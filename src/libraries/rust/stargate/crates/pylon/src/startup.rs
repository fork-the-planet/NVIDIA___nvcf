// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use pylon_lib::{
    AuthTokenProvider, BringupConfig, EngineStatsStreamConfig, EngineStatsStreamHandle,
    EngineStatsStreamMode, InferenceServerRegistrationClient, InferenceServerRegistrationConfig,
    MetricsServerHandle, OutputTokenParserFactory, PylonMetrics, PylonQueueMismatchRetryConfig,
    PylonRetryConfig, PylonRuntimeState, QuicHttpTunnelConfig, QuicHttpTunnelHandle,
    RequestQualityMonitorConfig, StatsCollectorConfig, StatsCollectorHandle,
    start_engine_stats_stream, start_metrics_server, start_quic_http_tunnel,
    start_stats_collector_with_engine_stats, stats_aggregator_update_channel,
};
use reqwest::header::HeaderName;
use stargate_proto::pb::InferenceServerStatus;
use stargate_runtime::wait_for_termination_signal;
use tokio::task::JoinError;
use tracing::{error, info};

use super::{Args, telemetry};

pub(super) async fn run(args: Args) -> Result<()> {
    let _telemetry_guard =
        telemetry::init_telemetry(args.otel_endpoint.as_deref(), &args.otel_service_name)?;
    let plan = PylonStartupPlan::from_args(&args)?;
    let runtime = start_pylon_runtime(&args, &plan).await?;

    if plan.reverse_mode() {
        info!(
            stargate = %args.stargate_address,
            inference_server_id = %args.inference_server_id,
            upstream = %plan.upstream,
            "registered with stargate (reverse tunnel mode)"
        );
    } else {
        info!(
            stargate = %args.stargate_address,
            inference_server_id = %args.inference_server_id,
            inference_server_url = %runtime.registration_inference_server_url,
            upstream = %plan.upstream,
            "registered with stargate"
        );
    }
    info!("pylon running");
    runtime
        .run_until_shutdown(wait_for_termination_signal())
        .await
}

#[derive(Clone)]
pub(crate) struct PylonStartupPlan {
    upstream: String,
    cluster_id: String,
    pylon_retry: PylonRetryConfig,
    queue_mismatch_retry: PylonQueueMismatchRetryConfig,
    fixed_last_mean_input_tps: Option<f64>,
    request_quality_monitor: RequestQualityMonitorConfig,
    metrics_addr: SocketAddr,
    auth_token_provider: Option<Arc<AuthTokenProvider>>,
    mode: PylonStartupMode,
}

#[derive(Clone, Copy)]
enum PylonStartupMode {
    Direct { listen_addr: SocketAddr },
    Reverse,
}

impl PylonStartupPlan {
    pub(crate) fn from_args(args: &Args) -> Result<Self> {
        Ok(Self {
            upstream: normalize_base_url(&args.upstream_http_base_url),
            cluster_id: effective_cluster_id(args),
            pylon_retry: pylon_retry_config_from_args(args)?,
            queue_mismatch_retry: pylon_queue_mismatch_retry_config_from_args(args)?,
            fixed_last_mean_input_tps: benchmark_fixed_last_mean_input_tps_from_args(args)?,
            request_quality_monitor: request_quality_monitor_config_from_args(args),
            metrics_addr: format!("{}:{}", args.metrics_host, args.metrics_port).parse()?,
            auth_token_provider: auth_token_provider_from_args(args),
            mode: startup_mode_from_args(args)?,
        })
    }

    pub(crate) fn direct_tunnel_listen_addr(&self) -> Option<SocketAddr> {
        match self.mode {
            PylonStartupMode::Direct { listen_addr } => Some(listen_addr),
            PylonStartupMode::Reverse => None,
        }
    }

    pub(crate) fn registration_inference_server_url(&self, direct_quic_url: &str) -> String {
        match self.mode {
            PylonStartupMode::Direct { .. } => direct_quic_url.to_string(),
            PylonStartupMode::Reverse => self.upstream.clone(),
        }
    }

    fn reverse_mode(&self) -> bool {
        matches!(self.mode, PylonStartupMode::Reverse)
    }
}

fn startup_mode_from_args(args: &Args) -> Result<PylonStartupMode> {
    if args.reverse_tunnel {
        Ok(PylonStartupMode::Reverse)
    } else {
        Ok(PylonStartupMode::Direct {
            listen_addr: args.quic_listen_addr.parse()?,
        })
    }
}

struct RunningPylon {
    registration_client: InferenceServerRegistrationClient,
    engine_stats_stream: Option<RunningEngineStatsStream>,
    stats_collector: StatsCollectorHandle,
    metrics_server: MetricsServerHandle,
    tunnel: Option<QuicHttpTunnelHandle>,
    registration_inference_server_url: String,
}

struct RunningEngineStatsStream {
    mode: EngineStatsStreamMode,
    handle: EngineStatsStreamHandle,
}

enum PylonRuntimeExit {
    Signal(std::io::Result<&'static str>),
    CriticalTask {
        name: &'static str,
        result: std::result::Result<(), JoinError>,
    },
    EngineStatsStream(std::result::Result<(), JoinError>),
}

impl RunningPylon {
    async fn run_until_shutdown<S>(mut self, signal: S) -> Result<()>
    where
        S: Future<Output = std::io::Result<&'static str>>,
    {
        tokio::pin!(signal);
        loop {
            match self.wait_for_exit(signal.as_mut()).await {
                PylonRuntimeExit::Signal(result) => {
                    let result = result.context("failed to receive pylon termination signal");
                    if let Ok(signal) = &result {
                        info!(signal, "received shutdown signal");
                    }
                    self.shutdown().await;
                    return result.map(|_| ());
                }
                PylonRuntimeExit::EngineStatsStream(result)
                    if engine_stats_exit_is_expected(
                        self.engine_stats_stream.as_ref().map(|stream| stream.mode),
                        &result,
                    ) =>
                {
                    info!("auto engine stats stream completed after enabling fallback");
                    self.engine_stats_stream = None;
                }
                PylonRuntimeExit::EngineStatsStream(result) => {
                    let error = critical_task_exit_error("engine stats stream", result);
                    error!(error = %error, "critical pylon task exited");
                    self.shutdown().await;
                    return Err(error);
                }
                PylonRuntimeExit::CriticalTask { name, result } => {
                    let error = critical_task_exit_error(name, result);
                    error!(error = %error, "critical pylon task exited");
                    self.shutdown().await;
                    return Err(error);
                }
            }
        }
    }

    async fn wait_for_exit<S>(&mut self, signal: Pin<&mut S>) -> PylonRuntimeExit
    where
        S: Future<Output = std::io::Result<&'static str>>,
    {
        tokio::select! {
            result = signal => PylonRuntimeExit::Signal(result),
            result = self.registration_client.wait_for_exit() => PylonRuntimeExit::CriticalTask {
                name: "registration session",
                result,
            },
            result = wait_for_engine_stats_exit(&mut self.engine_stats_stream) => {
                PylonRuntimeExit::EngineStatsStream(result)
            },
            result = self.stats_collector.wait_for_exit() => PylonRuntimeExit::CriticalTask {
                name: "stats collector",
                result,
            },
            result = self.metrics_server.wait_for_exit() => PylonRuntimeExit::CriticalTask {
                name: "metrics server",
                result,
            },
            result = wait_for_tunnel_exit(&mut self.tunnel) => PylonRuntimeExit::CriticalTask {
                name: "direct tunnel accept loop",
                result,
            },
        }
    }

    async fn shutdown(self) {
        let Self {
            mut registration_client,
            engine_stats_stream,
            stats_collector,
            metrics_server,
            tunnel,
            ..
        } = self;
        tokio::join!(
            registration_client.shutdown(),
            shutdown_engine_stats_stream(engine_stats_stream),
            stats_collector.shutdown(),
            metrics_server.shutdown(),
            shutdown_tunnel(tunnel),
        );
    }
}

async fn wait_for_engine_stats_exit(
    engine_stats_stream: &mut Option<RunningEngineStatsStream>,
) -> std::result::Result<(), JoinError> {
    let Some(engine_stats_stream) = engine_stats_stream else {
        return std::future::pending().await;
    };
    engine_stats_stream.handle.wait_for_exit().await
}

async fn wait_for_tunnel_exit(
    tunnel: &mut Option<QuicHttpTunnelHandle>,
) -> std::result::Result<(), JoinError> {
    let Some(tunnel) = tunnel else {
        return std::future::pending().await;
    };
    tunnel.wait_for_exit().await
}

async fn shutdown_engine_stats_stream(engine_stats_stream: Option<RunningEngineStatsStream>) {
    if let Some(engine_stats_stream) = engine_stats_stream {
        engine_stats_stream.handle.shutdown().await;
    }
}

async fn shutdown_tunnel(tunnel: Option<QuicHttpTunnelHandle>) {
    if let Some(tunnel) = tunnel {
        tunnel.shutdown().await;
    }
}

fn critical_task_exit_error(
    name: &'static str,
    result: std::result::Result<(), JoinError>,
) -> anyhow::Error {
    match result {
        Ok(()) => anyhow::anyhow!("{name} exited unexpectedly"),
        Err(error) => anyhow::anyhow!("{name} failed: {error}"),
    }
}

fn engine_stats_exit_is_expected(
    mode: Option<EngineStatsStreamMode>,
    result: &std::result::Result<(), JoinError>,
) -> bool {
    mode == Some(EngineStatsStreamMode::Auto) && result.is_ok()
}

async fn start_pylon_runtime(args: &Args, plan: &PylonStartupPlan) -> Result<RunningPylon> {
    let metrics = start_pylon_metrics()?;
    let metrics_server = start_metrics_server(plan.metrics_addr, metrics.registry()).await?;
    let stats_config =
        stats_collector_config_from_args(args, &plan.upstream, plan.fixed_last_mean_input_tps);
    let (runtime_state, request_observation_rx) = PylonRuntimeState::observed(
        InferenceServerStatus::Active,
        &args.model_name,
        stats_config.observation_channel_capacity,
        Some(metrics.clone()),
    );
    seed_runtime_state(
        &runtime_state,
        &args.model_name,
        plan.fixed_last_mean_input_tps,
    );
    let (engine_stats_stream, stats_update_rx) =
        start_engine_stats_runtime(args, plan, metrics.clone(), &stats_config);
    let tls_cert_pem = args.tls_cert_path.as_ref().map(std::fs::read).transpose()?;
    let tls_key_pem = args.tls_key_path.as_ref().map(std::fs::read).transpose()?;
    let tunnel = start_direct_tunnel_from_plan(
        args,
        plan,
        runtime_state.clone(),
        metrics.clone(),
        tls_cert_pem.clone(),
        tls_key_pem,
    )
    .await?;
    let registration_inference_server_url =
        registration_inference_server_url_from_tunnel(plan, tunnel.as_ref());
    let registration_config = registration_config_from_plan(
        args,
        plan,
        runtime_state.clone(),
        metrics,
        registration_inference_server_url.clone(),
        tls_cert_pem,
    );
    let mut registration_client = InferenceServerRegistrationClient::default();
    registration_client.start(registration_config)?;
    let stats_collector = start_stats_collector_with_engine_stats(
        stats_config,
        request_observation_rx,
        stats_update_rx,
        runtime_state,
    );

    Ok(RunningPylon {
        registration_client,
        engine_stats_stream,
        stats_collector,
        metrics_server,
        tunnel,
        registration_inference_server_url,
    })
}

fn start_engine_stats_runtime(
    args: &Args,
    plan: &PylonStartupPlan,
    metrics: Arc<PylonMetrics>,
    stats_config: &StatsCollectorConfig,
) -> (
    Option<RunningEngineStatsStream>,
    Option<flume::Receiver<pylon_lib::StatsAggregatorUpdate>>,
) {
    let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(stats_config);
    let mut config = EngineStatsStreamConfig::new(
        &plan.upstream,
        &args.engine_stats_stream_path,
        args.engine_stats_stream,
    );
    config.metrics = Some(metrics);
    let mode = config.mode;
    let engine_stats_stream = start_engine_stats_stream(config, stats_update_tx)
        .map(|handle| RunningEngineStatsStream { mode, handle });
    let stats_update_rx = engine_stats_stream.as_ref().map(|_| stats_update_rx);
    (engine_stats_stream, stats_update_rx)
}

async fn start_direct_tunnel_from_plan(
    args: &Args,
    plan: &PylonStartupPlan,
    runtime_state: PylonRuntimeState,
    metrics: Arc<PylonMetrics>,
    tls_cert_pem: Option<Vec<u8>>,
    tls_key_pem: Option<Vec<u8>>,
) -> Result<Option<QuicHttpTunnelHandle>> {
    let Some(tunnel_config) = direct_tunnel_config_from_plan(
        args,
        plan,
        runtime_state,
        metrics,
        tls_cert_pem,
        tls_key_pem,
    ) else {
        return Ok(None);
    };
    let tunnel = start_quic_http_tunnel(tunnel_config).await?;
    info!(addr = %tunnel.listen_addr(), url = %format!("quic://{}", tunnel.listen_addr()), "QUIC tunnel listening");
    Ok(Some(tunnel))
}

fn registration_inference_server_url_from_tunnel(
    plan: &PylonStartupPlan,
    tunnel: Option<&QuicHttpTunnelHandle>,
) -> String {
    let direct_quic_url = tunnel
        .map(|tunnel| format!("quic://{}", tunnel.listen_addr()))
        .unwrap_or_default();
    plan.registration_inference_server_url(&direct_quic_url)
}

fn seed_runtime_state(
    runtime_state: &PylonRuntimeState,
    model_names: &[String],
    fixed_last_mean_input_tps: Option<f64>,
) {
    if let Some(last_mean_input_tps) = fixed_last_mean_input_tps {
        for model_id in model_names {
            runtime_state.set_model_stats(
                model_id,
                pylon_lib::CurrentModelStats {
                    last_mean_input_tps,
                    ..Default::default()
                },
            );
        }
    }
}

fn start_pylon_metrics() -> Result<Arc<PylonMetrics>> {
    let metrics = PylonMetrics::new()?;
    metrics.observe_target_info(
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_NAME"),
        option_env!("GIT_COMMIT_HASH")
            .or(option_env!("GIT_COMMIT_SHA"))
            .unwrap_or(""),
    );
    Ok(metrics)
}

fn direct_tunnel_config_from_plan(
    args: &Args,
    plan: &PylonStartupPlan,
    runtime_state: PylonRuntimeState,
    metrics: Arc<PylonMetrics>,
    tls_cert_pem: Option<Vec<u8>>,
    tls_key_pem: Option<Vec<u8>>,
) -> Option<QuicHttpTunnelConfig> {
    let listen_addr = plan.direct_tunnel_listen_addr()?;
    Some(build_direct_tunnel_config(DirectTunnelConfigParams {
        listen_addr,
        upstream_http_base_url: plan.upstream.clone(),
        inference_server_id: args.inference_server_id.clone(),
        tls_cert_pem,
        tls_key_pem,
        quic_insecure: args.quic_insecure,
        tunnel_protocol: args.tunnel_protocol,
        retry: plan.pylon_retry.clone(),
        queue_mismatch_retry: plan.queue_mismatch_retry.clone(),
        runtime_state,
        request_quality_monitor: plan.request_quality_monitor.clone(),
        metrics,
    }))
}

fn registration_config_from_plan(
    args: &Args,
    plan: &PylonStartupPlan,
    runtime_state: PylonRuntimeState,
    metrics: Arc<PylonMetrics>,
    inference_server_url: String,
    tls_cert_pem: Option<Vec<u8>>,
) -> InferenceServerRegistrationConfig {
    InferenceServerRegistrationConfig {
        seeds: vec![args.stargate_address.clone()],
        inference_server_id: args.inference_server_id.clone(),
        cluster_id: plan.cluster_id.clone(),
        inference_server_url,
        upstream_http_base_url: Some(plan.upstream.clone()),
        min_update_interval: Duration::from_millis(args.min_update_interval_ms),
        reverse_tunnel: plan.reverse_mode(),
        tls_cert_pem,
        quic_insecure: args.quic_insecure,
        tunnel_protocol: args.tunnel_protocol,
        bringup: bringup_config_from_args(args),
        output_token_parser_factory: OutputTokenParserFactory::vllm(),
        runtime_state,
        request_quality_monitor: plan.request_quality_monitor.clone(),
        metrics: Some(metrics),
        retry: plan.pylon_retry.clone(),
        queue_mismatch_retry: plan.queue_mismatch_retry.clone(),
        auth_token_provider: plan.auth_token_provider.clone(),
    }
}

fn bringup_config_from_args(args: &Args) -> BringupConfig {
    BringupConfig {
        enabled: !args.disable_bringup,
        active_canary_interval: Duration::from_millis(args.active_canary_interval_ms),
        canary_timeout: Duration::from_millis(args.bringup_canary_timeout_ms),
        canary_max_generation_threshold: args.canary_max_generation_threshold,
        calibration_requests: args.calibration_requests,
        calibration_prompt_units: args.calibration_prompt_units,
        calibration_max_concurrency: args.calibration_max_concurrency,
        calibration_timeout: Duration::from_millis(args.bringup_calibration_timeout_ms),
    }
}

pub(crate) struct DirectTunnelConfigParams {
    pub(crate) listen_addr: SocketAddr,
    pub(crate) upstream_http_base_url: String,
    pub(crate) inference_server_id: String,
    pub(crate) tls_cert_pem: Option<Vec<u8>>,
    pub(crate) tls_key_pem: Option<Vec<u8>>,
    pub(crate) quic_insecure: bool,
    pub(crate) tunnel_protocol: pylon_lib::TunnelTransportProtocol,
    pub(crate) retry: PylonRetryConfig,
    pub(crate) queue_mismatch_retry: PylonQueueMismatchRetryConfig,
    pub(crate) runtime_state: PylonRuntimeState,
    pub(crate) request_quality_monitor: RequestQualityMonitorConfig,
    pub(crate) metrics: Arc<PylonMetrics>,
}

pub(crate) fn build_direct_tunnel_config(params: DirectTunnelConfigParams) -> QuicHttpTunnelConfig {
    let mut tunnel_config =
        QuicHttpTunnelConfig::new(params.listen_addr, params.upstream_http_base_url);
    tunnel_config.inference_server_id = Some(params.inference_server_id);
    tunnel_config.tls_cert_pem = params.tls_cert_pem;
    tunnel_config.tls_key_pem = params.tls_key_pem;
    tunnel_config.quic_insecure = params.quic_insecure;
    tunnel_config.tunnel_protocol = params.tunnel_protocol;
    tunnel_config.retry = params.retry;
    tunnel_config.queue_mismatch_retry = params.queue_mismatch_retry;
    tunnel_config.runtime_state = params.runtime_state;
    tunnel_config.request_quality_monitor = params.request_quality_monitor;
    tunnel_config.metrics = Some(params.metrics);
    tunnel_config
}

pub(crate) fn stats_collector_config_from_args(
    args: &Args,
    upstream: &str,
    fixed_last_mean_input_tps: Option<f64>,
) -> StatsCollectorConfig {
    let mut stats_config = StatsCollectorConfig {
        configured_model_ids: args.model_name.clone(),
        fixed_last_mean_input_tps,
        openai_fallback_stats_enabled: args.engine_stats_stream == EngineStatsStreamMode::Off,
        ..Default::default()
    };
    if let Some(path) = args.kv_cache_stats_path.as_deref() {
        // Mock benchmark backends can expose live KV-cache occupancy over HTTP;
        // real upstreams usually do not, so polling is explicit.
        stats_config.kv_cache_stats_url = Some(join_base_url_path(upstream, path));
    }
    stats_config
}

fn join_base_url_path(base_url: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

pub(crate) fn normalize_base_url(url: &str) -> String {
    url.trim_end_matches('/').to_string()
}

pub(crate) fn effective_cluster_id(args: &Args) -> String {
    args.cluster_id
        .clone()
        .unwrap_or_else(|| args.inference_server_id.clone())
}

pub(crate) fn pylon_retry_config_from_args(args: &Args) -> Result<PylonRetryConfig> {
    Ok(PylonRetryConfig {
        retryable_upstream_status_codes: parse_retryable_status_codes(
            &args.pylon_retryable_upstream_status_codes,
        )?,
        require_upstream_retry_header: args.pylon_require_upstream_retry_header,
        upstream_retry_header: HeaderName::from_str(args.pylon_upstream_retry_header.trim())
            .with_context(|| {
                format!(
                    "invalid pylon upstream retry header: {}",
                    args.pylon_upstream_retry_header
                )
            })?,
        propagate_retry_after: args.pylon_propagate_retry_after,
        local_connect_failures_retryable: args.pylon_local_connect_failures_retryable,
    })
}

pub(crate) fn pylon_queue_mismatch_retry_config_from_args(
    args: &Args,
) -> Result<PylonQueueMismatchRetryConfig> {
    ensure!(
        args.pylon_queue_mismatch_tolerance_factor.is_finite()
            && args.pylon_queue_mismatch_tolerance_factor > 0.0,
        "pylon queue mismatch tolerance factor must be finite and positive"
    );
    Ok(PylonQueueMismatchRetryConfig {
        enabled: args.pylon_queue_mismatch_retry_enabled,
        min_delta_ms: args.pylon_queue_mismatch_min_delta_ms,
        tolerance_factor: args.pylon_queue_mismatch_tolerance_factor,
        retry_after_ms: args.pylon_queue_mismatch_retry_after_ms,
    })
}

pub(crate) fn benchmark_fixed_last_mean_input_tps_from_args(args: &Args) -> Result<Option<f64>> {
    if let Some(last_mean_input_tps) = args.benchmark_fixed_last_mean_input_tps {
        ensure!(
            last_mean_input_tps.is_finite() && last_mean_input_tps > 0.0,
            "benchmark fixed last mean input TPS must be finite and positive"
        );
    }
    Ok(args.benchmark_fixed_last_mean_input_tps)
}

pub(crate) fn parse_retryable_status_codes(value: &str) -> Result<Vec<reqwest::StatusCode>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            let code = part
                .parse::<u16>()
                .with_context(|| format!("invalid pylon retryable status code: {part}"))?;
            reqwest::StatusCode::from_u16(code)
                .with_context(|| format!("invalid pylon retryable status code: {part}"))
        })
        .collect()
}

pub(crate) fn request_quality_monitor_config_from_args(args: &Args) -> RequestQualityMonitorConfig {
    RequestQualityMonitorConfig {
        collect_quality_metrics: args.collect_quality_metrics,
        collect_quality_metrics_min_tokens: args.collect_quality_metrics_min_tokens,
        output_tokens_threshold_min: args.quality_output_tokens_threshold_min,
        output_compression_threshold_max: args.quality_output_compression_threshold_max,
        output_degeneracy_threshold_min: args.quality_output_degeneracy_threshold_min,
        output_repetition_1gram_threshold_min: args.quality_output_repetition_1gram_threshold_min,
        output_repetition_2gram_threshold_min: args.quality_output_repetition_2gram_threshold_min,
        output_repetition_3gram_threshold_min: args.quality_output_repetition_3gram_threshold_min,
        median_logprob_threshold_max: args.quality_median_logprob_threshold_max,
    }
}

fn auth_token_provider_from_args(args: &Args) -> Option<Arc<AuthTokenProvider>> {
    if let Some(token) = args.auth_token.as_ref() {
        Some(Arc::new(AuthTokenProvider::Static(token.clone())))
    } else {
        args.auth_token_file
            .as_ref()
            .map(|path| Arc::new(AuthTokenProvider::File(path.clone().into())))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use clap::Parser;
    use pylon_lib::{
        EngineStatsStreamMode, PylonMetrics, RequestObservation, RequestObservationEndpoint,
        RequestObservationState, TunnelTransportProtocol,
    };
    use stargate_proto::pb::InferenceServerStatus;

    use super::*;

    fn parse_args(extra: &[&str]) -> Args {
        let mut args = vec![
            "pylon",
            "--upstream-http-base-url",
            "http://127.0.0.1:8090/",
            "--model-name",
            "model-a",
            "--model-name",
            "model-b",
        ];
        args.extend_from_slice(extra);
        <Args as Parser>::try_parse_from(args).expect("args should parse")
    }

    fn test_observation() -> RequestObservation {
        RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "request-a".to_string(),
            routing_key: None,
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 8,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: None,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::Queued,
            time_to_response_headers: None,
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::ZERO,
        }
    }

    async fn receive_queued_model_stats(
        runtime_state: &PylonRuntimeState,
        model_id: &str,
    ) -> pylon_lib::CurrentModelStats {
        tokio::time::timeout(Duration::from_secs(1), async {
            let mut poll = tokio::time::interval(Duration::from_millis(1));
            loop {
                poll.tick().await;
                if let Some(stats) = runtime_state.model_stats(model_id)
                    && stats.queue_size > 0
                {
                    break stats;
                }
            }
        })
        .await
        .expect("model stats should arrive")
    }

    async fn test_running_pylon(stats_collector: StatsCollectorHandle) -> RunningPylon {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        RunningPylon {
            registration_client: InferenceServerRegistrationClient::default(),
            engine_stats_stream: None,
            stats_collector,
            metrics_server: start_metrics_server(
                "127.0.0.1:0".parse().expect("metrics address should parse"),
                metrics.registry(),
            )
            .await
            .expect("metrics server should start"),
            tunnel: None,
            registration_inference_server_url: "quic://127.0.0.1:4567".to_string(),
        }
    }

    fn active_test_stats_collector() -> StatsCollectorHandle {
        let config = StatsCollectorConfig::default();
        let (runtime_state, request_observation_rx) = PylonRuntimeState::observed(
            InferenceServerStatus::Unknown,
            &[],
            config.observation_channel_capacity,
            None,
        );
        start_stats_collector_with_engine_stats(config, request_observation_rx, None, runtime_state)
    }

    #[test]
    fn engine_stats_runtime_off_mode_leaves_stats_updates_unclaimed() {
        let args = parse_args(&["--engine-stats-stream", "off"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = StatsCollectorConfig::default();
        let (engine_stats_stream, stats_update_rx) =
            start_engine_stats_runtime(&args, &plan, metrics, &config);

        assert!(engine_stats_stream.is_none());
        assert!(stats_update_rx.is_none());
    }

    #[tokio::test]
    async fn engine_stats_runtime_required_mode_claims_stats_updates() {
        let args = parse_args(&["--engine-stats-stream", "required"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = StatsCollectorConfig::default();
        let (engine_stats_stream, stats_update_rx) =
            start_engine_stats_runtime(&args, &plan, metrics, &config);

        assert!(stats_update_rx.is_some());
        engine_stats_stream
            .expect("required engine stats should start a stream task")
            .handle
            .shutdown()
            .await;
    }

    #[test]
    fn only_successful_auto_engine_stats_completion_is_nonfatal() {
        let completed = Ok(());

        assert!(engine_stats_exit_is_expected(
            Some(EngineStatsStreamMode::Auto),
            &completed,
        ));
        assert!(!engine_stats_exit_is_expected(
            Some(EngineStatsStreamMode::Required),
            &completed,
        ));
        assert!(!engine_stats_exit_is_expected(None, &completed));
    }

    #[tokio::test]
    async fn reverse_mode_direct_tunnel_startup_returns_no_tunnel_without_binding() {
        let args = parse_args(&["--reverse-tunnel"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let tunnel = start_direct_tunnel_from_plan(
            &args,
            &plan,
            PylonRuntimeState::default(),
            metrics,
            None,
            None,
        )
        .await
        .expect("reverse mode should not start a direct tunnel");

        assert!(tunnel.is_none());
        assert_eq!(
            registration_inference_server_url_from_tunnel(&plan, None),
            "http://127.0.0.1:8090"
        );
    }

    #[tokio::test]
    async fn direct_mode_direct_tunnel_startup_binds_and_reports_quic_url() {
        let args = parse_args(&["--quic-listen-addr", "127.0.0.1:0"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let tunnel = start_direct_tunnel_from_plan(
            &args,
            &plan,
            PylonRuntimeState::default(),
            metrics,
            None,
            None,
        )
        .await
        .expect("direct mode should bind a direct tunnel")
        .expect("direct mode should return the tunnel handle");

        let registration_url = registration_inference_server_url_from_tunnel(&plan, Some(&tunnel));
        assert!(registration_url.starts_with("quic://127.0.0.1:"));
        tunnel.shutdown().await;
    }

    #[test]
    fn direct_tunnel_config_from_plan_preserves_runtime_inputs() {
        let args = parse_args(&[
            "--inference-server-id",
            "pylon-a",
            "--quic-listen-addr",
            "127.0.0.1:4567",
            "--tunnel-protocol",
            "webtransport",
            "--pylon-local-connect-failures-retryable=true",
            "--pylon-queue-mismatch-retry-enabled=false",
            "--collect-quality-metrics",
        ]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let runtime_state = PylonRuntimeState::default();

        let config = direct_tunnel_config_from_plan(
            &args,
            &plan,
            runtime_state,
            metrics.clone(),
            Some(b"cert".to_vec()),
            Some(b"key".to_vec()),
        )
        .expect("direct mode should build tunnel config");

        assert_eq!(config.listen_addr, "127.0.0.1:4567".parse().unwrap());
        assert_eq!(config.upstream_http_base_url, "http://127.0.0.1:8090");
        assert_eq!(config.inference_server_id.as_deref(), Some("pylon-a"));
        assert_eq!(config.tls_cert_pem.as_deref(), Some(&b"cert"[..]));
        assert_eq!(config.tls_key_pem.as_deref(), Some(&b"key"[..]));
        assert_eq!(
            config.tunnel_protocol,
            TunnelTransportProtocol::WebTransport
        );
        assert!(config.retry.local_connect_failures_retryable);
        assert!(!config.queue_mismatch_retry.enabled);
        assert!(config.request_quality_monitor.collect_quality_metrics);
        assert!(Arc::ptr_eq(config.metrics.as_ref().unwrap(), &metrics));
    }

    #[test]
    fn direct_tunnel_config_is_absent_for_reverse_mode() {
        let args = parse_args(&["--reverse-tunnel"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");

        assert!(
            direct_tunnel_config_from_plan(
                &args,
                &plan,
                PylonRuntimeState::default(),
                metrics,
                None,
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn registration_config_from_plan_preserves_direct_registration_contract() {
        let args = parse_args(&[
            "--stargate-address",
            "http://stargate:50071",
            "--inference-server-id",
            "pylon-a",
            "--min-update-interval-ms",
            "250",
        ]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");

        let config = registration_config_from_plan(
            &args,
            &plan,
            PylonRuntimeState::default(),
            metrics,
            "quic://127.0.0.1:4567".to_string(),
            None,
        );

        assert_eq!(config.seeds, ["http://stargate:50071"]);
        assert_eq!(config.inference_server_id, "pylon-a");
        assert_eq!(config.inference_server_url, "quic://127.0.0.1:4567");
        assert_eq!(
            config.upstream_http_base_url.as_deref(),
            Some("http://127.0.0.1:8090")
        );
        assert_eq!(config.min_update_interval, Duration::from_millis(250));
        assert!(!config.reverse_tunnel);
    }

    #[test]
    fn registration_config_from_plan_preserves_reverse_registration_contract() {
        let args = parse_args(&[
            "--reverse-tunnel",
            "--stargate-address",
            "http://stargate:50071",
            "--inference-server-id",
            "pylon-a",
            "--cluster-id",
            "shared-cluster",
            "--min-update-interval-ms",
            "250",
            "--quic-insecure",
            "--tunnel-protocol",
            "http3",
            "--disable-bringup",
            "--active-canary-interval-ms",
            "123",
            "--canary-max-generation-threshold",
            "77",
            "--calibration-requests",
            "3",
            "--calibration-prompt-units",
            "512",
            "--calibration-max-concurrency",
            "2",
            "--bringup-canary-timeout-ms",
            "456",
            "--bringup-calibration-timeout-ms",
            "789",
            "--auth-token",
            "token-from-cli",
        ]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let runtime_state = PylonRuntimeState::default();

        let config = registration_config_from_plan(
            &args,
            &plan,
            runtime_state,
            metrics.clone(),
            "http://127.0.0.1:8090".to_string(),
            Some(b"trusted reverse cert".to_vec()),
        );

        assert_eq!(config.seeds, ["http://stargate:50071"]);
        assert_eq!(config.inference_server_id, "pylon-a");
        assert_eq!(config.cluster_id, "shared-cluster");
        assert_eq!(config.inference_server_url, "http://127.0.0.1:8090");
        assert_eq!(
            config.upstream_http_base_url.as_deref(),
            Some("http://127.0.0.1:8090")
        );
        assert_eq!(config.min_update_interval, Duration::from_millis(250));
        assert!(config.reverse_tunnel);
        assert_eq!(
            config.tls_cert_pem.as_deref(),
            Some(&b"trusted reverse cert"[..])
        );
        assert!(config.quic_insecure);
        assert_eq!(config.tunnel_protocol, TunnelTransportProtocol::Http3);
        assert!(!config.bringup.enabled);
        assert_eq!(
            config.bringup.active_canary_interval,
            Duration::from_millis(123)
        );
        assert_eq!(config.bringup.canary_max_generation_threshold, 77);
        assert_eq!(config.bringup.calibration_requests, 3);
        assert_eq!(config.bringup.calibration_prompt_units, 512);
        assert_eq!(config.bringup.calibration_max_concurrency, 2);
        assert_eq!(config.bringup.canary_timeout, Duration::from_millis(456));
        assert_eq!(
            config.bringup.calibration_timeout,
            Duration::from_millis(789)
        );
        assert!(Arc::ptr_eq(config.metrics.as_ref().unwrap(), &metrics));
        assert!(matches!(
            config.auth_token_provider.as_deref(),
            Some(AuthTokenProvider::Static(token)) if token == "token-from-cli"
        ));
    }

    #[test]
    fn stats_config_uses_normalized_upstream_and_fixed_rate() {
        let args = parse_args(&[
            "--engine-stats-stream",
            "required",
            "--kv-cache-stats-path",
            "kv/live",
            "--benchmark-fixed-last-mean-input-tps",
            "1200",
        ]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let stats =
            stats_collector_config_from_args(&args, &plan.upstream, plan.fixed_last_mean_input_tps);

        assert_eq!(stats.configured_model_ids, ["model-a", "model-b"]);
        assert_eq!(stats.fixed_last_mean_input_tps, Some(1200.0));
        assert_eq!(
            stats.kv_cache_stats_url.as_deref(),
            Some("http://127.0.0.1:8090/kv/live")
        );
        assert!(!stats.openai_fallback_stats_enabled);
    }

    #[tokio::test]
    async fn fixed_input_tps_seeds_queue_estimates_before_engine_stats() {
        let args = parse_args(&["--benchmark-fixed-last-mean-input-tps", "1000"]);
        let plan = PylonStartupPlan::from_args(&args).expect("startup plan should build");
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let mut config = stats_collector_config_from_args(&args, &plan.upstream, None);
        config.openai_fallback_stats_enabled = true;
        let (runtime_state, request_observation_rx) = PylonRuntimeState::observed(
            InferenceServerStatus::Unknown,
            &args.model_name,
            config.observation_channel_capacity,
            Some(metrics),
        );
        seed_runtime_state(
            &runtime_state,
            &args.model_name,
            plan.fixed_last_mean_input_tps,
        );
        let stats_collector = start_stats_collector_with_engine_stats(
            config,
            request_observation_rx,
            None,
            runtime_state.clone(),
        );
        let mut observation = test_observation();
        observation.input_tokens = 1000;

        runtime_state.observe_request(observation);
        let stats = receive_queued_model_stats(&runtime_state, "model-a").await;

        assert_eq!(stats.queue_size, 1);
        assert_eq!(
            stats
                .queue_time_estimate_ms_by_priority
                .as_ref()
                .and_then(|estimates| estimates.get(&0)),
            Some(&1000)
        );
        stats_collector.shutdown().await;
    }

    #[tokio::test]
    async fn running_pylon_shutdown_stops_owned_metrics_server() {
        let runtime = test_running_pylon(active_test_stats_collector()).await;

        tokio::time::timeout(Duration::from_secs(1), runtime.shutdown())
            .await
            .expect("owned metrics server should stop during shutdown");
    }

    #[tokio::test]
    async fn running_pylon_returns_error_when_stats_collector_exits() {
        let observation_rx = {
            let (_observation_tx, observation_rx) =
                flume::bounded::<pylon_lib::RequestObservationEvent>(1);
            observation_rx
        };
        let runtime = test_running_pylon(start_stats_collector_with_engine_stats(
            StatsCollectorConfig::default(),
            observation_rx,
            None,
            PylonRuntimeState::default(),
        ))
        .await;

        let error = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.run_until_shutdown(std::future::pending::<std::io::Result<&'static str>>()),
        )
        .await
        .expect("critical stats exit should wake the runtime")
        .expect_err("critical stats exit should fail the runtime");

        assert!(error.to_string().contains("stats collector exited"));
    }

    #[tokio::test]
    async fn running_pylon_treats_sigterm_as_clean_shutdown() {
        let runtime = test_running_pylon(active_test_stats_collector()).await;

        tokio::time::timeout(
            Duration::from_secs(1),
            runtime.run_until_shutdown(async { Ok("SIGTERM") }),
        )
        .await
        .expect("SIGTERM shutdown should finish")
        .expect("SIGTERM should be a clean runtime exit");
    }

    #[test]
    fn auth_token_file_is_used_when_static_token_is_absent() {
        let args = parse_args(&["--auth-token-file", "/tmp/pylon-token"]);

        assert!(matches!(
            auth_token_provider_from_args(&args).as_deref(),
            Some(AuthTokenProvider::File(path)) if path == std::path::Path::new("/tmp/pylon-token")
        ));
    }
}
