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

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail, ensure};
use bytes::Buf;
use futures::TryStreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use sonic_rs::JsonValueTrait;
use stargate_protocol::tunnel_contract::{
    HEADER_STARGATE_EXPECTED_QUEUE_MS, HEADER_STARGATE_RETRY_AFTER_MS,
    HEADER_STARGATE_RETRY_REASON, HEADER_STARGATE_RETRYABLE, HEADER_STARGATE_UPSTREAM_RETRYABLE,
};
use stargate_telemetry::{
    inject_trace_context, parent_context_from_headers, traceparent_from_headers,
};
use tracing::{Instrument, Span, field};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::output_token_parser::{
    OutputTokenParser, OutputTokenParserFactory, OutputTokenProgress,
};
use crate::queue_admission::{
    PylonQueueMismatchRetryConfig, QueueAdmissionDecision, QueueTrackedRequestGuard,
    RETRY_REASON_QUEUE_ESTIMATE_MISMATCH,
};
use crate::request_observer::{
    MissingRequiredHeaderError, RequestObservationEndpoint, RequestObserver, RequiredTunnelHeaders,
    TunnelRequestObserver, validate_required_tunnel_headers,
};
use crate::request_quality_monitor::{
    RequestOutputTokenProgress, RequestQualityMonitorConfig, RequestQualityRecorder,
};
use crate::runtime_state::PylonRuntimeState;
use crate::sse_message_stream::{
    ParsedSseMessage, SseMessage, SseReadTimeoutPhase, UpstreamSseMessageStream,
    UpstreamSseReadError, upstream_sse_message_stream,
};
use crate::stats::PylonMetrics;

pub(super) const DEFAULT_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
// This bounds only one upstream SSE event waiting for its blank-line delimiter,
// not the request body or completed response events. One MiB accommodates
// unusually large structured chunks while a missing delimiter cannot make the
// pylon retain unbounded upstream bytes.
pub const DEFAULT_MAX_SSE_BUFFER_BYTES: usize = 1024 * 1024;
pub(super) const DEFAULT_FIRST_OUTPUT_TIMEOUT: Duration = Duration::from_secs(30);
pub(super) const DEFAULT_OUTPUT_CHUNK_TIMEOUT: Duration = Duration::from_secs(30);
pub(super) const MAX_SPECULATIVE_REQUEST_BODY_PREALLOC_BYTES: usize = 64 * 1024;
pub(super) const RETRY_REASON_UPSTREAM_ADMISSION_REJECTED: &str = "upstream_admission_rejected";
pub(super) const RETRY_REASON_LOCAL_CONNECT_FAILURE: &str = "local_connect_failure";
pub(super) const WEBTRANSPORT_STREAM_HEADER_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Debug)]
pub struct PylonRetryConfig {
    pub retryable_upstream_status_codes: Vec<reqwest::StatusCode>,
    pub require_upstream_retry_header: bool,
    pub upstream_retry_header: HeaderName,
    pub propagate_retry_after: bool,
    pub local_connect_failures_retryable: bool,
}

impl Default for PylonRetryConfig {
    fn default() -> Self {
        Self {
            retryable_upstream_status_codes: vec![
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ],
            require_upstream_retry_header: true,
            upstream_retry_header: HeaderName::from_static(HEADER_STARGATE_UPSTREAM_RETRYABLE),
            propagate_retry_after: true,
            local_connect_failures_retryable: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TunnelForwardingConfig {
    pub upstream_http_base_url: String,
    pub max_request_body_bytes: usize,
    /// Maximum bytes in one upstream SSE event before its blank-line delimiter.
    /// Completed events are forwarded and released immediately, independently
    /// of the request-body limit.
    pub max_sse_buffer_bytes: usize,
    pub first_output_timeout: Duration,
    pub output_chunk_timeout: Duration,
    pub output_token_parser_factory: OutputTokenParserFactory,
    pub runtime_state: PylonRuntimeState,
    pub request_quality_monitor: RequestQualityMonitorConfig,
    pub retry: PylonRetryConfig,
    pub queue_mismatch_retry: PylonQueueMismatchRetryConfig,
    pub metrics: Option<Arc<PylonMetrics>>,
    #[cfg(test)]
    pub webtransport_stream_header_wait_tx: Option<flume::Sender<()>>,
}

impl TunnelForwardingConfig {
    pub fn new(upstream_http_base_url: String) -> Self {
        Self {
            upstream_http_base_url,
            max_request_body_bytes: DEFAULT_MAX_BODY_BYTES,
            max_sse_buffer_bytes: DEFAULT_MAX_SSE_BUFFER_BYTES,
            first_output_timeout: DEFAULT_FIRST_OUTPUT_TIMEOUT,
            output_chunk_timeout: DEFAULT_OUTPUT_CHUNK_TIMEOUT,
            output_token_parser_factory: OutputTokenParserFactory,
            runtime_state: PylonRuntimeState::default(),
            request_quality_monitor: RequestQualityMonitorConfig::default(),
            retry: PylonRetryConfig::default(),
            queue_mismatch_retry: PylonQueueMismatchRetryConfig::default(),
            metrics: None,
            #[cfg(test)]
            webtransport_stream_header_wait_tx: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct TunnelServerApp {
    pub(super) http_client: reqwest::Client,
    pub(super) inference_server_id: String,
    pub(super) upstream_http_base_url: String,
    pub(super) max_request_body_bytes: usize,
    pub(super) max_sse_buffer_bytes: usize,
    pub(super) first_output_timeout: Duration,
    pub(super) output_chunk_timeout: Duration,
    pub(super) output_token_parser_factory: OutputTokenParserFactory,
    pub(super) runtime_state: PylonRuntimeState,
    pub(super) request_quality_monitor: RequestQualityMonitorConfig,
    pub(super) retry: PylonRetryConfig,
    pub(super) queue_mismatch_retry: PylonQueueMismatchRetryConfig,
    pub(super) metrics: Option<Arc<PylonMetrics>>,
    #[cfg(test)]
    pub(super) webtransport_stream_header_wait_tx: Option<flume::Sender<()>>,
}

fn evaluate_queue_admission(
    app: &TunnelServerApp,
    required_tunnel_headers: &RequiredTunnelHeaders,
    request_headers: &HeaderMap,
) -> QueueAdmissionDecision {
    let decision = app.runtime_state.evaluate_queue_admission(
        &app.queue_mismatch_retry,
        required_tunnel_headers,
        request_headers,
    );
    if let Some(metrics) = app.metrics.as_deref() {
        metrics.observe_queue_admission_decision(
            &app.inference_server_id,
            &required_tunnel_headers.model_id,
            decision.result_label(),
            decision.expected_ms(),
            decision.actual_ms(),
        );
    }
    log_queue_admission_decision(&decision);
    decision
}

macro_rules! emit_queue_admission_decision {
    ($level:expr, $decision:expr) => {
        tracing::event!(
            $level,
            queue.expected_ms = $decision.expected_ms().unwrap_or_default(),
            queue.expected_present = $decision.expected_ms().is_some(),
            queue.actual_ms = $decision.actual_ms().unwrap_or_default(),
            queue.actual_present = $decision.actual_ms().is_some(),
            queue.admission_result = $decision.result_label(),
            queue.mismatch_threshold_ms = $decision.threshold_ms().unwrap_or_default(),
            queue.mismatch_threshold_present = $decision.threshold_ms().is_some(),
            "evaluated local queue mismatch admission"
        )
    };
}

fn log_queue_admission_decision(decision: &QueueAdmissionDecision) {
    match queue_admission_log_level(decision) {
        tracing::Level::INFO => emit_queue_admission_decision!(tracing::Level::INFO, decision),
        tracing::Level::DEBUG => emit_queue_admission_decision!(tracing::Level::DEBUG, decision),
        _ => unreachable!("queue admission log level should be INFO or DEBUG"),
    }
}

fn queue_admission_log_level(decision: &QueueAdmissionDecision) -> tracing::Level {
    if should_log_queue_admission_at_info(decision) {
        tracing::Level::INFO
    } else {
        tracing::Level::DEBUG
    }
}

fn should_log_queue_admission_at_info(decision: &QueueAdmissionDecision) -> bool {
    matches!(decision, QueueAdmissionDecision::Rejected { .. })
}

fn tracked_queue_request_for_required_headers(
    app: &TunnelServerApp,
    required_tunnel_headers: Option<&RequiredTunnelHeaders>,
) -> Option<QueueTrackedRequestGuard> {
    required_tunnel_headers.map(|required| app.runtime_state.track_request(required))
}

fn observe_queue_output(queue_request: &mut Option<QueueTrackedRequestGuard>) {
    if let Some(queue_request) = queue_request.as_mut() {
        queue_request.observe_output();
    }
}

fn cleanup_rejected_queue_request(app: &TunnelServerApp, required: &RequiredTunnelHeaders) {
    // Observers are created before admission so body validation and terminal
    // accounting keep their existing order. Remove the queue projection before
    // sending the rejection; the observed projection remains until fail()
    // terminalizes it and clears lifecycle metrics.
    app.runtime_state.finish_queue_request(&required.request_id);
}

enum TunnelRequestLifecycleInitError {
    BadRequiredHeaders(MissingRequiredHeaderError),
    Internal(anyhow::Error),
}

struct TunnelRequestLifecycle {
    required_tunnel_headers: Option<RequiredTunnelHeaders>,
    observer: Option<TunnelRequestObserver>,
    queue_request: Option<QueueTrackedRequestGuard>,
    quality_recorder: Option<RequestQualityRecorder>,
}

impl TunnelRequestLifecycle {
    fn new(
        app: &TunnelServerApp,
        method: &reqwest::Method,
        path_and_query: &str,
        request_headers: &HeaderMap,
    ) -> std::result::Result<Self, TunnelRequestLifecycleInitError> {
        let observation_endpoint = request_observation_endpoint(method, path_and_query);
        let required_tunnel_headers = if is_health_request_path(path_and_query) {
            None
        } else {
            Some(
                validate_required_tunnel_headers(request_headers)
                    .map_err(TunnelRequestLifecycleInitError::BadRequiredHeaders)?,
            )
        };
        let observer = if let Some(endpoint) = observation_endpoint {
            let required = required_tunnel_headers.clone().ok_or_else(|| {
                TunnelRequestLifecycleInitError::Internal(anyhow::anyhow!(
                    "required tunnel headers missing for observed request"
                ))
            })?;
            Some(TunnelRequestObserver::accepted(
                endpoint,
                required,
                app.runtime_state.clone(),
            ))
        } else {
            None
        };
        let quality_recorder = if observation_endpoint
            == Some(RequestObservationEndpoint::ChatCompletions)
            && app.request_quality_monitor.enabled()
        {
            Some(RequestQualityRecorder::new())
        } else {
            None
        };

        Ok(Self {
            required_tunnel_headers,
            observer,
            queue_request: None,
            quality_recorder,
        })
    }

    fn validate_body(
        &mut self,
        method: &reqwest::Method,
        path_and_query: &str,
        body_bytes: &[u8],
    ) -> std::result::Result<(), RequestBodyValidationError> {
        if let Some(observer) = self.observer.as_mut() {
            observer.observe_request_body(body_bytes);
        }
        if let Err(error) = validate_request_body(method, path_and_query, body_bytes) {
            self.fail();
            return Err(error);
        }
        Ok(())
    }

    fn reject_queue_mismatch(
        &mut self,
        app: &TunnelServerApp,
        request_headers: &HeaderMap,
    ) -> Option<QueueAdmissionDecision> {
        let required = self.required_tunnel_headers.as_ref()?;
        let decision = evaluate_queue_admission(app, required, request_headers);
        if !matches!(decision, QueueAdmissionDecision::Rejected { .. }) {
            return None;
        }

        cleanup_rejected_queue_request(app, required);
        self.fail();
        Some(decision)
    }

    fn start_queue_tracking(&mut self, app: &TunnelServerApp) {
        self.queue_request =
            tracked_queue_request_for_required_headers(app, self.required_tunnel_headers.as_ref());
    }

    fn fail(&mut self) {
        if let Some(observer) = self.observer.as_mut() {
            observer.fail();
        }
    }

    fn on_upstream_headers(&mut self, response_headers: &HeaderMap, status: reqwest::StatusCode) {
        if let Some(queue_request) = self.queue_request.as_mut() {
            queue_request.on_upstream_response_headers();
        }
        if let Some(observer) = self.observer.as_mut() {
            observer.on_upstream_response_headers(response_headers, status.as_u16());
        }
    }

    fn should_relay_sse(&self, response_headers: &HeaderMap) -> bool {
        self.observer
            .as_ref()
            .is_some_and(TunnelRequestObserver::is_streaming)
            && is_sse_response(response_headers)
    }

    fn observe_raw_success(&mut self, status: reqwest::StatusCode) {
        if status.is_success() {
            observe_queue_output(&mut self.queue_request);
        }
        if let Some(obs) = self
            .observer
            .as_mut()
            .and_then(TunnelRequestObserver::generation_mut)
            && status.is_success()
        {
            obs.observe_output_message();
        }
    }

    async fn relay_sse<Transport>(
        &mut self,
        app: &TunnelServerApp,
        response: reqwest::Response,
        transport: &mut Transport,
    ) -> Result<()>
    where
        Transport: TunnelRequestTransport,
    {
        let mut upstream_messages = upstream_sse_message_stream(
            response.bytes_stream(),
            app.first_output_timeout,
            app.output_chunk_timeout,
            app.max_sse_buffer_bytes,
        );
        let mut output_token_parser = app.output_token_parser_factory.create();
        let obs = self
            .observer
            .as_mut()
            .and_then(TunnelRequestObserver::generation_mut)
            .ok_or_else(|| anyhow::anyhow!("observer missing for observed streaming request"))?;
        relay_remaining_output(
            &mut upstream_messages,
            &mut output_token_parser,
            obs,
            self.quality_recorder.as_mut(),
            &mut self.queue_request,
            transport,
        )
        .await
    }

    fn finish(&mut self) {
        if let Some(observer) = self.observer.as_mut() {
            observer.finish();
        }
        if let Some(queue_request) = self.queue_request.as_mut() {
            queue_request.finish();
        }
    }

    fn finalize_quality_check(&self, app: &TunnelServerApp, request_headers: &HeaderMap) {
        finalize_quality_check(
            request_headers,
            self.quality_recorder.as_ref(),
            &app.request_quality_monitor,
            app.metrics.as_deref(),
        );
    }
}

pub(super) struct TunnelRequestParts {
    pub(super) method: reqwest::Method,
    pub(super) path_and_query: String,
    pub(super) headers: HeaderMap,
}

pub(super) trait TunnelRequestTransport: ResponseBodyEventSink {
    async fn read_request_body(
        &mut self,
        request_headers: &HeaderMap,
        max_request_body_bytes: usize,
    ) -> Result<Vec<u8>>;

    async fn send_success(
        &mut self,
        status: reqwest::StatusCode,
        response_headers: &HeaderMap,
        retry: &PylonRetryConfig,
        metrics: Option<&PylonMetrics>,
        inference_server_id: &str,
    ) -> Result<()>;

    async fn send_error(&mut self, status: reqwest::StatusCode, message: String) -> Result<()>;

    async fn send_queue_mismatch(
        &mut self,
        app: &TunnelServerApp,
        decision: &QueueAdmissionDecision,
    ) -> Result<()>;

    async fn send_local_connect_failure(
        &mut self,
        app: &TunnelServerApp,
        error: &UpstreamRequestError,
        retryable: bool,
    ) -> Result<()>;

    async fn finish_response(&mut self) -> Result<()>;
}

pub(super) async fn forward_tunnel_request<Transport>(
    app: &TunnelServerApp,
    request: TunnelRequestParts,
    transport: &mut Transport,
) -> Result<()>
where
    Transport: TunnelRequestTransport,
{
    let TunnelRequestParts {
        method,
        path_and_query,
        headers: request_headers,
    } = request;
    let mut lifecycle =
        match TunnelRequestLifecycle::new(app, &method, &path_and_query, &request_headers) {
            Ok(lifecycle) => lifecycle,
            Err(TunnelRequestLifecycleInitError::BadRequiredHeaders(error)) => {
                transport
                    .send_error(reqwest::StatusCode::BAD_REQUEST, error.message())
                    .await?;
                return Ok(());
            }
            Err(TunnelRequestLifecycleInitError::Internal(error)) => return Err(error),
        };

    let body_bytes = transport
        .read_request_body(&request_headers, app.max_request_body_bytes)
        .await?;
    if is_health_request_path(&path_and_query) {
        return forward_tunnel_health_request(
            app,
            method,
            &path_and_query,
            &request_headers,
            body_bytes,
            transport,
        )
        .await;
    }
    if let Err(error) = lifecycle.validate_body(&method, &path_and_query, &body_bytes) {
        transport
            .send_error(reqwest::StatusCode::BAD_REQUEST, error.to_string())
            .await?;
        return Ok(());
    }
    if let Some(decision) = lifecycle.reject_queue_mismatch(app, &request_headers) {
        transport.send_queue_mismatch(app, &decision).await?;
        return Ok(());
    }
    lifecycle.start_queue_tracking(app);

    let response = match send_traced_upstream_request(
        app,
        method,
        &path_and_query,
        &request_headers,
        body_bytes,
    )
    .await
    {
        Ok(response) => response,
        Err(error) if app.retry.local_connect_failures_retryable && error.is_connect_failure() => {
            lifecycle.fail();
            transport
                .send_local_connect_failure(app, &error, true)
                .await?;
            return Ok(());
        }
        Err(error) if error.is_connect_failure() => {
            lifecycle.fail();
            transport
                .send_local_connect_failure(app, &error, false)
                .await?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let status = response.status();
    let response_headers = response.headers().clone();
    transport
        .send_success(
            status,
            &response_headers,
            &app.retry,
            app.metrics.as_deref(),
            &app.inference_server_id,
        )
        .await?;
    lifecycle.on_upstream_headers(&response_headers, status);

    if lifecycle.should_relay_sse(&response_headers) {
        lifecycle.relay_sse(app, response, transport).await?;
    } else {
        lifecycle.observe_raw_success(status);
        relay_response_body_raw(response.bytes_stream(), transport).await?;
    }

    transport.finish_response().await?;
    lifecycle.finish();
    lifecycle.finalize_quality_check(app, &request_headers);

    Ok(())
}

async fn forward_tunnel_health_request<Transport>(
    app: &TunnelServerApp,
    method: reqwest::Method,
    path_and_query: &str,
    request_headers: &HeaderMap,
    body_bytes: Vec<u8>,
    transport: &mut Transport,
) -> Result<()>
where
    Transport: TunnelRequestTransport,
{
    let response = match send_untraced_upstream_request(
        app,
        method,
        path_and_query,
        request_headers,
        body_bytes,
    )
    .await
    {
        Ok(response) => response,
        Err(error) if app.retry.local_connect_failures_retryable && error.is_connect_failure() => {
            transport
                .send_local_connect_failure(app, &error, true)
                .await?;
            return Ok(());
        }
        Err(error) if error.is_connect_failure() => {
            transport
                .send_local_connect_failure(app, &error, false)
                .await?;
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let status = response.status();
    let response_headers = response.headers().clone();
    transport
        .send_success(
            status,
            &response_headers,
            &app.retry,
            app.metrics.as_deref(),
            &app.inference_server_id,
        )
        .await?;
    relay_response_body_raw(response.bytes_stream(), transport).await?;
    transport.finish_response().await
}

#[derive(Debug, thiserror::Error)]
enum RequestBodyValidationError {
    #[error("request body must be valid JSON")]
    InvalidJson,
    #[error("{endpoint} requests must set stream=true")]
    StreamingEndpointMustStream { endpoint: &'static str },
}

async fn send_traced_upstream_request(
    app: &TunnelServerApp,
    method: reqwest::Method,
    path_and_query: &str,
    request_headers: &HeaderMap,
    body_bytes: Vec<u8>,
) -> std::result::Result<reqwest::Response, UpstreamRequestError> {
    let span = pylon_upstream_http_span(app, &method, path_and_query, request_headers);
    let upstream_headers = headers_for_traced_upstream_request(request_headers, &span);
    let result =
        send_upstream_request_inner(app, method, path_and_query, &upstream_headers, body_bytes)
            .instrument(span.clone())
            .await;
    record_pylon_upstream_result_to_span(&span, &result);
    result
}

async fn send_untraced_upstream_request(
    app: &TunnelServerApp,
    method: reqwest::Method,
    path_and_query: &str,
    request_headers: &HeaderMap,
    body_bytes: Vec<u8>,
) -> std::result::Result<reqwest::Response, UpstreamRequestError> {
    send_upstream_request_inner(app, method, path_and_query, request_headers, body_bytes).await
}

async fn send_upstream_request_inner(
    app: &TunnelServerApp,
    method: reqwest::Method,
    path_and_query: &str,
    request_headers: &HeaderMap,
    body_bytes: Vec<u8>,
) -> std::result::Result<reqwest::Response, UpstreamRequestError> {
    let request_url = join_base_path(&app.upstream_http_base_url, path_and_query)
        .map_err(UpstreamRequestError::Build)?;
    let mut request = app
        .http_client
        .request(method, request_url)
        .body(body_bytes);
    for (name, value) in request_headers {
        if should_forward_header(name) {
            request = request.header(name, value);
        }
    }
    request.send().await.map_err(UpstreamRequestError::Send)
}

fn pylon_upstream_http_span(
    app: &TunnelServerApp,
    method: &reqwest::Method,
    path_and_query: &str,
    request_headers: &HeaderMap,
) -> Span {
    let span = tracing::info_span!(
        "pylon_upstream_http_request",
        otel_parent = field::Empty,
        http.method = %method,
        http.path = %path_and_query,
        inference_server.id = %app.inference_server_id,
        upstream.status = field::Empty,
        upstream.error = field::Empty,
    );
    let _ = span.set_parent(pylon_upstream_parent_context(request_headers));
    if let Some(otel_parent) = otel_parent_from_headers(request_headers) {
        span.record("otel_parent", otel_parent);
    }
    span
}

fn headers_for_traced_upstream_request(request_headers: &HeaderMap, span: &Span) -> HeaderMap {
    let mut upstream_headers = request_headers.clone();
    inject_trace_context(&mut upstream_headers, &span.context());
    upstream_headers
}

pub(super) fn pylon_upstream_parent_context(headers: &HeaderMap) -> opentelemetry::Context {
    parent_context_from_headers(headers)
}

pub(super) fn otel_parent_from_headers(headers: &HeaderMap) -> Option<&str> {
    traceparent_from_headers(headers)
}

fn record_pylon_upstream_result_to_span(
    span: &Span,
    result: &std::result::Result<reqwest::Response, UpstreamRequestError>,
) {
    match result {
        Ok(response) => {
            span.record("upstream.status", response.status().as_u16());
        }
        Err(error) => {
            let error = error.to_string();
            span.record("upstream.error", error.as_str());
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum UpstreamRequestError {
    #[error("failed to build upstream request: {0}")]
    Build(#[source] anyhow::Error),
    #[error("upstream http request failed: {0}")]
    Send(#[source] reqwest::Error),
}

impl UpstreamRequestError {
    fn is_connect_failure(&self) -> bool {
        matches!(self, Self::Send(error) if error.is_connect())
    }
}

fn validate_request_body(
    method: &reqwest::Method,
    path_and_query: &str,
    body_bytes: &[u8],
) -> Result<(), RequestBodyValidationError> {
    if sonic_rs::get(body_bytes, &[] as &[&str]).is_err() {
        return Err(RequestBodyValidationError::InvalidJson);
    }

    if let Some(endpoint) =
        request_observation_endpoint(method, path_and_query).and_then(stream_required_endpoint_path)
        && !sonic_rs::get(body_bytes, &["stream"])
            .ok()
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    {
        return Err(RequestBodyValidationError::StreamingEndpointMustStream { endpoint });
    }

    Ok(())
}

pub(super) fn is_health_request_path(path_and_query: &str) -> bool {
    path_and_query
        .split('?')
        .next()
        .is_some_and(|path| path == "/health")
}

fn request_observation_endpoint(
    method: &reqwest::Method,
    path_and_query: &str,
) -> Option<RequestObservationEndpoint> {
    if method != reqwest::Method::POST {
        return None;
    }

    match path_and_query.split('?').next() {
        Some("/v1/chat/completions") => Some(RequestObservationEndpoint::ChatCompletions),
        Some("/v1/responses") => Some(RequestObservationEndpoint::Responses),
        Some("/v1/embeddings") => Some(RequestObservationEndpoint::Embeddings),
        _ => None,
    }
}

fn stream_required_endpoint_path(endpoint: RequestObservationEndpoint) -> Option<&'static str> {
    match endpoint {
        RequestObservationEndpoint::ChatCompletions => Some("/v1/chat/completions"),
        RequestObservationEndpoint::Responses => Some("/v1/responses"),
        RequestObservationEndpoint::Embeddings => None,
    }
}

fn is_sse_response(headers: &HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"))
}

pub(super) trait ResponseBodyEventSink {
    async fn send_body_event(&mut self, event: bytes::Bytes) -> Result<()>;
}

async fn relay_remaining_output<Sink>(
    upstream_messages: &mut UpstreamSseMessageStream,
    output_token_parser: &mut OutputTokenParser,
    observer: &mut RequestObserver,
    quality_recorder: Option<&mut RequestQualityRecorder>,
    queue_request: &mut Option<QueueTrackedRequestGuard>,
    body_sink: &mut Sink,
) -> Result<()>
where
    Sink: ResponseBodyEventSink,
{
    let Some(first_message) =
        read_next_upstream_sse_message(upstream_messages, observer, false).await?
    else {
        return Ok(());
    };

    relay_chunk_stats_fallback_output(
        first_message,
        upstream_messages,
        output_token_parser,
        observer,
        quality_recorder,
        queue_request,
        body_sink,
    )
    .await
}

async fn relay_chunk_stats_fallback_output<Sink>(
    first_message: ParsedSseMessage,
    upstream_messages: &mut UpstreamSseMessageStream,
    output_token_parser: &mut OutputTokenParser,
    observer: &mut RequestObserver,
    quality_recorder: Option<&mut RequestQualityRecorder>,
    queue_request: &mut Option<QueueTrackedRequestGuard>,
    body_sink: &mut Sink,
) -> Result<()>
where
    Sink: ResponseBodyEventSink,
{
    let mut saw_output = false;
    let mut next_message = Some(first_message);
    let mut quality_recorder = quality_recorder;
    loop {
        let parsed_message = match next_message.take() {
            Some(parsed_message) => parsed_message,
            None => {
                let Some(parsed_message) =
                    read_next_upstream_sse_message(upstream_messages, observer, saw_output).await?
                else {
                    return Ok(());
                };
                parsed_message
            }
        };

        let forward_event = Some(parsed_message.raw_event.clone());
        observe_output_message_if_needed(&parsed_message, observer, queue_request, &mut saw_output);
        if let SseMessage::ChatCompletionChunk { raw_data } = &parsed_message.message {
            let output_progress = if !observer.is_terminal() {
                output_token_parser.parse_output_token_progress(raw_data)
            } else {
                None
            };
            if let Some(progress) = output_progress {
                observe_output_token_progress(observer, progress);
            }
            if let Some(recorder) = quality_recorder.as_deref_mut() {
                recorder.observe_sse_chunk_with_token_progress(
                    raw_data,
                    output_progress.map(request_quality_output_token_progress),
                );
            }
        }

        if let Some(event) = forward_event {
            body_sink.send_body_event(event).await?;
        }
    }
}

async fn read_next_upstream_sse_message(
    upstream_messages: &mut UpstreamSseMessageStream,
    observer: &mut RequestObserver,
    saw_output: bool,
) -> Result<Option<ParsedSseMessage>> {
    match upstream_messages.try_next().await {
        Ok(Some(parsed_message)) => Ok(Some(parsed_message)),
        Ok(None) if saw_output => Ok(None),
        Ok(None) => {
            observer.fail();
            bail!("upstream stream ended before first output event");
        }
        Err(UpstreamSseReadError::Timeout(SseReadTimeoutPhase::SubsequentOutput)) => {
            observer.fail();
            bail!("timed out waiting for subsequent output event from upstream");
        }
        Err(UpstreamSseReadError::Timeout(SseReadTimeoutPhase::FirstOutput)) => {
            observer.fail();
            bail!("timed out waiting for first output event from upstream");
        }
        Err(UpstreamSseReadError::BufferLimitExceeded {
            max_buffer_bytes,
            buffered_bytes,
        }) => {
            observer.fail();
            bail!(
                "upstream SSE buffer exceeded {max_buffer_bytes} bytes while waiting for an event boundary (buffered {buffered_bytes} bytes)"
            );
        }
        Err(UpstreamSseReadError::Upstream(error)) => {
            observer.fail();
            Err(error.context("failed to read upstream response message"))
        }
    }
}

fn observe_output_message_if_needed(
    parsed_message: &ParsedSseMessage,
    observer: &mut RequestObserver,
    queue_request: &mut Option<QueueTrackedRequestGuard>,
    saw_output: &mut bool,
) {
    if parsed_message.message.counts_as_output() && !observer.is_terminal() {
        *saw_output = true;
        observe_queue_output(queue_request);
        observer.observe_output_message();
    }
}

fn observe_output_token_progress(observer: &mut RequestObserver, progress: OutputTokenProgress) {
    match progress {
        OutputTokenProgress::ExplicitCumulative { tokens, .. } => {
            observer.observe_output_tokens_generated_so_far(tokens);
        }
        OutputTokenProgress::EstimatedDelta { delta } => {
            observer.observe_output_tokens(delta);
        }
    }
}

fn request_quality_output_token_progress(
    progress: OutputTokenProgress,
) -> RequestOutputTokenProgress {
    match progress {
        OutputTokenProgress::ExplicitCumulative { tokens, delta } => {
            RequestOutputTokenProgress::Cumulative { tokens, delta }
        }
        OutputTokenProgress::EstimatedDelta { delta } => RequestOutputTokenProgress::Delta(delta),
    }
}

fn finalize_quality_check(
    request_headers: &HeaderMap,
    quality_recorder: Option<&RequestQualityRecorder>,
    quality_config: &RequestQualityMonitorConfig,
    metrics: Option<&PylonMetrics>,
) {
    let Some(recorder) = quality_recorder else {
        return;
    };
    if !recorder.has_observed_stream_output() {
        return;
    }
    let model_id = request_headers
        .get("x-model")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let (_quality_metrics, quality_result) = recorder.evaluate(quality_config);
    if let Some(metrics) = metrics {
        let result_label = if !quality_result.evaluated {
            "skipped"
        } else if quality_result.threshold_match_reason.is_some() {
            "matched"
        } else {
            "clean"
        };
        metrics.observe_quality_check_result(model_id, result_label);
        if let Some(reason) = quality_result.threshold_match_reason {
            metrics.observe_quality_threshold_match(model_id, reason);
        }
    }
}

async fn relay_response_body_raw<BodyStream, Sink>(
    mut body_stream: BodyStream,
    body_sink: &mut Sink,
) -> Result<()>
where
    BodyStream: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
    Sink: ResponseBodyEventSink,
{
    while let Some(chunk) = body_stream
        .try_next()
        .await
        .context("failed to read upstream response body")?
    {
        body_sink.send_body_event(chunk).await?;
    }
    Ok(())
}

pub(super) fn request_body_buffer(
    request_headers: &HeaderMap,
    max_request_body_bytes: usize,
) -> Result<Vec<u8>> {
    let capacity = request_body_capacity(request_headers, max_request_body_bytes)?;
    Ok(Vec::with_capacity(capacity.unwrap_or(0)))
}

pub(super) fn request_body_capacity(
    request_headers: &HeaderMap,
    max_request_body_bytes: usize,
) -> Result<Option<usize>> {
    let Some(value) = request_headers.get(reqwest::header::CONTENT_LENGTH) else {
        return Ok(None);
    };
    let Ok(value) = value.to_str() else {
        return Ok(None);
    };
    let Ok(content_length) = value.trim().parse::<usize>() else {
        return Ok(None);
    };
    ensure!(
        content_length <= max_request_body_bytes,
        "request body too large"
    );
    // Preallocate for honest small Content-Length values, but cap speculative
    // allocation so a legal large body cannot reserve tens of MiB up front.
    Ok(Some(
        content_length.min(MAX_SPECULATIVE_REQUEST_BODY_PREALLOC_BYTES),
    ))
}

pub(super) fn next_body_len(
    current: usize,
    chunk_len: usize,
    max_request_body_bytes: usize,
) -> Result<usize> {
    let next = current
        .checked_add(chunk_len)
        .context("request body length overflowed")?;
    ensure!(next <= max_request_body_bytes, "request body too large");
    Ok(next)
}

pub(super) fn extend_body_from_buf<B>(body_bytes: &mut Vec<u8>, chunk: &mut B)
where
    B: Buf,
{
    while chunk.has_remaining() {
        // Copy each contiguous slice directly out of the Buf; this avoids
        // materializing another Bytes value while still handling segmented Buf
        // implementations.
        let bytes = chunk.chunk();
        body_bytes.extend_from_slice(bytes);
        chunk.advance(bytes.len());
    }
}

pub(super) fn build_response_headers(
    status: reqwest::StatusCode,
    response_headers: &HeaderMap,
    retry: &PylonRetryConfig,
    metrics: Option<&PylonMetrics>,
    inference_server_id: &str,
    omit_content_length: bool,
) -> Result<HeaderMap> {
    let mut header_frame = HeaderMap::new();
    let classification = classify_upstream_response(status, response_headers, retry);
    log_upstream_response_classification(status, classification);
    if let Some(metrics) = metrics
        && !status.is_success()
    {
        if classification.retryable {
            metrics
                .retryable_responses_total(
                    inference_server_id,
                    classification.reason,
                    &status.as_u16().to_string(),
                )
                .inc();
        } else {
            metrics
                .nonretryable_failures_total(inference_server_id, classification.reason)
                .inc();
        }
    }

    if classification.retryable {
        header_frame.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRYABLE),
            HeaderValue::from_static("true"),
        );
        header_frame.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRY_REASON),
            HeaderValue::from_static(RETRY_REASON_UPSTREAM_ADMISSION_REJECTED),
        );
        if retry.propagate_retry_after
            && let Some(retry_after_ms) = retry_after_millis(response_headers)
        {
            header_frame.insert(
                HeaderName::from_static(HEADER_STARGATE_RETRY_AFTER_MS),
                HeaderValue::from_str(&retry_after_ms.to_string())
                    .context("invalid retry-after millis")?,
            );
        }
    }
    for (name, value) in response_headers {
        if should_forward_response_header(name, retry)
            && !(omit_content_length && name == reqwest::header::CONTENT_LENGTH)
        {
            header_frame.append(name, value.clone());
        }
    }
    Ok(header_frame)
}

macro_rules! emit_upstream_response_classification {
    ($level:expr, $status:expr, $classification:expr) => {
        tracing::event!(
            $level,
            upstream.status = $status.as_u16(),
            tunnel.retryable = $classification.retryable,
            tunnel.retry_reason = $classification.reason,
            upstream.retry_header_present = $classification.upstream_retry_header_present,
            "classified upstream response"
        )
    };
}

fn log_upstream_response_classification(
    status: reqwest::StatusCode,
    classification: UpstreamRetryClassification,
) {
    match upstream_classification_log_level(status, classification) {
        tracing::Level::INFO => {
            emit_upstream_response_classification!(tracing::Level::INFO, status, classification);
        }
        tracing::Level::DEBUG => {
            emit_upstream_response_classification!(tracing::Level::DEBUG, status, classification);
        }
        _ => unreachable!("upstream classification log level should be INFO or DEBUG"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpstreamRetryClassification {
    retryable: bool,
    reason: &'static str,
    upstream_retry_header_present: bool,
}

fn classify_upstream_response(
    status: reqwest::StatusCode,
    response_headers: &HeaderMap,
    retry: &PylonRetryConfig,
) -> UpstreamRetryClassification {
    let upstream_retry_header_present = response_headers
        .get(&retry.upstream_retry_header)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"));
    let status_retryable = retry.retryable_upstream_status_codes.contains(&status);
    let retryable =
        status_retryable && (!retry.require_upstream_retry_header || upstream_retry_header_present);
    let reason = if retryable {
        RETRY_REASON_UPSTREAM_ADMISSION_REJECTED
    } else if status_retryable
        && retry.require_upstream_retry_header
        && !upstream_retry_header_present
    {
        "missing_upstream_retry_header"
    } else if !status.is_success() {
        "upstream_nonretryable_status"
    } else {
        ""
    };

    UpstreamRetryClassification {
        retryable,
        reason,
        upstream_retry_header_present,
    }
}

fn should_log_upstream_classification_at_info(
    status: reqwest::StatusCode,
    classification: UpstreamRetryClassification,
) -> bool {
    classification.retryable || !status.is_success()
}

fn upstream_classification_log_level(
    status: reqwest::StatusCode,
    classification: UpstreamRetryClassification,
) -> tracing::Level {
    if should_log_upstream_classification_at_info(status, classification) {
        tracing::Level::INFO
    } else {
        tracing::Level::DEBUG
    }
}

fn retry_after_millis(response_headers: &HeaderMap) -> Option<u64> {
    let value = response_headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return seconds.checked_mul(1000);
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    let duration = retry_at
        .duration_since(SystemTime::now())
        .unwrap_or(Duration::ZERO);
    u64::try_from(duration.as_millis()).ok()
}

pub(super) fn queue_mismatch_response_headers(
    app: &TunnelServerApp,
    decision: &QueueAdmissionDecision,
    include_custom_status: bool,
) -> Result<HeaderMap> {
    let status = reqwest::StatusCode::TOO_MANY_REQUESTS;
    if let Some(metrics) = app.metrics.as_deref() {
        metrics
            .retryable_responses_total(
                &app.inference_server_id,
                RETRY_REASON_QUEUE_ESTIMATE_MISMATCH,
                &status.as_u16().to_string(),
            )
            .inc();
    }

    let mut headers = HeaderMap::new();
    if include_custom_status {
        headers.insert(
            HeaderName::from_static("x-status"),
            HeaderValue::from_static("429"),
        );
    }
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/problem+json"),
    );
    headers.insert(
        HeaderName::from_static(HEADER_STARGATE_RETRYABLE),
        HeaderValue::from_static("true"),
    );
    headers.insert(
        HeaderName::from_static(HEADER_STARGATE_RETRY_REASON),
        HeaderValue::from_static(RETRY_REASON_QUEUE_ESTIMATE_MISMATCH),
    );
    if let QueueAdmissionDecision::Rejected {
        retry_after_ms: Some(retry_after_ms),
        ..
    } = decision
    {
        headers.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRY_AFTER_MS),
            HeaderValue::from_str(&retry_after_ms.to_string())
                .context("invalid queue mismatch retry-after millis")?,
        );
    }
    Ok(headers)
}

pub(super) fn queue_mismatch_body(decision: &QueueAdmissionDecision) -> String {
    let (expected_ms, actual_ms, threshold_ms) = match decision {
        QueueAdmissionDecision::Rejected {
            expected_ms,
            actual_ms,
            threshold_ms,
            ..
        } => (*expected_ms, *actual_ms, *threshold_ms),
        _ => (0, 0, 0),
    };
    serde_json::json!({
        "type": "about:blank",
        "title": "Too Many Requests",
        "status": reqwest::StatusCode::TOO_MANY_REQUESTS.as_u16(),
        "detail": "local queue estimate exceeded Stargate routing estimate",
        "reason": RETRY_REASON_QUEUE_ESTIMATE_MISMATCH,
        "expected_queue_ms": expected_ms,
        "actual_queue_ms": actual_ms,
        "threshold_ms": threshold_ms,
    })
    .to_string()
}

pub(super) fn record_local_connect_failure(
    app: &TunnelServerApp,
    error: &UpstreamRequestError,
    retryable: bool,
) -> reqwest::StatusCode {
    tracing::warn!(
        inference_server_id = %app.inference_server_id,
        error = %error,
        retryable,
        "local upstream connection failed"
    );

    let status = reqwest::StatusCode::SERVICE_UNAVAILABLE;
    if let Some(metrics) = app.metrics.as_deref() {
        if retryable {
            metrics
                .retryable_responses_total(
                    &app.inference_server_id,
                    RETRY_REASON_LOCAL_CONNECT_FAILURE,
                    &status.as_u16().to_string(),
                )
                .inc();
        } else {
            metrics
                .nonretryable_failures_total(
                    &app.inference_server_id,
                    RETRY_REASON_LOCAL_CONNECT_FAILURE,
                )
                .inc();
        }
    }

    status
}

pub(super) fn problem_details_body(
    status: reqwest::StatusCode,
    detail: impl Into<String>,
) -> String {
    serde_json::json!({
        "type": "about:blank",
        "title": status.canonical_reason().unwrap_or("Error"),
        "status": status.as_u16(),
        "detail": detail.into(),
    })
    .to_string()
}

pub(super) fn join_base_path(base: &str, path_and_query: &str) -> Result<url::Url> {
    let base = url::Url::parse(base).context("invalid upstream_http_base_url")?;
    let pq = if path_and_query.starts_with('/') {
        path_and_query.to_string()
    } else {
        format!("/{path_and_query}")
    };
    let joined = base.join(&pq).context("join upstream path failed")?;
    Ok(joined)
}

pub(super) fn should_forward_header(name: &HeaderName) -> bool {
    // `HeaderName` is normalized by http/reqwest, so `as_str()` gives a stable
    // lowercase key without allocating on the header-forwarding hot path.
    if name.as_str() == HEADER_STARGATE_EXPECTED_QUEUE_MS {
        return false;
    }
    !matches!(
        name.as_str(),
        "connection"
            | "proxy-connection"
            | "keep-alive"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "upgrade"
            | "host"
            | "x-method"
            | "x-path"
    )
}

pub(super) fn should_forward_response_header(name: &HeaderName, retry: &PylonRetryConfig) -> bool {
    if name == retry.upstream_retry_header {
        return false;
    }
    // Keep response filtering allocation-free; this runs for every upstream
    // response header before the frame is written back through the tunnel.
    let name = name.as_str();
    !matches!(
        name,
        "connection"
            | "proxy-connection"
            | "keep-alive"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "upgrade"
            | "content-length"
            | HEADER_STARGATE_UPSTREAM_RETRYABLE
            | HEADER_STARGATE_RETRYABLE
            | HEADER_STARGATE_RETRY_REASON
            | HEADER_STARGATE_RETRY_AFTER_MS
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_admission_info_logging_is_reserved_for_rejections() {
        let rejected = QueueAdmissionDecision::Rejected {
            expected_ms: 10,
            actual_ms: 20,
            threshold_ms: 5,
            retry_after_ms: Some(125),
        };
        assert!(should_log_queue_admission_at_info(&rejected));
        assert_eq!(queue_admission_log_level(&rejected), tracing::Level::INFO);

        let accepted = QueueAdmissionDecision::Accepted {
            expected_ms: 10,
            actual_ms: 11,
            threshold_ms: 5,
        };
        assert!(!should_log_queue_admission_at_info(&accepted));
        assert_eq!(queue_admission_log_level(&accepted), tracing::Level::DEBUG);

        let missing_estimate = QueueAdmissionDecision::MissingEstimate;
        assert!(!should_log_queue_admission_at_info(&missing_estimate));
        assert_eq!(
            queue_admission_log_level(&missing_estimate),
            tracing::Level::DEBUG
        );

        let unknown_local = QueueAdmissionDecision::UnknownLocalEstimate { expected_ms: 10 };
        assert!(!should_log_queue_admission_at_info(&unknown_local));
        assert_eq!(
            queue_admission_log_level(&unknown_local),
            tracing::Level::DEBUG
        );

        let disabled = QueueAdmissionDecision::Disabled;
        assert!(!should_log_queue_admission_at_info(&disabled));
        assert_eq!(queue_admission_log_level(&disabled), tracing::Level::DEBUG);
    }

    #[test]
    fn upstream_classification_info_logging_is_reserved_for_actionable_results() {
        let success = UpstreamRetryClassification {
            retryable: false,
            reason: "",
            upstream_retry_header_present: false,
        };
        assert!(!should_log_upstream_classification_at_info(
            reqwest::StatusCode::OK,
            success
        ));
        assert_eq!(
            upstream_classification_log_level(reqwest::StatusCode::OK, success),
            tracing::Level::DEBUG
        );

        let retryable = UpstreamRetryClassification {
            retryable: true,
            reason: RETRY_REASON_UPSTREAM_ADMISSION_REJECTED,
            upstream_retry_header_present: true,
        };
        assert!(should_log_upstream_classification_at_info(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            retryable
        ));
        assert_eq!(
            upstream_classification_log_level(reqwest::StatusCode::TOO_MANY_REQUESTS, retryable),
            tracing::Level::INFO
        );

        let non_success = UpstreamRetryClassification {
            retryable: false,
            reason: "upstream_nonretryable_status",
            upstream_retry_header_present: false,
        };
        assert!(should_log_upstream_classification_at_info(
            reqwest::StatusCode::BAD_GATEWAY,
            non_success
        ));
        assert_eq!(
            upstream_classification_log_level(reqwest::StatusCode::BAD_GATEWAY, non_success),
            tracing::Level::INFO
        );
    }
}
