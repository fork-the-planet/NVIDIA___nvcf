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

use anyhow::Result;
use stargate::registration::{
    DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT, DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT,
};
use stargate_protocol::TunnelTransportProtocol;
use stargate_runtime::wait_for_termination_signal;
use tracing::{error, info};

#[path = "main/startup.rs"]
mod startup;

mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

const DEFAULT_PROXY_MAX_REPLAY_BODY_BYTES: usize = 64 * 1024 * 1024;

fn parse_nonzero_millis(value: &str) -> std::result::Result<u64, String> {
    let millis = value
        .parse::<u64>()
        .map_err(|err| format!("invalid millisecond value: {err}"))?;

    if millis == 0 {
        return Err("value must be greater than 0".to_string());
    }

    Ok(millis)
}

fn parse_nonzero_usize(value: &str) -> std::result::Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|err| format!("invalid count: {err}"))?;

    if count == 0 {
        return Err("value must be greater than 0".to_string());
    }

    Ok(count)
}

#[derive(clap::Parser, Debug)]
#[command(name = "stargate")]
struct Args {
    /// Stable Stargate process or pod identity.
    #[arg(long, value_name = "ID")]
    stargate_id: String,

    /// Local TCP socket for backend-facing WatchStargates and registration.
    #[arg(long, default_value = "0.0.0.0:50071", value_name = "ADDR")]
    listen_addr: String,

    /// Local TCP socket for frontend-facing model discovery (`ListModels`).
    #[arg(long, default_value = "0.0.0.0:50073", value_name = "ADDR")]
    model_discovery_listen_addr: String,

    /// Local HTTP socket for proxy traffic, health probes, and metrics.
    #[arg(long, default_value = "0.0.0.0:8000", value_name = "ADDR")]
    http_listen_addr: String,

    /// Self gRPC address published by non-Kubernetes discovery and used as the
    /// port source for Kubernetes advertised hostnames.
    #[arg(long, value_name = "ADDR")]
    advertise_addr: SocketAddr,

    /// DNS name used for Stargate peer discovery.
    ///
    /// In Kubernetes this should be the headless Service so EndpointSlice
    /// readiness controls peer visibility and, when explicitly enabled,
    /// development-only relay targets.
    #[arg(long, value_name = "DNS_NAME")]
    stargate_discovery_dns_name: String,

    /// Additional WatchStargates endpoints for remote regions. Repeatable.
    ///
    /// These are recursive watch seeds, not registration targets. Pylons only
    /// register to concrete `stargates` entries returned by watch snapshots.
    #[arg(
        long,
        env = "STARGATE_REMOTE_WATCH_URLS",
        value_delimiter = ',',
        value_name = "URL"
    )]
    remote_stargate_url: Vec<String>,

    /// Optional pylon dial address for backend-facing Stargate gRPC.
    ///
    /// Stargate still advertises per-pod addresses as gRPC authority/SNI
    /// identity, and sends this address separately so pylons can connect
    /// through a TCP load balancer.
    #[arg(long, value_name = "ADDR")]
    grpc_pylon_dial_addr: Option<String>,

    /// Backend-facing advertised hostname template.
    ///
    /// Supports `{pod_name}` and `{namespace}`. In Kubernetes this rendered host
    /// becomes the pylon gRPC authority and reverse QUIC SNI so routers can
    /// select the intended Stargate pod.
    #[arg(long, value_name = "TEMPLATE")]
    advertised_hostname_template: Option<String>,

    /// Pod name used to render `--advertised-hostname-template`.
    #[arg(long, env = "POD_NAME", value_name = "NAME")]
    pod_name: Option<String>,

    /// Pod namespace used to render `--advertised-hostname-template`.
    #[arg(long, env = "POD_NAMESPACE", value_name = "NAMESPACE")]
    pod_namespace: Option<String>,

    /// Publish only this Stargate in WatchStargates instead of DNS-discovered peers.
    #[arg(long, default_value_t = false)]
    disable_dns_discovery: bool,

    /// Enable development-only peer relaying for backend gRPC and reverse QUIC traffic.
    ///
    /// Requires Kubernetes pod identity and DNS discovery. This relay must not run in
    /// production; use `stargate-k8s-router` or a supported load-balancer topology instead.
    #[arg(long, default_value_t = false)]
    enable_dev_peer_forwarding: bool,

    /// Interval for refreshing DNS-discovered Stargate peers.
    #[arg(long, default_value_t = 1000, value_parser = parse_nonzero_millis, value_name = "MS")]
    dns_poll_ms: u64,

    /// Maximum resolver cache TTL used by Stargate DNS discovery.
    #[arg(long, default_value_t = 1000, value_name = "MS")]
    dns_resolver_ttl_ms: u64,

    /// Maximum interval between unchanged WatchStargates snapshots.
    #[arg(long, default_value_t = 5000, value_name = "MS")]
    watch_heartbeat_ms: u64,

    /// Minimum idle timeout for heartbeat-aware registration streams; 0 disables all enforcement
    #[arg(
        long,
        default_value_t = DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT.as_millis() as u64,
        env = "STARGATE_REGISTRATION_UPDATE_IDLE_TIMEOUT_MS",
        value_name = "MS"
    )]
    registration_update_idle_timeout_ms: u64,

    /// Maximum idle timeout for heartbeat-aware registration streams; 0 disables all enforcement
    #[arg(
        long,
        default_value_t = DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT.as_millis() as u64,
        env = "STARGATE_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT_MS",
        value_name = "MS"
    )]
    registration_update_max_idle_timeout_ms: u64,

    /// Grace period for shutdown tasks after Stargate starts draining.
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    shutdown_drain_timeout_ms: u64,

    /// Timeout for establishing outbound direct QUIC connections and
    /// development-only peer relays.
    #[arg(long, default_value_t = 2000, value_name = "MS")]
    quic_connect_timeout_ms: u64,

    /// Timeout for each proxied request over an established QUIC tunnel.
    #[arg(long, default_value_t = 30000, value_name = "MS")]
    quic_request_timeout_ms: u64,

    /// Number of direct QUIC connections opened per backend.
    #[arg(
        long,
        default_value_t = 1,
        env = "STARGATE_DIRECT_QUIC_CONNECTIONS",
        value_parser = parse_nonzero_usize,
        value_name = "N"
    )]
    direct_quic_connections: usize,

    /// Maximum direct QUIC reconnect attempts on the proxy hot path
    #[arg(
        long,
        default_value_t = 2,
        env = "STARGATE_PROXY_MAX_CONNECT_RETRIES",
        value_name = "N"
    )]
    proxy_max_connect_retries: u32,

    /// Maximum retries for explicit retryable upstream responses
    #[arg(
        long,
        default_value_t = 2,
        env = "STARGATE_PROXY_MAX_REQUEST_RETRIES",
        value_name = "N"
    )]
    proxy_max_request_retries: u32,

    /// Maximum request body bytes buffered for proxy retry replay
    #[arg(
        long,
        default_value_t = DEFAULT_PROXY_MAX_REPLAY_BODY_BYTES,
        env = "STARGATE_PROXY_MAX_REPLAY_BODY_BYTES",
        value_name = "BYTES"
    )]
    proxy_max_replay_body_bytes: usize,

    /// Require pylon's explicit retry signal before retrying upstream status responses
    #[arg(
        long,
        action = clap::ArgAction::Set,
        default_value_t = true,
        env = "STARGATE_PROXY_REQUIRE_PYLON_RETRY_SIGNAL"
    )]
    proxy_require_pylon_retry_signal: bool,

    /// Request header carrying the retry budget in milliseconds; empty disables budget headers
    #[arg(
        long,
        default_value = "x-stargate-max-wait-ms",
        env = "STARGATE_PROXY_RETRY_BUDGET_HEADER",
        value_name = "HEADER"
    )]
    proxy_retry_budget_header: String,

    /// TLS certificate PEM for QUIC listeners. Generates self-signed if omitted.
    #[arg(long, env = "STARGATE_TLS_CERT_PATH", value_name = "PATH")]
    tls_cert_path: Option<String>,

    /// TLS private key PEM for QUIC listeners. Generates self-signed if omitted.
    #[arg(long, env = "STARGATE_TLS_KEY_PATH", value_name = "PATH")]
    tls_key_path: Option<String>,

    /// Skip QUIC TLS certificate verification for outbound connections and relays.
    #[arg(long, default_value_t = false, env = "STARGATE_QUIC_INSECURE")]
    quic_insecure: bool,

    /// Path to load balancer config JSON file (uses power-of-two default if omitted)
    #[arg(long, value_name = "PATH")]
    lb_config_path: Option<String>,

    /// OTLP/gRPC trace export endpoint. Tracing export is disabled if omitted.
    #[arg(long, value_name = "ENDPOINT")]
    otel_endpoint: Option<String>,

    /// OpenTelemetry service.name resource and tracer name.
    #[arg(long, default_value = stargate::telemetry::DEFAULT_SERVICE_NAME, value_name = "NAME")]
    otel_service_name: String,

    /// Port for Prometheus metrics HTTP server
    #[arg(long, default_value_t = 9090, value_name = "PORT")]
    metrics_port: u16,

    /// Prefix prepended to all Prometheus metric names.
    #[arg(long, default_value = stargate::metrics::DEFAULT_PREFIX, value_name = "PREFIX")]
    metrics_prefix: String,

    /// Local UDP socket for reverse QUIC tunnel connections from pylons.
    #[arg(long, value_name = "ADDR")]
    reverse_tunnel_listen_addr: Option<String>,

    /// Optional pylon dial address for reverse QUIC tunnels.
    ///
    /// Stargate still sends the per-pod reverse tunnel target as QUIC SNI
    /// identity, and sends this address separately so pylons can connect through
    /// a UDP load balancer.
    #[arg(long, value_name = "ADDR")]
    reverse_tunnel_pylon_dial_addr: Option<String>,

    /// Timeout waiting for a reverse tunnel connection after registration.
    #[arg(long, default_value_t = 10000, value_name = "MS")]
    reverse_tunnel_connect_timeout_ms: u64,

    /// Tunnel protocol used for proxied request streams; must match pylon.
    #[arg(long, default_value_t = TunnelTransportProtocol::Custom, value_name = "PROTOCOL")]
    tunnel_protocol: TunnelTransportProtocol,

    /// gRPC endpoint for worker authentication (e.g. http://llm-gateway:50051)
    #[arg(long, value_name = "URL")]
    worker_auth_endpoint: Option<String>,

    /// JSON secrets file path for worker-auth bearer tokens.
    #[arg(long, env = "SECRETS_PATH", value_name = "PATH")]
    secrets_path: Option<String>,

    /// Dot-separated JSON path to the auth token inside the secrets file.
    #[arg(long, env = "SECRETS_JSON_PATH", value_name = "PATH")]
    secrets_json_path: Option<String>,

    /// OAuth2 provider host for client-credentials worker auth. When set, the
    /// worker-auth client mints tokens at `<host>/token` using the id/secret
    /// from the secrets file instead of reading a static bearer token.
    #[arg(long, env = "OAUTH2_PROVIDER_HOST", value_name = "URL")]
    oauth2_provider_host: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let args = <Args as clap::Parser>::parse();
    run(args).await
}

async fn run(args: Args) -> Result<()> {
    let _telemetry_guard = stargate::telemetry::init_telemetry(
        args.otel_endpoint.as_deref(),
        &args.otel_service_name,
    )?;
    log_startup(&args);

    let startup::RuntimeStartup {
        runtime,
        shutdown_drain_timeout,
    } = startup::runtime_from_args(args).await?;

    let handle = runtime.start().await?;

    let shutdown_error = match wait_for_runtime_shutdown_trigger(
        wait_for_termination_signal(),
        handle.wait_for_critical_failure(),
    )
    .await
    {
        RuntimeShutdownTrigger::Signal(Ok(first_signal)) => {
            info!(
                signal = first_signal,
                "received termination signal, beginning graceful shutdown"
            );
            None
        }
        RuntimeShutdownTrigger::Signal(Err(error)) => {
            let error =
                anyhow::Error::new(error).context("failed to receive stargate termination signal");
            error!(error = %error, "termination signal handler failed, beginning graceful shutdown");
            Some(error)
        }
        RuntimeShutdownTrigger::Critical(failure) => {
            error!(error = %failure, "critical stargate task failed, beginning graceful shutdown");
            Some(failure.into())
        }
    };
    handle.begin_shutdown();

    tokio::select! {
        completed = handle.wait_for_shutdown(shutdown_drain_timeout) => {
            if completed {
                info!("graceful shutdown complete");
            } else {
                info!(timeout_ms = shutdown_drain_timeout.as_millis(), "graceful shutdown timed out; forcing exit");
                std::process::exit(1);
            }
        }
        second_signal = wait_for_termination_signal() => {
            match second_signal {
                Ok(signal) => info!(signal, "received second termination signal; forcing immediate exit"),
                Err(error) => error!(error = %error, "termination signal handler failed during shutdown; forcing immediate exit"),
            }
            std::process::exit(1);
        }
    };

    match shutdown_error {
        Some(error) => {
            info!("stargate shutdown complete after process failure");
            Err(error)
        }
        None => {
            info!("stargate stopped cleanly");
            Ok(())
        }
    }
}

fn log_startup(args: &Args) {
    info!(
        version = built_info::PKG_VERSION,
        commit_short_sha = built_info::GIT_COMMIT_HASH_SHORT.unwrap_or("unknown"),
        config = ?args,
        "starting stargate"
    );
}

enum RuntimeShutdownTrigger<Signal, Failure> {
    Signal(Signal),
    Critical(Failure),
}

async fn wait_for_runtime_shutdown_trigger<Signal, Failure, SignalOutput, FailureOutput>(
    signal: Signal,
    failure: Failure,
) -> RuntimeShutdownTrigger<SignalOutput, FailureOutput>
where
    Signal: Future<Output = SignalOutput>,
    Failure: Future<Output = FailureOutput>,
{
    tokio::select! {
        signal = signal => RuntimeShutdownTrigger::Signal(signal),
        failure = failure => RuntimeShutdownTrigger::Critical(failure),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::process::Command;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::*;

    #[derive(Clone)]
    struct TestLogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for TestLogWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("test log writer lock should not be poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn capture_startup_log(args: &Args) -> String {
        let output = Arc::new(Mutex::new(Vec::new()));
        let writer_output = Arc::clone(&output);
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_target(false)
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || TestLogWriter(Arc::clone(&writer_output)))
            .finish();

        tracing::subscriber::with_default(subscriber, || log_startup(args));

        String::from_utf8(
            output
                .lock()
                .expect("test log output lock should not be poisoned")
                .clone(),
        )
        .expect("startup log output should be UTF-8")
    }

    fn try_parse_args(extra: &[&str]) -> std::result::Result<Args, clap::Error> {
        let mut args = vec![
            "stargate",
            "--stargate-id",
            "sg-test",
            "--advertise-addr",
            "127.0.0.1:50071",
            "--stargate-discovery-dns-name",
            "stargate.local",
        ];
        args.extend_from_slice(extra);
        <Args as clap::Parser>::try_parse_from(args)
    }

    fn parse_args(extra: &[&str]) -> Args {
        try_parse_args(extra).expect("args should parse")
    }

    #[test]
    fn startup_log_includes_build_identity_and_complete_args() {
        let args = parse_args(&[
            "--proxy-max-connect-retries",
            "7",
            "--worker-auth-endpoint",
            "http://worker-auth.example.test:50051",
            "--secrets-path",
            "/var/run/secrets/worker-auth.json",
            "--secrets-json-path",
            "auth.token",
        ]);

        let output = capture_startup_log(&args);
        let expected_commit = built_info::GIT_COMMIT_HASH_SHORT.unwrap_or("unknown");

        assert!(
            output.contains(&format!("version=\"{}\"", built_info::PKG_VERSION)),
            "startup log: {output}"
        );
        assert!(
            output.contains(&format!("commit_short_sha=\"{expected_commit}\"")),
            "startup log: {output}"
        );
        assert!(
            output.contains(&format!("config={args:?}")),
            "startup log: {output}"
        );
    }

    #[test]
    fn build_identity_matches_checked_out_commit() {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git should be available when running repository tests");
        assert!(
            output.status.success(),
            "git rev-parse failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let expected_commit = String::from_utf8(output.stdout)
            .expect("git commit should be UTF-8")
            .trim()
            .to_owned();
        assert_eq!(
            built_info::GIT_COMMIT_HASH,
            Some(expected_commit.as_str()),
            "generated build identity should match the checked-out commit"
        );
    }

    #[test]
    fn dns_poll_ms_zero_is_rejected() {
        let error =
            try_parse_args(&["--dns-poll-ms", "0"]).expect_err("zero dns poll should be rejected");

        assert!(
            error.to_string().contains("greater than 0"),
            "unexpected clap error: {error}"
        );
    }

    #[test]
    fn registration_update_idle_timeout_default_matches_runtime_default() {
        let args = parse_args(&[]);

        assert_eq!(
            Duration::from_millis(args.registration_update_idle_timeout_ms),
            DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT
        );
        assert_eq!(
            Duration::from_millis(args.registration_update_max_idle_timeout_ms),
            DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT
        );
    }

    #[test]
    fn direct_quic_connections_default_and_override_parse() {
        let defaults = parse_args(&[]);
        assert_eq!(defaults.direct_quic_connections, 1);

        let overridden = parse_args(&["--direct-quic-connections", "4"]);
        assert_eq!(overridden.direct_quic_connections, 4);
    }

    #[test]
    fn direct_quic_connections_zero_is_rejected() {
        let error = try_parse_args(&["--direct-quic-connections", "0"])
            .expect_err("zero direct QUIC connections should be rejected");

        assert!(
            error.to_string().contains("greater than 0"),
            "unexpected clap error: {error}"
        );
    }

    #[test]
    fn dev_peer_forwarding_defaults_to_false_and_requires_explicit_opt_in() {
        let defaults = parse_args(&[]);
        assert!(!defaults.enable_dev_peer_forwarding);

        let opted_in = parse_args(&["--enable-dev-peer-forwarding"]);
        assert!(opted_in.enable_dev_peer_forwarding);
    }

    #[test]
    fn model_discovery_listen_addr_default_and_override_parse() {
        let defaults = parse_args(&[]);
        assert_eq!(defaults.model_discovery_listen_addr, "0.0.0.0:50073");

        let overridden = parse_args(&["--model-discovery-listen-addr", "127.0.0.1:50173"]);
        assert_eq!(overridden.model_discovery_listen_addr, "127.0.0.1:50173");
    }

    #[test]
    fn observability_names_default_and_override_parse() {
        let defaults = parse_args(&[]);
        assert_eq!(
            defaults.otel_service_name,
            stargate::telemetry::DEFAULT_SERVICE_NAME
        );
        assert_eq!(defaults.metrics_prefix, stargate::metrics::DEFAULT_PREFIX);

        let overridden = parse_args(&[
            "--otel-service-name",
            "llm-request-router",
            "--metrics-prefix",
            "llm_request_router_",
        ]);
        assert_eq!(overridden.otel_service_name, "llm-request-router");
        assert_eq!(overridden.metrics_prefix, "llm_request_router_");
    }

    #[tokio::test]
    async fn runtime_shutdown_trigger_reports_signal() {
        let trigger = wait_for_runtime_shutdown_trigger(
            async { "SIGTERM" },
            std::future::pending::<&'static str>(),
        )
        .await;

        assert!(matches!(trigger, RuntimeShutdownTrigger::Signal("SIGTERM")));
    }

    #[tokio::test]
    async fn runtime_shutdown_trigger_reports_critical_failure() {
        let trigger =
            wait_for_runtime_shutdown_trigger(std::future::pending::<&'static str>(), async {
                "HTTP proxy server"
            })
            .await;

        assert!(matches!(
            trigger,
            RuntimeShutdownTrigger::Critical("HTTP proxy server")
        ));
    }

    #[test]
    fn otel_endpoint_help_matches_grpc_exporter_transport() {
        let mut command = <Args as clap::CommandFactory>::command();
        let mut help = Vec::new();
        command
            .write_long_help(&mut help)
            .expect("help should render");
        let help = std::str::from_utf8(&help).expect("help should be UTF-8");

        assert!(help.contains("OTLP/gRPC trace export endpoint"));
        assert!(!help.contains("OTLP/HTTP/protobuf trace export endpoint"));
    }

    #[test]
    fn registration_update_idle_timeout_cli_override_is_applied() {
        let args = parse_args(&[
            "--registration-update-idle-timeout-ms",
            "120000",
            "--registration-update-max-idle-timeout-ms",
            "600000",
        ]);

        assert_eq!(args.registration_update_idle_timeout_ms, 120_000);
        assert_eq!(args.registration_update_max_idle_timeout_ms, 600_000);
    }

    #[test]
    fn registration_update_idle_timeout_zero_disables_enforcement() {
        let args = parse_args(&[
            "--registration-update-idle-timeout-ms",
            "0",
            "--registration-update-max-idle-timeout-ms",
            "0",
        ]);

        assert_eq!(args.registration_update_idle_timeout_ms, 0);
        assert_eq!(args.registration_update_max_idle_timeout_ms, 0);
    }

    #[test]
    fn tunnel_protocol_cli_defaults_to_custom() {
        let args = parse_args(&[]);

        assert_eq!(args.tunnel_protocol, TunnelTransportProtocol::Custom);
    }

    #[test]
    fn tunnel_protocol_cli_accepts_http3() {
        let args = parse_args(&["--tunnel-protocol", "http3"]);

        assert_eq!(args.tunnel_protocol, TunnelTransportProtocol::Http3);
    }

    #[test]
    fn tunnel_protocol_cli_accepts_webtransport() {
        let args = parse_args(&["--tunnel-protocol", "webtransport"]);

        assert_eq!(args.tunnel_protocol, TunnelTransportProtocol::WebTransport);
    }

    #[test]
    fn reverse_tunnel_pylon_dial_addr_cli_is_optional_and_parseable() {
        let defaults = parse_args(&[]);
        assert_eq!(defaults.reverse_tunnel_pylon_dial_addr, None);
        assert_eq!(defaults.grpc_pylon_dial_addr, None);

        let args = parse_args(&[
            "--grpc-pylon-dial-addr",
            "stargate-grpc-lb.stargate.svc.cluster.local:443",
            "--reverse-tunnel-listen-addr",
            "0.0.0.0:50072",
            "--reverse-tunnel-pylon-dial-addr",
            "stargate-quic-lb.stargate.svc.cluster.local:50072",
        ]);

        assert_eq!(
            args.reverse_tunnel_pylon_dial_addr.as_deref(),
            Some("stargate-quic-lb.stargate.svc.cluster.local:50072")
        );
        assert_eq!(
            args.grpc_pylon_dial_addr.as_deref(),
            Some("stargate-grpc-lb.stargate.svc.cluster.local:443")
        );
    }
}
