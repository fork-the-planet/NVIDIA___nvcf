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

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use stargate::discovery::{
    Discovery, DnsDiscovery, HeadlessDnsDiscovery, HeadlessDnsDiscoveryConfig, SelfOnlyDiscovery,
};
use stargate::proxy::{ProxyRetryConfig, ProxyTransportConfig};
use stargate::runtime::{ReverseTunnelConfig, StargateRuntime, StargateRuntimeConfig};
use stargate_forwarding::{ForwardingResolver, HeadlessDnsResolver, render_hostname};
use stargate_tls::ServerTlsIdentity;

use super::Args;

pub(super) struct RuntimeStartup {
    pub(super) runtime: StargateRuntime,
    pub(super) shutdown_drain_timeout: Duration,
}

struct DiscoveryAndForwarding {
    discovery: Box<dyn Discovery>,
    forwarding: Option<Arc<dyn ForwardingResolver>>,
}

struct WorkerAuthStartup {
    endpoint: String,
    token_provider: Option<stargate_auth::AuthTokenProvider>,
}

pub(super) async fn runtime_from_args(mut args: Args) -> Result<RuntimeStartup> {
    let DiscoveryAndForwarding {
        discovery,
        forwarding,
    } = make_discovery(&args)?;
    let proxy_retry_config = proxy_retry_config_from_args(&args)?;
    let reverse_tunnel = reverse_tunnel_config_from_args(&args)?;
    let proxy_transport =
        proxy_transport_config_from_args(&args, reverse_tunnel.is_some(), proxy_retry_config)?;
    let shutdown_drain_timeout = Duration::from_millis(args.shutdown_drain_timeout_ms);
    let worker_auth = worker_auth_startup_from_args(
        args.worker_auth_endpoint.take(),
        args.secrets_path.take(),
        args.secrets_json_path.take(),
        args.oauth2_provider_host.take(),
    )?;

    let runtime_config = runtime_config_from_args(args, proxy_transport, reverse_tunnel)?;
    let mut runtime = StargateRuntime::new(runtime_config, discovery);
    if let Some(forwarding) = forwarding {
        runtime = runtime.with_forwarding(forwarding);
    }
    if let Some(auth) = worker_auth {
        let authenticator =
            stargate::auth::GrpcWorkerAuthenticator::connect(&auth.endpoint, auth.token_provider)
                .await
                .context("failed to connect to worker auth endpoint")?;
        runtime = runtime.with_authenticator(Arc::new(authenticator));
    }

    Ok(RuntimeStartup {
        runtime,
        shutdown_drain_timeout,
    })
}

fn proxy_transport_config_from_args(
    args: &Args,
    reverse_tunnel_enabled: bool,
    retry: ProxyRetryConfig,
) -> Result<ProxyTransportConfig> {
    let tls_cert_pem = args.tls_cert_path.as_ref().map(std::fs::read).transpose()?;
    let tls_key_pem = args.tls_key_path.as_ref().map(std::fs::read).transpose()?;
    let server_tls_identity = server_tls_identity_for_reverse_listener(
        reverse_tunnel_enabled,
        tls_cert_pem.clone(),
        tls_key_pem,
    )?;
    Ok(ProxyTransportConfig {
        quic_connect_timeout: Duration::from_millis(args.quic_connect_timeout_ms),
        quic_request_timeout: Duration::from_millis(args.quic_request_timeout_ms),
        tls_cert_pem,
        server_tls_identity,
        quic_insecure: args.quic_insecure,
        tunnel_protocol: args.tunnel_protocol,
        direct_quic_connections: args.direct_quic_connections,
        retry,
    })
}

fn reverse_tunnel_config_from_args(args: &Args) -> Result<Option<ReverseTunnelConfig>> {
    let Some(listen_addr) = args
        .reverse_tunnel_listen_addr
        .as_deref()
        .map(str::parse)
        .transpose()?
    else {
        return Ok(None);
    };
    let hostname_template = args
        .advertised_hostname_template
        .as_deref()
        .unwrap_or("{pod_name}.stargate.external");
    let advertised_host = render_hostname(
        hostname_template,
        args.pod_name.as_deref().unwrap_or(&args.stargate_id),
        args.pod_namespace.as_deref().unwrap_or(""),
    );
    Ok(Some(ReverseTunnelConfig {
        listen_addr,
        advertised_host,
        pylon_dial_addr: args.reverse_tunnel_pylon_dial_addr.clone(),
        connect_timeout: Duration::from_millis(args.reverse_tunnel_connect_timeout_ms),
    }))
}

fn runtime_config_from_args(
    args: Args,
    proxy_transport: ProxyTransportConfig,
    reverse_tunnel: Option<ReverseTunnelConfig>,
) -> Result<StargateRuntimeConfig> {
    Ok(StargateRuntimeConfig {
        stargate_id: args.stargate_id,
        grpc_listen_addr: args.listen_addr.parse()?,
        model_discovery_listen_addr: args.model_discovery_listen_addr.parse()?,
        http_listen_addr: args.http_listen_addr.parse()?,
        metrics_listen_addr: Some(format!("0.0.0.0:{}", args.metrics_port).parse()?),
        advertise_addr: args.advertise_addr,
        stargate_discovery_dns_name: args.stargate_discovery_dns_name,
        remote_watch_stargate_urls: args.remote_stargate_url,
        grpc_pylon_dial_addr: args.grpc_pylon_dial_addr,
        dns_poll_interval: Duration::from_millis(args.dns_poll_ms),
        watch_heartbeat_interval: Duration::from_millis(args.watch_heartbeat_ms),
        registration_update_idle_timeout: Duration::from_millis(
            args.registration_update_idle_timeout_ms,
        ),
        registration_update_max_idle_timeout: Duration::from_millis(
            args.registration_update_max_idle_timeout_ms,
        ),
        proxy_transport,
        lb_config_path: args.lb_config_path,
        metrics_prefix: args.metrics_prefix,
        reverse_tunnel,
    })
}

fn make_discovery(args: &Args) -> Result<DiscoveryAndForwarding> {
    make_discovery_with_resolver(args, make_resolver)
}

fn make_discovery_with_resolver(
    args: &Args,
    make_resolver: impl FnOnce(Duration) -> Result<hickory_resolver::TokioAsyncResolver>,
) -> Result<DiscoveryAndForwarding> {
    if args.disable_dns_discovery {
        if args.enable_dev_peer_forwarding {
            bail!("--enable-dev-peer-forwarding cannot be combined with --disable-dns-discovery");
        }
        let http_listen_addr: SocketAddr = args.http_listen_addr.parse()?;
        return Ok(DiscoveryAndForwarding {
            discovery: Box::new(SelfOnlyDiscovery::new(
                args.advertise_addr,
                args.stargate_id.clone(),
                http_listen_addr.port(),
            )) as Box<dyn Discovery>,
            forwarding: None,
        });
    }

    match (&args.pod_name, &args.pod_namespace) {
        (Some(pod_name), Some(pod_namespace)) => {
            let dns_resolver_ttl = Duration::from_millis(args.dns_resolver_ttl_ms);
            let template = args
                .advertised_hostname_template
                .clone()
                .unwrap_or_else(|| "{pod_name}.stargate.external".to_string());
            let forwarding = if args.enable_dev_peer_forwarding {
                tracing::warn!(
                    development_only = true,
                    stargate_id = args.stargate_id.as_str(),
                    "development-only peer forwarding is enabled; it must not run in production"
                );
                Some(Arc::new(HeadlessDnsResolver {
                    self_pod_name: pod_name.clone(),
                    advertised_hostname_template: template.clone(),
                    namespace: pod_namespace.clone(),
                    headless_dns_suffix: args.stargate_discovery_dns_name.clone(),
                }) as Arc<dyn ForwardingResolver>)
            } else {
                None
            };
            let resolver = make_resolver(dns_resolver_ttl)?;
            let discovery = Box::new(HeadlessDnsDiscovery::new(HeadlessDnsDiscoveryConfig {
                self_pod_name: pod_name.clone(),
                pod_namespace: pod_namespace.clone(),
                advertised_hostname_template: template,
                discovery_dns_name: args.stargate_discovery_dns_name.clone(),
                resolver,
                grpc_port: args.advertise_addr.port(),
            })) as Box<dyn Discovery>;
            Ok(DiscoveryAndForwarding {
                discovery,
                forwarding,
            })
        }
        _ if args.enable_dev_peer_forwarding => {
            bail!("--enable-dev-peer-forwarding requires both --pod-name and --pod-namespace");
        }
        _ => {
            let http_listen_addr: SocketAddr = args.http_listen_addr.parse()?;
            let dns_resolver_ttl = Duration::from_millis(args.dns_resolver_ttl_ms);
            let resolver = make_resolver(dns_resolver_ttl)?;
            Ok(DiscoveryAndForwarding {
                discovery: Box::new(DnsDiscovery::new(
                    args.advertise_addr,
                    args.stargate_id.clone(),
                    args.stargate_discovery_dns_name.clone(),
                    resolver,
                    http_listen_addr.port(),
                )) as Box<dyn Discovery>,
                forwarding: None,
            })
        }
    }
}

fn proxy_retry_config_from_args(args: &Args) -> Result<ProxyRetryConfig> {
    let request_retry_budget_ms_header = match args.proxy_retry_budget_header.trim() {
        "" => None,
        header => Some(
            http::HeaderName::from_bytes(header.as_bytes())
                .with_context(|| format!("invalid proxy retry budget header: {header}"))?,
        ),
    };
    Ok(ProxyRetryConfig {
        max_connect_retries: args.proxy_max_connect_retries,
        max_request_retries: args.proxy_max_request_retries,
        max_replay_body_bytes: args.proxy_max_replay_body_bytes,
        require_pylon_retry_signal: args.proxy_require_pylon_retry_signal,
        request_retry_budget_ms_header,
        ..ProxyRetryConfig::default()
    })
}

fn make_resolver(ttl: Duration) -> Result<hickory_resolver::TokioAsyncResolver> {
    let (config, mut options) = hickory_resolver::system_conf::read_system_conf()
        .context("failed to read system resolver config")?;
    options.timeout = Duration::from_secs(1);
    options.attempts = 1;
    options.negative_max_ttl = Some(Duration::from_secs(0));
    options.positive_max_ttl = Some(ttl);
    Ok(hickory_resolver::TokioAsyncResolver::tokio(config, options))
}

/// OAuth2 scope the router requests at the worker-auth endpoint. Distinct from
/// the gateway's invocation scope.
const WORKER_AUTH_SCOPE: &str = "llm:check_worker";

/// Builds the worker-auth startup config from CLI args.
///
/// With a secrets path and no OAuth2 host, the static token from the secrets
/// file is used (unchanged). When an OAuth2 host is set, the router mints tokens
/// via the client-credentials grant, reading the id/secret from the secrets
/// file. An OAuth2 host without a secrets path is rejected.
fn worker_auth_startup_from_args(
    endpoint: Option<String>,
    secrets_path: Option<String>,
    secrets_json_path: Option<String>,
    oauth2_provider_host: Option<String>,
) -> Result<Option<WorkerAuthStartup>> {
    let Some(endpoint) = endpoint else {
        return Ok(None);
    };
    let token_provider = match (secrets_path, oauth2_provider_host) {
        (Some(path), None) => {
            let key = secrets_json_path
                .unwrap_or_else(|| "authToken".to_string())
                .split('.')
                .map(String::from)
                .collect();
            Some(stargate_auth::AuthTokenProvider::JsonFile {
                path: PathBuf::from(path),
                key,
            })
        }
        (Some(path), Some(host)) => Some(stargate_auth::AuthTokenProvider::client_credentials(
            &host,
            PathBuf::from(path),
            WORKER_AUTH_SCOPE,
        )),
        (None, Some(_)) => anyhow::bail!(
            "OAUTH2_PROVIDER_HOST is set but SECRETS_PATH is not; client-credentials worker auth needs the secrets file with the id/secret"
        ),
        (None, None) => None,
    };
    Ok(Some(WorkerAuthStartup {
        endpoint,
        token_provider,
    }))
}

fn server_tls_identity_for_reverse_listener(
    reverse_tunnel_enabled: bool,
    cert_pem: Option<Vec<u8>>,
    key_pem: Option<Vec<u8>>,
) -> Result<ServerTlsIdentity, anyhow::Error> {
    if reverse_tunnel_enabled {
        return ServerTlsIdentity::from_optional_pem(cert_pem, key_pem);
    }
    Ok(ServerTlsIdentity::SelfSigned)
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};

    use clap::Parser;

    use super::*;

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
        <Args as Parser>::try_parse_from(args)
    }

    fn parse_args(extra: &[&str]) -> Args {
        try_parse_args(extra).expect("args should parse")
    }

    fn test_resolver(_: Duration) -> Result<hickory_resolver::TokioAsyncResolver> {
        Ok(hickory_resolver::TokioAsyncResolver::tokio(
            hickory_resolver::config::ResolverConfig::default(),
            hickory_resolver::config::ResolverOpts::default(),
        ))
    }

    #[derive(Clone)]
    struct CapturedLog(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedLog {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("test log buffer should not be poisoned")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn proxy_retry_cli_defaults_match_runtime_defaults() {
        let args = parse_args(&[]);
        let retry = proxy_retry_config_from_args(&args).expect("retry config should parse");
        let defaults = ProxyRetryConfig::default();

        assert_eq!(retry.max_connect_retries, defaults.max_connect_retries);
        assert_eq!(retry.max_request_retries, defaults.max_request_retries);
        assert_eq!(retry.max_replay_body_bytes, defaults.max_replay_body_bytes);
        assert_eq!(
            retry.require_pylon_retry_signal,
            defaults.require_pylon_retry_signal
        );
        assert_eq!(
            retry.request_retry_budget_ms_header,
            defaults.request_retry_budget_ms_header
        );
    }

    #[test]
    fn proxy_retry_cli_overrides_are_applied() {
        let args = parse_args(&[
            "--proxy-max-connect-retries",
            "7",
            "--proxy-max-request-retries",
            "9",
            "--proxy-max-replay-body-bytes",
            "12345",
            "--proxy-require-pylon-retry-signal=false",
            "--proxy-retry-budget-header",
            "x-test-budget-ms",
        ]);
        let retry = proxy_retry_config_from_args(&args).expect("retry config should parse");

        assert_eq!(retry.max_connect_retries, 7);
        assert_eq!(retry.max_request_retries, 9);
        assert_eq!(retry.max_replay_body_bytes, 12345);
        assert!(!retry.require_pylon_retry_signal);
        assert_eq!(
            retry.request_retry_budget_ms_header,
            Some(http::HeaderName::from_static("x-test-budget-ms"))
        );
    }

    #[test]
    fn empty_proxy_retry_budget_header_disables_budget_header() {
        let args = parse_args(&["--proxy-retry-budget-header", ""]);
        let retry = proxy_retry_config_from_args(&args).expect("retry config should parse");

        assert_eq!(retry.request_retry_budget_ms_header, None);
    }

    #[test]
    fn direct_quic_tls_trust_cert_does_not_require_server_key() {
        let identity =
            server_tls_identity_for_reverse_listener(false, Some(b"cert".to_vec()), None)
                .expect("cert-only direct trust should not require a server key");

        assert_eq!(identity, ServerTlsIdentity::SelfSigned);
    }

    #[test]
    fn reverse_listener_tls_cert_still_requires_server_key() {
        let err = server_tls_identity_for_reverse_listener(true, Some(b"cert".to_vec()), None)
            .expect_err("reverse listener server TLS still needs a complete PEM pair");

        assert!(
            err.to_string().contains("TLS key PEM is required"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reverse_tunnel_config_owns_rendered_advertised_host_and_runtime_settings() {
        let args = parse_args(&[
            "--reverse-tunnel-listen-addr",
            "127.0.0.1:50072",
            "--advertised-hostname-template",
            "{pod_name}.stargate-headless.{namespace}.svc.cluster.local",
            "--pod-name",
            "stargate-3",
            "--pod-namespace",
            "inference",
            "--reverse-tunnel-pylon-dial-addr",
            "stargate-quic-lb.inference.svc.cluster.local:443",
            "--reverse-tunnel-connect-timeout-ms",
            "4321",
        ]);

        let config = reverse_tunnel_config_from_args(&args)
            .expect("reverse tunnel config should parse")
            .expect("listener should enable reverse tunnel config");

        assert_eq!(config.listen_addr, "127.0.0.1:50072".parse().unwrap());
        assert_eq!(
            config.advertised_host,
            "stargate-3.stargate-headless.inference.svc.cluster.local"
        );
        assert_eq!(
            config.pylon_dial_addr.as_deref(),
            Some("stargate-quic-lb.inference.svc.cluster.local:443")
        );
        assert_eq!(config.connect_timeout, Duration::from_millis(4321));
    }

    #[test]
    fn worker_auth_json_token_provider_uses_configured_key_path() {
        let startup = worker_auth_startup_from_args(
            Some("http://auth.example.test".to_string()),
            Some("/var/run/secrets/auth.json".to_string()),
            Some("nested.authToken".to_string()),
            None,
        )
        .expect("worker auth args should be valid")
        .expect("auth startup should exist");

        assert_eq!(startup.endpoint, "http://auth.example.test");
        match startup
            .token_provider
            .expect("secrets path should create token provider")
        {
            stargate_auth::AuthTokenProvider::JsonFile { path, key } => {
                assert_eq!(path, PathBuf::from("/var/run/secrets/auth.json"));
                assert_eq!(key, vec!["nested".to_string(), "authToken".to_string()]);
            }
            other => panic!("unexpected token provider: {other:?}"),
        }
    }

    #[test]
    fn worker_auth_json_token_provider_uses_default_key_path() {
        let startup = worker_auth_startup_from_args(
            Some("http://auth.example.test".to_string()),
            Some("/var/run/secrets/auth.json".to_string()),
            None,
            None,
        )
        .expect("worker auth args should be valid")
        .expect("auth startup should exist");

        match startup
            .token_provider
            .expect("secrets path should create token provider")
        {
            stargate_auth::AuthTokenProvider::JsonFile { key, .. } => {
                assert_eq!(key, vec!["authToken".to_string()]);
            }
            other => panic!("unexpected token provider: {other:?}"),
        }
    }

    #[test]
    fn worker_auth_uses_client_credentials_when_oauth_host_set() {
        let startup = worker_auth_startup_from_args(
            Some("http://auth.example.test".to_string()),
            Some("/var/run/secrets/auth.json".to_string()),
            None,
            Some("https://oauth.example.test".to_string()),
        )
        .expect("worker auth args should be valid")
        .expect("auth startup should exist");

        assert!(matches!(
            startup.token_provider,
            Some(stargate_auth::AuthTokenProvider::ClientCredentials(_))
        ));
    }

    #[test]
    fn worker_auth_errors_when_oauth_host_set_without_secrets() {
        let result = worker_auth_startup_from_args(
            Some("http://auth.example.test".to_string()),
            None,
            None,
            Some("https://oauth.example.test".to_string()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn worker_auth_absent_without_endpoint() {
        let startup = worker_auth_startup_from_args(None, None, None, None)
            .expect("worker auth args should be valid");
        assert!(startup.is_none());
    }

    #[test]
    fn self_only_discovery_uses_proxy_http_port() {
        let args = parse_args(&[
            "--disable-dns-discovery",
            "--http-listen-addr",
            "127.0.0.1:18000",
        ]);
        let startup = make_discovery(&args).expect("self-only discovery should build without DNS");

        assert!(startup.forwarding.is_none());
        let initial = startup.discovery.initial_stargates();
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].stargate_id, "sg-test");
        assert_eq!(initial[0].advertise_addr, "127.0.0.1:50071");
        assert_eq!(initial[0].http_advertise_addr, "127.0.0.1:18000");
    }

    #[test]
    fn kubernetes_identity_without_dev_peer_forwarding_has_no_forwarding_resolver() {
        let args = parse_args(&["--pod-name", "stargate-0", "--pod-namespace", "inference"]);

        let startup = make_discovery_with_resolver(&args, test_resolver)
            .expect("headless DNS discovery should build");

        assert!(startup.forwarding.is_none());
    }

    #[test]
    fn explicit_dev_peer_forwarding_attaches_a_forwarding_resolver_and_logs_warning() {
        let args = parse_args(&[
            "--pod-name",
            "stargate-0",
            "--pod-namespace",
            "inference",
            "--enable-dev-peer-forwarding",
        ]);
        let logs = Arc::new(Mutex::new(Vec::new()));
        let writer = {
            let logs = logs.clone();
            move || CapturedLog(logs.clone())
        };
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::WARN)
            .with_writer(writer)
            .finish();

        let startup = tracing::subscriber::with_default(subscriber, || {
            make_discovery_with_resolver(&args, test_resolver)
        })
        .expect("development peer forwarding should build");

        assert!(startup.forwarding.is_some());
        let logs = String::from_utf8(
            logs.lock()
                .expect("test log buffer should not be poisoned")
                .clone(),
        )
        .expect("captured logs should be UTF-8");
        assert_eq!(
            logs.matches(
                "development-only peer forwarding is enabled; it must not run in production"
            )
            .count(),
            1,
            "expected exactly one development-only warning: {logs}"
        );
        assert!(
            logs.contains("development_only=true"),
            "warning should include development_only=true: {logs}"
        );
        assert!(
            logs.contains("stargate_id=\"sg-test\""),
            "warning should include the Stargate identity: {logs}"
        );
    }

    #[tokio::test]
    async fn dev_peer_forwarding_without_pod_identity_is_rejected_before_runtime_construction() {
        let result = runtime_from_args(parse_args(&["--enable-dev-peer-forwarding"])).await;
        let error = match result {
            Ok(_) => panic!("development peer forwarding without pod identity must be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("requires both --pod-name and --pod-namespace"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn dev_peer_forwarding_with_disabled_dns_discovery_is_rejected() {
        let result = runtime_from_args(parse_args(&[
            "--enable-dev-peer-forwarding",
            "--pod-name",
            "stargate-0",
            "--pod-namespace",
            "inference",
            "--disable-dns-discovery",
        ]))
        .await;
        let error = match result {
            Ok(_) => panic!("development peer forwarding without DNS discovery must be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("cannot be combined with --disable-dns-discovery"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn runtime_startup_returns_process_owned_shutdown_timeout() {
        let startup = runtime_from_args(parse_args(&[
            "--disable-dns-discovery",
            "--metrics-port",
            "9191",
            "--shutdown-drain-timeout-ms",
            "1234",
        ]))
        .await
        .expect("runtime startup should build without DNS when discovery is disabled");

        assert_eq!(startup.shutdown_drain_timeout, Duration::from_millis(1234));
    }

    #[tokio::test]
    async fn occupied_metrics_port_fails_owned_runtime_startup() {
        let blocker =
            std::net::TcpListener::bind("127.0.0.1:0").expect("metrics blocker should bind");
        let metrics_port = blocker
            .local_addr()
            .expect("metrics blocker should have an address")
            .port();
        let startup = runtime_from_args(parse_args(&[
            "--disable-dns-discovery",
            "--metrics-port",
            &metrics_port.to_string(),
        ]))
        .await
        .expect("runtime startup should build");

        let error = match startup.runtime.start().await {
            Ok(_) => panic!("occupied metrics port should fail runtime startup"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("metrics"));
    }
}
