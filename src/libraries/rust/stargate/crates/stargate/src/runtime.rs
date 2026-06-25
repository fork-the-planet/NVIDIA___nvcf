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

mod server_tasks;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::info;

use stargate_forwarding::ForwardingResolver;
pub use stargate_runtime::CriticalTaskFailure;
use stargate_runtime::CriticalTaskGroup;

use crate::auth::{OpenAuthenticator, WorkerAuthenticator};
use crate::control_plane::{
    RegistrationConnectionConfig, ReverseTunnelRegistrationConfig, StargateService,
    StargateServiceConfig,
};
use crate::discovery::Discovery;
use crate::http_proxy::{ProxyAppState, ProxyTrafficState, ProxyTransportConfig, make_router};
use crate::load_balancer::{LoadBalancerConfig, LoadBalancerRouter};
use crate::metrics::StargateMetrics;
use crate::routing_state::StargateState;
use crate::tunnel::{QuicHttpProxy, QuicTunnelConfig};
use server_tasks::{
    bind_tcp_listener, spawn_control_plane_grpc_server, spawn_http_proxy_server,
    spawn_model_discovery_grpc_server,
};
#[derive(Debug, Clone)]
pub struct StargateRuntimeConfig {
    /// Stable process/pod identity used in logs, metrics, and routing snapshots.
    pub stargate_id: String,

    /// Local TCP socket for the backend-facing gRPC control plane:
    /// `WatchStargates` and `RegisterInferenceServer`.
    pub grpc_listen_addr: SocketAddr,

    /// Local TCP socket for the frontend-facing `ListModels` gRPC service.
    /// This is intentionally separate from the backend control plane.
    pub model_discovery_listen_addr: SocketAddr,

    /// Local HTTP socket for OpenAI-compatible proxy traffic and health probes.
    pub http_listen_addr: SocketAddr,

    /// Optional local TCP socket for Prometheus metrics.
    pub metrics_listen_addr: Option<SocketAddr>,

    /// Self address used by discovery implementations before any hostname
    /// template is applied. Outside Kubernetes this is usually what pylons see
    /// in `WatchStargates.stargates[*].advertise_addr`; in Kubernetes the
    /// advertised hostname template normally replaces the host.
    pub advertise_addr: SocketAddr,

    /// DNS name used for Stargate peer discovery. In Kubernetes this must be
    /// the headless Service name so EndpointSlice readiness controls which pods
    /// are visible to pylons and to the development-only peer relay when it is
    /// explicitly enabled.
    pub stargate_discovery_dns_name: String,

    /// Additional `WatchStargates` endpoints for remote regions. These are
    /// recursive watch seeds only; pylons register to concrete Stargate entries
    /// returned by those streams.
    pub remote_watch_stargate_urls: Vec<String>,

    /// Optional pylon dial address for backend-facing gRPC registration/watch.
    /// When set, `WatchStargates` preserves the per-pod `advertise_addr` as
    /// gRPC authority/SNI identity and sends this as `grpc_pylon_dial_addr` so
    /// pylons connect through a TCP load balancer.
    pub grpc_pylon_dial_addr: Option<String>,

    /// Poll cadence for DNS-based peer discovery.
    pub dns_poll_interval: Duration,

    /// Maximum interval between unchanged `WatchStargates` snapshots.
    pub watch_heartbeat_interval: Duration,

    /// Minimum idle timeout for heartbeat-aware registration streams. A zero
    /// value disables registration idle enforcement.
    pub registration_update_idle_timeout: Duration,

    /// Hard cap for heartbeat-aware registration idle timeout and fallback for
    /// streams that do not advertise heartbeat hints. A zero value disables idle enforcement.
    pub registration_update_max_idle_timeout: Duration,

    /// QUIC/TLS/tunnel-protocol and proxy retry configuration for backend
    /// request forwarding.
    pub proxy_transport: ProxyTransportConfig,

    /// Optional JSON load-balancer config path.
    pub lb_config_path: Option<String>,

    /// Prefix prepended to Prometheus metric names.
    pub metrics_prefix: String,

    /// Reverse listener, advertised identity, external dial address, and
    /// registration wait policy. `None` disables reverse registrations.
    pub reverse_tunnel: Option<ReverseTunnelConfig>,
}

#[derive(Debug, Clone)]
pub struct ReverseTunnelConfig {
    /// Local UDP socket Stargate binds for pylon-initiated reverse QUIC tunnels.
    pub listen_addr: SocketAddr,
    /// Already-rendered hostname used as reverse QUIC SNI and routing identity.
    pub advertised_host: String,
    /// Optional externally reachable UDP load-balancer address for pylons.
    pub pylon_dial_addr: Option<String>,
    /// How long registration waits for a reverse connection after advertising.
    pub connect_timeout: Duration,
}

pub struct StargateRuntime {
    config: StargateRuntimeConfig,
    discovery: Box<dyn Discovery>,
    forwarding: Option<Arc<dyn ForwardingResolver>>,
    authenticator: Arc<dyn WorkerAuthenticator>,
    /// Pre-bound listeners bypass the bind-after-allocation race where
    /// `ephemeral_addr()` releases a port that another process can steal
    /// before `start()` re-binds it. When set, `start()` uses the provided
    /// socket instead of binding to the configured address.
    grpc_listener: Option<std::net::TcpListener>,
    model_discovery_listener: Option<std::net::TcpListener>,
    http_listener: Option<std::net::TcpListener>,
    reverse_tunnel_socket: Option<std::net::UdpSocket>,
}

impl StargateRuntime {
    pub fn new(config: StargateRuntimeConfig, discovery: Box<dyn Discovery>) -> Self {
        Self {
            config,
            discovery,
            forwarding: None,
            authenticator: Arc::new(OpenAuthenticator),
            grpc_listener: None,
            model_discovery_listener: None,
            http_listener: None,
            reverse_tunnel_socket: None,
        }
    }

    pub fn with_forwarding(mut self, forwarding: Arc<dyn ForwardingResolver>) -> Self {
        self.forwarding = Some(forwarding);
        self
    }

    pub fn with_authenticator(mut self, authenticator: Arc<dyn WorkerAuthenticator>) -> Self {
        self.authenticator = authenticator;
        self
    }

    pub fn with_grpc_listener(mut self, listener: std::net::TcpListener) -> Self {
        self.grpc_listener = Some(listener);
        self
    }

    pub fn with_model_discovery_listener(mut self, listener: std::net::TcpListener) -> Self {
        self.model_discovery_listener = Some(listener);
        self
    }

    pub fn with_http_listener(mut self, listener: std::net::TcpListener) -> Self {
        self.http_listener = Some(listener);
        self
    }

    pub fn with_reverse_tunnel_socket(mut self, socket: std::net::UdpSocket) -> Self {
        self.reverse_tunnel_socket = Some(socket);
        self
    }

    pub async fn start(self) -> Result<StargateHandle> {
        let (tasks, critical_failure_rx) = CriticalTaskGroup::new("stargate");
        let startup_shutdown = StartupShutdownGuard::new(&tasks);
        let metrics = StargateMetrics::new_with_prefix(&self.config.metrics_prefix)
            .context("failed to create prometheus metrics registry")?;
        if let Some(metrics_addr) = self.config.metrics_listen_addr {
            let metrics_listener = bind_tcp_listener(None, metrics_addr, "metrics").await?;
            let metrics_registry = metrics.registry();
            tasks.spawn_critical("metrics server", move |stop| {
                crate::metrics::start_metrics_server(metrics_listener, metrics_registry, stop)
            });
        }

        let quic_proxy = QuicHttpProxy::new(
            QuicTunnelConfig {
                connect_timeout: self.config.proxy_transport.quic_connect_timeout,
                request_timeout: self.config.proxy_transport.quic_request_timeout,
                tls_cert_pem: self.config.proxy_transport.tls_cert_pem.clone(),
                server_tls_identity: self.config.proxy_transport.server_tls_identity.clone(),
                quic_insecure: self.config.proxy_transport.quic_insecure,
                tunnel_protocol: self.config.proxy_transport.tunnel_protocol,
                direct_quic_connections: self.config.proxy_transport.direct_quic_connections,
            },
            self.authenticator.clone(),
        )
        .context("failed to initialize quic proxy")?;
        let quic_proxy = Arc::new(quic_proxy);
        let shared_state = Arc::new(StargateState::new_with_metrics(metrics.clone()));

        let reverse_tunnel = if let Some(reverse_tunnel) = &self.config.reverse_tunnel {
            let bound_reverse_addr = quic_proxy
                .start_reverse_listener(
                    reverse_tunnel.listen_addr,
                    shared_state.clone(),
                    tasks.clone(),
                    self.forwarding.clone(),
                    self.reverse_tunnel_socket,
                )
                .await
                .context("failed to start reverse tunnel listener")?;
            Some(reverse_tunnel.registration_config(bound_reverse_addr.port()))
        } else {
            None
        };
        let registration_connection_config = RegistrationConnectionConfig {
            quic_proxy: quic_proxy.clone(),
            reverse_tunnel,
        };

        let lb_config = match &self.config.lb_config_path {
            Some(path) => {
                let bytes = std::fs::read(path)
                    .with_context(|| format!("failed to read lb config file: {path}"))?;
                serde_json::from_slice::<LoadBalancerConfig>(&bytes)
                    .with_context(|| format!("failed to parse lb config file: {path}"))?
            }
            None => LoadBalancerConfig::default(),
        };
        let lb_router = Arc::new(
            LoadBalancerRouter::from_config(&lb_config)
                .context("failed to create load balancer router")?,
        );
        info!(
            default_lb = %lb_config.default,
            model_overrides = lb_config.models.len(),
            "load balancer config loaded"
        );

        let model_discovery_listener = bind_tcp_listener(
            self.model_discovery_listener,
            self.config.model_discovery_listen_addr,
            "model-discovery",
        )
        .await?;
        let service = StargateService::new(StargateServiceConfig {
            stargate_id: self.config.stargate_id.clone(),
            advertise_addr: self.config.advertise_addr,
            discovery_dns_name: self.config.stargate_discovery_dns_name.clone(),
            discovery: self.discovery,
            remote_watch_stargate_urls: self.config.remote_watch_stargate_urls.clone(),
            grpc_pylon_dial_addr: self.config.grpc_pylon_dial_addr.clone(),
            discovery_poll_interval: self.config.dns_poll_interval,
            watch_heartbeat_interval: self.config.watch_heartbeat_interval,
            tasks: tasks.clone(),
            registration_update_idle_timeout: self.config.registration_update_idle_timeout,
            registration_update_max_idle_timeout: self.config.registration_update_max_idle_timeout,
            state: shared_state.clone(),
            registration_connection_config,
            forwarding: self.forwarding,
            authenticator: self.authenticator,
        });

        let proxy_router = make_router(ProxyAppState {
            state: service.state(),
            traffic: ProxyTrafficState {
                shutdown: tasks.shutdown_signal(),
            },
            quic_proxy: quic_proxy.clone(),
            lb_router,
            metrics: metrics.clone(),
            retry: self.config.proxy_transport.retry.clone(),
        });

        let grpc_listener =
            bind_tcp_listener(self.grpc_listener, self.config.grpc_listen_addr, "gRPC").await?;
        spawn_control_plane_grpc_server(&tasks, grpc_listener, service.clone());

        spawn_model_discovery_grpc_server(&tasks, model_discovery_listener, service.clone());

        let http_listener =
            bind_tcp_listener(self.http_listener, self.config.http_listen_addr, "HTTP").await?;
        spawn_http_proxy_server(&tasks, http_listener, proxy_router);

        startup_shutdown.disarm();
        Ok(StargateHandle {
            tasks,
            critical_failure_rx,
            metrics,
            state: service.state(),
        })
    }
}

struct StartupShutdownGuard<'a> {
    tasks: &'a CriticalTaskGroup,
    disarmed: bool,
}

impl<'a> StartupShutdownGuard<'a> {
    fn new(tasks: &'a CriticalTaskGroup) -> Self {
        Self {
            tasks,
            disarmed: false,
        }
    }

    fn disarm(mut self) {
        self.disarmed = true;
    }
}

impl Drop for StartupShutdownGuard<'_> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        self.tasks.begin_shutdown();
    }
}

pub struct StargateHandle {
    tasks: CriticalTaskGroup,
    critical_failure_rx: flume::Receiver<CriticalTaskFailure>,
    metrics: Arc<StargateMetrics>,
    state: Arc<StargateState>,
}

impl StargateHandle {
    pub fn metrics(&self) -> Arc<StargateMetrics> {
        self.metrics.clone()
    }

    pub fn state(&self) -> Arc<StargateState> {
        self.state.clone()
    }

    pub fn begin_shutdown(&self) {
        if !self.tasks.is_stopping() {
            info!("Entering draining mode");
            self.tasks.begin_shutdown();
        }
    }

    pub async fn wait_for_critical_failure(&self) -> CriticalTaskFailure {
        self.critical_failure_rx
            .recv_async()
            .await
            .expect("runtime task group outlives its critical-failure receiver")
    }

    pub async fn wait_for_shutdown(&self, timeout: Duration) -> bool {
        tokio::select! {
            _ = self.tasks.wait() => true,
            _ = tokio::time::sleep(timeout) => false,
        }
    }
}

impl Drop for StargateHandle {
    fn drop(&mut self) {
        self.tasks.begin_shutdown();
    }
}

impl ReverseTunnelConfig {
    fn registration_config(&self, bound_port: u16) -> ReverseTunnelRegistrationConfig {
        let target = format!("{}:{bound_port}", self.advertised_host);
        let pylon_dial_addr = self
            .pylon_dial_addr
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| target.clone());
        ReverseTunnelRegistrationConfig {
            target,
            pylon_dial_addr,
            connect_timeout: self.connect_timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_proxy::ProxyTransportConfig;
    use stargate_proto::pb::StargateInfo;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::oneshot;
    use tokio::time::Instant as TokioInstant;

    #[test]
    fn reverse_tunnel_registration_config_keeps_target_separate_from_pylon_dial_address() {
        let config = ReverseTunnelConfig {
            listen_addr: "0.0.0.0:50072".parse().unwrap(),
            advertised_host: "stargate-0.stargate-headless.stargate.svc.cluster.local".to_string(),
            pylon_dial_addr: Some("stargate-quic-lb.stargate.svc.cluster.local:50072".to_string()),
            connect_timeout: Duration::from_secs(10),
        }
        .registration_config(50072);

        assert_eq!(
            config.target,
            "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
        );
        assert_eq!(
            config.pylon_dial_addr,
            "stargate-quic-lb.stargate.svc.cluster.local:50072"
        );
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
    }

    #[test]
    fn reverse_tunnel_registration_config_uses_target_as_default_pylon_dial_address() {
        let config = ReverseTunnelConfig {
            listen_addr: "0.0.0.0:0".parse().unwrap(),
            advertised_host: "stargate-0.stargate-headless.stargate.svc.cluster.local".to_string(),
            pylon_dial_addr: Some("   ".to_string()),
            connect_timeout: Duration::from_secs(10),
        }
        .registration_config(50072);

        assert_eq!(
            config.target,
            "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
        );
        assert_eq!(config.pylon_dial_addr, config.target);
    }

    struct CountingDiscovery {
        active_count: Arc<AtomicUsize>,
        self_info: StargateInfo,
    }

    impl CountingDiscovery {
        fn new(active_count: Arc<AtomicUsize>, self_info: StargateInfo) -> Self {
            active_count.fetch_add(1, Ordering::SeqCst);
            Self {
                active_count,
                self_info,
            }
        }
    }

    impl Drop for CountingDiscovery {
        fn drop(&mut self) {
            self.active_count.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl Discovery for CountingDiscovery {
        fn initial_stargates(&self) -> Vec<StargateInfo> {
            vec![self.self_info.clone()]
        }

        async fn discover_stargates(&self) -> Vec<StargateInfo> {
            vec![self.self_info.clone()]
        }
    }

    struct BlockingDiscovery {
        active_calls: Arc<AtomicUsize>,
        self_info: StargateInfo,
    }

    struct ActiveDiscoveryCall {
        active_calls: Arc<AtomicUsize>,
    }

    impl Drop for ActiveDiscoveryCall {
        fn drop(&mut self) {
            self.active_calls.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl Discovery for BlockingDiscovery {
        fn initial_stargates(&self) -> Vec<StargateInfo> {
            vec![self.self_info.clone()]
        }

        async fn discover_stargates(&self) -> Vec<StargateInfo> {
            let _active_call = ActiveDiscoveryCall {
                active_calls: self.active_calls.clone(),
            };
            self.active_calls.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn start_failure_after_service_construction_stops_startup_tasks() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        let active_discoveries = Arc::new(AtomicUsize::new(0));
        let grpc_blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let grpc_addr = grpc_blocker.local_addr().unwrap();
        let http_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let discovery = CountingDiscovery::new(
            active_discoveries.clone(),
            StargateInfo {
                stargate_id: "test-startup-cleanup".to_string(),
                advertise_addr: grpc_addr.to_string(),
                http_advertise_addr: http_addr.to_string(),
                grpc_pylon_dial_addr: String::new(),
            },
        );
        let runtime = StargateRuntime::new(
            StargateRuntimeConfig {
                stargate_id: "test-startup-cleanup".to_string(),
                grpc_listen_addr: grpc_addr,
                model_discovery_listen_addr: "127.0.0.1:0".parse().unwrap(),
                http_listen_addr: http_addr,
                metrics_listen_addr: None,
                advertise_addr: grpc_addr,
                stargate_discovery_dns_name: "localhost".to_string(),
                remote_watch_stargate_urls: Vec::new(),
                grpc_pylon_dial_addr: None,
                dns_poll_interval: Duration::from_secs(60),
                watch_heartbeat_interval: Duration::from_secs(60),
                registration_update_idle_timeout:
                    crate::control_plane::DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT,
                registration_update_max_idle_timeout:
                    crate::control_plane::DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT,
                proxy_transport: ProxyTransportConfig {
                    quic_connect_timeout: Duration::from_secs(5),
                    quic_request_timeout: Duration::from_secs(10),
                    tls_cert_pem: None,
                    server_tls_identity: stargate_tls::ServerTlsIdentity::SelfSigned,
                    quic_insecure: true,
                    tunnel_protocol: Default::default(),
                    direct_quic_connections: 1,
                    retry: Default::default(),
                },
                lb_config_path: None,
                metrics_prefix: crate::metrics::DEFAULT_PREFIX.to_string(),
                reverse_tunnel: None,
            },
            Box::new(discovery),
        );

        let result = runtime.start().await;
        assert!(result.is_err(), "occupied gRPC port should fail startup");

        wait_for_count(&active_discoveries, 0, Duration::from_secs(2)).await;
    }

    #[tokio::test]
    async fn shutdown_cancels_in_flight_discovery_poll() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

        let active_calls = Arc::new(AtomicUsize::new(0));
        let grpc_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let http_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let runtime = StargateRuntime::new(
            StargateRuntimeConfig {
                stargate_id: "test-discovery-cancel".to_string(),
                grpc_listen_addr: grpc_addr,
                model_discovery_listen_addr: "127.0.0.1:0".parse().unwrap(),
                http_listen_addr: http_addr,
                metrics_listen_addr: None,
                advertise_addr: grpc_addr,
                stargate_discovery_dns_name: "localhost".to_string(),
                remote_watch_stargate_urls: Vec::new(),
                grpc_pylon_dial_addr: None,
                dns_poll_interval: Duration::from_secs(60),
                watch_heartbeat_interval: Duration::from_secs(60),
                registration_update_idle_timeout:
                    crate::control_plane::DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT,
                registration_update_max_idle_timeout:
                    crate::control_plane::DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT,
                proxy_transport: ProxyTransportConfig {
                    quic_connect_timeout: Duration::from_secs(5),
                    quic_request_timeout: Duration::from_secs(10),
                    tls_cert_pem: None,
                    server_tls_identity: stargate_tls::ServerTlsIdentity::SelfSigned,
                    quic_insecure: true,
                    tunnel_protocol: Default::default(),
                    direct_quic_connections: 1,
                    retry: Default::default(),
                },
                lb_config_path: None,
                metrics_prefix: crate::metrics::DEFAULT_PREFIX.to_string(),
                reverse_tunnel: None,
            },
            Box::new(BlockingDiscovery {
                active_calls: active_calls.clone(),
                self_info: StargateInfo {
                    stargate_id: "test-discovery-cancel".to_string(),
                    advertise_addr: grpc_addr.to_string(),
                    http_advertise_addr: http_addr.to_string(),
                    grpc_pylon_dial_addr: String::new(),
                },
            }),
        );

        let handle = runtime.start().await.expect("stargate should start");
        wait_for_count(&active_calls, 1, Duration::from_secs(2)).await;

        handle.begin_shutdown();
        assert!(
            handle.wait_for_shutdown(Duration::from_secs(2)).await,
            "shutdown should not wait for a blocked discovery call"
        );
        wait_for_count(&active_calls, 0, Duration::from_secs(2)).await;
    }

    #[tokio::test]
    async fn handle_surfaces_critical_root_failure_and_stops_runtime() {
        let (tasks, critical_failure_rx) = CriticalTaskGroup::new("stargate");
        let handle = StargateHandle {
            tasks: tasks.clone(),
            critical_failure_rx,
            metrics: StargateMetrics::new().expect("metrics should initialize"),
            state: Arc::new(StargateState::new()),
        };
        tasks.spawn_critical("test critical root", |_| async { Ok(()) });

        let failure =
            tokio::time::timeout(Duration::from_secs(1), handle.wait_for_critical_failure())
                .await
                .expect("critical root failure should reach the handle");

        assert_eq!(failure.task_name(), "test critical root");
        assert_eq!(failure.detail(), "exited unexpectedly");
        assert!(
            handle.wait_for_shutdown(Duration::from_secs(1)).await,
            "critical root failure should stop the runtime tree"
        );
    }

    #[tokio::test]
    async fn dropping_handle_cancels_owned_runtime_work() {
        let (tasks, critical_failure_rx) = CriticalTaskGroup::new("stargate");
        let stop = tasks.shutdown_signal();
        let (stopped_tx, stopped_rx) = oneshot::channel();
        tasks.task_tracker().spawn(async move {
            stop.cancelled().await;
            let _ = stopped_tx.send(());
        });
        let handle = StargateHandle {
            tasks,
            critical_failure_rx,
            metrics: StargateMetrics::new().expect("metrics should initialize"),
            state: Arc::new(StargateState::new()),
        };

        // Dropping the runtime owner is the behavior under test.
        drop(handle);

        tokio::time::timeout(Duration::from_secs(1), stopped_rx)
            .await
            .expect("dropping the handle should cancel owned runtime work")
            .expect("owned runtime work should publish completion");
    }

    async fn wait_for_count(count: &AtomicUsize, expected: usize, timeout: Duration) {
        let deadline = TokioInstant::now() + timeout;
        let mut interval = tokio::time::interval(Duration::from_millis(10));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let actual = count.load(Ordering::SeqCst);
            if actual == expected {
                return;
            }
            assert!(
                TokioInstant::now() < deadline,
                "count stayed at {actual}, expected {expected}"
            );
            interval.tick().await;
        }
    }
}
