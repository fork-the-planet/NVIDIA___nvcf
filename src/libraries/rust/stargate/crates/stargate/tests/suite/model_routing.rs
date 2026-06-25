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

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::common::{
    LifecycleDummyBackend, assert_model_routing, bind_ephemeral_udp, init_crypto,
    make_stargate_runtime, make_stargate_runtime_with_reverse,
    make_stargate_runtime_with_shared_discovery,
    make_stargate_runtime_with_shared_discovery_and_remote_watch_urls,
    make_stargate_runtime_with_shared_discovery_and_reverse, start_and_register_backend,
    start_and_register_backend_with_bringup, start_lifecycle_dummy_backend, wait_for_routing,
    wait_for_unroutable, wait_until,
};
use pylon_lib::{
    BringupConfig, InferenceServerRegistrationClient, InferenceServerRegistrationConfig,
    OutputTokenParserFactory, PylonMetrics, PylonRuntimeState, QuicHttpTunnelConfig,
    QuicHttpTunnelHandle, start_quic_http_tunnel,
};
use stargate::routing::RoutingTargetKey;
use stargate::runtime::StargateHandle;
use stargate_proto::pb::InferenceServerStatus;

const ROUTES: &[(&str, &str)] = &[
    ("model-alpha", "backend-alpha"),
    ("model-beta", "backend-beta"),
];

const GLOBAL_CALIBRATION_REQUESTS: usize = 3;

struct GlobalWatchNode {
    grpc_addr: std::net::SocketAddr,
    http_addr: std::net::SocketAddr,
    handle: Option<StargateHandle>,
}

struct GlobalWatchTopology {
    region_b: Arc<Mutex<Vec<stargate_proto::pb::StargateInfo>>>,
    nodes: Vec<GlobalWatchNode>,
    seed_addr: std::net::SocketAddr,
    remote_seed_addr: std::net::SocketAddr,
}

impl GlobalWatchTopology {
    async fn start() -> Self {
        let region_a = Arc::new(Mutex::new(Vec::new()));
        let region_b = Arc::new(Mutex::new(Vec::new()));

        let (grpc_b0, http_b0, runtime_b0) =
            make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
                "test-global-fault-b-0",
                region_b.clone(),
                Vec::new(),
            );
        let handle_b0 = runtime_b0
            .start()
            .await
            .expect("region B stargate 0 failed");
        let (grpc_a0, http_a0, runtime_a0) =
            make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
                "test-global-fault-a-0",
                region_a.clone(),
                vec![grpc_b0.to_string()],
            );
        let handle_a0 = runtime_a0.start().await.expect("region A stargate failed");
        let (grpc_b1, http_b1, runtime_b1) =
            make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
                "test-global-fault-b-1",
                region_b.clone(),
                vec![grpc_a0.to_string()],
            );
        let handle_b1 = runtime_b1
            .start()
            .await
            .expect("region B stargate 1 failed");

        Self {
            region_b,
            nodes: vec![
                GlobalWatchNode {
                    grpc_addr: grpc_b0,
                    http_addr: http_b0,
                    handle: Some(handle_b0),
                },
                GlobalWatchNode {
                    grpc_addr: grpc_a0,
                    http_addr: http_a0,
                    handle: Some(handle_a0),
                },
                GlobalWatchNode {
                    grpc_addr: grpc_b1,
                    http_addr: http_b1,
                    handle: Some(handle_b1),
                },
            ],
            seed_addr: grpc_a0,
            remote_seed_addr: grpc_a0,
        }
    }

    fn live_grpc_addrs(&self) -> Vec<std::net::SocketAddr> {
        self.nodes
            .iter()
            .filter(|node| node.handle.is_some())
            .map(|node| node.grpc_addr)
            .collect()
    }

    fn live_http_addrs(&self) -> Vec<std::net::SocketAddr> {
        self.nodes
            .iter()
            .filter(|node| node.handle.is_some())
            .map(|node| node.http_addr)
            .collect()
    }

    async fn stop_node(&mut self, index: usize) {
        let handle = self.nodes[index]
            .handle
            .take()
            .expect("global-watch node already stopped");
        handle.begin_shutdown();
        assert!(handle.wait_for_shutdown(Duration::from_secs(5)).await);
    }

    async fn start_replacement(&mut self) -> usize {
        let index = self.nodes.len();
        let (grpc_addr, http_addr, runtime) =
            make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
                &format!("test-global-fault-replacement-{index}"),
                self.region_b.clone(),
                vec![self.remote_seed_addr.to_string()],
            );
        let handle = runtime.start().await.expect("replacement stargate failed");
        self.nodes.push(GlobalWatchNode {
            grpc_addr,
            http_addr,
            handle: Some(handle),
        });
        index
    }

    async fn shutdown(mut self) {
        for node in &mut self.nodes {
            if let Some(handle) = node.handle.take() {
                handle.begin_shutdown();
                assert!(handle.wait_for_shutdown(Duration::from_secs(5)).await);
            }
        }
    }
}

struct GlobalCalibrationBackend {
    backend: LifecycleDummyBackend,
    tunnel: Option<QuicHttpTunnelHandle>,
    registration: InferenceServerRegistrationClient,
    metrics: Arc<PylonMetrics>,
}

impl GlobalCalibrationBackend {
    async fn start(
        seed_addr: std::net::SocketAddr,
        inference_server_id: &str,
        cluster_id: &str,
        model: &str,
        calibration_delay: Duration,
        gated_requests: Option<usize>,
    ) -> Self {
        let backend = start_lifecycle_dummy_backend(model).await;
        backend.set_calibration_delay(calibration_delay);
        if let Some(gated_requests) = gated_requests {
            backend.gate_next_calibration_requests(gated_requests);
        }
        let mut tunnel_config = QuicHttpTunnelConfig::new(
            "127.0.0.1:0".parse().unwrap(),
            format!("http://{}", backend.addr),
        );
        tunnel_config.quic_insecure = true;
        let tunnel = start_quic_http_tunnel(tunnel_config)
            .await
            .expect("global calibration tunnel failed to start");
        let inference_server_url = format!("quic://{}", tunnel.listen_addr());
        let metrics = PylonMetrics::new().expect("pylon metrics should initialize");
        let runtime_state =
            PylonRuntimeState::new(InferenceServerStatus::Active, &[model.to_string()]);
        let mut registration = InferenceServerRegistrationClient::default();
        registration
            .start(InferenceServerRegistrationConfig {
                seeds: vec![seed_addr.to_string()],
                inference_server_id: inference_server_id.to_string(),
                cluster_id: cluster_id.to_string(),
                inference_server_url,
                upstream_http_base_url: Some(format!("http://{}", backend.addr)),
                min_update_interval: Duration::from_millis(100),
                reverse_tunnel: false,
                tls_cert_pem: None,
                quic_insecure: true,
                tunnel_protocol: Default::default(),
                bringup: global_calibration_bringup(),
                output_token_parser_factory: OutputTokenParserFactory::vllm(),
                request_quality_monitor: pylon_lib::RequestQualityMonitorConfig::default(),
                metrics: Some(metrics.clone()),
                retry: pylon_lib::PylonRetryConfig::default(),
                queue_mismatch_retry: pylon_lib::PylonQueueMismatchRetryConfig::default(),
                runtime_state,
                auth_token_provider: None,
            })
            .expect("global calibration registration failed");
        Self {
            backend,
            tunnel: Some(tunnel),
            registration,
            metrics,
        }
    }

    fn stop(&mut self) {
        self.registration.stop();
    }

    async fn shutdown_tunnel(&mut self) {
        if let Some(tunnel) = self.tunnel.take() {
            tunnel.shutdown().await;
        }
    }
}

fn global_calibration_bringup() -> BringupConfig {
    BringupConfig {
        enabled: true,
        active_canary_interval: Duration::ZERO,
        calibration_requests: GLOBAL_CALIBRATION_REQUESTS,
        calibration_prompt_units: 256,
        calibration_max_concurrency: 1,
        calibration_timeout: Duration::from_secs(5),
        ..BringupConfig::default()
    }
}

async fn wait_for_calibration_requests(
    backend: &LifecycleDummyBackend,
    expected: usize,
    timeout: Duration,
) {
    wait_until(
        &format!("{expected} calibration requests"),
        timeout,
        Duration::from_millis(25),
        || async {
            let actual = backend.calibration_requests();
            if actual == expected {
                Ok(())
            } else {
                Err(actual)
            }
        },
    )
    .await;
}

async fn wait_for_registration_gauges(
    backend: &GlobalCalibrationBackend,
    router_addrs: &[std::net::SocketAddr],
    expected: i64,
    timeout: Duration,
) {
    wait_until(
        &format!("registration gauges equal {expected}"),
        timeout,
        Duration::from_millis(50),
        || async {
            let metrics = backend
                .metrics
                .gather_text()
                .expect("pylon metrics should encode");
            let mismatches = router_addrs
                .iter()
                .filter(|router_addr| {
                    let sample = format!(
                        r#"pylon_registration_stream_connected{{router="{router_addr}"}} {expected}"#
                    );
                    !metrics.lines().any(|line| line == sample)
                })
                .copied()
                .collect::<Vec<_>>();
            if mismatches.is_empty() {
                Ok(())
            } else {
                Err(format!("mismatched routers {mismatches:?}:\n{metrics}"))
            }
        },
    )
    .await;
}

async fn wait_for_empty_candidates(topology: &GlobalWatchTopology, model: &str) {
    let target = RoutingTargetKey {
        routing_key: None,
        model_id: model.to_string(),
    };
    for node in topology.nodes.iter().filter(|node| node.handle.is_some()) {
        let state = node.handle.as_ref().expect("live node").state();
        wait_until(
            &format!("empty candidates on {}", node.grpc_addr),
            Duration::from_secs(10),
            Duration::from_millis(50),
            || {
                let state = state.clone();
                let target = target.clone();
                async move {
                    let candidates = state.cluster_candidates_for_target(&target).await;
                    if candidates.is_empty() {
                        Ok(())
                    } else {
                        Err(format!("candidates={candidates:?}"))
                    }
                }
            },
        )
        .await;
    }
}

async fn calibrated_floors(topology: &GlobalWatchTopology, model: &str) -> Vec<f64> {
    let target = RoutingTargetKey {
        routing_key: None,
        model_id: model.to_string(),
    };
    let mut floors = Vec::new();
    for node in topology.nodes.iter().filter(|node| node.handle.is_some()) {
        let state = node.handle.as_ref().expect("live node").state();
        floors.push(
            wait_until(
                &format!("calibrated floor on {}", node.grpc_addr),
                Duration::from_secs(10),
                Duration::from_millis(50),
                || {
                    let state = state.clone();
                    let target = target.clone();
                    async move {
                        let candidates = state.cluster_candidates_for_target(&target).await;
                        let Some(candidate) = candidates.first() else {
                            return Err("no candidate".to_string());
                        };
                        if candidates.len() == 1 && candidate.stats.last_mean_input_tps > 0.0 {
                            Ok(candidate.stats.last_mean_input_tps)
                        } else {
                            Err(format!("candidates={candidates:?}"))
                        }
                    }
                },
            )
            .await,
        );
    }
    floors
}

#[tokio::test]
async fn two_models_forward_quic() {
    init_crypto();

    let (grpc_addr, http_addr, runtime) = make_stargate_runtime("test-sg-fwd-2m");
    let handle = runtime.start().await.expect("stargate failed to start");

    let seeds = vec![grpc_addr.to_string()];
    let mut alpha = start_and_register_backend(&seeds, "backend-alpha", "model-alpha", false).await;
    let mut beta = start_and_register_backend(&seeds, "backend-beta", "model-beta", false).await;

    wait_for_routing(http_addr, "model-alpha", Duration::from_secs(5)).await;
    wait_for_routing(http_addr, "model-beta", Duration::from_secs(5)).await;

    assert_model_routing(&[http_addr], ROUTES, 3).await;

    alpha.stop();
    beta.stop();
    handle.begin_shutdown();
    handle.wait_for_shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn two_models_reverse_tunnel() {
    init_crypto();

    let (reverse_addr, reverse_socket) = bind_ephemeral_udp();
    let (grpc_addr, http_addr, runtime) =
        make_stargate_runtime_with_reverse("test-sg-rev-2m", reverse_addr, Some(reverse_socket));
    let handle = runtime.start().await.expect("stargate failed to start");

    let seeds = vec![grpc_addr.to_string()];
    let mut alpha = start_and_register_backend(&seeds, "backend-alpha", "model-alpha", true).await;
    let mut beta = start_and_register_backend(&seeds, "backend-beta", "model-beta", true).await;

    wait_for_routing(http_addr, "model-alpha", Duration::from_secs(10)).await;
    wait_for_routing(http_addr, "model-beta", Duration::from_secs(10)).await;

    assert_model_routing(&[http_addr], ROUTES, 3).await;

    alpha.stop();
    beta.stop();
    handle.begin_shutdown();
    handle.wait_for_shutdown(Duration::from_secs(5)).await;
}

/// Two stargates with SharedDiscovery. Each backend seeds only stargate 1;
/// the client discovers stargate 2 via WatchStargates and registers with it.
#[tokio::test]
async fn two_models_multi_stargate_forward_quic() {
    init_crypto();

    let peers = Arc::new(Mutex::new(Vec::new()));
    let (grpc_addr_1, http_addr_1, runtime_1) =
        make_stargate_runtime_with_shared_discovery("test-sg-mfwd-1", peers.clone());
    let (_, http_addr_2, runtime_2) =
        make_stargate_runtime_with_shared_discovery("test-sg-mfwd-2", peers.clone());
    let handle_1 = runtime_1.start().await.expect("stargate 1 failed to start");
    let handle_2 = runtime_2.start().await.expect("stargate 2 failed to start");

    let seeds = vec![grpc_addr_1.to_string()];
    let mut alpha = start_and_register_backend(&seeds, "backend-alpha", "model-alpha", false).await;
    let mut beta = start_and_register_backend(&seeds, "backend-beta", "model-beta", false).await;

    for &http_addr in &[http_addr_1, http_addr_2] {
        wait_for_routing(http_addr, "model-alpha", Duration::from_secs(10)).await;
        wait_for_routing(http_addr, "model-beta", Duration::from_secs(10)).await;
    }

    assert_model_routing(&[http_addr_1, http_addr_2], ROUTES, 3).await;

    alpha.stop();
    beta.stop();
    handle_1.begin_shutdown();
    handle_2.begin_shutdown();
    handle_1.wait_for_shutdown(Duration::from_secs(5)).await;
    handle_2.wait_for_shutdown(Duration::from_secs(5)).await;
}

/// Two stargates with SharedDiscovery and reverse tunnel. Each backend seeds
/// only stargate 1; the client discovers stargate 2 via WatchStargates.
#[tokio::test]
async fn two_models_multi_stargate_reverse_tunnel() {
    init_crypto();

    let peers = Arc::new(Mutex::new(Vec::new()));
    let (reverse_addr_1, reverse_socket_1) = bind_ephemeral_udp();
    let (reverse_addr_2, reverse_socket_2) = bind_ephemeral_udp();
    let (grpc_addr_1, http_addr_1, runtime_1) =
        make_stargate_runtime_with_shared_discovery_and_reverse(
            "test-sg-mrev-1",
            peers.clone(),
            reverse_addr_1,
            Some(reverse_socket_1),
        );
    let (_, http_addr_2, runtime_2) = make_stargate_runtime_with_shared_discovery_and_reverse(
        "test-sg-mrev-2",
        peers.clone(),
        reverse_addr_2,
        Some(reverse_socket_2),
    );
    let handle_1 = runtime_1.start().await.expect("stargate 1 failed to start");
    let handle_2 = runtime_2.start().await.expect("stargate 2 failed to start");

    let seeds = vec![grpc_addr_1.to_string()];
    let mut alpha = start_and_register_backend(&seeds, "backend-alpha", "model-alpha", true).await;
    let mut beta = start_and_register_backend(&seeds, "backend-beta", "model-beta", true).await;

    for &http_addr in &[http_addr_1, http_addr_2] {
        wait_for_routing(http_addr, "model-alpha", Duration::from_secs(15)).await;
        wait_for_routing(http_addr, "model-beta", Duration::from_secs(15)).await;
    }

    assert_model_routing(&[http_addr_1, http_addr_2], ROUTES, 3).await;

    alpha.stop();
    beta.stop();
    handle_1.begin_shutdown();
    handle_2.begin_shutdown();
    handle_1.wait_for_shutdown(Duration::from_secs(5)).await;
    handle_2.wait_for_shutdown(Duration::from_secs(5)).await;
}

/// Region A advertises Region B's WatchStargates endpoint as a remote watch URL.
/// A backend seeded only with Region A must recursively watch Region B and
/// register with every discovered stargate pod in both regions.
#[tokio::test]
async fn backend_discovers_remote_region_watch_url_and_registers_globally() {
    init_crypto();

    let region_a = Arc::new(Mutex::new(Vec::new()));
    let region_b = Arc::new(Mutex::new(Vec::new()));
    let (_grpc_a0, http_a0, runtime_a0) =
        make_stargate_runtime_with_shared_discovery("test-global-a-0", region_a.clone());
    let (_, http_a1, runtime_a1) =
        make_stargate_runtime_with_shared_discovery("test-global-a-1", region_a.clone());
    let (grpc_b0, http_b0, runtime_b0) =
        make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
            "test-global-b-0",
            region_b.clone(),
            Vec::new(),
        );
    let (_, http_b1, runtime_b1) =
        make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
            "test-global-b-1",
            region_b.clone(),
            Vec::new(),
        );

    let handle_a0 = runtime_a0
        .start()
        .await
        .expect("region A stargate 0 failed");
    let handle_a1 = runtime_a1
        .start()
        .await
        .expect("region A stargate 1 failed");
    let handle_b0 = runtime_b0
        .start()
        .await
        .expect("region B stargate 0 failed");
    let handle_b1 = runtime_b1
        .start()
        .await
        .expect("region B stargate 1 failed");

    let (grpc_a_remote, http_a_remote, runtime_a_remote) =
        make_stargate_runtime_with_shared_discovery_and_remote_watch_urls(
            "test-global-a-remote",
            region_a.clone(),
            vec![grpc_b0.to_string()],
        );
    let handle_a_remote = runtime_a_remote
        .start()
        .await
        .expect("region A remote-advertising stargate failed");

    let seeds = vec![grpc_a_remote.to_string()];
    let mut backend = start_and_register_backend_with_bringup(
        &seeds,
        "backend-global",
        "model-global",
        false,
        // Disable calibration so this test isolates recursive WatchStargates
        // discovery and registration fanout.
        BringupConfig {
            enabled: false,
            ..BringupConfig::default()
        },
    )
    .await;

    for http_addr in [http_a0, http_a1, http_a_remote, http_b0, http_b1] {
        wait_for_routing(http_addr, "model-global", Duration::from_secs(15)).await;
    }

    backend.stop();
    handle_a0.begin_shutdown();
    handle_a1.begin_shutdown();
    handle_a_remote.begin_shutdown();
    handle_b0.begin_shutdown();
    handle_b1.begin_shutdown();
    handle_a0.wait_for_shutdown(Duration::from_secs(5)).await;
    handle_a1.wait_for_shutdown(Duration::from_secs(5)).await;
    handle_a_remote
        .wait_for_shutdown(Duration::from_secs(5))
        .await;
    handle_b0.wait_for_shutdown(Duration::from_secs(5)).await;
    handle_b1.wait_for_shutdown(Duration::from_secs(5)).await;
}

#[tokio::test]
async fn global_watch_coordinated_calibration_completes_independently_on_each_stargate() {
    init_crypto();

    let topology = GlobalWatchTopology::start().await;
    let model = "model-global-cal";
    let mut backend = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-cal-backend",
        "global-cal-cluster",
        model,
        Duration::ZERO,
        None,
    )
    .await;
    wait_for_registration_gauges(
        &backend,
        &topology.live_grpc_addrs(),
        1,
        Duration::from_secs(15),
    )
    .await;
    for http_addr in topology.live_http_addrs() {
        wait_for_routing(http_addr, model, Duration::from_secs(20)).await;
    }
    wait_for_calibration_requests(
        &backend.backend,
        GLOBAL_CALIBRATION_REQUESTS * topology.live_http_addrs().len(),
        Duration::from_secs(10),
    )
    .await;

    backend.stop();
    backend.shutdown_tunnel().await;
    topology.shutdown().await;
}

#[tokio::test]
async fn global_watch_calibration_reassigns_after_owner_stops() {
    init_crypto();

    let topology = GlobalWatchTopology::start().await;
    let model = "model-global-owner-failure";
    let live_stargates = topology.live_grpc_addrs().len();
    let mut owner = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-owner-a",
        "global-owner-cluster",
        model,
        Duration::ZERO,
        Some(live_stargates),
    )
    .await;
    wait_for_registration_gauges(
        &owner,
        &topology.live_grpc_addrs(),
        1,
        Duration::from_secs(15),
    )
    .await;
    wait_for_calibration_requests(&owner.backend, live_stargates, Duration::from_secs(10)).await;

    let mut replacement = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-owner-b",
        "global-owner-cluster",
        model,
        Duration::ZERO,
        Some(live_stargates),
    )
    .await;
    wait_for_registration_gauges(
        &replacement,
        &topology.live_grpc_addrs(),
        1,
        Duration::from_secs(15),
    )
    .await;
    assert_eq!(
        replacement.backend.calibration_requests(),
        0,
        "waiting sibling must not calibrate while the original owner is registered"
    );
    for http_addr in topology.live_http_addrs() {
        wait_for_unroutable(http_addr, model, Duration::from_secs(5)).await;
    }

    owner.stop();
    wait_for_registration_gauges(
        &owner,
        &topology.live_grpc_addrs(),
        0,
        Duration::from_secs(10),
    )
    .await;
    wait_for_calibration_requests(
        &replacement.backend,
        live_stargates,
        Duration::from_secs(10),
    )
    .await;
    // The disconnected-stream gauges above are the synchronization point.
    // Releasing the old gate now proves stopped ownership cannot dispatch more.
    owner.backend.release_calibration_gate();
    replacement.backend.release_calibration_gate();
    wait_for_calibration_requests(
        &replacement.backend,
        GLOBAL_CALIBRATION_REQUESTS * live_stargates,
        Duration::from_secs(10),
    )
    .await;
    for http_addr in topology.live_http_addrs() {
        wait_for_routing(http_addr, model, Duration::from_secs(10)).await;
    }
    assert_eq!(
        owner.backend.calibration_requests(),
        live_stargates,
        "stopped owner must not issue remaining calibration requests after its gate releases"
    );

    replacement.stop();
    owner.shutdown_tunnel().await;
    replacement.shutdown_tunnel().await;
    topology.shutdown().await;
}

#[tokio::test]
async fn global_watch_calibration_cancels_stopped_stargate_and_calibrates_replacement() {
    init_crypto();

    let mut topology = GlobalWatchTopology::start().await;
    let model = "model-global-stargate-failure";
    let started_stargates = topology.live_grpc_addrs().len();
    let mut backend = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-stargate-backend",
        "global-stargate-cluster",
        model,
        Duration::ZERO,
        Some(started_stargates),
    )
    .await;
    wait_for_registration_gauges(
        &backend,
        &topology.live_grpc_addrs(),
        1,
        Duration::from_secs(15),
    )
    .await;
    wait_for_calibration_requests(&backend.backend, started_stargates, Duration::from_secs(10))
        .await;

    let stopped_router = topology.nodes[2].grpc_addr;
    topology.stop_node(2).await;
    wait_for_registration_gauges(&backend, &[stopped_router], 0, Duration::from_secs(10)).await;
    backend.backend.release_calibration_gate();
    let surviving_stargates = started_stargates - 1;
    let expected_after_stop =
        started_stargates + surviving_stargates * (GLOBAL_CALIBRATION_REQUESTS - 1);
    wait_for_calibration_requests(
        &backend.backend,
        expected_after_stop,
        Duration::from_secs(10),
    )
    .await;
    for http_addr in topology.live_http_addrs() {
        wait_for_routing(http_addr, model, Duration::from_secs(10)).await;
    }

    let replacement_index = topology.start_replacement().await;
    let replacement_router = topology.nodes[replacement_index].grpc_addr;
    wait_for_registration_gauges(&backend, &[replacement_router], 1, Duration::from_secs(15)).await;
    wait_for_calibration_requests(
        &backend.backend,
        expected_after_stop + GLOBAL_CALIBRATION_REQUESTS,
        Duration::from_secs(10),
    )
    .await;
    wait_for_routing(
        topology.nodes[replacement_index].http_addr,
        model,
        Duration::from_secs(10),
    )
    .await;

    backend.stop();
    backend.shutdown_tunnel().await;
    topology.shutdown().await;
}

#[tokio::test]
async fn global_watch_calibration_floor_does_not_cross_zero_registration_boundary() {
    init_crypto();

    let topology = GlobalWatchTopology::start().await;
    let model = "model-global-floor-reset";
    let cluster = "global-floor-reset-cluster";
    let expected_requests = GLOBAL_CALIBRATION_REQUESTS * topology.live_grpc_addrs().len();
    let mut fast = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-floor-fast",
        cluster,
        model,
        Duration::ZERO,
        None,
    )
    .await;
    wait_for_calibration_requests(&fast.backend, expected_requests, Duration::from_secs(10)).await;
    for http_addr in topology.live_http_addrs() {
        wait_for_routing(http_addr, model, Duration::from_secs(10)).await;
    }
    let fast_floors = calibrated_floors(&topology, model).await;

    fast.stop();
    wait_for_empty_candidates(&topology, model).await;
    fast.shutdown_tunnel().await;

    let mut slow = GlobalCalibrationBackend::start(
        topology.seed_addr,
        "global-floor-slow",
        cluster,
        model,
        Duration::from_millis(30),
        None,
    )
    .await;
    wait_for_calibration_requests(&slow.backend, expected_requests, Duration::from_secs(15)).await;
    for http_addr in topology.live_http_addrs() {
        wait_for_routing(http_addr, model, Duration::from_secs(15)).await;
    }
    let slow_floors = calibrated_floors(&topology, model).await;
    assert_eq!(fast_floors.len(), slow_floors.len());
    for (fast_floor, slow_floor) in fast_floors.into_iter().zip(slow_floors) {
        assert!(
            slow_floor < fast_floor,
            "fresh slower calibration floor {slow_floor} must not be clamped by prior floor {fast_floor}"
        );
    }

    slow.stop();
    slow.shutdown_tunnel().await;
    topology.shutdown().await;
}
