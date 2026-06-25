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

use super::router_stream::{RouterAdvertisedStatusTracker, observe_advertised_statuses};
use super::urls::infer_upstream_http_base_url;
use super::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use crate::bringup::{BringupConfig, ModelBringupState};
use crate::output_token_parser::OutputTokenParserFactory;
use crate::queue_admission::PylonQueueMismatchRetryConfig;
use crate::quic_http_tunnel::{
    PylonRetryConfig, ReverseQuicTunnelHandle, TunnelError, TunnelForwardingConfig,
};
use crate::request_quality_monitor::RequestQualityMonitorConfig;
use crate::runtime_state::PylonRuntimeState;
use crate::stats::PylonMetrics;
use stargate_proto::pb::{
    CalibrationState, InferenceServerAck, InferenceServerModelRegistration,
    InferenceServerRegistration, InferenceServerStatus, ModelCalibrationDirective, ModelStats,
};
use stargate_protocol::TunnelTransportProtocol;
use stargate_runtime::OwnedTask;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use axum::Router;
use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio::sync::{Mutex as TokioMutex, mpsc, oneshot, watch};

struct DropNotifier(Option<tokio::sync::oneshot::Sender<()>>);

impl Drop for DropNotifier {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

fn grpc_endpoint(authority_addr: &str) -> StargateGrpcEndpoint {
    StargateGrpcEndpoint::new(authority_addr.to_string(), authority_addr.to_string())
        .expect("test endpoint authority should be non-empty")
}

fn grpc_endpoint_with_dial(authority_addr: &str, dial_addr: &str) -> StargateGrpcEndpoint {
    StargateGrpcEndpoint::new(authority_addr.to_string(), dial_addr.to_string())
        .expect("test endpoint authority should be non-empty")
}

#[test]
fn reverse_tunnel_connectivity_only_overrides_router_local_advertisement() {
    assert_eq!(
        router_advertised_status(InferenceServerStatus::Active, true, false),
        InferenceServerStatus::Inactive
    );
    assert_eq!(
        router_advertised_status(InferenceServerStatus::Active, true, true),
        InferenceServerStatus::Active
    );
    assert_eq!(
        router_advertised_status(InferenceServerStatus::Inactive, true, false),
        InferenceServerStatus::Inactive
    );
}

#[test]
fn stargate_grpc_debug_target_parses_host_port_and_tls() {
    let target = stargate_grpc_debug_target("https://stargate.example.svc:50051").unwrap();

    assert_eq!(
        target,
        StargateGrpcDebugTarget {
            endpoint: "https://stargate.example.svc:50051".to_string(),
            scheme: "https".to_string(),
            host: "stargate.example.svc".to_string(),
            port: 50051,
        }
    );
}

#[test]
fn stargate_grpc_debug_target_defaults_ports() {
    let http_target = stargate_grpc_debug_target("http://router-a").unwrap();
    let https_target = stargate_grpc_debug_target("https://router-a").unwrap();

    assert_eq!(http_target.port, 80);
    assert_eq!(https_target.port, 443);
}

#[test]
fn stargate_grpc_endpoint_rejects_empty_authority_and_formats_dial_overrides() {
    assert!(StargateGrpcEndpoint::new(" ", "stargate-grpc-lb:443").is_none());

    assert_eq!(
        grpc_endpoint("router-a:50071").to_string(),
        "router-a:50071"
    );
    assert_eq!(
        grpc_endpoint_with_dial("router-a:50071", "stargate-grpc-lb:443").to_string(),
        "router-a:50071 via stargate-grpc-lb:443"
    );
}

#[test]
fn bringup_gates_active_status_until_model_is_advertising() {
    assert_eq!(
        gated_model_status(
            InferenceServerStatus::Active,
            ModelBringupState::ConnectingUnavailable
        ),
        InferenceServerStatus::Inactive
    );
    assert_eq!(
        gated_model_status(InferenceServerStatus::Active, ModelBringupState::Recovering),
        InferenceServerStatus::Inactive
    );
    assert_eq!(
        gated_model_status(
            InferenceServerStatus::Active,
            ModelBringupState::AdvertisingActive
        ),
        InferenceServerStatus::Active
    );
}

#[test]
fn observes_router_advertised_status_from_registration_update() {
    let metrics = PylonMetrics::new().expect("metrics should initialize");
    let mut models = HashMap::new();
    models.insert(
        "model-a".to_string(),
        InferenceServerModelRegistration {
            stats: Some(ModelStats {
                last_mean_input_tps: 30.0,
                ..ModelStats::default()
            }),
            status: InferenceServerStatus::Active.into(),
        },
    );

    let registration_update = build_inference_server_registration(
        "client-a",
        "cluster-a",
        "quic://127.0.0.1:9000",
        &models,
        true,
        false,
        false,
    );
    let advertised = advertised_model_statuses(&registration_update);
    observe_advertised_statuses(Some(metrics.as_ref()), "router-a", &advertised);

    let body = metrics.gather_text().expect("metrics should encode");
    assert!(body.contains(
        r#"pylon_model_advertised_status{model="model-a",router="router-a",status="inactive"} 1"#
    ));
    assert!(body.contains(
        r#"pylon_model_advertised_status{model="model-a",router="router-a",status="active"} 0"#
    ));
}

#[test]
fn clears_router_advertised_status_when_tracker_drops() {
    let metrics = PylonMetrics::new().expect("metrics should initialize");
    let mut models = HashMap::new();
    models.insert(
        "model-a".to_string(),
        InferenceServerModelRegistration {
            stats: None,
            status: InferenceServerStatus::Active.into(),
        },
    );
    let registration_update = build_inference_server_registration(
        "client-a",
        "cluster-a",
        "quic://127.0.0.1:9000",
        &models,
        false,
        false,
        false,
    );

    {
        let mut tracker = RouterAdvertisedStatusTracker::new(Some(metrics.as_ref()), "router-a");
        tracker.record_successful_advertisement(advertised_model_statuses(&registration_update));
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(
            r#"pylon_model_advertised_status{model="model-a",router="router-a",status="active"} 1"#
        ));
        assert!(body.contains(r#"pylon_registration_stream_connected{router="router-a"} 1"#));
        tracker.record_reverse_tunnel_connected(true);
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(r#"pylon_reverse_tunnel_connected{router="router-a"} 1"#));
    }

    let body = metrics.gather_text().expect("metrics should encode");
    assert!(body.contains(
        r#"pylon_model_advertised_status{model="model-a",router="router-a",status="active"} 0"#
    ));
    assert!(body.contains(
        r#"pylon_model_advertised_status{model="model-a",router="router-a",status="inactive"} 1"#
    ));
    assert!(body.contains(r#"pylon_registration_stream_connected{router="router-a"} 0"#));
    assert!(body.contains(r#"pylon_reverse_tunnel_connected{router="router-a"} 0"#));
}

#[test]
fn infers_http_upstream_base_url_from_http_registration_url() {
    assert_eq!(
        infer_upstream_http_base_url("http://127.0.0.1:8000"),
        Some("http://127.0.0.1:8000".to_string())
    );
    assert_eq!(infer_upstream_http_base_url("quic://127.0.0.1:8000"), None);
}

fn test_registration_config() -> InferenceServerRegistrationConfig {
    InferenceServerRegistrationConfig {
        seeds: vec!["router-a".to_string()],
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-a".to_string(),
        inference_server_url: "quic://127.0.0.1:8443".to_string(),
        upstream_http_base_url: Some("http://127.0.0.1:8090".to_string()),
        min_update_interval: Duration::from_secs(2),
        reverse_tunnel: false,
        tls_cert_pem: None,
        quic_insecure: true,
        tunnel_protocol: TunnelTransportProtocol::Custom,
        bringup: BringupConfig::default(),
        output_token_parser_factory: OutputTokenParserFactory,
        runtime_state: PylonRuntimeState::new(
            InferenceServerStatus::Active,
            &["model-a".to_string()],
        ),
        request_quality_monitor: RequestQualityMonitorConfig::default(),
        metrics: None,
        retry: PylonRetryConfig::default(),
        queue_mismatch_retry: PylonQueueMismatchRetryConfig::default(),
        auth_token_provider: None,
    }
}

#[test]
fn registration_start_plan_normalizes_config_before_orchestration() {
    let mut config = test_registration_config();
    config.cluster_id = String::new();
    config.inference_server_url = "http://127.0.0.1:8090".to_string();
    config.upstream_http_base_url = None;
    config.reverse_tunnel = true;

    let plan = RegistrationStartPlan::from_config(&config).expect("plan should build");

    assert_eq!(plan.watch_seeds, vec!["router-a".to_string()]);
    assert_eq!(plan.cluster_id, "inst-a");
    assert_eq!(plan.upstream_http_base_url, "http://127.0.0.1:8090");
}

#[test]
fn registration_start_plan_rejects_invalid_startup_config() {
    let mut empty_seeds = test_registration_config();
    empty_seeds.seeds.clear();
    assert!(matches!(
        RegistrationStartPlan::from_config(&empty_seeds),
        Err(ClientError::Config(message)) if message == "stargate seeds are empty"
    ));

    let mut empty_runtime = test_registration_config();
    empty_runtime.runtime_state = PylonRuntimeState::default();
    assert!(matches!(
        RegistrationStartPlan::from_config(&empty_runtime),
        Err(ClientError::Config(message)) if message == "pylon runtime state has no configured models"
    ));

    let mut missing_upstream = test_registration_config();
    missing_upstream.inference_server_url = "quic://127.0.0.1:8443".to_string();
    missing_upstream.upstream_http_base_url = None;
    assert!(matches!(
        RegistrationStartPlan::from_config(&missing_upstream),
        Err(ClientError::Config(message))
            if message == "upstream_http_base_url is required when inference_server_url is not http(s)"
    ));

    let mut direct_http_url = test_registration_config();
    direct_http_url.inference_server_url = "http://127.0.0.1:8090".to_string();
    direct_http_url.upstream_http_base_url = None;
    direct_http_url.reverse_tunnel = false;
    assert!(matches!(
        RegistrationStartPlan::from_config(&direct_http_url),
        Err(ClientError::Config(message))
            if message == "direct registration inference_server_url must be quic://"
    ));
}

#[tokio::test]
async fn registration_start_preserves_owner_configured_runtime_state() {
    let runtime_state =
        PylonRuntimeState::new(InferenceServerStatus::Active, &["model-a".to_string()]);
    runtime_state.set_model_stats(
        "model-a",
        CurrentModelStats {
            output_tps: 12.5,
            ..CurrentModelStats::default()
        },
    );
    let mut config = test_registration_config();
    config.bringup.enabled = false;
    config.runtime_state = runtime_state.clone();

    let mut client = InferenceServerRegistrationClient::default();
    client.start(config).expect("registration should start");

    assert_eq!(
        runtime_state
            .model_stats("model-a")
            .expect("configured model should remain present")
            .output_tps,
        12.5
    );
    client.stop();
}

#[test]
fn reverse_tunnel_registration_advertises_upstream_http_url() {
    let register_config = InferenceServerRegistrationConfig {
        seeds: vec!["router-a".to_string()],
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-a".to_string(),
        inference_server_url: "quic://127.0.0.1:8443".to_string(),
        upstream_http_base_url: Some("http://127.0.0.1:8090".to_string()),
        min_update_interval: Duration::from_secs(2),
        reverse_tunnel: true,
        tls_cert_pem: None,
        quic_insecure: true,
        tunnel_protocol: TunnelTransportProtocol::Custom,
        bringup: BringupConfig::default(),
        output_token_parser_factory: OutputTokenParserFactory,
        runtime_state: PylonRuntimeState::default(),
        request_quality_monitor: RequestQualityMonitorConfig::default(),
        metrics: None,
        retry: PylonRetryConfig::default(),
        queue_mismatch_retry: PylonQueueMismatchRetryConfig::default(),
        auth_token_provider: None,
    };
    let (cluster_calibration_directive_tx, _cluster_calibration_directive_rx) = flume::bounded(1);

    let task_template = RouterRegistrationTaskTemplate::from_registration_config(
        &register_config,
        &register_config.cluster_id,
        register_config
            .upstream_http_base_url
            .as_deref()
            .expect("test config includes upstream HTTP base URL"),
        cluster_calibration_directive_tx,
    );

    let task_config = task_template.build_for_router(grpc_endpoint("router-a"));
    assert_eq!(task_config.inference_server_url, "http://127.0.0.1:8090");
    assert_eq!(
        task_config.forwarding.upstream_http_base_url,
        "http://127.0.0.1:8090"
    );
}

#[test]
fn build_inference_server_registration_includes_cluster_id() {
    let models = HashMap::new();

    let registration = build_inference_server_registration(
        "client-a",
        "cluster-shared",
        "quic://127.0.0.1:9000",
        &models,
        false,
        true,
        false,
    );

    assert_eq!(registration.inference_server_id, "client-a");
    assert_eq!(registration.cluster_id, "cluster-shared");
    assert!(registration.coordinated_calibration);
}

#[test]
fn watch_stargate_urls_are_discovery_seeds_not_registration_routers() {
    let snapshot = watch_endpoint_snapshot_from_response(
        "seed-a",
        stargate_proto::pb::WatchStargatesResponse {
            stargates: vec![stargate_proto::pb::StargateInfo {
                stargate_id: "stargate-0".to_string(),
                advertise_addr: "stargate-0.region-a:50071".to_string(),
                http_advertise_addr: "stargate-0.region-a:8000".to_string(),
                grpc_pylon_dial_addr: "lb.region-a:443".to_string(),
            }],
            watch_stargate_urls: vec!["stargate.region-b:50071".to_string()],
        },
    );

    assert_eq!(
        snapshot.registration_routers,
        BTreeSet::from([grpc_endpoint_with_dial(
            "stargate-0.region-a:50071",
            "lb.region-a:443"
        )])
    );
    assert_eq!(
        snapshot.watch_urls,
        BTreeSet::from(["stargate.region-b:50071".to_string()])
    );
}

#[test]
fn watch_stargate_snapshot_uses_advertise_addr_as_direct_dial_fallback() {
    let snapshot = watch_endpoint_snapshot_from_response(
        "seed-a",
        stargate_proto::pb::WatchStargatesResponse {
            stargates: vec![stargate_proto::pb::StargateInfo {
                stargate_id: "stargate-0".to_string(),
                advertise_addr: "stargate-0.region-a:50071".to_string(),
                http_advertise_addr: String::new(),
                grpc_pylon_dial_addr: String::new(),
            }],
            watch_stargate_urls: Vec::new(),
        },
    );

    assert_eq!(
        snapshot.registration_routers,
        BTreeSet::from([grpc_endpoint("stargate-0.region-a:50071")])
    );
}

#[test]
fn watch_stargate_snapshot_uses_stargate_id_when_advertise_addr_is_empty() {
    let snapshot = watch_endpoint_snapshot_from_response(
        "seed-a",
        stargate_proto::pb::WatchStargatesResponse {
            stargates: vec![stargate_proto::pb::StargateInfo {
                stargate_id: "stargate-0.region-a:50071".to_string(),
                advertise_addr: " ".to_string(),
                http_advertise_addr: String::new(),
                grpc_pylon_dial_addr: String::new(),
            }],
            watch_stargate_urls: Vec::new(),
        },
    );

    assert_eq!(
        snapshot.registration_routers,
        BTreeSet::from([grpc_endpoint("stargate-0.region-a:50071")])
    );
}

#[test]
fn grpc_connect_target_dials_lb_and_overrides_authority() {
    let endpoint = grpc_endpoint_with_dial(
        "stargate-0.region-a:50071",
        "https://stargate-grpc-lb.region-a:443",
    );

    let target = stargate_grpc_connect_target(&endpoint);

    assert_eq!(
        target.dial_endpoint,
        "https://stargate-grpc-lb.region-a:443"
    );
    assert_eq!(
        target.authority_endpoint,
        "https://stargate-0.region-a:50071"
    );
    assert!(target.override_authority);
    stargate_grpc_channel_endpoint(&target).expect("endpoint should be valid");
}

#[test]
fn recursive_watch_discovery_waits_for_remote_snapshots_before_registration_publish() {
    let seeds = BTreeSet::from(["stargate.region-a:50071".to_string()]);
    let mut snapshots = HashMap::from([(
        "stargate.region-a:50071".to_string(),
        WatchEndpointSnapshot {
            registration_routers: BTreeSet::from([
                grpc_endpoint("stargate-0.region-a:50071"),
                grpc_endpoint("stargate-1.region-a:50071"),
            ]),
            watch_urls: BTreeSet::from(["stargate.region-b:50071".to_string()]),
        },
    )]);

    let desired_urls = desired_watch_urls_from_snapshots(&seeds, &snapshots);
    assert_eq!(
        desired_urls,
        BTreeSet::from([
            "stargate.region-a:50071".to_string(),
            "stargate.region-b:50071".to_string(),
        ])
    );
    assert!(!all_desired_watch_urls_have_snapshots(
        &desired_urls,
        |watch_url| snapshots.contains_key(watch_url)
    ));

    snapshots.insert(
        "stargate.region-b:50071".to_string(),
        WatchEndpointSnapshot {
            registration_routers: BTreeSet::from([
                grpc_endpoint("stargate-0.region-b:50071"),
                grpc_endpoint("stargate-1.region-b:50071"),
            ]),
            watch_urls: BTreeSet::from(["stargate.region-a:50071".to_string()]),
        },
    );
    let desired_urls = desired_watch_urls_from_snapshots(&seeds, &snapshots);
    assert!(all_desired_watch_urls_have_snapshots(
        &desired_urls,
        |watch_url| snapshots.contains_key(watch_url)
    ));
    assert_eq!(
        active_registration_routers(snapshots.values()),
        BTreeSet::from([
            grpc_endpoint("stargate-0.region-a:50071"),
            grpc_endpoint("stargate-0.region-b:50071"),
            grpc_endpoint("stargate-1.region-a:50071"),
            grpc_endpoint("stargate-1.region-b:50071"),
        ])
    );
}

#[test]
fn recursive_watch_discovery_ignores_disconnected_snapshot_cycles() {
    let seeds = BTreeSet::from(["stargate.region-a:50071".to_string()]);
    let snapshots = HashMap::from([
        (
            "stargate.region-a:50071".to_string(),
            WatchEndpointSnapshot {
                registration_routers: BTreeSet::from([grpc_endpoint("stargate-0.region-a:50071")]),
                watch_urls: BTreeSet::new(),
            },
        ),
        (
            "stargate.region-b:50071".to_string(),
            WatchEndpointSnapshot {
                registration_routers: BTreeSet::from([grpc_endpoint("stargate-0.region-b:50071")]),
                watch_urls: BTreeSet::from(["stargate.region-c:50071".to_string()]),
            },
        ),
        (
            "stargate.region-c:50071".to_string(),
            WatchEndpointSnapshot {
                registration_routers: BTreeSet::from([grpc_endpoint("stargate-0.region-c:50071")]),
                watch_urls: BTreeSet::from(["stargate.region-b:50071".to_string()]),
            },
        ),
    ]);

    assert_eq!(
        desired_watch_urls_from_snapshots(&seeds, &snapshots),
        BTreeSet::from(["stargate.region-a:50071".to_string()])
    );
}

#[tokio::test]
async fn stop_watched_endpoint_signals_and_awaits_task() {
    let (exited_tx, exited_rx) = tokio::sync::oneshot::channel();
    let task = OwnedTask::spawn("watch stargate endpoint", move |stop| async move {
        stop.cancelled().await;
        let _ = exited_tx.send(());
    });
    let endpoint = WatchedEndpoint {
        generation: 0,
        task,
        state: WatchEndpointState::Connecting,
    };

    tokio::time::timeout(Duration::from_secs(1), stop_watched_endpoint(endpoint))
        .await
        .expect("watched endpoint should stop cooperatively");
    exited_rx
        .await
        .expect("watched endpoint task should publish exit");
}

#[tokio::test]
async fn watch_endpoint_update_send_wakes_on_cancellation_when_channel_is_full() {
    let update = |generation| WatchEndpointUpdate {
        watch_url: "stargate.region-b:50071".to_string(),
        generation,
        event: WatchEndpointEvent::Disconnected,
    };
    let (updates_tx, mut updates_rx) = mpsc::channel(1);
    updates_tx
        .send(update(1))
        .await
        .expect("seed update should fill channel");
    let stop = CancellationToken::new();
    let task_stop = stop.clone();

    let task = tokio::spawn(async move {
        send_watch_endpoint_update(&updates_tx, update(2), &task_stop).await
    });
    tokio::task::yield_now().await;
    stop.cancel();

    let sent = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("send wait should wake on local stop")
        .expect("send task should not panic");
    assert!(!sent, "stopped endpoint should not enqueue an update");
    assert_eq!(
        updates_rx
            .recv()
            .await
            .expect("first update should still be queued")
            .generation,
        1
    );
    assert!(updates_rx.try_recv().is_err());
}

#[tokio::test]
async fn registration_update_send_wakes_on_cancellation_when_channel_is_full() {
    let (updates_tx, _updates_rx) = tokio::sync::mpsc::channel(1);
    updates_tx
        .send(InferenceServerRegistration::default())
        .await
        .expect("seed update should fill channel");
    let stop = CancellationToken::new();
    let task_stop = stop.clone();

    let task = tokio::spawn(async move {
        send_registration_update(
            &updates_tx,
            InferenceServerRegistration::default(),
            &task_stop,
        )
        .await
    });
    tokio::task::yield_now().await;
    stop.cancel();

    let sent = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("registration update send should wake on cancellation")
        .expect("registration update send should not panic");
    assert!(!sent);
}

#[tokio::test]
async fn cluster_calibration_directive_publication_observes_stop_before_empty_batch() {
    let (directives_tx, _directives_rx) = flume::bounded(1);
    let stop = CancellationToken::new();
    let router_endpoint = grpc_endpoint("stargate.region-a:50051");
    stop.cancel();

    assert!(
        !publish_cluster_calibration_directives(
            &directives_tx,
            &router_endpoint,
            Vec::new(),
            &stop,
        )
        .await,
        "stopped registration stream must not treat an empty directive batch as completed work"
    );
}

#[tokio::test]
async fn cluster_calibration_directive_publication_translates_supported_states_in_order() {
    let model_directive = |model_id: &str, state: i32| ModelCalibrationDirective {
        model_id: model_id.to_string(),
        state,
        assignment_token: format!("{model_id}-assignment"),
    };
    let (directives_tx, directives_rx) = flume::unbounded();
    let router_endpoint = grpc_endpoint("stargate.region-a:50051");

    assert!(
        publish_cluster_calibration_directives(
            &directives_tx,
            &router_endpoint,
            vec![
                model_directive("unknown", i32::MAX),
                model_directive("waiting", CalibrationState::Waiting as i32),
                model_directive("run", CalibrationState::Run as i32),
                model_directive("complete", CalibrationState::Complete as i32),
            ],
            &CancellationToken::new(),
        )
        .await
    );

    let received = directives_rx
        .try_iter()
        .map(|directive| {
            let ClusterCalibrationDirective::Model {
                router_endpoint,
                model_id,
                state,
                assignment_token,
            } = directive
            else {
                panic!("model directive publication emitted a disconnect");
            };
            (router_endpoint, model_id, state, assignment_token)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        received,
        vec![
            (
                router_endpoint.clone(),
                "waiting".to_string(),
                ClusterCalibrationDirectiveState::Waiting,
                "waiting-assignment".to_string(),
            ),
            (
                router_endpoint.clone(),
                "run".to_string(),
                ClusterCalibrationDirectiveState::Run,
                "run-assignment".to_string(),
            ),
            (
                router_endpoint,
                "complete".to_string(),
                ClusterCalibrationDirectiveState::Complete,
                "complete-assignment".to_string(),
            ),
        ]
    );
}

#[tokio::test]
async fn cluster_calibration_directive_publication_wakes_on_cancellation_when_channel_is_full() {
    let (directives_tx, directives_rx) = flume::bounded(1);
    directives_tx
        .send(ClusterCalibrationDirective::Model {
            router_endpoint: grpc_endpoint("seed-router"),
            model_id: "seed-model".to_string(),
            state: ClusterCalibrationDirectiveState::Waiting,
            assignment_token: "seed-assignment".to_string(),
        })
        .expect("seed directive should fill the channel");
    let stop = CancellationToken::new();
    let task_stop = stop.clone();
    let router_endpoint = grpc_endpoint("stargate.region-a:50051");

    let task = tokio::spawn(async move {
        publish_cluster_calibration_directives(
            &directives_tx,
            &router_endpoint,
            vec![ModelCalibrationDirective {
                model_id: "blocked-model".to_string(),
                state: CalibrationState::Run as i32,
                assignment_token: "blocked-assignment".to_string(),
            }],
            &task_stop,
        )
        .await
    });
    tokio::task::yield_now().await;
    stop.cancel();

    let completed = tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("directive publication should wake on local stop")
        .expect("directive publication task should not panic");
    assert!(!completed);
    assert_eq!(directives_rx.len(), 1);
}

#[tokio::test]
async fn watch_endpoint_updates_ignore_removed_or_replaced_generations() {
    let snapshot = |router: &str| WatchEndpointSnapshot {
        registration_routers: BTreeSet::from([grpc_endpoint(router)]),
        watch_urls: BTreeSet::new(),
    };
    let watch_url = "stargate.region-b:50071".to_string();
    let mut watched = HashMap::<String, WatchedEndpoint>::new();

    assert!(!apply_watch_endpoint_update(
        &mut watched,
        WatchEndpointUpdate {
            watch_url: watch_url.clone(),
            generation: 0,
            event: WatchEndpointEvent::Snapshot(snapshot("stale-router")),
        }
    ));
    assert!(active_registration_routers(watched_endpoint_snapshots(&watched)).is_empty());

    let task = OwnedTask::spawn("watch stargate endpoint", |_| async {
        std::future::pending::<()>().await;
    });
    watched.insert(
        watch_url.clone(),
        WatchedEndpoint {
            generation: 1,
            task,
            state: WatchEndpointState::Connecting,
        },
    );

    assert!(!apply_watch_endpoint_update(
        &mut watched,
        WatchEndpointUpdate {
            watch_url: watch_url.clone(),
            generation: 0,
            event: WatchEndpointEvent::Snapshot(snapshot("stale-router")),
        }
    ));
    assert!(!all_desired_watch_urls_have_snapshots(
        &BTreeSet::from([watch_url.clone()]),
        |watch_url| watched
            .get(watch_url)
            .is_some_and(|endpoint| endpoint.state.has_snapshot())
    ));
    assert!(active_registration_routers(watched_endpoint_snapshots(&watched)).is_empty());

    assert!(apply_watch_endpoint_update(
        &mut watched,
        WatchEndpointUpdate {
            watch_url: watch_url.clone(),
            generation: 1,
            event: WatchEndpointEvent::Snapshot(snapshot("current-router")),
        }
    ));
    assert_eq!(
        active_registration_routers(watched_endpoint_snapshots(&watched)),
        BTreeSet::from([grpc_endpoint("current-router")])
    );

    assert!(apply_watch_endpoint_update(
        &mut watched,
        WatchEndpointUpdate {
            watch_url: watch_url.clone(),
            generation: 1,
            event: WatchEndpointEvent::Disconnected,
        }
    ));
    assert!(matches!(
        watched.get(&watch_url).map(|endpoint| &endpoint.state),
        Some(WatchEndpointState::Disconnected)
    ));
    assert!(active_registration_routers(watched_endpoint_snapshots(&watched)).is_empty());

    for endpoint in watched.into_values() {
        endpoint.task.abort();
    }
}

#[tokio::test]
async fn registration_router_topology_generation_is_shared_latest_value() {
    let empty = BTreeSet::new();
    let seed_router = BTreeSet::from([grpc_endpoint("stargate-0.region-a:50071")]);
    let global_routers = BTreeSet::from([
        grpc_endpoint("stargate-0.region-a:50071"),
        grpc_endpoint("stargate-0.region-b:50071"),
    ]);
    let (topology_tx, mut registration_rx) = watch::channel(RegistrationRouterTopology::default());
    let mut calibration_rx = registration_rx.clone();

    assert_eq!(registration_rx.borrow().published_routers(), None);
    assert!(!publish_registration_router_topology(
        &topology_tx,
        &seed_router,
        false
    ));
    assert!(!publish_registration_router_topology(
        &topology_tx,
        &empty,
        true
    ));
    assert!(!registration_rx.has_changed().unwrap());
    assert!(!calibration_rx.has_changed().unwrap());

    assert!(publish_registration_router_topology(
        &topology_tx,
        &global_routers,
        true
    ));
    registration_rx.changed().await.unwrap();
    calibration_rx.changed().await.unwrap();
    assert_eq!(
        registration_rx.borrow().published_routers(),
        Some(&global_routers)
    );
    assert_eq!(
        calibration_rx.borrow().published_routers(),
        Some(&global_routers)
    );

    assert!(!publish_registration_router_topology(
        &topology_tx,
        &global_routers,
        false
    ));
    assert!(!registration_rx.has_changed().unwrap());
    assert!(!calibration_rx.has_changed().unwrap());

    assert!(publish_registration_router_topology(
        &topology_tx,
        &seed_router,
        false
    ));
    registration_rx.changed().await.unwrap();
    calibration_rx.changed().await.unwrap();
    assert_eq!(
        registration_rx.borrow().published_routers(),
        Some(&seed_router)
    );
    assert_eq!(
        calibration_rx.borrow().published_routers(),
        Some(&seed_router)
    );

    assert!(publish_registration_router_topology(
        &topology_tx,
        &empty,
        false
    ));
    registration_rx.changed().await.unwrap();
    calibration_rx.changed().await.unwrap();
    assert_eq!(registration_rx.borrow().published_routers(), Some(&empty));
    assert_eq!(calibration_rx.borrow().published_routers(), Some(&empty));
}

#[tokio::test]
async fn watch_discovery_stops_when_cancelled() {
    let (topology_tx, _topology_rx) = watch::channel(RegistrationRouterTopology::default());
    let stop = CancellationToken::new();
    let task = tokio::spawn(run_watch_stargate_discovery(
        Vec::new(),
        topology_tx,
        stop.clone(),
    ));

    stop.cancel();

    tokio::time::timeout(Duration::from_secs(1), task)
        .await
        .expect("watch discovery should stop when cancelled")
        .expect("watch discovery should not panic");
}

#[tokio::test]
async fn coordinated_submission_attempt_times_out_without_waiting_forever() {
    let error = await_cluster_calibration_submission(
        Duration::from_millis(1),
        std::future::pending::<anyhow::Result<()>>(),
    )
    .await
    .expect_err("stalled calibration submission should time out");

    assert!(
        error
            .to_string()
            .contains("cluster calibration submission timed out")
    );
}

#[test]
fn successful_calibration_submission_keeps_non_retryable_assignment_token() {
    let key = (grpc_endpoint("http://router-a"), "model-a".to_string());
    let mut work = HashMap::from([(
        key.clone(),
        RouterCalibrationWork::PendingSubmission {
            result: PendingRouterCalibration {
                assignment_token: "token-a".to_string(),
                measured_last_mean_input_tps: 123.0,
            },
            task: None,
        },
    )]);

    finish_pending_cluster_calibration_submission(
        &mut work,
        CompletedCalibrationSubmission {
            key: key.clone(),
            assignment_token: "token-a".to_string(),
            result: Ok(()),
        },
    );

    assert!(
        matches!(
            work.get(&key),
            Some(RouterCalibrationWork::Submitted { assignment_token })
                if assignment_token == "token-a"
        ),
        "accepted calibration result must keep only assignment provenance"
    );
    let retry_keys = work
        .iter()
        .filter_map(|(key, state)| match state {
            RouterCalibrationWork::PendingSubmission { task: None, .. } => Some(key.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        retry_keys.is_empty(),
        "accepted calibration result must not remain retryable"
    );
    assert!(
        work.get(&key)
            .is_some_and(|state| state.assignment_token() == "token-a"),
        "stale RUN for an accepted assignment must not start a second sweep"
    );
}

#[tokio::test]
async fn failed_calibration_submission_remains_retryable() {
    let key = (grpc_endpoint("http://router-a"), "model-a".to_string());
    let mut submission_tasks = JoinSet::<CompletedCalibrationSubmission>::new();
    let abort_handle = submission_tasks
        .spawn(async { std::future::pending::<CompletedCalibrationSubmission>().await });
    let mut work = HashMap::from([(
        key.clone(),
        RouterCalibrationWork::PendingSubmission {
            result: PendingRouterCalibration {
                assignment_token: "token-a".to_string(),
                measured_last_mean_input_tps: 123.0,
            },
            task: Some(OwnedCalibrationTask::new(abort_handle)),
        },
    )]);

    finish_pending_cluster_calibration_submission(
        &mut work,
        CompletedCalibrationSubmission {
            key: key.clone(),
            assignment_token: "token-a".to_string(),
            result: Err(anyhow::anyhow!("temporary submission failure")),
        },
    );

    assert!(
        matches!(
            work.get(&key),
            Some(RouterCalibrationWork::PendingSubmission { task: None, .. })
        ),
        "failed calibration result should remain pending for retry"
    );
    submission_tasks.abort_all();
}

#[tokio::test]
async fn panicked_calibration_submission_remains_retryable() {
    let key = (grpc_endpoint("http://router-a"), "model-a".to_string());
    let mut submission_tasks = JoinSet::<CompletedCalibrationSubmission>::new();
    let abort_handle = submission_tasks
        .spawn(async { std::future::pending::<CompletedCalibrationSubmission>().await });
    let task_id = abort_handle.id();
    let mut work = HashMap::from([(
        key.clone(),
        RouterCalibrationWork::PendingSubmission {
            result: PendingRouterCalibration {
                assignment_token: "token-a".to_string(),
                measured_last_mean_input_tps: 123.0,
            },
            task: Some(OwnedCalibrationTask::new(abort_handle)),
        },
    )]);

    handle_panicked_calibration_task(&mut work, task_id);

    assert!(
        matches!(
            work.get(&key),
            Some(RouterCalibrationWork::PendingSubmission {
                result,
                task: None,
            }) if result.assignment_token == "token-a"
                && result.measured_last_mean_input_tps == 123.0
        ),
        "panicked submission task should keep its measured result retryable"
    );
    submission_tasks.abort_all();
}

#[tokio::test]
async fn panicked_calibration_sweep_removes_matching_work_only() {
    let sweep_key = (grpc_endpoint("http://router-a"), "model-a".to_string());
    let submitted_key = (grpc_endpoint("http://router-b"), "model-b".to_string());
    let mut sweep_tasks = JoinSet::<CompletedCalibrationSweep>::new();
    let abort_handle =
        sweep_tasks.spawn(async { std::future::pending::<CompletedCalibrationSweep>().await });
    let task_id = abort_handle.id();
    let mut work = HashMap::from([
        (
            sweep_key.clone(),
            RouterCalibrationWork::Sweeping {
                assignment_token: "token-a".to_string(),
                task: OwnedCalibrationTask::new(abort_handle),
            },
        ),
        (
            submitted_key.clone(),
            RouterCalibrationWork::Submitted {
                assignment_token: "token-b".to_string(),
            },
        ),
    ]);

    handle_panicked_calibration_task(&mut work, task_id);

    assert!(!work.contains_key(&sweep_key));
    assert!(
        matches!(
            work.get(&submitted_key),
            Some(RouterCalibrationWork::Submitted { assignment_token })
                if assignment_token == "token-b"
        ),
        "unmatched submitted work should remain untouched"
    );
    sweep_tasks.abort_all();
}

#[tokio::test]
async fn panicked_calibration_cleanup_ignores_unknown_task_id() {
    let key = (grpc_endpoint("http://router-a"), "model-a".to_string());
    let mut submission_tasks = JoinSet::<CompletedCalibrationSubmission>::new();
    let unknown_task_id = submission_tasks
        .spawn(async { std::future::pending::<CompletedCalibrationSubmission>().await })
        .id();
    let mut work = HashMap::from([(
        key.clone(),
        RouterCalibrationWork::PendingSubmission {
            result: PendingRouterCalibration {
                assignment_token: "token-a".to_string(),
                measured_last_mean_input_tps: 123.0,
            },
            task: None,
        },
    )]);
    work.insert(
        (grpc_endpoint("http://router-b"), "model-b".to_string()),
        RouterCalibrationWork::Submitted {
            assignment_token: "token-b".to_string(),
        },
    );

    handle_panicked_calibration_task(&mut work, unknown_task_id);

    assert!(
        matches!(
            work.get(&key),
            Some(RouterCalibrationWork::PendingSubmission {
                result,
                task: None,
            }) if result.assignment_token == "token-a"
                && result.measured_last_mean_input_tps == 123.0
        ),
        "unknown task ids should not alter calibration work"
    );
    submission_tasks.abort_all();
}

#[derive(Clone)]
struct BlockedCalibrationUpstream {
    entered_tx: Arc<TokioMutex<Option<oneshot::Sender<()>>>>,
    dropped_tx: Arc<TokioMutex<Option<oneshot::Sender<()>>>>,
}

async fn spawn_blocked_calibration_upstream()
-> (String, oneshot::Receiver<()>, oneshot::Receiver<()>) {
    let (entered_tx, entered_rx) = oneshot::channel();
    let (dropped_tx, dropped_rx) = oneshot::channel();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test upstream should bind");
    let addr = listener.local_addr().expect("bound listener has address");
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/chat/completions", post(block_calibration_completion))
        .with_state(BlockedCalibrationUpstream {
            entered_tx: Arc::new(TokioMutex::new(Some(entered_tx))),
            dropped_tx: Arc::new(TokioMutex::new(Some(dropped_tx))),
        });
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("test upstream should serve");
    });
    (format!("http://{addr}"), entered_rx, dropped_rx)
}

async fn block_calibration_completion(State(state): State<BlockedCalibrationUpstream>) -> Response {
    let _drop_notifier = DropNotifier(state.dropped_tx.lock().await.take());
    if let Some(entered_tx) = state.entered_tx.lock().await.take() {
        let _ = entered_tx.send(());
    }
    std::future::pending::<Response>().await
}

#[tokio::test]
async fn assigned_sweep_does_not_block_directive_consumption() {
    let (upstream_http_base_url, sweep_entered_rx, _sweep_dropped_rx) =
        spawn_blocked_calibration_upstream().await;
    let cancel_token = CancellationToken::new();
    let (directive_tx, directive_rx) = flume::bounded(1);
    let router_endpoint = grpc_endpoint("http://router-a");
    let (_router_topology_tx, router_topology_rx) =
        watch::channel(RegistrationRouterTopology::Published(BTreeSet::from([
            router_endpoint.clone(),
        ])));
    let executor = tokio::spawn(run_cluster_calibration_executor(
        ClusterCalibrationExecutorTaskConfig {
            inference_server_id: "pylon-a".to_string(),
            cluster_id: "cluster-a".to_string(),
            retry_interval: Duration::from_secs(30),
            upstream_http_base_url,
            bringup: BringupConfig {
                calibration_requests: 1,
                calibration_timeout: Duration::from_secs(30),
                canary_timeout: Duration::from_secs(1),
                ..BringupConfig::default()
            },
            metrics: None,
            auth_token_provider: None,
        },
        directive_rx,
        router_topology_rx,
        cancel_token.clone(),
    ));
    let directive =
        |model_id: &str, state, assignment_token: &str| ClusterCalibrationDirective::Model {
            router_endpoint: router_endpoint.clone(),
            model_id: model_id.to_string(),
            state,
            assignment_token: assignment_token.to_string(),
        };

    directive_tx
        .send_async(directive(
            "model-a",
            ClusterCalibrationDirectiveState::Run,
            "token-a",
        ))
        .await
        .expect("run directive should be accepted");
    tokio::time::timeout(Duration::from_secs(1), sweep_entered_rx)
        .await
        .expect("calibration sweep should begin before queue probe")
        .expect("calibration sweep should begin before queue probe");
    directive_tx
        .send_async(directive(
            "model-a",
            ClusterCalibrationDirectiveState::Complete,
            "",
        ))
        .await
        .expect("completion directive should fill the bounded queue");

    tokio::time::timeout(
        Duration::from_secs(1),
        directive_tx.send_async(directive(
            "model-b",
            ClusterCalibrationDirectiveState::Waiting,
            "",
        )),
    )
    .await
    .expect("executor must continue draining directives during a slow sweep")
    .expect("waiting directive should enqueue after draining completion");

    cancel_token.cancel();
    executor
        .await
        .expect("cancelled calibration executor should exit cleanly");
}

#[tokio::test]
async fn router_removal_aborts_assigned_sweep() {
    let (upstream_http_base_url, sweep_entered_rx, sweep_dropped_rx) =
        spawn_blocked_calibration_upstream().await;
    let cancel_token = CancellationToken::new();
    let (directive_tx, directive_rx) = flume::bounded(1);
    let router_endpoint = grpc_endpoint("http://router-a");
    let (router_topology_tx, router_topology_rx) =
        watch::channel(RegistrationRouterTopology::Published(BTreeSet::from([
            router_endpoint.clone(),
        ])));
    let executor = tokio::spawn(run_cluster_calibration_executor(
        ClusterCalibrationExecutorTaskConfig {
            inference_server_id: "pylon-a".to_string(),
            cluster_id: "cluster-a".to_string(),
            retry_interval: Duration::from_secs(30),
            upstream_http_base_url,
            bringup: BringupConfig {
                calibration_requests: 1,
                calibration_timeout: Duration::from_secs(30),
                canary_timeout: Duration::from_secs(1),
                ..BringupConfig::default()
            },
            metrics: None,
            auth_token_provider: None,
        },
        directive_rx,
        router_topology_rx,
        cancel_token.clone(),
    ));

    directive_tx
        .send_async(ClusterCalibrationDirective::Model {
            router_endpoint,
            model_id: "model-a".to_string(),
            state: ClusterCalibrationDirectiveState::Run,
            assignment_token: "token-a".to_string(),
        })
        .await
        .expect("run directive should be accepted");
    tokio::time::timeout(Duration::from_secs(1), sweep_entered_rx)
        .await
        .expect("calibration sweep should begin before router removal")
        .expect("calibration sweep should begin before router removal");

    router_topology_tx.send_replace(RegistrationRouterTopology::Published(BTreeSet::new()));
    tokio::time::timeout(Duration::from_secs(1), sweep_dropped_rx)
        .await
        .expect("router removal should abort the running calibration sweep")
        .expect("blocked calibration request should be dropped");

    cancel_token.cancel();
    executor
        .await
        .expect("cancelled calibration executor should exit cleanly");
}

#[tokio::test]
async fn registration_disconnect_aborts_assigned_sweep_while_router_remains_discovered() {
    let (upstream_http_base_url, sweep_entered_rx, sweep_dropped_rx) =
        spawn_blocked_calibration_upstream().await;
    let cancel_token = CancellationToken::new();
    let (directive_tx, directive_rx) = flume::bounded(1);
    let router_endpoint = grpc_endpoint("http://router-a");
    let (_router_topology_tx, router_topology_rx) =
        watch::channel(RegistrationRouterTopology::Published(BTreeSet::from([
            router_endpoint.clone(),
        ])));
    let executor = tokio::spawn(run_cluster_calibration_executor(
        ClusterCalibrationExecutorTaskConfig {
            inference_server_id: "pylon-a".to_string(),
            cluster_id: "cluster-a".to_string(),
            retry_interval: Duration::from_secs(30),
            upstream_http_base_url,
            bringup: BringupConfig {
                calibration_requests: 1,
                calibration_timeout: Duration::from_secs(30),
                canary_timeout: Duration::from_secs(1),
                ..BringupConfig::default()
            },
            metrics: None,
            auth_token_provider: None,
        },
        directive_rx,
        router_topology_rx,
        cancel_token.clone(),
    ));

    directive_tx
        .send_async(ClusterCalibrationDirective::Model {
            router_endpoint: router_endpoint.clone(),
            model_id: "model-a".to_string(),
            state: ClusterCalibrationDirectiveState::Run,
            assignment_token: "token-a".to_string(),
        })
        .await
        .expect("run directive should be accepted");
    tokio::time::timeout(Duration::from_secs(1), sweep_entered_rx)
        .await
        .expect("calibration sweep should begin before registration disconnect")
        .expect("calibration sweep should begin before registration disconnect");

    directive_tx
        .send_async(ClusterCalibrationDirective::RouterDisconnected { router_endpoint })
        .await
        .expect("registration disconnect should be accepted");
    tokio::time::timeout(Duration::from_secs(1), sweep_dropped_rx)
        .await
        .expect("registration disconnect should abort the running calibration sweep")
        .expect("blocked calibration request should be dropped");

    cancel_token.cancel();
    executor
        .await
        .expect("cancelled calibration executor should exit cleanly");
}

#[test]
fn runtime_state_directly_derives_one_coherent_registration_snapshot() {
    let runtime_state =
        PylonRuntimeState::new(InferenceServerStatus::Inactive, &["model-a".to_string()]);
    let advertised_state = runtime_state.clone();

    runtime_state.set_status(InferenceServerStatus::Active);
    runtime_state.set_model_stats(
        "model-a",
        CurrentModelStats {
            output_tps: 12.5,
            ..CurrentModelStats::default()
        },
    );
    runtime_state.set_model_bringup("model-a", ModelBringupState::AdvertisingActive);

    let snapshot = advertised_state.advertised_models();
    let model = &snapshot["model-a"];

    assert_eq!(model.status, InferenceServerStatus::Active as i32);
    assert_eq!(
        model
            .stats
            .as_ref()
            .expect("stats should be present")
            .output_tps,
        12.5
    );
}

#[test]
fn snapshot_forwards_collected_model_stats_exactly() {
    let runtime_state =
        PylonRuntimeState::new(InferenceServerStatus::Active, &["model-a".to_string()]);
    runtime_state.set_model_bringup("model-a", ModelBringupState::AdvertisingActive);

    let queue_time_estimate_ms_by_priority = HashMap::from([(0, 11), (2, 7)]);
    runtime_state.set_model_stats(
        "model-a",
        CurrentModelStats {
            output_tps: 2.5,
            embedding_item_tps: 0.0,
            last_mean_input_tps: 3.5,
            queue_size: 4,
            queued_input_size: 5,
            max_output_tps: 6.5,
            max_embedding_item_tps: 0.0,
            kv_cache_capacity_tokens: 7,
            kv_cache_used_tokens: 8,
            kv_cache_free_tokens: 9,
            num_running_queries: 10,
            max_engine_concurrency: Some(11),
            total_query_input_size: 12,
            input_processing_queries: 13,
            output_generation_queries: 14,
            stats_observed_at_unix_ms: 15,
            stats_capabilities: vec!["request.output.chunk_usage".to_string()],
            stats_sources: vec!["chunk_usage".to_string()],
            queue_time_estimate_ms_by_priority: Some(queue_time_estimate_ms_by_priority.clone()),
        },
    );

    let snapshot = runtime_state.advertised_models();
    let model = &snapshot["model-a"];
    assert_eq!(model.status, InferenceServerStatus::Active as i32);
    let stats = model.stats.as_ref().expect("stats should be present");
    assert_eq!(stats.output_tps, 2.5);
    assert_eq!(stats.last_mean_input_tps, 3.5);
    assert_eq!(stats.queue_size, 4);
    assert_eq!(stats.queued_input_size, 5);
    assert_eq!(stats.max_output_tps, 6.5);
    assert_eq!(stats.kv_cache_capacity_tokens, 7);
    assert_eq!(stats.kv_cache_used_tokens, 8);
    assert_eq!(stats.kv_cache_free_tokens, 9);
    assert_eq!(stats.num_running_queries, 10);
    assert_eq!(stats.max_engine_concurrency, 11);
    assert_eq!(stats.total_query_input_size, 12);
    assert_eq!(
        stats.queue_time_estimate_ms_by_priority,
        queue_time_estimate_ms_by_priority
    );
    assert_eq!(stats.input_processing_queries, 13);
    assert_eq!(stats.output_generation_queries, 14);
    assert_eq!(stats.stats_observed_at_unix_ms, 15);
    assert_eq!(
        stats.stats_capabilities,
        vec!["request.output.chunk_usage".to_string()]
    );
    assert_eq!(stats.stats_sources, vec!["chunk_usage".to_string()]);
}

#[test]
fn reverse_tunnel_config_propagates_metrics() {
    let metrics = PylonMetrics::new().expect("metrics should initialize");
    let mut forwarding = TunnelForwardingConfig::new("http://127.0.0.1:8090/".to_string());
    forwarding.max_sse_buffer_bytes = 1234;
    forwarding.metrics = Some(metrics.clone());
    let config = build_reverse_quic_tunnel_config(ReverseQuicTunnelConfigParams {
        dial_addr: "127.0.0.1:12345".to_string(),
        sni_override: None,
        inference_server_id: "inst-a".to_string(),
        tls_cert_pem: Some(b"trusted reverse cert".to_vec()),
        quic_insecure: true,
        tunnel_protocol: TunnelTransportProtocol::Http3,
        forwarding,
        auth_token_provider: None,
    });

    assert!(
        Arc::ptr_eq(config.metrics.as_ref().unwrap(), &metrics),
        "reverse tunnel config should carry pylon metrics"
    );
    assert_eq!(config.tunnel_protocol, TunnelTransportProtocol::Http3);
    assert_eq!(config.max_sse_buffer_bytes, 1234);
    assert_eq!(
        config.tls_cert_pem.as_deref(),
        Some(&b"trusted reverse cert"[..])
    );
}

#[test]
fn reverse_tunnel_endpoint_from_ack_uses_pylon_dial_addr_and_preserves_routing_sni() {
    let endpoint = reverse_tunnel_endpoint_from_ack(&InferenceServerAck {
        reverse_tunnel_target: "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
            .to_string(),
        reverse_tunnel_pylon_dial_addr: "stargate-quic-lb.stargate.svc.cluster.local:50072"
            .to_string(),
        model_calibration_directives: Vec::new(),
    })
    .expect("ack should contain reverse tunnel endpoint");

    assert_eq!(
        endpoint.pylon_dial_addr,
        "stargate-quic-lb.stargate.svc.cluster.local:50072"
    );
    assert_eq!(
        endpoint.routing_target_addr,
        "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
    );
    assert_eq!(
        endpoint.sni_override.as_deref(),
        Some("stargate-0.stargate-headless.stargate.svc.cluster.local")
    );
}

#[test]
fn reverse_tunnel_endpoint_from_ack_rejects_empty_pylon_dial_addr() {
    let endpoint = reverse_tunnel_endpoint_from_ack(&InferenceServerAck {
        reverse_tunnel_target: "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
            .to_string(),
        reverse_tunnel_pylon_dial_addr: String::new(),
        model_calibration_directives: Vec::new(),
    });

    assert!(endpoint.is_none());
}

#[test]
fn reverse_tunnel_state_replaces_connected_endpoint_atomically() {
    let endpoint_a = ReverseTunnelEndpoint {
        routing_target_addr: "router-a:50072".to_string(),
        pylon_dial_addr: "dial-a:50072".to_string(),
        sni_override: Some("router-a".to_string()),
    };
    let endpoint_b = ReverseTunnelEndpoint {
        routing_target_addr: "router-b:50072".to_string(),
        pylon_dial_addr: "dial-b:50072".to_string(),
        sni_override: Some("router-b".to_string()),
    };
    let mut state = ReverseTunnelState::default();

    assert!(state.replace_endpoint(Some(endpoint_a.clone())));
    assert!(state.mark_connected(&endpoint_a));
    assert!(state.is_connected());

    assert!(state.replace_endpoint(Some(endpoint_b.clone())));
    assert_eq!(state.endpoint(), Some(&endpoint_b));
    assert!(!state.is_connected());
    assert!(
        !state.mark_connected(&endpoint_a),
        "a delayed connection for the replaced endpoint must be rejected"
    );
    assert!(state.mark_connected(&endpoint_b));
    assert!(state.is_connected());
    assert!(!state.replace_endpoint(Some(endpoint_b.clone())));
    assert!(
        state.is_connected(),
        "a repeated ACK for the same endpoint must preserve connectivity"
    );
}

#[tokio::test]
async fn reverse_tunnel_connect_attempt_times_out() {
    let result = reverse_tunnel_connect_with_timeout(
        Duration::from_millis(1),
        std::future::pending::<Result<ReverseQuicTunnelHandle, TunnelError>>(),
    )
    .await;

    assert!(
        matches!(result, Err(TunnelError::ConnectTimeout { timeout_ms: 1 })),
        "expected timeout connect error"
    );
}

#[test]
fn reverse_tunnel_config_propagates_request_quality_monitor() {
    let request_quality_monitor = RequestQualityMonitorConfig {
        collect_quality_metrics: true,
        collect_quality_metrics_min_tokens: 7,
        output_tokens_threshold_min: Some(9),
        output_compression_threshold_max: Some(0.4),
        output_degeneracy_threshold_min: Some(0.5),
        output_repetition_1gram_threshold_min: Some(0.6),
        output_repetition_2gram_threshold_min: Some(0.7),
        output_repetition_3gram_threshold_min: Some(0.8),
        median_logprob_threshold_max: Some(-6.5),
    };

    let mut forwarding = TunnelForwardingConfig::new("http://127.0.0.1:8090/".to_string());
    forwarding.request_quality_monitor = request_quality_monitor.clone();
    let config = build_reverse_quic_tunnel_config(ReverseQuicTunnelConfigParams {
        dial_addr: "127.0.0.1:12345".to_string(),
        sni_override: None,
        inference_server_id: "inst-a".to_string(),
        tls_cert_pem: None,
        quic_insecure: true,
        tunnel_protocol: TunnelTransportProtocol::Custom,
        forwarding,
        auth_token_provider: None,
    });

    assert!(config.request_quality_monitor.collect_quality_metrics);
    assert_eq!(
        config
            .request_quality_monitor
            .collect_quality_metrics_min_tokens,
        7
    );
    assert_eq!(
        config.request_quality_monitor.output_tokens_threshold_min,
        Some(9)
    );
    assert_eq!(
        config
            .request_quality_monitor
            .output_compression_threshold_max,
        Some(0.4)
    );
    assert_eq!(
        config
            .request_quality_monitor
            .output_degeneracy_threshold_min,
        Some(0.5)
    );
    assert_eq!(
        config
            .request_quality_monitor
            .output_repetition_1gram_threshold_min,
        Some(0.6)
    );
    assert_eq!(
        config
            .request_quality_monitor
            .output_repetition_2gram_threshold_min,
        Some(0.7)
    );
    assert_eq!(
        config
            .request_quality_monitor
            .output_repetition_3gram_threshold_min,
        Some(0.8)
    );
    assert_eq!(
        config.request_quality_monitor.median_logprob_threshold_max,
        Some(-6.5)
    );
}

#[test]
fn router_registration_task_harness_propagates_request_quality_monitor_to_each_router() {
    let request_quality_monitor = RequestQualityMonitorConfig {
        collect_quality_metrics: true,
        collect_quality_metrics_min_tokens: 7,
        output_tokens_threshold_min: Some(9),
        output_compression_threshold_max: Some(0.4),
        output_degeneracy_threshold_min: Some(0.5),
        output_repetition_1gram_threshold_min: Some(0.6),
        output_repetition_2gram_threshold_min: Some(0.7),
        output_repetition_3gram_threshold_min: Some(0.8),
        median_logprob_threshold_max: Some(-6.5),
    };
    let register_config = InferenceServerRegistrationConfig {
        seeds: vec!["router-a".to_string(), "router-b".to_string()],
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-a".to_string(),
        inference_server_url: "quic://127.0.0.1:8443".to_string(),
        upstream_http_base_url: Some("http://127.0.0.1:8090".to_string()),
        min_update_interval: Duration::from_secs(2),
        reverse_tunnel: true,
        tls_cert_pem: Some(b"trusted reverse cert".to_vec()),
        quic_insecure: true,
        tunnel_protocol: TunnelTransportProtocol::Http3,
        bringup: BringupConfig::default(),
        output_token_parser_factory: OutputTokenParserFactory,
        runtime_state: PylonRuntimeState::default(),
        request_quality_monitor: request_quality_monitor.clone(),
        metrics: None,
        retry: PylonRetryConfig::default(),
        queue_mismatch_retry: PylonQueueMismatchRetryConfig::default(),
        auth_token_provider: None,
    };
    let (cluster_calibration_directive_tx, _cluster_calibration_directive_rx) = flume::bounded(1);
    let task_template = RouterRegistrationTaskTemplate::from_registration_config(
        &register_config,
        &register_config.cluster_id,
        register_config.upstream_http_base_url.as_deref().unwrap(),
        cluster_calibration_directive_tx,
    );

    for router in ["router-a", "router-b"] {
        let task_config = task_template.build_for_router(grpc_endpoint(router));
        assert_eq!(task_config.router_endpoint.authority_addr(), router);
        assert_eq!(task_config.inference_server_id, "inst-a");
        assert_eq!(task_config.cluster_id, "cluster-a");
        assert_eq!(task_config.inference_server_url, "http://127.0.0.1:8090");
        assert_eq!(task_config.min_update_interval, Duration::from_secs(2));
        assert!(task_config.reverse_tunnel);
        assert!(task_config.coordinated_calibration);
        assert_eq!(
            task_config.tls_cert_pem.as_deref(),
            Some(&b"trusted reverse cert"[..])
        );
        assert!(task_config.quic_insecure);
        assert_eq!(task_config.tunnel_protocol, TunnelTransportProtocol::Http3);
        assert_eq!(
            task_config.forwarding.upstream_http_base_url,
            "http://127.0.0.1:8090"
        );
        assert!(
            task_config
                .forwarding
                .request_quality_monitor
                .collect_quality_metrics
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .collect_quality_metrics_min_tokens,
            7
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_tokens_threshold_min,
            Some(9)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_compression_threshold_max,
            Some(0.4)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_degeneracy_threshold_min,
            Some(0.5)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_repetition_1gram_threshold_min,
            Some(0.6)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_repetition_2gram_threshold_min,
            Some(0.7)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .output_repetition_3gram_threshold_min,
            Some(0.8)
        );
        assert_eq!(
            task_config
                .forwarding
                .request_quality_monitor
                .median_logprob_threshold_max,
            Some(-6.5)
        );
    }
}
