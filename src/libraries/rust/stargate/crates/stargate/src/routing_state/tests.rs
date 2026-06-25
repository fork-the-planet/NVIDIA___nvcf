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

use super::registration::{
    RegistrationClusterGeneration, test_registration_generation,
    test_registration_generation_in_cluster,
};
use super::reservations::update_reserved_priority_queue_time;
use super::snapshots::{ClusterBackendUpsert, RoutedClusterState, RoutingTargetGeneration};
use super::*;
use crate::load_balancer::{
    LoadBalancerAlgorithm, LoadBalancerConfig, LoadBalancerRequest, LoadBalancerRouter,
};
use stargate_proto::pb::InferenceServerModelRegistration;
use std::collections::HashMap;
use std::sync::Arc;

fn model_registration(status: i32) -> InferenceServerModelRegistration {
    InferenceServerModelRegistration {
        stats: Some(ModelStats::default()),
        status,
    }
}

fn model_registration_with_stats(
    status: i32,
    stats: ModelStats,
) -> InferenceServerModelRegistration {
    InferenceServerModelRegistration {
        stats: Some(stats),
        status,
    }
}

fn running_registration(
    state: &StargateState,
    id: &str,
    url: &str,
    routing_key: Option<&str>,
) -> RunningRegistration {
    running_registration_in_cluster(state, id, id, url, routing_key)
}

fn running_registration_in_cluster(
    state: &StargateState,
    id: &str,
    cluster_id: &str,
    url: &str,
    routing_key: Option<&str>,
) -> RunningRegistration {
    let identity = RegistrationIdentity {
        inference_server_id: id.to_string(),
        cluster_id: cluster_id.to_string(),
        inference_server_url: url.to_string(),
        routing_key: routing_key.map(ToOwned::to_owned),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    state.begin_registration(&identity).unwrap()
}

fn running_coordinated_registration_in_cluster(
    state: &StargateState,
    id: &str,
    cluster_id: &str,
    url: &str,
    routing_key: Option<&str>,
) -> RunningRegistration {
    let identity = RegistrationIdentity {
        inference_server_id: id.to_string(),
        cluster_id: cluster_id.to_string(),
        inference_server_url: url.to_string(),
        routing_key: routing_key.map(ToOwned::to_owned),
        reverse_tunnel: false,
        coordinated_calibration: true,
    };
    state.begin_registration(&identity).unwrap()
}

fn make_target(routing_key: Option<&str>, model_id: &str) -> RoutingTargetKey {
    RoutingTargetKey {
        routing_key: routing_key.map(ToOwned::to_owned),
        model_id: model_id.to_string(),
    }
}

async fn selected_cluster_for_target(
    state: &StargateState,
    target: &RoutingTargetKey,
) -> SelectedRoutedCluster {
    state
        .routing_target_snapshot(target)
        .await
        .expect("target should be routable")
        .into_selected_cluster(0)
}

fn routed_backend(
    cluster_id: &str,
    inference_server_id: &str,
    output_tps: f64,
    last_mean_input_tps: f64,
    rtt: Duration,
) -> RoutedInferenceServerSnapshot {
    let registration = test_registration_generation(RegistrationIdentity {
        inference_server_id: inference_server_id.to_string(),
        cluster_id: cluster_id.to_string(),
        inference_server_url: format!("quic://{inference_server_id}"),
        routing_key: None,
        reverse_tunnel: false,
        coordinated_calibration: false,
    });
    RoutedInferenceServerSnapshot {
        registration,
        cluster_id: cluster_id.to_string(),
        inference_server_id: inference_server_id.to_string(),
        inference_server_url: format!("quic://{inference_server_id}"),
        stats: ModelStats {
            output_tps,
            last_mean_input_tps,
            ..ModelStats::default()
        },
        rtt,
        snapshot_updated_at: Instant::now(),
        status: InferenceServerStatus::Active,
        reverse_tunnel: false,
    }
}

fn routed_backend_in_cluster_generation(
    cluster_generation: Arc<RegistrationClusterGeneration>,
    cluster_id: &str,
    inference_server_id: &str,
    output_tps: f64,
    last_mean_input_tps: f64,
    rtt: Duration,
) -> RoutedInferenceServerSnapshot {
    let mut snapshot = routed_backend(
        cluster_id,
        inference_server_id,
        output_tps,
        last_mean_input_tps,
        rtt,
    );
    let registration = test_registration_generation_in_cluster(
        RegistrationIdentity {
            inference_server_id: snapshot.inference_server_id.clone(),
            cluster_id: snapshot.cluster_id.clone(),
            inference_server_url: snapshot.inference_server_url.clone(),
            routing_key: None,
            reverse_tunnel: snapshot.reverse_tunnel,
            coordinated_calibration: false,
        },
        cluster_generation,
    );
    snapshot.registration = registration;
    snapshot
}

fn routed_backend_snapshot_in_cluster_generation(
    mut snapshot: RoutedInferenceServerSnapshot,
    cluster_generation: Arc<RegistrationClusterGeneration>,
) -> RoutedInferenceServerSnapshot {
    let registration = test_registration_generation_in_cluster(
        RegistrationIdentity {
            inference_server_id: snapshot.inference_server_id.clone(),
            cluster_id: snapshot.cluster_id.clone(),
            inference_server_url: snapshot.inference_server_url.clone(),
            routing_key: None,
            reverse_tunnel: snapshot.reverse_tunnel,
            coordinated_calibration: false,
        },
        cluster_generation,
    );
    snapshot.registration = registration;
    snapshot
}

fn registration_update(
    running: &RunningRegistration,
    model_id: &str,
    status: i32,
    stats: ModelStats,
) -> InferenceServerRegistration {
    let identity = running.identity();
    InferenceServerRegistration {
        inference_server_id: identity.inference_server_id.clone(),
        cluster_id: identity.cluster_id.clone(),
        inference_server_url: identity.inference_server_url.clone(),
        models: HashMap::from([(
            model_id.to_string(),
            model_registration_with_stats(status, stats),
        )]),
        reverse_tunnel: identity.reverse_tunnel,
        coordinated_calibration: identity.coordinated_calibration,
    }
}

fn empty_registration_update(running: &RunningRegistration) -> InferenceServerRegistration {
    let identity = running.identity();
    InferenceServerRegistration {
        inference_server_id: identity.inference_server_id.clone(),
        cluster_id: identity.cluster_id.clone(),
        inference_server_url: identity.inference_server_url.clone(),
        models: HashMap::new(),
        reverse_tunnel: identity.reverse_tunnel,
        coordinated_calibration: identity.coordinated_calibration,
    }
}

async fn submit_assigned_calibration(
    state: &StargateState,
    routing_key: Option<&str>,
    inference_server_id: &str,
    cluster_id: &str,
    model_id: &str,
    assignment_token: &str,
    last_mean_input_tps: f64,
) {
    state
        .submit_cluster_calibration(
            routing_key.map(ToOwned::to_owned),
            &SubmitClusterCalibrationRequest {
                inference_server_id: inference_server_id.to_string(),
                cluster_id: cluster_id.to_string(),
                model_id: model_id.to_string(),
                assignment_token: assignment_token.to_string(),
                measured_last_mean_input_tps: last_mean_input_tps,
            },
        )
        .await
        .expect("assigned local calibration result should be accepted");
}

#[tokio::test]
async fn registration_cluster_generation_tracks_overlap_and_final_retirement() {
    let state = StargateState::default();
    let (running_a, running_b) = std::thread::scope(|scope| {
        let registration_a = scope.spawn(|| {
            running_registration_in_cluster(
                &state,
                "inst-cluster-generation-a",
                "cluster-generation",
                "quic://127.0.0.1:1111",
                Some("rk-cluster-generation"),
            )
        });
        let registration_b = scope.spawn(|| {
            running_registration_in_cluster(
                &state,
                "inst-cluster-generation-b",
                "cluster-generation",
                "quic://127.0.0.1:2222",
                Some("rk-cluster-generation"),
            )
        });
        (
            registration_a
                .join()
                .expect("registration A thread should not panic"),
            registration_b
                .join()
                .expect("registration B thread should not panic"),
        )
    });
    let first_generation = running_a.cluster_generation().clone();

    assert!(Arc::ptr_eq(
        &first_generation,
        running_b.cluster_generation()
    ));
    state.end_registration(running_a).await;
    assert!(!first_generation.is_retired());

    let running_c = running_registration_in_cluster(
        &state,
        "inst-cluster-generation-c",
        "cluster-generation",
        "quic://127.0.0.1:3333",
        Some("rk-cluster-generation"),
    );
    assert!(Arc::ptr_eq(
        &first_generation,
        running_c.cluster_generation()
    ));

    state.end_registration(running_b).await;
    state.end_registration(running_c).await;
    assert!(first_generation.is_retired());

    let replacement = running_registration_in_cluster(
        &state,
        "inst-cluster-generation-replacement",
        "cluster-generation",
        "quic://127.0.0.1:4444",
        Some("rk-cluster-generation"),
    );
    assert!(!Arc::ptr_eq(
        &first_generation,
        replacement.cluster_generation()
    ));
    assert!(!replacement.cluster_generation().is_retired());
    state.end_registration(replacement).await;
}

#[tokio::test]
async fn stale_registration_cleanup_cannot_remove_replacement_route() {
    let state = StargateState::default();
    let target = make_target(Some("rk-reused-generation"), "model-reused-generation");
    let old = running_registration_in_cluster(
        &state,
        "inst-reused-generation",
        "cluster-reused-generation",
        "quic://127.0.0.1:1111",
        Some("rk-reused-generation"),
    );
    let old_update = registration_update(
        &old,
        "model-reused-generation",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(&old, &old_update, true, Some(Duration::from_millis(5)))
        .await;

    let ended = state
        .registrations
        .end_registration(old)
        .expect("exact old registration should be removed");
    assert!(ended.registration.cluster_generation.is_retired());

    let replacement = running_registration_in_cluster(
        &state,
        "inst-reused-generation",
        "cluster-reused-generation",
        "quic://127.0.0.1:2222",
        Some("rk-reused-generation"),
    );
    let replacement_update = registration_update(
        &replacement,
        "model-reused-generation",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;
    let candidates = state.candidates_for_target(&target).await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].inference_server_url, "quic://127.0.0.1:2222");

    state
        .routing
        .remove_inference_server_targets(&ended.registration, &HashSet::from([target.clone()]))
        .await;

    let candidates = state.candidates_for_target(&target).await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].inference_server_url, "quic://127.0.0.1:2222");
    state.end_registration(replacement).await;
}

#[tokio::test]
async fn stale_cleanup_cannot_remove_same_cluster_generation_replacement_route() {
    let state = StargateState::default();
    let target = make_target(Some("rk-reused-overlap"), "model-reused-overlap");
    let old = running_registration_in_cluster(
        &state,
        "inst-reused-overlap",
        "cluster-reused-overlap",
        "quic://127.0.0.1:1111",
        Some("rk-reused-overlap"),
    );
    let peer = running_registration_in_cluster(
        &state,
        "inst-reused-overlap-peer",
        "cluster-reused-overlap",
        "quic://127.0.0.1:3333",
        Some("rk-reused-overlap"),
    );
    let old_update = registration_update(
        &old,
        "model-reused-overlap",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(&old, &old_update, true, Some(Duration::from_millis(5)))
        .await;

    let ended = state
        .registrations
        .end_registration(old)
        .expect("exact old registration should be removed");
    assert!(!ended.registration.cluster_generation.is_retired());

    let replacement = running_registration_in_cluster(
        &state,
        "inst-reused-overlap",
        "cluster-reused-overlap",
        "quic://127.0.0.1:2222",
        Some("rk-reused-overlap"),
    );
    assert!(Arc::ptr_eq(
        &ended.registration.cluster_generation,
        replacement.cluster_generation()
    ));
    let replacement_update = registration_update(
        &replacement,
        "model-reused-overlap",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    state
        .routing
        .remove_inference_server_targets(&ended.registration, &HashSet::from([target.clone()]))
        .await;

    let candidates = state.candidates_for_target(&target).await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].inference_server_url, "quic://127.0.0.1:2222");
    state.end_registration(peer).await;
    state.end_registration(replacement).await;
}

#[tokio::test]
async fn stale_selected_registration_cannot_reserve_same_id_replacement() {
    let state = StargateState::default();
    let target = make_target(Some("rk-stale-reservation"), "model-stale-reservation");
    let old = running_registration_in_cluster(
        &state,
        "inst-stale-reservation",
        "cluster-stale-reservation",
        "quic://127.0.0.1:1111",
        Some("rk-stale-reservation"),
    );
    let old_update = registration_update(
        &old,
        "model-stale-reservation",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 100.0,
            queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(&old, &old_update, true, Some(Duration::from_millis(5)))
        .await;
    let stale_selected = state
        .candidates_for_target(&target)
        .await
        .into_iter()
        .next()
        .expect("old registration should be routable");
    let selected_cluster = selected_cluster_for_target(&state, &target).await;

    state.end_registration(old).await;
    let replacement = running_registration_in_cluster(
        &state,
        "inst-stale-reservation",
        "cluster-stale-reservation",
        "quic://127.0.0.1:2222",
        Some("rk-stale-reservation"),
    );
    let replacement_update = registration_update(
        &replacement,
        "model-stale-reservation",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 100.0,
            queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    let reservation = selected_cluster.reserve_backend(&stale_selected.registration, 37, 4);

    assert!(
        reservation.is_none(),
        "a stale selected registration must not reserve its same-ID replacement"
    );
    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters[0].stats.num_running_queries, 0);
    assert_eq!(clusters[0].stats.total_query_input_size, 0);
}

#[tokio::test]
async fn stale_selected_cluster_cannot_choose_same_id_replacement() {
    let state = StargateState::default();
    let target = make_target(Some("rk-stale-cluster"), "model-stale-cluster");
    let old = running_registration_in_cluster(
        &state,
        "inst-stale-cluster",
        "cluster-stale-cluster",
        "quic://127.0.0.1:1111",
        Some("rk-stale-cluster"),
    );
    let old_update = registration_update(
        &old,
        "model-stale-cluster",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(&old, &old_update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = state
        .routing_target_snapshot(&target)
        .await
        .expect("old target should be routable")
        .into_selected_cluster(0);

    state.end_registration(old).await;
    let replacement = running_registration_in_cluster(
        &state,
        "inst-stale-cluster",
        "cluster-stale-cluster",
        "quic://127.0.0.1:2222",
        Some("rk-stale-cluster"),
    );
    let replacement_update = registration_update(
        &replacement,
        "model-stale-cluster",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    assert!(
        selected_cluster.select_backend(&HashSet::new()).is_none(),
        "a stale selected cluster must not choose from its same-ID replacement"
    );
}

#[tokio::test]
async fn inactive_fresh_cluster_generation_clears_retired_routing_state() {
    let state = StargateState::default();
    let target = make_target(Some("rk-retired-cluster"), "model-retired-cluster");
    let old = running_registration_in_cluster(
        &state,
        "inst-retired-cluster",
        "cluster-retired-cluster",
        "quic://127.0.0.1:1111",
        Some("rk-retired-cluster"),
    );
    let old_update = registration_update(
        &old,
        "model-retired-cluster",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(&old, &old_update, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(state.candidates_for_target(&target).await.len(), 1);

    let ended = state
        .registrations
        .end_registration(old)
        .expect("exact old registration should be removed");
    assert!(ended.registration.cluster_generation.is_retired());

    let replacement = running_registration_in_cluster(
        &state,
        "inst-retired-cluster",
        "cluster-retired-cluster",
        "quic://127.0.0.1:2222",
        Some("rk-retired-cluster"),
    );
    let inactive_update = registration_update(
        &replacement,
        "model-retired-cluster",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(
            &replacement,
            &inactive_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    assert!(state.candidates_for_target(&target).await.is_empty());
    state.end_registration(replacement).await;
}

#[tokio::test]
async fn apply_registration_update_removes_models_no_longer_advertised() {
    let state = StargateState::default();
    let running = running_registration(&state, "inst-1", "quic://127.0.0.1:1234", Some("rk-1"));
    let initial_update = InferenceServerRegistration {
        inference_server_id: "inst-1".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:1234".to_string(),
        models: HashMap::from([(
            "model-a".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    state
        .apply_registration_update(
            &running,
            &initial_update,
            true,
            Some(Duration::from_millis(10)),
        )
        .await;

    let update = InferenceServerRegistration {
        inference_server_id: "inst-1".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:1234".to_string(),
        models: HashMap::from([(
            "model-b".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(10)))
        .await;

    assert!(
        state
            .candidates_for_target(&make_target(Some("rk-1"), "model-a"))
            .await
            .is_empty()
    );
    assert_eq!(
        state
            .candidates_for_target(&make_target(Some("rk-1"), "model-b"))
            .await
            .len(),
        1
    );
    assert!(
        state
            .routing
            .target_state(&make_target(Some("rk-1"), "model-a"))
            .await
            .is_none()
    );
}

#[tokio::test]
async fn registered_inactive_model_is_known_without_routable_candidates() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-known-inactive",
        "quic://127.0.0.1:1234",
        Some("rk-known"),
    );
    let target = make_target(Some("rk-known"), "model-known");
    let update = registration_update(
        &running,
        "model-known",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(10)))
        .await;

    assert!(state.has_registered_model_for_target(&target));
    assert!(state.candidates_for_target(&target).await.is_empty());
    assert!(!state.has_registered_model_for_target(&make_target(Some("wrong-rk"), "model-known")));

    state.end_registration(running).await;
    assert!(!state.has_registered_model_for_target(&target));
}

#[tokio::test]
async fn registered_target_membership_survives_until_final_advertiser_leaves() {
    let state = StargateState::default();
    let running_a = running_registration(
        &state,
        "inst-shared-target-a",
        "quic://127.0.0.1:1234",
        Some("rk-shared-target"),
    );
    let running_b = running_registration(
        &state,
        "inst-shared-target-b",
        "quic://127.0.0.1:1235",
        Some("rk-shared-target"),
    );
    let target = make_target(Some("rk-shared-target"), "model-shared-target");

    for running in [&running_a, &running_b] {
        let update = registration_update(
            running,
            "model-shared-target",
            InferenceServerStatus::Inactive as i32,
            ModelStats::default(),
        );
        state
            .apply_registration_update(running, &update, true, Some(Duration::from_millis(10)))
            .await;
    }

    assert!(state.has_registered_model_for_target(&target));
    assert!(
        !state
            .has_registered_model_for_target(&make_target(Some("rk-other"), "model-shared-target"))
    );

    state.end_registration(running_a).await;
    assert!(state.has_registered_model_for_target(&target));

    let empty_update = empty_registration_update(&running_b);
    state
        .apply_registration_update(
            &running_b,
            &empty_update,
            true,
            Some(Duration::from_millis(10)),
        )
        .await;
    assert!(!state.has_registered_model_for_target(&target));

    state.end_registration(running_b).await;
}

#[tokio::test]
async fn registration_model_update_publishes_advertised_generation_and_retains_cleanup_union() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-generation",
        "quic://127.0.0.1:1234",
        Some("rk-generation"),
    );

    let first = BTreeSet::from(["model-a".to_string(), "model-b".to_string()]);
    assert!(
        state
            .registrations
            .begin_advertised_model_update(&running, first.clone())
            .is_empty()
    );
    assert!(state.has_registered_model_for_target(&make_target(Some("rk-generation"), "model-a")));
    assert!(state.has_registered_model_for_target(&make_target(Some("rk-generation"), "model-b")));
    assert_eq!(state.registrations.cleanup_model_ids(&running), first);
    state.registrations.finish_advertised_model_update(&running);

    let second = BTreeSet::from(["model-b".to_string(), "model-c".to_string()]);
    assert_eq!(
        state
            .registrations
            .begin_advertised_model_update(&running, second.clone()),
        BTreeSet::from(["model-a".to_string()])
    );
    assert!(!state.has_registered_model_for_target(&make_target(Some("rk-generation"), "model-a")));
    assert!(state.has_registered_model_for_target(&make_target(Some("rk-generation"), "model-b")));
    assert!(state.has_registered_model_for_target(&make_target(Some("rk-generation"), "model-c")));
    assert_eq!(
        state.registrations.cleanup_model_ids(&running),
        BTreeSet::from([
            "model-a".to_string(),
            "model-b".to_string(),
            "model-c".to_string(),
        ])
    );
    state.registrations.finish_advertised_model_update(&running);
    assert_eq!(state.registrations.cleanup_model_ids(&running), second);
}

#[tokio::test]
#[should_panic(expected = "registration model update started while another update is applying")]
async fn registration_model_update_rejects_overlapping_apply() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-overlap",
        "quic://127.0.0.1:1234",
        Some("rk-overlap"),
    );

    state
        .registrations
        .begin_advertised_model_update(&running, BTreeSet::new());
    state
        .registrations
        .begin_advertised_model_update(&running, BTreeSet::new());
}

#[tokio::test]
async fn registration_cleanup_during_applying_update_removes_previous_and_advertised_routes() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-cleanup-union",
        "quic://127.0.0.1:1234",
        Some("rk-cleanup-union"),
    );
    let initial = registration_update(
        &running,
        "model-a",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    state
        .apply_registration_update(&running, &initial, true, Some(Duration::from_millis(5)))
        .await;

    let model_a = make_target(Some("rk-cleanup-union"), "model-a");
    let model_b = make_target(Some("rk-cleanup-union"), "model-b");
    state
        .registrations
        .begin_advertised_model_update(&running, BTreeSet::from(["model-b".to_string()]));
    state
        .routing
        .upsert_inference_server_target(
            &model_b,
            RoutedInferenceServerSnapshot::new(
                running.generation(),
                ModelStats::default(),
                Duration::from_millis(5),
                Instant::now(),
                InferenceServerStatus::Active,
            ),
        )
        .await;

    state.end_registration(running).await;

    assert!(state.candidates_for_target(&model_a).await.is_empty());
    assert!(state.candidates_for_target(&model_b).await.is_empty());
}

#[tokio::test]
async fn active_registration_keeps_connection_rtt_in_snapshot() {
    let state = StargateState::default();
    let running = running_registration(&state, "inst-rtt", "quic://127.0.0.1:7777", Some("rk-rtt"));
    let update = InferenceServerRegistration {
        inference_server_id: "inst-rtt".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:7777".to_string(),
        models: HashMap::from([(
            "model-rtt".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    let expected_rtt = Duration::from_millis(42);
    state
        .apply_registration_update(&running, &update, true, Some(expected_rtt))
        .await;

    let candidates = state
        .candidates_for_target(&make_target(Some("rk-rtt"), "model-rtt"))
        .await;
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].rtt, expected_rtt);
    assert!(Arc::ptr_eq(
        &candidates[0].registration,
        &running.generation()
    ));
}

#[tokio::test]
async fn coordinated_calibration_decision_keeps_directive_and_routing_gate_consistent() {
    let state = StargateState::default();
    let running = running_coordinated_registration_in_cluster(
        &state,
        "inst-atomic-decision",
        "cluster-atomic-decision",
        "quic://127.0.0.1:1111",
        Some("rk-atomic-decision"),
    );

    let assigned = running
        .cluster_generation()
        .calibrations
        .registration_decision(running.identity(), "model-atomic-decision")
        .await;
    let (assigned_directive, assigned_routing_gated) = assigned.into_parts();
    let assigned_directive =
        assigned_directive.expect("coordinated registration should receive a directive");
    assert_eq!(assigned_directive.state, CalibrationState::Run as i32);
    assert!(assigned_routing_gated);

    submit_assigned_calibration(
        &state,
        Some("rk-atomic-decision"),
        "inst-atomic-decision",
        "cluster-atomic-decision",
        "model-atomic-decision",
        &assigned_directive.assignment_token,
        125.0,
    )
    .await;

    let completed = running
        .cluster_generation()
        .calibrations
        .registration_decision(running.identity(), "model-atomic-decision")
        .await;
    let (completed_directive, completed_routing_gated) = completed.into_parts();
    let completed_directive =
        completed_directive.expect("completed calibration should still return a directive");
    assert_eq!(completed_directive.state, CalibrationState::Complete as i32);
    assert!(!completed_routing_gated);
}

#[tokio::test]
async fn coordinated_calibration_submission_preserves_transition_results() {
    let state = StargateState::default();
    let running = running_coordinated_registration_in_cluster(
        &state,
        "inst-submission-results",
        "cluster-submission-results",
        "quic://127.0.0.1:1111",
        Some("rk-submission-results"),
    );
    let request = |inference_server_id: &str, assignment_token: &str, measured: f64| {
        SubmitClusterCalibrationRequest {
            inference_server_id: inference_server_id.to_string(),
            cluster_id: "cluster-submission-results".to_string(),
            model_id: "model-submission-results".to_string(),
            assignment_token: assignment_token.to_string(),
            measured_last_mean_input_tps: measured,
        }
    };

    let missing = state
        .submit_cluster_calibration(
            Some("rk-submission-results".to_string()),
            &request("inst-submission-results", "missing-token", 125.0),
        )
        .await
        .expect_err("submission without an assignment should fail");
    assert_eq!(
        missing.message(),
        "cluster calibration has no active local assignment"
    );

    let assigned = running
        .cluster_generation()
        .calibrations
        .registration_decision(running.identity(), "model-submission-results")
        .await;
    let (directive, routing_gated) = assigned.into_parts();
    let assignment_token = directive
        .expect("coordinated registration should receive an assignment")
        .assignment_token;
    assert!(routing_gated);

    let wrong_owner = state
        .submit_cluster_calibration(
            Some("rk-submission-results".to_string()),
            &request("inst-other", &assignment_token, 125.0),
        )
        .await
        .expect_err("a sibling should not own the assignment");
    assert_eq!(
        wrong_owner.message(),
        "cluster calibration submission does not own the local assignment"
    );

    let completed = request("inst-submission-results", &assignment_token, 125.0);
    state
        .submit_cluster_calibration(Some("rk-submission-results".to_string()), &completed)
        .await
        .expect("the assignment owner should complete calibration");
    state
        .submit_cluster_calibration(Some("rk-submission-results".to_string()), &completed)
        .await
        .expect("an exact repeated submission should be idempotent");

    let conflicting = state
        .submit_cluster_calibration(
            Some("rk-submission-results".to_string()),
            &request("inst-submission-results", &assignment_token, 126.0),
        )
        .await
        .expect_err("a conflicting repeated submission should fail");
    assert_eq!(
        conflicting.message(),
        "cluster calibration was already completed by another submission"
    );
}

#[tokio::test]
async fn coordinated_calibration_assigns_one_owner_and_gates_siblings_until_complete() {
    let state = StargateState::default();
    let running_a = running_coordinated_registration_in_cluster(
        &state,
        "inst-a",
        "cluster-cal",
        "quic://127.0.0.1:1111",
        Some("rk-cal"),
    );
    let running_b = running_coordinated_registration_in_cluster(
        &state,
        "inst-b",
        "cluster-cal",
        "quic://127.0.0.1:2222",
        Some("rk-cal"),
    );
    let target = make_target(Some("rk-cal"), "model-cal");

    let update_a = registration_update(
        &running_a,
        "model-cal",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].model_id, "model-cal");
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    assert!(!directives[0].assignment_token.is_empty());
    let assignment_token = directives[0].assignment_token.clone();

    let update_b_active_without_calibration = registration_update(
        &running_b,
        "model-cal",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_b,
            &update_b_active_without_calibration,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Waiting as i32);
    assert!(state.candidates_for_target(&target).await.is_empty());

    submit_assigned_calibration(
        &state,
        Some("rk-cal"),
        "inst-a",
        "cluster-cal",
        "model-cal",
        &assignment_token,
        150.0,
    )
    .await;
    let update_a_complete = registration_update(
        &running_a,
        "model-cal",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_a,
            &update_a_complete,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);

    let update_b_complete = registration_update(
        &running_b,
        "model-cal",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 120.0,
            ..ModelStats::default()
        },
    );
    let directives = state
        .apply_registration_update(
            &running_b,
            &update_b_complete,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].active_backend_count, 2);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 150.0);
}

#[tokio::test]
async fn coordinated_calibration_is_not_summed_with_runtime_reports() {
    let state = StargateState::default();
    let running_seed = running_coordinated_registration_in_cluster(
        &state,
        "inst-seed",
        "cluster-capacity-source",
        "quic://127.0.0.1:1111",
        Some("rk-capacity-source"),
    );
    let running_runtime = running_coordinated_registration_in_cluster(
        &state,
        "inst-runtime",
        "cluster-capacity-source",
        "quic://127.0.0.1:2222",
        Some("rk-capacity-source"),
    );
    let target = make_target(Some("rk-capacity-source"), "model-capacity-source");

    let seed_update = registration_update(
        &running_seed,
        "model-capacity-source",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let assignment = state
        .apply_registration_update(
            &running_seed,
            &seed_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    submit_assigned_calibration(
        &state,
        Some("rk-capacity-source"),
        "inst-seed",
        "cluster-capacity-source",
        "model-capacity-source",
        &assignment[0].assignment_token,
        150.0,
    )
    .await;
    state
        .apply_registration_update(
            &running_seed,
            &seed_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let runtime_update = registration_update(
        &running_runtime,
        "model-capacity-source",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 120.0,
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(
            &running_runtime,
            &runtime_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(
        clusters[0].stats.last_mean_input_tps, 150.0,
        "cluster calibration must remain independent from backend-local runtime capacity"
    );
}

#[tokio::test]
async fn coordinated_calibration_remains_cluster_capacity_floor_after_backends_report_runtime() {
    let state = StargateState::default();
    let running_owner = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-capacity-floor",
        "quic://127.0.0.1:1111",
        Some("rk-capacity-floor"),
    );
    let running_peer = running_coordinated_registration_in_cluster(
        &state,
        "inst-peer",
        "cluster-capacity-floor",
        "quic://127.0.0.1:2222",
        Some("rk-capacity-floor"),
    );
    let target = make_target(Some("rk-capacity-floor"), "model-capacity-floor");

    let calibration_update = registration_update(
        &running_owner,
        "model-capacity-floor",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let assignment = state
        .apply_registration_update(
            &running_owner,
            &calibration_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    submit_assigned_calibration(
        &state,
        Some("rk-capacity-floor"),
        "inst-owner",
        "cluster-capacity-floor",
        "model-capacity-floor",
        &assignment[0].assignment_token,
        150.0,
    )
    .await;
    state
        .apply_registration_update(
            &running_owner,
            &calibration_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let owner_runtime_update = registration_update(
        &running_owner,
        "model-capacity-floor",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 50.0,
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(
            &running_owner,
            &owner_runtime_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let peer_runtime_update = registration_update(
        &running_peer,
        "model-capacity-floor",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 40.0,
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(
            &running_peer,
            &peer_runtime_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(
        clusters[0].stats.last_mean_input_tps, 150.0,
        "a completed cluster calibration remains the floor after only smaller runtime observations are available"
    );
}

#[tokio::test]
async fn coordinated_calibration_reassigns_when_owner_disconnects_before_completion() {
    let state = StargateState::default();
    let running_a = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-reassign",
        "quic://127.0.0.1:1111",
        Some("rk-reassign"),
    );
    let running_b = running_coordinated_registration_in_cluster(
        &state,
        "inst-next",
        "cluster-reassign",
        "quic://127.0.0.1:2222",
        Some("rk-reassign"),
    );

    let update_a = registration_update(
        &running_a,
        "model-reassign",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    state.end_registration(running_a).await;

    let update_b = registration_update(
        &running_b,
        "model-reassign",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
}

#[tokio::test]
async fn coordinated_calibration_accepts_only_owner_submission() {
    let state = StargateState::default();
    let running = running_coordinated_registration_in_cluster(
        &state,
        "inst-precalibrated",
        "cluster-precalibrated",
        "quic://127.0.0.1:1111",
        Some("rk-precalibrated"),
    );
    let target = make_target(Some("rk-precalibrated"), "model-precalibrated");

    let update = registration_update(
        &running,
        "model-precalibrated",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let assignment = state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(assignment[0].state, CalibrationState::Run as i32);
    submit_assigned_calibration(
        &state,
        Some("rk-precalibrated"),
        "inst-precalibrated",
        "cluster-precalibrated",
        "model-precalibrated",
        &assignment[0].assignment_token,
        123.0,
    )
    .await;
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty(),
        "capacity submission alone must not assert backend activity or connectivity"
    );
    let directives = state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;

    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);
    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 123.0);
}

#[tokio::test]
async fn normal_registration_cannot_complete_another_pylons_assignment() {
    let state = StargateState::default();
    let running_sibling = running_coordinated_registration_in_cluster(
        &state,
        "inst-sibling",
        "cluster-fanout",
        "quic://127.0.0.1:2222",
        Some("rk-fanout"),
    );
    let running_owner = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-fanout",
        "quic://127.0.0.1:1111",
        Some("rk-fanout"),
    );
    let target = make_target(Some("rk-fanout"), "model-fanout");

    let sibling_update = registration_update(
        &running_sibling,
        "model-fanout",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_sibling,
            &sibling_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    let assignment_token = directives[0].assignment_token.clone();
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty(),
        "a RUN assignment must not trigger routing before the result RPC"
    );

    let unassigned_update = registration_update(
        &running_owner,
        "model-fanout",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_owner,
            &unassigned_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Waiting as i32);
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty(),
        "normal registration stats from a non-owner must not complete cluster calibration"
    );

    submit_assigned_calibration(
        &state,
        Some("rk-fanout"),
        "inst-sibling",
        "cluster-fanout",
        "model-fanout",
        &assignment_token,
        150.0,
    )
    .await;
    let directives = state
        .apply_registration_update(
            &running_owner,
            &unassigned_update,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);
    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 150.0);
}

#[tokio::test]
async fn runtime_last_mean_input_tps_without_complete_state_does_not_complete_calibration() {
    let state = StargateState::default();
    let running = running_coordinated_registration_in_cluster(
        &state,
        "inst-runtime-only",
        "cluster-runtime-only",
        "quic://127.0.0.1:1111",
        Some("rk-runtime-only"),
    );
    let target = make_target(Some("rk-runtime-only"), "model-runtime-only");

    let update_initial = registration_update(
        &running,
        "model-runtime-only",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running,
            &update_initial,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    let update_runtime_only = registration_update(
        &running,
        "model-runtime-only",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 999.0,
            ..ModelStats::default()
        },
    );
    let directives = state
        .apply_registration_update(
            &running,
            &update_runtime_only,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn coordinated_calibration_reassigns_when_owner_removes_model_before_completion() {
    let state = StargateState::default();
    let running_a = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-remove-model",
        "quic://127.0.0.1:1111",
        Some("rk-remove-model"),
    );
    let running_b = running_coordinated_registration_in_cluster(
        &state,
        "inst-next",
        "cluster-remove-model",
        "quic://127.0.0.1:2222",
        Some("rk-remove-model"),
    );

    let update_a = registration_update(
        &running_a,
        "model-remove",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    let update_b = registration_update(
        &running_b,
        "model-remove",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives[0].state, CalibrationState::Waiting as i32);

    let remove_model_update = empty_registration_update(&running_a);
    let directives = state
        .apply_registration_update(
            &running_a,
            &remove_model_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert!(directives.is_empty());

    let directives = state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
}

#[tokio::test]
async fn coordinated_calibration_does_not_complete_from_waiting_backend_stats() {
    let state = StargateState::default();
    let running_a = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-waiting-stats",
        "quic://127.0.0.1:1111",
        Some("rk-waiting-stats"),
    );
    let running_b = running_coordinated_registration_in_cluster(
        &state,
        "inst-waiting",
        "cluster-waiting-stats",
        "quic://127.0.0.1:2222",
        Some("rk-waiting-stats"),
    );

    let update_a = registration_update(
        &running_a,
        "model-waiting-stats",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    let update_b_active_with_capacity = registration_update(
        &running_b,
        "model-waiting-stats",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 999.0,
            ..ModelStats::default()
        },
    );
    let directives = state
        .apply_registration_update(
            &running_b,
            &update_b_active_with_capacity,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Waiting as i32);

    let remove_model_update = empty_registration_update(&running_a);
    state
        .apply_registration_update(
            &running_a,
            &remove_model_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let directives = state
        .apply_registration_update(
            &running_b,
            &update_b_active_with_capacity,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
}

#[tokio::test]
async fn completed_calibration_survives_model_removal_while_cluster_is_registered() {
    let state = StargateState::default();
    let running = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-readd-model",
        "quic://127.0.0.1:1111",
        Some("rk-readd-model"),
    );

    let update_initial = registration_update(
        &running,
        "model-readd",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running,
            &update_initial,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    let update_complete = registration_update(
        &running,
        "model-readd",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    submit_assigned_calibration(
        &state,
        Some("rk-readd-model"),
        "inst-owner",
        "cluster-readd-model",
        "model-readd",
        &directives[0].assignment_token,
        321.0,
    )
    .await;
    let directives = state
        .apply_registration_update(
            &running,
            &update_complete,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);

    let remove_model_update = empty_registration_update(&running);
    let directives = state
        .apply_registration_update(
            &running,
            &remove_model_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert!(directives.is_empty());

    let directives = state
        .apply_registration_update(
            &running,
            &update_initial,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);
}

#[tokio::test]
async fn coordinated_calibration_keeps_completed_model_while_peer_still_registered() {
    let state = StargateState::default();
    let running_a = running_coordinated_registration_in_cluster(
        &state,
        "inst-owner",
        "cluster-keep-model",
        "quic://127.0.0.1:1111",
        Some("rk-keep-model"),
    );
    let running_b = running_coordinated_registration_in_cluster(
        &state,
        "inst-peer",
        "cluster-keep-model",
        "quic://127.0.0.1:2222",
        Some("rk-keep-model"),
    );

    let update_a_initial = registration_update(
        &running_a,
        "model-keep",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_a,
            &update_a_initial,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);

    let update_a_complete = registration_update(
        &running_a,
        "model-keep",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    submit_assigned_calibration(
        &state,
        Some("rk-keep-model"),
        "inst-owner",
        "cluster-keep-model",
        "model-keep",
        &directives[0].assignment_token,
        654.0,
    )
    .await;
    let directives = state
        .apply_registration_update(
            &running_a,
            &update_a_complete,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);

    let update_b = registration_update(
        &running_b,
        "model-keep",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);

    let remove_model_update = empty_registration_update(&running_a);
    state
        .apply_registration_update(
            &running_a,
            &remove_model_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let directives = state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Complete as i32);
}

#[tokio::test]
async fn final_cluster_disconnect_drops_local_snapshot_and_completed_calibration() {
    let state = StargateState::default();
    let owner = running_coordinated_registration_in_cluster(
        &state,
        "inst-old",
        "cluster-return",
        "quic://127.0.0.1:1111",
        Some("rk-return"),
    );
    let target = make_target(Some("rk-return"), "model-return");
    let active_update = registration_update(
        &owner,
        "model-return",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let assignment = state
        .apply_registration_update(&owner, &active_update, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(assignment[0].state, CalibrationState::Run as i32);
    let first_token = assignment[0].assignment_token.clone();
    submit_assigned_calibration(
        &state,
        Some("rk-return"),
        "inst-old",
        "cluster-return",
        "model-return",
        &first_token,
        321.0,
    )
    .await;
    state
        .apply_registration_update(&owner, &active_update, true, Some(Duration::from_millis(5)))
        .await;
    assert_eq!(
        state.cluster_candidates_for_target(&target).await[0]
            .stats
            .last_mean_input_tps,
        321.0
    );

    state.end_registration(owner).await;
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty()
    );

    let replacement = running_coordinated_registration_in_cluster(
        &state,
        "inst-new",
        "cluster-return",
        "quic://127.0.0.1:2222",
        Some("rk-return"),
    );
    let replacement_update = registration_update(
        &replacement,
        "model-return",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    assert_ne!(directives[0].assignment_token, first_token);
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn completed_calibration_floor_remains_until_final_local_registration_drops() {
    let state = StargateState::default();
    let running_coordinated = running_coordinated_registration_in_cluster(
        &state,
        "inst-coordinated",
        "cluster-clear-capacity",
        "quic://127.0.0.1:1111",
        Some("rk-clear-capacity"),
    );
    let running_local = running_registration_in_cluster(
        &state,
        "inst-local",
        "cluster-clear-capacity",
        "quic://127.0.0.1:2222",
        Some("rk-clear-capacity"),
    );
    let target = make_target(Some("rk-clear-capacity"), "model-clear-capacity");

    let update_initial = registration_update(
        &running_coordinated,
        "model-clear-capacity",
        InferenceServerStatus::Inactive as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &running_coordinated,
            &update_initial,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    let first_token = directives[0].assignment_token.clone();

    let update_complete = registration_update(
        &running_coordinated,
        "model-clear-capacity",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    submit_assigned_calibration(
        &state,
        Some("rk-clear-capacity"),
        "inst-coordinated",
        "cluster-clear-capacity",
        "model-clear-capacity",
        &first_token,
        777.0,
    )
    .await;
    state
        .apply_registration_update(
            &running_coordinated,
            &update_complete,
            true,
            Some(Duration::from_millis(5)),
        )
        .await;

    let update_local = registration_update(
        &running_local,
        "model-clear-capacity",
        InferenceServerStatus::Active as i32,
        ModelStats {
            last_mean_input_tps: 100.0,
            ..ModelStats::default()
        },
    );
    state
        .apply_registration_update(
            &running_local,
            &update_local,
            true,
            Some(Duration::from_millis(6)),
        )
        .await;

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 777.0);

    let remove_model_update = empty_registration_update(&running_coordinated);
    state
        .apply_registration_update(
            &running_coordinated,
            &remove_model_update,
            true,
            Some(Duration::from_millis(7)),
        )
        .await;

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].active_backend_count, 1);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 777.0);

    state.end_registration(running_coordinated).await;
    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].stats.last_mean_input_tps, 777.0);

    state.end_registration(running_local).await;
    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty()
    );

    let replacement = running_coordinated_registration_in_cluster(
        &state,
        "inst-replacement",
        "cluster-clear-capacity",
        "quic://127.0.0.1:3333",
        Some("rk-clear-capacity"),
    );
    let replacement_update = registration_update(
        &replacement,
        "model-clear-capacity",
        InferenceServerStatus::Active as i32,
        ModelStats::default(),
    );
    let directives = state
        .apply_registration_update(
            &replacement,
            &replacement_update,
            true,
            Some(Duration::from_millis(8)),
        )
        .await;
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0].state, CalibrationState::Run as i32);
    assert_ne!(directives[0].assignment_token, first_token);
}

#[tokio::test]
async fn reservation_updates_local_snapshot_until_next_registration_update() {
    let state = StargateState::default();
    let running = running_registration(&state, "inst-res", "quic://127.0.0.1:8888", Some("rk-res"));
    let target = make_target(Some("rk-res"), "model-res");
    let update = InferenceServerRegistration {
        inference_server_id: "inst-res".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:8888".to_string(),
        models: HashMap::from([(
            "model-res".to_string(),
            InferenceServerModelRegistration {
                stats: Some(ModelStats {
                    last_mean_input_tps: 100.0,
                    max_engine_concurrency: 8,
                    queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
                    ..ModelStats::default()
                }),
                status: InferenceServerStatus::Active.into(),
            },
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    {
        let _successful_attempt_reservation = selected_cluster
            .reserve_backend(&running.generation(), 37, 4)
            .expect("active backend should accept reservation");
    }

    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(candidates[0].stats.num_running_queries, 1);
    assert_eq!(candidates[0].stats.queue_size, 1);
    assert_eq!(candidates[0].stats.total_query_input_size, 37);
    assert_eq!(candidates[0].stats.queued_input_size, 37);
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&4),
        Some(&375)
    );

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;

    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(candidates[0].stats.num_running_queries, 0);
    assert_eq!(candidates[0].stats.queue_size, 0);
    assert_eq!(candidates[0].stats.total_query_input_size, 0);
    assert_eq!(candidates[0].stats.queued_input_size, 0);
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&4),
        Some(&5)
    );

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters[0].stats.num_running_queries, 0);
    assert_eq!(clusters[0].stats.queue_size, 0);
    assert_eq!(clusters[0].stats.total_query_input_size, 0);
    assert_eq!(clusters[0].stats.queued_input_size, 0);
    assert_eq!(
        clusters[0].stats.queue_time_estimate_ms_by_priority.get(&4),
        Some(&5)
    );
}

#[tokio::test]
async fn released_reservation_restores_local_snapshot_before_registration_update() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-release",
        "quic://127.0.0.1:8888",
        Some("rk-release"),
    );
    let target = make_target(Some("rk-release"), "model-release");
    let update = InferenceServerRegistration {
        inference_server_id: "inst-release".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:8888".to_string(),
        models: HashMap::from([(
            "model-release".to_string(),
            InferenceServerModelRegistration {
                stats: Some(ModelStats {
                    last_mean_input_tps: 100.0,
                    queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
                    ..ModelStats::default()
                }),
                status: InferenceServerStatus::Active.into(),
            },
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let reservation = selected_cluster
        .reserve_backend(&running.generation(), 37, 4)
        .expect("active backend should accept reservation");

    reservation.release();

    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(candidates[0].stats.num_running_queries, 0);
    assert_eq!(candidates[0].stats.total_query_input_size, 0);
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&4),
        Some(&5)
    );

    let consumed_by_heartbeat = selected_cluster
        .reserve_backend(&running.generation(), 10, 4)
        .expect("active backend should accept reservation");
    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let still_pending = selected_cluster
        .reserve_backend(&running.generation(), 20, 4)
        .expect("active backend should accept reservation");

    consumed_by_heartbeat.release();
    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(candidates[0].stats.num_running_queries, 1);
    assert_eq!(candidates[0].stats.total_query_input_size, 20);
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&4),
        Some(&205)
    );

    still_pending.release();
    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(candidates[0].stats.num_running_queries, 0);
    assert_eq!(candidates[0].stats.total_query_input_size, 0);
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&4),
        Some(&5)
    );
}

#[tokio::test]
async fn shared_cluster_reservation_updates_cluster_snapshot_even_when_other_backend_is_latest() {
    let state = StargateState::default();
    let running_a = running_registration_in_cluster(
        &state,
        "inst-a",
        "cluster-reserved",
        "quic://127.0.0.1:1111",
        Some("rk-reserved"),
    );
    let running_b = running_registration_in_cluster(
        &state,
        "inst-b",
        "cluster-reserved",
        "quic://127.0.0.1:2222",
        Some("rk-reserved"),
    );
    let target = make_target(Some("rk-reserved"), "model-reserved");

    let update_a = InferenceServerRegistration {
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-reserved".to_string(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        models: HashMap::from([(
            "model-reserved".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    output_tps: 0.0,
                    last_mean_input_tps: 100.0,
                    max_output_tps: 50.0,
                    queue_size: 0,
                    queued_input_size: 0,
                    kv_cache_capacity_tokens: 1000,
                    kv_cache_used_tokens: 100,
                    kv_cache_free_tokens: 900,
                    num_running_queries: 3,
                    max_engine_concurrency: 8,
                    total_query_input_size: 30,
                    queue_time_estimate_ms_by_priority: HashMap::from([(4, 10)]),
                    ..ModelStats::default()
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    let update_b = InferenceServerRegistration {
        inference_server_id: "inst-b".to_string(),
        cluster_id: "cluster-reserved".to_string(),
        inference_server_url: "quic://127.0.0.1:2222".to_string(),
        models: HashMap::from([(
            "model-reserved".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    output_tps: 0.0,
                    last_mean_input_tps: 100.0,
                    max_output_tps: 60.0,
                    queue_size: 0,
                    queued_input_size: 0,
                    kv_cache_capacity_tokens: 2000,
                    kv_cache_used_tokens: 500,
                    kv_cache_free_tokens: 1500,
                    num_running_queries: 7,
                    max_engine_concurrency: 9,
                    total_query_input_size: 70,
                    queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
                    ..ModelStats::default()
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;

    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let _reservation = selected_cluster.reserve_backend(&running_a.generation(), 37, 4);

    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    let cluster = &clusters[0];
    assert_eq!(cluster.stats.queue_size, 1);
    assert_eq!(cluster.stats.queued_input_size, 37);
    assert_eq!(cluster.stats.num_running_queries, 8);
    assert_eq!(cluster.stats.total_query_input_size, 107);
    // Reservation delta uses summed backend input capacity:
    // existing 5ms + ceil(37 tokens / 200 input TPS * 1000) = 190ms.
    assert_eq!(
        cluster.stats.queue_time_estimate_ms_by_priority.get(&4),
        Some(&190)
    );
}

#[tokio::test]
async fn reservation_inserts_request_priority_and_preserves_more_urgent_bucket() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-priority-res",
        "quic://127.0.0.1:8888",
        Some("rk-priority-res"),
    );
    let target = make_target(Some("rk-priority-res"), "model-priority-res");
    let update = InferenceServerRegistration {
        inference_server_id: "inst-priority-res".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:8888".to_string(),
        models: HashMap::from([(
            "model-priority-res".to_string(),
            InferenceServerModelRegistration {
                stats: Some(ModelStats {
                    last_mean_input_tps: 100.0,
                    queue_time_estimate_ms_by_priority: HashMap::from([(2, 5)]),
                    ..ModelStats::default()
                }),
                status: InferenceServerStatus::Active.into(),
            },
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let _reservation = selected_cluster.reserve_backend(&running.generation(), 10, 3);

    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&2),
        Some(&5)
    );
    assert_eq!(
        candidates[0]
            .stats
            .queue_time_estimate_ms_by_priority
            .get(&3),
        Some(&105)
    );
}

#[tokio::test]
async fn reservation_updates_lower_urgency_cumulative_priority_buckets() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-priority-cumulative",
        "quic://127.0.0.1:8888",
        Some("rk-priority-cumulative"),
    );
    let target = make_target(Some("rk-priority-cumulative"), "model-priority-cumulative");
    let update = InferenceServerRegistration {
        inference_server_id: "inst-priority-cumulative".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:8888".to_string(),
        models: HashMap::from([(
            "model-priority-cumulative".to_string(),
            InferenceServerModelRegistration {
                stats: Some(ModelStats {
                    last_mean_input_tps: 100.0,
                    queue_time_estimate_ms_by_priority: HashMap::from([(1, 10), (4, 40)]),
                    ..ModelStats::default()
                }),
                status: InferenceServerStatus::Active.into(),
            },
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let _reservation = selected_cluster.reserve_backend(&running.generation(), 10, 2);

    let candidates = state.cluster_candidates_for_target(&target).await;
    let priority_estimates = &candidates[0].stats.queue_time_estimate_ms_by_priority;
    assert_eq!(priority_estimates.get(&1), Some(&10));
    assert_eq!(priority_estimates.get(&2), Some(&110));
    assert_eq!(priority_estimates.get(&4), Some(&140));
}

#[test]
fn reservation_updates_existing_request_priority_and_lower_urgency_buckets() {
    let mut stats = ModelStats {
        last_mean_input_tps: 100.0,
        queue_time_estimate_ms_by_priority: HashMap::from([(1, 10), (2, 100), (4, 400)]),
        ..ModelStats::default()
    };

    update_reserved_priority_queue_time(&mut stats, 10, 2);

    let priority_estimates = &stats.queue_time_estimate_ms_by_priority;
    assert_eq!(priority_estimates.get(&1), Some(&10));
    assert_eq!(priority_estimates.get(&2), Some(&200));
    assert_eq!(priority_estimates.get(&4), Some(&500));
}

#[test]
fn reservation_saturates_priority_queue_estimates() {
    let mut stats = ModelStats {
        last_mean_input_tps: 1.0,
        queue_time_estimate_ms_by_priority: HashMap::from([(0, u64::MAX - 1), (2, u64::MAX - 2)]),
        ..ModelStats::default()
    };

    update_reserved_priority_queue_time(&mut stats, 10, 0);

    let priority_estimates = &stats.queue_time_estimate_ms_by_priority;
    assert_eq!(priority_estimates.get(&0), Some(&u64::MAX));
    assert_eq!(priority_estimates.get(&2), Some(&u64::MAX));
}

#[tokio::test]
async fn reservation_clears_priority_map_when_delta_cannot_be_computed() {
    let mut stats = ModelStats {
        last_mean_input_tps: 0.0,
        queue_time_estimate_ms_by_priority: HashMap::from([(1, 10), (4, 40)]),
        ..ModelStats::default()
    };

    update_reserved_priority_queue_time(&mut stats, 10, 2);

    assert!(stats.queue_time_estimate_ms_by_priority.is_empty());
}

#[test]
fn queue_time_estimate_helper_uses_sparse_priority_and_aggregate_fallback() {
    let priority_stats = ModelStats {
        last_mean_input_tps: 100.0,
        queued_input_size: 25,
        queue_time_estimate_ms_by_priority: HashMap::from([(1, 10), (4, 40)]),
        ..ModelStats::default()
    };
    assert_eq!(
        crate::queue_estimate::queue_time_estimate_ms_for_priority(&priority_stats, 3),
        Some(10)
    );

    let aggregate_stats = ModelStats {
        last_mean_input_tps: 100.0,
        queued_input_size: 25,
        ..ModelStats::default()
    };
    assert_eq!(
        crate::queue_estimate::queue_time_estimate_ms_for_priority(&aggregate_stats, 3),
        Some(250)
    );

    let invalid_capacity_stats = ModelStats {
        last_mean_input_tps: 0.0,
        queued_input_size: 25,
        ..ModelStats::default()
    };
    assert_eq!(
        crate::queue_estimate::queue_time_estimate_ms_for_priority(&invalid_capacity_stats, 3),
        None
    );
}

#[test]
fn queue_time_estimate_helper_treats_lower_priority_only_work_as_known_zero() {
    let stats = ModelStats {
        last_mean_input_tps: 100.0,
        queued_input_size: 25,
        queue_time_estimate_ms_by_priority: HashMap::from([(4, 250)]),
        ..ModelStats::default()
    };

    assert_eq!(
        crate::queue_estimate::queue_time_estimate_ms_for_priority(&stats, 0),
        Some(0)
    );
}

#[tokio::test]
async fn reservation_inserts_high_priority_estimate_when_only_lower_priority_work_exists() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-priority-clear",
        "quic://127.0.0.1:8888",
        Some("rk-priority-clear"),
    );
    let target = make_target(Some("rk-priority-clear"), "model-priority-clear");
    let update = InferenceServerRegistration {
        inference_server_id: "inst-priority-clear".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:8888".to_string(),
        models: HashMap::from([(
            "model-priority-clear".to_string(),
            InferenceServerModelRegistration {
                stats: Some(ModelStats {
                    last_mean_input_tps: 100.0,
                    queue_time_estimate_ms_by_priority: HashMap::from([(4, 5)]),
                    ..ModelStats::default()
                }),
                status: InferenceServerStatus::Active as i32,
            },
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(5)))
        .await;
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let _reservation = selected_cluster.reserve_backend(&running.generation(), 10, 0);

    let candidates = state.cluster_candidates_for_target(&target).await;
    assert_eq!(
        candidates[0].stats.queue_time_estimate_ms_by_priority,
        HashMap::from([(0, 100), (4, 105)])
    );
}

#[tokio::test]
async fn inactive_registration_is_not_routable() {
    let state = StargateState::default();
    let running = running_registration(
        &state,
        "inst-inactive",
        "quic://127.0.0.1:9999",
        Some("rk-in"),
    );
    let update = InferenceServerRegistration {
        inference_server_id: "inst-inactive".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:9999".to_string(),
        models: HashMap::from([(
            "model-r".to_string(),
            model_registration(InferenceServerStatus::Inactive as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running, &update, true, Some(Duration::from_millis(7)))
        .await;
    assert!(
        state
            .candidates_for_target(&make_target(Some("rk-in"), "model-r"))
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn list_active_models_reports_only_routable_models() {
    let state = StargateState::default();
    let active = running_registration(
        &state,
        "inst-active",
        "quic://127.0.0.1:1111",
        Some("rk-list"),
    );
    let active_without_rtt = running_registration(
        &state,
        "inst-no-rtt",
        "quic://127.0.0.1:2222",
        Some("rk-list"),
    );
    let inactive = running_registration(
        &state,
        "inst-inactive-list",
        "quic://127.0.0.1:3333",
        Some("rk-list"),
    );

    state
        .apply_registration_update(
            &active,
            &InferenceServerRegistration {
                inference_server_id: "inst-active".to_string(),
                cluster_id: String::new(),
                inference_server_url: "quic://127.0.0.1:1111".to_string(),
                models: HashMap::from([(
                    "model-listed".to_string(),
                    model_registration(InferenceServerStatus::Active as i32),
                )]),
                reverse_tunnel: false,
                coordinated_calibration: false,
            },
            true,
            Some(Duration::from_millis(7)),
        )
        .await;
    state
        .apply_registration_update(
            &active_without_rtt,
            &InferenceServerRegistration {
                inference_server_id: "inst-no-rtt".to_string(),
                cluster_id: String::new(),
                inference_server_url: "quic://127.0.0.1:2222".to_string(),
                models: HashMap::from([(
                    "model-not-ready".to_string(),
                    model_registration(InferenceServerStatus::Active as i32),
                )]),
                reverse_tunnel: false,
                coordinated_calibration: false,
            },
            true,
            None,
        )
        .await;
    state
        .apply_registration_update(
            &inactive,
            &InferenceServerRegistration {
                inference_server_id: "inst-inactive-list".to_string(),
                cluster_id: String::new(),
                inference_server_url: "quic://127.0.0.1:3333".to_string(),
                models: HashMap::from([(
                    "model-inactive".to_string(),
                    model_registration(InferenceServerStatus::Inactive as i32),
                )]),
                reverse_tunnel: false,
                coordinated_calibration: false,
            },
            true,
            Some(Duration::from_millis(7)),
        )
        .await;

    let models = state.list_active_models(Some("rk-list"), &[]).await;
    assert_eq!(models.len(), 1);
    assert_eq!(models[0], "model-listed");

    let filtered = state
        .list_active_models(Some("rk-list"), &["model-not-ready".to_string()])
        .await;
    assert!(filtered.is_empty(), "got: {filtered:?}");
}

#[tokio::test]
async fn list_active_models_ignores_empty_target_generations() {
    let state = StargateState::default();
    let target = make_target(Some("rk-intermediate"), "model-intermediate");
    let _target_state = state.routing.target_state_or_insert(&target).await;

    assert!(
        state
            .cluster_candidates_for_target(&target)
            .await
            .is_empty(),
        "proxy routing source of truth should not consider an uninitialized generation routable"
    );

    let listed = state.list_active_models(Some("rk-intermediate"), &[]).await;
    assert!(
        listed.is_empty(),
        "ListModels must not advertise targets without routable cluster generations: {listed:?}"
    );
}

#[test]
#[should_panic(expected = "routed snapshot inference-server ID must match exact registration")]
fn routed_cluster_rejects_snapshot_identity_mismatch() {
    let mut backend = routed_backend(
        "cluster-snapshot-identity",
        "backend-snapshot-identity",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let cluster_state = RoutedClusterState::new(backend.registration.cluster_generation.clone());
    backend.inference_server_id = "different-exported-id".to_string();

    cluster_state.upsert_backend(Arc::new(backend));
}

#[test]
fn cluster_generation_publishes_matching_sorted_backend_and_aggregate_views() {
    let backend_b = routed_backend(
        "cluster-generation",
        "backend-b",
        2.0,
        20.0,
        Duration::from_millis(20),
    );
    let backend_a = routed_backend_in_cluster_generation(
        backend_b.registration.cluster_generation.clone(),
        "cluster-generation",
        "backend-a",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let cluster_state = RoutedClusterState::new(backend_b.registration.cluster_generation.clone());

    cluster_state.upsert_backend(Arc::new(backend_b.clone()));
    cluster_state.upsert_backend(Arc::new(backend_a.clone()));

    let backends = cluster_state.backend_snapshot_values();
    assert_eq!(
        backends
            .iter()
            .map(|backend| backend.inference_server_id.as_str())
            .collect::<Vec<_>>(),
        vec!["backend-a", "backend-b"]
    );
    let snapshot = cluster_state
        .routing_snapshot(None)
        .expect("published backends should have one matching aggregate");
    assert_eq!(snapshot.active_backend_count, backends.len());
    assert_eq!(snapshot.stats.output_tps, 3.0);
    assert_eq!(snapshot.stats.last_mean_input_tps, 30.0);
    assert_eq!(snapshot.rtt, Duration::from_millis(10));

    cluster_state.remove_backend(&backend_a.registration);
    let backends = cluster_state.backend_snapshot_values();
    let snapshot = cluster_state
        .routing_snapshot(None)
        .expect("remaining backend should keep the cluster routable");
    assert_eq!(backends.len(), 1);
    assert_eq!(backends[0].inference_server_id, "backend-b");
    assert_eq!(snapshot.active_backend_count, backends.len());
    assert_eq!(snapshot.stats.output_tps, 2.0);
    assert_eq!(snapshot.rtt, Duration::from_millis(20));

    cluster_state.remove_backend(&backend_b.registration);
    assert!(cluster_state.routing_snapshot(None).is_none());
    assert!(cluster_state.select_backend(&HashSet::new()).is_none());
}

#[test]
fn cluster_generation_derives_cluster_source_from_stored_backends() {
    let observed_at = Instant::now();
    let mut backend_a = routed_backend(
        "cluster-derived-source",
        "backend-a",
        1.0,
        100.0,
        Duration::from_millis(10),
    );
    backend_a.stats.max_output_tps = 10.0;
    backend_a.snapshot_updated_at = observed_at;
    let cluster_state = Arc::new(RoutedClusterState::new(
        backend_a.registration.cluster_generation.clone(),
    ));
    let backend_a_registration = backend_a.registration.clone();
    let mut backend_c = routed_backend_in_cluster_generation(
        backend_a.registration.cluster_generation.clone(),
        "cluster-derived-source",
        "backend-c",
        3.0,
        100.0,
        Duration::from_millis(30),
    );
    backend_c.stats.max_output_tps = 30.0;
    backend_c.snapshot_updated_at = observed_at + Duration::from_millis(1);
    let mut backend_b = routed_backend_in_cluster_generation(
        backend_a.registration.cluster_generation.clone(),
        "cluster-derived-source",
        "backend-b",
        2.0,
        100.0,
        Duration::from_millis(20),
    );
    backend_b.stats.max_output_tps = 20.0;
    backend_b.snapshot_updated_at = observed_at + Duration::from_millis(2);
    let backend_b_registration = backend_b.registration.clone();

    cluster_state.upsert_backend(Arc::new(backend_a));
    cluster_state.upsert_backend(Arc::new(backend_c));
    cluster_state.upsert_backend(Arc::new(backend_b.clone()));
    let _reservation = cluster_state
        .reserve_backend(&backend_a_registration, 25, 0)
        .expect("stored backend should accept reservation");

    let snapshot = cluster_state
        .routing_snapshot(None)
        .expect("latest backend should publish the cluster-scoped source");
    assert_eq!(snapshot.stats.max_output_tps, 20.0);
    assert_eq!(snapshot.stats.queue_size, 1);

    backend_b.stats.max_output_tps = 25.0;
    cluster_state.upsert_backend(Arc::new(backend_b));
    let snapshot = cluster_state
        .routing_snapshot(None)
        .expect("source heartbeat should retain another backend's reservation");
    assert_eq!(snapshot.stats.max_output_tps, 25.0);
    assert_eq!(snapshot.stats.queue_size, 1);

    cluster_state.remove_backend(&backend_b_registration);
    let snapshot = cluster_state
        .routing_snapshot(None)
        .expect("source removal should derive a surviving source");
    assert_eq!(snapshot.active_backend_count, 2);
    assert_eq!(snapshot.stats.max_output_tps, 30.0);
    assert_eq!(snapshot.stats.queue_size, 1);
}

#[test]
fn replaced_backend_registration_cannot_reserve_same_id_successor() {
    let old = routed_backend(
        "cluster-exact-registration",
        "backend-same-id",
        1.0,
        100.0,
        Duration::from_millis(10),
    );
    let cluster_generation = old.registration.cluster_generation.clone();
    let stale_registration = old.registration.clone();
    let cluster_state = Arc::new(RoutedClusterState::new(cluster_generation.clone()));
    assert_eq!(
        cluster_state.upsert_backend(Arc::new(old)),
        ClusterBackendUpsert::Inserted
    );

    let replacement = routed_backend_in_cluster_generation(
        cluster_generation,
        "cluster-exact-registration",
        "backend-same-id",
        2.0,
        100.0,
        Duration::from_millis(5),
    );
    let replacement_registration = replacement.registration.clone();
    assert_eq!(
        cluster_state.upsert_backend(Arc::new(replacement)),
        ClusterBackendUpsert::Replaced
    );

    assert!(
        cluster_state
            .reserve_backend(&stale_registration, 25, 0)
            .is_none(),
        "a replaced registration must not reserve its same-ID successor"
    );
    assert!(
        cluster_state
            .reserve_backend(&replacement_registration, 25, 0)
            .is_some(),
        "the exact stored replacement should accept a reservation"
    );
}

#[test]
fn reservation_release_only_cancels_exact_pending_reservation() {
    let old_backend = routed_backend(
        "cluster-exact-release",
        "backend-same-id",
        1.0,
        100.0,
        Duration::from_millis(10),
    );
    let old_registration = old_backend.registration.clone();
    let old_cluster = Arc::new(RoutedClusterState::new(
        old_registration.cluster_generation.clone(),
    ));
    old_cluster.upsert_backend(Arc::new(old_backend));
    let old_reservation = old_cluster
        .reserve_backend(&old_registration, 10, 0)
        .expect("old cluster should accept reservation");

    let replacement_backend = routed_backend(
        "cluster-exact-release",
        "backend-same-id",
        2.0,
        100.0,
        Duration::from_millis(5),
    );
    let replacement_registration = replacement_backend.registration.clone();
    let replacement_cluster = Arc::new(RoutedClusterState::new(
        replacement_registration.cluster_generation.clone(),
    ));
    replacement_cluster.upsert_backend(Arc::new(replacement_backend));
    let _replacement_reservation = replacement_cluster
        .reserve_backend(&replacement_registration, 20, 0)
        .expect("replacement cluster should accept its independent reservation");

    old_reservation.release();

    assert_eq!(
        old_cluster
            .routing_snapshot(None)
            .expect("old cluster should retain its backend")
            .stats
            .queue_size,
        0
    );
    assert_eq!(
        replacement_cluster
            .routing_snapshot(None)
            .expect("replacement cluster should retain its backend")
            .stats
            .queue_size,
        1,
        "releasing the old token must not cancel the replacement reservation"
    );
}

#[test]
fn reservation_release_does_not_wait_for_cluster_generation_lock() {
    let backend = routed_backend(
        "cluster-lock-independent-release",
        "backend-lock-independent-release",
        1.0,
        100.0,
        Duration::from_millis(10),
    );
    let registration = backend.registration.clone();
    let cluster = Arc::new(RoutedClusterState::new(
        registration.cluster_generation.clone(),
    ));
    cluster.upsert_backend(Arc::new(backend));
    let reservation = cluster
        .reserve_backend(&registration, 10, 0)
        .expect("active backend should accept reservation");

    let (release_finished, release_thread, _finished_receiver) = {
        let _generation = cluster.generation.lock();
        let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(1);
        let (finished_sender, finished_receiver) = std::sync::mpsc::sync_channel(1);
        let release_thread = std::thread::spawn(move || {
            started_sender
                .send(())
                .expect("release start receiver should remain alive");
            reservation.release();
            finished_sender
                .send(())
                .expect("release completion receiver should remain alive");
        });
        started_receiver
            .recv()
            .expect("reservation release thread should start");
        let release_finished = finished_receiver
            .recv_timeout(Duration::from_secs(1))
            .is_ok();
        (release_finished, release_thread, finished_receiver)
    };

    release_thread
        .join()
        .expect("reservation release thread should finish");
    assert!(
        release_finished,
        "reservation release should not wait for the cluster-generation lock"
    );
    assert_eq!(
        cluster
            .routing_snapshot(None)
            .expect("cluster should retain its backend")
            .stats
            .queue_size,
        0
    );
}

#[test]
fn cluster_generation_applies_current_calibration_without_storing_it() {
    let backend = routed_backend(
        "cluster-calibration-generation",
        "backend-a",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let cluster_state = RoutedClusterState::new(backend.registration.cluster_generation.clone());
    cluster_state.upsert_backend(Arc::new(backend.clone()));

    assert_eq!(
        cluster_state
            .routing_snapshot(Some(100.0))
            .expect("cluster should be routable")
            .stats
            .last_mean_input_tps,
        100.0
    );
    assert_eq!(
        cluster_state
            .routing_snapshot(Some(5.0))
            .expect("cluster should be routable")
            .stats
            .last_mean_input_tps,
        10.0
    );
    assert_eq!(
        cluster_state
            .routing_snapshot(None)
            .expect("cluster should be routable")
            .stats
            .last_mean_input_tps,
        10.0
    );
}

#[test]
fn cluster_generation_round_robin_uses_stored_order_and_filtered_sequence() {
    let backend_b = routed_backend(
        "cluster-round-robin-generation",
        "backend-b",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let cluster_generation = backend_b.registration.cluster_generation.clone();
    let cluster_state = RoutedClusterState::new(cluster_generation.clone());
    cluster_state.upsert_backend(Arc::new(backend_b));
    for inference_server_id in ["backend-a", "backend-c"] {
        let backend = routed_backend_in_cluster_generation(
            cluster_generation.clone(),
            "cluster-round-robin-generation",
            inference_server_id,
            1.0,
            10.0,
            Duration::from_millis(10),
        );
        cluster_state.upsert_backend(Arc::new(backend.clone()));
    }

    let mut selected = Vec::new();
    for _ in 0..3 {
        selected.push(
            cluster_state
                .select_backend(&HashSet::new())
                .expect("stored backend should be selected")
                .inference_server_id
                .clone(),
        );
    }
    assert_eq!(selected, vec!["backend-a", "backend-b", "backend-c"]);

    let failed = HashSet::from(["backend-a".to_string()]);
    assert_eq!(
        cluster_state
            .select_backend(&failed)
            .expect("filtered backend should be selected")
            .inference_server_id,
        "backend-c"
    );
    assert_eq!(
        cluster_state
            .select_backend(&failed)
            .expect("filtered sequence should advance")
            .inference_server_id,
        "backend-b"
    );
}

#[test]
fn backend_selection_shares_published_snapshot_until_republication() {
    let backend = routed_backend(
        "cluster-shared-backend-snapshot",
        "backend-shared-snapshot",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let registration = backend.registration.clone();
    let cluster_state = RoutedClusterState::new(registration.cluster_generation.clone());
    let publication = Arc::new(backend);
    cluster_state.upsert_backend(publication.clone());

    let first = cluster_state
        .select_backend(&HashSet::new())
        .expect("published backend should be selected");
    let second = cluster_state
        .select_backend(&HashSet::new())
        .expect("same published backend should be selected again");
    assert!(
        Arc::ptr_eq(&publication, &first),
        "cluster storage should retain the exact immutable publication"
    );
    assert!(
        Arc::ptr_eq(&first, &second),
        "repeated selection should share the immutable published snapshot"
    );

    let republished_publication = Arc::new(RoutedInferenceServerSnapshot::new(
        registration,
        ModelStats {
            output_tps: 99.0,
            last_mean_input_tps: 10.0,
            ..ModelStats::default()
        },
        Duration::from_millis(10),
        Instant::now(),
        InferenceServerStatus::Active,
    ));
    cluster_state.upsert_backend(republished_publication.clone());
    let republished = cluster_state
        .select_backend(&HashSet::new())
        .expect("republished backend should be selected");

    assert!(Arc::ptr_eq(&republished_publication, &republished));
    assert!(
        !Arc::ptr_eq(&first, &republished),
        "heartbeat republication should replace the immutable published snapshot"
    );
    assert_eq!(first.stats.output_tps, 1.0);
    assert_eq!(republished.stats.output_tps, 99.0);
}

#[test]
fn cluster_backend_aggregate_dedupes_sources_and_uses_fastest_rtt() {
    let observed_at = Instant::now();

    let backend_a = RoutedInferenceServerSnapshot {
        registration: test_registration_generation(RegistrationIdentity {
            inference_server_id: "backend-a".to_string(),
            cluster_id: "cluster-aggregate".to_string(),
            inference_server_url: "quic://127.0.0.1:1111".to_string(),
            routing_key: None,
            reverse_tunnel: false,
            coordinated_calibration: false,
        }),
        cluster_id: "cluster-aggregate".to_string(),
        inference_server_id: "backend-a".to_string(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        stats: ModelStats {
            output_tps: 1.5,
            last_mean_input_tps: 10.0,
            queue_size: 2,
            queued_input_size: 20,
            input_processing_queries: 1,
            output_generation_queries: 3,
            stats_observed_at_unix_ms: 10,
            stats_capabilities: vec!["cap-a".to_string(), "shared".to_string()],
            stats_sources: vec!["source-a".to_string(), "shared".to_string()],
            ..ModelStats::default()
        },
        rtt: Duration::from_millis(20),
        snapshot_updated_at: observed_at,
        status: InferenceServerStatus::Active,
        reverse_tunnel: false,
    };
    let cluster_state = RoutedClusterState::new(backend_a.registration.cluster_generation.clone());
    cluster_state.upsert_backend(Arc::new(backend_a.clone()));
    let backend_b = routed_backend_snapshot_in_cluster_generation(
        RoutedInferenceServerSnapshot {
            registration: test_registration_generation(RegistrationIdentity {
                inference_server_id: "backend-b".to_string(),
                cluster_id: "cluster-aggregate".to_string(),
                inference_server_url: "quic://127.0.0.1:2222".to_string(),
                routing_key: None,
                reverse_tunnel: false,
                coordinated_calibration: false,
            }),
            cluster_id: "cluster-aggregate".to_string(),
            inference_server_id: "backend-b".to_string(),
            inference_server_url: "quic://127.0.0.1:2222".to_string(),
            stats: ModelStats {
                output_tps: 2.5,
                last_mean_input_tps: 30.0,
                queue_size: 4,
                queued_input_size: 40,
                input_processing_queries: 5,
                output_generation_queries: 7,
                stats_observed_at_unix_ms: 25,
                stats_capabilities: vec!["shared".to_string(), "cap-b".to_string()],
                stats_sources: vec!["shared".to_string(), "source-b".to_string()],
                ..ModelStats::default()
            },
            rtt: Duration::from_millis(5),
            snapshot_updated_at: observed_at,
            status: InferenceServerStatus::Active,
            reverse_tunnel: false,
        },
        backend_a.registration.cluster_generation.clone(),
    );
    cluster_state.upsert_backend(Arc::new(backend_b.clone()));

    let (stats, rtt, active_backend_count) = cluster_state
        .backend_aggregate()
        .expect("two live backends should aggregate");

    assert_eq!(active_backend_count, 2);
    assert_eq!(rtt, Duration::from_millis(5));
    assert_eq!(stats.output_tps, 4.0);
    assert_eq!(stats.last_mean_input_tps, 40.0);
    assert_eq!(stats.queue_size, 6);
    assert_eq!(stats.queued_input_size, 60);
    assert_eq!(stats.input_processing_queries, 6);
    assert_eq!(stats.output_generation_queries, 10);
    assert_eq!(stats.stats_observed_at_unix_ms, 25);
    let mut capabilities = stats.stats_capabilities.clone();
    capabilities.sort();
    assert_eq!(capabilities, vec!["cap-a", "cap-b", "shared"]);
    assert_eq!(stats.stats_capabilities.len(), 3);

    let mut sources = stats.stats_sources.clone();
    sources.sort();
    assert_eq!(sources, vec!["shared", "source-a", "source-b"]);
    assert_eq!(stats.stats_sources.len(), 3);
}

#[tokio::test]
async fn list_active_models_filters_by_routing_key() {
    let state = StargateState::default();
    let running_a = running_registration(
        &state,
        "inst-list-rk-a",
        "quic://127.0.0.1:1111",
        Some("rk-a"),
    );
    let running_b = running_registration(
        &state,
        "inst-list-rk-b",
        "quic://127.0.0.1:2222",
        Some("rk-b"),
    );

    let update_a = InferenceServerRegistration {
        inference_server_id: "inst-list-rk-a".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        models: HashMap::from([
            (
                "z-list-model".to_string(),
                model_registration(InferenceServerStatus::Active as i32),
            ),
            (
                "shared-list-model".to_string(),
                model_registration(InferenceServerStatus::Active as i32),
            ),
            (
                "a-list-model".to_string(),
                model_registration(InferenceServerStatus::Active as i32),
            ),
        ]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    let update_b = InferenceServerRegistration {
        inference_server_id: "inst-list-rk-b".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:2222".to_string(),
        models: HashMap::from([(
            "shared-list-model".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;
    let unscoped = state.list_active_models(None, &[]).await;
    assert!(
        unscoped.is_empty(),
        "unscoped ListModels must not include keyed registrations: {unscoped:?}"
    );

    let models_a = state.list_active_models(Some("rk-a"), &[]).await;
    assert_eq!(
        models_a,
        vec!["a-list-model", "shared-list-model", "z-list-model"]
    );

    let models_b = state.list_active_models(Some("rk-b"), &[]).await;
    assert_eq!(models_b, vec!["shared-list-model"]);

    let wrong_key = state.list_active_models(Some("rk-c"), &[]).await;
    assert!(
        wrong_key.is_empty(),
        "ListModels must not leak models across routing keys: {wrong_key:?}"
    );

    let filtered = state
        .list_active_models(
            Some("rk-a"),
            &[
                "z-list-model".to_string(),
                "shared-list-model".to_string(),
                "z-list-model".to_string(),
            ],
        )
        .await;
    assert_eq!(filtered, vec!["shared-list-model", "z-list-model"]);
}

#[tokio::test]
async fn list_active_models_reads_authoritative_routing_generations() {
    let state = StargateState::default();
    let target = make_target(None, "model-list-authoritative");
    let running = running_registration_in_cluster(
        &state,
        "backend-list-authoritative",
        "cluster-list-authoritative",
        "quic://127.0.0.1:2222",
        None,
    );
    let registration = running.generation();
    let snapshot = RoutedInferenceServerSnapshot::new(
        registration.clone(),
        ModelStats::default(),
        Duration::from_millis(5),
        Instant::now(),
        InferenceServerStatus::Active,
    );

    state
        .routing
        .upsert_inference_server_target(&target, snapshot)
        .await;

    let candidate = state
        .candidates_for_target(&target)
        .await
        .into_iter()
        .next()
        .expect("directly published snapshot should be routable");
    assert!(Arc::ptr_eq(&candidate.registration, &registration));
    assert_eq!(candidate.cluster_id, registration.identity.cluster_id);
    assert_eq!(
        candidate.inference_server_id,
        registration.identity.inference_server_id
    );
    assert_eq!(
        candidate.inference_server_url,
        registration.identity.inference_server_url
    );
    assert_eq!(
        candidate.reverse_tunnel,
        registration.identity.reverse_tunnel
    );

    let listed = state.list_active_models(None, &[]).await;
    assert_eq!(listed.len(), 1, "got: {listed:?}");
    assert_eq!(listed[0], "model-list-authoritative");

    state
        .routing
        .remove_inference_server_from_target(&registration, &target)
        .await;
    let after_removal = state.list_active_models(None, &[]).await;
    assert!(
        after_removal.is_empty(),
        "removed model should disappear from authoritative discovery immediately: {after_removal:?}"
    );
    state.end_registration(running).await;
}

#[tokio::test]
async fn different_routing_keys_isolate_candidates() {
    let state = StargateState::default();
    let running_a = running_registration(&state, "inst-a", "quic://127.0.0.1:1111", Some("rk-a"));
    let running_b = running_registration(&state, "inst-b", "quic://127.0.0.1:2222", Some("rk-b"));

    let update_a = InferenceServerRegistration {
        inference_server_id: "inst-a".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    let update_b = InferenceServerRegistration {
        inference_server_id: "inst-b".to_string(),
        cluster_id: String::new(),
        inference_server_url: "quic://127.0.0.1:2222".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration(InferenceServerStatus::Active as i32),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(5)))
        .await;
    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;

    let candidates_a = state
        .candidates_for_target(&make_target(Some("rk-a"), "shared-model"))
        .await;
    let candidates_b = state
        .candidates_for_target(&make_target(Some("rk-b"), "shared-model"))
        .await;
    assert_eq!(candidates_a.len(), 1);
    assert_eq!(candidates_b.len(), 1);
    assert_eq!(candidates_a[0].inference_server_id, "inst-a");
    assert_eq!(candidates_b[0].inference_server_id, "inst-b");
}

#[test]
fn routing_target_membership_follows_selected_snapshot_registration() {
    let target_state = RoutingTargetState::default();
    let selected_snapshot = routed_backend(
        "cluster-selected",
        "backend-selected",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    target_state
        .upsert_backend(Arc::new(selected_snapshot.clone()))
        .expect("active routing target should accept backend");

    let selected_cluster = {
        let generation = target_state.generation.lock();
        let RoutingTargetGeneration::Active { clusters, .. } = &*generation else {
            panic!("new routing target should remain active");
        };
        assert_eq!(clusters.len(), 1);
        clusters
            .get(&selected_snapshot.registration.identity.cluster_id)
            .cloned()
            .expect(
                "target membership must follow the registration exposed to selected backend work",
            )
    };
    let stored = selected_cluster
        .select_backend(&HashSet::new())
        .expect("stored backend should be selectable");
    assert!(Arc::ptr_eq(
        &stored.registration,
        &selected_snapshot.registration
    ));
}

#[test]
fn routing_target_generation_tracks_backend_membership_without_heartbeat_double_count() {
    let target_state = RoutingTargetState::default();
    let backend_a = routed_backend(
        "cluster-a",
        "backend-a",
        1.0,
        10.0,
        Duration::from_millis(10),
    );
    let backend_b = routed_backend(
        "cluster-b",
        "backend-b",
        2.0,
        20.0,
        Duration::from_millis(20),
    );
    let backend_a_registration = backend_a.registration.clone();
    let backend_b_registration = backend_b.registration.clone();

    assert_eq!(target_state.active_backend_count(), 0);
    target_state
        .upsert_backend(Arc::new(backend_a.clone()))
        .unwrap();
    assert_eq!(target_state.active_backend_count(), 1);

    let mut heartbeat = backend_a;
    heartbeat.stats.output_tps = 3.0;
    target_state.upsert_backend(Arc::new(heartbeat)).unwrap();
    assert_eq!(
        target_state.active_backend_count(),
        1,
        "replacing one backend heartbeat must not duplicate membership"
    );

    target_state.upsert_backend(Arc::new(backend_b)).unwrap();
    assert_eq!(target_state.active_backend_count(), 2);

    target_state.remove_backend(&backend_a_registration);
    assert_eq!(target_state.active_backend_count(), 1);
    target_state.remove_backend(&backend_a_registration);
    assert_eq!(
        target_state.active_backend_count(),
        1,
        "removing absent membership must not decrement the target summary"
    );

    target_state.remove_backend(&backend_b_registration);
    assert_eq!(target_state.active_backend_count(), 0);
}

#[tokio::test]
async fn routing_target_generation_recreates_reachable_state_after_retirement() {
    let state = StargateState::default();
    let target = make_target(Some("rk-retired"), "model-retired");
    let retired = state.routing.target_state_or_insert(&target).await;

    assert!(
        state
            .routing
            .targets
            .remove_if_empty(&target, retired.clone())
            .await
    );
    let rejected_publication = Arc::new(routed_backend(
        "cluster-retired",
        "backend-retired",
        1.0,
        10.0,
        Duration::from_millis(10),
    ));
    let rejected = retired
        .upsert_backend(rejected_publication.clone())
        .expect_err("a retained stale owner must not accept unreachable backend state");
    assert!(
        Arc::ptr_eq(&rejected, &rejected_publication),
        "retired-target retry should retain the exact immutable publication"
    );

    let running = running_registration_in_cluster(
        &state,
        "backend-current",
        "cluster-current",
        "quic://backend-current",
        Some("rk-retired"),
    );
    state
        .routing
        .upsert_inference_server_target(
            &target,
            RoutedInferenceServerSnapshot::new(
                running.generation(),
                ModelStats::default(),
                Duration::from_millis(5),
                Instant::now(),
                InferenceServerStatus::Active,
            ),
        )
        .await;

    let current = state
        .routing
        .target_state(&target)
        .await
        .expect("normal upsert should publish a replacement target generation");
    assert!(!Arc::ptr_eq(&retired, &current));
    assert_eq!(state.candidates_for_target(&target).await.len(), 1);
    state.end_registration(running).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routing_target_recreation_starts_fresh_load_balancer_state() {
    let state = StargateState::default();
    let target = make_target(Some("rk-lifecycle"), "model-lifecycle");
    let router = LoadBalancerRouter::from_config(&LoadBalancerConfig {
        default: LoadBalancerAlgorithm::RoundRobin,
        request_algorithms: HashMap::new(),
        models: HashMap::new(),
    })
    .expect("round-robin router should initialize");
    let request = LoadBalancerRequest {
        routing_target: &target,
        cache_affinity_key: None,
        input_tokens: None,
        priority: 0,
        received_at: Instant::now(),
        request_slo: None,
        excluded_cluster_ids: None,
    };
    let candidates = [
        RoutedClusterSnapshot {
            cluster_id: "cluster-0".to_string(),
            stats: ModelStats::default(),
            rtt: Duration::from_millis(1),
            snapshot_updated_at: Instant::now(),
            status: InferenceServerStatus::Active,
            active_backend_count: 1,
        },
        RoutedClusterSnapshot {
            cluster_id: "cluster-1".to_string(),
            stats: ModelStats::default(),
            rtt: Duration::from_millis(1),
            snapshot_updated_at: Instant::now(),
            status: InferenceServerStatus::Active,
            active_backend_count: 1,
        },
    ];
    let first_generation = state.routing.target_state_or_insert(&target).await;

    let first = router
        .choose_candidate(&first_generation.load_balancers, &request, &candidates)
        .expect("first generation should select a candidate");
    let second = router
        .choose_candidate(&first_generation.load_balancers, &request, &candidates)
        .expect("first generation should advance its sequence");
    state
        .routing
        .targets
        .remove_if_empty(&target, first_generation.clone())
        .await;
    let replacement = state.routing.target_state_or_insert(&target).await;
    let replacement_first = router
        .choose_candidate(&replacement.load_balancers, &request, &candidates)
        .expect("replacement generation should select a candidate");

    assert_eq!(first.candidate_index, 0);
    assert_eq!(second.candidate_index, 1);
    assert!(!Arc::ptr_eq(&first_generation, &replacement));
    assert_eq!(replacement_first.candidate_index, 0);
}

#[tokio::test]
async fn routing_target_snapshot_retains_removed_generation_until_selection_finishes() {
    let state = StargateState::default();
    let target = make_target(Some("rk-snapshot"), "model-snapshot");
    let target_state = state.routing.target_state_or_insert(&target).await;
    let weak_target_state = Arc::downgrade(&target_state);
    let snapshot = state
        .routing_target_snapshot(&target)
        .await
        .expect("existing target should produce a routed snapshot");

    state
        .routing
        .targets
        .remove_if_empty(&target, target_state.clone())
        .await;
    assert!(state.routing.target_state(&target).await.is_none());
    // Release the test's direct owner so only the in-flight snapshot keeps the old generation alive.
    drop(target_state);
    assert!(weak_target_state.upgrade().is_some());

    // Finishing selection releases the final owner of the removed target generation.
    drop(snapshot);
    assert!(weak_target_state.upgrade().is_none());
}

#[tokio::test]
async fn shared_cluster_registration_exposes_one_aggregated_cluster_candidate() {
    let state = StargateState::default();
    let running_a = running_registration_in_cluster(
        &state,
        "inst-a",
        "cluster-shared",
        "quic://127.0.0.1:1111",
        Some("rk-a"),
    );
    let running_b = running_registration_in_cluster(
        &state,
        "inst-b",
        "cluster-shared",
        "quic://127.0.0.1:2222",
        Some("rk-a"),
    );

    let update_a = InferenceServerRegistration {
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-shared".to_string(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    output_tps: 2.0,
                    last_mean_input_tps: 100.0,
                    max_output_tps: 50.0,
                    queue_size: 1,
                    queued_input_size: 100,
                    input_processing_queries: 1,
                    output_generation_queries: 2,
                    stats_observed_at_unix_ms: 1000,
                    stats_capabilities: vec!["request.output.chunk_usage".to_string()],
                    stats_sources: vec!["chunk_usage".to_string()],
                    kv_cache_capacity_tokens: 1000,
                    kv_cache_used_tokens: 100,
                    kv_cache_free_tokens: 900,
                    num_running_queries: 11,
                    max_engine_concurrency: 111,
                    total_query_input_size: 1111,
                    queue_time_estimate_ms_by_priority: HashMap::from([(1, 111)]),
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    let update_b = InferenceServerRegistration {
        inference_server_id: "inst-b".to_string(),
        cluster_id: "cluster-shared".to_string(),
        inference_server_url: "quic://127.0.0.1:2222".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    output_tps: 5.0,
                    last_mean_input_tps: 120.0,
                    max_output_tps: 60.0,
                    queue_size: 2,
                    queued_input_size: 200,
                    input_processing_queries: 3,
                    output_generation_queries: 4,
                    stats_observed_at_unix_ms: 2000,
                    stats_capabilities: vec![
                        "request.output.chunk_usage".to_string(),
                        "machine.kv_cache.http".to_string(),
                    ],
                    stats_sources: vec!["chunk_usage".to_string(), "kv_cache_stats".to_string()],
                    kv_cache_capacity_tokens: 2000,
                    kv_cache_used_tokens: 500,
                    kv_cache_free_tokens: 1500,
                    num_running_queries: 7,
                    max_engine_concurrency: 77,
                    total_query_input_size: 777,
                    queue_time_estimate_ms_by_priority: HashMap::from([(1, 222), (2, 333)]),
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(10)))
        .await;
    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;

    let target = make_target(Some("rk-a"), "shared-model");
    let backend_candidates = state.candidates_for_target(&target).await;
    assert_eq!(backend_candidates.len(), 2);

    let clusters = state.cluster_candidates_for_target(&target).await;
    assert_eq!(clusters.len(), 1);
    let cluster = &clusters[0];
    assert_eq!(cluster.cluster_id, "cluster-shared");
    assert_eq!(cluster.active_backend_count, 2);
    assert_eq!(cluster.stats.last_mean_input_tps, 220.0);
    assert_eq!(cluster.stats.output_tps, 7.0);
    assert_eq!(cluster.stats.queue_size, 3);
    assert_eq!(cluster.stats.queued_input_size, 300);
    assert_eq!(cluster.stats.input_processing_queries, 4);
    assert_eq!(cluster.stats.output_generation_queries, 6);
    assert_eq!(cluster.stats.stats_observed_at_unix_ms, 2000);
    assert_eq!(
        cluster.stats.stats_capabilities,
        vec![
            "request.output.chunk_usage".to_string(),
            "machine.kv_cache.http".to_string(),
        ]
    );
    assert_eq!(
        cluster.stats.stats_sources,
        vec!["chunk_usage".to_string(), "kv_cache_stats".to_string()]
    );
    assert_eq!(cluster.stats.max_output_tps, 60.0);
    assert_eq!(cluster.stats.kv_cache_capacity_tokens, 2000);
    assert_eq!(cluster.stats.kv_cache_used_tokens, 500);
    assert_eq!(cluster.stats.kv_cache_free_tokens, 1500);
    assert_eq!(cluster.stats.num_running_queries, 7);
    assert_eq!(cluster.stats.max_engine_concurrency, 77);
    assert_eq!(cluster.stats.total_query_input_size, 777);
    assert_eq!(
        cluster.stats.queue_time_estimate_ms_by_priority,
        HashMap::from([(1, 222), (2, 333)])
    );
    assert_eq!(cluster.rtt, Duration::from_millis(5));
}

#[tokio::test]
async fn shared_cluster_recomputes_cluster_stats_when_source_backend_is_removed() {
    let state = StargateState::default();
    let running_a = running_registration_in_cluster(
        &state,
        "inst-a",
        "cluster-shared",
        "quic://127.0.0.1:1111",
        Some("rk-a"),
    );
    let running_b = running_registration_in_cluster(
        &state,
        "inst-b",
        "cluster-shared",
        "quic://127.0.0.1:2222",
        Some("rk-a"),
    );

    let update_a = InferenceServerRegistration {
        inference_server_id: "inst-a".to_string(),
        cluster_id: "cluster-shared".to_string(),
        inference_server_url: "quic://127.0.0.1:1111".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    last_mean_input_tps: 100.0,
                    max_output_tps: 50.0,
                    kv_cache_capacity_tokens: 1000,
                    kv_cache_used_tokens: 100,
                    kv_cache_free_tokens: 900,
                    num_running_queries: 11,
                    max_engine_concurrency: 111,
                    total_query_input_size: 1111,
                    queue_time_estimate_ms_by_priority: HashMap::from([(1, 111)]),
                    ..ModelStats::default()
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };
    let update_b = InferenceServerRegistration {
        inference_server_id: "inst-b".to_string(),
        cluster_id: "cluster-shared".to_string(),
        inference_server_url: "quic://127.0.0.1:2222".to_string(),
        models: HashMap::from([(
            "shared-model".to_string(),
            model_registration_with_stats(
                InferenceServerStatus::Active as i32,
                ModelStats {
                    last_mean_input_tps: 120.0,
                    max_output_tps: 60.0,
                    kv_cache_capacity_tokens: 2000,
                    kv_cache_used_tokens: 500,
                    kv_cache_free_tokens: 1500,
                    num_running_queries: 7,
                    max_engine_concurrency: 77,
                    total_query_input_size: 777,
                    queue_time_estimate_ms_by_priority: HashMap::from([(1, 222), (2, 333)]),
                    ..ModelStats::default()
                },
            ),
        )]),
        reverse_tunnel: false,
        coordinated_calibration: false,
    };

    state
        .apply_registration_update(&running_a, &update_a, true, Some(Duration::from_millis(10)))
        .await;
    state
        .apply_registration_update(&running_b, &update_b, true, Some(Duration::from_millis(5)))
        .await;

    state.end_registration(running_b).await;

    let clusters = state
        .cluster_candidates_for_target(&make_target(Some("rk-a"), "shared-model"))
        .await;
    assert_eq!(clusters.len(), 1);
    let cluster = &clusters[0];
    assert_eq!(cluster.active_backend_count, 1);
    assert_eq!(cluster.stats.last_mean_input_tps, 100.0);
    assert_eq!(cluster.stats.max_output_tps, 50.0);
    assert_eq!(cluster.stats.kv_cache_capacity_tokens, 1000);
    assert_eq!(cluster.stats.kv_cache_used_tokens, 100);
    assert_eq!(cluster.stats.kv_cache_free_tokens, 900);
    assert_eq!(cluster.stats.num_running_queries, 11);
    assert_eq!(cluster.stats.max_engine_concurrency, 111);
    assert_eq!(cluster.stats.total_query_input_size, 1111);
    assert_eq!(
        cluster.stats.queue_time_estimate_ms_by_priority,
        HashMap::from([(1, 111)])
    );
}

#[tokio::test]
async fn shared_cluster_selects_active_backends_round_robin() {
    let state = StargateState::default();
    let running_a = running_registration_in_cluster(
        &state,
        "inst-a",
        "cluster-shared",
        "quic://127.0.0.1:1111",
        Some("rk-a"),
    );
    let running_b = running_registration_in_cluster(
        &state,
        "inst-b",
        "cluster-shared",
        "quic://127.0.0.1:2222",
        Some("rk-a"),
    );
    for (running, inst, url) in [
        (&running_a, "inst-a", "quic://127.0.0.1:1111"),
        (&running_b, "inst-b", "quic://127.0.0.1:2222"),
    ] {
        let update = InferenceServerRegistration {
            inference_server_id: inst.to_string(),
            cluster_id: "cluster-shared".to_string(),
            inference_server_url: url.to_string(),
            models: HashMap::from([(
                "shared-model".to_string(),
                model_registration(InferenceServerStatus::Active as i32),
            )]),
            reverse_tunnel: false,
            coordinated_calibration: false,
        };
        state
            .apply_registration_update(running, &update, true, Some(Duration::from_millis(5)))
            .await;
    }

    let target = make_target(Some("rk-a"), "shared-model");
    let selected_cluster = selected_cluster_for_target(&state, &target).await;
    let first = selected_cluster
        .select_backend(&HashSet::new())
        .expect("first backend should be selected");
    let second = selected_cluster
        .select_backend(&HashSet::new())
        .expect("second backend should be selected");
    let third = selected_cluster
        .select_backend(&HashSet::new())
        .expect("third backend should be selected");

    assert_eq!(first.inference_server_id, "inst-a");
    assert_eq!(second.inference_server_id, "inst-b");
    assert_eq!(third.inference_server_id, "inst-a");

    let selected = selected_cluster
        .select_backend(&HashSet::from(["inst-a".to_string()]))
        .expect("remaining backend should be selected");
    assert_eq!(selected.inference_server_id, "inst-b");
}
