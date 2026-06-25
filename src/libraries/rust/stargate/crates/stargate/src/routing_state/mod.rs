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

use scc::HashMap as SccHashMap;
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};
use tonic::Status;
use tracing::warn;

use crate::metrics::StargateMetrics;
use stargate_proto::pb::{
    CalibrationState, InferenceServerRegistration, InferenceServerStatus,
    ModelCalibrationDirective, ModelStats, SubmitClusterCalibrationRequest,
};

mod calibration;
mod cluster_snapshots;
mod clusters;
mod keys;
mod registration;
mod reservations;
mod snapshots;

pub use keys::RoutingTargetKey;
pub use snapshots::{RoutedClusterSnapshot, RoutedInferenceServerSnapshot};
pub(crate) use snapshots::{RoutingTargetSnapshot, SelectedRoutedCluster};

pub(crate) use keys::RegistrationIdentity;
#[cfg(test)]
pub(crate) use registration::test_registration_generation;
pub(crate) use registration::{RegistrationGeneration, RunningRegistration};
pub(crate) use reservations::RoutingReservation;

#[cfg(test)]
use snapshots::RoutingTargetState;

#[derive(Debug)]
pub struct StargateState {
    registrations: registration::RegistrationRegistry,
    routing: clusters::RoutingLifecycle,
}

impl Default for StargateState {
    fn default() -> Self {
        Self::new()
    }
}

impl StargateState {
    pub fn new() -> Self {
        Self::new_inner(None)
    }

    pub fn new_with_metrics(metrics: Arc<StargateMetrics>) -> Self {
        Self::new_inner(Some(metrics))
    }

    fn new_inner(metrics: Option<Arc<StargateMetrics>>) -> Self {
        Self {
            registrations: registration::RegistrationRegistry::default(),
            routing: clusters::RoutingLifecycle::new(metrics),
        }
    }

    pub(crate) fn begin_registration(
        &self,
        identity: &RegistrationIdentity,
    ) -> Result<RunningRegistration, Status> {
        self.registrations.begin_registration(identity)
    }

    pub(crate) async fn end_registration(&self, running: RunningRegistration) {
        let Some(ended) = self.registrations.end_registration(running) else {
            return;
        };

        let registration = ended.registration;
        let model_ids = ended.cleanup_model_ids;
        registration
            .cluster_generation
            .calibrations
            .release_owned_assignments(&registration.identity, &model_ids)
            .await;
        let routing_key = registration.identity.routing_key.clone();
        let targets: HashSet<RoutingTargetKey> = model_ids
            .into_iter()
            .map(|model_id| RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id,
            })
            .collect();
        self.routing
            .remove_inference_server_targets(&registration, &targets)
            .await;
    }

    pub(crate) async fn submit_cluster_calibration(
        &self,
        routing_key: Option<String>,
        request: &SubmitClusterCalibrationRequest,
    ) -> Result<(), Status> {
        self.registrations
            .submit_cluster_calibration(routing_key, request)
            .await
    }

    pub(crate) async fn apply_registration_update(
        &self,
        running: &RunningRegistration,
        update: &InferenceServerRegistration,
        reverse_connected: bool,
        rtt: Option<Duration>,
    ) -> Vec<ModelCalibrationDirective> {
        let registration = running.generation();
        let identity = &registration.identity;
        let routing_key = &identity.routing_key;
        let mut calibration_directives = Vec::new();

        // Publish the full heartbeat membership before per-target awaits while
        // retaining the cleanup union until every routing mutation commits.
        let current_models: BTreeSet<String> = update.models.keys().cloned().collect();
        let removed_models = self
            .registrations
            .begin_advertised_model_update(running, current_models);
        let removed_targets: HashSet<RoutingTargetKey> = removed_models
            .iter()
            .map(|model_id| RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id: model_id.clone(),
            })
            .collect();
        registration
            .cluster_generation
            .calibrations
            .release_owned_assignments(identity, &removed_models)
            .await;
        self.routing
            .remove_inference_server_targets(&registration, &removed_targets)
            .await;

        for (model_id, model) in &update.models {
            // Identical stats across consecutive updates are expected because
            // heartbeat sends carry full registration snapshots.
            let target = RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id: model_id.clone(),
            };
            let (calibration_directive, calibration_pending) = registration
                .cluster_generation
                .calibrations
                .registration_decision(identity, model_id)
                .await
                .into_parts();
            if let Some(directive) = calibration_directive {
                calibration_directives.push(directive);
            }
            let stats = model.stats.clone().unwrap_or_default();
            let model_status = InferenceServerStatus::try_from(model.status)
                .unwrap_or(InferenceServerStatus::Unknown);
            let effective_status =
                if (identity.reverse_tunnel && !reverse_connected) || calibration_pending {
                    InferenceServerStatus::Inactive
                } else if model.stats.is_none() {
                    warn!(
                        inference_server_id = %identity.inference_server_id,
                        model_id = %model_id,
                        "missing model stats in registration; setting model status to inactive"
                    );
                    InferenceServerStatus::Inactive
                } else {
                    model_status
                };

            if effective_status == InferenceServerStatus::Active {
                let Some(current_rtt) = rtt else {
                    warn!(
                        inference_server_id = %identity.inference_server_id,
                        model_id = %model_id,
                        "active model registration missing connection RTT; skipping routing update"
                    );
                    self.routing
                        .remove_inference_server_from_target(&registration, &target)
                        .await;
                    continue;
                };
                self.routing
                    .upsert_inference_server_target(
                        &target,
                        RoutedInferenceServerSnapshot::new(
                            registration.clone(),
                            stats,
                            current_rtt,
                            Instant::now(),
                            effective_status,
                        ),
                    )
                    .await;
            } else {
                self.routing
                    .remove_inference_server_from_target(&registration, &target)
                    .await;
            }
        }

        self.registrations.finish_advertised_model_update(running);
        calibration_directives
    }

    /// Returns all active inference server snapshots for a
    /// `(routing_key, model_id)` pair. The HTTP proxy calls this to get the
    /// candidate set that the load balancer chooses from.
    pub async fn candidates_for_target(
        &self,
        target: &RoutingTargetKey,
    ) -> Vec<RoutedInferenceServerSnapshot> {
        self.routing.candidates_for_target(target).await
    }

    pub async fn cluster_candidates_for_target(
        &self,
        target: &RoutingTargetKey,
    ) -> Vec<RoutedClusterSnapshot> {
        self.routing.cluster_candidates_for_target(target).await
    }

    pub(crate) async fn routing_target_snapshot(
        &self,
        target: &RoutingTargetKey,
    ) -> Option<RoutingTargetSnapshot> {
        self.routing.routing_target_snapshot(target).await
    }

    pub fn has_registered_model_for_target(&self, target: &RoutingTargetKey) -> bool {
        self.registrations.has_registered_model_for_target(target)
    }

    pub async fn list_active_models(
        &self,
        routing_key: Option<&str>,
        model_ids: &[String],
    ) -> Vec<String> {
        self.routing
            .list_active_models(routing_key, model_ids)
            .await
    }

    /// Looks up the registration for an inference server that declared
    /// `reverse_tunnel = true` during gRPC registration. Returns `None` if
    /// the server is not registered or was registered without reverse tunnel
    /// mode.
    ///
    /// Called during the QUIC reverse-tunnel handshake to confirm the
    /// connecting server was expected and to retrieve the auth-derived
    /// routing key for comparison against the QUIC handshake's own auth
    /// result.
    pub(crate) fn reverse_tunnel_registration(
        &self,
        inference_server_id: &str,
    ) -> Option<Arc<RegistrationGeneration>> {
        self.registrations
            .reverse_tunnel_registration(inference_server_id)
    }
}

#[cfg(test)]
mod tests;
