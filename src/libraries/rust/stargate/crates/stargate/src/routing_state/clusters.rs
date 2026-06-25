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

use super::keys::RoutingTargetKey;
use super::registration::RegistrationGeneration;
use super::snapshots::{
    ClusterBackendRemoval, ClusterBackendUpsert, RoutedClusterSnapshot, RoutedClusterState,
    RoutedInferenceServerSnapshot, RoutingTargetGeneration, RoutingTargetSnapshot,
    RoutingTargetState,
};
use super::*;

#[derive(Debug, Default)]
pub(super) struct RoutingLifecycle {
    pub(super) targets: RoutingTargetStore,
}

#[derive(Debug, Default)]
pub(super) struct RoutingTargetStore {
    targets: SccHashMap<RoutingTargetKey, Arc<RoutingTargetState>>,
    metrics: Option<Arc<StargateMetrics>>,
}

impl RoutingTargetState {
    pub(super) fn upsert_backend(
        &self,
        backend: Arc<RoutedInferenceServerSnapshot>,
    ) -> Result<(), Arc<RoutedInferenceServerSnapshot>> {
        let cluster_id = backend.registration.identity.cluster_id.clone();
        let cluster_generation = backend.registration.cluster_generation.clone();
        assert!(
            !cluster_generation.is_retired(),
            "retired registration cluster generation cannot publish routing state"
        );
        let mut generation = self.generation.lock();
        let RoutingTargetGeneration::Active {
            clusters,
            active_backend_count,
        } = &mut *generation
        else {
            return Err(backend);
        };

        let Some(cluster_state) = clusters.get(&cluster_id).cloned() else {
            let cluster_state = Arc::new(RoutedClusterState::new(cluster_generation));
            assert_eq!(
                cluster_state.upsert_backend(backend),
                ClusterBackendUpsert::Inserted,
                "new routed cluster generation must insert its first backend"
            );
            clusters.insert(cluster_id, cluster_state);
            *active_backend_count = active_backend_count
                .checked_add(1)
                .expect("routing target active backend count overflow");
            return Ok(());
        };

        if Arc::ptr_eq(cluster_state.cluster_generation(), &cluster_generation) {
            if cluster_state.upsert_backend(backend) == ClusterBackendUpsert::Inserted {
                *active_backend_count = active_backend_count
                    .checked_add(1)
                    .expect("routing target active backend count overflow");
            }
            return Ok(());
        }

        assert!(
            cluster_state.cluster_generation().is_retired(),
            "one routing target cannot contain different active generations of the same cluster"
        );
        let stale_backend_count = cluster_state.backend_count();
        let replacement = Arc::new(RoutedClusterState::new(cluster_generation));
        assert_eq!(
            replacement.upsert_backend(backend),
            ClusterBackendUpsert::Inserted,
            "replacement routed cluster generation must insert its first backend"
        );
        let replaced = clusters
            .insert(cluster_id, replacement)
            .expect("existing routed cluster generation should be replaced");
        assert!(
            Arc::ptr_eq(&replaced, &cluster_state),
            "replaced routed cluster generation should match the inspected generation"
        );
        *active_backend_count = active_backend_count
            .checked_sub(stale_backend_count)
            .and_then(|count| count.checked_add(1))
            .expect("routing target active backend count replacement overflow");
        Ok(())
    }

    pub(super) fn remove_backend(&self, registration: &Arc<RegistrationGeneration>) {
        let cluster_id = &registration.identity.cluster_id;
        let mut generation = self.generation.lock();
        let RoutingTargetGeneration::Active {
            clusters,
            active_backend_count,
        } = &mut *generation
        else {
            return;
        };
        let Some(cluster_state) = clusters.get(cluster_id).cloned() else {
            return;
        };

        if !Arc::ptr_eq(
            cluster_state.cluster_generation(),
            &registration.cluster_generation,
        ) {
            if !cluster_state.cluster_generation().is_retired() {
                assert!(
                    registration.cluster_generation.is_retired(),
                    "one routing target cannot contain different active generations of the same cluster"
                );
                return;
            }
            let removed = clusters.remove(cluster_id).expect(
                "retired routed cluster generation should remain target-owned until removal",
            );
            assert!(
                Arc::ptr_eq(&removed, &cluster_state),
                "removed routed cluster generation should match the inspected generation"
            );
            *active_backend_count = active_backend_count
                .checked_sub(cluster_state.backend_count())
                .expect("routing target active backend count underflow");
            return;
        }

        match cluster_state.remove_backend(registration) {
            ClusterBackendRemoval::Missing => return,
            ClusterBackendRemoval::Removed => {}
            ClusterBackendRemoval::Emptied => {
                let removed = clusters
                    .remove(cluster_id)
                    .expect("emptied cluster should remain target-owned until removal");
                assert!(
                    Arc::ptr_eq(&removed, &cluster_state),
                    "removed cluster should match the mutated target-owned generation"
                );
            }
        }
        *active_backend_count = active_backend_count
            .checked_sub(1)
            .expect("routing target active backend count underflow");
    }

    fn cluster_states(&self) -> Vec<Arc<RoutedClusterState>> {
        match &*self.generation.lock() {
            RoutingTargetGeneration::Active { clusters, .. } => {
                clusters.values().cloned().collect()
            }
            RoutingTargetGeneration::Retired => Vec::new(),
        }
    }

    pub(super) fn active_backend_count(&self) -> usize {
        match &*self.generation.lock() {
            RoutingTargetGeneration::Active {
                active_backend_count,
                ..
            } => *active_backend_count,
            RoutingTargetGeneration::Retired => 0,
        }
    }

    fn is_routable(&self) -> bool {
        self.active_backend_count() > 0
    }

    fn retire_if_empty(&self) -> bool {
        let mut generation = self.generation.lock();
        match &*generation {
            RoutingTargetGeneration::Retired => return true,
            RoutingTargetGeneration::Active {
                clusters,
                active_backend_count,
            } if clusters.is_empty() => {
                assert_eq!(
                    *active_backend_count, 0,
                    "empty routing target generation must not retain active backends"
                );
            }
            RoutingTargetGeneration::Active { .. } => return false,
        }
        *generation = RoutingTargetGeneration::Retired;
        true
    }
}

impl RoutingTargetStore {
    fn new(metrics: Option<Arc<StargateMetrics>>) -> Self {
        Self {
            targets: SccHashMap::default(),
            metrics,
        }
    }

    pub(super) async fn target_state(
        &self,
        target: &RoutingTargetKey,
    ) -> Option<Arc<RoutingTargetState>> {
        self.targets
            .read_async(target, |_key, state| state.clone())
            .await
    }

    pub(super) async fn target_state_or_insert(
        &self,
        target: &RoutingTargetKey,
    ) -> Arc<RoutingTargetState> {
        loop {
            if let Some(existing) = self.target_state(target).await {
                return existing;
            }

            let candidate = Arc::new(RoutingTargetState::default());
            if self
                .targets
                .insert_async(target.clone(), candidate.clone())
                .await
                .is_ok()
            {
                return candidate;
            }

            if let Some(existing) = self.target_state(target).await {
                return existing;
            }
        }
    }

    pub(super) async fn targets(&self) -> Vec<(RoutingTargetKey, Arc<RoutingTargetState>)> {
        let mut targets = Vec::new();
        let _ = self
            .targets
            .iter_async(|target, target_state| {
                targets.push((target.clone(), target_state.clone()));
                true
            })
            .await;
        targets
    }

    pub(super) async fn matching_targets(
        &self,
        routing_key: Option<&str>,
        model_ids: &[String],
    ) -> Vec<(RoutingTargetKey, Arc<RoutingTargetState>)> {
        if model_ids.is_empty() {
            return self
                .targets()
                .await
                .into_iter()
                .filter(|(target, _state)| target.routing_key.as_deref() == routing_key)
                .collect();
        }

        let routing_key = routing_key.map(ToOwned::to_owned);
        let mut targets = Vec::new();
        for model_id in model_ids.iter().cloned().collect::<BTreeSet<_>>() {
            let target = RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id,
            };
            if let Some(target_state) = self.target_state(&target).await {
                targets.push((target, target_state));
            }
        }
        targets
    }

    async fn publish_active_backend_count(&self, target: &RoutingTargetKey) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        loop {
            let before = self.target_state(target).await;
            let count = before
                .as_ref()
                .map_or(0, |target_state| target_state.active_backend_count());
            metrics.set_active_inference_servers(
                target.routing_key.as_deref(),
                &target.model_id,
                count,
            );

            let after = self.target_state(target).await;
            let unchanged = match (&before, &after) {
                (None, None) => true,
                (Some(before), Some(after)) => {
                    Arc::ptr_eq(before, after) && after.active_backend_count() == count
                }
                _ => false,
            };
            if unchanged {
                return;
            }
        }
    }

    pub(super) async fn remove_if_empty(
        &self,
        target: &RoutingTargetKey,
        target_state: Arc<RoutingTargetState>,
    ) -> bool {
        // Retire while the current map entry is write-locked so a stale owner
        // cannot publish into detached state after a replacement is inserted.
        self.targets
            .remove_if_async(target, move |current| {
                Arc::ptr_eq(current, &target_state) && current.retire_if_empty()
            })
            .await
            .is_some()
    }
}

impl RoutingLifecycle {
    pub(super) fn new(metrics: Option<Arc<StargateMetrics>>) -> Self {
        Self {
            targets: RoutingTargetStore::new(metrics),
        }
    }

    pub(super) async fn target_state(
        &self,
        target: &RoutingTargetKey,
    ) -> Option<Arc<RoutingTargetState>> {
        self.targets.target_state(target).await
    }

    pub(super) async fn target_state_or_insert(
        &self,
        target: &RoutingTargetKey,
    ) -> Arc<RoutingTargetState> {
        self.targets.target_state_or_insert(target).await
    }

    pub(super) async fn upsert_inference_server_target(
        &self,
        target: &RoutingTargetKey,
        snapshot: RoutedInferenceServerSnapshot,
    ) {
        let mut snapshot = Arc::new(snapshot);
        loop {
            let target_state = self.target_state_or_insert(target).await;
            let Err(rejected) = target_state.upsert_backend(snapshot) else {
                break;
            };
            snapshot = rejected;
            let _ = self.targets.remove_if_empty(target, target_state).await;
        }
        self.targets.publish_active_backend_count(target).await;
    }

    pub(super) async fn remove_inference_server_from_target(
        &self,
        registration: &Arc<RegistrationGeneration>,
        target: &RoutingTargetKey,
    ) {
        let Some(target_state) = self.target_state(target).await else {
            return;
        };

        target_state.remove_backend(registration);
        let _ = self
            .targets
            .remove_if_empty(target, target_state.clone())
            .await;
        self.targets.publish_active_backend_count(target).await;
    }

    pub(super) async fn remove_inference_server_targets(
        &self,
        registration: &Arc<RegistrationGeneration>,
        targets: &HashSet<RoutingTargetKey>,
    ) {
        for target in targets {
            self.remove_inference_server_from_target(registration, target)
                .await;
        }
    }

    pub(super) async fn candidates_for_target(
        &self,
        target: &RoutingTargetKey,
    ) -> Vec<RoutedInferenceServerSnapshot> {
        let Some(target_state) = self.target_state(target).await else {
            return Vec::new();
        };

        let mut candidates = Vec::new();
        for cluster_state in target_state.cluster_states() {
            candidates.extend(cluster_state.backend_snapshot_values());
        }
        candidates
    }

    pub(super) async fn cluster_candidates_for_target(
        &self,
        target: &RoutingTargetKey,
    ) -> Vec<RoutedClusterSnapshot> {
        self.routing_target_snapshot(target)
            .await
            .map(RoutingTargetSnapshot::into_clusters)
            .unwrap_or_default()
    }

    pub(super) async fn routing_target_snapshot(
        &self,
        target: &RoutingTargetKey,
    ) -> Option<RoutingTargetSnapshot> {
        let target_state = self.target_state(target).await?;

        let mut clusters = Vec::new();
        for cluster_state in target_state.cluster_states() {
            let calibrated_last_mean_input_tps = cluster_state
                .cluster_generation()
                .calibrations
                .completed_last_mean_input_tps(&target.model_id)
                .await;
            if let Some(snapshot) = cluster_state.routing_snapshot(calibrated_last_mean_input_tps) {
                clusters.push((snapshot, cluster_state));
            }
        }
        Some(RoutingTargetSnapshot::new(target_state, clusters))
    }

    pub(super) async fn list_active_models(
        &self,
        routing_key: Option<&str>,
        model_ids: &[String],
    ) -> Vec<String> {
        let mut active_models = BTreeSet::new();
        for (target, target_state) in self.targets.matching_targets(routing_key, model_ids).await {
            if target_state.is_routable() {
                active_models.insert(target.model_id);
            }
        }
        active_models.into_iter().collect()
    }
}
