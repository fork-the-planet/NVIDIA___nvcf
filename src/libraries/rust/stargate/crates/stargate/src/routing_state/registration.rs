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

use super::calibration::{ClusterCalibrations, validate_cluster_calibration_submission};
use super::keys::{RegistrationIdentity, RoutingTargetKey};
use super::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::tunnel::RegistrationConnections;

#[derive(Debug, Default)]
pub(super) struct RegistrationRegistry {
    membership: RwLock<RegistrationMembership>,
}

#[derive(Debug, Default)]
struct RegistrationMembership {
    registrations_by_id: HashMap<String, RegistrationRecord>,
    active_clusters: HashMap<RegistrationClusterKey, ActiveRegistrationCluster>,
    advertised_targets: HashMap<RoutingTargetKey, usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RegistrationClusterKey {
    routing_key: Option<String>,
    cluster_id: String,
}

#[derive(Debug)]
struct RegistrationRecord {
    state: Arc<RegistrationGeneration>,
    models: RegisteredModelState,
}

#[derive(Debug)]
struct ActiveRegistrationCluster {
    generation: Arc<RegistrationClusterGeneration>,
    registration_count: usize,
}

#[derive(Debug)]
pub(super) struct RegistrationClusterGeneration {
    retired: AtomicBool,
    pub(super) calibrations: ClusterCalibrations,
}

#[derive(Debug)]
pub(crate) struct RegistrationGeneration {
    pub(super) identity: RegistrationIdentity,
    pub(super) cluster_generation: Arc<RegistrationClusterGeneration>,
    tunnel_connections: RegistrationConnections,
}

#[derive(Debug)]
enum RegisteredModelState {
    Stable(BTreeSet<String>),
    Applying {
        previous: BTreeSet<String>,
        advertised: BTreeSet<String>,
    },
}

struct AdvertisedModelDelta {
    added: BTreeSet<String>,
    removed: BTreeSet<String>,
}

pub(crate) struct RunningRegistration {
    state: Arc<RegistrationGeneration>,
}

pub(super) struct EndedRegistration {
    pub(super) registration: Arc<RegistrationGeneration>,
    pub(super) cleanup_model_ids: BTreeSet<String>,
}

impl RunningRegistration {
    pub(crate) fn identity(&self) -> &RegistrationIdentity {
        &self.state.identity
    }

    #[cfg(test)]
    pub(super) fn cluster_generation(&self) -> &Arc<RegistrationClusterGeneration> {
        &self.state.cluster_generation
    }

    pub(crate) fn generation(&self) -> Arc<RegistrationGeneration> {
        self.state.clone()
    }

    fn into_registration_generation(self) -> Arc<RegistrationGeneration> {
        self.state
    }
}

impl RegistrationGeneration {
    pub(crate) fn inference_server_id(&self) -> &str {
        &self.identity.inference_server_id
    }

    pub(crate) fn inference_server_url(&self) -> &str {
        &self.identity.inference_server_url
    }

    pub(crate) fn routing_key(&self) -> &Option<String> {
        &self.identity.routing_key
    }

    pub(crate) fn reverse_tunnel(&self) -> bool {
        self.identity.reverse_tunnel
    }

    pub(crate) fn tunnel_connections(&self) -> &RegistrationConnections {
        &self.tunnel_connections
    }
}

impl RegistrationClusterGeneration {
    fn new() -> Self {
        Self {
            retired: AtomicBool::new(false),
            calibrations: ClusterCalibrations::default(),
        }
    }

    fn retire(&self) {
        assert!(
            !self.retired.swap(true, Ordering::AcqRel),
            "registration cluster generation retired more than once"
        );
    }

    pub(super) fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }
}

impl RegisteredModelState {
    fn begin_update(&mut self, advertised: BTreeSet<String>) -> AdvertisedModelDelta {
        let Self::Stable(previous) = self else {
            panic!("registration model update started while another update is applying");
        };
        let previous = std::mem::take(previous);
        let added = advertised.difference(&previous).cloned().collect();
        let removed = previous.difference(&advertised).cloned().collect();
        *self = Self::Applying {
            previous,
            advertised,
        };
        AdvertisedModelDelta { added, removed }
    }

    fn finish_update(&mut self) {
        let Self::Applying { advertised, .. } = self else {
            panic!("registration model update finished without an applying update");
        };
        let advertised = std::mem::take(advertised);
        *self = Self::Stable(advertised);
    }

    fn advertised_models(&self) -> &BTreeSet<String> {
        match self {
            Self::Stable(advertised) | Self::Applying { advertised, .. } => advertised,
        }
    }

    fn cleanup_model_ids(&self) -> BTreeSet<String> {
        match self {
            Self::Stable(advertised) => advertised.clone(),
            Self::Applying {
                previous,
                advertised,
            } => previous.union(advertised).cloned().collect(),
        }
    }
}

impl RegistrationClusterKey {
    fn from_identity(identity: &RegistrationIdentity) -> Self {
        Self {
            routing_key: identity.routing_key.clone(),
            cluster_id: identity.cluster_id.clone(),
        }
    }

    fn from_submission(
        routing_key: Option<String>,
        request: &SubmitClusterCalibrationRequest,
    ) -> Self {
        Self {
            routing_key,
            cluster_id: request.cluster_id.clone(),
        }
    }
}

impl RegistrationMembership {
    fn begin_registration(
        &mut self,
        identity: &RegistrationIdentity,
    ) -> Result<RunningRegistration, Status> {
        if let Some(existing) = self.registrations_by_id.get(&identity.inference_server_id) {
            return Err(duplicate_registration_status(
                &identity.inference_server_id,
                &existing.state,
            ));
        }

        let cluster_key = RegistrationClusterKey::from_identity(identity);
        let cluster_generation = match self.active_clusters.entry(cluster_key) {
            Entry::Occupied(mut existing) => {
                let cluster = existing.get_mut();
                cluster.registration_count = cluster
                    .registration_count
                    .checked_add(1)
                    .expect("registration cluster generation count overflow");
                cluster.generation.clone()
            }
            Entry::Vacant(vacant) => {
                let generation = Arc::new(RegistrationClusterGeneration::new());
                vacant.insert(ActiveRegistrationCluster {
                    generation: generation.clone(),
                    registration_count: 1,
                });
                generation
            }
        };
        let state = Arc::new(RegistrationGeneration {
            identity: identity.clone(),
            cluster_generation,
            tunnel_connections: RegistrationConnections::new(),
        });
        let replaced = self.registrations_by_id.insert(
            identity.inference_server_id.clone(),
            RegistrationRecord {
                state: state.clone(),
                models: RegisteredModelState::Stable(BTreeSet::new()),
            },
        );
        assert!(
            replaced.is_none(),
            "duplicate registration inserted after exact-id check"
        );
        Ok(RunningRegistration { state })
    }

    fn end_registration(&mut self, running: RunningRegistration) -> Option<EndedRegistration> {
        let registration = running.into_registration_generation();
        let inference_server_id = &registration.identity.inference_server_id;
        let current = self.registrations_by_id.get(inference_server_id)?;
        if !Arc::ptr_eq(&current.state, &registration) {
            return None;
        }

        let removed = self
            .registrations_by_id
            .remove(inference_server_id)
            .expect("exact registration should remain present until removal");
        self.remove_advertised_targets(
            &registration.identity.routing_key,
            removed.models.advertised_models(),
        );
        let cleanup_model_ids = removed.models.cleanup_model_ids();

        let cluster_key = RegistrationClusterKey::from_identity(&registration.identity);
        let remove_cluster = {
            let cluster = self
                .active_clusters
                .get_mut(&cluster_key)
                .expect("live registration must retain its active cluster generation");
            assert!(
                Arc::ptr_eq(&cluster.generation, &registration.cluster_generation),
                "active cluster entry must match exact registration generation"
            );
            if cluster.registration_count == 1 {
                cluster.generation.retire();
                true
            } else {
                cluster.registration_count = cluster
                    .registration_count
                    .checked_sub(1)
                    .expect("registration cluster generation count underflow");
                false
            }
        };
        if remove_cluster {
            let removed_cluster = self
                .active_clusters
                .remove(&cluster_key)
                .expect("final registration must remove its active cluster entry");
            assert!(
                Arc::ptr_eq(
                    &removed_cluster.generation,
                    &registration.cluster_generation
                ),
                "removed cluster entry must match exact registration generation"
            );
        }

        Some(EndedRegistration {
            registration,
            cleanup_model_ids,
        })
    }

    fn begin_advertised_model_update(
        &mut self,
        registration: &Arc<RegistrationGeneration>,
        advertised_models: BTreeSet<String>,
    ) -> BTreeSet<String> {
        let inference_server_id = &registration.identity.inference_server_id;
        let update = {
            let record = self
                .registrations_by_id
                .get_mut(inference_server_id)
                .expect("running registration must remain indexed during model update");
            assert!(
                Arc::ptr_eq(&record.state, registration),
                "model update must target the exact running registration"
            );
            record.models.begin_update(advertised_models)
        };
        self.remove_advertised_targets(&registration.identity.routing_key, &update.removed);
        self.add_advertised_targets(&registration.identity.routing_key, &update.added);
        update.removed
    }

    fn finish_advertised_model_update(&mut self, registration: &Arc<RegistrationGeneration>) {
        let inference_server_id = &registration.identity.inference_server_id;
        let record = self
            .registrations_by_id
            .get_mut(inference_server_id)
            .expect("running registration must remain indexed when model update finishes");
        assert!(
            Arc::ptr_eq(&record.state, registration),
            "finished model update must target the exact running registration"
        );
        record.models.finish_update();
    }

    fn cluster_generation_for_submission(
        &self,
        routing_key: Option<String>,
        request: &SubmitClusterCalibrationRequest,
    ) -> Option<Arc<RegistrationClusterGeneration>> {
        if let Some(record) = self.registrations_by_id.get(&request.inference_server_id)
            && record.state.identity.routing_key == routing_key
            && record.state.identity.cluster_id == request.cluster_id
        {
            return Some(record.state.cluster_generation.clone());
        }
        self.active_clusters
            .get(&RegistrationClusterKey::from_submission(
                routing_key,
                request,
            ))
            .map(|cluster| cluster.generation.clone())
    }

    fn has_registered_model_for_target(&self, target: &RoutingTargetKey) -> bool {
        self.advertised_targets.contains_key(target)
    }

    fn reverse_tunnel_registration(
        &self,
        inference_server_id: &str,
    ) -> Option<Arc<RegistrationGeneration>> {
        let registration = &self.registrations_by_id.get(inference_server_id)?.state;
        registration
            .identity
            .reverse_tunnel
            .then(|| registration.clone())
    }

    fn add_advertised_targets(
        &mut self,
        routing_key: &Option<String>,
        model_ids: &BTreeSet<String>,
    ) {
        for model_id in model_ids {
            let target = RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id: model_id.clone(),
            };
            let advertiser_count = self.advertised_targets.entry(target).or_default();
            *advertiser_count = advertiser_count
                .checked_add(1)
                .expect("registered target advertiser count overflow");
        }
    }

    fn remove_advertised_targets(
        &mut self,
        routing_key: &Option<String>,
        model_ids: &BTreeSet<String>,
    ) {
        for model_id in model_ids {
            let target = RoutingTargetKey {
                routing_key: routing_key.clone(),
                model_id: model_id.clone(),
            };
            match self.advertised_targets.entry(target) {
                Entry::Occupied(existing) if *existing.get() == 1 => {
                    existing.remove();
                }
                Entry::Occupied(mut existing) => {
                    *existing.get_mut() = existing
                        .get()
                        .checked_sub(1)
                        .expect("registered target advertiser count underflow");
                }
                Entry::Vacant(_) => {
                    panic!("registered target advertiser removed without active membership");
                }
            }
        }
    }
}

impl RegistrationRegistry {
    pub(super) fn begin_registration(
        &self,
        identity: &RegistrationIdentity,
    ) -> Result<RunningRegistration, Status> {
        self.membership.write().begin_registration(identity)
    }

    pub(super) fn end_registration(
        &self,
        running: RunningRegistration,
    ) -> Option<EndedRegistration> {
        self.membership.write().end_registration(running)
    }

    pub(super) fn begin_advertised_model_update(
        &self,
        running: &RunningRegistration,
        advertised_models: BTreeSet<String>,
    ) -> BTreeSet<String> {
        self.membership
            .write()
            .begin_advertised_model_update(&running.state, advertised_models)
    }

    pub(super) fn finish_advertised_model_update(&self, running: &RunningRegistration) {
        self.membership
            .write()
            .finish_advertised_model_update(&running.state);
    }

    #[cfg(test)]
    pub(super) fn cleanup_model_ids(&self, running: &RunningRegistration) -> BTreeSet<String> {
        let membership = self.membership.read();
        let record = membership
            .registrations_by_id
            .get(&running.state.identity.inference_server_id)
            .expect("running registration must remain indexed for cleanup inspection");
        assert!(
            Arc::ptr_eq(&record.state, &running.state),
            "cleanup inspection must target the exact running registration"
        );
        record.models.cleanup_model_ids()
    }

    pub(super) async fn submit_cluster_calibration(
        &self,
        routing_key: Option<String>,
        request: &SubmitClusterCalibrationRequest,
    ) -> Result<(), Status> {
        validate_cluster_calibration_submission(request)?;
        let cluster_generation = {
            let membership = self.membership.read();
            membership.cluster_generation_for_submission(routing_key, request)
        };
        let Some(cluster_generation) = cluster_generation else {
            return Err(Status::failed_precondition(
                "cluster calibration has no active local assignment",
            ));
        };
        cluster_generation
            .calibrations
            .submit_validated(request)
            .await
    }

    pub(super) fn has_registered_model_for_target(&self, target: &RoutingTargetKey) -> bool {
        self.membership
            .read()
            .has_registered_model_for_target(target)
    }

    pub(super) fn reverse_tunnel_registration(
        &self,
        inference_server_id: &str,
    ) -> Option<Arc<RegistrationGeneration>> {
        self.membership
            .read()
            .reverse_tunnel_registration(inference_server_id)
    }
}

#[cfg(test)]
pub(crate) fn test_registration_generation(
    identity: RegistrationIdentity,
) -> Arc<RegistrationGeneration> {
    test_registration_generation_in_cluster(
        identity,
        Arc::new(RegistrationClusterGeneration::new()),
    )
}

#[cfg(test)]
pub(super) fn test_registration_generation_in_cluster(
    identity: RegistrationIdentity,
    cluster_generation: Arc<RegistrationClusterGeneration>,
) -> Arc<RegistrationGeneration> {
    Arc::new(RegistrationGeneration {
        identity,
        cluster_generation,
        tunnel_connections: RegistrationConnections::new(),
    })
}

pub(super) fn duplicate_registration_status(
    inference_server_id: &str,
    existing: &Arc<RegistrationGeneration>,
) -> Status {
    warn!(
        inference_server_id = %inference_server_id,
        existing_url = %existing.identity.inference_server_url,
        existing_reverse_tunnel = existing.identity.reverse_tunnel,
        "duplicate inference_server_id: another stream already registered this id"
    );
    Status::already_exists(format!(
        "inference_server_id '{}' is already registered",
        inference_server_id
    ))
}
