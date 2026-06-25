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

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use scc::HashMap as SccHashMap;

use super::{LoadBalancer, LoadBalancerAlgorithmConfig, create_load_balancer_with_config};

// The non-zero-sized token guarantees each Arc has a distinct stable pointer.
struct LoadBalancerDefinitionToken {
    _unique_allocation: u8,
}

#[derive(Clone)]
struct LoadBalancerDefinitionId(Arc<LoadBalancerDefinitionToken>);

impl LoadBalancerDefinitionId {
    fn new() -> Self {
        Self(Arc::new(LoadBalancerDefinitionToken {
            _unique_allocation: 0,
        }))
    }
}

impl fmt::Debug for LoadBalancerDefinitionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LoadBalancerDefinitionId")
    }
}

impl PartialEq for LoadBalancerDefinitionId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for LoadBalancerDefinitionId {}

impl Hash for LoadBalancerDefinitionId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Clone, Debug)]
pub(super) struct LoadBalancerDefinition {
    id: LoadBalancerDefinitionId,
    config: LoadBalancerAlgorithmConfig,
}

impl LoadBalancerDefinition {
    pub(super) fn new(config: LoadBalancerAlgorithmConfig) -> anyhow::Result<Self> {
        let _ = create_load_balancer_with_config(&config)?;
        Ok(Self {
            id: LoadBalancerDefinitionId::new(),
            config,
        })
    }

    pub(super) fn config(&self) -> &LoadBalancerAlgorithmConfig {
        &self.config
    }
}

/// Stateful load-balancer instances owned by one routing-target generation.
///
/// Keep this value alive exactly as long as its routing target. Replacing it
/// intentionally resets target-local counters and caches.
#[derive(Default)]
pub struct LoadBalancerTargetState {
    instances: SccHashMap<LoadBalancerDefinitionId, Arc<dyn LoadBalancer>>,
}

impl LoadBalancerTargetState {
    pub(super) fn load_balancer(
        &self,
        definition: &LoadBalancerDefinition,
    ) -> Arc<dyn LoadBalancer> {
        if let Some(lb) = self
            .instances
            .read_sync(&definition.id, |_definition_id, lb| lb.clone())
        {
            return lb;
        }

        let lb = create_load_balancer_with_config(definition.config())
            .expect("load balancer config validated during router construction");
        if self
            .instances
            .insert_sync(definition.id.clone(), lb.clone())
            .is_ok()
        {
            return lb;
        }

        self.instances
            .read_sync(&definition.id, |_definition_id, lb| lb.clone())
            .expect("target-local load balancer should exist after insert race")
    }

    #[cfg(test)]
    pub(super) fn instance_count(&self) -> usize {
        self.instances.len()
    }
}

impl fmt::Debug for LoadBalancerTargetState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LoadBalancerTargetState")
            .field("instance_count", &self.instances.len())
            .finish()
    }
}
