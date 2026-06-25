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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::RwLock;
use xxhash_rust::xxh3::xxh3_64;

use super::{GroqMultiregionConfig, double_excluded_cluster_ids, single_excluded_cluster_id};
use crate::load_balancer::{
    HashInputBuilder, LoadBalancerRequest, cache_affinity_key_is_cacheable,
};
use crate::routing_state::{RoutedClusterSnapshot, RoutingTargetKey};

const SELECTION_CACHE_LIMIT: usize = 4096;

#[derive(Default)]
pub(super) struct CacheAffinitySelector {
    cache: RwLock<CacheAffinityRingCache>,
}

impl CacheAffinitySelector {
    pub(super) fn candidate_indices(
        &self,
        config: &GroqMultiregionConfig,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<Arc<Vec<usize>>> {
        let cache_affinity_key = request.cache_affinity_key?;
        let selection_count = config.cache_affinity_backend_selection_count()?;
        if candidates.is_empty() {
            return None;
        }

        let selection_count = selection_count.min(candidates.len());
        let single_excluded_cluster_id = single_excluded_cluster_id(request);
        let double_excluded_cluster_ids = double_excluded_cluster_ids(request);
        let cacheable_selection = cache_affinity_key_is_cacheable(cache_affinity_key)
            && (!request.has_excluded_clusters()
                || single_excluded_cluster_id.is_some()
                || double_excluded_cluster_ids.is_some());
        let selected_indices = match self.cached_or_computed_selection(
            config,
            request,
            candidates,
            cache_affinity_key,
            selection_count,
            cacheable_selection,
        ) {
            CacheAffinitySelectionLookup::Hit(indices) => indices,
            CacheAffinitySelectionLookup::Computed(indices) => {
                if cacheable_selection && !indices.is_empty() {
                    let mut cache = self.cache.write();
                    if cache.matches(request.routing_target, candidates) {
                        cache.insert_selection(
                            cache_affinity_key,
                            single_excluded_cluster_id,
                            double_excluded_cluster_ids,
                            indices.clone(),
                        );
                    }
                }
                indices
            }
            CacheAffinitySelectionLookup::Stale => {
                let ring = build_ring(config, request, candidates);
                let indices = Arc::new(select_candidate_indices(
                    request,
                    &ring,
                    cache_affinity_key,
                    selection_count,
                    config,
                ));
                let mut cache = self.cache.write();
                cache.replace(request.routing_target, candidates, ring);
                if cacheable_selection && !indices.is_empty() {
                    cache.insert_selection(
                        cache_affinity_key,
                        single_excluded_cluster_id,
                        double_excluded_cluster_ids,
                        indices.clone(),
                    );
                }
                indices
            }
        };

        (!selected_indices.is_empty()).then_some(selected_indices)
    }

    fn cached_or_computed_selection(
        &self,
        config: &GroqMultiregionConfig,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        cache_affinity_key: &str,
        selection_count: usize,
        cacheable_key: bool,
    ) -> CacheAffinitySelectionLookup {
        let cache = self.cache.read();
        if !cache.matches(request.routing_target, candidates) {
            return CacheAffinitySelectionLookup::Stale;
        }

        if cacheable_key {
            if let Some(excluded_cluster_id) = single_excluded_cluster_id(request) {
                if let Some(indices) =
                    cache.single_excluded_selection(cache_affinity_key, excluded_cluster_id)
                {
                    return CacheAffinitySelectionLookup::Hit(indices);
                }
            } else if let Some((first_excluded, second_excluded)) =
                double_excluded_cluster_ids(request)
            {
                if let Some(indices) = cache.two_excluded_selection(
                    cache_affinity_key,
                    first_excluded,
                    second_excluded,
                ) {
                    return CacheAffinitySelectionLookup::Hit(indices);
                }
            } else if let Some(indices) = cache.selection(cache_affinity_key) {
                return CacheAffinitySelectionLookup::Hit(indices);
            }
        }

        let indices = Arc::new(select_candidate_indices(
            request,
            &cache.ring,
            cache_affinity_key,
            selection_count,
            config,
        ));
        CacheAffinitySelectionLookup::Computed(indices)
    }

    #[cfg(test)]
    pub(super) fn cached_key_bytes(&self) -> usize {
        let cache = self.cache.read();
        cache.cached_key_bytes()
    }
}

enum CacheAffinitySelectionLookup {
    Hit(Arc<Vec<usize>>),
    Computed(Arc<Vec<usize>>),
    Stale,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CacheAffinityRingEntry {
    hash: u64,
    cluster_id: String,
    candidate_index: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ExcludedClusterPair {
    first: String,
    second: String,
}

impl ExcludedClusterPair {
    fn new(first: &str, second: &str) -> Self {
        if first <= second {
            Self {
                first: first.to_string(),
                second: second.to_string(),
            }
        } else {
            Self {
                first: second.to_string(),
                second: first.to_string(),
            }
        }
    }

    #[cfg(test)]
    fn key_bytes(&self) -> usize {
        self.first.len() + self.second.len()
    }
}

#[derive(Debug, Default)]
struct CacheAffinityRingCache {
    target: Option<RoutingTargetKey>,
    candidate_cluster_ids: Vec<String>,
    ring: Vec<CacheAffinityRingEntry>,
    selections: HashMap<String, Arc<Vec<usize>>>,
    selection_order: VecDeque<String>,
    // Retry selections depend on both the affinity key and the backend excluded
    // by the previous attempt. Keep them separate from the no-exclusion cache so
    // the hot first-attempt lookup can still borrow just `&str` without building
    // a composite owned key on every request.
    single_excluded_selections: HashMap<String, HashMap<String, Arc<Vec<usize>>>>,
    single_excluded_selection_order: VecDeque<(String, String)>,
    single_excluded_selection_count: usize,
    // Two-exclusion retries happen after two failed attempts. Cache those
    // separately so the common no-exclusion and single-exclusion lookup shapes
    // stay simple while repeated retry keys avoid another affinity-ring walk.
    two_excluded_selections: HashMap<String, HashMap<ExcludedClusterPair, Arc<Vec<usize>>>>,
    two_excluded_selection_order: VecDeque<(String, ExcludedClusterPair)>,
    two_excluded_selection_count: usize,
}

impl CacheAffinityRingCache {
    fn replace(
        &mut self,
        target: &RoutingTargetKey,
        candidates: &[RoutedClusterSnapshot],
        ring: Vec<CacheAffinityRingEntry>,
    ) {
        self.target = Some(target.clone());
        self.candidate_cluster_ids = candidates
            .iter()
            .map(|candidate| candidate.cluster_id.clone())
            .collect();
        self.ring = ring;
        self.selections.clear();
        self.selection_order.clear();
        self.single_excluded_selections.clear();
        self.single_excluded_selection_order.clear();
        self.single_excluded_selection_count = 0;
        self.two_excluded_selections.clear();
        self.two_excluded_selection_order.clear();
        self.two_excluded_selection_count = 0;
    }

    fn matches(&self, target: &RoutingTargetKey, candidates: &[RoutedClusterSnapshot]) -> bool {
        self.target.as_ref() == Some(target)
            && self.candidate_cluster_ids.len() == candidates.len()
            && self
                .candidate_cluster_ids
                .iter()
                .zip(candidates)
                .all(|(cached, candidate)| cached == &candidate.cluster_id)
    }

    fn selection(&self, cache_affinity_key: &str) -> Option<Arc<Vec<usize>>> {
        self.selections.get(cache_affinity_key).cloned()
    }

    fn single_excluded_selection(
        &self,
        cache_affinity_key: &str,
        excluded_cluster_id: &str,
    ) -> Option<Arc<Vec<usize>>> {
        self.single_excluded_selections
            .get(cache_affinity_key)
            .and_then(|by_excluded| by_excluded.get(excluded_cluster_id))
            .cloned()
    }

    fn two_excluded_selection(
        &self,
        cache_affinity_key: &str,
        first_excluded: &str,
        second_excluded: &str,
    ) -> Option<Arc<Vec<usize>>> {
        let excluded_pair = ExcludedClusterPair::new(first_excluded, second_excluded);
        self.two_excluded_selections
            .get(cache_affinity_key)
            .and_then(|by_excluded| by_excluded.get(&excluded_pair))
            .cloned()
    }

    fn insert_selection(
        &mut self,
        cache_affinity_key: &str,
        single_excluded_cluster_id: Option<&str>,
        double_excluded_cluster_ids: Option<(&str, &str)>,
        selected_indices: Arc<Vec<usize>>,
    ) {
        if let Some(excluded_cluster_id) = single_excluded_cluster_id {
            self.insert_single_excluded_selection(
                cache_affinity_key,
                excluded_cluster_id,
                selected_indices,
            );
            return;
        }
        if let Some((first_excluded, second_excluded)) = double_excluded_cluster_ids {
            self.insert_two_excluded_selection(
                cache_affinity_key,
                first_excluded,
                second_excluded,
                selected_indices,
            );
            return;
        }

        if let Some(existing) = self.selections.get_mut(cache_affinity_key) {
            *existing = selected_indices;
            return;
        }

        self.evict_selection_entries();
        self.selection_order
            .push_back(cache_affinity_key.to_string());
        self.selections
            .insert(cache_affinity_key.to_string(), selected_indices);
    }

    fn insert_single_excluded_selection(
        &mut self,
        cache_affinity_key: &str,
        excluded_cluster_id: &str,
        selected_indices: Arc<Vec<usize>>,
    ) {
        if let Some(by_excluded) = self.single_excluded_selections.get_mut(cache_affinity_key)
            && let Some(existing) = by_excluded.get_mut(excluded_cluster_id)
        {
            *existing = selected_indices;
            return;
        }

        self.evict_selection_entries();
        self.single_excluded_selections
            .entry(cache_affinity_key.to_string())
            .or_default()
            .insert(excluded_cluster_id.to_string(), selected_indices);
        self.single_excluded_selection_order.push_back((
            cache_affinity_key.to_string(),
            excluded_cluster_id.to_string(),
        ));
        self.single_excluded_selection_count += 1;
    }

    fn insert_two_excluded_selection(
        &mut self,
        cache_affinity_key: &str,
        first_excluded: &str,
        second_excluded: &str,
        selected_indices: Arc<Vec<usize>>,
    ) {
        let excluded_pair = ExcludedClusterPair::new(first_excluded, second_excluded);
        if let Some(by_excluded) = self.two_excluded_selections.get_mut(cache_affinity_key)
            && let Some(existing) = by_excluded.get_mut(&excluded_pair)
        {
            *existing = selected_indices;
            return;
        }

        self.evict_selection_entries();
        self.two_excluded_selections
            .entry(cache_affinity_key.to_string())
            .or_default()
            .insert(excluded_pair.clone(), selected_indices);
        self.two_excluded_selection_order
            .push_back((cache_affinity_key.to_string(), excluded_pair));
        self.two_excluded_selection_count += 1;
    }

    fn evict_selection_entries(&mut self) {
        // All affinity-selection caches share the same entry budget. On
        // pressure, evict retry-specific entries first so normal affinity hits
        // keep the same behavior they had before retry caching existed.
        while self.selection_count() >= SELECTION_CACHE_LIMIT {
            if let Some((cache_affinity_key, excluded_pair)) =
                self.two_excluded_selection_order.pop_front()
            {
                self.remove_two_excluded_selection(&cache_affinity_key, &excluded_pair);
                continue;
            }

            if let Some((cache_affinity_key, excluded_cluster_id)) =
                self.single_excluded_selection_order.pop_front()
            {
                self.remove_single_excluded_selection(&cache_affinity_key, &excluded_cluster_id);
                continue;
            }

            let Some(oldest) = self.selection_order.pop_front() else {
                break;
            };
            self.selections.remove(&oldest);
        }
    }

    fn remove_single_excluded_selection(
        &mut self,
        cache_affinity_key: &str,
        excluded_cluster_id: &str,
    ) {
        let Some(by_excluded) = self.single_excluded_selections.get_mut(cache_affinity_key) else {
            return;
        };
        if by_excluded.remove(excluded_cluster_id).is_some() {
            self.single_excluded_selection_count -= 1;
        }
        if by_excluded.is_empty() {
            self.single_excluded_selections.remove(cache_affinity_key);
        }
    }

    fn remove_two_excluded_selection(
        &mut self,
        cache_affinity_key: &str,
        excluded_pair: &ExcludedClusterPair,
    ) {
        let Some(by_excluded) = self.two_excluded_selections.get_mut(cache_affinity_key) else {
            return;
        };
        if by_excluded.remove(excluded_pair).is_some() {
            self.two_excluded_selection_count -= 1;
        }
        if by_excluded.is_empty() {
            self.two_excluded_selections.remove(cache_affinity_key);
        }
    }

    fn selection_count(&self) -> usize {
        self.selections.len()
            + self.single_excluded_selection_count
            + self.two_excluded_selection_count
    }

    #[cfg(test)]
    fn cached_key_bytes(&self) -> usize {
        let plain_key_bytes = self.selections.keys().map(String::len).sum::<usize>();
        let single_excluded_key_bytes = self
            .single_excluded_selections
            .iter()
            .map(|(cache_affinity_key, by_excluded)| {
                cache_affinity_key.len() * by_excluded.len()
                    + by_excluded.keys().map(String::len).sum::<usize>()
            })
            .sum::<usize>();
        let two_excluded_key_bytes = self
            .two_excluded_selections
            .iter()
            .map(|(cache_affinity_key, by_excluded)| {
                cache_affinity_key.len() * by_excluded.len()
                    + by_excluded
                        .keys()
                        .map(ExcludedClusterPair::key_bytes)
                        .sum::<usize>()
            })
            .sum::<usize>();
        plain_key_bytes + single_excluded_key_bytes + two_excluded_key_bytes
    }
}

#[cfg(test)]
pub(in crate::load_balancer) fn cache_affinity_candidate_indices(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
) -> Option<Vec<usize>> {
    let cache_affinity_key = request.cache_affinity_key?;
    let selection_count = config.cache_affinity_backend_selection_count()?;
    if candidates.is_empty() {
        return None;
    }

    let selection_count = selection_count.min(candidates.len());
    let ring = build_ring(config, request, candidates);
    let selected_indices =
        select_candidate_indices(request, &ring, cache_affinity_key, selection_count, config);

    (!selected_indices.is_empty()).then_some(selected_indices)
}

#[cfg(test)]
pub(in crate::load_balancer) fn cache_affinity_candidates(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
) -> Option<Vec<RoutedClusterSnapshot>> {
    cache_affinity_candidate_indices(config, request, candidates).map(|selected_indices| {
        selected_indices
            .into_iter()
            .map(|index| candidates[index].clone())
            .collect()
    })
}

fn build_ring(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
) -> Vec<CacheAffinityRingEntry> {
    let mut ring = Vec::with_capacity(candidates.len() * config.cache_affinity_virtual_nodes());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for virtual_node in 0..config.cache_affinity_virtual_nodes() {
            ring.push(CacheAffinityRingEntry {
                hash: cache_affinity_virtual_node_hash(config, request, candidate, virtual_node),
                cluster_id: candidate.cluster_id.clone(),
                candidate_index,
            });
        }
    }
    ring.sort_unstable_by(|a, b| {
        a.hash
            .cmp(&b.hash)
            .then_with(|| a.cluster_id.cmp(&b.cluster_id))
            .then_with(|| a.candidate_index.cmp(&b.candidate_index))
    });
    ring
}

fn select_candidate_indices(
    request: &LoadBalancerRequest<'_>,
    ring: &[CacheAffinityRingEntry],
    cache_affinity_key: &str,
    selection_count: usize,
    config: &GroqMultiregionConfig,
) -> Vec<usize> {
    if ring.is_empty() {
        return Vec::new();
    }
    let key_hash = cache_affinity_key_hash(config, request, cache_affinity_key);
    let start_index = ring
        .binary_search_by(|entry| entry.hash.cmp(&key_hash))
        .unwrap_or_else(|index| index);
    if let Some(excluded_cluster_id) = single_excluded_cluster_id(request) {
        // Affinity retries normally exclude the single backend that failed the
        // prior attempt. Keep the ring walk identical, but compare against that
        // borrowed id directly instead of paying for a HashSet lookup at each
        // virtual node.
        return select_candidate_indices_from_ring(
            ring,
            start_index,
            selection_count,
            |cluster_id| cluster_id == excluded_cluster_id,
        );
    }
    if let Some((first_excluded, second_excluded)) = double_excluded_cluster_ids(request) {
        // The third attempt excludes the two earlier failed clusters. The pair
        // case is common enough to avoid a HashSet probe during the affinity
        // ring walk, but still small enough to keep as direct borrowed compares.
        return select_candidate_indices_from_ring(
            ring,
            start_index,
            selection_count,
            |cluster_id| cluster_id == first_excluded || cluster_id == second_excluded,
        );
    }

    select_candidate_indices_from_ring(ring, start_index, selection_count, |cluster_id| {
        request.excludes_cluster(cluster_id)
    })
}

fn select_candidate_indices_from_ring(
    ring: &[CacheAffinityRingEntry],
    start_index: usize,
    selection_count: usize,
    mut excludes_cluster: impl FnMut(&str) -> bool,
) -> Vec<usize> {
    let mut selected_indices = Vec::with_capacity(selection_count);
    let mut selected_cluster_ids = Vec::with_capacity(selection_count);
    for offset in 0..ring.len() {
        let entry = &ring[(start_index + offset) % ring.len()];
        if excludes_cluster(&entry.cluster_id) {
            continue;
        }
        if selected_cluster_ids
            .iter()
            .all(|cluster_id| *cluster_id != entry.cluster_id.as_str())
        {
            selected_cluster_ids.push(entry.cluster_id.as_str());
            selected_indices.push(entry.candidate_index);
            if selected_indices.len() >= selection_count {
                break;
            }
        }
    }

    selected_indices
}

const HASH_VERSION: u8 = 1;

fn cache_affinity_key_hash(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    cache_affinity_key: &str,
) -> u64 {
    let mut bytes = HashInputBuilder::new();
    append_ring_prefix(&mut bytes, config, request);
    bytes.append_tagged_bytes(b"cache_affinity_key", cache_affinity_key.as_bytes());
    xxh3_64(bytes.as_slice())
}

pub(in crate::load_balancer) fn cache_affinity_virtual_node_hash(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidate: &RoutedClusterSnapshot,
    virtual_node: usize,
) -> u64 {
    let mut bytes = HashInputBuilder::new();
    append_ring_prefix(&mut bytes, config, request);
    bytes.append_tagged_bytes(b"cluster_id", candidate.cluster_id.as_bytes());
    bytes.append_tagged_bytes(b"virtual_node", &virtual_node.to_le_bytes());
    xxh3_64(bytes.as_slice())
}

fn append_ring_prefix(
    bytes: &mut HashInputBuilder,
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
) {
    bytes.push(HASH_VERSION);
    bytes.append_tagged_bytes(b"seed", config.seed().unwrap_or("").as_bytes());
    bytes.append_tagged_bytes(
        b"routing_key",
        request
            .routing_target
            .routing_key
            .as_deref()
            .unwrap_or("")
            .as_bytes(),
    );
    bytes.append_tagged_bytes(b"model_id", request.routing_target.model_id.as_bytes());
}
