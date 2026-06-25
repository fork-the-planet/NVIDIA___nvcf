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

mod cache_affinity;
mod estimates;
mod fast_path;

use std::cmp::Ordering;
use std::fmt;
use std::time::Duration;

use rand::Rng;

use super::{
    GroqMultiregionAlgorithmConfig, LoadBalancer, LoadBalancerAlgorithmConfig,
    LoadBalancerCandidateChoice, LoadBalancerRequest,
};
use crate::routing_state::RoutedClusterSnapshot;
use cache_affinity::CacheAffinitySelector;
use estimates::{
    CandidateEstimateAccumulator, TtftEstimate, compare_least_queue_time, estimate_ttft_ms,
    has_capacity,
};

#[cfg(test)]
pub(super) use cache_affinity::{
    cache_affinity_candidate_indices, cache_affinity_candidates, cache_affinity_virtual_node_hash,
};

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct GroqMultiregionTtftComponents {
    pub(super) queue_ms: f64,
    pub(super) ttft_ms: f64,
}

#[cfg(test)]
pub(super) fn groq_multiregion_ttft_components(
    candidate: &RoutedClusterSnapshot,
    input_tokens: Option<u64>,
    priority: u32,
    ignore_queue_time: bool,
    ignore_input_processing_time: bool,
) -> GroqMultiregionTtftComponents {
    let estimate = estimate_ttft_ms(
        candidate,
        input_tokens,
        priority,
        ignore_queue_time,
        ignore_input_processing_time,
    );
    GroqMultiregionTtftComponents {
        queue_ms: estimate.queue_ms,
        ttft_ms: estimate.ttft_ms,
    }
}

// Parity notes vs lpu-router MultiRegion:
// - Stargate uses only the sticky last_mean_input_tps capacity signal for
//   queue/prefill estimates; there is no separate live input-TPS fallback.
// - Sparse priority queue maps use the nearest published priority at or below the request priority;
//   lpu-router clamps only priorities above the max and otherwise treats missing entries as zero.
// - Stargate does not currently model lpu-router batch-folding queue estimates, LoRA dynamic-model
//   penalties, backend/datacenter request filters, backend-id overrides, or utilization-max rejection.
pub(super) struct GroqMultiregionLoadBalancer {
    config: GroqMultiregionConfig,
    cache_affinity: CacheAffinitySelector,
}

#[derive(Clone, Debug)]
pub(super) struct GroqMultiregionConfig {
    seed: Option<String>,
    cache_affinity_virtual_nodes: usize,
    cache_affinity_backend_selection_count: Option<usize>,
    queue_slo: Option<QueueSloConfig>,
    ttft_bucket_size: Duration,
    next_bucket_unlock_factor: f64,
    sample_count: usize,
    max_queued: u64,
    ignore_queue_time: bool,
    ignore_input_processing_time: bool,
}

#[derive(Clone, Debug)]
struct QueueSloConfig {
    floor: Duration,
    ceil: Duration,
}

impl GroqMultiregionLoadBalancer {
    pub(super) fn new(config: GroqMultiregionConfig) -> Self {
        Self {
            config,
            cache_affinity: CacheAffinitySelector::default(),
        }
    }

    pub(super) fn has_queue_slo(&self, request: &LoadBalancerRequest<'_>) -> bool {
        self.config.max_queue_time(request).is_some()
    }

    #[cfg(test)]
    pub(super) fn cached_affinity_key_bytes(&self) -> usize {
        self.cache_affinity.cached_key_bytes()
    }
}

impl GroqMultiregionConfig {
    pub(super) fn from_algorithm_config(config: &LoadBalancerAlgorithmConfig) -> Self {
        let config = config
            .multiregion_settings()
            .expect("multiregion settings should match load-balancer algorithm");
        Self::from_settings(config)
    }

    pub(super) fn from_settings(config: &GroqMultiregionAlgorithmConfig) -> Self {
        let queue_slo = match (
            config.max_queue_time_floor_ms,
            config.max_queue_time_ceil_ms,
        ) {
            (Some(floor_ms), Some(ceil_ms)) => Some(QueueSloConfig {
                floor: Duration::from_millis(floor_ms),
                ceil: Duration::from_millis(ceil_ms),
            }),
            _ => None,
        };

        // Zero virtual nodes would make affinity routing degenerate; keep the historical minimum.
        let cache_affinity_virtual_nodes =
            config.cache_affinity_virtual_nodes.unwrap_or(150).max(1);
        // Sampling at least one backend keeps the algorithm meaningful even when config sets n=0.
        let sample_count = config.n.unwrap_or(2).max(1);

        Self {
            seed: config.seed.clone(),
            cache_affinity_virtual_nodes,
            cache_affinity_backend_selection_count: config
                .cache_affinity_backend_selection_count
                .filter(|count| *count > 0),
            queue_slo,
            ttft_bucket_size: Duration::from_millis(config.ttft_bucket_size_ms.unwrap_or(20)),
            next_bucket_unlock_factor: config.next_bucket_unlock_factor.unwrap_or(0.25),
            sample_count,
            max_queued: config.max_queued.unwrap_or(0),
            ignore_queue_time: config.ignore_queue_time.unwrap_or(false),
            ignore_input_processing_time: config.ignore_input_processing_time.unwrap_or(false),
        }
    }

    pub(super) fn cache_affinity_virtual_nodes(&self) -> usize {
        self.cache_affinity_virtual_nodes
    }

    pub(super) fn cache_affinity_backend_selection_count(&self) -> Option<usize> {
        self.cache_affinity_backend_selection_count
    }

    pub(super) fn max_queue_time(&self, request: &LoadBalancerRequest<'_>) -> Option<Duration> {
        let queue_slo = self.queue_slo.as_ref()?;

        let floor_ms = queue_slo.floor.as_secs_f64() * 1000.0;
        let ceil_ms = queue_slo.ceil.as_secs_f64() * 1000.0;
        let slo_elapsed_percentage = request_slo_elapsed_percentage(request);
        let max_queue_time_ms = floor_ms + (ceil_ms - floor_ms) * slo_elapsed_percentage;
        Some(Duration::from_secs_f64(max_queue_time_ms / 1000.0))
    }

    pub(super) fn ttft_bucket_size(&self) -> Duration {
        self.ttft_bucket_size
    }

    pub(super) fn next_bucket_unlock_factor(&self) -> f64 {
        self.next_bucket_unlock_factor
    }

    pub(super) fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub(super) fn max_queued(&self) -> u64 {
        self.max_queued
    }

    pub(super) fn ignore_queue_time(&self) -> bool {
        self.ignore_queue_time
    }

    pub(super) fn ignore_input_processing_time(&self) -> bool {
        self.ignore_input_processing_time
    }

    fn seed(&self) -> Option<&str> {
        self.seed.as_deref()
    }
}

fn request_slo_elapsed_percentage(request: &LoadBalancerRequest<'_>) -> f64 {
    let Some(request_slo) = request.request_slo else {
        return 1.0;
    };
    if request_slo.is_zero() {
        return 1.0;
    }

    (request.received_at.elapsed().as_secs_f64() / request_slo.as_secs_f64()).clamp(0.0, 1.0)
}

impl fmt::Display for GroqMultiregionLoadBalancer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "groq-multiregion")
    }
}

impl LoadBalancer for GroqMultiregionLoadBalancer {
    fn choose_candidate(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        if let Some(affinity_indices) =
            self.cache_affinity
                .candidate_indices(&self.config, request, candidates)
            && let Some(choice) =
                self.choose_from_candidate_indices(request, candidates, affinity_indices.as_slice())
        {
            return Some(choice);
        }

        self.choose_from_candidates(request, candidates)
    }
}

impl GroqMultiregionLoadBalancer {
    fn choose_from_candidates(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        if candidates.is_empty() {
            return None;
        }
        if let Some(choice) =
            fast_path::choose_from_rtt_only_single_bucket(&self.config, request, candidates)
        {
            return Some(choice);
        }
        if let Some(choice) =
            fast_path::choose_from_queue_ignored_single_bucket(&self.config, request, candidates)
        {
            return Some(choice);
        }
        self.choose_from_candidate_iter(request, candidates, candidates.len(), candidates)
    }

    pub(super) fn choose_from_candidate_indices(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        candidate_indices: &[usize],
    ) -> Option<LoadBalancerCandidateChoice> {
        if candidate_indices.is_empty() {
            return None;
        }
        if let Some(choice) =
            self.choose_from_two_ready_affinity_candidates(request, candidates, candidate_indices)
        {
            return Some(choice);
        }

        // Cache-affinity selection stores indices into the current candidate
        // slice. Routing over references avoids cloning snapshots into a
        // temporary Vec for every affinity hit.
        self.choose_from_candidate_iter(
            request,
            candidate_indices.iter().map(|index| &candidates[*index]),
            candidate_indices.len(),
            candidates,
        )
    }

    fn choose_from_two_ready_affinity_candidates(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        candidate_indices: &[usize],
    ) -> Option<LoadBalancerCandidateChoice> {
        if candidate_indices.len() != 2 || self.config.sample_count() < 2 {
            return None;
        }
        if self.config.max_queue_time(request).is_some() {
            return None;
        }

        let candidate_a = &candidates[candidate_indices[0]];
        let candidate_b = &candidates[candidate_indices[1]];
        if request.excludes_cluster(&candidate_a.cluster_id)
            || request.excludes_cluster(&candidate_b.cluster_id)
        {
            return None;
        }

        let max_queued = self.config.max_queued();
        if !has_capacity(candidate_a, max_queued) || !has_capacity(candidate_b, max_queued) {
            return None;
        }

        let estimate_a = estimate_ttft_ms(
            candidate_a,
            request.input_tokens,
            request.priority,
            self.config.ignore_queue_time(),
            self.config.ignore_input_processing_time(),
        );
        let estimate_b = estimate_ttft_ms(
            candidate_b,
            request.input_tokens,
            request.priority,
            self.config.ignore_queue_time(),
            self.config.ignore_input_processing_time(),
        );
        if !estimate_a.ttft_ms.is_finite() || !estimate_b.ttft_ms.is_finite() {
            return None;
        }

        let bucket_size_ms = self.config.ttft_bucket_size().as_secs_f64() * 1000.0;
        if (estimate_a.ttft_ms - estimate_b.ttft_ms).abs() > bucket_size_ms {
            return None;
        }

        // With exactly two affinity-selected candidates and n >= 2, the normal
        // shuffle samples both candidates. Equal candidates used to be decided
        // by the shuffled order, so keep that random tie-break explicitly while
        // avoiding the allocation and prefix-shuffle overhead on cache hits.
        let mut rng = rand::rng();
        let candidate = choose_less_queued_candidate(
            candidate_a,
            &estimate_a,
            candidate_b,
            &estimate_b,
            &mut rng,
        );
        let candidate_index = if std::ptr::eq(candidate, candidate_a) {
            candidate_indices[0]
        } else {
            candidate_indices[1]
        };
        Some(LoadBalancerCandidateChoice::with_rank_depth_1(
            candidate_index,
        ))
    }

    fn choose_from_candidate_iter<'a>(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: impl IntoIterator<Item = &'a RoutedClusterSnapshot>,
        candidate_capacity: usize,
        candidate_index_source: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        let candidates = candidates.into_iter();

        let mut estimates =
            CandidateEstimateAccumulator::new(&self.config, request, candidate_capacity);
        if !estimates.filters_by_queue_slo() && !request.has_excluded_clusters() {
            // This is the steady-state proxy path: first-attempt routing with no
            // queue SLO. Keep it separate so every candidate does not pay for
            // retry exclusion and queue-SLO option checks.
            for candidate in candidates {
                estimates.push_estimate(candidate);
            }
        } else if !estimates.filters_by_queue_slo() {
            if let Some(excluded_cluster_id) = single_excluded_cluster_id(request) {
                // Most retries exclude exactly the backend that failed the prior
                // attempt. Compare against that one borrowed id directly instead
                // of doing a HashSet lookup for every candidate. Queue-SLO work is
                // still skipped because this branch is only for configs without a
                // queue SLO.
                for candidate in candidates {
                    if candidate.cluster_id == excluded_cluster_id {
                        continue;
                    }
                    estimates.push_estimate(candidate);
                }
            } else {
                // Multi-exclusion retries are less common, but they still avoid
                // queue-SLO filtering when the algorithm is configured without a
                // queue SLO. Keep the exact exclusion semantics for this fallback.
                for candidate in candidates {
                    if request.excludes_cluster(&candidate.cluster_id) {
                        continue;
                    }
                    estimates.push_estimate(candidate);
                }
            }
        } else {
            for candidate in candidates {
                if request.excludes_cluster(&candidate.cluster_id) {
                    continue;
                }
                estimates.push_estimate(candidate);
            }
        }

        if estimates.is_empty() {
            return None;
        }
        if !estimates.has_finite_fastest_ttft() {
            return None;
        }

        let bucket_size_ms = self.config.ttft_bucket_size().as_secs_f64() * 1000.0;
        if estimates.all_estimates_in_first_bucket(bucket_size_ms) {
            return self.choose_from_unlocked_candidates(
                estimates.into_estimates(),
                candidate_index_source,
            );
        }

        let mut estimated = estimates.into_estimates();
        estimated.sort_unstable_by(|(candidate_a, estimate_a), (candidate_b, estimate_b)| {
            estimate_a
                .ttft_ms
                .total_cmp(&estimate_b.ttft_ms)
                .then_with(|| candidate_a.cluster_id.cmp(&candidate_b.cluster_id))
        });

        let unlock_factor = self.config.next_bucket_unlock_factor();
        let mut slept_for_ms = request.received_at.elapsed().as_secs_f64() * 1000.0;
        let mut prev_bucket_start_ttft = None;
        let mut unlocked = Vec::with_capacity(estimated.len());

        for (candidate, estimate) in estimated {
            if !estimate.ttft_ms.is_finite() {
                break;
            }

            if let Some(prev_ttft) = prev_bucket_start_ttft {
                let gap_ms = estimate.ttft_ms - prev_ttft;
                if gap_ms > bucket_size_ms {
                    let sleep_for_at_least_ms = gap_ms * unlock_factor;
                    if slept_for_ms < sleep_for_at_least_ms {
                        break;
                    }
                    slept_for_ms -= sleep_for_at_least_ms;
                    prev_bucket_start_ttft = Some(estimate.ttft_ms);
                }
            } else {
                prev_bucket_start_ttft = Some(estimate.ttft_ms);
            }

            unlocked.push((candidate, estimate));
        }

        self.choose_from_unlocked_candidates(unlocked, candidate_index_source)
    }

    fn choose_from_unlocked_candidates(
        &self,
        mut unlocked_with_capacity: Vec<(&RoutedClusterSnapshot, TtftEstimate)>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        let max_queued = self.config.max_queued();
        // The caller already owns this candidate buffer. Retaining in place
        // preserves the later shuffle semantics while avoiding a second Vec
        // allocation on every routing decision.
        unlocked_with_capacity.retain(|(candidate, _)| has_capacity(candidate, max_queued));

        if unlocked_with_capacity.is_empty() {
            return None;
        }

        let sample_count = self.config.sample_count();
        if sample_count == 1 {
            return choose_one_unlocked_candidate(&unlocked_with_capacity, candidates);
        }
        if sample_count == 2 {
            return choose_two_unlocked_candidates(&unlocked_with_capacity, candidates);
        }

        let sampled_count = sample_count.min(unlocked_with_capacity.len());
        shuffle_prefix(&mut unlocked_with_capacity, sampled_count);

        unlocked_with_capacity
            .into_iter()
            .take(sampled_count)
            .min_by(|(candidate_a, estimate_a), (candidate_b, estimate_b)| {
                compare_least_queue_time(candidate_a, estimate_a, candidate_b, estimate_b)
            })
            .map(|(candidate, _)| choice_for_candidate(candidates, candidate, 1))
    }
}

fn choose_one_unlocked_candidate(
    unlocked_with_capacity: &[(&RoutedClusterSnapshot, TtftEstimate)],
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if unlocked_with_capacity.is_empty() {
        return None;
    }

    let selected_index = rand::rng().random_range(0..unlocked_with_capacity.len());
    Some(choice_for_candidate(
        candidates,
        unlocked_with_capacity[selected_index].0,
        1,
    ))
}

fn choose_two_unlocked_candidates(
    unlocked_with_capacity: &[(&RoutedClusterSnapshot, TtftEstimate)],
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if unlocked_with_capacity.len() < 2 {
        return choose_one_unlocked_candidate(unlocked_with_capacity, candidates);
    }

    let mut rng = rand::rng();
    let candidate_a_index = rng.random_range(0..unlocked_with_capacity.len());
    let mut candidate_b_index = rng.random_range(0..unlocked_with_capacity.len() - 1);
    if candidate_b_index >= candidate_a_index {
        candidate_b_index += 1;
    }

    // This is equivalent to shuffling a two-element prefix: it samples two
    // distinct unlocked candidates uniformly, then applies the same least-queue
    // comparison with an explicit random tie-break for candidates that the old
    // shuffled order would have treated as equal.
    let (candidate_a, estimate_a) = unlocked_with_capacity[candidate_a_index];
    let (candidate_b, estimate_b) = unlocked_with_capacity[candidate_b_index];
    let candidate =
        choose_less_queued_candidate(candidate_a, &estimate_a, candidate_b, &estimate_b, &mut rng);
    Some(choice_for_candidate(candidates, candidate, 1))
}

fn choice_for_candidate(
    candidates: &[RoutedClusterSnapshot],
    selected: &RoutedClusterSnapshot,
    rank_depth: usize,
) -> LoadBalancerCandidateChoice {
    let base = candidates.as_ptr() as usize;
    let selected_ptr = std::ptr::from_ref(selected) as usize;
    let stride = std::mem::size_of::<RoutedClusterSnapshot>();
    let byte_offset = selected_ptr
        .checked_sub(base)
        .expect("selected candidate should come from candidate slice");
    assert_eq!(
        byte_offset % stride,
        0,
        "selected candidate should align with candidate slice"
    );
    let candidate_index = byte_offset / stride;
    assert!(
        candidate_index < candidates.len(),
        "selected candidate should come from candidate slice"
    );
    LoadBalancerCandidateChoice {
        candidate_index,
        rank_depth,
        selected_after_kv_free_tokens_skip: false,
    }
}

fn choose_less_queued_candidate<'a>(
    candidate_a: &'a RoutedClusterSnapshot,
    estimate_a: &TtftEstimate,
    candidate_b: &'a RoutedClusterSnapshot,
    estimate_b: &TtftEstimate,
    rng: &mut impl Rng,
) -> &'a RoutedClusterSnapshot {
    match compare_least_queue_time(candidate_a, estimate_a, candidate_b, estimate_b) {
        Ordering::Less => candidate_a,
        Ordering::Equal if rng.random_bool(0.5) => candidate_a,
        Ordering::Equal | Ordering::Greater => candidate_b,
    }
}

fn single_excluded_cluster_id<'a>(request: &LoadBalancerRequest<'a>) -> Option<&'a str> {
    let excluded = request.excluded_cluster_ids?;
    if excluded.len() == 1 {
        excluded.iter().next().map(String::as_str)
    } else {
        None
    }
}

fn double_excluded_cluster_ids<'a>(
    request: &LoadBalancerRequest<'a>,
) -> Option<(&'a str, &'a str)> {
    let excluded = request.excluded_cluster_ids?;
    if excluded.len() != 2 {
        return None;
    }
    let mut excluded_ids = excluded.iter().map(String::as_str);
    let first = excluded_ids.next()?;
    let second = excluded_ids.next()?;
    if first <= second {
        Some((first, second))
    } else {
        Some((second, first))
    }
}

fn shuffle_prefix<T>(items: &mut [T], count: usize) {
    let mut rng = rand::rng();
    for index in 0..count {
        let swap_index = rng.random_range(index..items.len());
        items.swap(index, swap_index);
    }
}
