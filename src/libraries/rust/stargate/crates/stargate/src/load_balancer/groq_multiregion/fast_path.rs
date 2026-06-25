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

use rand::Rng;

use crate::load_balancer::{LoadBalancerCandidateChoice, LoadBalancerRequest};
use crate::routing_state::RoutedClusterSnapshot;

use super::estimates::{
    compare_least_queue_time, estimate_queue_comparison, has_capacity, queue_ignored_ttft_ms,
    rtt_ms,
};
use super::{
    GroqMultiregionConfig, choice_for_candidate, choose_less_queued_candidate, shuffle_prefix,
    single_excluded_cluster_id,
};

pub(super) fn choose_from_queue_ignored_single_bucket(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if !config.ignore_queue_time() || config.ignore_input_processing_time() {
        return None;
    }
    if config.max_queue_time(request).is_some() {
        return None;
    }

    match request.excluded_cluster_ids {
        Some(excluded) if excluded.is_empty() => {
            choose_from_queue_ignored_single_bucket_filtered(config, request, candidates, |_| false)
        }
        None => {
            choose_from_queue_ignored_single_bucket_filtered(config, request, candidates, |_| false)
        }
        Some(excluded) if excluded.len() == 1 => {
            let excluded_cluster_id = single_excluded_cluster_id(request)?;
            choose_from_queue_ignored_single_bucket_filtered(
                config,
                request,
                candidates,
                |candidate| candidate.cluster_id == excluded_cluster_id,
            )
        }
        Some(excluded) if excluded.len() == 2 => {
            let mut excluded_ids = excluded.iter().map(String::as_str);
            let first_excluded = excluded_ids.next()?;
            let second_excluded = excluded_ids.next()?;
            choose_from_queue_ignored_single_bucket_filtered(
                config,
                request,
                candidates,
                |candidate| {
                    let cluster_id = candidate.cluster_id.as_str();
                    cluster_id == first_excluded || cluster_id == second_excluded
                },
            )
        }
        // Larger retry exclusion sets are uncommon and need more filtering
        // work per candidate. Keep them on the general path instead of
        // making the steady-state ignore-queue fast path carry a HashSet probe.
        Some(_) => None,
    }
}

fn choose_from_queue_ignored_single_bucket_filtered(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
    mut excludes_candidate: impl FnMut(&RoutedClusterSnapshot) -> bool,
) -> Option<LoadBalancerCandidateChoice> {
    let input_tokens = request.input_tokens.unwrap_or(0) as f64;
    let mut fastest_ttft = f64::INFINITY;
    let mut slowest_ttft = f64::NEG_INFINITY;
    for candidate in candidates {
        if excludes_candidate(candidate) {
            continue;
        }
        let ttft_ms = queue_ignored_ttft_ms(candidate, input_tokens);
        if !ttft_ms.is_finite() {
            return None;
        }
        fastest_ttft = fastest_ttft.min(ttft_ms);
        slowest_ttft = slowest_ttft.max(ttft_ms);
    }

    let bucket_size_ms = config.ttft_bucket_size().as_secs_f64() * 1000.0;
    if !fastest_ttft.is_finite() || slowest_ttft - fastest_ttft > bucket_size_ms {
        return None;
    }

    let max_queued = config.max_queued();
    let mut unlocked_with_capacity = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if excludes_candidate(candidate) {
            continue;
        }
        if has_capacity(candidate, max_queued) {
            unlocked_with_capacity.push(candidate);
        }
    }

    // Queue is intentionally ignored for TTFT bucket construction in this
    // config, but it is still the primary sampled-candidate comparator. Keep
    // queue estimation to the sample instead of paying it for every backend.
    choose_from_unlocked_candidate_refs(config, request, unlocked_with_capacity, candidates)
}

pub(super) fn choose_from_rtt_only_single_bucket(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if !config.ignore_queue_time() || !config.ignore_input_processing_time() {
        return None;
    }
    if config.max_queue_time(request).is_some() {
        return None;
    }

    match request.excluded_cluster_ids {
        Some(excluded) if excluded.is_empty() => {
            choose_from_rtt_only_single_bucket_filtered(config, request, candidates, |_| false)
        }
        None => choose_from_rtt_only_single_bucket_filtered(config, request, candidates, |_| false),
        Some(excluded) if excluded.len() == 1 => {
            let excluded_cluster_id = single_excluded_cluster_id(request)?;
            choose_from_rtt_only_single_bucket_filtered(config, request, candidates, |candidate| {
                candidate.cluster_id == excluded_cluster_id
            })
        }
        Some(excluded) if excluded.len() == 2 => {
            let mut excluded_ids = excluded.iter().map(String::as_str);
            let first_excluded = excluded_ids.next()?;
            let second_excluded = excluded_ids.next()?;
            choose_from_rtt_only_single_bucket_filtered(config, request, candidates, |candidate| {
                let cluster_id = candidate.cluster_id.as_str();
                cluster_id == first_excluded || cluster_id == second_excluded
            })
        }
        // Larger retry exclusion sets are uncommon and need more filtering
        // work per candidate. Keep them on the general path instead of
        // making the steady-state RTT-only fast path carry a HashSet probe.
        Some(_) => None,
    }
}

fn choose_from_rtt_only_single_bucket_filtered(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    candidates: &[RoutedClusterSnapshot],
    mut excludes_candidate: impl FnMut(&RoutedClusterSnapshot) -> bool,
) -> Option<LoadBalancerCandidateChoice> {
    let mut fastest_ttft = f64::INFINITY;
    let mut slowest_ttft = f64::NEG_INFINITY;
    for candidate in candidates {
        if excludes_candidate(candidate) {
            continue;
        }
        let ttft_ms = rtt_ms(candidate);
        fastest_ttft = fastest_ttft.min(ttft_ms);
        slowest_ttft = slowest_ttft.max(ttft_ms);
    }

    let bucket_size_ms = config.ttft_bucket_size().as_secs_f64() * 1000.0;
    if !fastest_ttft.is_finite() || slowest_ttft - fastest_ttft > bucket_size_ms {
        return None;
    }

    let max_queued = config.max_queued();
    let mut unlocked_with_capacity = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if excludes_candidate(candidate) {
            continue;
        }
        if has_capacity(candidate, max_queued) {
            unlocked_with_capacity.push(candidate);
        }
    }

    // When both non-RTT TTFT components are ignored and every candidate is
    // already in the first bucket, routing only needs queue estimates for
    // the sampled candidates. Computing sparse priority queue estimates for
    // all non-excluded backends would preserve correctness but wastes work
    // on the common wide-bucket, n=2 deployment shape.
    choose_from_unlocked_candidate_refs(config, request, unlocked_with_capacity, candidates)
}

fn choose_from_unlocked_candidate_refs(
    config: &GroqMultiregionConfig,
    request: &LoadBalancerRequest<'_>,
    mut unlocked_with_capacity: Vec<&RoutedClusterSnapshot>,
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if unlocked_with_capacity.is_empty() {
        return None;
    }

    let sample_count = config.sample_count();
    if sample_count == 1 {
        let selected_index = rand::rng().random_range(0..unlocked_with_capacity.len());
        return Some(choice_for_candidate(
            candidates,
            unlocked_with_capacity[selected_index],
            1,
        ));
    }
    if sample_count == 2 {
        return choose_two_rtt_only_candidates(
            &unlocked_with_capacity,
            request.priority,
            candidates,
        );
    }

    let sampled_count = sample_count.min(unlocked_with_capacity.len());
    shuffle_prefix(&mut unlocked_with_capacity, sampled_count);
    unlocked_with_capacity
        .into_iter()
        .take(sampled_count)
        .map(|candidate| {
            (
                candidate,
                estimate_queue_comparison(candidate, request.priority),
            )
        })
        .min_by(|(candidate_a, estimate_a), (candidate_b, estimate_b)| {
            compare_least_queue_time(candidate_a, estimate_a, candidate_b, estimate_b)
        })
        .map(|(candidate, _)| choice_for_candidate(candidates, candidate, 1))
}

fn choose_two_rtt_only_candidates(
    unlocked_with_capacity: &[&RoutedClusterSnapshot],
    priority: u32,
    candidates: &[RoutedClusterSnapshot],
) -> Option<LoadBalancerCandidateChoice> {
    if unlocked_with_capacity.len() < 2 {
        if unlocked_with_capacity.is_empty() {
            return None;
        }
        let selected_index = rand::rng().random_range(0..unlocked_with_capacity.len());
        return Some(choice_for_candidate(
            candidates,
            unlocked_with_capacity[selected_index],
            1,
        ));
    }

    let mut rng = rand::rng();
    let candidate_a_index = rng.random_range(0..unlocked_with_capacity.len());
    let mut candidate_b_index = rng.random_range(0..unlocked_with_capacity.len() - 1);
    if candidate_b_index >= candidate_a_index {
        candidate_b_index += 1;
    }

    let candidate_a = unlocked_with_capacity[candidate_a_index];
    let candidate_b = unlocked_with_capacity[candidate_b_index];
    let estimate_a = estimate_queue_comparison(candidate_a, priority);
    let estimate_b = estimate_queue_comparison(candidate_b, priority);
    let candidate =
        choose_less_queued_candidate(candidate_a, &estimate_a, candidate_b, &estimate_b, &mut rng);
    Some(choice_for_candidate(candidates, candidate, 1))
}
