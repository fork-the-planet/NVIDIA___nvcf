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

mod ranking;

use std::fmt;
use std::sync::Arc;

use super::{
    LoadBalancer, LoadBalancerAlgorithmConfig, LoadBalancerCandidateChoice, LoadBalancerRequest,
};
use crate::routing_state::RoutedClusterSnapshot;
use ranking::{
    PulsarRankingLookup, PulsarRankingStore, PulsarScorer, ScoredCandidate,
    compare_ranked_candidate, has_valid_input_capacity,
};

#[cfg(test)]
pub(super) use ranking::{pulsar_hash64, pulsar_ranked_indices};

pub(super) struct PulsarLoadBalancer {
    config: LoadBalancerAlgorithmConfig,
    rankings: PulsarRankingStore,
}

impl PulsarLoadBalancer {
    pub(super) fn new(config: LoadBalancerAlgorithmConfig) -> Self {
        Self {
            config,
            rankings: PulsarRankingStore::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn cached_affinity_key_bytes(&self) -> usize {
        self.rankings.cached_key_bytes()
    }
}

impl fmt::Display for PulsarLoadBalancer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pulsar")
    }
}

impl LoadBalancer for PulsarLoadBalancer {
    fn choose_candidate(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        let (lookup, cached_choice) = self.ranking_lookup(request, candidates)?;
        match lookup {
            PulsarRankingLookup::Hit => cached_choice,
            PulsarRankingLookup::MissCacheable => {
                let ranking = Arc::new(self.compute_ranking(request, candidates));
                if ranking.is_empty() {
                    return None;
                }
                let choice = self.choose_from_ranked_indices(request, candidates, &ranking);
                if let Some(cached_choice) = self.rankings.insert_or_choose_existing(
                    request,
                    candidates,
                    ranking,
                    |cached| self.choose_from_ranked_indices(request, candidates, cached),
                ) {
                    return Some(cached_choice);
                }
                choice
            }
            PulsarRankingLookup::MissBypass => self.choose_by_score_scan(request, candidates),
        }
    }
}

impl PulsarLoadBalancer {
    /*
    PULSAR selection model

    Treat consistent hashing as a ranking generator, not a direct destination selector.
    For one request we score every candidate, sort by descending score, then choose the
    first candidate that is currently feasible.

    Request-specific ranking
    ------------------------
    The cache-affinity key is the stable request identity for KV reuse. A different
    affinity key gets a different deterministic ranking.

        request key K1                     request key K2
        ---------------                    ---------------
        score(A) = 0.91  rank 1            score(A) = 0.77  rank 2
        score(B) = 0.63  rank 2            score(B) = 0.82  rank 1
        score(C) = 0.18  rank 3            score(C) = 0.11  rank 3

    Progressive unlocking
    ---------------------
    We do not send all overflow to "the next node on a ring". We walk each request's own
    rendezvous ranking until we find the first feasible backend.

        K1 ranking: A -> B -> C
        K2 ranking: B -> A -> C
        K3 ranking: A -> C -> B

        if A saturates:
          K1 unlocks to B
          K2 stays on B
          K3 unlocks to C

    That scattering behavior is the whole point. Shared-primary keys do not all collapse
    onto one shared successor.

    ASCII picture
    -------------

        before saturation

          K1 ---> A
          K2 ---> A
          K3 ---> A

        ring-successor overflow would do this

          K1 ---> B
          K2 ---> B
          K3 ---> B

        PULSAR does this instead

          K1 ---> B
          K2 ---> D
          K3 ---> C

    Feasibility invariants
    ----------------------
    Feasibility must depend only on pre-hash information:

      - request headers (`x-cache-affinity-key`, `x-input-tokens`)
      - backend snapshots (`last_mean_input_tps`, KV metrics)
      - router-local admission state in future extensions

    It must not depend on the scores themselves. The ranking answers "in what order should
    I try backends for this key?" Feasibility answers "is this backend safe right now?"

    Current implementation
    ----------------------
    This implementation is the first useful slice, not the full paper:

      - ranking: weighted rendezvous hashing
      - key material: routing target + cache-affinity key + cluster_id + optional seed
      - weight: `last_mean_input_tps`
      - feasibility: retry exclusions, plus an optional reported-free-KV gate
        against request input tokens

    So the control flow is:

        rank all candidates for this request
             |
             v
        candidate[0] feasible? -- yes --> choose it
             |
             no
             v
        candidate[1] feasible? -- yes --> choose it
             |
             no
             v
        ...
             |
             v
           none feasible --> proxy service-unavailable response
    */
    fn choose_from_ranked_indices(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        ranked_indices: &[usize],
    ) -> Option<LoadBalancerCandidateChoice> {
        let mut selected_after_kv_free_tokens_skip = false;
        for (index, candidate_index) in ranked_indices.iter().enumerate() {
            let candidate = &candidates[*candidate_index];
            match self.feasibility(request, candidate) {
                PulsarCandidateFeasibility::Eligible => {
                    return Some(LoadBalancerCandidateChoice {
                        candidate_index: *candidate_index,
                        rank_depth: index + 1,
                        selected_after_kv_free_tokens_skip,
                    });
                }
                reason if reason.skipped_for_kv_free_tokens() => {
                    selected_after_kv_free_tokens_skip = true;
                }
                _ => {}
            }
        }

        None
    }

    fn ranking_lookup(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<(PulsarRankingLookup, Option<LoadBalancerCandidateChoice>)> {
        self.rankings.lookup_choice(request, candidates, |ranking| {
            self.choose_from_ranked_indices(request, candidates, ranking)
        })
    }

    pub(super) fn compute_ranking(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Vec<usize> {
        ranking::pulsar_ranked_indices(self.config.seed(), request, candidates)
    }

    fn choose_by_score_scan(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
    ) -> Option<LoadBalancerCandidateChoice> {
        let mut scorer = PulsarScorer::new(self.config.seed(), request);
        let mut best_overall = None;
        let mut best_feasible = None;
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let Some(score) = scorer.score(candidate) else {
                continue;
            };
            let is_best_overall = best_overall.as_ref().is_none_or(|best: &ScoredCandidate| {
                compare_ranked_candidate(
                    score,
                    candidate,
                    best.score,
                    &candidates[best.candidate_index],
                )
                .is_lt()
            });
            if is_best_overall {
                best_overall = Some(ScoredCandidate {
                    score,
                    candidate_index,
                });
            }

            if !self.feasibility(request, candidate).is_eligible() {
                continue;
            }
            let is_best_feasible = best_feasible.as_ref().is_none_or(|best: &ScoredCandidate| {
                compare_ranked_candidate(
                    score,
                    candidate,
                    best.score,
                    &candidates[best.candidate_index],
                )
                .is_lt()
            });
            if is_best_feasible {
                best_feasible = Some(ScoredCandidate {
                    score,
                    candidate_index,
                });
            }
        }

        let best = best_feasible?;
        let chosen = &candidates[best.candidate_index];
        // The common all-feasible case has rank depth 1 and does not need the
        // second score pass used to preserve rank-depth semantics after fallback.
        let rank_depth = if best_overall
            .as_ref()
            .is_some_and(|overall| overall.candidate_index == best.candidate_index)
        {
            1
        } else {
            self.rank_depth_for_score(request, candidates, chosen, best.score)
        };
        Some(LoadBalancerCandidateChoice {
            candidate_index: best.candidate_index,
            rank_depth,
            selected_after_kv_free_tokens_skip: rank_depth > 1
                && self.higher_rank_kv_free_tokens_skip(request, candidates, chosen, best.score),
        })
    }

    fn higher_rank_kv_free_tokens_skip(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        chosen: &RoutedClusterSnapshot,
        chosen_score: f64,
    ) -> bool {
        let mut scorer = PulsarScorer::new(self.config.seed(), request);
        candidates.iter().any(|candidate| {
            let Some(score) = scorer.score(candidate) else {
                return false;
            };
            compare_ranked_candidate(score, candidate, chosen_score, chosen).is_lt()
                && self
                    .feasibility(request, candidate)
                    .skipped_for_kv_free_tokens()
        })
    }

    fn rank_depth_for_score(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidates: &[RoutedClusterSnapshot],
        chosen: &RoutedClusterSnapshot,
        chosen_score: f64,
    ) -> usize {
        let mut scorer = PulsarScorer::new(self.config.seed(), request);
        let mut rank_depth = 1;
        for candidate in candidates {
            let Some(score) = scorer.score(candidate) else {
                continue;
            };
            if compare_ranked_candidate(score, candidate, chosen_score, chosen).is_lt() {
                rank_depth += 1;
            }
        }
        rank_depth
    }

    #[cfg(test)]
    pub(super) fn weight(&self, candidate: &RoutedClusterSnapshot) -> Option<f64> {
        ranking::pulsar_weight(candidate)
    }

    pub(super) fn feasibility(
        &self,
        request: &LoadBalancerRequest<'_>,
        candidate: &RoutedClusterSnapshot,
    ) -> PulsarCandidateFeasibility {
        candidate_feasibility(&self.config, request, candidate)
    }
}

pub(super) fn input_work_admission_candidate(
    config: &LoadBalancerAlgorithmConfig,
    request: &LoadBalancerRequest<'_>,
    candidate: &RoutedClusterSnapshot,
) -> bool {
    has_valid_input_capacity(candidate)
        && candidate_feasibility(config, request, candidate).is_eligible()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PulsarCandidateFeasibility {
    Eligible,
    RetryExcluded,
    MissingRequiredInputTokens,
    MissingKvFreeTokens,
    InsufficientKvFreeTokens,
}

impl PulsarCandidateFeasibility {
    pub(super) fn is_eligible(self) -> bool {
        self == Self::Eligible
    }

    pub(super) fn skipped_for_kv_free_tokens(self) -> bool {
        matches!(
            self,
            Self::MissingKvFreeTokens | Self::InsufficientKvFreeTokens
        )
    }
}

fn candidate_feasibility(
    config: &LoadBalancerAlgorithmConfig,
    request: &LoadBalancerRequest<'_>,
    candidate: &RoutedClusterSnapshot,
) -> PulsarCandidateFeasibility {
    if request.excludes_cluster(&candidate.cluster_id) {
        return PulsarCandidateFeasibility::RetryExcluded;
    }

    if !config.considers_kv_free_tokens() {
        return PulsarCandidateFeasibility::Eligible;
    }

    let Some(input_tokens) = request.input_tokens else {
        return PulsarCandidateFeasibility::MissingRequiredInputTokens;
    };
    if !has_kv_free_token_metrics(candidate) {
        return PulsarCandidateFeasibility::MissingKvFreeTokens;
    }
    if candidate.stats.kv_cache_free_tokens < input_tokens {
        return PulsarCandidateFeasibility::InsufficientKvFreeTokens;
    }

    PulsarCandidateFeasibility::Eligible
}

fn has_kv_free_token_metrics(candidate: &RoutedClusterSnapshot) -> bool {
    candidate.stats.kv_cache_capacity_tokens > 0
        || candidate.stats.kv_cache_used_tokens > 0
        || candidate.stats.kv_cache_free_tokens > 0
}
