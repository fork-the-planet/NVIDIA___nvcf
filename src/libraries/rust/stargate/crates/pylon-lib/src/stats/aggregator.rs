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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use tokio::time::Instant as TokioInstant;

use crate::{CurrentModelStats, PylonRuntimeState};

use super::collector::{
    FinalizeRequestUpdate, RequestCounterUpdate, StatsAggregatorUpdate, StatsCollectorConfig,
    StatsUpdateSource,
};
use super::token_metrics::TpsDistribution;

#[derive(Debug, Default)]
pub(super) struct ModelMetricsState {
    pub(super) last_mean_input_tps: f64,
    pub(super) chat_output_tps_samples: VecDeque<f64>,
    pub(super) chat_output_tps_sum: f64,
    pub(super) embedding_item_tps_samples: VecDeque<f64>,
    pub(super) embedding_item_tps_sum: f64,
    pub(super) max_chat_output_tps: f64,
    pub(super) max_embedding_item_tps: f64,
    pub(super) kv_cache: KvCacheStatsSnapshot,
    pub(super) input_tps_distribution: TpsDistribution,
    aggregate_state_counted: bool,
    pub(super) counter_output_tps_authoritative: bool,
    pub(super) chunk_usage_stats_observed: bool,
    pub(super) kv_cache_stats_observed: bool,
    pub(super) engine_stream_stats_observed: bool,
    pub(super) last_stats_event_at: Option<TokioInstant>,
    pub(super) stats_observed_at_unix_ms: u64,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(super) struct KvCacheStatsSnapshot {
    pub(super) model: String,
    pub(super) kv_cache_capacity_tokens: u64,
    pub(super) kv_cache_used_tokens: u64,
    pub(super) kv_cache_free_tokens: u64,
}

#[derive(Debug)]
struct RequestCounterState {
    model_id: String,
    input: CounterSampleState,
    output: CounterSampleState,
    last_seen_at: TokioInstant,
}

impl RequestCounterState {
    fn from_observed(
        model_id: String,
        tokens_processed: u64,
        tokens_generated: u64,
        observed_at: TokioInstant,
    ) -> Self {
        Self {
            model_id,
            input: CounterSampleState::from_observed(tokens_processed, observed_at),
            output: CounterSampleState::from_observed(tokens_generated, observed_at),
            last_seen_at: observed_at,
        }
    }

    fn from_zero_baseline(model_id: String, observed_at: TokioInstant) -> Self {
        Self {
            model_id,
            input: CounterSampleState::from_observed(0, observed_at),
            output: CounterSampleState::from_observed(0, observed_at),
            last_seen_at: observed_at,
        }
    }
}

#[derive(Debug)]
enum RequestCounterLifecycle {
    Live(RequestCounterState),
    Finalized { observed_at: TokioInstant },
}

#[derive(Debug)]
struct CounterSampleState {
    observed: u64,
    sampled: u64,
    sampled_at: TokioInstant,
}

impl CounterSampleState {
    fn from_observed(observed: u64, observed_at: TokioInstant) -> Self {
        Self {
            observed,
            sampled: observed,
            sampled_at: observed_at,
        }
    }

    fn is_regression(&self, next: u64) -> bool {
        next < self.observed
    }

    fn observe(
        &mut self,
        next: u64,
        observed_at: TokioInstant,
        min_units: u64,
        duration_floor: Duration,
    ) -> Option<CounterSample> {
        let prior_observed = self.observed;
        self.observed = next;

        let units = next.saturating_sub(self.sampled);
        if units == 0 || units < min_units {
            return None;
        }

        let mut duration = observed_at
            .checked_duration_since(self.sampled_at)
            .unwrap_or(Duration::ZERO);
        if duration < duration_floor && self.sampled == 0 && prior_observed == 0 {
            duration = duration_floor;
        }
        if duration < duration_floor {
            return None;
        }

        self.sampled = next;
        self.sampled_at = observed_at;
        Some(CounterSample { units, duration })
    }
}

#[derive(Debug)]
struct CounterSample {
    units: u64,
    duration: Duration,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct InputThroughputSample {
    pub(super) units: u64,
    pub(super) duration: Duration,
    pub(super) clamp_duration_to_floor: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EmbeddingThroughputSample {
    pub(super) items: u64,
    pub(super) duration: Duration,
}

#[derive(Debug)]
struct RequestCounterEvent {
    source: StatsUpdateSource,
    request_id: String,
    model_id: String,
    tokens_processed: Option<u64>,
    tokens_generated: Option<u64>,
    finished: bool,
    observed_at: TokioInstant,
}

impl RequestCounterEvent {
    fn from_update(update: RequestCounterUpdate) -> Self {
        let RequestCounterUpdate {
            source,
            request_id,
            model_id,
            tokens_processed,
            tokens_generated,
            finished,
            observed_at,
        } = update;

        Self {
            source,
            request_id,
            model_id,
            tokens_processed,
            tokens_generated,
            finished,
            observed_at,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CounterSamplingPolicy {
    duration_floor: Duration,
    min_input_tokens: u64,
    min_output_tokens: u64,
}

#[derive(Debug, Default)]
struct RequestCounterSamples {
    input: Option<CounterSample>,
    output: Option<CounterSample>,
}

pub(super) struct StatsAggregator {
    pub(super) config: StatsCollectorConfig,
    pub(super) runtime_state: PylonRuntimeState,
    pub(super) per_model: HashMap<String, ModelMetricsState>,
    request_counters: HashMap<String, RequestCounterLifecycle>,
    live_request_count: usize,
    aggregate_model_state_count: usize,
    unix_ms_anchor: u64,
    instant_anchor: TokioInstant,
}

impl StatsAggregator {
    pub(super) fn new(config: StatsCollectorConfig, runtime_state: PylonRuntimeState) -> Self {
        Self {
            config,
            runtime_state,
            per_model: HashMap::new(),
            request_counters: HashMap::new(),
            live_request_count: 0,
            aggregate_model_state_count: 0,
            unix_ms_anchor: current_unix_millis(),
            instant_anchor: TokioInstant::now(),
        }
    }

    #[cfg(test)]
    pub(super) fn apply_update(
        &mut self,
        update: StatsAggregatorUpdate,
    ) -> Vec<(String, CurrentModelStats)> {
        let mut updated_models = Vec::new();
        self.apply_update_into(update, &mut updated_models);
        updated_models
    }

    pub(super) fn apply_update_into(
        &mut self,
        update: StatsAggregatorUpdate,
        updated_models: &mut Vec<(String, CurrentModelStats)>,
    ) {
        match update {
            StatsAggregatorUpdate::RequestCounters(update) => {
                self.apply_request_counters_into(update, updated_models);
            }
            StatsAggregatorUpdate::FinalizeRequest(update) => {
                updated_models.extend(self.finalize_request(update));
            }
            StatsAggregatorUpdate::EnableOpenAiFallback => {
                self.enable_openai_fallback();
            }
        }
    }

    pub(super) fn apply_control_update(&mut self, update: &StatsAggregatorUpdate) -> bool {
        if !matches!(update, StatsAggregatorUpdate::EnableOpenAiFallback) {
            return false;
        }
        self.enable_openai_fallback();
        true
    }

    pub(super) fn openai_fallback_stats_enabled(&self) -> bool {
        self.config.openai_fallback_stats_enabled
    }

    pub(super) fn live_request_count(&self) -> usize {
        self.live_request_count
    }

    #[cfg(test)]
    pub(super) fn request_counter_identity_count(&self) -> usize {
        self.request_counters.len()
    }

    pub(super) fn model_state_count(&self) -> usize {
        self.aggregate_model_state_count
    }

    pub(super) fn has_request_counter(&self, request_id: &str) -> bool {
        matches!(
            self.request_counters.get(request_id),
            Some(RequestCounterLifecycle::Live(_))
        )
    }

    pub(super) fn unix_millis_at(&self, observed_at: TokioInstant) -> u64 {
        if let Some(elapsed) = observed_at.checked_duration_since(self.instant_anchor) {
            return self
                .unix_ms_anchor
                .saturating_add(duration_millis_u64(elapsed));
        }
        let elapsed = self
            .instant_anchor
            .checked_duration_since(observed_at)
            .unwrap_or(Duration::ZERO);
        self.unix_ms_anchor
            .saturating_sub(duration_millis_u64(elapsed))
    }

    pub(super) fn sweep_stale(&mut self, now: TokioInstant) -> Vec<(String, CurrentModelStats)> {
        let mut dirty_models = Vec::new();
        let request_ttl = self.config.engine_stats_request_ttl;
        if !request_ttl.is_zero() {
            let metrics = self.runtime_state.metrics();
            let live_request_count = &mut self.live_request_count;
            self.request_counters
                .retain(|request_id, lifecycle| match lifecycle {
                    RequestCounterLifecycle::Live(state) => {
                        if elapsed_since(now, state.last_seen_at) < request_ttl {
                            return true;
                        }
                        let model_id = state.model_id.clone();
                        *lifecycle = RequestCounterLifecycle::Finalized { observed_at: now };
                        *live_request_count = (*live_request_count)
                            .checked_sub(1)
                            .expect("live request counter count underflowed during stale cleanup");
                        tracing::warn!(
                            request_id,
                            model_id,
                            ttl_ms = request_ttl.as_millis(),
                            "removing stale engine stats request entry"
                        );
                        if let Some(metrics) = metrics {
                            metrics.observe_engine_stats_stale_cleanup(
                                "request",
                                "engine_stats_stream",
                            );
                        }
                        push_dirty_model(&mut dirty_models, model_id);
                        true
                    }
                    RequestCounterLifecycle::Finalized { observed_at } => {
                        elapsed_since(now, *observed_at) < request_ttl
                    }
                });
        }

        let model_ttl = self.config.engine_stats_model_ttl;
        if !model_ttl.is_zero() {
            for (model_id, state) in &mut self.per_model {
                if state
                    .last_stats_event_at
                    .is_some_and(|observed_at| elapsed_since(now, observed_at) >= model_ttl)
                    && state.clear_live_output_tps()
                {
                    state.stats_observed_at_unix_ms = current_unix_millis();
                    tracing::warn!(
                        model_id,
                        ttl_ms = model_ttl.as_millis(),
                        "clearing stale engine stats output TPS"
                    );
                    if let Some(metrics) = self.runtime_state.metrics() {
                        metrics.observe_engine_stats_stale_cleanup("stats", "engine_stats_stream");
                    }
                    push_dirty_model(&mut dirty_models, model_id.clone());
                }
            }
        }

        if let Some(metrics) = self.runtime_state.metrics() {
            metrics
                .observe_engine_stats_model_states("engine_stats_stream", self.model_state_count());
            for _ in &dirty_models {
                metrics.observe_engine_stats_dirty_snapshot("engine_stats_stream", "stale");
            }
        }

        dirty_models
            .into_iter()
            .map(|model_id| {
                let stats = self.snapshot(&model_id);
                (model_id, stats)
            })
            .collect()
    }

    pub(super) fn apply_request_counters_into(
        &mut self,
        update: RequestCounterUpdate,
        updated_models: &mut Vec<(String, CurrentModelStats)>,
    ) {
        let Some(event) = self.admit_request_counter_update(update) else {
            return;
        };
        let Some(samples) = self.apply_request_counter_transition(&event) else {
            return;
        };
        self.publish_request_counter_model(event, samples, updated_models);
    }

    fn admit_request_counter_update(
        &self,
        update: RequestCounterUpdate,
    ) -> Option<RequestCounterEvent> {
        let event = RequestCounterEvent::from_update(update);

        if matches!(
            self.request_counters.get(&event.request_id),
            Some(RequestCounterLifecycle::Finalized { .. })
        ) {
            tracing::warn!(
                request_id = %event.request_id,
                source = ?event.source,
                "ignoring stats event after request finalization"
            );
            if let Some(metrics) = self.runtime_state.metrics() {
                metrics.observe_engine_stats_invalid_event("post_finalize");
            }
            return None;
        }

        if !self.configured_model_allowed(&event.model_id) {
            tracing::warn!(
                model_id = %event.model_id,
                configured_models = ?self.config.configured_model_ids,
                "dropping stats event for unconfigured model"
            );
            if let Some(metrics) = self.runtime_state.metrics() {
                metrics.observe_engine_stats_invalid_event("unconfigured_model");
            }
            return None;
        }

        Some(event)
    }

    fn apply_request_counter_transition(
        &mut self,
        event: &RequestCounterEvent,
    ) -> Option<RequestCounterSamples> {
        self.reset_request_counter_model_if_changed(event);
        if self.request_counter_regressed(event) {
            return None;
        }

        let policy = self.counter_sampling_policy();
        let mut new_request_state = None;
        let samples = match self.request_counters.get_mut(&event.request_id) {
            Some(RequestCounterLifecycle::Live(state)) => {
                let samples = observe_request_counter_samples(event, state, policy);
                state.last_seen_at = event.observed_at;
                samples
            }
            Some(RequestCounterLifecycle::Finalized { .. }) => {
                panic!("finalized request counter passed admission")
            }
            None => {
                let (state, samples) = new_request_counter_state(event, policy);
                if !event.finished {
                    new_request_state = Some(state);
                }
                samples
            }
        };

        self.store_request_counter_state(event, new_request_state);
        Some(samples)
    }

    fn reset_request_counter_model_if_changed(&mut self, event: &RequestCounterEvent) {
        let prior_model = match self.request_counters.get(&event.request_id) {
            Some(RequestCounterLifecycle::Live(state)) if state.model_id != event.model_id => {
                state.model_id.clone()
            }
            Some(RequestCounterLifecycle::Live(_)) | None => return,
            Some(RequestCounterLifecycle::Finalized { .. }) => {
                panic!("finalized request counter passed admission")
            }
        };
        let removed = self.remove_request_counter_lifecycle(&event.request_id);
        assert!(
            matches!(removed, Some(RequestCounterLifecycle::Live(_))),
            "live request counter disappeared during model reset"
        );
        tracing::warn!(
            request_id = %event.request_id,
            prior_model,
            model_id = %event.model_id,
            "resetting request stats after model changed"
        );
    }

    fn request_counter_regressed(&self, event: &RequestCounterEvent) -> bool {
        let state = match self.request_counters.get(&event.request_id) {
            Some(RequestCounterLifecycle::Live(state)) => state,
            None => return false,
            Some(RequestCounterLifecycle::Finalized { .. }) => {
                panic!("finalized request counter passed admission")
            }
        };
        if !request_counter_event_regressed(event, state) {
            return false;
        }

        tracing::warn!(
            request_id = %event.request_id,
            model_id = %event.model_id,
            prior_tokens_processed = state.input.observed,
            tokens_processed = event.tokens_processed.unwrap_or(state.input.observed),
            prior_tokens_generated = state.output.observed,
            tokens_generated = event.tokens_generated.unwrap_or(state.output.observed),
            source = ?event.source,
            "ignoring regressing request stats counters"
        );
        if let Some(metrics) = self.runtime_state.metrics() {
            metrics.observe_engine_stats_invalid_event("regressing_counters");
        }
        true
    }

    fn counter_sampling_policy(&self) -> CounterSamplingPolicy {
        CounterSamplingPolicy {
            duration_floor: self.config.duration_floor,
            min_input_tokens: self.config.min_input_tokens,
            min_output_tokens: self.config.min_output_tokens,
        }
    }

    fn store_request_counter_state(
        &mut self,
        event: &RequestCounterEvent,
        new_request_state: Option<RequestCounterState>,
    ) {
        if event.finished {
            self.replace_request_counter_lifecycle(
                event.request_id.clone(),
                RequestCounterLifecycle::Finalized {
                    observed_at: event.observed_at,
                },
            );
        } else if let Some(state) = new_request_state {
            let previous = self.replace_request_counter_lifecycle(
                event.request_id.clone(),
                RequestCounterLifecycle::Live(state),
            );
            assert!(
                previous.is_none(),
                "new request counter state replaced an existing lifecycle"
            );
        }
    }

    fn replace_request_counter_lifecycle(
        &mut self,
        request_id: String,
        next: RequestCounterLifecycle,
    ) -> Option<RequestCounterLifecycle> {
        let next_is_live = matches!(&next, RequestCounterLifecycle::Live(_));
        let previous = self.request_counters.insert(request_id, next);
        let previous_was_live = matches!(previous.as_ref(), Some(RequestCounterLifecycle::Live(_)));
        match (previous_was_live, next_is_live) {
            (false, true) => {
                self.live_request_count = self
                    .live_request_count
                    .checked_add(1)
                    .expect("live request counter count overflowed");
            }
            (true, false) => {
                self.live_request_count = self
                    .live_request_count
                    .checked_sub(1)
                    .expect("live request counter count underflowed");
            }
            (false, false) | (true, true) => {}
        }
        previous
    }

    fn remove_request_counter_lifecycle(
        &mut self,
        request_id: &str,
    ) -> Option<RequestCounterLifecycle> {
        let removed = self.request_counters.remove(request_id);
        if matches!(removed.as_ref(), Some(RequestCounterLifecycle::Live(_))) {
            self.live_request_count = self
                .live_request_count
                .checked_sub(1)
                .expect("live request counter count underflowed");
        }
        removed
    }

    fn publish_request_counter_model(
        &mut self,
        event: RequestCounterEvent,
        samples: RequestCounterSamples,
        updated_models: &mut Vec<(String, CurrentModelStats)>,
    ) {
        let stats_observed_at_unix_ms = self.unix_millis_at(event.observed_at);
        let dirty = {
            let config = &self.config;
            let model_state = aggregate_model_state(
                &mut self.per_model,
                &mut self.aggregate_model_state_count,
                event.model_id.clone(),
            );
            model_state.last_stats_event_at = Some(event.observed_at);
            model_state.stats_observed_at_unix_ms = stats_observed_at_unix_ms;

            let mut dirty = mark_engine_stream_observed(model_state, event.source);
            if let Some(sample) = samples.input {
                dirty |= apply_input_throughput_sample(
                    config,
                    &self.runtime_state,
                    &event.model_id,
                    model_state,
                    InputThroughputSample {
                        units: sample.units,
                        duration: sample.duration,
                        clamp_duration_to_floor: false,
                    },
                );
            }
            dirty |= apply_output_counter_sample(
                model_state,
                samples.output,
                config.duration_floor,
                config.smoothing_window_size,
            );

            dirty
        };
        if dirty {
            let stats = self.snapshot(&event.model_id);
            updated_models.push((event.model_id, stats));
        }
    }

    pub(super) fn finalize_request(
        &mut self,
        update: FinalizeRequestUpdate,
    ) -> Vec<(String, CurrentModelStats)> {
        let previous = self.replace_request_counter_lifecycle(
            update.request_id.clone(),
            RequestCounterLifecycle::Finalized {
                observed_at: update.observed_at,
            },
        );
        let Some(RequestCounterLifecycle::Live(state)) = previous else {
            return Vec::new();
        };
        let stats_observed_at_unix_ms = self.unix_millis_at(update.observed_at);
        if let Some(model_state) = self.per_model.get_mut(&state.model_id) {
            model_state.stats_observed_at_unix_ms = stats_observed_at_unix_ms;
        }
        tracing::debug!(
            request_id = update.request_id,
            source = ?update.source,
            "finalized request stats"
        );
        vec![(state.model_id.clone(), self.snapshot(&state.model_id))]
    }

    pub(super) fn configured_model_allowed(&self, model_id: &str) -> bool {
        self.config.configured_model_ids.is_empty()
            || self
                .config
                .configured_model_ids
                .iter()
                .any(|configured| configured == model_id)
    }

    pub(super) fn record_engine_embedding_sample(
        &mut self,
        model_id: &str,
        sample: EmbeddingThroughputSample,
    ) -> bool {
        if !self.configured_model_allowed(model_id) {
            return false;
        }
        let duration_floor = self.config.duration_floor;
        let Some(embedding_item_tps) = tps_for_units(
            sample.items,
            sample.duration.max(duration_floor),
            duration_floor,
        ) else {
            return false;
        };
        let model_state = aggregate_model_state(
            &mut self.per_model,
            &mut self.aggregate_model_state_count,
            model_id.to_string(),
        );
        model_state.stats_observed_at_unix_ms = current_unix_millis();
        model_state.max_embedding_item_tps =
            model_state.max_embedding_item_tps.max(embedding_item_tps);
        push_sample(
            &mut model_state.embedding_item_tps_samples,
            &mut model_state.embedding_item_tps_sum,
            embedding_item_tps,
            self.config.smoothing_window_size,
        );
        true
    }

    fn enable_openai_fallback(&mut self) {
        if self.config.openai_fallback_stats_enabled {
            return;
        }
        self.config.openai_fallback_stats_enabled = true;
        tracing::warn!("OpenAI fallback stats enabled after engine stats stream was unsupported");
        if let Some(metrics) = self.runtime_state.metrics() {
            metrics.observe_engine_stats_source_transition(
                "engine_stats_stream",
                "openai_fallback",
                "unsupported",
            );
        }
    }
}

fn aggregate_model_state<'a>(
    per_model: &'a mut HashMap<String, ModelMetricsState>,
    aggregate_model_state_count: &mut usize,
    model_id: String,
) -> &'a mut ModelMetricsState {
    let model_state = per_model.entry(model_id).or_default();
    if !model_state.aggregate_state_counted {
        model_state.aggregate_state_counted = true;
        *aggregate_model_state_count += 1;
    }
    model_state
}

fn request_counter_event_regressed(
    event: &RequestCounterEvent,
    state: &RequestCounterState,
) -> bool {
    event
        .tokens_processed
        .is_some_and(|next| state.input.is_regression(next))
        || event
            .tokens_generated
            .is_some_and(|next| state.output.is_regression(next))
}

fn new_request_counter_state(
    event: &RequestCounterEvent,
    policy: CounterSamplingPolicy,
) -> (RequestCounterState, RequestCounterSamples) {
    let next_tokens_processed = event.tokens_processed.unwrap_or(0);
    let next_tokens_generated = event.tokens_generated.unwrap_or(0);
    let mut state = if event.source == StatsUpdateSource::EngineStatsStream {
        RequestCounterState::from_zero_baseline(event.model_id.clone(), event.observed_at)
    } else {
        RequestCounterState::from_observed(
            event.model_id.clone(),
            next_tokens_processed,
            next_tokens_generated,
            event.observed_at,
        )
    };

    let samples = if event.source == StatsUpdateSource::EngineStatsStream {
        observe_request_counter_samples(event, &mut state, policy)
    } else {
        RequestCounterSamples::default()
    };
    (state, samples)
}

fn observe_request_counter_samples(
    event: &RequestCounterEvent,
    state: &mut RequestCounterState,
    policy: CounterSamplingPolicy,
) -> RequestCounterSamples {
    let mut samples = RequestCounterSamples::default();
    if let Some(next_tokens_processed) = event.tokens_processed {
        samples.input = state.input.observe(
            next_tokens_processed,
            event.observed_at,
            policy.min_input_tokens,
            policy.duration_floor,
        );
    }
    if let Some(next_tokens_generated) = event.tokens_generated {
        samples.output = state.output.observe(
            next_tokens_generated,
            event.observed_at,
            policy.min_output_tokens,
            policy.duration_floor,
        );
    }
    samples
}

fn mark_engine_stream_observed(
    model_state: &mut ModelMetricsState,
    source: StatsUpdateSource,
) -> bool {
    if source != StatsUpdateSource::EngineStatsStream || model_state.engine_stream_stats_observed {
        return false;
    }
    model_state.engine_stream_stats_observed = true;
    true
}

pub(super) fn apply_input_throughput_sample(
    config: &StatsCollectorConfig,
    runtime_state: &PylonRuntimeState,
    model_id: &str,
    model_state: &mut ModelMetricsState,
    sample: InputThroughputSample,
) -> bool {
    if sample.units < config.min_input_tokens {
        return false;
    }
    let duration = if sample.clamp_duration_to_floor {
        sample.duration.max(config.duration_floor)
    } else {
        sample.duration
    };
    let Some(input_tps) = tps_for_units(sample.units, duration, config.duration_floor) else {
        return false;
    };

    model_state.input_tps_distribution.update(input_tps);
    if !model_state.input_tps_distribution.has_sufficient_data()
        || !valid_last_mean_input_tps(model_state.input_tps_distribution.mean)
    {
        return false;
    }

    let last_mean_input_tps =
        effective_last_mean_input_tps(config, model_state.input_tps_distribution.mean);
    if model_state.last_mean_input_tps == last_mean_input_tps {
        return false;
    }

    model_state.last_mean_input_tps = last_mean_input_tps;
    runtime_state.update_model_throughput(model_id, last_mean_input_tps);
    true
}

fn apply_output_counter_sample(
    model_state: &mut ModelMetricsState,
    sample: Option<CounterSample>,
    duration_floor: Duration,
    smoothing_window_size: usize,
) -> bool {
    let Some(sample) = sample else {
        return false;
    };
    let Some(output_tps) = tps_for_units(sample.units, sample.duration, duration_floor) else {
        return false;
    };

    model_state.max_chat_output_tps = model_state.max_chat_output_tps.max(output_tps);
    model_state.counter_output_tps_authoritative = true;
    push_sample(
        &mut model_state.chat_output_tps_samples,
        &mut model_state.chat_output_tps_sum,
        output_tps,
        smoothing_window_size,
    );
    true
}

fn elapsed_since(now: TokioInstant, then: TokioInstant) -> Duration {
    now.checked_duration_since(then).unwrap_or(Duration::ZERO)
}

fn push_dirty_model(models: &mut Vec<String>, model_id: String) {
    if !models.iter().any(|existing| existing == &model_id) {
        models.push(model_id);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ModelStatsSnapshotInputs {
    pub(super) active_chat_output_tps: f64,
    pub(super) queue_size: u64,
    pub(super) queued_input_size: u64,
    pub(super) num_running_queries: u64,
    pub(super) total_query_input_size: u64,
    pub(super) input_processing_queries: u64,
    pub(super) output_generation_queries: u64,
}

impl ModelMetricsState {
    pub(super) fn clear_live_output_tps(&mut self) -> bool {
        self.last_stats_event_at = None;
        if self.chat_output_tps_samples.is_empty() {
            return false;
        }
        self.chat_output_tps_samples.clear();
        self.chat_output_tps_sum = 0.0;
        self.counter_output_tps_authoritative = false;
        true
    }

    pub(super) fn current_stats(&self, inputs: ModelStatsSnapshotInputs) -> CurrentModelStats {
        let (stats_capabilities, stats_sources) = self.stats_labels();
        let active_chat_output_tps = if self.counter_output_tps_authoritative {
            0.0
        } else {
            inputs.active_chat_output_tps
        };
        CurrentModelStats {
            last_mean_input_tps: self.last_mean_input_tps,
            output_tps: active_chat_output_tps.max(average_with_sum(
                &self.chat_output_tps_samples,
                self.chat_output_tps_sum,
            )),
            embedding_item_tps: average_with_sum(
                &self.embedding_item_tps_samples,
                self.embedding_item_tps_sum,
            ),
            max_output_tps: self.max_chat_output_tps,
            max_embedding_item_tps: self.max_embedding_item_tps,
            queue_size: inputs.queue_size,
            queued_input_size: inputs.queued_input_size,
            kv_cache_capacity_tokens: self.kv_cache.kv_cache_capacity_tokens,
            kv_cache_used_tokens: self.kv_cache.kv_cache_used_tokens,
            kv_cache_free_tokens: self.kv_cache.kv_cache_free_tokens,
            num_running_queries: inputs.num_running_queries,
            max_engine_concurrency: None,
            total_query_input_size: inputs.total_query_input_size,
            queue_time_estimate_ms_by_priority: None,
            input_processing_queries: inputs.input_processing_queries,
            output_generation_queries: inputs.output_generation_queries,
            stats_observed_at_unix_ms: self.stats_observed_at_unix_ms,
            stats_capabilities,
            stats_sources,
        }
    }

    pub(super) fn stats_labels(&self) -> (Vec<String>, Vec<String>) {
        // These labels are sticky per model metrics state. They describe
        // contract surfaces pylon has observed from this backend, not just the
        // surfaces exercised by the most recent request.
        let mut capabilities = Vec::new();
        let mut sources = Vec::new();
        if self.chunk_usage_stats_observed {
            capabilities.push("request.output.chunk_usage".to_string());
            sources.push("chunk_usage".to_string());
        }
        if self.engine_stream_stats_observed {
            capabilities.push("model.throughput.engine_stream".to_string());
            sources.push("engine_stats_stream".to_string());
        }
        if self.kv_cache_stats_observed {
            capabilities.push("machine.kv_cache.http".to_string());
            sources.push("kv_cache_stats".to_string());
        }
        (capabilities, sources)
    }
}

pub(super) fn current_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or_default()
}

pub(super) fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

pub(super) fn output_decode_duration(
    total_duration: Duration,
    time_to_first_output: Option<Duration>,
    time_to_first_token: Option<Duration>,
    duration_floor: Duration,
) -> Option<Duration> {
    if let Some(time_to_first_token) = time_to_first_token {
        // Observation timestamps can arrive with the same coarse clock tick; never underflow decode time.
        let token_duration = total_duration.saturating_sub(time_to_first_token);
        if token_duration >= duration_floor {
            return Some(token_duration);
        }
    }

    time_to_first_output
        // Observation timestamps can arrive with the same coarse clock tick; never underflow decode time.
        .map(|time_to_first_output| total_duration.saturating_sub(time_to_first_output))
}

pub(super) fn tps_for_units(
    units: u64,
    duration: Duration,
    duration_floor: Duration,
) -> Option<f64> {
    if units == 0 || duration < duration_floor {
        return None;
    }
    Some(units as f64 / duration.as_secs_f64())
}

pub(super) fn valid_last_mean_input_tps(last_mean_input_tps: f64) -> bool {
    last_mean_input_tps > 0.0 && last_mean_input_tps.is_finite()
}

pub(super) fn fixed_last_mean_input_tps(config: &StatsCollectorConfig) -> Option<f64> {
    config
        .fixed_last_mean_input_tps
        .filter(|value| valid_last_mean_input_tps(*value))
}

pub(super) fn effective_last_mean_input_tps(config: &StatsCollectorConfig, observed: f64) -> f64 {
    fixed_last_mean_input_tps(config).unwrap_or(observed)
}

pub(super) fn apply_fixed_last_mean_input_tps(
    config: &StatsCollectorConfig,
    stats: &mut CurrentModelStats,
) {
    if let Some(last_mean_input_tps) = fixed_last_mean_input_tps(config) {
        stats.last_mean_input_tps = last_mean_input_tps;
    }
}

pub(super) fn push_sample(
    samples: &mut VecDeque<f64>,
    sum: &mut f64,
    sample: f64,
    window_size: usize,
) {
    if window_size == 0 {
        return;
    }
    samples.push_back(sample);
    *sum += sample;
    while samples.len() > window_size {
        if let Some(removed) = samples.pop_front() {
            *sum -= removed;
        }
    }
}

pub(super) fn average_with_sum(samples: &VecDeque<f64>, sum: f64) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        sum / samples.len() as f64
    }
}
