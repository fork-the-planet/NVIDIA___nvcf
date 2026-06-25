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

use std::collections::HashMap;
use tokio::time::Instant as TokioInstant;

use crate::request_observer::{RequestObservationEndpoint, RequestObservationState};
use crate::{CurrentModelStats, RequestObservation, RequestObservationEvent};

use super::aggregator::{
    EmbeddingThroughputSample, InputThroughputSample, KvCacheStatsSnapshot, ModelMetricsState,
    ModelStatsSnapshotInputs, StatsAggregator, apply_fixed_last_mean_input_tps,
    apply_input_throughput_sample, current_unix_millis, output_decode_duration, push_sample,
    tps_for_units,
};
use super::collector::{
    FinalizeRequestUpdate, RequestCounterUpdate, RequestCounterUpdateInput, StatsAggregatorUpdate,
    StatsCollectorConfig, StatsUpdateSource,
};

impl StatsAggregator {
    pub(super) fn apply_fallback_observation(
        &mut self,
        event: &RequestObservationEvent,
    ) -> Vec<(String, CurrentModelStats)> {
        let observation = &event.observation;
        let mut changed_models = self.record_fallback_observation(event);
        let mut counter_updates = Vec::new();
        if let Some(update) = fallback_update_from_observation(observation) {
            self.apply_update_into(update, &mut counter_updates);
        }
        for (model_id, _) in counter_updates {
            push_changed_model(&mut changed_models, model_id);
        }
        self.snapshots(changed_models)
    }

    pub(super) fn apply_stream_observation(
        &mut self,
        event: &RequestObservationEvent,
    ) -> Vec<(String, CurrentModelStats)> {
        let observation = &event.observation;
        let mut changed_models = self.record_lifecycle_event(event);
        if let Some(sample) = observed_stream_embedding_sample(observation)
            && self.record_engine_embedding_sample(&observation.model_id, sample)
        {
            push_changed_model(&mut changed_models, observation.model_id.clone());
        }
        self.snapshots(changed_models)
    }

    pub(super) fn apply_kv_cache_stats(
        &mut self,
        kv_cache: KvCacheStatsSnapshot,
    ) -> Option<(String, CurrentModelStats)> {
        if !self.configured_model_allowed(&kv_cache.model) {
            return None;
        }
        let model_id = kv_cache.model.clone();
        let model_state = self.per_model.entry(model_id.clone()).or_default();
        model_state.kv_cache = kv_cache;
        model_state.kv_cache_stats_observed = true;
        model_state.stats_observed_at_unix_ms = current_unix_millis();
        let stats = self.snapshot(&model_id);
        Some((model_id, stats))
    }

    pub(super) fn snapshot(&self, model_id: &str) -> CurrentModelStats {
        snapshot_model_stats(&self.config, &self.runtime_state, &self.per_model, model_id)
    }

    fn record_fallback_observation(&mut self, event: &RequestObservationEvent) -> Vec<String> {
        let observation = &event.observation;
        let counter_already_observed =
            observation.output_tokens_explicit && self.has_request_counter(&observation.request_id);
        let mut changed_models = self.record_lifecycle_event(event);
        let input_sample = observed_input_throughput_sample(observation);
        let model_state = self
            .per_model
            .entry(observation.model_id.clone())
            .or_default();
        if observation.output_tokens_from_chunk_usage {
            model_state.chunk_usage_stats_observed = true;
        }
        if observation.state == RequestObservationState::Complete {
            match observation.endpoint {
                RequestObservationEndpoint::ChatCompletions
                | RequestObservationEndpoint::Responses => {
                    if !counter_already_observed
                        && let Some(output_tps) = observed_output_tps(&self.config, observation)
                    {
                        model_state.max_chat_output_tps =
                            model_state.max_chat_output_tps.max(output_tps);
                        push_sample(
                            &mut model_state.chat_output_tps_samples,
                            &mut model_state.chat_output_tps_sum,
                            output_tps,
                            self.config.smoothing_window_size,
                        );
                    }
                }
                RequestObservationEndpoint::Embeddings => {
                    if let Some(embedding_item_tps) =
                        observed_embeddings_item_tps(&self.config, observation)
                    {
                        model_state.max_embedding_item_tps =
                            model_state.max_embedding_item_tps.max(embedding_item_tps);
                        push_sample(
                            &mut model_state.embedding_item_tps_samples,
                            &mut model_state.embedding_item_tps_sum,
                            embedding_item_tps,
                            self.config.smoothing_window_size,
                        );
                    }
                }
            }
        }
        if input_sample.is_some_and(|sample| {
            apply_input_throughput_sample(
                &self.config,
                &self.runtime_state,
                &observation.model_id,
                model_state,
                sample,
            )
        }) {
            push_changed_model(&mut changed_models, observation.model_id.clone());
        }
        changed_models
    }

    fn record_lifecycle_event(&mut self, event: &RequestObservationEvent) -> Vec<String> {
        let observation = &event.observation;
        let active_chat_output_tps = self
            .config
            .openai_fallback_stats_enabled
            .then(|| observed_output_tps(&self.config, observation))
            .flatten();
        let mut changed_models = event.changed_model_ids.clone();
        if let Some(model_id) = self
            .runtime_state
            .update_request_active_output_tps(&observation.request_id, active_chat_output_tps)
        {
            push_changed_model(&mut changed_models, model_id);
        }
        self.per_model
            .entry(observation.model_id.clone())
            .or_default()
            .stats_observed_at_unix_ms = current_unix_millis();
        changed_models
    }

    fn snapshots(&self, model_ids: Vec<String>) -> Vec<(String, CurrentModelStats)> {
        model_ids
            .into_iter()
            .map(|model_id| {
                let stats = self.snapshot(&model_id);
                (model_id, stats)
            })
            .collect()
    }
}

pub(super) fn fallback_update_from_observation(
    observation: &RequestObservation,
) -> Option<StatsAggregatorUpdate> {
    let observed_at = TokioInstant::now();
    let tokens_generated = observation
        .output_tokens_explicit
        .then_some(observation.output_tokens);
    if tokens_generated.is_some() {
        return Some(StatsAggregatorUpdate::RequestCounters(
            RequestCounterUpdate::new(RequestCounterUpdateInput {
                source: StatsUpdateSource::OpenAiFallback,
                request_id: observation.request_id.clone(),
                model_id: observation.model_id.clone(),
                tokens_processed: None,
                tokens_generated,
                finished: observation.is_terminal(),
                observed_at,
            }),
        ));
    }
    if observation.is_terminal() {
        return Some(StatsAggregatorUpdate::FinalizeRequest(
            FinalizeRequestUpdate::new(
                StatsUpdateSource::OpenAiFallback,
                observation.request_id.clone(),
                observed_at,
            ),
        ));
    }
    None
}

fn observed_stream_embedding_sample(
    observation: &RequestObservation,
) -> Option<EmbeddingThroughputSample> {
    if observation.endpoint != RequestObservationEndpoint::Embeddings
        || observation.state != RequestObservationState::Complete
    {
        return None;
    }

    let embedding_duration = observation
        .time_to_response_headers
        .map(|response_headers| observation.total_duration.saturating_sub(response_headers));

    if !observation.embedding_items_observed || embedding_duration.is_none() {
        return None;
    }

    Some(EmbeddingThroughputSample {
        items: observation.embedding_items,
        duration: embedding_duration.expect("checked above"),
    })
}

fn snapshot_model_stats(
    config: &StatsCollectorConfig,
    runtime_state: &crate::PylonRuntimeState,
    per_model: &HashMap<String, ModelMetricsState>,
    model_id: &str,
) -> CurrentModelStats {
    let queue_snapshot = runtime_state.snapshot_live_model(model_id);

    let snapshot_inputs = ModelStatsSnapshotInputs {
        active_chat_output_tps: queue_snapshot.active_chat_output_tps,
        queue_size: queue_snapshot.queue_size,
        queued_input_size: queue_snapshot.queued_input_size,
        num_running_queries: queue_snapshot.num_running_queries,
        total_query_input_size: queue_snapshot.total_query_input_size,
        input_processing_queries: queue_snapshot.input_processing_queries,
        output_generation_queries: queue_snapshot.output_generation_queries,
    };
    let mut stats = per_model
        .get(model_id)
        .map(|model_state| model_state.current_stats(snapshot_inputs))
        .unwrap_or_else(|| ModelMetricsState::default().current_stats(snapshot_inputs));
    stats.queue_time_estimate_ms_by_priority = queue_snapshot.queue_time_estimate_ms_by_priority;
    apply_fixed_last_mean_input_tps(config, &mut stats);
    stats
}

pub(super) fn observed_input_throughput_sample(
    observation: &RequestObservation,
) -> Option<InputThroughputSample> {
    if observation.state == RequestObservationState::Complete {
        let (input_tokens, duration) = match observation.endpoint {
            RequestObservationEndpoint::ChatCompletions | RequestObservationEndpoint::Responses => {
                (observation.input_tokens, observation.time_to_first_output?)
            }
            RequestObservationEndpoint::Embeddings => {
                let duration = observation
                    .time_to_response_headers
                    .unwrap_or(observation.total_duration);
                (observation.input_tokens, duration)
            }
        };
        return Some(InputThroughputSample {
            units: input_tokens,
            duration,
            clamp_duration_to_floor: observation.endpoint == RequestObservationEndpoint::Embeddings,
        });
    }
    None
}

fn push_changed_model(models: &mut Vec<String>, model_id: String) {
    if !models.iter().any(|existing| existing == &model_id) {
        models.push(model_id);
    }
}

pub(super) fn observed_output_tps(
    config: &StatsCollectorConfig,
    observation: &RequestObservation,
) -> Option<f64> {
    if observation.endpoint == RequestObservationEndpoint::Embeddings {
        return None;
    }
    if observation.output_tokens < config.min_output_tokens {
        return None;
    }
    let duration = output_decode_duration(
        observation.total_duration,
        observation.time_to_first_output,
        observation.time_to_first_token,
        config.duration_floor,
    )?;
    tps_for_units(observation.output_tokens, duration, config.duration_floor)
}

pub(super) fn observed_embeddings_item_tps(
    config: &StatsCollectorConfig,
    observation: &RequestObservation,
) -> Option<f64> {
    if observation.endpoint != RequestObservationEndpoint::Embeddings {
        return None;
    }
    let response_headers_duration = observation.time_to_response_headers?;
    let relay_duration = observation
        .total_duration
        .saturating_sub(response_headers_duration);
    let duration = relay_duration.max(config.duration_floor);
    tps_for_units(observation.embedding_items, duration, config.duration_floor)
}
