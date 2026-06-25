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

use std::time::Duration;

use indexmap::IndexMap;
use tokio::time::Instant as TokioInstant;
use tokio_util::sync::CancellationToken;

#[cfg(test)]
use crate::RequestObservation;
use crate::{CurrentModelStats, PylonRuntimeState, RequestObservationEvent};
use stargate_runtime::{OwnedTask, TASK_SHUTDOWN_TIMEOUT};

use super::aggregator::{KvCacheStatsSnapshot, StatsAggregator, fixed_last_mean_input_tps};
#[cfg(test)]
use super::metrics::PylonMetrics;

const DEFAULT_OBSERVATION_CHANNEL_CAPACITY: usize = 1024;
const DEFAULT_SMOOTHING_WINDOW_SIZE: usize = 8;
const DEFAULT_MIN_INPUT_TOKENS: u64 = 1;
const DEFAULT_MIN_OUTPUT_TOKENS: u64 = 1;
const DEFAULT_DURATION_FLOOR: Duration = Duration::from_millis(10);
const DEFAULT_KV_CACHE_POLL_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_KV_CACHE_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);
const DEFAULT_ENGINE_STATS_REQUEST_TTL: Duration = Duration::from_secs(300);
const DEFAULT_ENGINE_STATS_MODEL_TTL: Duration = Duration::from_secs(30);
const DEFAULT_ENGINE_STATS_SWEEP_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
pub struct StatsCollectorConfig {
    pub observation_channel_capacity: usize,
    pub smoothing_window_size: usize,
    pub min_input_tokens: u64,
    pub min_output_tokens: u64,
    pub duration_floor: Duration,
    pub configured_model_ids: Vec<String>,
    /// Pins input throughput for deterministic benchmarks instead of publishing learned samples.
    pub fixed_last_mean_input_tps: Option<f64>,
    pub kv_cache_stats_url: Option<String>,
    pub kv_cache_poll_interval: Duration,
    pub kv_cache_request_timeout: Duration,
    pub engine_stats_request_ttl: Duration,
    pub engine_stats_model_ttl: Duration,
    pub engine_stats_sweep_interval: Duration,
    pub openai_fallback_stats_enabled: bool,
}

impl Default for StatsCollectorConfig {
    fn default() -> Self {
        Self {
            observation_channel_capacity: DEFAULT_OBSERVATION_CHANNEL_CAPACITY,
            smoothing_window_size: DEFAULT_SMOOTHING_WINDOW_SIZE,
            min_input_tokens: DEFAULT_MIN_INPUT_TOKENS,
            min_output_tokens: DEFAULT_MIN_OUTPUT_TOKENS,
            duration_floor: DEFAULT_DURATION_FLOOR,
            configured_model_ids: Vec::new(),
            fixed_last_mean_input_tps: None,
            kv_cache_stats_url: None,
            kv_cache_poll_interval: DEFAULT_KV_CACHE_POLL_INTERVAL,
            kv_cache_request_timeout: DEFAULT_KV_CACHE_REQUEST_TIMEOUT,
            engine_stats_request_ttl: DEFAULT_ENGINE_STATS_REQUEST_TTL,
            engine_stats_model_ttl: DEFAULT_ENGINE_STATS_MODEL_TTL,
            engine_stats_sweep_interval: DEFAULT_ENGINE_STATS_SWEEP_INTERVAL,
            openai_fallback_stats_enabled: true,
        }
    }
}

pub fn stats_aggregator_update_channel(
    config: &StatsCollectorConfig,
) -> (
    flume::Sender<StatsAggregatorUpdate>,
    flume::Receiver<StatsAggregatorUpdate>,
) {
    flume::bounded(config.observation_channel_capacity)
}

pub struct StatsCollectorHandle {
    task: OwnedTask,
}

impl StatsCollectorHandle {
    pub async fn wait_for_exit(&mut self) -> Result<(), tokio::task::JoinError> {
        self.task.wait_for_exit().await
    }

    pub async fn shutdown(self) {
        self.task.shutdown(TASK_SHUTDOWN_TIMEOUT).await;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsUpdateSource {
    EngineStatsStream,
    OpenAiFallback,
}

#[derive(Debug, Clone)]
pub enum StatsAggregatorUpdate {
    RequestCounters(RequestCounterUpdate),
    FinalizeRequest(FinalizeRequestUpdate),
    EnableOpenAiFallback,
}

#[derive(Debug, Clone)]
pub struct RequestCounterUpdate {
    pub(crate) source: StatsUpdateSource,
    pub(crate) request_id: String,
    pub(crate) model_id: String,
    pub(crate) tokens_processed: Option<u64>,
    pub(crate) tokens_generated: Option<u64>,
    pub(crate) finished: bool,
    pub(crate) observed_at: TokioInstant,
}

#[derive(Debug, Clone)]
pub struct RequestCounterUpdateInput {
    pub source: StatsUpdateSource,
    pub request_id: String,
    pub model_id: String,
    pub tokens_processed: Option<u64>,
    pub tokens_generated: Option<u64>,
    pub finished: bool,
    pub observed_at: tokio::time::Instant,
}

impl RequestCounterUpdate {
    pub fn new(input: RequestCounterUpdateInput) -> Self {
        let RequestCounterUpdateInput {
            source,
            request_id,
            model_id,
            tokens_processed,
            tokens_generated,
            finished,
            observed_at,
        } = input;
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

#[derive(Debug, Clone)]
pub struct FinalizeRequestUpdate {
    pub(crate) source: StatsUpdateSource,
    pub(crate) request_id: String,
    pub(crate) observed_at: TokioInstant,
}

impl FinalizeRequestUpdate {
    pub fn new(
        source: StatsUpdateSource,
        request_id: impl Into<String>,
        observed_at: TokioInstant,
    ) -> Self {
        Self {
            source,
            request_id: request_id.into(),
            observed_at,
        }
    }
}

pub fn start_stats_collector(
    config: StatsCollectorConfig,
    observation_rx: flume::Receiver<RequestObservationEvent>,
    runtime_state: PylonRuntimeState,
) -> StatsCollectorHandle {
    start_stats_collector_with_engine_stats(config, observation_rx, None, runtime_state)
}

pub fn start_stats_collector_with_engine_stats(
    mut config: StatsCollectorConfig,
    observation_rx: flume::Receiver<RequestObservationEvent>,
    stats_update_rx: Option<flume::Receiver<StatsAggregatorUpdate>>,
    runtime_state: PylonRuntimeState,
) -> StatsCollectorHandle {
    if stats_update_rx.is_some() {
        // A wired engine stats stream is the throughput source of truth. Auto
        // mode falls back only after the stream task sends EnableOpenAiFallback.
        config.openai_fallback_stats_enabled = false;
    }
    let task = OwnedTask::spawn("stats collector", move |stop| async move {
        run_stats_collector(config, observation_rx, stats_update_rx, runtime_state, stop).await;
    });
    StatsCollectorHandle { task }
}

fn publish_model_stats_update(
    runtime_state: &PylonRuntimeState,
    model_id: String,
    stats: CurrentModelStats,
) {
    observe_model_metric(runtime_state, &model_id, &stats);
    runtime_state.set_model_stats(model_id, stats);
}

fn publish_model_stats_updates(
    runtime_state: &PylonRuntimeState,
    updates: Vec<(String, CurrentModelStats)>,
) {
    for (model_id, stats) in updates {
        publish_model_stats_update(runtime_state, model_id, stats);
    }
}

async fn run_stats_collector(
    config: StatsCollectorConfig,
    observation_rx: flume::Receiver<RequestObservationEvent>,
    mut stats_update_rx: Option<flume::Receiver<StatsAggregatorUpdate>>,
    runtime_state: PylonRuntimeState,
    stop: CancellationToken,
) {
    let mut aggregator = StatsAggregator::new(config.clone(), runtime_state.clone());
    if let Some(last_mean_input_tps) = fixed_last_mean_input_tps(&config) {
        for model_id in &config.configured_model_ids {
            runtime_state.update_model_throughput(model_id, last_mean_input_tps);
            let stats = aggregator.snapshot(model_id);
            publish_model_stats_update(&runtime_state, model_id.clone(), stats);
        }
    }
    let http_client = reqwest::Client::new();
    let mut kv_cache_poll = tokio::time::interval(config.kv_cache_poll_interval);
    kv_cache_poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut engine_stats_sweep = tokio::time::interval(config.engine_stats_sweep_interval);
    engine_stats_sweep.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut stats_aggregator_updated_models = Vec::with_capacity(2);
    let mut stats_aggregator_latest_models = IndexMap::with_capacity(2);

    'collector: loop {
        tokio::select! {
            _ = stop.cancelled() => break 'collector,
            event = observation_rx.recv_async() => {
                let Ok(event) = event else {
                    break 'collector;
                };
                let updated_models = if aggregator.openai_fallback_stats_enabled() {
                    aggregator.apply_fallback_observation(&event)
                } else {
                    aggregator.apply_stream_observation(&event)
                };
                publish_model_stats_updates(&runtime_state, updated_models);
            }
            update = async {
                match &stats_update_rx {
                    Some(rx) => rx.recv_async().await.ok(),
                    None => std::future::pending().await,
                }
            } => {
                let Some(update) = update else {
                    stats_update_rx = None;
                    continue;
                };
                if aggregator.apply_control_update(&update) {
                    continue;
                }
                stats_aggregator_updated_models.clear();
                aggregator.apply_update_into(update, &mut stats_aggregator_updated_models);
                if let Some(rx) = &stats_update_rx {
                    while let Ok(update) = rx.try_recv() {
                        if aggregator.apply_control_update(&update) {
                            continue;
                        }
                        aggregator.apply_update_into(update, &mut stats_aggregator_updated_models);
                    }
                }
                retain_latest_model_updates(
                    &mut stats_aggregator_updated_models,
                    &mut stats_aggregator_latest_models,
                );
                if let Some(metrics) = runtime_state.metrics() {
                    metrics.observe_engine_stats_live_requests(
                        "engine_stats_stream",
                        aggregator.live_request_count(),
                    );
                    metrics.observe_engine_stats_model_states(
                        "engine_stats_stream",
                        aggregator.model_state_count(),
                    );
                }
                publish_model_stats_updates(
                    &runtime_state,
                    std::mem::take(&mut stats_aggregator_updated_models),
                );
            }
            _ = engine_stats_sweep.tick() => {
                let updated_models = aggregator.sweep_stale(TokioInstant::now());
                if let Some(metrics) = runtime_state.metrics() {
                    metrics.observe_engine_stats_live_requests(
                        "engine_stats_stream",
                        aggregator.live_request_count(),
                    );
                }
                publish_model_stats_updates(&runtime_state, updated_models);
            }
            _ = kv_cache_poll.tick(), if config.kv_cache_stats_url.is_some() => {
                let Some(kv_cache) = stop
                    .run_until_cancelled(poll_kv_cache_stats(&config, &http_client))
                    .await
                else {
                    break 'collector;
                };
                let Some(kv_cache) = kv_cache else {
                    continue;
                };
                if kv_cache.model.is_empty() {
                    tracing::warn!("dropping KV-cache stats without model id");
                    continue;
                }
                let model_id = kv_cache.model.clone();
                let Some((model_id, updated_stats)) = aggregator.apply_kv_cache_stats(kv_cache)
                else {
                    tracing::warn!(
                        model_id,
                        configured_models = ?config.configured_model_ids,
                        "dropping KV-cache stats for unconfigured model"
                    );
                    continue;
                };
                publish_model_stats_update(&runtime_state, model_id, updated_stats);
            }
        }
    }
}

async fn poll_kv_cache_stats(
    config: &StatsCollectorConfig,
    http_client: &reqwest::Client,
) -> Option<KvCacheStatsSnapshot> {
    let url = config.kv_cache_stats_url.as_ref()?;
    let response = match http_client
        .get(url)
        .timeout(config.kv_cache_request_timeout)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(url, error = %error, "failed to poll KV-cache stats");
            return None;
        }
    };
    if !response.status().is_success() {
        tracing::warn!(url, status = %response.status(), "KV-cache stats endpoint returned non-success status");
        return None;
    }
    match response.json().await {
        Ok(stats) => Some(stats),
        Err(error) => {
            tracing::warn!(url, error = %error, "failed to parse KV-cache stats");
            None
        }
    }
}

fn observe_model_metric(
    runtime_state: &PylonRuntimeState,
    model_id: &str,
    stats: &CurrentModelStats,
) {
    let Some(metrics) = runtime_state.metrics() else {
        return;
    };

    metrics.observe_model_stats(model_id, stats);
}

fn retain_latest_model_updates(
    updates: &mut Vec<(String, CurrentModelStats)>,
    latest_by_model: &mut IndexMap<String, CurrentModelStats>,
) {
    latest_by_model.clear();
    while let Some((model_id, stats)) = updates.pop() {
        latest_by_model.entry(model_id).or_insert(stats);
    }
    while let Some(update) = latest_by_model.pop() {
        updates.push(update);
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::super::aggregator::{KvCacheStatsSnapshot, StatsAggregator};
    use super::super::projection::fallback_update_from_observation;
    use super::*;
    use crate::request_observer::RequestObservationEndpoint;
    use crate::request_observer::RequestObservationState;
    use axum::{Json, Router, routing::get};
    use tokio::net::TcpListener;

    const MODEL_STATS_TEST_TIMEOUT: Duration = Duration::from_millis(500);

    fn observed_runtime(
        config: &StatsCollectorConfig,
    ) -> (PylonRuntimeState, flume::Receiver<RequestObservationEvent>) {
        PylonRuntimeState::observed(
            stargate_proto::pb::InferenceServerStatus::Unknown,
            &[],
            config.observation_channel_capacity,
            None,
        )
    }

    fn observed_runtime_with_metrics(
        config: &StatsCollectorConfig,
        metrics: Arc<PylonMetrics>,
    ) -> (PylonRuntimeState, flume::Receiver<RequestObservationEvent>) {
        PylonRuntimeState::observed(
            stargate_proto::pb::InferenceServerStatus::Unknown,
            &[],
            config.observation_channel_capacity,
            Some(metrics),
        )
    }

    fn apply_fallback_observation(
        aggregator: &mut StatsAggregator,
        observation: &RequestObservation,
    ) -> Vec<(String, CurrentModelStats)> {
        let event = aggregator
            .runtime_state
            .transition_request_observation(observation.clone());
        aggregator.apply_fallback_observation(&event)
    }

    fn apply_stream_observation(
        aggregator: &mut StatsAggregator,
        observation: &RequestObservation,
    ) -> Vec<(String, CurrentModelStats)> {
        let event = aggregator
            .runtime_state
            .transition_request_observation(observation.clone());
        aggregator.apply_stream_observation(&event)
    }

    fn completed_observation(
        input_tokens: u64,
        output_messages: u64,
        output_tokens: u64,
        time_to_first_output: Duration,
        total_duration: Duration,
    ) -> RequestObservation {
        RequestObservation {
            endpoint: crate::request_observer::RequestObservationEndpoint::ChatCompletions,
            request_id: "req-1".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages,
            output_tokens,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::Complete,
            time_to_response_headers: Some(Duration::from_millis(20)),
            time_to_first_output: Some(time_to_first_output),
            time_to_first_token: Some(time_to_first_output),
            total_duration,
        }
    }

    fn completed_embeddings_observation(
        input_tokens: u64,
        embedding_items: u64,
        time_to_response_headers: Duration,
        total_duration: Duration,
    ) -> RequestObservation {
        RequestObservation {
            endpoint: crate::request_observer::RequestObservationEndpoint::Embeddings,
            request_id: "req-embedding".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens,
            embedding_items,
            embedding_items_observed: true,
            upstream_status: Some(200),
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::Complete,
            time_to_response_headers: Some(time_to_response_headers),
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration,
        }
    }

    fn active_chat_observation(
        request_id: &str,
        state: RequestObservationState,
    ) -> RequestObservation {
        let time_to_first_output = (state == RequestObservationState::OutputGeneration)
            .then_some(Duration::from_millis(50));
        RequestObservation {
            endpoint: crate::request_observer::RequestObservationEndpoint::ChatCompletions,
            request_id: request_id.to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 32,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages: 1,
            output_tokens: 2,
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
            state,
            time_to_response_headers: Some(Duration::from_millis(10)),
            time_to_first_output,
            time_to_first_token: time_to_first_output,
            total_duration: Duration::from_millis(100),
        }
    }

    #[test]
    fn latest_model_update_retention_keeps_last_snapshot_per_model() {
        let mut updates = vec![
            (
                "model-a".to_string(),
                CurrentModelStats {
                    output_tps: 1.0,
                    ..Default::default()
                },
            ),
            (
                "model-b".to_string(),
                CurrentModelStats {
                    output_tps: 2.0,
                    ..Default::default()
                },
            ),
            (
                "model-a".to_string(),
                CurrentModelStats {
                    output_tps: 3.0,
                    ..Default::default()
                },
            ),
            (
                "model-c".to_string(),
                CurrentModelStats {
                    output_tps: 4.0,
                    ..Default::default()
                },
            ),
            (
                "model-b".to_string(),
                CurrentModelStats {
                    output_tps: 5.0,
                    ..Default::default()
                },
            ),
        ];
        let mut latest_by_model = indexmap::IndexMap::new();

        retain_latest_model_updates(&mut updates, &mut latest_by_model);

        assert_eq!(updates.len(), 3);
        assert_eq!(updates[0].0, "model-a");
        assert_eq!(updates[0].1.output_tps, 3.0);
        assert_eq!(updates[1].0, "model-c");
        assert_eq!(updates[1].1.output_tps, 4.0);
        assert_eq!(updates[2].0, "model-b");
        assert_eq!(updates[2].1.output_tps, 5.0);
        assert!(latest_by_model.is_empty());
    }

    fn stream_counter_update(
        request_id: &str,
        tokens_processed: u64,
        tokens_generated: u64,
        finished: bool,
        observed_at: TokioInstant,
    ) -> StatsAggregatorUpdate {
        StatsAggregatorUpdate::RequestCounters(RequestCounterUpdate {
            source: StatsUpdateSource::EngineStatsStream,
            request_id: request_id.to_string(),
            model_id: "model-a".to_string(),
            tokens_processed: Some(tokens_processed),
            tokens_generated: Some(tokens_generated),
            finished,
            observed_at,
        })
    }

    fn stream_counter_partial_update(
        request_id: &str,
        tokens_processed: Option<u64>,
        tokens_generated: Option<u64>,
        finished: bool,
        observed_at: TokioInstant,
    ) -> StatsAggregatorUpdate {
        StatsAggregatorUpdate::RequestCounters(RequestCounterUpdate {
            source: StatsUpdateSource::EngineStatsStream,
            request_id: request_id.to_string(),
            model_id: "model-a".to_string(),
            tokens_processed,
            tokens_generated,
            finished,
            observed_at,
        })
    }

    fn fallback_counter_update(
        request_id: &str,
        tokens_processed: u64,
        tokens_generated: u64,
        finished: bool,
        observed_at: TokioInstant,
    ) -> StatsAggregatorUpdate {
        StatsAggregatorUpdate::RequestCounters(RequestCounterUpdate {
            source: StatsUpdateSource::OpenAiFallback,
            request_id: request_id.to_string(),
            model_id: "model-a".to_string(),
            tokens_processed: Some(tokens_processed),
            tokens_generated: Some(tokens_generated),
            finished,
            observed_at,
        })
    }

    fn counter_update_for_model(
        source: StatsUpdateSource,
        request_id: &str,
        model_id: &str,
        tokens_processed: u64,
        tokens_generated: u64,
        finished: bool,
        observed_at: TokioInstant,
    ) -> StatsAggregatorUpdate {
        StatsAggregatorUpdate::RequestCounters(RequestCounterUpdate {
            source,
            request_id: request_id.to_string(),
            model_id: model_id.to_string(),
            tokens_processed: Some(tokens_processed),
            tokens_generated: Some(tokens_generated),
            finished,
            observed_at,
        })
    }

    async fn wait_for_model_stats(
        runtime_state: &PylonRuntimeState,
        model_id: &str,
        context: &str,
        predicate: impl Fn(&CurrentModelStats) -> bool,
    ) -> CurrentModelStats {
        tokio::time::timeout(MODEL_STATS_TEST_TIMEOUT, async {
            let mut poll = tokio::time::interval(Duration::from_millis(1));
            loop {
                poll.tick().await;
                if let Some(stats) = runtime_state.model_stats(model_id)
                    && predicate(&stats)
                {
                    return stats;
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{context}"))
    }

    #[test]
    fn stats_stream_cumulative_request_counters_drive_stats_aggregator() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-a", 0, 0, false, start));
        let updates = aggregator.apply_update(stream_counter_update(
            "req-a",
            10,
            4,
            false,
            start + Duration::from_millis(100),
        ));
        let stats = updates
            .into_iter()
            .find(|(model_id, _)| model_id == "model-a")
            .expect("model stats should update")
            .1;

        assert_eq!(stats.output_tps, 40.0);
        assert_eq!(stats.max_output_tps, 40.0);
        assert_eq!(stats.stats_sources, vec!["engine_stats_stream".to_string()]);

        for tick in 2..=5 {
            let updates = aggregator.apply_update(stream_counter_update(
                "req-a",
                tick * 10,
                4,
                false,
                start + Duration::from_millis(tick * 100),
            ));
            if tick < 5 {
                continue;
            }
            let stats = updates
                .into_iter()
                .find(|(model_id, _)| model_id == "model-a")
                .expect("fifth input sample should publish sticky mean")
                .1;
            assert_eq!(stats.last_mean_input_tps, 100.0);
        }
    }

    #[test]
    fn fixed_input_tps_is_preserved_across_engine_stats_updates() {
        let config = StatsCollectorConfig {
            fixed_last_mean_input_tps: Some(2_200.0),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let stats = aggregator
            .apply_update(stream_counter_update("req-a", 0, 0, false, start))
            .pop()
            .expect("first stream update should publish source labels")
            .1;
        assert_eq!(stats.last_mean_input_tps, 2_200.0);

        let mut published = None;
        for tick in 1..=5 {
            published = aggregator
                .apply_update(stream_counter_update(
                    "req-a",
                    tick * 10,
                    0,
                    false,
                    start + Duration::from_millis(tick * 100),
                ))
                .pop()
                .map(|(_, stats)| stats)
                .or(published);
        }
        assert_eq!(
            published
                .expect("sufficient input samples should publish stats")
                .last_mean_input_tps,
            2_200.0
        );
    }

    #[test]
    fn first_engine_stream_counter_without_zero_baseline_contributes_tps() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(100),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-first-output",
                0,
                10,
                true,
                start,
            ))
            .pop()
            .expect("first output counter should publish stats")
            .1;
        assert_eq!(stats.output_tps, 100.0);
        assert_eq!(stats.max_output_tps, 100.0);

        let mut latest = None;
        for index in 0..5 {
            latest = aggregator
                .apply_update(stream_counter_update(
                    &format!("req-first-input-{index}"),
                    10,
                    0,
                    true,
                    start + Duration::from_secs(index + 1),
                ))
                .pop()
                .map(|(_, stats)| stats);
        }
        let stats = latest.expect("fifth first input counter should publish mean input stats");
        assert_eq!(stats.last_mean_input_tps, 100.0);
    }

    #[test]
    fn first_post_baseline_engine_stream_delta_under_floor_contributes_tps() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(100),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let label_stats = aggregator
            .apply_update(stream_counter_update("req-fast", 0, 0, false, start))
            .pop()
            .expect("first engine stream event should publish source labels")
            .1;
        assert_eq!(
            label_stats.stats_sources,
            vec!["engine_stats_stream".to_string()]
        );
        assert_eq!(label_stats.output_tps, 0.0);

        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-fast",
                0,
                10,
                true,
                start + Duration::from_millis(1),
            ))
            .pop()
            .expect("first real counter delta should publish stats");
        assert_eq!(stats.1.output_tps, 100.0);
        assert_eq!(stats.1.max_output_tps, 100.0);
    }

    #[test]
    fn engine_stream_sub_floor_deltas_accumulate_after_fast_first_sample() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(10),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-live", 0, 0, false, start));
        let first_stats = aggregator
            .apply_update(stream_counter_update(
                "req-live",
                0,
                1,
                false,
                start + Duration::from_millis(1),
            ))
            .pop()
            .expect("first fast counter delta should publish with the duration floor")
            .1;
        assert_eq!(first_stats.output_tps, 100.0);
        assert_eq!(first_stats.max_output_tps, 100.0);

        for tick in 2..10 {
            let updates = aggregator.apply_update(stream_counter_update(
                "req-live",
                0,
                tick,
                false,
                start + Duration::from_millis(tick),
            ));
            assert!(
                updates.is_empty(),
                "sub-floor deltas should accumulate without publishing noisy snapshots"
            );
        }

        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-live",
                0,
                11,
                false,
                start + Duration::from_millis(11),
            ))
            .pop()
            .expect("accumulated sub-floor deltas should publish once the sample window is valid")
            .1;
        assert_eq!(stats.max_output_tps, 1_000.0);
        assert_eq!(stats.output_tps, 550.0);
    }

    #[test]
    fn engine_stream_missing_counter_fields_do_not_sample_stale_dimensions() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(10),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_partial_update(
            "req-partial",
            None,
            Some(0),
            false,
            start,
        ));
        let first_stats = aggregator
            .apply_update(stream_counter_partial_update(
                "req-partial",
                None,
                Some(1),
                false,
                start + Duration::from_millis(1),
            ))
            .pop()
            .expect("first output counter should publish with the duration floor")
            .1;
        assert_eq!(first_stats.output_tps, 100.0);

        assert!(
            aggregator
                .apply_update(stream_counter_partial_update(
                    "req-partial",
                    None,
                    Some(2),
                    false,
                    start + Duration::from_millis(2),
                ))
                .is_empty(),
            "second output counter is still below the duration floor"
        );

        let input_only_updates = aggregator.apply_update(stream_counter_partial_update(
            "req-partial",
            Some(1),
            None,
            false,
            start + Duration::from_millis(11),
        ));
        assert!(
            input_only_updates.is_empty(),
            "input-only updates must not publish a stale output TPS sample"
        );
    }

    #[test]
    fn engine_stream_sub_minimum_deltas_accumulate_until_publishable() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(10),
            min_output_tokens: 5,
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-min", 0, 0, false, start));
        for tick in 1..10 {
            let updates = aggregator.apply_update(stream_counter_update(
                "req-min",
                0,
                tick,
                false,
                start + Duration::from_millis(tick),
            ));
            assert!(
                updates.is_empty(),
                "tokens below the minimum or duration floor should remain accumulated"
            );
        }

        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-min",
                0,
                10,
                false,
                start + Duration::from_millis(10),
            ))
            .pop()
            .expect("accumulated tokens should publish after reaching the floor")
            .1;
        assert_eq!(stats.output_tps, 1_000.0);
        assert_eq!(stats.max_output_tps, 1_000.0);
    }

    #[test]
    fn fallback_and_stream_cumulative_counters_share_stats_math() {
        let config = StatsCollectorConfig::default();
        let start = TokioInstant::now();
        let mut stream_aggregator =
            StatsAggregator::new(config.clone(), PylonRuntimeState::default());
        let mut fallback_aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        for tick in 0..=5 {
            let observed_at = start + Duration::from_millis(tick * 100);
            let tokens_processed = tick * 10;
            let tokens_generated = tick * 2;
            let stream_updates = stream_aggregator.apply_update(stream_counter_update(
                "req-shared",
                tokens_processed,
                tokens_generated,
                tick == 5,
                observed_at,
            ));
            let fallback_updates = fallback_aggregator.apply_update(fallback_counter_update(
                "req-shared",
                tokens_processed,
                tokens_generated,
                tick == 5,
                observed_at,
            ));
            if tick == 0 {
                assert_eq!(stream_updates.len(), 1);
                assert!(fallback_updates.is_empty());
                continue;
            }
            assert_eq!(stream_updates.len(), fallback_updates.len());
            for ((_, stream_stats), (_, fallback_stats)) in
                stream_updates.iter().zip(fallback_updates.iter())
            {
                assert_eq!(
                    stream_stats.last_mean_input_tps,
                    fallback_stats.last_mean_input_tps
                );
                assert_eq!(stream_stats.output_tps, fallback_stats.output_tps);
                assert_eq!(stream_stats.max_output_tps, fallback_stats.max_output_tps);
            }
        }
    }

    #[test]
    fn request_counter_model_reset_finalizes_without_late_replay() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let original_model = aggregator
            .apply_update(counter_update_for_model(
                StatsUpdateSource::EngineStatsStream,
                "req-reused",
                "model-a",
                0,
                0,
                false,
                start,
            ))
            .pop()
            .expect("first stream event should publish model-a source labels");
        assert_eq!(original_model.0, "model-a");
        assert_eq!(original_model.1.output_tps, 0.0);
        assert_eq!(aggregator.live_request_count(), 1);

        let replacement_model = aggregator
            .apply_update(counter_update_for_model(
                StatsUpdateSource::EngineStatsStream,
                "req-reused",
                "model-b",
                0,
                0,
                false,
                start + Duration::from_millis(50),
            ))
            .pop()
            .expect("model change should reset request state and publish model-b source labels");
        assert_eq!(replacement_model.0, "model-b");
        assert_eq!(replacement_model.1.output_tps, 0.0);
        assert_eq!(aggregator.live_request_count(), 1);

        let finalized = aggregator
            .apply_update(counter_update_for_model(
                StatsUpdateSource::OpenAiFallback,
                "req-reused",
                "model-b",
                10,
                4,
                true,
                start + Duration::from_millis(150),
            ))
            .pop()
            .expect("fallback finalization should publish the replacement model snapshot");
        assert_eq!(finalized.0, "model-b");
        assert_eq!(finalized.1.output_tps, 40.0);
        assert_eq!(finalized.1.max_output_tps, 40.0);
        assert_eq!(
            finalized.1.stats_sources,
            vec!["engine_stats_stream".to_string()]
        );
        assert_eq!(aggregator.live_request_count(), 0);

        let late_replay = aggregator.apply_update(counter_update_for_model(
            StatsUpdateSource::EngineStatsStream,
            "req-reused",
            "model-b",
            10,
            8,
            true,
            start + Duration::from_millis(200),
        ));
        assert!(
            late_replay.is_empty(),
            "late stream replay after fallback finalization must not double-count"
        );
        assert_eq!(aggregator.live_request_count(), 0);
    }

    #[test]
    fn dirty_fallback_counter_snapshots_preserve_lifecycle_load() {
        let config = StatsCollectorConfig::default();
        let start = TokioInstant::now();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = active_chat_observation(
            "req-fallback-live-load",
            RequestObservationState::OutputGeneration,
        );

        apply_stream_observation(&mut aggregator, &observation);
        assert!(
            aggregator
                .apply_update(fallback_counter_update(
                    "req-fallback-live-load",
                    0,
                    2,
                    false,
                    start
                ))
                .is_empty(),
            "first fallback counter is a baseline"
        );

        let stats = aggregator
            .apply_update(fallback_counter_update(
                "req-fallback-live-load",
                0,
                4,
                false,
                start + Duration::from_millis(100),
            ))
            .pop()
            .expect("second fallback counter should publish output TPS")
            .1;

        assert_eq!(stats.output_tps, 20.0);
        assert_eq!(stats.num_running_queries, 1);
        assert_eq!(stats.total_query_input_size, 32);
        assert_eq!(stats.input_processing_queries, 0);
        assert_eq!(stats.output_generation_queries, 1);
    }

    #[test]
    fn engine_stream_snapshots_preserve_local_kv_cache_stats() {
        let config = StatsCollectorConfig::default();
        let start = TokioInstant::now();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        aggregator.apply_kv_cache_stats(KvCacheStatsSnapshot {
            model: "model-a".to_string(),
            kv_cache_capacity_tokens: 1_000,
            kv_cache_used_tokens: 400,
            kv_cache_free_tokens: 600,
        });

        let stats = aggregator
            .apply_update(stream_counter_update("req-stream-kv", 0, 10, true, start))
            .pop()
            .expect("stream counter should publish stats with owned KV state")
            .1;

        assert_eq!(stats.kv_cache_capacity_tokens, 1_000);
        assert_eq!(stats.kv_cache_used_tokens, 400);
        assert_eq!(stats.kv_cache_free_tokens, 600);
        assert_eq!(
            stats.stats_capabilities,
            vec![
                "model.throughput.engine_stream".to_string(),
                "machine.kv_cache.http".to_string(),
            ]
        );
        assert_eq!(
            stats.stats_sources,
            vec![
                "engine_stats_stream".to_string(),
                "kv_cache_stats".to_string(),
            ]
        );
    }

    #[test]
    fn stats_aggregator_owns_lifecycle_engine_and_kv_state() {
        let config = StatsCollectorConfig {
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = active_chat_observation(
            "req-single-owner-lifecycle",
            RequestObservationState::OutputGeneration,
        );

        let lifecycle_stats = apply_stream_observation(&mut aggregator, &observation)
            .pop()
            .expect("lifecycle observation should publish a snapshot")
            .1;
        assert_eq!(lifecycle_stats.num_running_queries, 1);
        assert_eq!(lifecycle_stats.output_generation_queries, 1);

        let kv_stats = aggregator
            .apply_kv_cache_stats(KvCacheStatsSnapshot {
                model: "model-a".to_string(),
                kv_cache_capacity_tokens: 1_000,
                kv_cache_used_tokens: 400,
                kv_cache_free_tokens: 600,
            })
            .expect("KV stats should publish a snapshot")
            .1;
        assert_eq!(kv_stats.num_running_queries, 1);
        assert_eq!(kv_stats.kv_cache_capacity_tokens, 1_000);

        let start = TokioInstant::now();
        aggregator.apply_update(stream_counter_update(
            "req-single-owner-stream",
            0,
            0,
            false,
            start,
        ));
        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-single-owner-stream",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .pop()
            .expect("engine counters should publish the complete owned snapshot")
            .1;

        assert_eq!(stats.output_tps, 10.0);
        assert_eq!(stats.num_running_queries, 1);
        assert_eq!(stats.output_generation_queries, 1);
        assert_eq!(stats.kv_cache_capacity_tokens, 1_000);
        assert_eq!(
            stats.stats_sources,
            vec![
                "engine_stats_stream".to_string(),
                "kv_cache_stats".to_string(),
            ]
        );
    }

    #[test]
    fn engine_stats_model_state_count_excludes_lifecycle_and_kv_only_models() {
        let config = StatsCollectorConfig {
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        apply_stream_observation(
            &mut aggregator,
            &active_chat_observation(
                "req-lifecycle-only",
                RequestObservationState::OutputGeneration,
            ),
        );
        aggregator.apply_kv_cache_stats(KvCacheStatsSnapshot {
            model: "model-b".to_string(),
            kv_cache_capacity_tokens: 1_000,
            kv_cache_used_tokens: 400,
            kv_cache_free_tokens: 600,
        });
        assert_eq!(aggregator.model_state_count(), 0);

        aggregator.apply_update(stream_counter_update(
            "req-engine-state",
            0,
            0,
            false,
            TokioInstant::now(),
        ));
        assert_eq!(aggregator.model_state_count(), 1);
    }

    #[test]
    fn stats_aggregator_keeps_embeddings_observation_with_stream_output_stats() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-stream", 0, 0, false, start));
        let stats = aggregator
            .apply_update(stream_counter_update(
                "req-stream",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .pop()
            .expect("stream output counters should publish stats")
            .1;
        assert_eq!(stats.output_tps, 10.0);
        assert_eq!(stats.max_output_tps, 10.0);

        let mut latest = None;
        for index in 0..5 {
            let observation = RequestObservation {
                request_id: format!("req-embedding-{index}"),
                ..completed_embeddings_observation(
                    20,
                    2,
                    Duration::from_secs(1),
                    Duration::from_secs(2),
                )
            };
            latest = apply_stream_observation(&mut aggregator, &observation)
                .into_iter()
                .find(|(model_id, _)| model_id == "model-a")
                .map(|(_, stats)| stats);
        }

        let stats = latest.expect("fifth embeddings observation should publish stats");
        assert_eq!(stats.output_tps, 10.0);
        assert_eq!(stats.max_output_tps, 10.0);
        assert_eq!(stats.last_mean_input_tps, 0.0);
        assert_eq!(stats.embedding_item_tps, 2.0);
        assert_eq!(stats.max_embedding_item_tps, 2.0);
        assert_eq!(stats.stats_sources, vec!["engine_stats_stream".to_string()]);
        assert!(
            !stats
                .stats_capabilities
                .contains(&"request.embeddings_item_throughput".to_string())
        );
    }

    #[test]
    fn stream_mode_embeddings_do_not_double_count_stream_input_tps() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(100),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let mut latest = None;
        for index in 0..5 {
            latest = aggregator
                .apply_update(stream_counter_update(
                    &format!("req-stream-input-{index}"),
                    10,
                    0,
                    true,
                    start + Duration::from_secs(index + 1),
                ))
                .pop()
                .map(|(_, stats)| stats);
        }
        let stats = latest.expect("stream input counters should publish mean input stats");
        assert_eq!(stats.last_mean_input_tps, 100.0);

        let mut latest = None;
        for index in 0..5 {
            let observation = RequestObservation {
                request_id: format!("req-embedding-{index}"),
                ..completed_embeddings_observation(
                    20,
                    2,
                    Duration::from_secs(1),
                    Duration::from_secs(2),
                )
            };
            latest = apply_stream_observation(&mut aggregator, &observation)
                .into_iter()
                .find(|(model_id, _)| model_id == "model-a")
                .map(|(_, stats)| stats);
        }

        let stats = latest.expect("embeddings observations should publish item throughput");
        assert_eq!(stats.last_mean_input_tps, 100.0);
        assert_eq!(stats.embedding_item_tps, 2.0);
        assert_eq!(stats.max_embedding_item_tps, 2.0);
    }

    #[tokio::test]
    async fn stats_collector_enables_openai_fallback_only_after_control_update() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let (runtime_state, observation_rx) =
            observed_runtime_with_metrics(&config, metrics.clone());
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
            stop.clone(),
        ));

        let fallback_observation = RequestObservation {
            request_id: "req-fallback-disabled".to_string(),
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
            ..completed_observation(20, 1, 10, Duration::from_secs(1), Duration::from_secs(3))
        };
        runtime_state.observe_request(fallback_observation);
        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "fallback-disabled observation should publish lifecycle-only stats",
            |_| true,
        )
        .await;
        assert_eq!(stats.output_tps, 0.0);
        assert!(!stats.stats_sources.contains(&"chunk_usage".to_string()));

        stats_update_tx
            .send_async(StatsAggregatorUpdate::EnableOpenAiFallback)
            .await
            .expect("collector should receive fallback control update");
        for _ in 0..50 {
            let body = metrics.gather_text().expect("metrics should encode");
            if body.contains(
                r#"pylon_engine_stats_source_transitions_total{from="engine_stats_stream",reason="unsupported",to="openai_fallback"} 1"#,
            ) {
                break;
            }
            tokio::task::yield_now().await;
        }
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(
            body.contains(
                r#"pylon_engine_stats_source_transitions_total{from="engine_stats_stream",reason="unsupported",to="openai_fallback"} 1"#
            ),
            "collector should process fallback control update before fallback observations are accepted"
        );
        runtime_state.observe_request(RequestObservation {
            request_id: "req-fallback-enabled".to_string(),
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
            ..completed_observation(20, 1, 10, Duration::from_secs(1), Duration::from_secs(3))
        });

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "fallback-enabled observation should publish model stats",
            |stats| stats.output_tps == 5.0,
        )
        .await;
        assert_eq!(stats.output_tps, 5.0);
        assert!(stats.stats_sources.contains(&"chunk_usage".to_string()));

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn stats_collector_keeps_lifecycle_load_when_fallback_stats_disabled() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
            stop.clone(),
        ));

        let start = TokioInstant::now();
        stats_update_tx
            .send_async(stream_counter_update(
                "req-prior-stream",
                0,
                0,
                false,
                start,
            ))
            .await
            .expect("collector should receive stream start");
        stats_update_tx
            .send_async(stream_counter_update(
                "req-prior-stream",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .await
            .expect("collector should receive stream finish");
        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "stream finish should publish stats",
            |stats| stats.output_tps == 10.0,
        )
        .await;

        runtime_state.observe_request(active_chat_observation(
            "req-stream-lifecycle",
            RequestObservationState::InputProcessing,
        ));

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "stream mode lifecycle observation should publish stats",
            |stats| stats.input_processing_queries == 1,
        )
        .await;
        assert_eq!(stats.num_running_queries, 1);
        assert_eq!(stats.queue_size, 1);
        assert_eq!(stats.queued_input_size, 32);
        assert_eq!(stats.total_query_input_size, 32);
        assert_eq!(stats.input_processing_queries, 1);
        assert_eq!(stats.output_generation_queries, 0);
        assert_eq!(stats.output_tps, 10.0);

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn stats_collector_accepts_late_stream_finish_after_terminal_observation() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
            stop.clone(),
        ));

        let start = TokioInstant::now();
        stats_update_tx
            .send_async(stream_counter_update("req-stream-race", 0, 0, false, start))
            .await
            .expect("collector should receive stream start");

        let mut terminal_observation =
            completed_observation(32, 1, 10, Duration::from_millis(50), Duration::from_secs(1));
        terminal_observation.request_id = "req-stream-race".to_string();
        runtime_state.observe_request(terminal_observation);
        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "terminal observation should publish lifecycle stats",
            |_| true,
        )
        .await;

        stats_update_tx
            .send_async(stream_counter_update(
                "req-stream-race",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .await
            .expect("collector should receive late stream finish");

        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "late stream finish should publish stats",
            |stats| stats.output_tps == 10.0 && stats.max_output_tps == 10.0,
        )
        .await;

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn stats_collector_helper_defaults_stats_stream_to_authoritative() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let collector = start_stats_collector_with_engine_stats(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
        );

        let start = TokioInstant::now();
        stats_update_tx
            .send_async(stream_counter_update(
                "req-helper-stream",
                0,
                0,
                false,
                start,
            ))
            .await
            .expect("collector should receive stream start");

        let mut terminal_observation =
            completed_observation(32, 0, 0, Duration::from_millis(50), Duration::from_secs(1));
        terminal_observation.request_id = "req-helper-stream".to_string();
        runtime_state.observe_request(terminal_observation);

        stats_update_tx
            .send_async(stream_counter_update(
                "req-helper-stream",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .await
            .expect("collector should receive delayed stream finish");

        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "delayed stream finish should publish stats",
            |stats| stats.output_tps == 10.0 && stats.max_output_tps == 10.0,
        )
        .await;

        collector.shutdown().await;
    }

    #[tokio::test(start_paused = true)]
    async fn stats_collector_sweeps_stream_state_after_stats_receiver_closes() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            engine_stats_request_ttl: Duration::from_secs(1),
            engine_stats_model_ttl: Duration::from_secs(60),
            engine_stats_sweep_interval: Duration::from_secs(1),
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let (runtime_state, observation_rx) =
            observed_runtime_with_metrics(&config, metrics.clone());
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
            stop.clone(),
        ));

        let start = TokioInstant::now();
        stats_update_tx
            .send_async(stream_counter_update(
                "req-stream-stale",
                0,
                0,
                false,
                start,
            ))
            .await
            .expect("collector should receive stream start");
        drop(stats_update_tx);

        let label_stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "initial stream label snapshot should publish",
            |stats| stats.stats_sources == ["engine_stats_stream"],
        )
        .await;
        assert_eq!(
            label_stats.stats_sources,
            vec!["engine_stats_stream".to_string()]
        );

        tokio::time::advance(Duration::from_secs(2)).await;
        for _ in 0..50 {
            let body = metrics.gather_text().expect("metrics should encode");
            if body.contains(r#"pylon_engine_stats_live_requests{source="engine_stats_stream"} 0"#)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(
            body.contains(r#"pylon_engine_stats_live_requests{source="engine_stats_stream"} 0"#)
        );

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn fallback_counter_snapshots_preserve_lifecycle_load() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            None,
            runtime_state.clone(),
            stop.clone(),
        ));

        runtime_state.observe_request(active_chat_observation(
            "req-fallback-live-load",
            RequestObservationState::OutputGeneration,
        ));

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "fallback observation should publish stats",
            |stats| stats.output_generation_queries == 1,
        )
        .await;
        assert_eq!(stats.num_running_queries, 1);
        assert_eq!(stats.total_query_input_size, 32);
        assert_eq!(stats.input_processing_queries, 0);
        assert_eq!(stats.output_generation_queries, 1);

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn terminal_only_fallback_counter_does_not_clear_observed_output_tps() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            None,
            runtime_state.clone(),
            stop.clone(),
        ));

        runtime_state.observe_request(RequestObservation {
            request_id: "req-terminal-only-fallback".to_string(),
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
            ..completed_observation(20, 1, 10, Duration::from_secs(1), Duration::from_secs(3))
        });

        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "terminal observation should publish stats",
            |stats| stats.output_tps == 5.0,
        )
        .await;

        for _ in 0..20 {
            tokio::task::yield_now().await;
        }

        let stats = runtime_state
            .model_stats("model-a")
            .expect("terminal observation should leave model stats");
        assert_eq!(stats.output_tps, 5.0);

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[tokio::test]
    async fn stats_collector_keeps_embeddings_observation_when_fallback_stats_disabled() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 32,
            openai_fallback_stats_enabled: false,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let (stats_update_tx, stats_update_rx) = stats_aggregator_update_channel(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            Some(stats_update_rx),
            runtime_state.clone(),
            stop.clone(),
        ));

        let start = TokioInstant::now();
        stats_update_tx
            .send_async(stream_counter_update("req-stream", 0, 0, false, start))
            .await
            .expect("collector should receive stream start");
        stats_update_tx
            .send_async(stream_counter_update(
                "req-stream",
                0,
                10,
                true,
                start + Duration::from_secs(1),
            ))
            .await
            .expect("collector should receive stream finish");

        wait_for_model_stats(
            &runtime_state,
            "model-a",
            "collector should publish stream stats",
            |stats| stats.output_tps == 10.0 && stats.max_output_tps == 10.0,
        )
        .await;

        for index in 0..5 {
            runtime_state.observe_request(RequestObservation {
                request_id: format!("req-embedding-{index}"),
                ..completed_embeddings_observation(
                    20,
                    2,
                    Duration::from_secs(1),
                    Duration::from_secs(2),
                )
            });
        }

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "embeddings observations should publish stream-mode stats",
            |stats| stats.embedding_item_tps > 0.0,
        )
        .await;
        assert_eq!(stats.output_tps, 10.0);
        assert_eq!(stats.max_output_tps, 10.0);
        assert_eq!(stats.last_mean_input_tps, 0.0);
        assert_eq!(stats.embedding_item_tps, 2.0);
        assert_eq!(stats.stats_sources, vec!["engine_stats_stream".to_string()]);

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[test]
    fn stats_aggregator_ignores_regressions_and_post_finalize_events() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-final", 10, 2, false, start));
        aggregator.apply_update(stream_counter_update(
            "req-final",
            20,
            4,
            true,
            start + Duration::from_millis(100),
        ));
        assert_eq!(aggregator.live_request_count(), 0);

        let updates = aggregator.apply_update(stream_counter_update(
            "req-final",
            30,
            8,
            false,
            start + Duration::from_millis(200),
        ));
        assert!(updates.is_empty());

        aggregator.apply_update(stream_counter_update("req-live", 20, 4, false, start));
        let updates = aggregator.apply_update(stream_counter_update(
            "req-live",
            19,
            5,
            false,
            start + Duration::from_millis(100),
        ));
        assert!(updates.is_empty());
    }

    #[test]
    fn stats_aggregator_rejects_unconfigured_counter_models() {
        let config = StatsCollectorConfig {
            configured_model_ids: vec!["model-a".to_string()],
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        let updates = aggregator.apply_update(counter_update_for_model(
            StatsUpdateSource::EngineStatsStream,
            "req-unconfigured",
            "model-b",
            10,
            4,
            false,
            start,
        ));

        assert!(updates.is_empty());
        assert_eq!(aggregator.live_request_count(), 0);
        assert_eq!(
            aggregator.snapshot("model-b").stats_sources,
            Vec::<String>::new()
        );
    }

    #[test]
    fn fallback_terminal_observation_without_trusted_counters_finalizes_stream_request() {
        let mut observation = completed_observation(
            11,
            12,
            10,
            Duration::from_millis(100),
            Duration::from_millis(1_000),
        );
        observation.request_id = "req-stream-race".to_string();

        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();
        aggregator.apply_update(stream_counter_update("req-stream-race", 5, 3, false, start));
        assert_eq!(aggregator.live_request_count(), 1);

        let fallback_update =
            fallback_update_from_observation(&observation).expect("terminal update should exist");
        let stats = aggregator
            .apply_update(fallback_update)
            .pop()
            .expect("terminal request observation should publish the finalized stream snapshot")
            .1;

        assert_eq!(stats.stats_sources, vec!["engine_stats_stream".to_string()]);
        assert_eq!(aggregator.live_request_count(), 0);

        let updates = aggregator.apply_update(stream_counter_update(
            "req-stream-race",
            11,
            10,
            true,
            start + Duration::from_millis(100),
        ));
        assert!(
            updates.is_empty(),
            "post-finalization stream stats must not double-count"
        );
    }

    #[test]
    fn stats_aggregator_sweeps_stale_request_and_model_state() {
        let config = StatsCollectorConfig {
            engine_stats_request_ttl: Duration::from_secs(1),
            engine_stats_model_ttl: Duration::from_secs(1),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        for tick in 0..=5 {
            aggregator.apply_update(stream_counter_update(
                "req-stale",
                tick * 10,
                tick * 2,
                false,
                start + Duration::from_millis(tick * 100),
            ));
        }
        assert_eq!(aggregator.live_request_count(), 1);

        let updates = aggregator.sweep_stale(start + Duration::from_secs(2));
        assert_eq!(aggregator.live_request_count(), 0);
        let stats = updates
            .into_iter()
            .find(|(model_id, _)| model_id == "model-a")
            .expect("stale cleanup should publish a dirty model snapshot")
            .1;

        assert_eq!(stats.last_mean_input_tps, 100.0);
        assert_eq!(stats.output_tps, 0.0);
        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.queued_input_size, 0);
        assert_eq!(stats.num_running_queries, 0);
        assert_eq!(stats.input_processing_queries, 0);
        assert_eq!(stats.output_generation_queries, 0);
        assert_eq!(stats.stats_sources, vec!["engine_stats_stream".to_string()]);
    }

    #[test]
    fn stats_aggregator_tombstones_stale_request_before_late_finish() {
        let config = StatsCollectorConfig {
            engine_stats_request_ttl: Duration::from_secs(1),
            engine_stats_model_ttl: Duration::from_secs(60),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-stale-late", 0, 0, false, start));
        aggregator.apply_update(stream_counter_update(
            "req-stale-late",
            100,
            10,
            false,
            start + Duration::from_millis(100),
        ));
        assert_eq!(aggregator.live_request_count(), 1);

        let stale_updates = aggregator.sweep_stale(start + Duration::from_secs(2));
        assert_eq!(aggregator.live_request_count(), 0);
        assert!(
            stale_updates
                .iter()
                .any(|(model_id, _)| model_id == "model-a"),
            "stale cleanup should publish a dirty model snapshot"
        );

        let late_updates = aggregator.apply_update(stream_counter_update(
            "req-stale-late",
            100,
            20,
            true,
            start + Duration::from_millis(2_100),
        ));
        assert!(
            late_updates.is_empty(),
            "late cumulative finish after stale cleanup must not be replayed from zero"
        );
    }

    #[test]
    fn stats_aggregator_request_counter_identity_has_one_lifecycle_entry() {
        let config = StatsCollectorConfig {
            engine_stats_request_ttl: Duration::from_secs(1),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.apply_update(stream_counter_update("req-lifecycle", 0, 0, false, start));
        assert_eq!(aggregator.live_request_count(), 1);
        assert_eq!(aggregator.request_counter_identity_count(), 1);

        aggregator.apply_update(stream_counter_update(
            "req-lifecycle",
            10,
            2,
            true,
            start + Duration::from_millis(100),
        ));
        assert_eq!(aggregator.live_request_count(), 0);
        assert_eq!(aggregator.request_counter_identity_count(), 1);

        let late_updates = aggregator.apply_update(stream_counter_update(
            "req-lifecycle",
            20,
            4,
            true,
            start + Duration::from_millis(200),
        ));
        assert!(late_updates.is_empty());
        assert_eq!(aggregator.request_counter_identity_count(), 1);

        aggregator.sweep_stale(start + Duration::from_secs(2));
        assert_eq!(aggregator.request_counter_identity_count(), 0);
    }

    #[test]
    fn repeated_request_finalization_refreshes_tombstone_expiry() {
        let config = StatsCollectorConfig {
            engine_stats_request_ttl: Duration::from_secs(1),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();

        aggregator.finalize_request(FinalizeRequestUpdate::new(
            StatsUpdateSource::OpenAiFallback,
            "req-finalized-twice",
            start,
        ));
        aggregator.finalize_request(FinalizeRequestUpdate::new(
            StatsUpdateSource::OpenAiFallback,
            "req-finalized-twice",
            start + Duration::from_millis(800),
        ));

        aggregator.sweep_stale(start + Duration::from_millis(1_500));
        let late_updates = aggregator.apply_update(stream_counter_update(
            "req-finalized-twice",
            10,
            2,
            true,
            start + Duration::from_millis(1_600),
        ));

        assert!(late_updates.is_empty());
        assert_eq!(aggregator.request_counter_identity_count(), 1);

        aggregator.sweep_stale(start + Duration::from_secs(2));
        assert_eq!(aggregator.request_counter_identity_count(), 0);
    }

    #[test]
    fn stats_aggregator_keeps_bounded_request_state_for_many_cumulative_updates() {
        const REQUESTS: usize = 256;
        const EVENTS: usize = 10_000;

        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let start = TokioInstant::now();
        let mut latest = vec![(0u64, 0u64); REQUESTS];

        for index in 0..EVENTS {
            let request_index = index % REQUESTS;
            let step = (index / REQUESTS + 1) as u64;
            let tokens_processed = step * 8;
            let tokens_generated = step;
            latest[request_index] = (tokens_processed, tokens_generated);
            aggregator.apply_update(stream_counter_update(
                &format!("req-{request_index}"),
                tokens_processed,
                tokens_generated,
                false,
                start + Duration::from_millis(index as u64),
            ));
        }

        assert_eq!(aggregator.live_request_count(), REQUESTS);

        for (request_index, (tokens_processed, tokens_generated)) in latest.into_iter().enumerate()
        {
            aggregator.apply_update(stream_counter_update(
                &format!("req-{request_index}"),
                tokens_processed,
                tokens_generated,
                true,
                start + Duration::from_secs(60) + Duration::from_millis(request_index as u64),
            ));
        }

        assert_eq!(aggregator.live_request_count(), 0);
    }

    #[test]
    fn last_mean_input_tps_stays_sticky_without_new_samples() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        for request_index in 0..5 {
            apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-sticky-{request_index}"),
                    ..completed_observation(
                        20,
                        1,
                        1,
                        Duration::from_secs(2),
                        Duration::from_secs(2),
                    )
                },
            );
        }

        let stats = aggregator.snapshot("model-a");
        assert_eq!(stats.last_mean_input_tps, 10.0);
    }

    #[test]
    fn fallback_input_throughput_is_owned_by_stats_aggregator() {
        let config = StatsCollectorConfig::default();
        let runtime_state = PylonRuntimeState::default();
        let mut aggregator = StatsAggregator::new(config, runtime_state.clone());

        for request_index in 0..5 {
            let updates = apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-openai-stream-{request_index}"),
                    ..completed_observation(
                        50,
                        1,
                        8,
                        Duration::from_millis(500),
                        Duration::from_secs(1),
                    )
                },
            );
            assert_eq!(updates.len(), 1, "each observation should publish once");
            let stats = updates
                .into_iter()
                .find(|(model_id, _)| model_id == "model-a")
                .map(|(_, stats)| stats)
                .expect("lifecycle update should publish");

            assert_eq!(
                stats.last_mean_input_tps,
                if request_index < 4 { 0.0 } else { 100.0 }
            );
        }

        let distribution = &aggregator.per_model["model-a"].input_tps_distribution;
        assert_eq!(distribution.count, 5);
        assert_eq!(distribution.mean, 100.0);

        let _queued =
            runtime_state.track_request(&crate::request_observer::RequiredTunnelHeaders {
                request_id: "req-queued-after-fallback-samples".to_string(),
                routing_key: None,
                model_id: "model-a".to_string(),
                priority: 0,
                input_tokens: 50,
                accepted_at: std::time::Instant::now(),
            });
        runtime_state.transition_request_observation(active_chat_observation(
            "req-queued-after-fallback-samples",
            RequestObservationState::Queued,
        ));
        assert_eq!(
            runtime_state
                .snapshot_live_model("model-a")
                .queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(0, 320)]))
        );
    }

    #[test]
    fn fallback_input_throughput_ignores_non_terminal_observations() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        for state in [
            RequestObservationState::InputProcessing,
            RequestObservationState::OutputGeneration,
        ] {
            let updates = apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-live-{state:?}"),
                    state,
                    ..completed_observation(
                        50,
                        1,
                        8,
                        Duration::from_millis(500),
                        Duration::from_secs(1),
                    )
                },
            );
            assert_eq!(updates[0].1.last_mean_input_tps, 0.0);
        }

        assert_eq!(
            aggregator.per_model["model-a"].input_tps_distribution.count,
            0
        );
    }

    #[test]
    fn terminal_only_samples_use_request_duration_instead_of_tick_window() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        for request_index in 0..5 {
            apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-final-only-{request_index}"),
                    ..completed_observation(
                        100,
                        1,
                        1,
                        Duration::from_secs(2),
                        Duration::from_secs(2),
                    )
                },
            );
        }
        assert_eq!(aggregator.snapshot("model-a").last_mean_input_tps, 50.0);
        let distribution = &aggregator.per_model["model-a"].input_tps_distribution;
        assert_eq!(distribution.count, 5);
        assert_eq!(distribution.mean, 50.0);
    }

    #[test]
    fn terminal_only_samples_do_not_sum_same_tick_request_rates() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        for request_index in 0..5 {
            apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-final-only-sequential-{request_index}"),
                    ..completed_observation(
                        100,
                        1,
                        1,
                        Duration::from_millis(10),
                        Duration::from_millis(10),
                    )
                },
            );
        }
        assert_eq!(aggregator.snapshot("model-a").last_mean_input_tps, 10_000.0);
        let distribution = &aggregator.per_model["model-a"].input_tps_distribution;
        assert_eq!(distribution.count, 5);
        assert_eq!(distribution.mean, 10_000.0);
    }

    #[test]
    fn completed_request_stats_keep_exact_output_rate_formula() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation =
            completed_observation(120, 6, 30, Duration::from_secs(3), Duration::from_secs(9));

        let stats = apply_fallback_observation(&mut aggregator, &observation)
            .into_iter()
            .find(|(model_id, _)| model_id == "model-a")
            .unwrap()
            .1;

        assert_eq!(stats.last_mean_input_tps, 0.0);
        assert_eq!(stats.output_tps, 5.0);
        assert_eq!(stats.max_output_tps, 5.0);
    }

    #[test]
    fn ignores_observations_below_duration_floor() {
        let config = StatsCollectorConfig {
            duration_floor: Duration::from_millis(50),
            ..Default::default()
        };
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = completed_observation(
            20,
            4,
            8,
            Duration::from_millis(10),
            Duration::from_millis(20),
        );

        let stats = apply_fallback_observation(&mut aggregator, &observation);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.last_mean_input_tps, 0.0);
        assert_eq!(stats[0].1.output_tps, 0.0);
    }

    #[test]
    fn terminal_usage_chunks_use_first_output_for_output_tps() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = RequestObservation {
            time_to_first_token: Some(Duration::from_millis(5_995)),
            ..completed_observation(20, 4, 8, Duration::from_secs(2), Duration::from_secs(6))
        };

        let stats = apply_fallback_observation(&mut aggregator, &observation);

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.output_tps, 2.0);
        assert_eq!(stats[0].1.max_output_tps, 2.0);
    }

    #[test]
    fn embeddings_stats_update_last_mean_input_tps_without_claiming_output_tps() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation =
            completed_embeddings_observation(20, 4, Duration::from_secs(2), Duration::from_secs(4));

        for request_index in 0..4 {
            let stats = apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-embedding-{request_index}"),
                    ..observation.clone()
                },
            );
            assert_eq!(stats[0].1.last_mean_input_tps, 0.0);
        }
        let stats = apply_fallback_observation(
            &mut aggregator,
            &RequestObservation {
                request_id: "req-embedding-4".to_string(),
                ..observation
            },
        );

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.last_mean_input_tps, 10.0);
        assert_eq!(stats[0].1.output_tps, 0.0);
        assert_eq!(stats[0].1.max_output_tps, 0.0);
        assert_eq!(stats[0].1.embedding_item_tps, 2.0);
        assert_eq!(stats[0].1.max_embedding_item_tps, 2.0);
        assert!(stats[0].1.stats_capabilities.is_empty());
        assert!(stats[0].1.stats_sources.is_empty());

        let live_chat = RequestObservation {
            request_id: "req-live-chat".to_string(),
            state: RequestObservationState::OutputGeneration,
            output_tokens: 20,
            time_to_first_output: Some(Duration::from_secs(1)),
            time_to_first_token: Some(Duration::from_secs(1)),
            total_duration: Duration::from_secs(3),
            ..completed_observation(10, 1, 20, Duration::from_secs(1), Duration::from_secs(3))
        };
        let stats = apply_fallback_observation(&mut aggregator, &live_chat);

        assert_eq!(stats[0].1.output_tps, 10.0);
    }

    #[test]
    fn fast_embeddings_input_samples_clamp_to_duration_floor() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = completed_embeddings_observation(
            20,
            4,
            Duration::from_millis(1),
            Duration::from_millis(4),
        );

        for sample_index in 0..5 {
            apply_fallback_observation(
                &mut aggregator,
                &RequestObservation {
                    request_id: format!("req-fast-embedding-{sample_index}"),
                    ..observation.clone()
                },
            );
        }

        assert_eq!(aggregator.snapshot("model-a").last_mean_input_tps, 2000.0);
        let distribution = &aggregator.per_model["model-a"].input_tps_distribution;
        assert_eq!(distribution.count, 5);
        assert_eq!(distribution.mean, 2000.0);
    }

    #[test]
    fn embeddings_item_tps_clamps_fast_response_relay_duration() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = completed_embeddings_observation(
            20,
            2,
            Duration::from_millis(2),
            Duration::from_millis(5),
        );

        let stats = apply_fallback_observation(&mut aggregator, &observation);

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.output_tps, 0.0);
        assert_eq!(stats[0].1.max_output_tps, 0.0);
        assert_eq!(stats[0].1.embedding_item_tps, 200.0);
        assert_eq!(stats[0].1.max_embedding_item_tps, 200.0);
    }

    #[test]
    fn embeddings_stats_do_not_replace_chat_output_tps() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        let chat = completed_observation(20, 1, 10, Duration::from_secs(1), Duration::from_secs(3));
        let stats = apply_fallback_observation(&mut aggregator, &chat);
        assert_eq!(stats[0].1.output_tps, 5.0);

        let embeddings =
            completed_embeddings_observation(20, 2, Duration::from_secs(1), Duration::from_secs(2));
        let stats = apply_fallback_observation(&mut aggregator, &embeddings);

        assert_eq!(stats[0].1.output_tps, 5.0);
        assert_eq!(stats[0].1.max_output_tps, 5.0);
        assert_eq!(stats[0].1.embedding_item_tps, 2.0);
        assert_eq!(stats[0].1.max_embedding_item_tps, 2.0);
        assert!(stats[0].1.stats_capabilities.is_empty());
        assert!(stats[0].1.stats_sources.is_empty());
    }

    #[test]
    fn embeddings_observations_do_not_add_output_throughput_labels() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        let chat = completed_observation(20, 1, 10, Duration::from_secs(1), Duration::from_secs(3));
        let stats = apply_fallback_observation(&mut aggregator, &chat);
        assert_eq!(stats[0].1.output_tps, 5.0);

        let failed_embeddings = RequestObservation {
            state: RequestObservationState::Failed,
            ..completed_embeddings_observation(
                20,
                2,
                Duration::from_secs(1),
                Duration::from_secs(2),
            )
        };
        let stats = apply_fallback_observation(&mut aggregator, &failed_embeddings);

        assert_eq!(
            stats[0].1.output_tps, 5.0,
            "failed embeddings requests must not replace the last completed output sample"
        );
        assert_eq!(stats[0].1.embedding_item_tps, 0.0);
        assert_eq!(stats[0].1.max_embedding_item_tps, 0.0);
        assert!(stats[0].1.stats_capabilities.is_empty());
        assert!(stats[0].1.stats_sources.is_empty());

        let live_embeddings = RequestObservation {
            state: RequestObservationState::UpstreamConnecting,
            total_duration: Duration::ZERO,
            ..completed_embeddings_observation(
                20,
                2,
                Duration::from_secs(1),
                Duration::from_secs(2),
            )
        };
        let stats = apply_fallback_observation(&mut aggregator, &live_embeddings);

        assert_eq!(stats[0].1.output_tps, 5.0);
        assert_eq!(stats[0].1.embedding_item_tps, 0.0);
        assert!(stats[0].1.stats_capabilities.is_empty());
        assert!(stats[0].1.stats_sources.is_empty());
    }

    #[test]
    fn ignores_non_complete_observations() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let observation = RequestObservation {
            state: RequestObservationState::Failed,
            ..completed_observation(20, 4, 8, Duration::from_secs(2), Duration::from_secs(6))
        };

        let stats = apply_fallback_observation(&mut aggregator, &observation);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1.last_mean_input_tps, 0.0);
        assert_eq!(stats[0].1.output_tps, 0.0);
    }

    #[test]
    fn publishes_live_queue_and_active_stats() {
        let config = StatsCollectorConfig::default();
        let runtime_state = PylonRuntimeState::default();
        runtime_state.update_model_throughput("model-a", 100.0);
        let mut aggregator = StatsAggregator::new(config, runtime_state);

        let queued = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-live".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 2,
            input_tokens: 24,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::InputProcessing,
            time_to_response_headers: Some(Duration::from_millis(5)),
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::from_millis(5),
        };
        let queued_stats = apply_fallback_observation(&mut aggregator, &queued);
        assert_eq!(queued_stats[0].1.queue_size, 1);
        assert_eq!(queued_stats[0].1.queued_input_size, 24);
        assert_eq!(
            queued_stats[0].1.queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(0, 240)]))
        );
        assert_eq!(queued_stats[0].1.num_running_queries, 1);
        assert_eq!(queued_stats[0].1.total_query_input_size, 24);
        assert_eq!(queued_stats[0].1.input_processing_queries, 1);
        assert_eq!(queued_stats[0].1.output_generation_queries, 0);
        assert_eq!(queued_stats[0].1.last_mean_input_tps, 0.0);

        let generating = RequestObservation {
            output_messages: 2,
            output_tokens: 8,
            state: RequestObservationState::OutputGeneration,
            time_to_first_output: Some(Duration::from_secs(2)),
            time_to_first_token: Some(Duration::from_secs(2)),
            total_duration: Duration::from_secs(3),
            ..queued
        };
        let active_stats = apply_fallback_observation(&mut aggregator, &generating);
        assert_eq!(active_stats[0].1.queue_size, 0);
        assert_eq!(active_stats[0].1.queued_input_size, 0);
        assert_eq!(active_stats[0].1.num_running_queries, 1);
        assert_eq!(active_stats[0].1.total_query_input_size, 24);
        assert_eq!(active_stats[0].1.input_processing_queries, 0);
        assert_eq!(active_stats[0].1.output_generation_queries, 1);
        assert_eq!(active_stats[0].1.last_mean_input_tps, 0.0);
        assert_eq!(active_stats[0].1.output_tps, 8.0);
    }

    #[test]
    fn live_stats_math_is_exact_for_simultaneous_queued_and_generating_requests() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        let queued = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-queued".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 30,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::InputProcessing,
            time_to_response_headers: Some(Duration::from_millis(5)),
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::from_millis(5),
        };
        let generating = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-generating".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 20,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages: 3,
            output_tokens: 6,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::OutputGeneration,
            time_to_response_headers: Some(Duration::from_millis(5)),
            time_to_first_output: Some(Duration::from_secs(2)),
            time_to_first_token: Some(Duration::from_secs(2)),
            total_duration: Duration::from_secs(5),
        };

        apply_fallback_observation(&mut aggregator, &queued);
        let stats = apply_fallback_observation(&mut aggregator, &generating)
            .into_iter()
            .find(|(model_id, _)| model_id == "model-a")
            .unwrap()
            .1;

        assert_eq!(stats.queue_size, 1);
        assert_eq!(stats.queued_input_size, 30);
        assert_eq!(stats.num_running_queries, 2);
        assert_eq!(stats.total_query_input_size, 50);
        assert_eq!(stats.last_mean_input_tps, 0.0);
        assert_eq!(stats.output_tps, 2.0);
    }

    #[test]
    fn live_input_processing_keeps_full_requested_input_without_retired_progress() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        let observation = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-input-processing".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 100,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: Some(200),
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::InputProcessing,
            time_to_response_headers: Some(Duration::from_secs(2)),
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::from_secs(30),
        };

        let stats = apply_fallback_observation(&mut aggregator, &observation);

        assert_eq!(stats[0].1.last_mean_input_tps, 0.0);
        assert_eq!(stats[0].1.queued_input_size, 100);
        assert!(stats[0].1.stats_capabilities.is_empty());
        assert!(stats[0].1.stats_sources.is_empty());
        assert!(stats[0].1.stats_observed_at_unix_ms > 0);
    }

    #[test]
    fn chunk_usage_observations_claim_only_chunk_usage_stats() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());

        let observation = RequestObservation {
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
            ..completed_observation(
                12,
                1,
                7,
                Duration::from_millis(100),
                Duration::from_millis(500),
            )
        };

        let stats = apply_fallback_observation(&mut aggregator, &observation);

        assert_eq!(
            stats[0].1.stats_capabilities,
            vec!["request.output.chunk_usage".to_string()]
        );
        assert_eq!(stats[0].1.stats_sources, vec!["chunk_usage".to_string()]);
    }
    #[test]
    fn snapshot_includes_polled_kv_cache_stats() {
        let config = StatsCollectorConfig::default();
        let mut aggregator = StatsAggregator::new(config, PylonRuntimeState::default());
        let stats = aggregator
            .apply_kv_cache_stats(KvCacheStatsSnapshot {
                model: "model-a".to_string(),
                kv_cache_capacity_tokens: 1_000,
                kv_cache_used_tokens: 400,
                kv_cache_free_tokens: 600,
            })
            .expect("KV-cache stats should publish")
            .1;

        assert_eq!(stats.kv_cache_capacity_tokens, 1_000);
        assert_eq!(stats.kv_cache_used_tokens, 400);
        assert_eq!(stats.kv_cache_free_tokens, 600);
    }

    #[tokio::test]
    async fn kv_cache_poll_updates_model_metrics() {
        async fn kv_cache_stats() -> Json<serde_json::Value> {
            Json(serde_json::json!({
                "model": "model-a",
                "kv_cache_capacity_tokens": 1000,
                "kv_cache_used_tokens": 400,
                "kv_cache_free_tokens": 600
            }))
        }

        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have address");
        let server = tokio::spawn(async move {
            let app = Router::new().route("/kv-cache", get(kv_cache_stats));
            axum::serve(listener, app)
                .await
                .expect("KV-cache test server should run");
        });

        let config = StatsCollectorConfig {
            kv_cache_stats_url: Some(format!("http://{addr}/kv-cache")),
            kv_cache_poll_interval: Duration::from_millis(10),
            kv_cache_request_timeout: Duration::from_secs(1),
            ..Default::default()
        };
        let (runtime_state, observation_rx) =
            observed_runtime_with_metrics(&config, metrics.clone());
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            None,
            runtime_state.clone(),
            stop.clone(),
        ));

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "KV-cache stats should be published",
            |stats| stats.kv_cache_capacity_tokens == 1000,
        )
        .await;
        assert_eq!(stats.kv_cache_capacity_tokens, 1000);
        assert_eq!(stats.kv_cache_used_tokens, 400);
        assert_eq!(stats.kv_cache_free_tokens, 600);

        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(r#"pylon_model_kv_cache_capacity_tokens{model="model-a"} 1000"#));
        assert!(body.contains(r#"pylon_model_kv_cache_used_tokens{model="model-a"} 400"#));
        assert!(body.contains(r#"pylon_model_kv_cache_free_tokens{model="model-a"} 600"#));

        stop.cancel();
        tokio::time::timeout(Duration::from_secs(2), collector)
            .await
            .expect("collector should stop")
            .expect("collector task should join");
        server.abort();
    }

    #[tokio::test]
    async fn stats_collector_shutdown_interrupts_blocked_kv_cache_poll() {
        let poll_entered = Arc::new(tokio::sync::Barrier::new(2));
        let server_poll_entered = poll_entered.clone();
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have address");
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/kv-cache",
                get(move || {
                    let poll_entered = server_poll_entered.clone();
                    async move {
                        poll_entered.wait().await;
                        std::future::pending::<Json<serde_json::Value>>().await
                    }
                }),
            );
            axum::serve(listener, app)
                .await
                .expect("KV-cache test server should run");
        });
        let config = StatsCollectorConfig {
            kv_cache_stats_url: Some(format!("http://{addr}/kv-cache")),
            kv_cache_poll_interval: Duration::from_millis(1),
            kv_cache_request_timeout: Duration::from_secs(60),
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let collector =
            start_stats_collector_with_engine_stats(config, observation_rx, None, runtime_state);
        poll_entered.wait().await;

        let stopped = tokio::time::timeout(Duration::from_secs(1), collector.shutdown()).await;
        server.abort();

        stopped.expect("collector shutdown should interrupt blocked KV-cache poll");
    }

    #[tokio::test]
    async fn stats_collector_publishes_mean_input_tps_from_completed_observations() {
        let config = StatsCollectorConfig {
            observation_channel_capacity: 16,
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            None,
            runtime_state.clone(),
            stop.clone(),
        ));

        for request_index in 0..5 {
            runtime_state.observe_request(RequestObservation {
                request_id: format!("req-stats-openai-{request_index}"),
                output_messages: 1,
                output_tokens: 2,
                time_to_first_output: Some(Duration::from_millis(500)),
                time_to_first_token: Some(Duration::from_millis(600)),
                total_duration: Duration::from_secs(1),
                ..completed_observation(
                    50,
                    1,
                    2,
                    Duration::from_millis(500),
                    Duration::from_secs(1),
                )
            });
        }
        tokio::task::yield_now().await;

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "mean input TPS should be published",
            |stats| stats.last_mean_input_tps == 100.0,
        )
        .await;
        assert_eq!(stats.last_mean_input_tps, 100.0);
        assert_eq!(stats.output_tps, 5.0);

        stop.cancel();
        collector.await.expect("collector task should join");
    }
    #[tokio::test]
    async fn stats_collector_seeds_fixed_input_tps_for_queue_admission() {
        let config = StatsCollectorConfig {
            configured_model_ids: vec!["model-a".to_string()],
            fixed_last_mean_input_tps: Some(2_200.0),
            ..Default::default()
        };
        let (runtime_state, observation_rx) = observed_runtime(&config);
        let stop = CancellationToken::new();
        let collector = tokio::spawn(run_stats_collector(
            config,
            observation_rx,
            None,
            runtime_state.clone(),
            stop.clone(),
        ));

        let stats = wait_for_model_stats(
            &runtime_state,
            "model-a",
            "fixed TPS stats should be published",
            |stats| stats.last_mean_input_tps == 2_200.0,
        )
        .await;
        assert_eq!(stats.last_mean_input_tps, 2_200.0);

        let _queued =
            runtime_state.track_request(&crate::request_observer::RequiredTunnelHeaders {
                request_id: "req-queued".to_string(),
                routing_key: None,
                model_id: "model-a".to_string(),
                priority: 0,
                input_tokens: 32,
                accepted_at: std::time::Instant::now(),
            });
        runtime_state.transition_request_observation(active_chat_observation(
            "req-queued",
            RequestObservationState::Queued,
        ));
        assert_eq!(
            runtime_state
                .snapshot_live_model("model-a")
                .queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(0, 15)]))
        );

        stop.cancel();
        collector.await.expect("collector task should join");
    }

    #[test]
    fn records_metrics_when_configured() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let config = StatsCollectorConfig::default();
        let (runtime_state, _observation_rx) = PylonRuntimeState::observed(
            stargate_proto::pb::InferenceServerStatus::Unknown,
            &[],
            config.observation_channel_capacity,
            Some(metrics.clone()),
        );
        let mut aggregator = StatsAggregator::new(config, runtime_state.clone());
        let observation =
            completed_observation(20, 2, 10, Duration::from_secs(2), Duration::from_secs(4));

        for _ in 0..5 {
            let updated_stats = apply_fallback_observation(&mut aggregator, &observation);
            for (model_id, stats) in updated_stats {
                observe_model_metric(&runtime_state, &model_id, &stats);
            }
        }
        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(
            r#"pylon_requests_total{model="model-a",routing_key="rk-1",status="complete"} 5"#
        ));
        assert!(body.contains(r#"pylon_model_last_mean_input_tps{model="model-a"} 10"#));
        assert!(body.contains(r#"pylon_model_output_tps{model="model-a"} 5"#));
    }

    #[test]
    fn rejects_kv_cache_stats_for_unconfigured_models() {
        let config = StatsCollectorConfig {
            configured_model_ids: vec!["model-a".to_string()],
            ..Default::default()
        };
        let kv_cache = KvCacheStatsSnapshot {
            model: "model-b".to_string(),
            kv_cache_capacity_tokens: 1_000,
            kv_cache_used_tokens: 400,
            kv_cache_free_tokens: 600,
        };

        assert!(
            StatsAggregator::new(config, PylonRuntimeState::default())
                .apply_kv_cache_stats(kv_cache)
                .is_none()
        );
    }
}
