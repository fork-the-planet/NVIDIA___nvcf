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
use std::sync::Arc;

use parking_lot::Mutex;
use stargate_proto::pb::{InferenceServerModelRegistration, InferenceServerStatus, ModelStats};

use crate::bringup::ModelBringupState;
use crate::queue_admission::{
    LiveRequestState, PylonQueueMismatchRetryConfig, QueueAdmissionDecision, QueueModelSnapshot,
    QueueTrackedRequestGuard, RequestObservationTransition,
};
use crate::request_observer::{RequestObservation, RequiredTunnelHeaders};
use crate::stats::PylonMetrics;
use reqwest::header::HeaderMap;

#[derive(Debug, Clone, Default)]
pub struct CurrentModelStats {
    // Sticky runtime-observed mean input TPS for this backend.
    pub last_mean_input_tps: f64,
    // Token/sec output rate for streaming generation endpoints. Embeddings item
    // cardinality is observed separately and is not exported through this field.
    pub output_tps: f64,
    pub embedding_item_tps: f64,
    pub queue_size: u64,
    pub queued_input_size: u64,

    // Cluster-scoped shared-hardware/scheduler state. Stargate currently keeps
    // the latest active backend snapshot for these fields when multiple
    // backends share a cluster.
    // Same token/sec unit as `output_tps`; embeddings item rates are not folded in.
    pub max_output_tps: f64,
    pub max_embedding_item_tps: f64,
    pub kv_cache_capacity_tokens: u64,
    pub kv_cache_used_tokens: u64,
    pub kv_cache_free_tokens: u64,
    pub num_running_queries: u64,
    pub max_engine_concurrency: Option<u64>,
    pub total_query_input_size: u64,
    pub queue_time_estimate_ms_by_priority: Option<HashMap<u32, u64>>,
    pub input_processing_queries: u64,
    pub output_generation_queries: u64,
    pub stats_observed_at_unix_ms: u64,
    pub stats_capabilities: Vec<String>,
    pub stats_sources: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PylonRuntimeState {
    advertised: Arc<Mutex<AdvertisedRuntimeState>>,
    live_requests: LiveRequestState,
    metrics: Option<Arc<PylonMetrics>>,
    observation_tx: Option<flume::Sender<RequestObservationEvent>>,
}

#[derive(Clone, Debug)]
pub struct RequestObservationEvent {
    pub(crate) observation: RequestObservation,
    pub(crate) changed_model_ids: Vec<String>,
}

#[derive(Debug)]
struct AdvertisedRuntimeState {
    base_status: InferenceServerStatus,
    models: HashMap<String, AdvertisedModelState>,
}

#[derive(Debug)]
struct AdvertisedModelState {
    stats: CurrentModelStats,
    bringup: ModelBringupState,
}

impl PylonRuntimeState {
    pub fn new(initial_status: InferenceServerStatus, model_ids: &[String]) -> Self {
        Self {
            advertised: Arc::new(Mutex::new(AdvertisedRuntimeState::new(
                initial_status,
                model_ids,
            ))),
            live_requests: LiveRequestState::default(),
            metrics: None,
            observation_tx: None,
        }
    }

    pub fn observed(
        initial_status: InferenceServerStatus,
        model_ids: &[String],
        observation_capacity: usize,
        metrics: Option<Arc<PylonMetrics>>,
    ) -> (Self, flume::Receiver<RequestObservationEvent>) {
        let (observation_tx, observation_rx) = flume::bounded(observation_capacity);
        let mut runtime_state = Self::new(initial_status, model_ids);
        runtime_state.metrics = metrics;
        runtime_state.observation_tx = Some(observation_tx);
        (runtime_state, observation_rx)
    }

    pub fn set_status(&self, status: InferenceServerStatus) {
        self.advertised.lock().base_status = status;
    }

    pub(crate) fn model_ids(&self) -> Vec<String> {
        let mut model_ids = self
            .advertised
            .lock()
            .models
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        model_ids.sort();
        model_ids
    }

    pub fn set_model_stats(&self, model_id: impl Into<String>, stats: CurrentModelStats) {
        let model_id = model_id.into();
        self.live_requests
            .update_model_throughput(&model_id, stats.last_mean_input_tps);
        let mut advertised = self.advertised.lock();
        advertised
            .models
            .entry(model_id)
            .or_insert_with(|| AdvertisedModelState {
                stats: CurrentModelStats::default(),
                bringup: ModelBringupState::ConnectingUnavailable,
            })
            .stats = stats;
    }

    pub fn model_stats(&self, model_id: &str) -> Option<CurrentModelStats> {
        self.advertised
            .lock()
            .models
            .get(model_id)
            .map(|model| model.stats.clone())
    }

    pub(crate) fn set_model_bringup(
        &self,
        model_id: impl Into<String>,
        bringup: ModelBringupState,
    ) {
        let mut advertised = self.advertised.lock();
        advertised
            .models
            .entry(model_id.into())
            .or_insert_with(|| AdvertisedModelState {
                stats: CurrentModelStats::default(),
                bringup: ModelBringupState::ConnectingUnavailable,
            })
            .bringup = bringup;
    }

    #[cfg(test)]
    pub(crate) fn model_bringup(&self, model_id: &str) -> Option<ModelBringupState> {
        self.advertised
            .lock()
            .models
            .get(model_id)
            .map(|model| model.bringup)
    }

    pub(crate) fn advertised_models(&self) -> HashMap<String, InferenceServerModelRegistration> {
        self.advertised.lock().advertised_models()
    }

    pub fn observe_request(&self, observation: RequestObservation) {
        let event = self.transition_request_observation(observation);
        let request_id = event.observation.request_id.clone();
        if let Some(tx) = &self.observation_tx
            && let Err(error) = tx.try_send(event)
        {
            tracing::warn!(
                request_id,
                error = %error,
                "dropping request observation"
            );
        }
    }

    pub(crate) fn update_model_throughput(&self, model_id: &str, last_mean_input_tps: f64) {
        self.live_requests
            .update_model_throughput(model_id, last_mean_input_tps);
    }

    pub(crate) fn metrics(&self) -> Option<&PylonMetrics> {
        self.metrics.as_deref()
    }

    pub(crate) fn transition_request_observation(
        &self,
        observation: RequestObservation,
    ) -> RequestObservationEvent {
        let RequestObservationTransition {
            changed_model_ids,
            prior,
            current,
        } = self.live_requests.transition_observation(&observation);
        if let Some(metrics) = &self.metrics {
            metrics.observe_request_transition(&observation, prior.as_ref(), current.as_ref());
        }
        RequestObservationEvent {
            observation,
            changed_model_ids,
        }
    }

    pub(crate) fn update_request_active_output_tps(
        &self,
        request_id: &str,
        active_chat_output_tps: Option<f64>,
    ) -> Option<String> {
        self.live_requests
            .update_active_output_tps(request_id, active_chat_output_tps)
    }

    pub(crate) fn snapshot_live_model(&self, model_id: &str) -> QueueModelSnapshot {
        self.live_requests.snapshot_model(model_id)
    }

    pub(crate) fn evaluate_queue_admission(
        &self,
        config: &PylonQueueMismatchRetryConfig,
        required: &RequiredTunnelHeaders,
        headers: &HeaderMap,
    ) -> QueueAdmissionDecision {
        self.live_requests.evaluate(config, required, headers)
    }

    pub(crate) fn track_request(
        &self,
        required: &RequiredTunnelHeaders,
    ) -> QueueTrackedRequestGuard {
        self.live_requests.track_request(required)
    }

    pub(crate) fn finish_queue_request(&self, request_id: &str) {
        self.live_requests.finish_queue_request(request_id);
    }

    #[cfg(test)]
    pub(crate) fn tracked_request_count(&self) -> usize {
        self.live_requests.tracked_request_count()
    }
}

impl Default for PylonRuntimeState {
    fn default() -> Self {
        Self::new(InferenceServerStatus::Unknown, &[])
    }
}

impl RequestObservationEvent {
    pub fn observation(&self) -> &RequestObservation {
        &self.observation
    }

    pub fn into_observation(self) -> RequestObservation {
        self.observation
    }
}

impl AdvertisedRuntimeState {
    fn new(base_status: InferenceServerStatus, model_ids: &[String]) -> Self {
        let models = model_ids
            .iter()
            .cloned()
            .map(|model_id| {
                (
                    model_id,
                    AdvertisedModelState {
                        stats: CurrentModelStats::default(),
                        bringup: ModelBringupState::ConnectingUnavailable,
                    },
                )
            })
            .collect();
        Self {
            base_status,
            models,
        }
    }

    fn advertised_models(&self) -> HashMap<String, InferenceServerModelRegistration> {
        self.models
            .iter()
            .map(|(model_id, model)| {
                let stats = &model.stats;
                let status = gated_model_status(self.base_status, model.bringup);
                let registration = InferenceServerModelRegistration {
                    stats: Some(ModelStats {
                        last_mean_input_tps: stats.last_mean_input_tps,
                        output_tps: stats.output_tps,
                        max_output_tps: stats.max_output_tps,
                        queue_size: stats.queue_size,
                        queued_input_size: stats.queued_input_size,
                        kv_cache_capacity_tokens: stats.kv_cache_capacity_tokens,
                        kv_cache_used_tokens: stats.kv_cache_used_tokens,
                        kv_cache_free_tokens: stats.kv_cache_free_tokens,
                        num_running_queries: stats.num_running_queries,
                        max_engine_concurrency: stats.max_engine_concurrency.unwrap_or_default(),
                        total_query_input_size: stats.total_query_input_size,
                        queue_time_estimate_ms_by_priority: stats
                            .queue_time_estimate_ms_by_priority
                            .clone()
                            .unwrap_or_default(),
                        input_processing_queries: stats.input_processing_queries,
                        output_generation_queries: stats.output_generation_queries,
                        stats_observed_at_unix_ms: stats.stats_observed_at_unix_ms,
                        stats_capabilities: stats.stats_capabilities.clone(),
                        stats_sources: stats.stats_sources.clone(),
                    }),
                    status: status.into(),
                };
                (model_id.clone(), registration)
            })
            .collect()
    }
}

pub(crate) fn gated_model_status(
    base_status: InferenceServerStatus,
    bringup_state: ModelBringupState,
) -> InferenceServerStatus {
    if base_status != InferenceServerStatus::Active {
        return base_status;
    }
    match bringup_state {
        ModelBringupState::AdvertisingActive => InferenceServerStatus::Active,
        ModelBringupState::ConnectingUnavailable | ModelBringupState::Recovering => {
            InferenceServerStatus::Inactive
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use stargate_proto::pb::InferenceServerStatus;

    use super::PylonRuntimeState;
    use crate::PylonMetrics;
    use crate::request_observer::{
        RequestObservation, RequestObservationEndpoint, RequestObservationState,
    };

    #[test]
    fn publishing_observation_updates_live_state_and_emits_one_event() {
        let (runtime_state, observation_rx) =
            PylonRuntimeState::observed(InferenceServerStatus::Active, &[], 4, None);
        let mut observation = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-runtime-owner".to_string(),
            routing_key: None,
            model_id: "model-runtime-owner".to_string(),
            priority: 0,
            input_tokens: 42,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: None,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::UpstreamConnecting,
            time_to_response_headers: None,
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::ZERO,
        };

        runtime_state.observe_request(observation.clone());

        let emitted = observation_rx.try_recv().expect("observation should emit");
        assert_eq!(emitted.observation().request_id, observation.request_id);
        assert_eq!(emitted.observation().model_id, observation.model_id);
        assert_eq!(emitted.observation().state, observation.state);
        let live = runtime_state.snapshot_live_model("model-runtime-owner");
        assert_eq!(live.queue_size, 1);
        assert_eq!(live.queued_input_size, 42);

        observation.state = RequestObservationState::Complete;
        runtime_state.observe_request(observation);
        assert_eq!(
            runtime_state
                .snapshot_live_model("model-runtime-owner")
                .queue_size,
            0
        );
    }

    #[test]
    fn queue_cleanup_preserves_observed_identity_until_terminal_metrics_transition() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let (runtime_state, _observation_rx) = PylonRuntimeState::observed(
            InferenceServerStatus::Active,
            &[],
            4,
            Some(metrics.clone()),
        );
        let mut observation = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-local-rejection".to_string(),
            routing_key: Some("rk-a".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 42,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: None,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::UpstreamConnecting,
            time_to_response_headers: None,
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::ZERO,
        };

        runtime_state.observe_request(observation.clone());
        runtime_state.finish_queue_request(&observation.request_id);
        observation.state = RequestObservationState::Failed;
        runtime_state.observe_request(observation);

        let body = metrics.gather_text().expect("metrics should encode");
        assert!(
            body.contains(r#"pylon_requests_state{model="model-a",state="upstream_connecting"} 0"#),
            "terminal transition should remove the prior observed state: {body}"
        );
        assert!(
            body.contains(
                r#"pylon_requests_total{model="model-a",routing_key="rk-a",status="failed"} 1"#
            ),
            "terminal transition should record the failed request: {body}"
        );
    }

    #[test]
    fn lifecycle_metrics_do_not_wait_for_stats_channel_capacity() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let (runtime_state, _observation_rx) = PylonRuntimeState::observed(
            InferenceServerStatus::Active,
            &[],
            1,
            Some(metrics.clone()),
        );
        let mut observation = RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: "req-full-stats-channel".to_string(),
            routing_key: Some("rk-a".to_string()),
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 42,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: None,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: RequestObservationState::UpstreamConnecting,
            time_to_response_headers: None,
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::ZERO,
        };

        runtime_state.observe_request(observation.clone());
        observation.state = RequestObservationState::Failed;
        runtime_state.observe_request(observation);

        let body = metrics.gather_text().expect("metrics should encode");
        assert!(
            body.contains(r#"pylon_requests_state{model="model-a",state="upstream_connecting"} 0"#),
            "terminal source transition should clear the prior gauge: {body}"
        );
        assert!(
            body.contains(
                r#"pylon_requests_total{model="model-a",routing_key="rk-a",status="failed"} 1"#
            ),
            "terminal source transition should record metrics before channel send: {body}"
        );
    }
}
