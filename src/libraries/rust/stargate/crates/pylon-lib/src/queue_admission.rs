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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use parking_lot::Mutex;
use reqwest::header::HeaderMap;
use stargate_protocol::common::{queue_time_delta_ms, valid_last_mean_input_tps};
use stargate_protocol::tunnel_contract::HEADER_STARGATE_EXPECTED_QUEUE_MS;

use crate::request_observer::{RequestObservation, RequestObservationState, RequiredTunnelHeaders};

pub(crate) const RETRY_REASON_QUEUE_ESTIMATE_MISMATCH: &str = "queue_estimate_mismatch";

#[derive(Debug, Clone)]
pub struct PylonQueueMismatchRetryConfig {
    pub enabled: bool,
    pub min_delta_ms: u64,
    pub tolerance_factor: f64,
    pub retry_after_ms: Option<u64>,
}

impl Default for PylonQueueMismatchRetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_delta_ms: 25,
            tolerance_factor: 1.25,
            retry_after_ms: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LiveRequestState {
    inner: Arc<Mutex<QueueAdmissionState>>,
}

#[derive(Debug, Default)]
struct QueueAdmissionState {
    requests: HashMap<String, LiveRequest>,
    models: HashMap<String, QueueModelState>,
}

#[derive(Debug, Default)]
struct QueueModelState {
    last_mean_input_tps: Option<f64>,
    running_requests: usize,
    // Keep input totals exact internally so public saturation does not make
    // current-request exclusion or later removal lossy.
    total_input_tokens: u128,
    input_processing_requests: usize,
    output_generation_requests: usize,
    active_chat_output_tps_sum: f64,
    active_chat_output_requests: usize,
    prompt_work_by_priority: BTreeMap<u32, PromptWork>,
}

#[derive(Debug, Default)]
struct PromptWork {
    request_count: usize,
    input_tokens: u128,
}

#[derive(Clone, Debug)]
struct TrackedPromptRequest {
    model_id: String,
    priority: u32,
    input_tokens: u64,
    phase: TrackedPromptPhase,
    active_chat_output_tps: Option<f64>,
}

#[derive(Clone, Debug)]
enum LiveRequest {
    QueueOnly(TrackedPromptRequest),
    ObservedOnly(ObservedRequestState),
    QueueAndObserved {
        queue: TrackedPromptRequest,
        observed: ObservedRequestState,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ObservedRequestState {
    pub(crate) model_id: String,
    pub(crate) state: RequestObservationState,
    pub(crate) input_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RequestObservationTransition {
    pub(crate) changed_model_ids: Vec<String>,
    pub(crate) prior: Option<ObservedRequestState>,
    pub(crate) current: Option<ObservedRequestState>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TrackedPromptPhase {
    Pending,
    InputProcessing,
    OutputGeneration,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct QueueModelSnapshot {
    pub active_chat_output_tps: f64,
    pub queue_size: u64,
    pub queued_input_size: u64,
    pub num_running_queries: u64,
    pub total_query_input_size: u64,
    pub input_processing_queries: u64,
    pub output_generation_queries: u64,
    pub queue_time_estimate_ms_by_priority: Option<HashMap<u32, u64>>,
}

#[derive(Debug)]
pub(crate) struct QueueTrackedRequestGuard {
    live_requests: LiveRequestState,
    request_id: String,
    finished: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum QueueAdmissionDecision {
    Accepted {
        expected_ms: u64,
        actual_ms: u64,
        threshold_ms: u64,
    },
    Rejected {
        expected_ms: u64,
        actual_ms: u64,
        threshold_ms: u64,
        retry_after_ms: Option<u64>,
    },
    MissingEstimate,
    UnknownLocalEstimate {
        expected_ms: u64,
    },
    Disabled,
}

impl QueueAdmissionDecision {
    pub(crate) fn result_label(&self) -> &'static str {
        match self {
            Self::Accepted { .. } => "accepted",
            Self::Disabled => "disabled",
            Self::Rejected { .. } => "rejected",
            Self::MissingEstimate => "missing_estimate",
            Self::UnknownLocalEstimate { .. } => "unknown_local_estimate",
        }
    }

    pub(crate) fn expected_ms(&self) -> Option<u64> {
        match self {
            Self::Accepted { expected_ms, .. }
            | Self::Rejected { expected_ms, .. }
            | Self::UnknownLocalEstimate { expected_ms } => Some(*expected_ms),
            Self::MissingEstimate | Self::Disabled => None,
        }
    }

    pub(crate) fn actual_ms(&self) -> Option<u64> {
        match self {
            Self::Accepted { actual_ms, .. } | Self::Rejected { actual_ms, .. } => Some(*actual_ms),
            Self::MissingEstimate | Self::UnknownLocalEstimate { .. } | Self::Disabled => None,
        }
    }

    pub(crate) fn threshold_ms(&self) -> Option<u64> {
        match self {
            Self::Accepted { threshold_ms, .. } | Self::Rejected { threshold_ms, .. } => {
                Some(*threshold_ms)
            }
            Self::MissingEstimate | Self::UnknownLocalEstimate { .. } | Self::Disabled => None,
        }
    }
}

impl LiveRequestState {
    pub(crate) fn update_model_throughput(&self, model_id: &str, last_mean_input_tps: f64) {
        self.inner
            .lock()
            .update_model_throughput(model_id, last_mean_input_tps);
    }

    pub(crate) fn evaluate(
        &self,
        config: &PylonQueueMismatchRetryConfig,
        required: &RequiredTunnelHeaders,
        headers: &HeaderMap,
    ) -> QueueAdmissionDecision {
        if !config.enabled {
            return QueueAdmissionDecision::Disabled;
        }
        let Some(expected_ms) = parse_expected_queue_ms(headers) else {
            return QueueAdmissionDecision::MissingEstimate;
        };
        let Some(actual_ms) = self.queue_estimate_ms_for_priority_excluding(
            &required.model_id,
            required.priority,
            &required.request_id,
        ) else {
            return QueueAdmissionDecision::UnknownLocalEstimate { expected_ms };
        };
        let threshold_ms = mismatch_threshold_ms(expected_ms, config);
        if actual_ms > threshold_ms {
            QueueAdmissionDecision::Rejected {
                expected_ms,
                actual_ms,
                threshold_ms,
                retry_after_ms: config.retry_after_ms,
            }
        } else {
            QueueAdmissionDecision::Accepted {
                expected_ms,
                actual_ms,
                threshold_ms,
            }
        }
    }

    pub(crate) fn track_request(
        &self,
        required: &RequiredTunnelHeaders,
    ) -> QueueTrackedRequestGuard {
        let request = TrackedPromptRequest {
            model_id: required.model_id.clone(),
            priority: required.priority,
            input_tokens: required.input_tokens,
            phase: TrackedPromptPhase::Pending,
            active_chat_output_tps: None,
        };
        self.inner
            .lock()
            .start_queue_request(required.request_id.clone(), request);
        QueueTrackedRequestGuard {
            live_requests: self.clone(),
            request_id: required.request_id.clone(),
            finished: false,
        }
    }

    pub(crate) fn snapshot_model(&self, model_id: &str) -> QueueModelSnapshot {
        let state = self.inner.lock();
        state.snapshot_model(model_id)
    }

    pub(crate) fn transition_observation(
        &self,
        observation: &RequestObservation,
    ) -> RequestObservationTransition {
        self.inner.lock().transition_observation(observation)
    }

    pub(crate) fn update_active_output_tps(
        &self,
        request_id: &str,
        active_chat_output_tps: Option<f64>,
    ) -> Option<String> {
        self.inner
            .lock()
            .update_active_output_tps(request_id, active_chat_output_tps)
    }

    fn queue_estimate_ms_for_priority_excluding(
        &self,
        model_id: &str,
        priority: u32,
        excluded_request_id: &str,
    ) -> Option<u64> {
        let state = self.inner.lock();
        state.queue_estimate_ms_for_priority_excluding(model_id, priority, excluded_request_id)
    }

    fn advance_request_phase(&self, request_id: &str, phase: TrackedPromptPhase) {
        let _ = self.inner.lock().advance_request_phase(request_id, phase);
    }

    pub(crate) fn finish_queue_request(&self, request_id: &str) {
        self.inner.lock().finish_queue_request(request_id);
    }

    #[cfg(test)]
    pub(crate) fn tracked_request_count(&self) -> usize {
        self.inner
            .lock()
            .requests
            .values()
            .filter(|request| request.queue().is_some())
            .count()
    }
}

impl QueueAdmissionState {
    fn update_model_throughput(&mut self, model_id: &str, last_mean_input_tps: f64) {
        if valid_last_mean_input_tps(last_mean_input_tps) {
            self.models
                .entry(model_id.to_string())
                .or_default()
                .last_mean_input_tps = Some(last_mean_input_tps);
            return;
        }

        let remove_model = self.models.get_mut(model_id).is_some_and(|model| {
            model.last_mean_input_tps = None;
            model.is_unused()
        });
        if remove_model {
            self.models.remove(model_id);
        }
    }

    fn start_queue_request(&mut self, request_id: String, request: TrackedPromptRequest) {
        let (_, observed) = self.take_request(&request_id);
        self.put_request(request_id, Some(request), observed);
    }

    fn finish_queue_request(&mut self, request_id: &str) {
        let (_, observed) = self.take_request(request_id);
        self.put_request(request_id.to_string(), None, observed);
    }

    fn transition_observation(
        &mut self,
        observation: &RequestObservation,
    ) -> RequestObservationTransition {
        let (prior_queue, prior_observed) = self.take_request(&observation.request_id);
        let changed_model_ids = changed_model_ids(
            &observation.model_id,
            [
                prior_queue.as_ref().map(|request| request.model_id.clone()),
                prior_observed
                    .as_ref()
                    .map(|request| request.model_id.clone()),
            ],
        );
        if observation.is_terminal() {
            return RequestObservationTransition {
                changed_model_ids,
                prior: prior_observed,
                current: None,
            };
        }
        let phase = match observation.state {
            RequestObservationState::Queued | RequestObservationState::UpstreamConnecting => {
                TrackedPromptPhase::Pending
            }
            RequestObservationState::InputProcessing => TrackedPromptPhase::InputProcessing,
            RequestObservationState::OutputGeneration => TrackedPromptPhase::OutputGeneration,
            RequestObservationState::Complete
            | RequestObservationState::Failed
            | RequestObservationState::Cancelled => unreachable!("terminal observations returned"),
        };
        let phase = prior_queue
            .as_ref()
            .map(|prior| phase.max(prior.phase))
            .unwrap_or(phase);
        let active_chat_output_tps = if phase == TrackedPromptPhase::OutputGeneration {
            prior_queue
                .as_ref()
                .and_then(|request| request.active_chat_output_tps)
        } else {
            None
        };
        let current = ObservedRequestState {
            model_id: observation.model_id.clone(),
            state: observation.state,
            input_tokens: observation.input_tokens,
        };
        self.put_request(
            observation.request_id.clone(),
            Some(TrackedPromptRequest {
                model_id: observation.model_id.clone(),
                priority: observation.priority,
                input_tokens: observation.input_tokens,
                phase,
                active_chat_output_tps: active_chat_output_tps
                    .filter(|output_tps| output_tps.is_finite() && *output_tps > 0.0),
            }),
            Some(current.clone()),
        );
        RequestObservationTransition {
            changed_model_ids,
            prior: prior_observed,
            current: Some(current),
        }
    }

    fn update_active_output_tps(
        &mut self,
        request_id: &str,
        active_chat_output_tps: Option<f64>,
    ) -> Option<String> {
        let (queue, observed) = self.take_request(request_id);
        let Some(mut queue) = queue else {
            self.put_request(request_id.to_string(), None, observed);
            return None;
        };
        queue.active_chat_output_tps =
            active_chat_output_tps.filter(|output_tps| output_tps.is_finite() && *output_tps > 0.0);
        let model_id = queue.model_id.clone();
        self.put_request(request_id.to_string(), Some(queue), observed);
        Some(model_id)
    }

    fn advance_request_phase(&mut self, request_id: &str, next_phase: TrackedPromptPhase) -> bool {
        let Self {
            requests, models, ..
        } = self;
        let Some(request) = requests
            .get_mut(request_id)
            .and_then(LiveRequest::queue_mut)
        else {
            return false;
        };
        if next_phase <= request.phase {
            return true;
        }
        models
            .get_mut(&request.model_id)
            .expect("tracked request should have model queue state")
            .advance_request_phase(request, next_phase);
        request.phase = next_phase;
        true
    }

    fn take_request(
        &mut self,
        request_id: &str,
    ) -> (Option<TrackedPromptRequest>, Option<ObservedRequestState>) {
        let Some(request) = self.requests.remove(request_id) else {
            return (None, None);
        };
        let (queue, observed) = request.into_parts();
        if let Some(queue) = &queue {
            self.remove_request_from_model(queue);
        }
        (queue, observed)
    }

    fn put_request(
        &mut self,
        request_id: String,
        queue: Option<TrackedPromptRequest>,
        observed: Option<ObservedRequestState>,
    ) {
        let Some(request) = LiveRequest::from_parts(queue, observed) else {
            return;
        };
        if let Some(queue) = request.queue() {
            self.add_request_to_model(queue);
        }
        self.requests.insert(request_id, request);
    }

    fn snapshot_model(&self, model_id: &str) -> QueueModelSnapshot {
        self.models
            .get(model_id)
            .map(QueueModelState::snapshot)
            .unwrap_or_default()
    }

    fn queue_estimate_ms_for_priority_excluding(
        &self,
        model_id: &str,
        priority: u32,
        excluded_request_id: &str,
    ) -> Option<u64> {
        let model = self.models.get(model_id)?;
        let excluded_request = self
            .requests
            .get(excluded_request_id)
            .and_then(LiveRequest::queue)
            .filter(|request| request.model_id == model_id);
        model.queue_estimate_ms_for_priority_excluding(priority, excluded_request)
    }

    fn add_request_to_model(&mut self, request: &TrackedPromptRequest) {
        self.models
            .entry(request.model_id.clone())
            .or_default()
            .add_request(request);
    }

    fn remove_request_from_model(&mut self, request: &TrackedPromptRequest) {
        let remove_model = {
            let model = self
                .models
                .get_mut(&request.model_id)
                .expect("tracked request should have model queue state");
            model.remove_request(request);
            model.is_unused()
        };
        if remove_model {
            self.models.remove(&request.model_id);
        }
    }
}

impl QueueModelState {
    fn add_request(&mut self, request: &TrackedPromptRequest) {
        self.running_requests = self
            .running_requests
            .checked_add(1)
            .expect("model running request count overflowed");
        self.total_input_tokens = self
            .total_input_tokens
            .checked_add(u128::from(request.input_tokens))
            .expect("model total input tokens overflowed");
        self.add_phase(request.phase, request.priority, request.input_tokens);
        if let Some(output_tps) = request.active_chat_output_tps {
            self.active_chat_output_tps_sum += output_tps;
            self.active_chat_output_requests = self
                .active_chat_output_requests
                .checked_add(1)
                .expect("model active output request count overflowed");
        }
    }

    fn remove_request(&mut self, request: &TrackedPromptRequest) {
        self.running_requests = self
            .running_requests
            .checked_sub(1)
            .expect("model running request count underflowed");
        self.total_input_tokens = self
            .total_input_tokens
            .checked_sub(u128::from(request.input_tokens))
            .expect("model total input tokens underflowed");
        self.remove_phase(request.phase, request.priority, request.input_tokens);
        if let Some(output_tps) = request.active_chat_output_tps {
            self.active_chat_output_tps_sum -= output_tps;
            self.active_chat_output_requests = self
                .active_chat_output_requests
                .checked_sub(1)
                .expect("model active output request count underflowed");
        }
    }

    fn advance_request_phase(
        &mut self,
        request: &TrackedPromptRequest,
        next_phase: TrackedPromptPhase,
    ) {
        self.remove_phase(request.phase, request.priority, request.input_tokens);
        self.add_phase(next_phase, request.priority, request.input_tokens);
    }

    fn add_phase(&mut self, phase: TrackedPromptPhase, priority: u32, input_tokens: u64) {
        match phase {
            TrackedPromptPhase::Pending => {}
            TrackedPromptPhase::InputProcessing => {
                self.input_processing_requests = self
                    .input_processing_requests
                    .checked_add(1)
                    .expect("model input-processing request count overflowed");
            }
            TrackedPromptPhase::OutputGeneration => {
                self.output_generation_requests = self
                    .output_generation_requests
                    .checked_add(1)
                    .expect("model output-generation request count overflowed");
            }
        }
        if let Some((effective_priority, remaining)) =
            TrackedPromptRequest::prompt_work_for(phase, priority, input_tokens)
        {
            let work = self
                .prompt_work_by_priority
                .entry(effective_priority)
                .or_default();
            work.request_count = work
                .request_count
                .checked_add(1)
                .expect("model prompt request count overflowed");
            work.input_tokens = work
                .input_tokens
                .checked_add(u128::from(remaining))
                .expect("model prompt input tokens overflowed");
        }
    }

    fn remove_phase(&mut self, phase: TrackedPromptPhase, priority: u32, input_tokens: u64) {
        match phase {
            TrackedPromptPhase::Pending => {}
            TrackedPromptPhase::InputProcessing => {
                self.input_processing_requests = self
                    .input_processing_requests
                    .checked_sub(1)
                    .expect("model input-processing request count underflowed");
            }
            TrackedPromptPhase::OutputGeneration => {
                self.output_generation_requests = self
                    .output_generation_requests
                    .checked_sub(1)
                    .expect("model output-generation request count underflowed");
            }
        }
        let Some((effective_priority, remaining)) =
            TrackedPromptRequest::prompt_work_for(phase, priority, input_tokens)
        else {
            return;
        };
        let remove_priority = {
            let work = self
                .prompt_work_by_priority
                .get_mut(&effective_priority)
                .expect("tracked prompt request should have priority work");
            work.request_count = work
                .request_count
                .checked_sub(1)
                .expect("model prompt request count underflowed");
            work.input_tokens = work
                .input_tokens
                .checked_sub(u128::from(remaining))
                .expect("model prompt input tokens underflowed");
            work.request_count == 0
        };
        if remove_priority {
            let work = self
                .prompt_work_by_priority
                .remove(&effective_priority)
                .expect("empty priority work should remain present until removal");
            assert_eq!(
                work.input_tokens, 0,
                "empty prompt priority should not retain input tokens"
            );
        }
    }

    fn snapshot(&self) -> QueueModelSnapshot {
        let mut queue_size = 0usize;
        let mut queued_input_tokens = 0u128;
        let mut cumulative_input_tokens = 0u128;
        let mut estimates = self.last_mean_input_tps.map(|_| HashMap::new());

        for (priority, work) in &self.prompt_work_by_priority {
            queue_size = queue_size
                .checked_add(work.request_count)
                .expect("model queued request count overflowed");
            queued_input_tokens = queued_input_tokens
                .checked_add(work.input_tokens)
                .expect("model queued input tokens overflowed");
            cumulative_input_tokens = cumulative_input_tokens
                .checked_add(work.input_tokens)
                .expect("model cumulative input tokens overflowed");
            if let (Some(last_mean_input_tps), Some(estimates)) =
                (self.last_mean_input_tps, estimates.as_mut())
            {
                estimates.insert(
                    *priority,
                    saturated_queue_time_ms(cumulative_input_tokens, last_mean_input_tps),
                );
            }
        }

        QueueModelSnapshot {
            active_chat_output_tps: if self.active_chat_output_requests == 0 {
                0.0
            } else {
                self.active_chat_output_tps_sum / self.active_chat_output_requests as f64
            },
            queue_size: saturated_count(queue_size),
            queued_input_size: saturated_input_tokens(queued_input_tokens),
            num_running_queries: saturated_count(self.running_requests),
            total_query_input_size: saturated_input_tokens(self.total_input_tokens),
            input_processing_queries: saturated_count(self.input_processing_requests),
            output_generation_queries: saturated_count(self.output_generation_requests),
            // Valid throughput and no prompt work is an explicit empty map:
            // downstream merges must clear previously published queues.
            queue_time_estimate_ms_by_priority: estimates,
        }
    }

    fn queue_estimate_ms_for_priority_excluding(
        &self,
        priority: u32,
        excluded_request: Option<&TrackedPromptRequest>,
    ) -> Option<u64> {
        let last_mean_input_tps = self.last_mean_input_tps?;
        let mut input_tokens = self
            .prompt_work_by_priority
            .range(..=priority)
            .try_fold(0u128, |total, (_, work)| {
                total.checked_add(work.input_tokens)
            })
            .expect("model cumulative input tokens overflowed");
        if let Some((excluded_priority, excluded_input_tokens)) =
            excluded_request.and_then(TrackedPromptRequest::prompt_work)
            && excluded_priority <= priority
        {
            input_tokens = input_tokens
                .checked_sub(u128::from(excluded_input_tokens))
                .expect("excluded request should be present in model prompt work");
        }
        Some(saturated_queue_time_ms(input_tokens, last_mean_input_tps))
    }

    fn is_unused(&self) -> bool {
        self.running_requests == 0 && self.last_mean_input_tps.is_none()
    }
}

fn changed_model_ids(model_id: &str, prior_model_ids: [Option<String>; 2]) -> Vec<String> {
    let mut changed = vec![model_id.to_string()];
    for prior_model_id in prior_model_ids.into_iter().flatten() {
        if prior_model_id != model_id {
            changed.push(prior_model_id);
        }
    }
    changed.sort();
    changed.dedup();
    changed
}

fn saturated_count(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

fn saturated_input_tokens(input_tokens: u128) -> u64 {
    u64::try_from(input_tokens).unwrap_or(u64::MAX)
}

fn saturated_queue_time_ms(input_tokens: u128, last_mean_input_tps: f64) -> u64 {
    debug_assert!(valid_last_mean_input_tps(last_mean_input_tps));
    let Ok(input_tokens) = u64::try_from(input_tokens) else {
        return u64::MAX;
    };
    queue_time_delta_ms(input_tokens, last_mean_input_tps).unwrap_or(u64::MAX)
}

impl TrackedPromptRequest {
    fn prompt_work(&self) -> Option<(u32, u64)> {
        Self::prompt_work_for(self.phase, self.priority, self.input_tokens)
    }

    fn prompt_work_for(
        phase: TrackedPromptPhase,
        priority: u32,
        input_tokens: u64,
    ) -> Option<(u32, u64)> {
        match phase {
            TrackedPromptPhase::OutputGeneration => None,
            TrackedPromptPhase::Pending => Some((priority, input_tokens)),
            TrackedPromptPhase::InputProcessing => Some((0, input_tokens)),
        }
        .filter(|(_, remaining)| *remaining > 0)
    }
}

impl LiveRequest {
    fn from_parts(
        queue: Option<TrackedPromptRequest>,
        observed: Option<ObservedRequestState>,
    ) -> Option<Self> {
        match (queue, observed) {
            (Some(queue), Some(observed)) => Some(Self::QueueAndObserved { queue, observed }),
            (Some(queue), None) => Some(Self::QueueOnly(queue)),
            (None, Some(observed)) => Some(Self::ObservedOnly(observed)),
            (None, None) => None,
        }
    }

    fn into_parts(self) -> (Option<TrackedPromptRequest>, Option<ObservedRequestState>) {
        match self {
            Self::QueueOnly(queue) => (Some(queue), None),
            Self::ObservedOnly(observed) => (None, Some(observed)),
            Self::QueueAndObserved { queue, observed } => (Some(queue), Some(observed)),
        }
    }

    fn queue(&self) -> Option<&TrackedPromptRequest> {
        match self {
            Self::QueueOnly(queue) | Self::QueueAndObserved { queue, .. } => Some(queue),
            Self::ObservedOnly(_) => None,
        }
    }

    fn queue_mut(&mut self) -> Option<&mut TrackedPromptRequest> {
        match self {
            Self::QueueOnly(queue) | Self::QueueAndObserved { queue, .. } => Some(queue),
            Self::ObservedOnly(_) => None,
        }
    }
}

impl QueueTrackedRequestGuard {
    pub(crate) fn on_upstream_response_headers(&mut self) {
        self.live_requests
            .advance_request_phase(&self.request_id, TrackedPromptPhase::InputProcessing);
    }

    pub(crate) fn observe_output(&mut self) {
        self.live_requests
            .advance_request_phase(&self.request_id, TrackedPromptPhase::OutputGeneration);
    }

    pub(crate) fn finish(&mut self) {
        if !self.finished {
            self.live_requests.finish_queue_request(&self.request_id);
            self.finished = true;
        }
    }
}

impl Drop for QueueTrackedRequestGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

fn parse_expected_queue_ms(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(HEADER_STARGATE_EXPECTED_QUEUE_MS)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
}

pub(crate) fn mismatch_threshold_ms(
    expected_ms: u64,
    config: &PylonQueueMismatchRetryConfig,
) -> u64 {
    let additive_threshold = expected_ms.saturating_add(config.min_delta_ms);
    let factor = if config.tolerance_factor.is_finite() && config.tolerance_factor > 0.0 {
        config.tolerance_factor
    } else {
        1.0
    };
    let multiplicative = ((expected_ms as f64) * factor).ceil();
    let multiplicative_threshold =
        if multiplicative.is_finite() && multiplicative <= u64::MAX as f64 {
            multiplicative as u64
        } else {
            u64::MAX
        };
    additive_threshold.max(multiplicative_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_observer::RequestObservationEndpoint;
    use reqwest::header::HeaderValue;
    use std::time::{Duration, Instant};

    fn required(request_id: &str, priority: u32, input_tokens: u64) -> RequiredTunnelHeaders {
        required_for_model(request_id, "model-a", priority, input_tokens)
    }

    fn required_for_model(
        request_id: &str,
        model_id: &str,
        priority: u32,
        input_tokens: u64,
    ) -> RequiredTunnelHeaders {
        RequiredTunnelHeaders {
            request_id: request_id.to_string(),
            routing_key: None,
            model_id: model_id.to_string(),
            priority,
            input_tokens,
            accepted_at: Instant::now(),
        }
    }

    fn headers_with_expected(expected_ms: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_STARGATE_EXPECTED_QUEUE_MS,
            HeaderValue::from_str(expected_ms).unwrap(),
        );
        headers
    }

    fn observation(request_id: &str, state: RequestObservationState) -> RequestObservation {
        RequestObservation {
            endpoint: RequestObservationEndpoint::ChatCompletions,
            request_id: request_id.to_string(),
            routing_key: None,
            model_id: "model-a".to_string(),
            priority: 0,
            input_tokens: 100,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: None,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state,
            time_to_response_headers: None,
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: Duration::ZERO,
        }
    }

    #[test]
    fn one_live_request_transition_updates_queue_and_active_output_load() {
        let live_requests = LiveRequestState::default();
        let _first = live_requests.track_request(&required("req-a", 0, 100));
        let _second = live_requests.track_request(&required("req-b", 0, 100));

        live_requests.transition_observation(&observation(
            "req-a",
            RequestObservationState::OutputGeneration,
        ));
        live_requests.update_active_output_tps("req-a", Some(20.0));
        live_requests.transition_observation(&observation(
            "req-b",
            RequestObservationState::OutputGeneration,
        ));
        live_requests.update_active_output_tps("req-b", Some(10.0));

        let active = live_requests.snapshot_model("model-a");
        assert_eq!(active.output_generation_queries, 2);
        assert_eq!(active.active_chat_output_tps, 15.0);

        live_requests
            .transition_observation(&observation("req-a", RequestObservationState::Complete));

        let terminal = live_requests.snapshot_model("model-a");
        assert_eq!(terminal.output_generation_queries, 1);
        assert_eq!(terminal.active_chat_output_tps, 10.0);
    }

    #[test]
    fn threshold_helper_accepts_at_threshold_and_rejects_above() {
        let config = PylonQueueMismatchRetryConfig::default();
        assert_eq!(mismatch_threshold_ms(100, &config), 125);
        assert!(126 > mismatch_threshold_ms(100, &config));
        assert!(125 <= mismatch_threshold_ms(100, &config));
        assert_eq!(mismatch_threshold_ms(0, &config), 25);
    }

    #[test]
    fn live_request_state_publishes_cumulative_priority_queue_estimates() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 100.0);
        let _priority_two = live_requests.track_request(&required("req-p2", 2, 20));
        let mut priority_four = live_requests.track_request(&required("req-p4", 4, 30));
        priority_four.on_upstream_response_headers();

        let snapshot = live_requests.snapshot_model("model-a");

        assert_eq!(snapshot.queue_size, 2);
        assert_eq!(snapshot.queued_input_size, 50);
        assert_eq!(
            snapshot.queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(0, 300), (2, 500)]))
        );
    }

    #[test]
    fn live_request_state_excludes_output_generation_and_drains_on_drop() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 100.0);
        let mut request = live_requests.track_request(&required("req-output", 2, 20));
        request.observe_output();

        let snapshot = live_requests.snapshot_model("model-a");
        assert_eq!(snapshot.queue_size, 0);
        assert_eq!(snapshot.queued_input_size, 0);
        assert_eq!(snapshot.output_generation_queries, 1);

        drop(request);
        assert_eq!(live_requests.tracked_request_count(), 0);
    }

    #[test]
    fn live_request_state_publishes_explicit_empty_priority_estimates_after_prompt_queue_drains() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 100.0);
        let request = live_requests.track_request(&required("req-drained", 2, 20));

        assert_eq!(
            live_requests
                .snapshot_model("model-a")
                .queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(2, 200)]))
        );

        drop(request);
        assert_eq!(
            live_requests
                .snapshot_model("model-a")
                .queue_time_estimate_ms_by_priority,
            Some(HashMap::new())
        );
    }

    #[test]
    fn queue_model_state_moves_replaced_request_identity_between_models() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 100.0);
        live_requests.update_model_throughput("model-b", 100.0);
        let _first =
            live_requests.track_request(&required_for_model("req-shared", "model-a", 4, 20));
        let _replacement =
            live_requests.track_request(&required_for_model("req-shared", "model-b", 2, 30));

        assert_eq!(
            live_requests.snapshot_model("model-a"),
            QueueModelSnapshot {
                queue_time_estimate_ms_by_priority: Some(HashMap::new()),
                ..QueueModelSnapshot::default()
            }
        );
        assert_eq!(
            live_requests.snapshot_model("model-b"),
            QueueModelSnapshot {
                queue_size: 1,
                queued_input_size: 30,
                num_running_queries: 1,
                total_query_input_size: 30,
                queue_time_estimate_ms_by_priority: Some(HashMap::from([(2, 300)])),
                ..QueueModelSnapshot::default()
            }
        );
    }

    #[test]
    fn queue_cleanup_preserves_observed_state_until_terminal_transition() {
        let live_requests = LiveRequestState::default();
        let mut request = live_requests.track_request(&required("req-finished", 0, 100));
        let live = live_requests.transition_observation(&observation(
            "req-finished",
            RequestObservationState::UpstreamConnecting,
        ));
        request.finish();
        assert_eq!(live_requests.tracked_request_count(), 0);

        let mut terminal_observation = observation("req-finished", RequestObservationState::Failed);
        terminal_observation.model_id = "model-b".to_string();
        let terminal = live_requests.transition_observation(&terminal_observation);

        assert_eq!(terminal.prior, live.current);
        assert_eq!(terminal.current, None);
        assert_eq!(
            terminal.changed_model_ids,
            ["model-a".to_string(), "model-b".to_string()]
        );
        assert_eq!(live_requests.tracked_request_count(), 0);
    }

    #[test]
    fn delayed_earlier_observation_does_not_requeue_output_generation() {
        let live_requests = LiveRequestState::default();
        let mut request = live_requests.track_request(&required("req-output", 0, 100));
        request.observe_output();

        live_requests.transition_observation(&observation(
            "req-output",
            RequestObservationState::UpstreamConnecting,
        ));

        let snapshot = live_requests.snapshot_model("model-a");
        assert_eq!(snapshot.queue_size, 0);
        assert_eq!(snapshot.output_generation_queries, 1);
    }

    #[test]
    fn queue_model_state_keeps_guard_phase_updates_monotonic() {
        let live_requests = LiveRequestState::default();
        let mut request = live_requests.track_request(&required("req-output", 0, 100));
        request.observe_output();
        request.on_upstream_response_headers();

        let snapshot = live_requests.snapshot_model("model-a");
        assert_eq!(snapshot.queue_size, 0);
        assert_eq!(snapshot.input_processing_queries, 0);
        assert_eq!(snapshot.output_generation_queries, 1);
    }

    #[test]
    fn queue_model_state_saturates_unrepresentable_valid_throughput_estimates() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", f64::MIN_POSITIVE);
        let _queued = live_requests.track_request(&required("req-huge", 0, u64::MAX));

        assert_eq!(
            live_requests
                .snapshot_model("model-a")
                .queue_time_estimate_ms_by_priority,
            Some(HashMap::from([(0, u64::MAX)]))
        );
        assert_eq!(
            live_requests.evaluate(
                &PylonQueueMismatchRetryConfig::default(),
                &required("req-new", 0, 1),
                &headers_with_expected("0"),
            ),
            QueueAdmissionDecision::Rejected {
                expected_ms: 0,
                actual_ms: u64::MAX,
                threshold_ms: 25,
                retry_after_ms: None,
            }
        );
    }

    #[test]
    fn queue_model_state_excludes_current_request_after_public_totals_saturate() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 1.0);
        let current = required("req-current-huge", 0, u64::MAX);
        let _current = live_requests.track_request(&current);
        let _other = live_requests.track_request(&required("req-other", 0, 1));

        let snapshot = live_requests.snapshot_model("model-a");
        assert_eq!(snapshot.queued_input_size, u64::MAX);
        assert_eq!(
            live_requests.evaluate(
                &PylonQueueMismatchRetryConfig::default(),
                &current,
                &headers_with_expected("0"),
            ),
            QueueAdmissionDecision::Rejected {
                expected_ms: 0,
                actual_ms: 1000,
                threshold_ms: 25,
                retry_after_ms: None,
            }
        );
    }

    #[test]
    fn upstream_response_headers_do_not_apply_retired_progress_contract() {
        let live_requests = LiveRequestState::default();
        let mut request = live_requests.track_request(&required("req-progress", 0, 100));

        request.on_upstream_response_headers();

        let snapshot = live_requests.snapshot_model("model-a");
        assert_eq!(snapshot.queued_input_size, 100);
        assert_eq!(snapshot.input_processing_queries, 1);
    }

    #[test]
    fn admission_transitions_from_unknown_to_rejected_after_throughput_update() {
        let live_requests = LiveRequestState::default();
        let config = PylonQueueMismatchRetryConfig::default();
        let request = required("req-inflight", 0, 100);
        let _guard = live_requests.track_request(&request);
        let incoming = required("req-new", 0, 1);
        let headers = headers_with_expected("0");

        assert_eq!(
            live_requests.evaluate(&config, &incoming, &headers),
            QueueAdmissionDecision::UnknownLocalEstimate { expected_ms: 0 }
        );

        live_requests.update_model_throughput("model-a", 100.0);
        assert_eq!(
            live_requests.evaluate(&config, &incoming, &headers),
            QueueAdmissionDecision::Rejected {
                expected_ms: 0,
                actual_ms: 1000,
                threshold_ms: 25,
                retry_after_ms: None,
            }
        );
    }

    #[test]
    fn admission_excludes_current_request_even_when_observed_before_evaluation() {
        let live_requests = LiveRequestState::default();
        let config = PylonQueueMismatchRetryConfig::default();
        let current = required("req-current", 0, 10);
        let _guard = live_requests.track_request(&current);
        live_requests.update_model_throughput("model-a", 100.0);

        assert_eq!(
            live_requests.evaluate(&config, &current, &headers_with_expected("0")),
            QueueAdmissionDecision::Accepted {
                expected_ms: 0,
                actual_ms: 0,
                threshold_ms: 25,
            }
        );
    }

    #[test]
    fn disabled_admission_accepts_with_distinct_metric_label() {
        let live_requests = LiveRequestState::default();
        live_requests.update_model_throughput("model-a", 100.0);
        let inflight = required("req-inflight", 0, 100);
        let _guard = live_requests.track_request(&inflight);
        let config = PylonQueueMismatchRetryConfig {
            enabled: false,
            ..PylonQueueMismatchRetryConfig::default()
        };

        let decision = live_requests.evaluate(
            &config,
            &required("req-new", 0, 1),
            &headers_with_expected("0"),
        );
        assert_eq!(decision, QueueAdmissionDecision::Disabled);
        assert_eq!(decision.result_label(), "disabled");
    }

    #[test]
    fn admission_accepts_missing_or_invalid_expected_queue_header() {
        let live_requests = LiveRequestState::default();
        let config = PylonQueueMismatchRetryConfig::default();
        let incoming = required("req-new", 0, 1);
        assert_eq!(
            live_requests.evaluate(&config, &incoming, &HeaderMap::new()),
            QueueAdmissionDecision::MissingEstimate
        );
        assert_eq!(
            live_requests.evaluate(&config, &incoming, &headers_with_expected("nope")),
            QueueAdmissionDecision::MissingEstimate
        );
    }
}
