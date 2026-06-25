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

use std::time::Instant;

use super::{
    RequestObservation, RequestObservationEndpoint, RequestObservationState, RequiredTunnelHeaders,
};
use crate::runtime_state::PylonRuntimeState;

pub(super) struct EmbeddingsRequestObserver {
    required: RequiredTunnelHeaders,
    started_at: Instant,
    embedding_items: Option<u64>,
    upstream_status: Option<u16>,
    response_headers_at: Option<Instant>,
    state: RequestObservationState,
    runtime_state: PylonRuntimeState,
}

impl EmbeddingsRequestObserver {
    pub(super) fn accepted(
        required: RequiredTunnelHeaders,
        runtime_state: PylonRuntimeState,
    ) -> Self {
        Self::with_embedding_items(required, None, runtime_state)
    }

    fn with_embedding_items(
        required: RequiredTunnelHeaders,
        embedding_items: Option<u64>,
        runtime_state: PylonRuntimeState,
    ) -> Self {
        let started_at = required.accepted_at;
        let mut observer = Self {
            required,
            started_at,
            embedding_items,
            upstream_status: None,
            response_headers_at: None,
            state: RequestObservationState::UpstreamConnecting,
            runtime_state,
        };
        observer.emit();
        observer
    }

    pub(super) fn update_embedding_items(&mut self, embedding_items: Option<u64>) {
        if self.embedding_items == embedding_items {
            return;
        }
        self.embedding_items = embedding_items;
        self.emit();
    }

    pub(super) fn on_upstream_response_headers(&mut self, status: u16) {
        if self.is_terminal() {
            panic!(
                "invalid response-header transition for request_id={} from state={:?}",
                self.required.request_id, self.state
            );
        }
        self.upstream_status = Some(status);
        self.response_headers_at = Some(Instant::now());
        self.state = RequestObservationState::InputProcessing;
        self.emit();
    }

    pub(super) fn finish(&mut self) {
        if self.is_terminal() {
            panic!(
                "invalid finish transition for request_id={} from state={:?}",
                self.required.request_id, self.state
            );
        }
        self.state = if self
            .upstream_status
            .is_some_and(|status| (200..300).contains(&status))
        {
            RequestObservationState::Complete
        } else {
            RequestObservationState::Failed
        };
        self.emit();
    }

    pub(super) fn fail(&mut self) {
        if self.is_terminal() {
            panic!(
                "invalid fail transition for request_id={} from state={:?}",
                self.required.request_id, self.state
            );
        }
        self.state = RequestObservationState::Failed;
        self.emit();
    }

    pub(super) fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            RequestObservationState::Complete
                | RequestObservationState::Failed
                | RequestObservationState::Cancelled
        )
    }

    fn cancel(&mut self) {
        if self.is_terminal() {
            panic!(
                "invalid cancel transition for request_id={} from state={:?}",
                self.required.request_id, self.state
            );
        }
        self.state = RequestObservationState::Cancelled;
        self.emit();
    }

    fn emit(&mut self) {
        let observation = RequestObservation {
            endpoint: RequestObservationEndpoint::Embeddings,
            request_id: self.required.request_id.clone(),
            routing_key: self.required.routing_key.clone(),
            model_id: self.required.model_id.clone(),
            priority: self.required.priority,
            input_tokens: self.required.input_tokens,
            embedding_items: self.embedding_items.unwrap_or_default(),
            embedding_items_observed: self.embedding_items.is_some(),
            upstream_status: self.upstream_status,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
            state: self.state,
            time_to_response_headers: self
                .response_headers_at
                .map(|instant| instant.saturating_duration_since(self.started_at)),
            time_to_first_output: None,
            time_to_first_token: None,
            total_duration: self.started_at.elapsed(),
        };

        tracing::info!(
            request_id = observation.request_id,
            endpoint = ?observation.endpoint,
            routing_key = observation.routing_key.as_deref().unwrap_or(""),
            model_id = observation.model_id.as_str(),
            priority = observation.priority,
            input_tokens = observation.input_tokens,
            embedding_items = ?observation
                .embedding_items_observed
                .then_some(observation.embedding_items),
            upstream_status = observation.upstream_status.unwrap_or_default(),
            state = ?observation.state,
            time_to_response_headers_ms = observation
                .time_to_response_headers
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or_default(),
            total_duration_ms = observation.total_duration.as_secs_f64() * 1000.0,
            "embeddings request observed"
        );

        self.runtime_state.observe_request(observation);
    }
}

impl Drop for EmbeddingsRequestObserver {
    fn drop(&mut self) {
        if !self.is_terminal() {
            self.cancel();
        }
    }
}

pub(super) fn embedding_items_from_request_body(body_bytes: &[u8]) -> Option<u64> {
    let value = serde_json::from_slice::<serde_json::Value>(body_bytes).ok()?;
    let input = value.get("input")?;
    match input {
        serde_json::Value::String(_) => Some(1),
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                return Some(0);
            }
            if items.iter().all(serde_json::Value::is_number) {
                Some(1)
            } else {
                u64::try_from(items.len()).ok()
            }
        }
        _ => None,
    }
}
