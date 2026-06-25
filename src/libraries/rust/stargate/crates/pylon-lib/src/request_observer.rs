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

use std::time::{Duration, Instant};

use reqwest::header::HeaderMap;
#[cfg(test)]
use stargate_protocol::tunnel_contract::{
    HEADER_INPUT_TOKENS, HEADER_MODEL, HEADER_REQUEST_ID, HEADER_ROUTING_KEY,
};

use crate::runtime_state::PylonRuntimeState;

mod embeddings;
mod headers;
mod tunnel;

#[cfg(test)]
use embeddings::{EmbeddingsRequestObserver, embedding_items_from_request_body};
#[cfg(test)]
use headers::RequiredHeaderErrorKind;
pub(crate) use headers::{
    MissingRequiredHeaderError, RequiredTunnelHeaders, validate_required_tunnel_headers,
};
pub(crate) use tunnel::TunnelRequestObserver;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestObservationState {
    Queued,
    UpstreamConnecting,
    InputProcessing,
    OutputGeneration,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestObservationEndpoint {
    ChatCompletions,
    Responses,
    Embeddings,
}

#[derive(Debug)]
enum RequestLifecycleState {
    Queued,
    UpstreamConnecting,
    InputProcessing(ResponsePhaseData),
    OutputGeneration {
        response: ResponsePhaseData,
        first_output_at: Instant,
        first_token_at: Option<Instant>,
    },
    Complete {
        response: ResponsePhaseData,
        first_output_at: Option<Instant>,
        first_token_at: Option<Instant>,
    },
    Failed {
        response: Option<ResponsePhaseData>,
        first_output_at: Option<Instant>,
        first_token_at: Option<Instant>,
    },
    Cancelled {
        response: Option<ResponsePhaseData>,
        first_output_at: Option<Instant>,
        first_token_at: Option<Instant>,
    },
}

impl RequestLifecycleState {
    fn observation_state(&self) -> RequestObservationState {
        match self {
            Self::Queued => RequestObservationState::Queued,
            Self::UpstreamConnecting => RequestObservationState::UpstreamConnecting,
            Self::InputProcessing(_) => RequestObservationState::InputProcessing,
            Self::OutputGeneration { .. } => RequestObservationState::OutputGeneration,
            Self::Complete { .. } => RequestObservationState::Complete,
            Self::Failed { .. } => RequestObservationState::Failed,
            Self::Cancelled { .. } => RequestObservationState::Cancelled,
        }
    }

    fn observation_state_name(&self) -> &'static str {
        match self.observation_state() {
            RequestObservationState::Queued => "Queued",
            RequestObservationState::UpstreamConnecting => "UpstreamConnecting",
            RequestObservationState::InputProcessing => "InputProcessing",
            RequestObservationState::OutputGeneration => "OutputGeneration",
            RequestObservationState::Complete => "Complete",
            RequestObservationState::Failed => "Failed",
            RequestObservationState::Cancelled => "Cancelled",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestObservation {
    pub endpoint: RequestObservationEndpoint,
    pub request_id: String,
    pub routing_key: Option<String>,
    pub model_id: String,
    pub priority: u32,
    pub input_tokens: u64,
    pub embedding_items: u64,
    pub embedding_items_observed: bool,
    pub upstream_status: Option<u16>,
    pub output_messages: u64,
    pub output_tokens: u64,
    pub output_tokens_explicit: bool,
    pub output_tokens_from_chunk_usage: bool,
    pub state: RequestObservationState,
    pub time_to_response_headers: Option<Duration>,
    pub time_to_first_output: Option<Duration>,
    pub time_to_first_token: Option<Duration>,
    pub total_duration: Duration,
}

impl RequestObservation {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            RequestObservationState::Complete
                | RequestObservationState::Failed
                | RequestObservationState::Cancelled
        )
    }
}

#[derive(Debug)]
struct ResponsePhaseData {
    upstream_status: u16,
    response_headers_at: Instant,
    output_messages: u64,
    output_tokens: u64,
    output_tokens_explicit: bool,
    output_tokens_from_chunk_usage: bool,
}

pub(crate) struct RequestObserver {
    endpoint: RequestObservationEndpoint,
    request_id: String,
    started_at: Instant,
    routing_key: Option<String>,
    model_id: String,
    priority: u32,
    input_tokens: u64,
    state: RequestLifecycleState,
    runtime_state: PylonRuntimeState,
}

impl RequestObserver {
    #[cfg(test)]
    pub(crate) fn new(
        request_headers: &HeaderMap,
        runtime_state: PylonRuntimeState,
    ) -> Result<Self, MissingRequiredHeaderError> {
        Ok(Self::from_required(
            RequestObservationEndpoint::ChatCompletions,
            validate_required_tunnel_headers(request_headers)?,
            runtime_state,
        ))
    }

    pub(crate) fn from_required(
        endpoint: RequestObservationEndpoint,
        required: RequiredTunnelHeaders,
        runtime_state: PylonRuntimeState,
    ) -> Self {
        let RequiredTunnelHeaders {
            request_id,
            routing_key,
            model_id,
            priority,
            input_tokens,
            accepted_at,
        } = required;
        let mut observer = Self {
            endpoint,
            request_id,
            started_at: accepted_at,
            routing_key,
            model_id,
            priority,
            input_tokens,
            state: RequestLifecycleState::UpstreamConnecting,
            runtime_state,
        };
        observer.emit();
        observer
    }

    pub(crate) fn on_upstream_response_headers(
        &mut self,
        _response_headers: &HeaderMap,
        status: u16,
    ) {
        match self.state {
            RequestLifecycleState::UpstreamConnecting => {}
            RequestLifecycleState::Queued
            | RequestLifecycleState::InputProcessing(_)
            | RequestLifecycleState::OutputGeneration { .. }
            | RequestLifecycleState::Complete { .. }
            | RequestLifecycleState::Failed { .. }
            | RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid response-header transition for request_id={} from state={}",
                self.request_id,
                self.state.observation_state_name()
            ),
        }

        let response_headers_at = Instant::now();
        self.state = RequestLifecycleState::InputProcessing(ResponsePhaseData {
            upstream_status: status,
            response_headers_at,
            output_messages: 0,
            output_tokens: 0,
            output_tokens_explicit: false,
            output_tokens_from_chunk_usage: false,
        });
        self.emit();
    }

    pub(crate) fn observe_output_message(&mut self) {
        match &mut self.state {
            RequestLifecycleState::InputProcessing(_) => {
                self.record_output_message();
                self.emit();
            }
            RequestLifecycleState::OutputGeneration { response, .. } => {
                response.output_messages += 1;
                self.emit();
            }
            RequestLifecycleState::Queued
            | RequestLifecycleState::UpstreamConnecting
            | RequestLifecycleState::Complete { .. }
            | RequestLifecycleState::Failed { .. }
            | RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid output observation transition for request_id={} from state={}",
                self.request_id,
                self.state.observation_state_name()
            ),
        }
    }

    pub(crate) fn observe_output_tokens(&mut self, output_tokens: u64) {
        if output_tokens == 0 {
            return;
        }

        match &mut self.state {
            RequestLifecycleState::InputProcessing(response) => {
                if response.output_tokens_explicit {
                    return;
                }
                self.record_output_tokens(output_tokens);
                self.emit();
            }
            RequestLifecycleState::OutputGeneration {
                response,
                first_token_at,
                ..
            } => {
                if response.output_tokens_explicit {
                    return;
                }
                response.output_tokens += output_tokens;
                if first_token_at.is_none() {
                    *first_token_at = Some(Instant::now());
                }
                self.emit();
            }
            RequestLifecycleState::Queued
            | RequestLifecycleState::UpstreamConnecting
            | RequestLifecycleState::Complete { .. }
            | RequestLifecycleState::Failed { .. }
            | RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid output token observation transition for request_id={} from state={}",
                self.request_id,
                self.state.observation_state_name()
            ),
        }
    }

    pub(crate) fn observe_output_tokens_generated_so_far(&mut self, output_tokens: u64) {
        match &mut self.state {
            RequestLifecycleState::InputProcessing(_) => {
                if self.record_output_tokens_generated_so_far(output_tokens) {
                    self.emit();
                }
            }
            RequestLifecycleState::OutputGeneration {
                response,
                first_token_at,
                ..
            } => {
                if response.output_tokens_explicit && output_tokens < response.output_tokens {
                    tracing::warn!(
                        request_id = self.request_id,
                        prior_output_tokens = response.output_tokens,
                        output_tokens_generated_so_far = output_tokens,
                        "ignoring regressing explicit output token counter"
                    );
                    return;
                }
                if response.output_tokens_explicit
                    && response.output_tokens_from_chunk_usage
                    && output_tokens == response.output_tokens
                {
                    return;
                }
                let should_emit = output_tokens > 0 || output_tokens != response.output_tokens;
                response.output_tokens = output_tokens;
                response.output_tokens_explicit = true;
                response.output_tokens_from_chunk_usage = true;
                if output_tokens > 0 && first_token_at.is_none() {
                    *first_token_at = Some(Instant::now());
                }
                if should_emit {
                    self.emit();
                }
            }
            RequestLifecycleState::Queued
            | RequestLifecycleState::UpstreamConnecting
            | RequestLifecycleState::Complete { .. }
            | RequestLifecycleState::Failed { .. }
            | RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid output token observation transition for request_id={} from state={}",
                self.request_id,
                self.state.observation_state_name()
            ),
        }
    }

    pub(crate) fn finish(&mut self) {
        let state = self.take_state();
        self.state = match state {
            RequestLifecycleState::InputProcessing(response) => {
                if (200..300).contains(&response.upstream_status) && response.output_messages == 0 {
                    panic!(
                        "invalid finish transition for request_id={} from state=InputProcessing without observed output",
                        self.request_id
                    )
                } else {
                    RequestLifecycleState::Failed {
                        response: Some(response),
                        first_output_at: None,
                        first_token_at: None,
                    }
                }
            }
            RequestLifecycleState::OutputGeneration {
                response,
                first_output_at,
                first_token_at,
            } => {
                if (200..300).contains(&response.upstream_status) {
                    RequestLifecycleState::Complete {
                        response,
                        first_output_at: Some(first_output_at),
                        first_token_at,
                    }
                } else {
                    RequestLifecycleState::Failed {
                        response: Some(response),
                        first_output_at: Some(first_output_at),
                        first_token_at,
                    }
                }
            }
            RequestLifecycleState::Failed { .. } => panic!(
                "invalid finish transition for request_id={} from state=Failed",
                self.request_id
            ),
            RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid finish transition for request_id={} from state=Cancelled",
                self.request_id
            ),
            RequestLifecycleState::Queued => panic!(
                "invalid finish transition for request_id={} from state=Queued",
                self.request_id
            ),
            RequestLifecycleState::UpstreamConnecting => RequestLifecycleState::Failed {
                response: None,
                first_output_at: None,
                first_token_at: None,
            },
            RequestLifecycleState::Complete { .. } => panic!(
                "invalid finish transition for request_id={} from state=Complete",
                self.request_id
            ),
        };

        self.emit();
    }

    pub(crate) fn fail(&mut self) {
        let state = self.take_state();
        self.state = match state {
            RequestLifecycleState::InputProcessing(response) => RequestLifecycleState::Failed {
                response: Some(response),
                first_output_at: None,
                first_token_at: None,
            },
            RequestLifecycleState::OutputGeneration {
                response,
                first_output_at,
                first_token_at,
            } => RequestLifecycleState::Failed {
                response: Some(response),
                first_output_at: Some(first_output_at),
                first_token_at,
            },
            RequestLifecycleState::Complete { .. } => panic!(
                "invalid fail transition for request_id={} from state=Complete",
                self.request_id
            ),
            RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid fail transition for request_id={} from state=Cancelled",
                self.request_id
            ),
            RequestLifecycleState::Failed { .. } => panic!(
                "invalid fail transition for request_id={} from state=Failed",
                self.request_id
            ),
            RequestLifecycleState::Queued | RequestLifecycleState::UpstreamConnecting => {
                RequestLifecycleState::Failed {
                    response: None,
                    first_output_at: None,
                    first_token_at: None,
                }
            }
        };
        self.emit();
    }

    fn cancel(&mut self) {
        let state = self.take_state();
        self.state = match state {
            RequestLifecycleState::InputProcessing(response) => RequestLifecycleState::Cancelled {
                response: Some(response),
                first_output_at: None,
                first_token_at: None,
            },
            RequestLifecycleState::OutputGeneration {
                response,
                first_output_at,
                first_token_at,
            } => RequestLifecycleState::Cancelled {
                response: Some(response),
                first_output_at: Some(first_output_at),
                first_token_at,
            },
            RequestLifecycleState::Queued | RequestLifecycleState::UpstreamConnecting => {
                RequestLifecycleState::Cancelled {
                    response: None,
                    first_output_at: None,
                    first_token_at: None,
                }
            }
            RequestLifecycleState::Complete { .. } => panic!(
                "invalid cancel transition for request_id={} from state=Complete",
                self.request_id
            ),
            RequestLifecycleState::Failed { .. } => panic!(
                "invalid cancel transition for request_id={} from state=Failed",
                self.request_id
            ),
            RequestLifecycleState::Cancelled { .. } => panic!(
                "invalid cancel transition for request_id={} from state=Cancelled",
                self.request_id
            ),
        };
        self.emit();
    }

    pub(crate) fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            RequestLifecycleState::Complete { .. }
                | RequestLifecycleState::Failed { .. }
                | RequestLifecycleState::Cancelled { .. }
        )
    }

    fn record_output_message(&mut self) {
        let state = self.take_state();
        self.state = match state {
            RequestLifecycleState::InputProcessing(mut response) => {
                response.output_messages += 1;
                RequestLifecycleState::OutputGeneration {
                    response,
                    first_output_at: Instant::now(),
                    first_token_at: None,
                }
            }
            RequestLifecycleState::OutputGeneration {
                mut response,
                first_output_at,
                first_token_at,
            } => {
                response.output_messages += 1;
                RequestLifecycleState::OutputGeneration {
                    response,
                    first_output_at,
                    first_token_at,
                }
            }
            other => {
                debug_assert!(
                    matches!(
                        other,
                        RequestLifecycleState::InputProcessing(_)
                            | RequestLifecycleState::OutputGeneration { .. }
                    ),
                    "record_output_message called from invalid state {}",
                    other.observation_state_name()
                );
                other
            }
        };
    }

    fn record_output_tokens(&mut self, output_tokens: u64) {
        let state = self.take_state();
        self.state = match state {
            RequestLifecycleState::InputProcessing(mut response) => {
                if response.output_tokens_explicit {
                    RequestLifecycleState::InputProcessing(response)
                } else {
                    let now = Instant::now();
                    response.output_tokens += output_tokens;
                    RequestLifecycleState::OutputGeneration {
                        response,
                        first_output_at: now,
                        first_token_at: Some(now),
                    }
                }
            }
            other => {
                debug_assert!(
                    matches!(other, RequestLifecycleState::InputProcessing(_)),
                    "record_output_tokens called from invalid state {}",
                    other.observation_state_name()
                );
                other
            }
        };
    }

    fn record_output_tokens_generated_so_far(&mut self, output_tokens: u64) -> bool {
        let state = self.take_state();
        let changed;
        self.state = match state {
            RequestLifecycleState::InputProcessing(mut response) => {
                if response.output_tokens_explicit && output_tokens < response.output_tokens {
                    tracing::warn!(
                        request_id = self.request_id,
                        prior_output_tokens = response.output_tokens,
                        output_tokens_generated_so_far = output_tokens,
                        "ignoring regressing explicit output token counter"
                    );
                    changed = false;
                    RequestLifecycleState::InputProcessing(response)
                } else if response.output_tokens_explicit
                    && response.output_tokens_from_chunk_usage
                    && output_tokens == response.output_tokens
                {
                    changed = false;
                    RequestLifecycleState::InputProcessing(response)
                } else {
                    changed = output_tokens > 0;
                    response.output_tokens = output_tokens;
                    response.output_tokens_explicit = true;
                    response.output_tokens_from_chunk_usage = true;
                    if output_tokens == 0 {
                        RequestLifecycleState::InputProcessing(response)
                    } else {
                        let now = Instant::now();
                        RequestLifecycleState::OutputGeneration {
                            response,
                            first_output_at: now,
                            first_token_at: Some(now),
                        }
                    }
                }
            }
            other => {
                debug_assert!(
                    matches!(other, RequestLifecycleState::InputProcessing(_)),
                    "record_output_tokens_generated_so_far called from invalid state {}",
                    other.observation_state_name()
                );
                changed = false;
                other
            }
        };
        changed
    }

    fn take_state(&mut self) -> RequestLifecycleState {
        // Queued is a mechanical placeholder used only to move the enum out for transition logic.
        std::mem::replace(&mut self.state, RequestLifecycleState::Queued)
    }

    fn response_snapshot(&self) -> (Option<&ResponsePhaseData>, Option<Instant>, Option<Instant>) {
        match &self.state {
            RequestLifecycleState::InputProcessing(response) => (Some(response), None, None),
            RequestLifecycleState::OutputGeneration {
                response,
                first_output_at,
                first_token_at,
            } => (Some(response), Some(*first_output_at), *first_token_at),
            RequestLifecycleState::Complete {
                response,
                first_output_at,
                first_token_at,
            } => (Some(response), *first_output_at, *first_token_at),
            RequestLifecycleState::Failed {
                response,
                first_output_at,
                first_token_at,
            }
            | RequestLifecycleState::Cancelled {
                response,
                first_output_at,
                first_token_at,
            } => (response.as_ref(), *first_output_at, *first_token_at),
            RequestLifecycleState::Queued | RequestLifecycleState::UpstreamConnecting => {
                (None, None, None)
            }
        }
    }

    fn emit(&mut self) {
        let (response, first_output_at, first_token_at) = self.response_snapshot();
        let observation = RequestObservation {
            endpoint: self.endpoint,
            request_id: self.request_id.clone(),
            routing_key: self.routing_key.clone(),
            model_id: self.model_id.clone(),
            priority: self.priority,
            input_tokens: self.input_tokens,
            embedding_items: 0,
            embedding_items_observed: false,
            upstream_status: response.map(|response| response.upstream_status),
            output_messages: response
                .map(|response| response.output_messages)
                .unwrap_or(0),
            output_tokens: response.map(|response| response.output_tokens).unwrap_or(0),
            output_tokens_explicit: response
                .map(|response| response.output_tokens_explicit)
                .unwrap_or(false),
            output_tokens_from_chunk_usage: response
                .map(|response| response.output_tokens_from_chunk_usage)
                .unwrap_or(false),
            state: self.state.observation_state(),
            // Observation timestamps can be coarser than event sequencing; never underflow
            // durations when two instants collapse to the same clock tick.
            time_to_response_headers: response.map(|response| {
                response
                    .response_headers_at
                    .saturating_duration_since(self.started_at)
            }),
            time_to_first_output: first_output_at
                .map(|instant| instant.saturating_duration_since(self.started_at)),
            time_to_first_token: first_token_at
                .map(|instant| instant.saturating_duration_since(self.started_at)),
            total_duration: self.started_at.elapsed(),
        };

        tracing::info!(
            request_id = observation.request_id,
            endpoint = ?observation.endpoint,
            routing_key = observation.routing_key.as_deref().unwrap_or(""),
            model_id = observation.model_id.as_str(),
            priority = observation.priority,
            input_tokens = observation.input_tokens,
            upstream_status = observation.upstream_status.unwrap_or_default(),
            output_messages = observation.output_messages,
            output_tokens = observation.output_tokens,
            output_tokens_explicit = observation.output_tokens_explicit,
            output_tokens_from_chunk_usage = observation.output_tokens_from_chunk_usage,
            state = ?observation.state,
            time_to_response_headers_ms = observation
                .time_to_response_headers
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or_default(),
            time_to_first_output_ms = observation
                .time_to_first_output
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or_default(),
            time_to_first_token_ms = observation
                .time_to_first_token
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or_default(),
            total_duration_ms = observation.total_duration.as_secs_f64() * 1000.0,
            "client request observed"
        );

        self.runtime_state.observe_request(observation);
    }
}

impl Drop for RequestObserver {
    fn drop(&mut self) {
        if !self.is_terminal() {
            self.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use stargate_proto::pb::InferenceServerStatus;

    use super::*;

    fn observed_runtime(
        capacity: usize,
    ) -> (
        PylonRuntimeState,
        flume::Receiver<crate::RequestObservationEvent>,
    ) {
        PylonRuntimeState::observed(InferenceServerStatus::Unknown, &[], capacity, None)
    }

    async fn recv_observation(
        rx: &flume::Receiver<crate::RequestObservationEvent>,
        context: &'static str,
    ) -> RequestObservation {
        tokio::time::timeout(Duration::from_secs(1), rx.recv_async())
            .await
            .expect(context)
            .expect("observation channel should remain open")
            .into_observation()
    }

    fn recv_observation_blocking(
        rx: &flume::Receiver<crate::RequestObservationEvent>,
        context: &'static str,
    ) -> RequestObservation {
        rx.try_recv().expect(context).into_observation()
    }

    #[test]
    fn validate_required_tunnel_headers_accepts_required_values() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());
        headers.insert("x-priority", "7".parse().unwrap());

        let required = validate_required_tunnel_headers(&headers).unwrap();

        assert_eq!(required.request_id, "req-1");
        assert_eq!(required.routing_key.as_deref(), Some("rk-1"));
        assert_eq!(required.model_id, "model-a");
        assert_eq!(required.input_tokens, 42);
        assert_eq!(required.priority, 7);
    }

    #[test]
    fn validate_required_tunnel_headers_defaults_missing_priority_to_zero() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let required = validate_required_tunnel_headers(&headers).unwrap();

        assert_eq!(required.priority, 0);
    }

    #[test]
    fn validate_required_tunnel_headers_rejects_malformed_priority() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());
        headers.insert("x-priority", "not-a-priority".parse().unwrap());

        let error = validate_required_tunnel_headers(&headers).unwrap_err();

        assert_eq!(error.header_name, "x-priority");
        assert_eq!(error.kind, RequiredHeaderErrorKind::Invalid);
    }

    #[test]
    fn validate_required_tunnel_headers_rejects_malformed_input_tokens() {
        for value in [
            "not-a-token-count".parse().unwrap(),
            "".parse().unwrap(),
            " ".parse().unwrap(),
            reqwest::header::HeaderValue::from_bytes(&[0xff]).unwrap(),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
            headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
            headers.insert(HEADER_INPUT_TOKENS, value);

            let error = validate_required_tunnel_headers(&headers).unwrap_err();

            assert_eq!(error.header_name, HEADER_INPUT_TOKENS);
            assert_eq!(error.kind, RequiredHeaderErrorKind::Invalid);
        }
    }

    #[test]
    fn validate_required_tunnel_headers_rejects_missing_required_values() {
        for missing in [HEADER_REQUEST_ID, HEADER_MODEL, HEADER_INPUT_TOKENS] {
            let mut headers = HeaderMap::new();
            headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
            headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
            headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());
            headers.remove(missing);

            let error = validate_required_tunnel_headers(&headers).unwrap_err();

            assert_eq!(error.header_name, missing);
            assert_eq!(error.kind, RequiredHeaderErrorKind::Missing);
        }
    }

    #[test]
    fn embeddings_observation_counts_input_cardinality() {
        assert_eq!(
            embedding_items_from_request_body(br#"{"input":"hello"}"#).unwrap(),
            1
        );
        assert_eq!(
            embedding_items_from_request_body(br#"{"input":["a","b"]}"#).unwrap(),
            2
        );
        assert_eq!(
            embedding_items_from_request_body(br#"{"input":[1,2,3]}"#).unwrap(),
            1
        );
        assert_eq!(
            embedding_items_from_request_body(br#"{"input":[[1,2],[3,4],[5]]}"#).unwrap(),
            3
        );
        assert_eq!(
            embedding_items_from_request_body(br#"{"input":[]}"#).unwrap(),
            0
        );
        assert_eq!(embedding_items_from_request_body(br#"{"input":"#), None);
    }

    fn embeddings_required_headers() -> RequiredTunnelHeaders {
        RequiredTunnelHeaders {
            request_id: "req-embeddings-terminal".to_string(),
            routing_key: Some("rk-1".to_string()),
            model_id: "model-embed".to_string(),
            priority: 0,
            input_tokens: 12,
            accepted_at: Instant::now(),
        }
    }

    #[test]
    fn tunnel_observer_drop_fails_each_observed_endpoint() {
        for endpoint in [
            RequestObservationEndpoint::ChatCompletions,
            RequestObservationEndpoint::Responses,
            RequestObservationEndpoint::Embeddings,
        ] {
            let (runtime_state, rx) = observed_runtime(4);
            let observer = TunnelRequestObserver::accepted(
                endpoint,
                embeddings_required_headers(),
                runtime_state,
            );

            // Dropping a live tunnel observer is the terminal behavior under test.
            drop(observer);

            let observations = rx
                .try_iter()
                .map(crate::RequestObservationEvent::into_observation)
                .collect::<Vec<_>>();
            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].endpoint, endpoint);
            assert_eq!(
                observations[0].state,
                RequestObservationState::UpstreamConnecting
            );
            assert_eq!(observations[1].endpoint, endpoint);
            assert_eq!(observations[1].state, RequestObservationState::Failed);
        }
    }

    #[test]
    fn terminal_tunnel_observer_drop_emits_nothing() {
        for endpoint in [
            RequestObservationEndpoint::ChatCompletions,
            RequestObservationEndpoint::Responses,
            RequestObservationEndpoint::Embeddings,
        ] {
            let (runtime_state, rx) = observed_runtime(8);
            let mut observer = TunnelRequestObserver::accepted(
                endpoint,
                embeddings_required_headers(),
                runtime_state,
            );
            observer.on_upstream_response_headers(&HeaderMap::new(), 200);
            if let Some(generation) = observer.generation_mut() {
                generation.observe_output_message();
            }
            observer.finish();
            let observations_before_drop = rx.len();

            // Dropping a terminal observer proves the wrapper does not emit again.
            drop(observer);

            assert_eq!(rx.len(), observations_before_drop);
        }
    }

    #[test]
    fn embeddings_observer_uses_request_acceptance_time() {
        let mut required = embeddings_required_headers();
        required.accepted_at = Instant::now()
            .checked_sub(Duration::from_millis(50))
            .expect("test acceptance time should be representable");
        let (runtime_state, rx) = observed_runtime(4);

        let _observer = EmbeddingsRequestObserver::accepted(required, runtime_state);
        let observation = recv_observation_blocking(&rx, "accepted embeddings observation");

        assert!(observation.total_duration >= Duration::from_millis(40));
    }

    #[test]
    fn embeddings_observer_can_record_cardinality_after_acceptance() {
        let (runtime_state, rx) = observed_runtime(4);
        let mut observer =
            EmbeddingsRequestObserver::accepted(embeddings_required_headers(), runtime_state);

        let accepted = recv_observation_blocking(&rx, "accepted embeddings observation");
        assert_eq!(accepted.embedding_items, 0);
        assert!(!accepted.embedding_items_observed);

        observer.update_embedding_items(Some(0));
        let parsed = recv_observation_blocking(&rx, "parsed embeddings observation");
        assert_eq!(parsed.embedding_items, 0);
        assert!(parsed.embedding_items_observed);
    }

    #[test]
    #[should_panic(expected = "invalid fail transition")]
    fn embeddings_observer_rejects_terminal_fail_transition() {
        let mut observer = EmbeddingsRequestObserver::accepted(
            embeddings_required_headers(),
            PylonRuntimeState::default(),
        );
        observer.update_embedding_items(Some(1));
        observer.on_upstream_response_headers(200);
        observer.finish();
        observer.fail();
    }

    #[test]
    #[should_panic(expected = "invalid finish transition")]
    fn embeddings_observer_rejects_terminal_finish_transition() {
        let mut observer = EmbeddingsRequestObserver::accepted(
            embeddings_required_headers(),
            PylonRuntimeState::default(),
        );
        observer.update_embedding_items(Some(1));
        observer.fail();
        observer.finish();
    }

    #[tokio::test]
    async fn counts_sse_events_across_chunk_boundaries() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let mut observer = RequestObserver::new(&headers, PylonRuntimeState::default()).unwrap();
        observer.on_upstream_response_headers(&response_headers, 200);
        observer.observe_output_message();
        observer.observe_output_message();

        observer.finish();

        let (response, _, _) = observer.response_snapshot();
        let response = response.unwrap();
        assert_eq!(response.output_messages, 2);
        assert_eq!(response.output_tokens, 0);
        assert_eq!(
            observer.state.observation_state(),
            RequestObservationState::Complete
        );
    }

    #[tokio::test]
    async fn non_terminal_updates_are_emitted() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-live".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();

        let initial = recv_observation(&rx, "initial observation should be emitted").await;
        assert_eq!(initial.state, RequestObservationState::UpstreamConnecting);
        assert!(!initial.is_terminal());

        observer.on_upstream_response_headers(&response_headers, 200);
        let first = recv_observation(&rx, "response-header observation should be emitted").await;
        assert_eq!(first.state, RequestObservationState::InputProcessing);
        assert!(!first.is_terminal());

        observer.observe_output_message();
        let second = recv_observation(&rx, "output observation should be emitted").await;
        assert_eq!(second.state, RequestObservationState::OutputGeneration);
        assert_eq!(second.output_messages, 1);
        assert!(!second.is_terminal());
    }

    #[tokio::test]
    async fn upstream_connecting_observation_is_emitted_when_request_starts() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-connect".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let (runtime_state, rx) = observed_runtime(8);
        let _observer = RequestObserver::new(&headers, runtime_state).unwrap();

        let observation = recv_observation(&rx, "initial observation should be emitted").await;
        assert_eq!(observation.request_id, "req-connect");
        assert_eq!(
            observation.state,
            RequestObservationState::UpstreamConnecting
        );
        assert_eq!(observation.upstream_status, None);
        assert!(!observation.is_terminal());
    }

    #[tokio::test]
    async fn dropping_nonterminal_observer_emits_cancelled() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-cancel".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let (runtime_state, rx) = observed_runtime(8);
        let observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let initial = recv_observation(&rx, "initial observation should be emitted").await;
        assert_eq!(initial.state, RequestObservationState::UpstreamConnecting);

        drop(observer);

        let terminal = recv_observation(&rx, "drop should emit terminal observation").await;
        assert_eq!(terminal.state, RequestObservationState::Cancelled);
        assert!(terminal.is_terminal());
    }

    #[tokio::test]
    async fn accumulates_output_tokens() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-1".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let mut observer = RequestObserver::new(&headers, PylonRuntimeState::default()).unwrap();
        observer.on_upstream_response_headers(&response_headers, 200);
        observer.observe_output_message();
        observer.observe_output_tokens(3);
        observer.observe_output_message();
        observer.observe_output_tokens(2);
        observer.finish();

        let (response, _, _) = observer.response_snapshot();
        let response = response.unwrap();
        assert_eq!(response.output_messages, 2);
        assert_eq!(response.output_tokens, 5);
    }

    #[tokio::test]
    async fn first_positive_output_tokens_start_real_ttft() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-token".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let _ = recv_observation(&rx, "initial observation should be emitted").await;
        observer.on_upstream_response_headers(&response_headers, 200);
        let _ = recv_observation(&rx, "response-header observation should be emitted").await;

        observer.observe_output_tokens(3);
        let token_observation = recv_observation(&rx, "token observation should be emitted").await;
        assert_eq!(
            token_observation.state,
            RequestObservationState::OutputGeneration
        );
        assert_eq!(token_observation.output_tokens, 3);
        assert!(token_observation.time_to_first_output.is_some());
        assert!(token_observation.time_to_first_token.is_some());
    }

    #[tokio::test]
    async fn response_headers_do_not_disable_text_fallback() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-zero-header-output".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let response_headers = HeaderMap::new();

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let _ = recv_observation(&rx, "initial observation should be emitted").await;
        observer.on_upstream_response_headers(&response_headers, 200);
        let header_observation =
            recv_observation(&rx, "response-header observation should be emitted").await;
        assert_eq!(header_observation.output_tokens, 0);
        assert!(!header_observation.output_tokens_explicit);

        observer.observe_output_tokens(3);
        let estimated_observation =
            recv_observation(&rx, "fallback token observation should be emitted").await;
        assert_eq!(
            estimated_observation.state,
            RequestObservationState::OutputGeneration
        );
        assert_eq!(estimated_observation.output_tokens, 3);
        assert!(!estimated_observation.output_tokens_explicit);
        assert!(estimated_observation.time_to_first_token.is_some());
    }

    #[tokio::test]
    async fn explicit_output_counter_corrects_prior_estimated_tokens() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-explicit-output".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let _ = recv_observation(&rx, "initial observation should be emitted").await;
        observer.on_upstream_response_headers(&response_headers, 200);
        let _ = recv_observation(&rx, "response-header observation should be emitted").await;

        observer.observe_output_tokens(5);
        let estimated =
            recv_observation(&rx, "estimated token observation should be emitted").await;
        assert_eq!(estimated.output_tokens, 5);
        assert!(!estimated.output_tokens_explicit);

        observer.observe_output_tokens_generated_so_far(3);
        let explicit = recv_observation(&rx, "explicit token observation should be emitted").await;
        assert_eq!(explicit.output_tokens, 3);
        assert!(explicit.output_tokens_explicit);
        assert!(explicit.output_tokens_from_chunk_usage);

        observer.observe_output_tokens_generated_so_far(3);
        assert!(
            rx.is_empty(),
            "repeated explicit counters with no value change should not emit"
        );

        observer.observe_output_tokens(10);
        assert!(
            rx.is_empty(),
            "fallback deltas should not emit after explicit counters"
        );
    }

    #[tokio::test]
    async fn zero_explicit_counter_marks_chunk_usage_without_extra_live_emit() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-zero-explicit".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let _ = recv_observation(&rx, "initial observation should be emitted").await;
        observer.on_upstream_response_headers(&response_headers, 200);
        let _ = recv_observation(&rx, "response-header observation should be emitted").await;

        observer.observe_output_message();
        let output_observation =
            recv_observation(&rx, "output observation should be emitted").await;
        assert_eq!(
            output_observation.state,
            RequestObservationState::OutputGeneration
        );
        assert_eq!(output_observation.output_tokens, 0);
        assert!(!output_observation.output_tokens_from_chunk_usage);

        observer.observe_output_tokens_generated_so_far(0);
        assert!(
            rx.is_empty(),
            "zero-token explicit counters should not emit a duplicate live update"
        );

        observer.finish();
        let terminal_observation =
            recv_observation(&rx, "terminal observation should be emitted").await;
        assert!(terminal_observation.is_terminal());
        assert_eq!(terminal_observation.output_tokens, 0);
        assert!(terminal_observation.output_tokens_explicit);
        assert!(terminal_observation.output_tokens_from_chunk_usage);
    }

    #[tokio::test]
    async fn chunk_usage_counter_before_output_controls_live_token_accounting() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-early-chunk-usage".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "42".parse().unwrap());

        let (runtime_state, rx) = observed_runtime(8);
        let mut observer = RequestObserver::new(&headers, runtime_state).unwrap();
        let _ = recv_observation(&rx, "initial observation should be emitted").await;
        observer.on_upstream_response_headers(&HeaderMap::new(), 200);
        let _ = recv_observation(&rx, "response-header observation should be emitted").await;

        observer.observe_output_tokens_generated_so_far(0);
        assert!(
            rx.is_empty(),
            "zero-token chunk usage before output should not emit a duplicate live update"
        );

        observer.observe_output_tokens(3);
        assert!(
            rx.is_empty(),
            "fallback token estimates should not emit after chunk usage becomes explicit"
        );

        observer.observe_output_tokens_generated_so_far(4);
        let explicit = recv_observation(&rx, "positive chunk usage should be emitted").await;
        assert_eq!(explicit.state, RequestObservationState::OutputGeneration);
        assert_eq!(explicit.output_tokens, 4);
        assert!(explicit.output_tokens_explicit);
        assert!(explicit.output_tokens_from_chunk_usage);
        assert!(explicit.time_to_first_output.is_some());
        assert!(explicit.time_to_first_token.is_some());

        observer.observe_output_tokens_generated_so_far(4);
        assert!(
            rx.is_empty(),
            "repeated chunk usage counters with no value change should not emit"
        );

        observer.observe_output_tokens_generated_so_far(2);
        assert!(
            rx.is_empty(),
            "regressing chunk usage counters should not emit"
        );
    }

    #[test]
    fn input_processing_chunk_usage_counter_rejects_regression() {
        let mut observer = make_test_observer();
        let started_at = Instant::now() - Duration::from_secs(1);
        observer.started_at = started_at;
        observer.state = RequestLifecycleState::InputProcessing(ResponsePhaseData {
            upstream_status: 200,
            response_headers_at: started_at + Duration::from_millis(10),
            output_messages: 0,
            output_tokens: 5,
            output_tokens_explicit: true,
            output_tokens_from_chunk_usage: true,
        });

        observer.observe_output_tokens_generated_so_far(3);

        let (response, first_output_at, first_token_at) = observer.response_snapshot();
        let response = response.unwrap();
        assert_eq!(response.output_tokens, 5);
        assert!(response.output_tokens_explicit);
        assert!(response.output_tokens_from_chunk_usage);
        assert_eq!(
            observer.state.observation_state(),
            RequestObservationState::InputProcessing
        );
        assert_eq!(first_output_at, None);
        assert_eq!(first_token_at, None);
    }

    #[test]
    fn late_usage_tokens_preserve_actual_first_token_time() {
        let mut observer = make_test_observer();
        let started_at = Instant::now() - Duration::from_secs(10);
        let first_output_at = started_at + Duration::from_secs(2);
        observer.started_at = started_at;
        observer.state = RequestLifecycleState::OutputGeneration {
            response: ResponsePhaseData {
                upstream_status: 200,
                response_headers_at: started_at + Duration::from_millis(50),
                output_messages: 2,
                output_tokens: 0,
                output_tokens_explicit: false,
                output_tokens_from_chunk_usage: false,
            },
            first_output_at,
            first_token_at: None,
        };

        let before_token_observation = Instant::now();
        observer.observe_output_tokens(7);

        let (_, observed_first_output_at, observed_first_token_at) = observer.response_snapshot();
        assert_eq!(observed_first_output_at, Some(first_output_at));
        let observed_first_token_at = observed_first_token_at.unwrap();
        assert!(observed_first_token_at >= before_token_observation);
        assert!(observed_first_token_at > first_output_at);
    }

    #[test]
    #[should_panic(expected = "invalid finish transition")]
    fn finish_without_output_panics() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-2".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "13".parse().unwrap());
        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let mut observer = RequestObserver::new(&headers, PylonRuntimeState::default()).unwrap();
        observer.on_upstream_response_headers(&response_headers, 200);
        observer.finish();
    }

    #[test]
    #[should_panic(expected = "invalid fail transition")]
    fn fail_after_complete_panics() {
        let mut observer = make_test_observer();
        observer.on_upstream_response_headers(&HeaderMap::new(), 200);
        observer.observe_output_message();
        observer.finish();
        observer.fail();
    }

    #[test]
    fn response_headers_after_terminal_state_panics() {
        fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
            if let Some(message) = panic.downcast_ref::<String>() {
                return message.clone();
            }
            if let Some(message) = panic.downcast_ref::<&'static str>() {
                return (*message).to_string();
            }
            panic!("unexpected non-string panic payload");
        }

        fn assert_terminal_response_header_panic(
            terminalize: impl FnOnce(&mut RequestObserver),
            expected_state: &str,
        ) {
            let mut observer = make_test_observer();
            terminalize(&mut observer);

            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                observer.on_upstream_response_headers(&HeaderMap::new(), 200);
            }))
            .expect_err("response headers after terminal state should panic");

            let message = panic_message(panic);
            assert!(message.contains("invalid response-header transition"));
            assert!(message.contains("request_id=req-inv"));
            assert!(message.contains(expected_state));
        }

        assert_terminal_response_header_panic(
            |observer| {
                observer.on_upstream_response_headers(&HeaderMap::new(), 200);
                observer.observe_output_message();
                observer.finish();
            },
            "state=Complete",
        );
        assert_terminal_response_header_panic(|observer| observer.fail(), "state=Failed");
        assert_terminal_response_header_panic(|observer| observer.cancel(), "state=Cancelled");
    }

    #[tokio::test]
    async fn failed_response_stays_failed() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-3".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "14".parse().unwrap());
        let mut response_headers = HeaderMap::new();
        response_headers.insert(
            reqwest::header::CONTENT_TYPE,
            "text/event-stream".parse().unwrap(),
        );

        let mut observer = RequestObserver::new(&headers, PylonRuntimeState::default()).unwrap();
        observer.on_upstream_response_headers(&response_headers, 503);
        observer.finish();

        let (response, _, _) = observer.response_snapshot();
        let response = response.unwrap();
        assert_eq!(response.output_messages, 0);
        assert_eq!(
            observer.state.observation_state(),
            RequestObservationState::Failed
        );
    }

    #[test]
    fn missing_request_id_is_rejected() {
        let headers = HeaderMap::new();
        let result = RequestObserver::new(&headers, PylonRuntimeState::default());
        assert!(matches!(
            result,
            Err(MissingRequiredHeaderError {
                header_name: HEADER_REQUEST_ID,
                kind: RequiredHeaderErrorKind::Missing,
            })
        ));
    }

    #[test]
    fn missing_model_is_rejected() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-4".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "12".parse().unwrap());
        let result = RequestObserver::new(&headers, PylonRuntimeState::default());
        assert!(matches!(
            result,
            Err(MissingRequiredHeaderError {
                header_name: HEADER_MODEL,
                kind: RequiredHeaderErrorKind::Missing,
            })
        ));
    }

    #[test]
    fn missing_input_tokens_is_rejected() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-5".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        let result = RequestObserver::new(&headers, PylonRuntimeState::default());
        assert!(matches!(
            result,
            Err(MissingRequiredHeaderError {
                header_name: HEADER_INPUT_TOKENS,
                kind: RequiredHeaderErrorKind::Missing,
            })
        ));
    }

    fn make_test_observer() -> RequestObserver {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_ID, "req-inv".parse().unwrap());
        headers.insert(HEADER_ROUTING_KEY, "rk-1".parse().unwrap());
        headers.insert(HEADER_MODEL, "model-a".parse().unwrap());
        headers.insert(HEADER_INPUT_TOKENS, "10".parse().unwrap());
        RequestObserver::new(&headers, PylonRuntimeState::default()).unwrap()
    }

    #[test]
    fn is_terminal_reports_correctly() {
        let mut observer = make_test_observer();
        assert!(!observer.is_terminal());
        observer.fail();
        assert!(observer.is_terminal());
    }
}
