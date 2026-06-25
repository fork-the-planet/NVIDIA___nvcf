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

use axum::http::HeaderMap;
use tracing::{Span, field};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::routing_state::{RoutedClusterSnapshot, RoutedInferenceServerSnapshot};
use crate::telemetry::parent_context_from_headers;

pub(super) fn proxy_openai_request_span(headers: &HeaderMap) -> Span {
    let span = tracing::info_span!(
        "proxy_openai_request",
        request.endpoint = field::Empty,
        request.routing_key = field::Empty,
        request.model_id = field::Empty,
        request.path = field::Empty,
        request.input_tokens = field::Empty,
        request.priority = field::Empty,
        request.max_wait_ms = field::Empty,
        request.slo_ms = field::Empty,
        request.cache_affinity_key_present = field::Empty,
        routing.requested_algorithm = field::Empty,
        routing.invalid_requested_algorithm = field::Empty,
        selected_cluster.id = field::Empty,
        selected_inst.id = field::Empty,
        selected_inst.output_tps = field::Empty,
        selected_inst.last_mean_input_tps = field::Empty,
        selected_inst.max_output_tps = field::Empty,
        selected_inst.queue_size = field::Empty,
        selected_inst.queued_input_size = field::Empty,
        selected_inst.num_running_queries = field::Empty,
        selected_inst.max_engine_concurrency = field::Empty,
        selected_inst.total_query_input_size = field::Empty,
        selected_inst.kv_cache_capacity_tokens = field::Empty,
        selected_inst.kv_cache_used_tokens = field::Empty,
        selected_inst.kv_cache_free_tokens = field::Empty,
        selected_inst.rtt_ms = field::Empty,
        selected_inst.snapshot_age_ms = field::Empty,
        routing.algorithm = field::Empty,
        routing.num_candidates = field::Empty,
        routing.rank_depth = field::Empty,
        routing.selected_after_kv_free_tokens_skip = field::Empty,
        routing.retry_attempts = field::Empty,
        routing.admission_rejection_reason = field::Empty,
        proxy.upstream_status = field::Empty,
        proxy.time_to_first_byte_ms = field::Empty,
        proxy.attempt = field::Empty,
        proxy.connect_attempts = field::Empty,
        proxy.request_retries = field::Empty,
        proxy.failed_backends = field::Empty,
        proxy.queue.expected_ms = field::Empty,
        proxy.retry_reason = field::Empty,
        proxy.replay_body_bytes = field::Empty,
    );
    let _ = span.set_parent(parent_context_from_headers(headers));
    span
}

pub(super) struct RequestTraceFields<'a> {
    pub(super) routing_key: Option<&'a str>,
    pub(super) model_id: &'a str,
    pub(super) request_path: &'a str,
    pub(super) input_tokens: u64,
    pub(super) priority: u32,
    pub(super) max_wait_ms: Option<u64>,
    pub(super) request_slo_ms: Option<u64>,
    pub(super) cache_affinity_key_present: bool,
}

pub(super) fn record_request_to_span(span: &Span, fields: RequestTraceFields<'_>) {
    span.record("request.routing_key", fields.routing_key.unwrap_or(""));
    span.record("request.model_id", fields.model_id);
    span.record("request.path", fields.request_path);
    span.record("request.input_tokens", fields.input_tokens);
    span.record("request.priority", fields.priority);
    span.record("request.max_wait_ms", fields.max_wait_ms.unwrap_or(0));
    span.record("request.slo_ms", fields.request_slo_ms.unwrap_or(0));
    span.record(
        "request.cache_affinity_key_present",
        fields.cache_affinity_key_present,
    );
}

pub(super) struct RoutingTraceFields<'a> {
    pub(super) routing_algorithm: &'a str,
    pub(super) requested_algorithm: Option<&'a str>,
    pub(super) num_candidates: usize,
    pub(super) rank_depth: usize,
    pub(super) selected_after_kv_free_tokens_skip: bool,
    pub(super) cluster: &'a RoutedClusterSnapshot,
    pub(super) chosen: &'a RoutedInferenceServerSnapshot,
}

pub(super) fn record_routing_to_span(span: &Span, routing: RoutingTraceFields<'_>) {
    let fields = SelectedInstanceTraceFields::from_route(routing.cluster, routing.chosen);
    span.record("routing.algorithm", routing.routing_algorithm);
    span.record(
        "routing.requested_algorithm",
        routing.requested_algorithm.unwrap_or(""),
    );
    span.record("routing.num_candidates", routing.num_candidates);
    span.record("routing.rank_depth", routing.rank_depth as i64);
    span.record(
        "routing.selected_after_kv_free_tokens_skip",
        routing.selected_after_kv_free_tokens_skip,
    );
    span.record("selected_cluster.id", &routing.cluster.cluster_id);
    span.record("selected_inst.id", &fields.inference_server_id);
    span.record("selected_inst.output_tps", fields.output_tps);
    span.record(
        "selected_inst.last_mean_input_tps",
        fields.last_mean_input_tps,
    );
    span.record("selected_inst.max_output_tps", fields.max_output_tps);
    span.record("selected_inst.queue_size", fields.queue_size);
    span.record("selected_inst.queued_input_size", fields.queued_input_size);
    span.record(
        "selected_inst.num_running_queries",
        fields.num_running_queries,
    );
    span.record(
        "selected_inst.max_engine_concurrency",
        fields.max_engine_concurrency,
    );
    span.record(
        "selected_inst.total_query_input_size",
        fields.total_query_input_size,
    );
    span.record(
        "selected_inst.kv_cache_capacity_tokens",
        fields.kv_cache_capacity_tokens,
    );
    span.record(
        "selected_inst.kv_cache_used_tokens",
        fields.kv_cache_used_tokens,
    );
    span.record(
        "selected_inst.kv_cache_free_tokens",
        fields.kv_cache_free_tokens,
    );
    span.record("selected_inst.rtt_ms", fields.rtt_ms);
    span.record("selected_inst.snapshot_age_ms", fields.snapshot_age_ms);
}

#[derive(Debug, Clone)]
struct SelectedInstanceTraceFields {
    inference_server_id: String,
    output_tps: f64,
    last_mean_input_tps: f64,
    max_output_tps: f64,
    queue_size: u64,
    queued_input_size: u64,
    kv_cache_capacity_tokens: u64,
    kv_cache_used_tokens: u64,
    kv_cache_free_tokens: u64,
    num_running_queries: u64,
    max_engine_concurrency: u64,
    total_query_input_size: u64,
    rtt_ms: f64,
    snapshot_age_ms: f64,
}

impl SelectedInstanceTraceFields {
    fn from_route(
        cluster: &RoutedClusterSnapshot,
        backend: &RoutedInferenceServerSnapshot,
    ) -> Self {
        Self {
            inference_server_id: backend.inference_server_id.clone(),
            output_tps: cluster.stats.output_tps,
            last_mean_input_tps: cluster.stats.last_mean_input_tps,
            max_output_tps: cluster.stats.max_output_tps,
            queue_size: cluster.stats.queue_size,
            queued_input_size: cluster.stats.queued_input_size,
            kv_cache_capacity_tokens: cluster.stats.kv_cache_capacity_tokens,
            kv_cache_used_tokens: cluster.stats.kv_cache_used_tokens,
            kv_cache_free_tokens: cluster.stats.kv_cache_free_tokens,
            num_running_queries: cluster.stats.num_running_queries,
            max_engine_concurrency: cluster.stats.max_engine_concurrency,
            total_query_input_size: cluster.stats.total_query_input_size,
            rtt_ms: cluster.rtt.as_secs_f64() * 1000.0,
            snapshot_age_ms: cluster.snapshot_updated_at.elapsed().as_secs_f64() * 1000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use axum::http::{HeaderMap, HeaderName, HeaderValue};
    use opentelemetry::global;
    use opentelemetry::trace::TraceContextExt;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use stargate_proto::pb::{InferenceServerStatus, ModelStats};

    use crate::routing_state::{
        RegistrationIdentity, RoutedClusterSnapshot, RoutedInferenceServerSnapshot,
        test_registration_generation,
    };
    use crate::telemetry::parent_context_from_headers;

    use super::*;

    #[test]
    fn selected_instance_trace_fields_exclude_url_and_include_pulsar_metrics() {
        let registration = test_registration_generation(RegistrationIdentity {
            inference_server_id: "inst-a".to_string(),
            cluster_id: "cluster-a".to_string(),
            inference_server_url: "quic://127.0.0.1:5000".to_string(),
            routing_key: None,
            reverse_tunnel: false,
            coordinated_calibration: false,
        });
        let snapshot = RoutedInferenceServerSnapshot {
            registration,
            cluster_id: "cluster-a".to_string(),
            inference_server_id: "inst-a".to_string(),
            inference_server_url: "quic://127.0.0.1:5000".to_string(),
            stats: ModelStats {
                output_tps: 20.0,
                last_mean_input_tps: 30.0,
                max_output_tps: 40.0,
                queue_size: 5,
                queued_input_size: 6,
                kv_cache_capacity_tokens: 4096,
                kv_cache_used_tokens: 1024,
                kv_cache_free_tokens: 3072,
                num_running_queries: 3,
                max_engine_concurrency: 8,
                total_query_input_size: 512,
                queue_time_estimate_ms_by_priority: std::collections::HashMap::new(),
                ..ModelStats::default()
            },
            rtt: Duration::from_millis(12),
            snapshot_updated_at: Instant::now(),
            status: InferenceServerStatus::Active,
            reverse_tunnel: false,
        };
        let cluster = RoutedClusterSnapshot {
            cluster_id: "cluster-a".to_string(),
            stats: ModelStats {
                output_tps: 20.0,
                last_mean_input_tps: 30.0,
                max_output_tps: 40.0,
                queue_size: 5,
                queued_input_size: 6,
                kv_cache_capacity_tokens: 4096,
                kv_cache_used_tokens: 1024,
                kv_cache_free_tokens: 3072,
                ..ModelStats::default()
            },
            rtt: Duration::from_millis(12),
            snapshot_updated_at: Instant::now(),
            status: InferenceServerStatus::Active,
            active_backend_count: 1,
        };

        let fields = SelectedInstanceTraceFields::from_route(&cluster, &snapshot);
        assert_eq!(fields.inference_server_id, "inst-a");
        assert_eq!(fields.kv_cache_capacity_tokens, 4096);
        assert_eq!(fields.kv_cache_used_tokens, 1024);
        assert_eq!(fields.kv_cache_free_tokens, 3072);
        assert_eq!(fields.rtt_ms, 12.0);
        assert!(fields.snapshot_age_ms >= 0.0);
    }

    #[test]
    fn traceparent_header_extracts_remote_parent_context() {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("traceparent"),
            HeaderValue::from_static("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
        );

        let parent_context = parent_context_from_headers(&headers);
        let span_context = parent_context.span().span_context().clone();

        assert!(span_context.is_valid());
        assert!(span_context.is_remote());
        assert_eq!(
            span_context.trace_id().to_string(),
            "4bf92f3577b34da6a3ce929d0e0e4736"
        );
        assert_eq!(span_context.span_id().to_string(), "00f067aa0ba902b7");
    }
}
