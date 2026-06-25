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

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use stargate_protocol::tunnel_contract::HEADER_INFERENCE_SERVER_ID as HEADER_CHOSEN_INFERENCE_SERVER_ID;
use tracing::{Instrument, Span, field, info, warn};

use crate::routing_state::{
    RoutedInferenceServerSnapshot, RoutingReservation, RoutingTargetKey, SelectedRoutedCluster,
};

use super::ProxyAppState;
use super::request::ProxyRequestInputs;
use super::retry::{
    FinalRetryDisposition, ReplayableRequestBody, RetryDecision, UpstreamRetry,
    decide_proxy_error_retry, decide_upstream_response_retry, retry_budget_has_remaining,
    should_release_queue_mismatch_reservation,
};
use super::upstream::{
    UpstreamStreamingResponse, copy_forwardable_headers, headers_for_upstream_attempt,
    proxy_attempt_result, proxy_via_quic_streaming,
};

const HEADER_CHOSEN_INFERENCE_SERVER_URL: &str = "x-inference-server-url";
const HEADER_CHOSEN_CLUSTER_ID: &str = "x-stargate-cluster-id";

#[derive(Default)]
pub(super) struct ProxyAttemptCounters {
    pub(super) attempt: u32,
    connect_retries: u32,
    request_retries: u32,
}

pub(super) struct ProxyAttemptContext<'a> {
    pub(super) app: &'a ProxyAppState,
    pub(super) target: &'a RoutingTargetKey,
    pub(super) request_inputs: &'a ProxyRequestInputs,
    pub(super) endpoint_name: &'static str,
    pub(super) method: &'a Method,
    pub(super) path_and_query: &'a str,
    pub(super) forwarded_headers: &'a HeaderMap,
    pub(super) retry_deadline: Option<Instant>,
    pub(super) request_start: Instant,
}

impl ProxyAttemptContext<'_> {
    fn routing_key(&self) -> Option<&str> {
        self.target.routing_key.as_deref()
    }

    fn model_id(&self) -> &str {
        self.target.model_id.as_str()
    }
}

pub(super) struct ProxyAttemptRoute<'a> {
    pub(super) cluster: &'a SelectedRoutedCluster,
    pub(super) chosen: &'a Arc<RoutedInferenceServerSnapshot>,
    pub(super) routing_algorithm: &'a str,
    pub(super) requested_algorithm: Option<&'a str>,
    pub(super) expected_queue_ms: Option<u64>,
}

pub(super) enum ProxyAttemptOutcome {
    ReturnFinal(Response<Body>),
    ProxyError(StatusCode),
    RetrySameBackend {
        chosen: Arc<RoutedInferenceServerSnapshot>,
    },
    RetryAlternateBackend {
        inference_server_id: String,
    },
    RetryAlternateCluster {
        cluster_id: String,
    },
}

pub(super) async fn run_proxy_attempt(
    context: ProxyAttemptContext<'_>,
    route: ProxyAttemptRoute<'_>,
    counters: &mut ProxyAttemptCounters,
    replay_body: &mut ReplayableRequestBody,
    failed_backend_count: usize,
) -> ProxyAttemptOutcome {
    counters.attempt += 1;
    Span::current().record("proxy.attempt", counters.attempt as i64);
    Span::current().record("proxy.connect_attempts", counters.connect_retries as i64);
    Span::current().record("proxy.request_retries", counters.request_retries as i64);
    Span::current().record("proxy.failed_backends", failed_backend_count as i64);

    if let Some(outcome) = ensure_hot_path_connection(&context, route.chosen, counters).await {
        return outcome;
    }

    let reservation = route.cluster.reserve_backend(
        &route.chosen.registration,
        context.request_inputs.input_tokens,
        context.request_inputs.priority,
    );
    record_proxy_attempt_start(&context, &route, counters);

    let upstream_span = proxy_upstream_attempt_span(&context, &route, counters.attempt);
    Span::current().record(
        "proxy.queue.expected_ms",
        route
            .expected_queue_ms
            .map(|value| value as i64)
            .unwrap_or_default(),
    );
    Span::current().record(
        "proxy.queue.expected_present",
        route.expected_queue_ms.is_some(),
    );
    let attempt_headers = headers_for_upstream_attempt(
        context.forwarded_headers,
        &upstream_span,
        route.expected_queue_ms,
    );
    let upstream_start = Instant::now();
    let upstream = proxy_via_quic_streaming(
        context.app,
        &route.chosen.registration,
        context.method.clone(),
        context.path_and_query,
        attempt_headers,
        || replay_body.body_for_attempt(),
    )
    .instrument(upstream_span.clone())
    .await;
    record_proxy_attempt_result(
        &context,
        &route,
        replay_body,
        &upstream,
        &upstream_span,
        upstream_start,
    );

    match upstream {
        Ok(upstream) => {
            handle_upstream_response_attempt(
                &context,
                &route,
                counters,
                replay_body,
                reservation,
                upstream,
            )
            .await
        }
        Err(status) => {
            handle_proxy_error_attempt(&context, &route, counters, replay_body, status).await
        }
    }
}

async fn ensure_hot_path_connection(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    counters: &mut ProxyAttemptCounters,
) -> Option<ProxyAttemptOutcome> {
    if chosen.reverse_tunnel
        || counters.connect_retries >= context.app.retry.max_connect_retries
        || context
            .app
            .quic_proxy
            .has_healthy_connection(&chosen.registration)
    {
        return None;
    }

    counters.connect_retries += 1;
    match reconnect_direct(context, chosen, "stale_connection").await {
        Ok(()) => None,
        Err(error) => {
            warn!(
                inference_server_id = %chosen.inference_server_id,
                error = %error,
                connect_retries = counters.connect_retries,
                "failed to reconnect stale QUIC upstream"
            );
            Some(ProxyAttemptOutcome::RetryAlternateBackend {
                inference_server_id: chosen.inference_server_id.clone(),
            })
        }
    }
}

fn record_proxy_attempt_start(
    context: &ProxyAttemptContext<'_>,
    route: &ProxyAttemptRoute<'_>,
    counters: &ProxyAttemptCounters,
) {
    info!(
        routing_key = ?context.target.routing_key,
        model_id = %context.model_id(),
        input_tokens = context.request_inputs.input_tokens,
        requested_algorithm = route.requested_algorithm.unwrap_or(""),
        routing_algorithm = %route.routing_algorithm,
        inference_server_id = %route.chosen.inference_server_id,
        inference_server_url = %route.chosen.inference_server_url,
        connect_retries = counters.connect_retries,
        request_retries = counters.request_retries,
        "proxying request"
    );
}

fn proxy_upstream_attempt_span(
    context: &ProxyAttemptContext<'_>,
    route: &ProxyAttemptRoute<'_>,
    attempt: u32,
) -> Span {
    tracing::info_span!(
        "proxy_upstream_http_request",
        request.endpoint = context.endpoint_name,
        http.method = %context.method,
        http.path = %context.path_and_query,
        proxy.attempt = attempt as i64,
        selected_cluster.id = %route.chosen.cluster_id,
        selected_inst.id = %route.chosen.inference_server_id,
        routing.algorithm = %route.routing_algorithm,
        proxy.queue.expected_ms = route.expected_queue_ms.map(|value| value as i64).unwrap_or_default(),
        proxy.queue.expected_present = route.expected_queue_ms.is_some(),
        proxy.upstream_status = field::Empty,
        proxy.error = field::Empty,
        proxy.time_to_first_byte_ms = field::Empty,
    )
}

fn record_proxy_attempt_result(
    context: &ProxyAttemptContext<'_>,
    route: &ProxyAttemptRoute<'_>,
    replay_body: &ReplayableRequestBody,
    upstream: &Result<UpstreamStreamingResponse, StatusCode>,
    upstream_span: &Span,
    upstream_start: Instant,
) {
    let replay_body_bytes = replay_body.buffered_len();
    Span::current().record("proxy.replay_body_bytes", replay_body_bytes as i64);
    context
        .app
        .metrics
        .proxy_replay_buffer_bytes(context.model_id())
        .observe(replay_body_bytes as f64);

    record_upstream_result_to_span(upstream_span, upstream, upstream_start.elapsed(), true);
    let attempt_result = proxy_attempt_result(upstream);
    context
        .app
        .metrics
        .proxy_attempts_total(
            context.routing_key(),
            context.model_id(),
            &route.chosen.inference_server_id,
            &attempt_result,
        )
        .inc();
    let ttfb = context.request_start.elapsed();
    record_upstream_result_to_span(&Span::current(), upstream, ttfb, false);
    if upstream.is_ok() {
        context
            .app
            .metrics
            .proxy_duration_seconds(
                context.routing_key(),
                context.model_id(),
                &route.chosen.inference_server_id,
            )
            .observe(ttfb.as_secs_f64());
    }
}

async fn handle_upstream_response_attempt(
    context: &ProxyAttemptContext<'_>,
    route: &ProxyAttemptRoute<'_>,
    counters: &mut ProxyAttemptCounters,
    replay_body: &mut ReplayableRequestBody,
    reservation: Option<RoutingReservation>,
    upstream: UpstreamStreamingResponse,
) -> ProxyAttemptOutcome {
    if should_release_queue_mismatch_reservation(upstream.status, &upstream.headers)
        && let Some(reservation) = reservation
    {
        reservation.release();
    }
    match decide_upstream_response_retry(
        upstream.status,
        &upstream.headers,
        &context.app.retry,
        retry_budget_has_remaining(context.retry_deadline),
        counters.request_retries,
        replay_body.replay_readiness(),
    ) {
        RetryDecision::Final(disposition) => {
            finalize_upstream_response(context, route.chosen, disposition, upstream)
        }
        RetryDecision::Retry(UpstreamRetry::AlternateBackend(retry_reason)) => {
            record_request_retry(context, counters, &retry_reason);
            warn!(
                inference_server_id = %route.chosen.inference_server_id,
                cluster_id = %route.chosen.cluster_id,
                status = %upstream.status,
                request_retries = counters.request_retries,
                retry_reason = %retry_reason,
                "retrying request on a sibling backend after local queue mismatch"
            );
            ProxyAttemptOutcome::RetryAlternateBackend {
                inference_server_id: route.chosen.inference_server_id.clone(),
            }
        }
        RetryDecision::Retry(UpstreamRetry::AlternateCluster(retry_reason)) => {
            record_request_retry(context, counters, &retry_reason);
            warn!(
                inference_server_id = %route.chosen.inference_server_id,
                cluster_id = %route.chosen.cluster_id,
                status = %upstream.status,
                request_retries = counters.request_retries,
                retry_reason = %retry_reason,
                "retrying request after retryable upstream response"
            );
            ProxyAttemptOutcome::RetryAlternateCluster {
                cluster_id: route.chosen.cluster_id.clone(),
            }
        }
    }
}

fn finalize_upstream_response(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    disposition: FinalRetryDisposition,
    upstream: UpstreamStreamingResponse,
) -> ProxyAttemptOutcome {
    if let Some(status) =
        record_final_retry_disposition(context, chosen, upstream.status, disposition, "response")
    {
        return final_proxy_error_outcome(context, chosen, status);
    }
    final_upstream_response_outcome(context, chosen, upstream)
}

async fn handle_proxy_error_attempt(
    context: &ProxyAttemptContext<'_>,
    route: &ProxyAttemptRoute<'_>,
    counters: &mut ProxyAttemptCounters,
    replay_body: &mut ReplayableRequestBody,
    status: StatusCode,
) -> ProxyAttemptOutcome {
    match decide_proxy_error_retry(
        status,
        &context.app.retry,
        retry_budget_has_remaining(context.retry_deadline),
        counters.connect_retries,
        replay_body.replay_readiness(),
    ) {
        RetryDecision::Final(disposition) => {
            finalize_proxy_error(context, route.chosen, disposition, status)
        }
        RetryDecision::Retry(()) => {
            retry_connection_or_failover(context, route.chosen, counters).await
        }
    }
}

fn finalize_proxy_error(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    disposition: FinalRetryDisposition,
    status: StatusCode,
) -> ProxyAttemptOutcome {
    let status =
        record_final_retry_disposition(context, chosen, status, disposition, "proxy error")
            .unwrap_or(status);
    final_proxy_error_outcome(context, chosen, status)
}

fn record_final_retry_disposition(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    status: StatusCode,
    disposition: FinalRetryDisposition,
    source: &'static str,
) -> Option<StatusCode> {
    match disposition {
        FinalRetryDisposition::PassThrough => {}
        FinalRetryDisposition::Exhausted(retry_reason) => {
            record_retry_exhausted(context, &retry_reason);
        }
        FinalRetryDisposition::ReplayIncomplete(retry_reason) => {
            Span::current().record("proxy.retry_reason", retry_reason.as_str());
            warn!(
                inference_server_id = %chosen.inference_server_id,
                status = %status,
                retry_reason = %retry_reason,
                source,
                "not retrying because request body replay buffer is incomplete"
            );
        }
        FinalRetryDisposition::PayloadTooLarge(retry_reason) => {
            if let Some(retry_reason) = retry_reason {
                Span::current().record("proxy.retry_reason", retry_reason.as_str());
            }
            return Some(StatusCode::PAYLOAD_TOO_LARGE);
        }
    }
    None
}

async fn retry_connection_or_failover(
    context: &ProxyAttemptContext<'_>,
    chosen: &Arc<RoutedInferenceServerSnapshot>,
    counters: &mut ProxyAttemptCounters,
) -> ProxyAttemptOutcome {
    counters.connect_retries += 1;
    if !chosen.reverse_tunnel {
        match reconnect_direct(context, chosen, "proxy_error").await {
            Ok(()) => {
                warn!(
                    inference_server_id = %chosen.inference_server_id,
                    connect_retries = counters.connect_retries,
                    "reconnected QUIC upstream after proxy failure"
                );
                return ProxyAttemptOutcome::RetrySameBackend {
                    chosen: Arc::clone(chosen),
                };
            }
            Err(error) => {
                warn!(
                    inference_server_id = %chosen.inference_server_id,
                    error = %error,
                    connect_retries = counters.connect_retries,
                    "failed to reconnect QUIC upstream"
                );
            }
        }
    }
    context
        .app
        .metrics
        .proxy_retries_total(
            context.routing_key(),
            context.model_id(),
            "connect_failover",
        )
        .inc();
    Span::current().record("proxy.retry_reason", "connect_failover");
    ProxyAttemptOutcome::RetryAlternateBackend {
        inference_server_id: chosen.inference_server_id.clone(),
    }
}

async fn reconnect_direct(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    eviction_reason: &'static str,
) -> anyhow::Result<()> {
    context
        .app
        .metrics
        .quic_connection_evictions_total(&chosen.inference_server_id, eviction_reason)
        .inc();
    let result = context
        .app
        .quic_proxy
        .connect_direct_registration(&chosen.registration)
        .await;
    context
        .app
        .metrics
        .quic_hot_path_reconnect_total(
            &chosen.inference_server_id,
            if result.is_ok() { "success" } else { "error" },
        )
        .inc();
    if result.is_ok() {
        context
            .app
            .metrics
            .proxy_retries_total(
                context.routing_key(),
                context.model_id(),
                "hot_path_reconnect",
            )
            .inc();
        Span::current().record("proxy.retry_reason", "hot_path_reconnect");
    }
    result
}

fn record_retry_exhausted(context: &ProxyAttemptContext<'_>, retry_reason: &str) {
    context
        .app
        .metrics
        .proxy_retry_exhausted_total(context.routing_key(), context.model_id(), retry_reason)
        .inc();
    Span::current().record("proxy.retry_reason", retry_reason);
}

fn record_request_retry(
    context: &ProxyAttemptContext<'_>,
    counters: &mut ProxyAttemptCounters,
    retry_reason: &str,
) {
    Span::current().record("proxy.retry_reason", retry_reason);
    counters.request_retries += 1;
    context
        .app
        .metrics
        .proxy_retries_total(context.routing_key(), context.model_id(), retry_reason)
        .inc();
}

fn final_upstream_response_outcome(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    upstream: UpstreamStreamingResponse,
) -> ProxyAttemptOutcome {
    let status = upstream.status;
    final_outcome(
        context,
        chosen,
        status,
        build_proxy_response(upstream, chosen),
    )
}

fn final_proxy_error_outcome(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    status: StatusCode,
) -> ProxyAttemptOutcome {
    final_outcome(context, chosen, status, Err(status))
}

fn final_outcome(
    context: &ProxyAttemptContext<'_>,
    chosen: &RoutedInferenceServerSnapshot,
    status: StatusCode,
    result: Result<Response<Body>, StatusCode>,
) -> ProxyAttemptOutcome {
    context
        .app
        .metrics
        .requests_total(
            context.routing_key(),
            context.model_id(),
            &chosen.inference_server_id,
            &status.as_u16().to_string(),
        )
        .inc();
    match result {
        Ok(response) => ProxyAttemptOutcome::ReturnFinal(response),
        Err(status) => ProxyAttemptOutcome::ProxyError(status),
    }
}

fn build_proxy_response(
    upstream: UpstreamStreamingResponse,
    chosen: &RoutedInferenceServerSnapshot,
) -> Result<Response<Body>, StatusCode> {
    let mut response = Response::builder().status(upstream.status);
    {
        let response_headers = response
            .headers_mut()
            .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
        copy_forwardable_headers(&upstream.headers, response_headers);
        response_headers.insert(
            HeaderName::from_static(HEADER_CHOSEN_INFERENCE_SERVER_ID),
            HeaderValue::from_str(&chosen.inference_server_id)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        );
        response_headers.insert(
            HeaderName::from_static(HEADER_CHOSEN_INFERENCE_SERVER_URL),
            HeaderValue::from_str(&chosen.inference_server_url)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        );
        response_headers.insert(
            HeaderName::from_static(HEADER_CHOSEN_CLUSTER_ID),
            HeaderValue::from_str(&chosen.cluster_id)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        );
    }
    response
        .body(upstream.body)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

fn record_upstream_result_to_span(
    span: &Span,
    result: &Result<UpstreamStreamingResponse, StatusCode>,
    ttfb: std::time::Duration,
    error_field: bool,
) {
    span.record("proxy.time_to_first_byte_ms", ttfb.as_secs_f64() * 1000.0);
    let (status, field) = match result {
        Ok(response) => (response.status, "proxy.upstream_status"),
        Err(status) if error_field => (*status, "proxy.error"),
        Err(status) => (*status, "proxy.upstream_status"),
    };
    span.record(field, status.as_u16().to_string());
}
