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

use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use stargate_protocol::tunnel_contract::{
    HEADER_STARGATE_EXPECTED_QUEUE_MS, HEADER_STARGATE_RETRY_AFTER_MS,
    HEADER_STARGATE_RETRY_REASON, HEADER_STARGATE_RETRYABLE,
};
use tracing::{Span, warn};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::routing_state::RegistrationGeneration;
use crate::telemetry::inject_trace_context;

use super::{HEADER_ROUTING_METHOD, HEADER_STARGATE_ERROR_CODE, ProxyAppState};

pub(super) struct UpstreamStreamingResponse {
    pub(super) status: StatusCode,
    pub(super) headers: HeaderMap,
    pub(super) body: Body,
}

pub(super) async fn proxy_via_quic_streaming(
    app: &ProxyAppState,
    registration: &std::sync::Arc<RegistrationGeneration>,
    method: Method,
    path_and_query: &str,
    forwarded_headers: HeaderMap,
    request_body: impl FnOnce() -> Result<Body, StatusCode> + Send,
) -> Result<UpstreamStreamingResponse, StatusCode> {
    let inference_server_id = registration.inference_server_id();
    let streaming_resp = app
        .quic_proxy
        .open_streaming_request(registration, method, path_and_query, forwarded_headers)
        .await
        .map_err(|error| {
            warn!(inference_server_id = %inference_server_id, error = %error, "quic upstream request failed");
            StatusCode::BAD_GATEWAY
        })?
        .send_body_and_recv_response(request_body()?)
        .await
        .map_err(|error| {
            warn!(inference_server_id = %inference_server_id, error = %error, "quic upstream request failed");
            StatusCode::BAD_GATEWAY
        })?;

    let status = streaming_resp.status;
    let headers = streaming_resp.headers;
    let mut body_stream = streaming_resp.body_stream;

    let body = Body::from_stream(async_stream::stream! {
        loop {
            match body_stream.recv_body().await {
                Ok(Some(chunk)) => yield Ok::<_, std::io::Error>(chunk),
                Ok(None) => break,
                Err(e) => {
                    yield Err(std::io::Error::other(e.to_string()));
                    break;
                }
            }
        }
    });

    Ok(UpstreamStreamingResponse {
        status,
        headers,
        body,
    })
}

pub(super) fn prepare_forwarded_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded_headers = HeaderMap::new();
    copy_forwardable_headers(headers, &mut forwarded_headers);
    forwarded_headers
}

pub(super) fn headers_for_upstream_attempt(
    forwarded_headers: &HeaderMap,
    span: &Span,
    expected_queue_ms: Option<u64>,
) -> HeaderMap {
    let mut attempt_headers = forwarded_headers.clone();
    let context = span.context();
    inject_trace_context(&mut attempt_headers, &context);
    if let Some(expected_queue_ms) = expected_queue_ms {
        attempt_headers.insert(
            HeaderName::from_static(HEADER_STARGATE_EXPECTED_QUEUE_MS),
            HeaderValue::from_str(&expected_queue_ms.to_string())
                .expect("decimal queue estimate should be a valid header value"),
        );
    }
    attempt_headers
}

pub(super) fn proxy_attempt_result(
    result: &Result<UpstreamStreamingResponse, StatusCode>,
) -> String {
    match result {
        Ok(response) => format!("upstream_{}", response.status.as_u16()),
        Err(status) => format!("proxy_{}", status.as_u16()),
    }
}

fn should_forward_header(name: &HeaderName) -> bool {
    // `HeaderName` stores normalized lowercase names, so matching the borrowed
    // str avoids allocating a lowercase copy for every proxied header.
    !matches!(
        name.as_str(),
        "connection"
            | "proxy-connection"
            | "keep-alive"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "upgrade"
            | "host"
            | HEADER_ROUTING_METHOD
            | HEADER_STARGATE_RETRYABLE
            | HEADER_STARGATE_RETRY_REASON
            | HEADER_STARGATE_RETRY_AFTER_MS
            | HEADER_STARGATE_EXPECTED_QUEUE_MS
            | HEADER_STARGATE_ERROR_CODE
    )
}

pub(super) fn copy_forwardable_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for (name, value) in from {
        if should_forward_header(name) {
            to.append(name, value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Bytes;
    use stargate_protocol::tunnel_contract::HEADER_MODEL;

    use crate::routing_state::{RegistrationIdentity, test_registration_generation};

    use super::super::retry::ReplayableRequestBody;
    use super::super::test_support::test_proxy_app_state;
    use super::*;

    #[test]
    fn prepare_forwarded_headers_strips_internal_proxy_headers() {
        let mut source = HeaderMap::new();
        source.insert(
            HeaderName::from_static("connection"),
            HeaderValue::from_static("close"),
        );
        source.insert(
            HeaderName::from_static("host"),
            HeaderValue::from_static("example.test"),
        );
        source.insert(
            HeaderName::from_static("x-routing-method"),
            HeaderValue::from_static("random"),
        );
        source.insert(
            HeaderName::from_static(HEADER_STARGATE_ERROR_CODE),
            HeaderValue::from_static("no_eligible_candidates"),
        );
        source.insert(
            HeaderName::from_static(HEADER_MODEL),
            HeaderValue::from_static("gpt"),
        );
        source.insert(
            HeaderName::from_static("x-upstream-header"),
            HeaderValue::from_static("kept"),
        );

        let forwarded = prepare_forwarded_headers(&source);

        assert!(!forwarded.contains_key("connection"));
        assert!(!forwarded.contains_key("host"));
        assert!(!forwarded.contains_key("x-routing-method"));
        assert!(!forwarded.contains_key(HEADER_STARGATE_ERROR_CODE));
        assert_eq!(
            forwarded.get(HEADER_MODEL),
            Some(&HeaderValue::from_static("gpt"))
        );
        assert_eq!(
            forwarded.get("x-upstream-header"),
            Some(&HeaderValue::from_static("kept"))
        );
    }

    #[test]
    fn headers_for_upstream_attempt_preserves_headers_and_adds_queue_estimate() {
        let span = tracing::info_span!("attempt_header_test");
        let mut forwarded_headers = HeaderMap::new();
        forwarded_headers.insert(
            HeaderName::from_static(HEADER_MODEL),
            HeaderValue::from_static("gpt"),
        );

        let attempt_headers = headers_for_upstream_attempt(&forwarded_headers, &span, Some(42));

        assert_eq!(
            attempt_headers.get(HEADER_MODEL),
            Some(&HeaderValue::from_static("gpt"))
        );
        assert_eq!(
            attempt_headers.get(HEADER_STARGATE_EXPECTED_QUEUE_MS),
            Some(&HeaderValue::from_static("42"))
        );
    }

    #[test]
    fn copy_forwardable_headers_strips_internal_retry_headers() {
        let mut upstream = HeaderMap::new();
        upstream.insert(
            HeaderName::from_static(HEADER_STARGATE_ERROR_CODE),
            HeaderValue::from_static("no_eligible_candidates"),
        );
        upstream.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRYABLE),
            HeaderValue::from_static("true"),
        );
        upstream.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRY_REASON),
            HeaderValue::from_static("retryable_proxy_error"),
        );
        upstream.insert(
            HeaderName::from_static(HEADER_STARGATE_RETRY_AFTER_MS),
            HeaderValue::from_static("25"),
        );
        upstream.insert(
            HeaderName::from_static(HEADER_STARGATE_EXPECTED_QUEUE_MS),
            HeaderValue::from_static("123"),
        );
        upstream.insert(
            HeaderName::from_static("x-upstream-header"),
            HeaderValue::from_static("preserved"),
        );

        let mut downstream = HeaderMap::new();
        copy_forwardable_headers(&upstream, &mut downstream);

        assert!(!downstream.contains_key(HEADER_STARGATE_ERROR_CODE));
        assert!(!downstream.contains_key(HEADER_STARGATE_RETRYABLE));
        assert!(!downstream.contains_key(HEADER_STARGATE_RETRY_REASON));
        assert!(!downstream.contains_key(HEADER_STARGATE_RETRY_AFTER_MS));
        assert!(!downstream.contains_key(HEADER_STARGATE_EXPECTED_QUEUE_MS));
        assert_eq!(
            downstream.get("x-upstream-header"),
            Some(&HeaderValue::from_static("preserved"))
        );
    }

    #[tokio::test]
    async fn transport_setup_failure_does_not_consume_first_replay_body() {
        let app = test_proxy_app_state();
        let registration = test_registration_generation(RegistrationIdentity {
            inference_server_id: "missing-connection".to_string(),
            cluster_id: "missing-connection".to_string(),
            inference_server_url: "quic://127.0.0.1:1".to_string(),
            routing_key: None,
            reverse_tunnel: false,
            coordinated_calibration: false,
        });
        let body = Body::from("still-available");
        let mut replay_body = ReplayableRequestBody::new(&HeaderMap::new(), body, 1024).unwrap();

        let result = proxy_via_quic_streaming(
            &app,
            &registration,
            Method::POST,
            "/v1/chat/completions",
            HeaderMap::new(),
            || replay_body.body_for_attempt(),
        )
        .await;

        assert_eq!(result.err(), Some(StatusCode::BAD_GATEWAY));

        let attempt_body = replay_body.body_for_attempt().unwrap();
        let attempt_bytes = axum::body::to_bytes(attempt_body, 1024).await.unwrap();
        assert_eq!(attempt_bytes, Bytes::from_static(b"still-available"));
    }
}
