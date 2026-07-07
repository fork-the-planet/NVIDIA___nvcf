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

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::StatusCode;
use serde::Deserialize;
use stargate_protocol::tunnel_contract::{HEADER_INPUT_TOKENS, HEADER_MODEL, HEADER_REQUEST_ID};
use uuid::Uuid;

static PYLON_GENERATED_REQUEST_SCOPE: LazyLock<Uuid> = LazyLock::new(Uuid::new_v4);
static PYLON_GENERATED_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) async fn check_upstream_health(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    timeout: Duration,
) -> bool {
    let health_url = format!("{}/health", upstream_http_base_url.trim_end_matches('/'));
    matches!(
        http_client.get(health_url).timeout(timeout).send().await,
        Ok(response) if response.status().is_success()
    )
}

pub(super) async fn send_canary_request(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    timeout: Duration,
    canary_max_generation_threshold: u32,
) -> Result<(), BringupError> {
    let request = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "1+1="}],
        "max_tokens": canary_max_generation_threshold,
        "seed": 33,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": false,
    });

    let completion = send_completion_request(
        http_client,
        upstream_http_base_url,
        timeout,
        request,
        "canary",
    )
    .await?;
    if completion.usage.completion_tokens == canary_max_generation_threshold {
        return Err(BringupError::RunawayGeneration {
            tokens: completion.usage.completion_tokens,
        });
    }
    Ok(())
}

pub(super) async fn send_completion_request(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    timeout: Duration,
    request: serde_json::Value,
    request_id_prefix: &str,
) -> Result<ChatCompletionResponse, BringupError> {
    let request_id = next_pylon_generated_request_id(request_id_prefix);
    let response = http_client
        .post(format!(
            "{}/v1/chat/completions",
            upstream_http_base_url.trim_end_matches('/')
        ))
        .timeout(timeout)
        .header(HEADER_REQUEST_ID, request_id)
        .header(HEADER_MODEL, request["model"].as_str().unwrap_or_default())
        .header(
            HEADER_INPUT_TOKENS,
            request
                .pointer("/messages/0/content")
                .and_then(serde_json::Value::as_str)
                .map(|text| text.len().to_string())
                .unwrap_or_else(|| "1".to_string()),
        )
        .json(&request)
        .send()
        .await?;

    let status = response.status();
    let body = response.bytes().await?;
    if status.is_success() {
        serde_json::from_slice(&body)
            .map_err(|error| BringupError::InvalidResponse(error.to_string()))
    } else {
        let message = extract_error_message(&body);
        if is_prompt_too_long(status, &message) {
            Err(BringupError::PromptTooLong)
        } else {
            Err(BringupError::Api {
                status,
                message: message.unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned()),
            })
        }
    }
}

fn next_pylon_generated_request_id(prefix: &str) -> String {
    let counter = PYLON_GENERATED_REQUEST_COUNTER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |counter| {
            counter.checked_add(1)
        })
        .expect("pylon-generated request id counter exhausted");
    format!("{prefix}-{}-{counter}", *PYLON_GENERATED_REQUEST_SCOPE)
}

fn extract_error_message(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<ErrorResponse>(body)
        .ok()
        .map(|error| error.error.message)
}

pub(super) fn is_prompt_too_long(status: StatusCode, message: &Option<String>) -> bool {
    status.is_client_error()
        && message.as_ref().is_some_and(|message| {
            let message = message.to_ascii_lowercase();
            ["prompt too long", "context length", "maximum context"]
                .iter()
                .any(|needle| message.contains(needle))
        })
}

#[derive(Debug, thiserror::Error)]
pub enum BringupError {
    #[error("invalid calibration configuration: {0}")]
    InvalidCalibrationConfig(&'static str),
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("upstream health check failed during pylon startup")]
    UnhealthyUpstream,
    #[error("upstream rejected request ({status}): {message}")]
    Api { status: StatusCode, message: String },
    #[error("calibration prompt too long")]
    PromptTooLong,
    #[error("runaway generation detected at completion_tokens={tokens}")]
    RunawayGeneration { tokens: u32 },
    #[error("invalid completion response: {0}")]
    InvalidResponse(String),
    #[error("calibration produced only {valid_samples} valid samples")]
    InsufficientCalibrationSamples { valid_samples: usize },
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatCompletionResponse {
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct Usage {
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    message: String,
}
