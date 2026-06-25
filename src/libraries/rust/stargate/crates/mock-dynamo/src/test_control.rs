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

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::AppState;

const BRINGUP_REQUEST_ID_PREFIX: &str = "bringup-";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TestEndpoint {
    ChatCompletions,
    Responses,
    Embeddings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TestRequestClass {
    Bringup,
    NonBringup,
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct ModelTestControl {
    pub(crate) chat_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TestCounterKey {
    endpoint: TestEndpoint,
    model: String,
    request_class: TestRequestClass,
}

#[derive(Debug, Default)]
struct TestControlInner {
    models: BTreeMap<String, ModelTestControl>,
    counters: BTreeMap<TestCounterKey, u64>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TestControlState {
    inner: Arc<Mutex<TestControlInner>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TestCounterSnapshot {
    pub(crate) endpoint: TestEndpoint,
    pub(crate) model: String,
    pub(crate) request_class: TestRequestClass,
    pub(crate) count: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TestControlSnapshot {
    pub(crate) models: BTreeMap<String, ModelTestControl>,
    pub(crate) counters: Vec<TestCounterSnapshot>,
}

impl TestControlState {
    pub(crate) async fn set_chat_failure(&self, model: &str, enabled: bool) {
        self.inner
            .lock()
            .await
            .models
            .entry(model.to_string())
            .or_default()
            .chat_failure = enabled;
    }

    pub(crate) async fn chat_failure_enabled(&self, model: &str) -> bool {
        self.inner
            .lock()
            .await
            .models
            .get(model)
            .is_some_and(|control| control.chat_failure)
    }

    pub(crate) async fn record_request(
        &self,
        endpoint: TestEndpoint,
        model: &str,
        request_class: TestRequestClass,
    ) {
        let mut inner = self.inner.lock().await;
        let count = inner
            .counters
            .entry(TestCounterKey {
                endpoint,
                model: model.to_string(),
                request_class,
            })
            .or_default();
        *count = count.saturating_add(1);
    }

    pub(crate) async fn snapshot(&self) -> TestControlSnapshot {
        let inner = self.inner.lock().await;
        TestControlSnapshot {
            models: inner.models.clone(),
            counters: inner
                .counters
                .iter()
                .map(|(key, count)| TestCounterSnapshot {
                    endpoint: key.endpoint,
                    model: key.model.clone(),
                    request_class: key.request_class,
                    count: *count,
                })
                .collect(),
        }
    }
}

impl TestControlSnapshot {
    #[cfg(test)]
    pub(crate) fn counter(
        &self,
        endpoint: TestEndpoint,
        model: &str,
        request_class: TestRequestClass,
    ) -> u64 {
        self.counters
            .iter()
            .find(|counter| {
                counter.endpoint == endpoint
                    && counter.model == model
                    && counter.request_class == request_class
            })
            .map_or(0, |counter| counter.count)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct UpdateModelTestControl {
    chat_failure: bool,
}

pub(crate) async fn update_model_test_control(
    State(state): State<AppState>,
    Path(model): Path<String>,
    Json(update): Json<UpdateModelTestControl>,
) -> Json<TestControlSnapshot> {
    state
        .test_control
        .set_chat_failure(&model, update.chat_failure)
        .await;
    Json(state.test_control.snapshot().await)
}

pub(crate) async fn test_control_snapshot(
    State(state): State<AppState>,
) -> Json<TestControlSnapshot> {
    Json(state.test_control.snapshot().await)
}

pub(crate) fn request_class(headers: &HeaderMap) -> TestRequestClass {
    if headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|request_id| request_id.starts_with(BRINGUP_REQUEST_ID_PREFIX))
    {
        TestRequestClass::Bringup
    } else {
        TestRequestClass::NonBringup
    }
}
