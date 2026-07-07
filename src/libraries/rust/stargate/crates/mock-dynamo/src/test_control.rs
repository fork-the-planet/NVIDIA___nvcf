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
use tokio::sync::{Mutex, Notify};

use crate::AppState;

const CALIBRATION_REQUEST_ID_PREFIX: &str = "calibration-";
const CANARY_REQUEST_ID_PREFIX: &str = "canary-";

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
    PylonGenerated,
    ApiGateway,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct ModelTestControl {
    pub(crate) chat_failure: bool,
    pub(crate) bringup_blocked: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ModelTestControlUpdate {
    pub(crate) chat_failure: Option<bool>,
    pub(crate) bringup_blocked: Option<bool>,
}

type TestCounterKey = (TestEndpoint, String, TestRequestClass);

#[derive(Debug, Default)]
struct TestControlInner {
    models: BTreeMap<String, ModelTestControl>,
    counters: BTreeMap<TestCounterKey, u64>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TestControlState {
    inner: Arc<Mutex<TestControlInner>>,
    bringup_released: Arc<Notify>,
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
    pub(crate) async fn update_model(&self, model: &str, update: ModelTestControlUpdate) {
        let released = {
            let mut inner = self.inner.lock().await;
            let control = inner.models.entry(model.to_string()).or_default();
            if let Some(chat_failure) = update.chat_failure {
                control.chat_failure = chat_failure;
            }
            if let Some(bringup_blocked) = update.bringup_blocked {
                control.bringup_blocked = bringup_blocked;
            }
            !control.bringup_blocked
        };
        if released {
            self.bringup_released.notify_waiters();
        }
    }

    pub(crate) async fn chat_failure_enabled(&self, model: &str) -> bool {
        self.inner
            .lock()
            .await
            .models
            .get(model)
            .is_some_and(|control| control.chat_failure)
    }

    pub(crate) async fn wait_for_bringup_release(&self, model: &str) {
        loop {
            let released = self.bringup_released.notified();
            let blocked = self
                .inner
                .lock()
                .await
                .models
                .get(model)
                .is_some_and(|control| control.bringup_blocked);
            if !blocked {
                return;
            }
            released.await;
        }
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
            .entry((endpoint, model.to_string(), request_class))
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
                .map(
                    |((endpoint, model, request_class), count)| TestCounterSnapshot {
                        endpoint: *endpoint,
                        model: model.clone(),
                        request_class: *request_class,
                        count: *count,
                    },
                )
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

pub(crate) async fn update_model_test_control(
    State(state): State<AppState>,
    Path(model): Path<String>,
    Json(update): Json<ModelTestControlUpdate>,
) -> Json<TestControlSnapshot> {
    state.test_control.update_model(&model, update).await;
    Json(state.test_control.snapshot().await)
}

pub(crate) async fn test_control_snapshot(
    State(state): State<AppState>,
) -> Json<TestControlSnapshot> {
    Json(state.test_control.snapshot().await)
}

pub(crate) fn request_class(headers: &HeaderMap) -> TestRequestClass {
    match headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
    {
        Some(request_id)
            if request_id.starts_with(CALIBRATION_REQUEST_ID_PREFIX)
                || request_id.starts_with(CANARY_REQUEST_ID_PREFIX) =>
        {
            TestRequestClass::PylonGenerated
        }
        _ => TestRequestClass::ApiGateway,
    }
}
