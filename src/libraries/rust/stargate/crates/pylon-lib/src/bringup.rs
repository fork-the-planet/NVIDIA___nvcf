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
use std::time::Duration;

use crate::stats::PylonMetrics;
#[cfg(test)]
use reqwest::StatusCode;

mod calibration;
mod lifecycle;
mod upstream;

pub(crate) use calibration::run_assigned_cluster_calibration;
#[cfg(test)]
use calibration::*;
pub(crate) use lifecycle::start_bringup_supervisor;
#[cfg(test)]
use lifecycle::*;
pub(crate) use upstream::BringupError;
#[cfg(test)]
use upstream::*;

const DEFAULT_ACTIVE_CANARY_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_CANARY_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_CANARY_MAX_GENERATION_THRESHOLD: u32 = 237;
const DEFAULT_CALIBRATION_REQUESTS: usize = 5;
const DEFAULT_CALIBRATION_PROMPT_UNITS: usize = 4096;
const DEFAULT_CALIBRATION_MAX_CONCURRENCY: usize = 4;
const DEFAULT_CALIBRATION_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct BringupConfig {
    pub enabled: bool,
    pub active_canary_interval: Duration,
    pub canary_timeout: Duration,
    pub canary_max_generation_threshold: u32,
    pub calibration_requests: usize,
    pub calibration_prompt_units: usize,
    pub calibration_max_concurrency: usize,
    pub calibration_timeout: Duration,
}

impl Default for BringupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            active_canary_interval: DEFAULT_ACTIVE_CANARY_INTERVAL,
            canary_timeout: DEFAULT_CANARY_TIMEOUT,
            canary_max_generation_threshold: DEFAULT_CANARY_MAX_GENERATION_THRESHOLD,
            calibration_requests: DEFAULT_CALIBRATION_REQUESTS,
            calibration_prompt_units: DEFAULT_CALIBRATION_PROMPT_UNITS,
            calibration_max_concurrency: DEFAULT_CALIBRATION_MAX_CONCURRENCY,
            calibration_timeout: DEFAULT_CALIBRATION_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelBringupState {
    ConnectingUnavailable,
    Recovering,
    AdvertisingActive,
}

#[derive(Debug, Clone)]
pub(crate) struct BringupTaskConfig {
    pub upstream_http_base_url: String,
    pub model_id: String,
    pub config: BringupConfig,
    pub metrics: Option<Arc<PylonMetrics>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use axum::Json;
    use axum::Router;
    use axum::extract::State;
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use serde_json::Value;
    use stargate_proto::pb::InferenceServerStatus;
    use tokio::net::TcpListener;
    use tokio::sync::{Barrier, Mutex, Notify};
    use tokio_util::sync::CancellationToken;

    use crate::runtime_state::PylonRuntimeState;

    async fn wait_for_bringup_state(
        runtime_state: &PylonRuntimeState,
        expected: ModelBringupState,
    ) {
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut poll = tokio::time::interval(Duration::from_millis(1));
            loop {
                poll.tick().await;
                if runtime_state.model_bringup("test-model") == Some(expected) {
                    return;
                }
            }
        })
        .await
        .expect("bringup task should publish expected state");
    }

    async fn wait_for_bringup_notification(notification: &Notify, expected_event: &str) {
        if tokio::time::timeout(Duration::from_secs(2), notification.notified())
            .await
            .is_err()
        {
            panic!("bringup task should {expected_event}");
        }
    }

    #[test]
    fn detects_prompt_too_long_errors() {
        assert!(is_prompt_too_long(
            StatusCode::BAD_REQUEST,
            &Some("Prompt too long for model context length".to_string())
        ));
        assert!(!is_prompt_too_long(
            StatusCode::INTERNAL_SERVER_ERROR,
            &Some("prompt too long".to_string())
        ));
    }

    #[test]
    fn bringup_lifecycle_state_classifies_actions() {
        let mut lifecycle = BringupLifecycleState::default();
        assert_eq!(
            lifecycle.next_action(),
            BringupLifecycleAction::AdvertiseInitialActive
        );

        lifecycle.complete_initial_bringup();
        assert_eq!(lifecycle, BringupLifecycleState::Active);
        assert_eq!(
            lifecycle.next_action(),
            BringupLifecycleAction::AdvertiseActive
        );

        lifecycle.require_recovery_canary();
        assert_eq!(lifecycle, BringupLifecycleState::Recovering);
        assert_eq!(
            lifecycle.next_action(),
            BringupLifecycleAction::RunRecoveryCanary
        );

        lifecycle.complete_recovery_canary();
        assert_eq!(lifecycle, BringupLifecycleState::Active);
    }

    #[tokio::test]
    async fn wait_or_stop_returns_when_cancelled() {
        let stop = CancellationToken::new();
        stop.cancel();

        assert!(wait_or_stop(&stop, Duration::from_secs(60)).await);
    }

    #[tokio::test]
    async fn canary_request_detects_runaway_generation() {
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 7,
            prompt_too_long_above: None,
            calibration_barrier: None,
            completion_delay: None,
            in_flight: None,
            max_in_flight: None,
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let client = reqwest::Client::new();
        let error =
            send_canary_request(&client, &base_url, "test-model", Duration::from_secs(1), 7)
                .await
                .expect_err("expected runaway generation");
        assert!(matches!(
            error,
            BringupError::RunawayGeneration { tokens: 7 }
        ));
    }

    #[tokio::test]
    async fn calibration_reduces_prompt_size_after_prompt_too_long() {
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: Some(700),
            calibration_barrier: None,
            completion_delay: None,
            in_flight: None,
            max_in_flight: None,
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let client = reqwest::Client::new();
        let config = BringupConfig {
            calibration_requests: 5,
            calibration_prompt_units: 1536,
            calibration_timeout: Duration::from_secs(1),
            ..BringupConfig::default()
        };

        let observed = run_calibration(&client, &base_url, "test-model", &config)
            .await
            .expect("calibration should back off and succeed");
        assert!(observed.is_finite());
        assert!(observed > 0.0);
    }

    #[tokio::test]
    async fn single_request_calibration_completes_after_prompt_backoff() {
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: Some(700),
            calibration_barrier: None,
            completion_delay: None,
            in_flight: None,
            max_in_flight: None,
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let client = reqwest::Client::new();
        let config = BringupConfig {
            calibration_requests: 1,
            calibration_prompt_units: 1536,
            calibration_timeout: Duration::from_secs(1),
            ..BringupConfig::default()
        };

        let observed = run_calibration(&client, &base_url, "test-model", &config)
            .await
            .expect("single-request calibration should complete with one valid configured sample");
        assert!(observed.is_finite());
        assert!(observed > 0.0);
    }

    #[test]
    fn calibration_plan_sweeps_tokens_at_increasing_concurrency_levels() {
        let config = BringupConfig {
            calibration_requests: 5,
            calibration_prompt_units: 1024,
            calibration_max_concurrency: 4,
            ..BringupConfig::default()
        };

        let plan = calibration_plan(&config);

        assert_eq!(calibration_plan_request_count(&plan), 5);
        assert_eq!(
            plan,
            vec![
                CalibrationBatch {
                    prompt_units: CALIBRATION_PROMPT_UNITS_FLOOR,
                    concurrency: 1,
                },
                CalibrationBatch {
                    prompt_units: 1024,
                    concurrency: 4,
                },
            ]
        );
    }

    #[test]
    fn calibration_plan_preserves_linear_ramp_when_quadrants_cannot_be_sampled() {
        let config = BringupConfig {
            calibration_requests: 3,
            calibration_prompt_units: 1024,
            calibration_max_concurrency: 4,
            ..BringupConfig::default()
        };

        let plan = calibration_plan(&config);

        assert_eq!(calibration_plan_request_count(&plan), 3);
        assert_eq!(
            plan,
            vec![
                CalibrationBatch {
                    prompt_units: CALIBRATION_PROMPT_UNITS_FLOOR,
                    concurrency: 1,
                },
                CalibrationBatch {
                    prompt_units: 1024,
                    concurrency: 2,
                },
            ]
        );
    }

    #[test]
    fn single_calibration_request_does_not_expand_to_max_concurrency() {
        let config = BringupConfig {
            calibration_requests: 1,
            calibration_prompt_units: 1024,
            calibration_max_concurrency: 4,
            ..BringupConfig::default()
        };

        let plan = calibration_plan(&config);

        assert_eq!(calibration_plan_request_count(&plan), 1);
        assert_eq!(
            plan,
            vec![CalibrationBatch {
                prompt_units: 1024,
                concurrency: 1,
            }]
        );
    }

    fn calibration_plan_request_count(plan: &[CalibrationBatch]) -> usize {
        plan.iter().map(|batch| batch.concurrency).sum()
    }

    #[tokio::test]
    async fn calibration_batch_sends_requests_concurrently() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: None,
            calibration_barrier: Some(Arc::new(Barrier::new(3))),
            completion_delay: None,
            in_flight: Some(in_flight),
            max_in_flight: Some(max_in_flight.clone()),
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let client = reqwest::Client::new();

        let observed = send_calibration_batch(
            &client,
            &base_url,
            "test-model",
            Duration::from_secs(1),
            256,
            3,
        )
        .await
        .expect("calibration batch should succeed");

        assert_eq!(observed.len(), 3);
        assert!(observed.iter().all(|sample| *sample > 0.0));
        assert_eq!(max_in_flight.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn calibration_batch_aggregate_capacity_scales_with_concurrency() {
        let serial = aggregate_input_tps(256, 1, Duration::from_millis(100));
        let concurrent = aggregate_input_tps(256, 3, Duration::from_millis(100));
        let immediate = aggregate_input_tps(256, 3, Duration::ZERO);

        assert!((serial - 2_560.0).abs() < f64::EPSILON);
        assert!((concurrent - 7_680.0).abs() < f64::EPSILON);
        assert!(immediate.is_finite());
    }

    #[tokio::test]
    async fn assigned_cluster_calibration_records_duration_metric() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: None,
            calibration_barrier: None,
            completion_delay: None,
            in_flight: None,
            max_in_flight: None,
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let last_mean_input_tps = run_assigned_cluster_calibration(&BringupTaskConfig {
            upstream_http_base_url: base_url,
            model_id: "test-model".to_string(),
            config: BringupConfig {
                calibration_requests: 5,
                active_canary_interval: Duration::ZERO,
                canary_timeout: Duration::from_secs(1),
                calibration_timeout: Duration::from_secs(1),
                ..BringupConfig::default()
            },
            metrics: Some(metrics.clone()),
        })
        .await
        .expect("assigned calibration should produce a local measurement");
        assert!(last_mean_input_tps > 0.0);

        let body = metrics.gather_text().expect("metrics should encode");
        assert!(body.contains(
            r#"pylon_model_calibration_duration_ms_count{model="test-model",outcome="success"} 1"#
        ));
        assert!(!body.contains(
            r#"pylon_model_calibration_duration_ms_count{model="test-model",outcome="failure"}"#
        ));
    }

    #[tokio::test]
    async fn assigned_cluster_calibration_does_not_measure_unhealthy_upstream() {
        let health_ok = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: None,
            calibration_barrier: None,
            completion_delay: None,
            in_flight: Some(in_flight),
            max_in_flight: Some(max_in_flight.clone()),
            canary_failures_remaining: None,
            health_ok: Some(health_ok),
        })
        .await;

        let error = run_assigned_cluster_calibration(&BringupTaskConfig {
            upstream_http_base_url: base_url,
            model_id: "test-model".to_string(),
            config: BringupConfig {
                calibration_requests: 1,
                canary_timeout: Duration::from_secs(1),
                calibration_timeout: Duration::from_secs(1),
                ..BringupConfig::default()
            },
            metrics: None,
        })
        .await
        .expect_err("unhealthy upstream must not be measured for an assignment");

        assert!(matches!(error, BringupError::UnhealthyUpstream));
        assert_eq!(
            max_in_flight.load(Ordering::SeqCst),
            0,
            "calibration requests must wait until upstream health succeeds"
        );
    }

    #[tokio::test]
    async fn recovery_canary_runs_after_initial_health_activation() {
        let canary_requests = Arc::new(AtomicUsize::new(0));
        let initial_canary_started = Arc::new(Notify::new());
        let release_initial_canary = Arc::new(Notify::new());
        let recovery_canary_started = Arc::new(Notify::new());
        let release_recovery_canary = Arc::new(Notify::new());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_canary_requests = canary_requests.clone();
        let server_initial_canary_started = initial_canary_started.clone();
        let server_release_initial_canary = release_initial_canary.clone();
        let server_recovery_canary_started = recovery_canary_started.clone();
        let server_release_recovery_canary = release_recovery_canary.clone();
        let server = tokio::spawn(async move {
            let app = Router::new()
                .route("/health", get(|| async { StatusCode::OK }))
                .route(
                    "/v1/chat/completions",
                    post(move |Json(request): Json<Value>| {
                        let canary_requests = server_canary_requests.clone();
                        let initial_canary_started = server_initial_canary_started.clone();
                        let release_initial_canary = server_release_initial_canary.clone();
                        let recovery_canary_started = server_recovery_canary_started.clone();
                        let release_recovery_canary = server_release_recovery_canary.clone();
                        async move {
                            let prompt = request
                                .get("messages")
                                .and_then(|value| value.as_array())
                                .and_then(|messages| messages.first())
                                .and_then(|message| message.get("content"))
                                .and_then(|value| value.as_str())
                                .unwrap_or_default();
                            let completion_tokens = if prompt == "1+1=" {
                                match canary_requests.fetch_add(1, Ordering::SeqCst) {
                                    0 => {
                                        initial_canary_started.notify_one();
                                        release_initial_canary.notified().await;
                                        7
                                    }
                                    1 => {
                                        recovery_canary_started.notify_one();
                                        release_recovery_canary.notified().await;
                                        1
                                    }
                                    _ => 1,
                                }
                            } else {
                                1
                            };
                            Json(serde_json::json!({
                                "usage": {"completion_tokens": completion_tokens}
                            }))
                            .into_response()
                        }
                    }),
                );
            axum::serve(listener, app).await.unwrap();
        });
        let runtime_state =
            PylonRuntimeState::new(InferenceServerStatus::Active, &["test-model".into()]);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            BringupTaskConfig {
                upstream_http_base_url: format!("http://{addr}"),
                model_id: "test-model".to_string(),
                config: BringupConfig {
                    calibration_requests: 0,
                    active_canary_interval: Duration::from_millis(10),
                    canary_timeout: Duration::from_secs(1),
                    canary_max_generation_threshold: 7,
                    ..BringupConfig::default()
                },
                metrics: None,
            },
            runtime_state.clone(),
            stop.clone(),
        ));

        wait_for_bringup_notification(&initial_canary_started, "start the initial canary").await;
        assert_eq!(
            runtime_state.model_bringup("test-model"),
            Some(ModelBringupState::AdvertisingActive)
        );
        release_initial_canary.notify_one();

        wait_for_bringup_notification(&recovery_canary_started, "start the recovery canary").await;
        assert_eq!(
            runtime_state.model_bringup("test-model"),
            Some(ModelBringupState::Recovering)
        );
        release_recovery_canary.notify_one();

        wait_for_bringup_state(&runtime_state, ModelBringupState::AdvertisingActive).await;

        stop.cancel();
        let task_result = task.await;
        server.abort();
        task_result.unwrap();
    }

    #[tokio::test]
    async fn active_bringup_stops_when_cancelled() {
        let base_url = spawn_test_server(TestServerState {
            completion_tokens: 1,
            prompt_too_long_above: None,
            calibration_barrier: None,
            completion_delay: None,
            in_flight: None,
            max_in_flight: None,
            canary_failures_remaining: None,
            health_ok: None,
        })
        .await;
        let runtime_state =
            PylonRuntimeState::new(InferenceServerStatus::Active, &["test-model".into()]);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            BringupTaskConfig {
                upstream_http_base_url: base_url,
                model_id: "test-model".to_string(),
                config: BringupConfig {
                    calibration_requests: 0,
                    active_canary_interval: Duration::from_secs(60),
                    canary_timeout: Duration::from_secs(1),
                    ..BringupConfig::default()
                },
                metrics: None,
            },
            runtime_state.clone(),
            stop.clone(),
        ));

        wait_for_bringup_state(&runtime_state, ModelBringupState::AdvertisingActive).await;

        stop.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("bringup task should stop when cancelled")
            .expect("bringup task should not panic");
    }

    #[tokio::test]
    async fn disabled_bringup_is_applied_without_ending_supervisor() {
        let runtime_state =
            PylonRuntimeState::new(InferenceServerStatus::Active, &["test-model".into()]);
        let parent_stop = CancellationToken::new();
        let supervisor = start_bringup_supervisor(
            &parent_stop,
            vec![BringupTaskConfig {
                upstream_http_base_url: "http://127.0.0.1:1".to_string(),
                model_id: "test-model".to_string(),
                config: BringupConfig {
                    enabled: false,
                    ..BringupConfig::default()
                },
                metrics: None,
            }],
            runtime_state.clone(),
        );

        wait_for_bringup_state(&runtime_state, ModelBringupState::AdvertisingActive).await;

        assert!(
            !parent_stop.is_cancelled(),
            "synchronously applied disabled bringup must not end the session"
        );
        supervisor.shutdown(Duration::from_secs(1)).await;
    }

    #[tokio::test]
    async fn bringup_cancellation_interrupts_blocked_health_check() {
        let health_entered = Arc::new(Barrier::new(2));
        let server_health_entered = health_entered.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let app = Router::new().route(
                "/health",
                get(move || {
                    let health_entered = server_health_entered.clone();
                    async move {
                        health_entered.wait().await;
                        std::future::pending::<&'static str>().await
                    }
                }),
            );
            axum::serve(listener, app).await.unwrap();
        });
        let runtime_state =
            PylonRuntimeState::new(InferenceServerStatus::Active, &["test-model".into()]);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            BringupTaskConfig {
                upstream_http_base_url: format!("http://{addr}"),
                model_id: "test-model".to_string(),
                config: BringupConfig {
                    canary_timeout: Duration::from_secs(60),
                    ..BringupConfig::default()
                },
                metrics: None,
            },
            runtime_state,
            stop.clone(),
        ));
        health_entered.wait().await;

        stop.cancel();
        let stopped = tokio::time::timeout(Duration::from_secs(1), task).await;
        server.abort();

        stopped
            .expect("bringup cancellation should interrupt blocked health check")
            .expect("bringup task should not panic");
    }

    #[derive(Clone)]
    struct TestServerState {
        completion_tokens: u32,
        prompt_too_long_above: Option<usize>,
        calibration_barrier: Option<Arc<Barrier>>,
        completion_delay: Option<Duration>,
        in_flight: Option<Arc<AtomicUsize>>,
        max_in_flight: Option<Arc<AtomicUsize>>,
        canary_failures_remaining: Option<Arc<AtomicUsize>>,
        health_ok: Option<Arc<AtomicBool>>,
    }

    async fn spawn_test_server(state: TestServerState) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(state));
        let app = Router::new()
            .route("/health", get(test_health))
            .route("/v1/chat/completions", post(test_chat_completion))
            .with_state(state);
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn test_health(
        State(state): State<Arc<Mutex<TestServerState>>>,
    ) -> axum::response::Response {
        let state = state.lock().await.clone();
        if state
            .health_ok
            .as_ref()
            .is_some_and(|health_ok| !health_ok.load(Ordering::SeqCst))
        {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        "ok".into_response()
    }

    async fn test_chat_completion(
        State(state): State<Arc<Mutex<TestServerState>>>,
        Json(request): Json<Value>,
    ) -> axum::response::Response {
        let state = state.lock().await.clone();
        let prompt = request
            .get("messages")
            .and_then(|value| value.as_array())
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("content"))
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        let prompt_len = prompt.len();
        if let Some(in_flight) = &state.in_flight {
            let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some(max_in_flight) = &state.max_in_flight {
                let mut observed = max_in_flight.load(Ordering::SeqCst);
                while current > observed {
                    match max_in_flight.compare_exchange(
                        observed,
                        current,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(next_observed) => observed = next_observed,
                    }
                }
            }
        }
        if let Some(barrier) = &state.calibration_barrier {
            barrier.wait().await;
        }
        if let Some(delay) = state.completion_delay {
            tokio::time::sleep(delay).await;
        }
        if let Some(in_flight) = &state.in_flight {
            in_flight.fetch_sub(1, Ordering::SeqCst);
        }
        if let Some(limit) = state.prompt_too_long_above
            && prompt_len > limit
        {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {"message": "Prompt too long"}
                })),
            )
                .into_response();
        }

        let mut completion_tokens = state.completion_tokens;
        if prompt == "1+1="
            && state
                .canary_failures_remaining
                .as_ref()
                .is_some_and(|remaining| {
                    remaining
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                            (value > 0).then(|| value - 1)
                        })
                        .is_ok()
                })
        {
            completion_tokens = request
                .get("max_tokens")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(completion_tokens);
        }

        Json(serde_json::json!({
            "usage": {"completion_tokens": completion_tokens}
        }))
        .into_response()
    }
}
