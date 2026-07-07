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

use std::time::Duration;

mod calibration;
mod lifecycle;
mod upstream;

pub use calibration::run_startup_calibration;
pub use lifecycle::{BringupHandle, start_bringup};
pub use upstream::BringupError;

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
}

#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    pub health_timeout: Duration,
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
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            health_timeout: DEFAULT_CANARY_TIMEOUT,
            calibration_requests: DEFAULT_CALIBRATION_REQUESTS,
            calibration_prompt_units: DEFAULT_CALIBRATION_PROMPT_UNITS,
            calibration_max_concurrency: DEFAULT_CALIBRATION_MAX_CONCURRENCY,
            calibration_timeout: DEFAULT_CALIBRATION_TIMEOUT,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BringupTaskConfig {
    pub upstream_http_base_url: String,
    pub model_id: String,
    pub config: BringupConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{calibration::*, lifecycle::*, upstream::*};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use axum::Json;
    use axum::Router;
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::response::IntoResponse;
    use axum::routing::{get, post};
    use reqwest::StatusCode;
    use serde_json::Value;
    use stargate_proto::pb::InferenceServerStatus;
    use stargate_protocol::tunnel_contract::HEADER_REQUEST_ID;
    use tokio::net::TcpListener;
    use tokio::sync::{Barrier, Mutex, Notify};
    use tokio_util::sync::CancellationToken;
    use uuid::Uuid;

    use crate::runtime_state::PylonRuntimeState;
    use crate::stats::PylonMetrics;

    async fn wait_for_bringup_ready(runtime_state: &PylonRuntimeState, expected: bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut poll = tokio::time::interval(Duration::from_millis(1));
            loop {
                poll.tick().await;
                if runtime_state.model_bringup_ready("test-model") == Some(expected) {
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

    fn test_runtime_state() -> PylonRuntimeState {
        PylonRuntimeState::new(InferenceServerStatus::Active, &["test-model".into()])
    }

    fn test_task_config(
        upstream_http_base_url: String,
        config: BringupConfig,
    ) -> BringupTaskConfig {
        BringupTaskConfig {
            upstream_http_base_url,
            model_id: "test-model".to_string(),
            config,
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
            ..TestServerState::default()
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
    async fn pylon_generated_request_ids_include_kind_uuid_and_monotonic_counter() {
        let request_ids = Arc::new(Mutex::new(Vec::new()));
        let base_url = spawn_test_server(TestServerState {
            request_ids: Some(request_ids.clone()),
            ..TestServerState::default()
        })
        .await;
        let client = reqwest::Client::new();

        send_canary_request(&client, &base_url, "test-model", Duration::from_secs(1), 7)
            .await
            .expect("canary request should succeed");
        send_calibration_batch(
            &client,
            &base_url,
            "test-model",
            Duration::from_secs(1),
            CALIBRATION_PROMPT_UNITS_FLOOR,
            2,
        )
        .await
        .expect("calibration batch should succeed");

        let request_ids = request_ids.lock().await.clone();
        assert_eq!(request_ids.len(), 3);

        let parsed = request_ids
            .iter()
            .map(|request_id| parse_pylon_generated_request_id(request_id))
            .collect::<Vec<_>>();
        assert_eq!(parsed[0].kind, "canary");
        assert!(
            parsed[1..]
                .iter()
                .all(|parsed| parsed.kind == "calibration")
        );
        let request_scope = parsed[0].uuid;
        assert!(
            parsed
                .iter()
                .all(|request_id| request_id.uuid == request_scope)
        );
        assert!(parsed.iter().all(|request_id| request_id.counter > 0));

        let canary_counter = parsed[0].counter;
        let mut calibration_counters = parsed[1..]
            .iter()
            .map(|parsed| parsed.counter)
            .collect::<Vec<_>>();
        calibration_counters.sort_unstable();
        calibration_counters.dedup();
        assert_eq!(calibration_counters.len(), 2);
        assert!(
            calibration_counters
                .iter()
                .all(|counter| *counter > canary_counter)
        );
    }

    #[tokio::test]
    async fn calibration_reduces_prompt_size_after_prompt_too_long() {
        let observed = calibrate_after_prompt_backoff(5).await;
        assert!(observed.is_finite());
        assert!(observed > 0.0);
    }

    #[tokio::test]
    async fn single_request_calibration_completes_after_prompt_backoff() {
        let observed = calibrate_after_prompt_backoff(1).await;
        assert!(observed.is_finite());
        assert!(observed > 0.0);
    }

    async fn calibrate_after_prompt_backoff(calibration_requests: usize) -> f64 {
        let base_url = spawn_test_server(TestServerState {
            prompt_too_long_above: Some(700),
            ..TestServerState::default()
        })
        .await;
        let client = reqwest::Client::new();
        let config = CalibrationConfig {
            calibration_requests,
            calibration_prompt_units: 1536,
            calibration_timeout: Duration::from_secs(1),
            ..CalibrationConfig::default()
        };

        run_calibration(&client, &base_url, "test-model", &config)
            .await
            .expect("calibration should back off and succeed")
    }

    #[test]
    fn calibration_plan_sweeps_tokens_at_increasing_concurrency_levels() {
        assert_calibration_plan(5, 4, &[(CALIBRATION_PROMPT_UNITS_FLOOR, 1), (1024, 4)]);
    }

    #[test]
    fn calibration_plan_preserves_linear_ramp_when_quadrants_cannot_be_sampled() {
        assert_calibration_plan(3, 4, &[(CALIBRATION_PROMPT_UNITS_FLOOR, 1), (1024, 2)]);
    }

    #[test]
    fn serial_calibration_plan_preserves_full_prompt_ramp() {
        assert_calibration_plan(
            3,
            1,
            &[(CALIBRATION_PROMPT_UNITS_FLOOR, 1), (640, 1), (1024, 1)],
        );
    }

    #[test]
    fn single_calibration_request_does_not_expand_to_max_concurrency() {
        assert_calibration_plan(1, 4, &[(1024, 1)]);
    }

    fn assert_calibration_plan(
        calibration_requests: usize,
        calibration_max_concurrency: usize,
        expected: &[(usize, usize)],
    ) {
        let config = CalibrationConfig {
            calibration_requests,
            calibration_prompt_units: 1024,
            calibration_max_concurrency,
            ..CalibrationConfig::default()
        };
        let plan = calibration_plan(&config);

        assert_eq!(
            plan.iter().map(|batch| batch.concurrency).sum::<usize>(),
            calibration_requests
        );
        assert_eq!(
            plan.iter()
                .map(|batch| (batch.prompt_units, batch.concurrency))
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[tokio::test]
    async fn calibration_batch_sends_requests_concurrently() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_test_server(TestServerState {
            calibration_barrier: Some(Arc::new(Barrier::new(3))),
            in_flight: Some(in_flight),
            max_in_flight: Some(max_in_flight.clone()),
            ..TestServerState::default()
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
    async fn startup_calibration_records_duration_metric() {
        let metrics = PylonMetrics::new().expect("metrics should initialize");
        let base_url = spawn_test_server(TestServerState::default()).await;
        let model_ids = vec!["test-model".to_string()];
        let calibration = run_startup_calibration(
            &base_url,
            &model_ids,
            &CalibrationConfig {
                health_timeout: Duration::from_secs(1),
                calibration_requests: 5,
                calibration_timeout: Duration::from_secs(1),
                ..CalibrationConfig::default()
            },
            Some(metrics.clone()),
        )
        .await
        .expect("startup calibration should produce a local measurement");
        let last_mean_input_tps = calibration["test-model"];
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
    async fn startup_calibration_rejects_zero_requests() {
        let base_url = spawn_test_server(TestServerState::default()).await;

        let error = run_startup_calibration(
            &base_url,
            &["test-model".to_string()],
            &CalibrationConfig {
                calibration_requests: 0,
                ..CalibrationConfig::default()
            },
            None,
        )
        .await
        .expect_err("zero requests cannot produce a valid input-TPS bootstrap");

        assert!(matches!(
            error,
            BringupError::InvalidCalibrationConfig(
                "calibration_requests must be greater than zero"
            )
        ));
    }

    #[tokio::test]
    async fn startup_calibration_does_not_measure_unhealthy_upstream() {
        let health_ok = Arc::new(AtomicBool::new(false));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_test_server(TestServerState {
            in_flight: Some(in_flight),
            max_in_flight: Some(max_in_flight.clone()),
            health_ok: Some(health_ok),
            ..TestServerState::default()
        })
        .await;

        let error = run_startup_calibration(
            &base_url,
            &["test-model".to_string()],
            &CalibrationConfig {
                health_timeout: Duration::from_secs(1),
                calibration_requests: 1,
                calibration_timeout: Duration::from_secs(1),
                ..CalibrationConfig::default()
            },
            None,
        )
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
        let canaries = CanarySequence::default();
        let health_requests = Arc::new(AtomicUsize::new(0));
        let base_url = spawn_test_server(TestServerState {
            canaries: Some(canaries.clone()),
            health_requests: Some(health_requests.clone()),
            ..TestServerState::default()
        })
        .await;
        let runtime_state = test_runtime_state();
        runtime_state.set_model_bringup_ready("test-model", true);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            test_task_config(
                base_url,
                BringupConfig {
                    active_canary_interval: Duration::from_millis(10),
                    canary_timeout: Duration::from_secs(1),
                    canary_max_generation_threshold: 7,
                    ..BringupConfig::default()
                },
            ),
            runtime_state.clone(),
            stop.clone(),
        ));

        wait_for_bringup_notification(&canaries.started[0], "start the initial canary").await;
        assert_eq!(runtime_state.model_bringup_ready("test-model"), Some(true));
        canaries.release[0].notify_one();

        wait_for_bringup_notification(&canaries.started[1], "start the recovery canary").await;
        assert_eq!(
            health_requests.load(Ordering::SeqCst),
            1,
            "one recovery attempt should perform one health check"
        );
        assert_eq!(runtime_state.model_bringup_ready("test-model"), Some(false));
        canaries.release[1].notify_one();

        wait_for_bringup_ready(&runtime_state, true).await;

        stop.cancel();
        task.await.unwrap();
    }

    #[tokio::test]
    async fn active_bringup_stops_when_cancelled() {
        let base_url = spawn_test_server(TestServerState::default()).await;
        let runtime_state = test_runtime_state();
        runtime_state.set_model_bringup_ready("test-model", true);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            test_task_config(
                base_url,
                BringupConfig {
                    active_canary_interval: Duration::from_secs(60),
                    canary_timeout: Duration::from_secs(1),
                    ..BringupConfig::default()
                },
            ),
            runtime_state.clone(),
            stop.clone(),
        ));

        wait_for_bringup_ready(&runtime_state, true).await;

        stop.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("bringup task should stop when cancelled")
            .expect("bringup task should not panic");
    }

    #[tokio::test]
    async fn disabled_bringup_starts_no_supervisor() {
        let runtime_state = test_runtime_state();
        let supervisor = start_bringup(
            "http://127.0.0.1:1",
            BringupConfig {
                enabled: false,
                ..BringupConfig::default()
            },
            runtime_state.clone(),
        )
        .await
        .expect("disabled bringup should start");

        wait_for_bringup_ready(&runtime_state, true).await;
        assert!(supervisor.is_none());
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
        let runtime_state = test_runtime_state();
        runtime_state.set_model_bringup_ready("test-model", true);
        let stop = CancellationToken::new();
        let task = tokio::spawn(run_bringup_task(
            test_task_config(
                format!("http://{addr}"),
                BringupConfig {
                    active_canary_interval: Duration::from_millis(1),
                    canary_timeout: Duration::from_secs(60),
                    ..BringupConfig::default()
                },
            ),
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
        canaries: Option<CanarySequence>,
        health_ok: Option<Arc<AtomicBool>>,
        health_requests: Option<Arc<AtomicUsize>>,
        request_ids: Option<Arc<Mutex<Vec<String>>>>,
    }

    impl Default for TestServerState {
        fn default() -> Self {
            Self {
                completion_tokens: 1,
                prompt_too_long_above: None,
                calibration_barrier: None,
                completion_delay: None,
                in_flight: None,
                max_in_flight: None,
                canary_failures_remaining: None,
                canaries: None,
                health_ok: None,
                health_requests: None,
                request_ids: None,
            }
        }
    }

    #[derive(Debug)]
    struct ParsedPylonGeneratedRequestId<'a> {
        kind: &'a str,
        uuid: Uuid,
        counter: u64,
    }

    fn parse_pylon_generated_request_id(request_id: &str) -> ParsedPylonGeneratedRequestId<'_> {
        let (kind, suffix) = request_id
            .split_once('-')
            .expect("request id should include kind prefix");
        let (uuid, counter) = suffix
            .rsplit_once('-')
            .expect("request id should end with a counter suffix");
        ParsedPylonGeneratedRequestId {
            kind,
            uuid: Uuid::parse_str(uuid).expect("request id should include a UUID"),
            counter: counter
                .parse()
                .expect("request id should include a decimal counter"),
        }
    }

    #[derive(Clone, Default)]
    struct CanarySequence {
        requests: Arc<AtomicUsize>,
        started: [Arc<Notify>; 2],
        release: [Arc<Notify>; 2],
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
        if let Some(health_requests) = &state.health_requests {
            health_requests.fetch_add(1, Ordering::SeqCst);
        }
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
        headers: HeaderMap,
        Json(request): Json<Value>,
    ) -> axum::response::Response {
        let state = state.lock().await.clone();
        if let Some(request_ids) = &state.request_ids
            && let Some(request_id) = headers
                .get(HEADER_REQUEST_ID)
                .and_then(|value| value.to_str().ok())
        {
            request_ids.lock().await.push(request_id.to_string());
        }
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
        if prompt == "1+1=" {
            if let Some(canaries) = &state.canaries {
                let request = canaries.requests.fetch_add(1, Ordering::SeqCst);
                if let (Some(started), Some(release)) =
                    (canaries.started.get(request), canaries.release.get(request))
                {
                    started.notify_one();
                    release.notified().await;
                }
                completion_tokens = if request == 0 { 7 } else { 1 };
            } else if state
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
        }

        Json(serde_json::json!({
            "usage": {"completion_tokens": completion_tokens}
        }))
        .into_response()
    }
}
