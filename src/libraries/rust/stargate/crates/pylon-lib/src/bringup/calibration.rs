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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;

use crate::stats::PylonMetrics;
use crate::stats::token_metrics::{SNAPSHOT_THRESHOLD, TpsDistribution};

use super::CalibrationConfig;
use super::upstream::{BringupError, check_upstream_health, send_completion_request};

pub(super) const CALIBRATION_PROMPT_UNITS_FLOOR: usize = 256;

pub async fn run_startup_calibration(
    upstream_http_base_url: &str,
    model_ids: &[String],
    config: &CalibrationConfig,
    metrics: Option<Arc<PylonMetrics>>,
) -> Result<HashMap<String, f64>, BringupError> {
    if config.calibration_requests == 0 {
        return Err(BringupError::InvalidCalibrationConfig(
            "calibration_requests must be greater than zero",
        ));
    }

    let client = reqwest::Client::new();
    let mut bootstrap_input_tps = HashMap::with_capacity(model_ids.len());
    for model_id in model_ids {
        let started_at = Instant::now();
        let result = if check_upstream_health(
            &client,
            upstream_http_base_url,
            config.health_timeout,
        )
        .await
        {
            run_calibration(&client, upstream_http_base_url, model_id, config).await
        } else {
            Err(BringupError::UnhealthyUpstream)
        };
        if let Some(metrics) = metrics.as_deref() {
            metrics.observe_model_calibration_duration(
                model_id,
                started_at.elapsed(),
                result.is_ok(),
            );
        }
        bootstrap_input_tps.insert(model_id.clone(), result?);
    }
    Ok(bootstrap_input_tps)
}

pub(super) async fn run_calibration(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    config: &CalibrationConfig,
) -> Result<f64, BringupError> {
    let mut distribution = TpsDistribution::default();
    let mut last_error = None;

    for batch in calibration_plan(config) {
        match send_calibration_batch_with_prompt_backoff(
            http_client,
            upstream_http_base_url,
            model_id,
            config.calibration_timeout,
            batch,
        )
        .await
        {
            Ok(observed_input_tps_samples) => {
                for sample in observed_input_tps_samples {
                    distribution.update(sample);
                }
            }
            Err(BringupError::PromptTooLong) => last_error = Some(BringupError::PromptTooLong),
            Err(error) => return Err(error),
        }
    }

    let required_samples = config.calibration_requests.min(SNAPSHOT_THRESHOLD);
    if distribution.count >= required_samples {
        return Ok(distribution.mean);
    }
    Err(
        last_error.unwrap_or(BringupError::InsufficientCalibrationSamples {
            valid_samples: distribution.count,
        }),
    )
}

async fn send_calibration_batch_with_prompt_backoff(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    timeout: Duration,
    batch: CalibrationBatch,
) -> Result<Vec<f64>, BringupError> {
    let mut prompt_units = batch.prompt_units.max(CALIBRATION_PROMPT_UNITS_FLOOR);

    loop {
        match send_calibration_batch(
            http_client,
            upstream_http_base_url,
            model_id,
            timeout,
            prompt_units,
            batch.concurrency,
        )
        .await
        {
            Err(BringupError::PromptTooLong) if prompt_units > CALIBRATION_PROMPT_UNITS_FLOOR => {
                prompt_units = CALIBRATION_PROMPT_UNITS_FLOOR
                    + (prompt_units - CALIBRATION_PROMPT_UNITS_FLOOR) / 2;
            }
            result => return result,
        }
    }
}

pub(super) struct CalibrationBatch {
    pub(super) prompt_units: usize,
    pub(super) concurrency: usize,
}

pub(super) fn calibration_plan(config: &CalibrationConfig) -> Vec<CalibrationBatch> {
    let requests = config.calibration_requests;
    let max_prompt_units = config
        .calibration_prompt_units
        .max(CALIBRATION_PROMPT_UNITS_FLOOR);
    // A zero calibration concurrency override degrades to serial calibration.
    let max_concurrency = config.calibration_max_concurrency.max(1).min(requests);
    if requests == 1 {
        return vec![CalibrationBatch {
            prompt_units: max_prompt_units,
            concurrency: 1,
        }];
    }

    let final_concurrency = max_concurrency.min(requests - 1);
    let single_request_runs = requests - final_concurrency;
    let prompt_span = max_prompt_units - CALIBRATION_PROMPT_UNITS_FLOOR;
    (0..single_request_runs)
        .map(|index| CalibrationBatch {
            prompt_units: CALIBRATION_PROMPT_UNITS_FLOOR
                + (prompt_span as u128 * index as u128 / single_request_runs as u128) as usize,
            concurrency: 1,
        })
        .chain(std::iter::once(CalibrationBatch {
            prompt_units: max_prompt_units,
            concurrency: final_concurrency,
        }))
        .collect()
}

pub(super) fn aggregate_input_tps(
    prompt_units: usize,
    concurrency: usize,
    elapsed: Duration,
) -> f64 {
    // Sub-millisecond localhost measurements should not report infinite
    // calibrated throughput.
    let elapsed = elapsed.max(Duration::from_millis(1));
    (prompt_units as f64 * concurrency as f64) / elapsed.as_secs_f64()
}

pub(super) async fn send_calibration_batch(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    timeout: Duration,
    prompt_units: usize,
    concurrency: usize,
) -> Result<Vec<f64>, BringupError> {
    assert!(concurrency > 0, "calibration batch concurrency must be > 0");
    let request = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "1".repeat(prompt_units)}],
        "max_tokens": 1,
        "seed": 33,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": false,
    });

    let batch_started_at = Instant::now();
    let responses = std::iter::repeat_n(request, concurrency).map(|request| {
        send_completion_request(
            http_client,
            upstream_http_base_url,
            timeout,
            request,
            "calibration",
        )
    });
    for response in join_all(responses).await {
        response?;
    }
    let aggregate_input_tps =
        aggregate_input_tps(prompt_units, concurrency, batch_started_at.elapsed());
    Ok(vec![aggregate_input_tps; concurrency])
}
