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

use futures::future::join_all;

use crate::stats::token_metrics::{SNAPSHOT_THRESHOLD, TpsDistribution};

use super::upstream::{BringupError, check_upstream_health, send_completion_request};
use super::{BringupConfig, BringupTaskConfig};

pub(super) const CALIBRATION_PROMPT_UNITS_FLOOR: usize = 256;

pub(crate) async fn run_assigned_cluster_calibration(
    task_config: &BringupTaskConfig,
) -> Result<f64, BringupError> {
    let client = reqwest::Client::new();
    let started_at = Instant::now();
    let result = if check_upstream_health(
        &client,
        &task_config.upstream_http_base_url,
        task_config.config.canary_timeout,
    )
    .await
    {
        run_calibration(
            &client,
            &task_config.upstream_http_base_url,
            &task_config.model_id,
            &task_config.config,
        )
        .await
    } else {
        Err(BringupError::UnhealthyUpstream)
    };
    if let Some(metrics) = task_config.metrics.as_deref() {
        metrics.observe_model_calibration_duration(
            &task_config.model_id,
            started_at.elapsed(),
            result.is_ok(),
        );
    }
    result
}

pub(super) async fn run_calibration(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    config: &BringupConfig,
) -> Result<f64, BringupError> {
    if config.calibration_requests == 0 {
        return Ok(0.0);
    }

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
            Err(BringupError::PromptTooLong) => {
                last_error = Some(BringupError::PromptTooLong);
            }
            Err(error) => return Err(error),
        }
    }

    let required_samples = config.calibration_requests.min(SNAPSHOT_THRESHOLD);
    if distribution.count >= required_samples {
        Ok(distribution.mean)
    } else if let Some(error) = last_error {
        Err(error)
    } else {
        Err(BringupError::InsufficientCalibrationSamples {
            valid_samples: distribution.count,
        })
    }
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
                let next_prompt_units = ((prompt_units + CALIBRATION_PROMPT_UNITS_FLOOR) / 2)
                    .max(CALIBRATION_PROMPT_UNITS_FLOOR);
                if next_prompt_units >= prompt_units {
                    return Err(BringupError::PromptTooLong);
                }
                prompt_units = next_prompt_units;
            }
            result => return result,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CalibrationBatch {
    pub(super) prompt_units: usize,
    pub(super) concurrency: usize,
}

pub(super) fn calibration_plan(config: &BringupConfig) -> Vec<CalibrationBatch> {
    let requests = config.calibration_requests;
    if requests == 0 {
        return Vec::new();
    }

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

    if max_concurrency == 1 {
        return (0..requests)
            .map(|index| {
                let prompt_units = interpolate_usize(
                    CALIBRATION_PROMPT_UNITS_FLOOR,
                    max_prompt_units,
                    index,
                    requests - 1,
                );
                let concurrency = interpolate_usize(1, max_concurrency, index, requests - 1);
                CalibrationBatch {
                    prompt_units,
                    concurrency,
                }
            })
            .collect();
    }

    let final_concurrency = max_concurrency.min(requests - 1);
    let single_request_runs = requests - final_concurrency;
    let mut batches = Vec::with_capacity(single_request_runs + 1);
    for index in 0..single_request_runs {
        batches.push(CalibrationBatch {
            prompt_units: interpolate_usize(
                CALIBRATION_PROMPT_UNITS_FLOOR,
                max_prompt_units,
                index,
                single_request_runs,
            ),
            concurrency: 1,
        });
    }
    batches.push(CalibrationBatch {
        prompt_units: max_prompt_units,
        concurrency: final_concurrency,
    });

    batches
}

fn interpolate_usize(start: usize, end: usize, index: usize, last_index: usize) -> usize {
    if last_index == 0 {
        return end;
    }
    let span = end - start;
    start + (span * index / last_index)
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
    let prompt = "1".repeat(prompt_units);
    let request = serde_json::json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "seed": 33,
        "temperature": 0.7,
        "top_p": 1.0,
        "stream": false,
    });

    let batch_started_at = Instant::now();
    let requests = (0..concurrency).map(|_| {
        let request = request.clone();
        async move {
            send_completion_request(http_client, upstream_http_base_url, timeout, request).await?;
            Ok::<_, BringupError>(())
        }
    });
    let _: Vec<()> = join_all(requests)
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;
    let aggregate_input_tps =
        aggregate_input_tps(prompt_units, concurrency, batch_started_at.elapsed());
    Ok(vec![aggregate_input_tps; concurrency])
}
