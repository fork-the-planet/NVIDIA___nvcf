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

use tokio_util::sync::CancellationToken;

use crate::runtime_state::PylonRuntimeState;
use stargate_runtime::{OwnedTask, TASK_SHUTDOWN_TIMEOUT};

use super::upstream::{check_upstream_health, send_canary_request};
use super::{BringupConfig, BringupError, BringupTaskConfig};

const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(1);

pub struct BringupHandle {
    task: OwnedTask,
}

impl BringupHandle {
    pub async fn wait_for_exit(&mut self) -> Result<(), tokio::task::JoinError> {
        self.task.wait_for_exit().await
    }

    pub async fn shutdown(self) {
        self.task.shutdown(TASK_SHUTDOWN_TIMEOUT).await;
    }
}

pub async fn start_bringup(
    upstream_http_base_url: &str,
    config: BringupConfig,
    runtime_state: PylonRuntimeState,
) -> Result<Option<BringupHandle>, BringupError> {
    if !config.enabled {
        runtime_state.mark_initial_bringup_complete();
        return Ok(None);
    }

    if !check_upstream_health(
        &reqwest::Client::new(),
        upstream_http_base_url,
        config.canary_timeout,
    )
    .await
    {
        return Err(BringupError::UnhealthyUpstream);
    }

    runtime_state.mark_initial_bringup_complete();

    let task_configs = runtime_state
        .model_ids()
        .into_iter()
        .map(|model_id| BringupTaskConfig {
            upstream_http_base_url: upstream_http_base_url.to_string(),
            model_id,
            config: config.clone(),
        })
        .collect();
    let task = OwnedTask::spawn("bringup supervisor", move |stop| async move {
        run_bringup_supervisor(task_configs, runtime_state, stop).await;
    });
    Ok(Some(BringupHandle { task }))
}

async fn run_bringup_supervisor(
    task_configs: Vec<BringupTaskConfig>,
    runtime_state: PylonRuntimeState,
    stop: CancellationToken,
) {
    let mut tasks = Vec::with_capacity(task_configs.len());
    for task_config in task_configs {
        let runtime_state = runtime_state.clone();
        tasks.push(OwnedTask::spawn_child(
            "model bringup",
            &stop,
            move |model_stop| run_bringup_task(task_config, runtime_state, model_stop),
        ));
    }

    stop.cancelled().await;
    OwnedTask::shutdown_all(tasks, TASK_SHUTDOWN_TIMEOUT).await;
}

pub(super) async fn run_bringup_task(
    task_config: BringupTaskConfig,
    runtime_state: PylonRuntimeState,
    stop: CancellationToken,
) {
    let BringupTaskConfig {
        upstream_http_base_url,
        model_id,
        config,
    } = task_config;
    let http_client = reqwest::Client::new();

    loop {
        if !wait_for_active_canary_failure(
            &http_client,
            &upstream_http_base_url,
            &model_id,
            &config,
            &stop,
        )
        .await
        {
            return;
        }
        runtime_state.set_model_bringup_ready(&model_id, false);

        loop {
            let Some(upstream_healthy) = stop
                .run_until_cancelled(check_upstream_health(
                    &http_client,
                    &upstream_http_base_url,
                    config.canary_timeout,
                ))
                .await
            else {
                return;
            };
            if !upstream_healthy {
                if wait_or_stop(&stop, CONNECT_RETRY_INTERVAL).await {
                    return;
                }
                continue;
            }

            let Some(canary_result) = stop
                .run_until_cancelled(send_canary_request(
                    &http_client,
                    &upstream_http_base_url,
                    &model_id,
                    config.canary_timeout,
                    config.canary_max_generation_threshold,
                ))
                .await
            else {
                return;
            };
            if let Err(error) = canary_result {
                tracing::warn!(model_id, error = %error, "bringup recovery canary failed");
                if wait_or_stop(&stop, CONNECT_RETRY_INTERVAL).await {
                    return;
                }
                continue;
            }

            runtime_state.set_model_bringup_ready(&model_id, true);
            break;
        }
    }
}

async fn wait_for_active_canary_failure(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    config: &BringupConfig,
    stop: &CancellationToken,
) -> bool {
    if config.active_canary_interval.is_zero() {
        stop.cancelled().await;
        return false;
    }

    let mut canary_interval = tokio::time::interval(config.active_canary_interval);
    canary_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    canary_interval.tick().await;

    loop {
        tokio::select! {
            _ = stop.cancelled() => return false,
            _ = canary_interval.tick() => {
                let Some(canary_result) = stop
                    .run_until_cancelled(send_canary_request(
                        http_client,
                        upstream_http_base_url,
                        model_id,
                        config.canary_timeout,
                        config.canary_max_generation_threshold,
                    ))
                    .await
                else {
                    return false;
                };
                if let Err(error) = canary_result {
                    tracing::warn!(model_id, error = %error, "active canary failed");
                    return true;
                }
            }
        }
    }
}

pub(super) async fn wait_or_stop(stop: &CancellationToken, duration: Duration) -> bool {
    stop.run_until_cancelled(tokio::time::sleep(duration))
        .await
        .is_none()
}
