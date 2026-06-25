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
use super::{BringupTaskConfig, ModelBringupState};

const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum BringupLifecycleState {
    #[default]
    Initializing,
    Active,
    Recovering,
}

impl BringupLifecycleState {
    pub(super) fn next_action(self) -> BringupLifecycleAction {
        match self {
            Self::Recovering => BringupLifecycleAction::RunRecoveryCanary,
            Self::Active => BringupLifecycleAction::AdvertiseActive,
            Self::Initializing => BringupLifecycleAction::AdvertiseInitialActive,
        }
    }

    pub(super) fn complete_initial_bringup(&mut self) {
        match self {
            Self::Initializing => *self = Self::Active,
            Self::Active => panic!("initial bringup completed after model was already active"),
            Self::Recovering => panic!("initial bringup completed while recovery was pending"),
        }
    }

    pub(super) fn require_recovery_canary(&mut self) {
        match self {
            Self::Active => *self = Self::Recovering,
            Self::Recovering => {}
            Self::Initializing => panic!("recovery canary requested before initial bringup"),
        }
    }

    pub(super) fn complete_recovery_canary(&mut self) {
        match self {
            Self::Recovering => *self = Self::Active,
            Self::Initializing => panic!("recovery canary completed before initial bringup"),
            Self::Active => panic!("recovery canary completed while model was already active"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BringupLifecycleAction {
    RunRecoveryCanary,
    AdvertiseActive,
    AdvertiseInitialActive,
}

enum ActiveMonitorOutcome {
    Stop,
    CanaryFailed(ModelBringupState),
}

pub(crate) fn start_bringup_supervisor(
    parent_stop: &CancellationToken,
    task_configs: Vec<BringupTaskConfig>,
    runtime_state: PylonRuntimeState,
) -> OwnedTask {
    OwnedTask::spawn_child("bringup supervisor", parent_stop, move |stop| async move {
        let mut tasks = Vec::new();
        for task_config in task_configs {
            if !task_config.config.enabled {
                publish_state(
                    &runtime_state,
                    &task_config.model_id,
                    ModelBringupState::AdvertisingActive,
                );
                continue;
            }
            let runtime_state = runtime_state.clone();
            let task = OwnedTask::spawn_child("model bringup", &stop, move |model_stop| {
                run_bringup_task(task_config, runtime_state, model_stop)
            });
            tasks.push(task);
        }

        stop.cancelled().await;
        OwnedTask::shutdown_all(tasks, TASK_SHUTDOWN_TIMEOUT).await;
    })
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
        metrics: _,
    } = task_config;
    let http_client = reqwest::Client::new();
    let mut lifecycle = BringupLifecycleState::default();

    assert!(
        config.enabled,
        "disabled bringup is applied by the supervisor"
    );

    loop {
        if stop.is_cancelled() {
            return;
        }

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
            publish_state(
                &runtime_state,
                &model_id,
                ModelBringupState::ConnectingUnavailable,
            );
            if wait_or_stop(&stop, CONNECT_RETRY_INTERVAL).await {
                return;
            }
            continue;
        }

        match lifecycle.next_action() {
            BringupLifecycleAction::RunRecoveryCanary => {
                publish_state(&runtime_state, &model_id, ModelBringupState::Recovering);
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
                match canary_result {
                    Ok(()) => {
                        lifecycle.complete_recovery_canary();
                        continue;
                    }
                    Err(error) => {
                        tracing::warn!(model_id, error = %error, "bringup recovery canary failed");
                        if wait_or_stop(&stop, CONNECT_RETRY_INTERVAL).await {
                            return;
                        }
                        continue;
                    }
                }
            }
            BringupLifecycleAction::AdvertiseActive => {
                publish_state(
                    &runtime_state,
                    &model_id,
                    ModelBringupState::AdvertisingActive,
                );
            }
            BringupLifecycleAction::AdvertiseInitialActive => {
                publish_state(
                    &runtime_state,
                    &model_id,
                    ModelBringupState::AdvertisingActive,
                );
                lifecycle.complete_initial_bringup();
            }
        }

        match monitor_active_model(
            &http_client,
            &upstream_http_base_url,
            &model_id,
            config.active_canary_interval,
            config.canary_timeout,
            config.canary_max_generation_threshold,
            &stop,
        )
        .await
        {
            ActiveMonitorOutcome::Stop => return,
            ActiveMonitorOutcome::CanaryFailed(next_state) => {
                lifecycle.require_recovery_canary();
                publish_state(&runtime_state, &model_id, next_state);
            }
        }
    }
}

async fn monitor_active_model(
    http_client: &reqwest::Client,
    upstream_http_base_url: &str,
    model_id: &str,
    active_canary_interval: Duration,
    canary_timeout: Duration,
    canary_max_generation_threshold: u32,
    stop: &CancellationToken,
) -> ActiveMonitorOutcome {
    if active_canary_interval.is_zero() {
        stop.cancelled().await;
        return ActiveMonitorOutcome::Stop;
    }

    let mut canary_interval = tokio::time::interval(active_canary_interval);
    canary_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    canary_interval.tick().await;

    loop {
        tokio::select! {
            _ = stop.cancelled() => return ActiveMonitorOutcome::Stop,
            _ = canary_interval.tick() => {
                let Some(canary_result) = stop
                    .run_until_cancelled(send_canary_request(
                        http_client,
                        upstream_http_base_url,
                        model_id,
                        canary_timeout,
                        canary_max_generation_threshold,
                    ))
                    .await
                else {
                    return ActiveMonitorOutcome::Stop;
                };
                if let Err(error) = canary_result {
                    tracing::warn!(model_id, error = %error, "active canary failed");
                    let Some(upstream_healthy) = stop
                        .run_until_cancelled(check_upstream_health(
                            http_client,
                            upstream_http_base_url,
                            canary_timeout,
                        ))
                        .await
                    else {
                        return ActiveMonitorOutcome::Stop;
                    };
                    let next_state = if upstream_healthy {
                        ModelBringupState::Recovering
                    } else {
                        ModelBringupState::ConnectingUnavailable
                    };
                    return ActiveMonitorOutcome::CanaryFailed(next_state);
                }
            }
        }
    }
}

fn publish_state(runtime_state: &PylonRuntimeState, model_id: &str, state: ModelBringupState) {
    runtime_state.set_model_bringup(model_id, state);
}

pub(super) async fn wait_or_stop(stop: &CancellationToken, duration: Duration) -> bool {
    stop.run_until_cancelled(tokio::time::sleep(duration))
        .await
        .is_none()
}
