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

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::bringup::{self, start_bringup_supervisor};
use crate::runtime_state::PylonRuntimeState;
use stargate_runtime::{OwnedTask, TASK_SHUTDOWN_TIMEOUT};

use super::calibration::{
    ClusterCalibrationDirective, ClusterCalibrationExecutorTaskConfig,
    run_cluster_calibration_executor,
};
use super::discovery::run_watch_stargate_discovery;
use super::grpc_endpoint::StargateGrpcEndpoint;
use super::router_stream::{RouterRegistrationTaskTemplate, run_router_registration_stream};
use super::topology::RegistrationRouterTopology;
use super::types::{ClientError, InferenceServerRegistrationConfig, RegistrationStartPlan};

fn spawn_registration_session(
    config: InferenceServerRegistrationConfig,
    start_plan: RegistrationStartPlan,
) -> OwnedTask {
    OwnedTask::spawn("registration session", move |stop| {
        run_registration_session(config, start_plan, stop)
    })
}

async fn run_registration_session(
    config: InferenceServerRegistrationConfig,
    start_plan: RegistrationStartPlan,
    stop: CancellationToken,
) {
    let runtime_state = config.runtime_state.clone();
    let model_ids = runtime_state.model_ids();
    let (cluster_calibration_directive_tx, cluster_calibration_directive_rx) =
        flume::bounded::<ClusterCalibrationDirective>(256);

    let (router_topology_tx, router_topology_rx) =
        watch::channel(RegistrationRouterTopology::default());
    let watch_seeds = start_plan.watch_seeds.clone();
    let watch_task = OwnedTask::spawn_child("watch stargate discovery", &stop, move |watch_stop| {
        run_watch_stargate_discovery(watch_seeds, router_topology_tx, watch_stop)
    });

    let task_template = RouterRegistrationTaskTemplate::from_registration_config(
        &config,
        &start_plan.cluster_id,
        &start_plan.upstream_http_base_url,
        cluster_calibration_directive_tx,
    );
    let coordinated_calibration = task_template.coordinated_calibration;

    let bringup_task = start_bringup_supervisor(
        &stop,
        model_ids
            .iter()
            .cloned()
            .map(|model_id| bringup::BringupTaskConfig {
                upstream_http_base_url: start_plan.upstream_http_base_url.clone(),
                model_id,
                config: config.bringup.clone(),
                metrics: config.metrics.clone(),
            })
            .collect(),
        runtime_state.clone(),
    );
    let calibration_topology_rx = router_topology_rx.clone();
    let calibration_task = coordinated_calibration.then(|| {
        OwnedTask::spawn_child("cluster calibration executor", &stop, move |cancel_token| {
            run_cluster_calibration_executor(
                ClusterCalibrationExecutorTaskConfig {
                    inference_server_id: config.inference_server_id.clone(),
                    cluster_id: start_plan.cluster_id.clone(),
                    retry_interval: config.min_update_interval,
                    upstream_http_base_url: start_plan.upstream_http_base_url.clone(),
                    bringup: config.bringup.clone(),
                    metrics: config.metrics.clone(),
                    auth_token_provider: config.auth_token_provider.clone(),
                },
                cluster_calibration_directive_rx,
                calibration_topology_rx,
                cancel_token,
            )
        })
    });

    let registration_task =
        OwnedTask::spawn_child("registration supervisor", &stop, move |registration_stop| {
            run_registration_supervisor(
                task_template,
                runtime_state,
                router_topology_rx,
                registration_stop,
            )
        });

    let mut tasks = Vec::with_capacity(4);
    tasks.push(watch_task);
    tasks.push(bringup_task);
    if let Some(task) = calibration_task {
        tasks.push(task);
    }
    tasks.push(registration_task);

    stop.cancelled().await;
    OwnedTask::shutdown_all(tasks, TASK_SHUTDOWN_TIMEOUT).await;
}

async fn run_registration_supervisor(
    task_template: RouterRegistrationTaskTemplate,
    runtime_state: PylonRuntimeState,
    mut router_topology_rx: watch::Receiver<RegistrationRouterTopology>,
    stop: CancellationToken,
) {
    let mut per_router_tasks: HashMap<StargateGrpcEndpoint, OwnedTask> = HashMap::new();

    loop {
        if stop.is_cancelled() {
            break;
        }

        let desired_routers = router_topology_rx
            .borrow_and_update()
            .published_routers()
            .cloned()
            .unwrap_or_default();
        let current_routers: Vec<StargateGrpcEndpoint> = per_router_tasks.keys().cloned().collect();
        for router in current_routers {
            if desired_routers.contains(&router) {
                continue;
            }
            if let Some(task) = per_router_tasks.remove(&router) {
                task.shutdown(TASK_SHUTDOWN_TIMEOUT).await;
            }
        }

        for router in &desired_routers {
            if per_router_tasks.contains_key(router) {
                continue;
            }

            let task_config = task_template.build_for_router(router.clone());
            let runtime_state = runtime_state.clone();
            let task =
                OwnedTask::spawn_child("router registration stream", &stop, move |worker_stop| {
                    run_router_registration_stream(task_config, runtime_state, worker_stop)
                });
            per_router_tasks.insert(router.clone(), task);
        }

        tokio::select! {
            _ = stop.cancelled() => break,
            changed = router_topology_rx.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }
    }

    OwnedTask::shutdown_all(
        per_router_tasks.into_values().collect(),
        TASK_SHUTDOWN_TIMEOUT,
    )
    .await;
}

#[derive(Default)]
pub struct InferenceServerRegistrationClient {
    running: Option<OwnedTask>,
}

impl InferenceServerRegistrationClient {
    pub fn stop(&mut self) {
        if let Some(running) = self.running.take() {
            running.abort();
        }
    }

    pub async fn shutdown(&mut self) {
        if let Some(running) = self.running.take() {
            running.shutdown(TASK_SHUTDOWN_TIMEOUT).await;
        }
    }

    pub async fn wait_for_exit(&mut self) -> std::result::Result<(), tokio::task::JoinError> {
        let Some(running) = self.running.as_mut() else {
            return std::future::pending().await;
        };
        let result = running.wait_for_exit().await;
        self.running = None;
        result
    }

    pub fn start(&mut self, config: InferenceServerRegistrationConfig) -> Result<(), ClientError> {
        self.stop();
        let start_plan = RegistrationStartPlan::from_config(&config)?;
        self.running = Some(spawn_registration_session(config, start_plan));
        Ok(())
    }
}

impl Drop for InferenceServerRegistrationClient {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use std::future::pending;
    use std::time::Duration;

    use tokio::sync::oneshot;

    use super::*;

    struct DropNotifier(Option<oneshot::Sender<()>>);

    impl Drop for DropNotifier {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    async fn spawn_pending_owned_task() -> (OwnedTask, oneshot::Receiver<()>) {
        let (entered_tx, entered_rx) = oneshot::channel();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let task = OwnedTask::spawn("pending registration session task", |_| async move {
            let _drop_notifier = DropNotifier(Some(dropped_tx));
            let _ = entered_tx.send(());
            pending::<()>().await;
        });
        entered_rx.await.expect("owned task should start");
        (task, dropped_rx)
    }

    #[tokio::test]
    async fn registration_client_drop_aborts_running_session_root() {
        let (running, dropped) = spawn_pending_owned_task().await;
        let client = InferenceServerRegistrationClient {
            running: Some(running),
        };

        // Dropping the client must tear down its one physical session root.
        drop(client);

        tokio::time::timeout(Duration::from_secs(1), dropped)
            .await
            .expect("session root should be aborted when the client drops")
            .expect("session root drop notifier should send");
    }
}
