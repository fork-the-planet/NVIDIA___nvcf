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

use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::routing_state::RegistrationGeneration;
use crate::tunnel::QuicHttpProxy;

const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const HEALTH_CHECK_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

pub(super) struct HealthCheckHandle {
    quic_proxy: Arc<QuicHttpProxy>,
    registration: Arc<RegistrationGeneration>,
    stop: CancellationToken,
    task: HealthCheckTask,
    rx: watch::Receiver<HealthCheckStatus>,
}

impl HealthCheckHandle {
    pub(super) fn start(
        quic_proxy: Arc<QuicHttpProxy>,
        registration: Arc<RegistrationGeneration>,
    ) -> Self {
        let stop = CancellationToken::new();
        let task_stop = stop.clone();
        let task_quic_proxy = quic_proxy.clone();
        let task_registration = registration.clone();
        let (tx, rx) = watch::channel(HealthCheckStatus::Pending);
        let task = HealthCheckTask::new(tokio::spawn(async move {
            loop {
                let status = tokio::select! {
                    _ = task_stop.cancelled() => break,
                    status = sample_health_check_status(&task_quic_proxy, &task_registration) => status,
                };
                let _ = tx.send_replace(status);
                tokio::select! {
                    _ = task_stop.cancelled() => break,
                    _ = tokio::time::sleep(HEALTH_CHECK_INTERVAL) => {}
                }
            }
        }));

        Self {
            quic_proxy,
            registration,
            stop,
            task,
            rx,
        }
    }

    pub(super) async fn shutdown(mut self) {
        self.stop.cancel();
        self.task.shutdown(HEALTH_CHECK_SHUTDOWN_TIMEOUT).await;
    }

    pub(super) async fn latest_ready_rtt_or_probe(&mut self) -> Option<Duration> {
        match self.latest_status() {
            HealthCheckStatus::Ready(rtt) => Some(rtt),
            HealthCheckStatus::Pending => self
                .quic_proxy
                .health_check_rtt(&self.registration)
                .await
                .ok(),
        }
    }

    fn latest_status(&mut self) -> HealthCheckStatus {
        self.rx.borrow_and_update().clone()
    }

    #[cfg(test)]
    pub(super) async fn changed(&mut self) -> Result<(), watch::error::RecvError> {
        self.rx.changed().await
    }
}

struct HealthCheckTask {
    task: Option<JoinHandle<()>>,
}

impl HealthCheckTask {
    fn new(task: JoinHandle<()>) -> Self {
        Self { task: Some(task) }
    }

    fn abort(&self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }

    async fn join(&mut self) -> Result<(), tokio::task::JoinError> {
        self.task
            .as_mut()
            .expect("health-check task should not be disarmed before join")
            .await
    }

    fn disarm(&mut self) {
        let _completed = self.task.take();
    }

    async fn shutdown(&mut self, timeout: Duration) {
        match tokio::time::timeout(timeout, self.join()).await {
            Ok(result) => {
                self.disarm();
                finish_health_check_task(result);
            }
            Err(_) => {
                warn!(
                    timeout_ms = timeout.as_millis(),
                    "health-check task did not stop before shutdown timeout"
                );
                // Cooperative shutdown missed the timeout; abort is the final fallback.
                self.abort();
                let result = self.join().await;
                self.disarm();
                finish_health_check_task(result);
            }
        }
    }
}

impl Drop for HealthCheckTask {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            // A registration processor or cleanup future can be cancelled; abort
            // before dropping the join handle so the task is not detached.
            task.abort();
        }
    }
}

fn finish_health_check_task(result: Result<(), tokio::task::JoinError>) {
    match result {
        Ok(()) => {}
        Err(error) if error.is_cancelled() => {}
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => warn!(error = %error, "health-check task join failed"),
    }
}

#[derive(Clone, Debug)]
enum HealthCheckStatus {
    Pending,
    Ready(Duration),
}

async fn sample_health_check_status(
    quic_proxy: &QuicHttpProxy,
    registration: &Arc<RegistrationGeneration>,
) -> HealthCheckStatus {
    if quic_proxy.has_healthy_connection(registration) {
        match quic_proxy.health_check_rtt(registration).await {
            Ok(rtt) => HealthCheckStatus::Ready(rtt),
            Err(error) => {
                warn!(
                    inference_server_id = %registration.inference_server_id(),
                    error = %error,
                    "health check failed"
                );
                HealthCheckStatus::Pending
            }
        }
    } else {
        HealthCheckStatus::Pending
    }
}

#[cfg(test)]
mod tests {
    use futures::future::pending;
    use tokio::sync::oneshot;

    use super::*;

    struct DropSignal(Option<oneshot::Sender<()>>);

    impl Drop for DropSignal {
        fn drop(&mut self) {
            if let Some(tx) = self.0.take() {
                let _ = tx.send(());
            }
        }
    }

    #[tokio::test]
    async fn dropped_health_check_task_aborts_child() {
        let (started_tx, started_rx) = oneshot::channel();
        let (dropped_tx, dropped_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let _drop_signal = DropSignal(Some(dropped_tx));
            let _ = started_tx.send(());
            pending::<()>().await;
        });
        let task = HealthCheckTask::new(task);
        started_rx
            .await
            .expect("health-check child should report startup");

        // Drop the sole task owner to exercise cancellation-safe cleanup.
        drop(task);

        tokio::time::timeout(Duration::from_secs(1), dropped_rx)
            .await
            .expect("aborted health-check child should be dropped")
            .expect("drop signal sender should remain alive until child cleanup");
    }
}
