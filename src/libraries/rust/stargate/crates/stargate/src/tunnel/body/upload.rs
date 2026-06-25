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

use std::future::Future;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::http::StatusCode;
use tracing::{error, warn};

pub(in crate::tunnel) struct RequestBodySendTask {
    label: &'static str,
    completion_timeout: Duration,
    handle: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl RequestBodySendTask {
    pub(in crate::tunnel) fn new(
        label: &'static str,
        completion_timeout: Duration,
        handle: tokio::task::JoinHandle<Result<()>>,
    ) -> Self {
        Self {
            label,
            completion_timeout,
            handle: Some(handle),
        }
    }

    pub(in crate::tunnel) async fn finish(mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        let mut handle = AbortOnDropRequestBodySendHandle::new(handle);

        match tokio::time::timeout(self.completion_timeout, handle.join()).await {
            Ok(result) => {
                handle.disarm();
                finish_request_body_send_result(self.label, result)
            }
            Err(_) => {
                warn!(
                    task = self.label,
                    timeout_ms = self.completion_timeout.as_millis(),
                    "request body send task did not finish before response EOF timeout"
                );
                // The upstream response is already complete; abort the upload
                // producer so response finalization cannot stall forever.
                handle.abort();
                let result = handle.join().await;
                handle.disarm();
                finish_timed_out_request_body_send(self.label, result)
            }
        }
    }

    pub(super) fn abort(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct AbortOnDropRequestBodySendHandle {
    handle: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl AbortOnDropRequestBodySendHandle {
    fn new(handle: tokio::task::JoinHandle<Result<()>>) -> Self {
        Self {
            handle: Some(handle),
        }
    }

    fn abort(&self) {
        if let Some(handle) = &self.handle {
            handle.abort();
        }
    }

    async fn join(&mut self) -> std::result::Result<Result<()>, tokio::task::JoinError> {
        self.handle
            .as_mut()
            .expect("request body send handle should not be disarmed before join")
            .await
    }

    fn disarm(&mut self) {
        let _completed = self.handle.take();
    }
}

impl Drop for AbortOnDropRequestBodySendHandle {
    fn drop(&mut self) {
        if let Some(handle) = &self.handle {
            // Response EOF finalization can be cancelled by downstream disconnect;
            // abort before dropping the handle so the upload task is not detached.
            handle.abort();
        }
    }
}

fn finish_request_body_send_result(
    label: &'static str,
    result: std::result::Result<Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    match result.with_context(|| format!("failed to join {label} send task"))? {
        Ok(()) => Ok(()),
        Err(error) => Err(error.context(format!("failed to send {label}"))),
    }
}

fn finish_timed_out_request_body_send(
    label: &'static str,
    result: std::result::Result<Result<()>, tokio::task::JoinError>,
) -> Result<()> {
    match result {
        Ok(result) => match result {
            Ok(()) => Ok(()),
            Err(error) => Err(error.context(format!("failed to send {label}"))),
        },
        Err(error) if error.is_cancelled() => Ok(()),
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(error).with_context(|| format!("failed to join {label} send task")),
    }
}

impl Drop for RequestBodySendTask {
    fn drop(&mut self) {
        // If callers drop the response before EOF, stop the producer so it
        // cannot keep reading user body bytes after the response is abandoned.
        self.abort();
    }
}

#[derive(Clone, Copy)]
pub(super) enum ResponseHeadRaceBias {
    // Custom QUIC can see a peer reset and body-producer error become ready
    // together; prefer the local body error because it is more actionable.
    SendFirst,
    // HTTP/3 and WebTransport can receive an early server response while the
    // upload is still active; preserve that response before upload errors.
    ResponseFirst,
}

pub(super) struct ResponseHeadRaceConfig {
    pub(super) upload_label: &'static str,
    pub(super) upload_panic_context: &'static str,
    pub(super) upload_error_context: &'static str,
    pub(super) response_header_timeout: Duration,
    pub(super) bias: ResponseHeadRaceBias,
}

pub(super) struct ResponseHeadRaceOutcome<Head> {
    pub(super) head: Head,
    send_done: bool,
    send_task: tokio::task::JoinHandle<Result<()>>,
    upload_label: &'static str,
    response_header_timeout: Duration,
}

impl<Head> ResponseHeadRaceOutcome<Head> {
    pub(super) fn request_body_send_task_if_success(
        self,
        status: StatusCode,
    ) -> (Head, Option<RequestBodySendTask>) {
        let Self {
            head,
            send_done,
            send_task,
            upload_label,
            response_header_timeout,
        } = self;
        let request_body_send_task = if status.is_success() && !send_done {
            Some(RequestBodySendTask::new(
                upload_label,
                response_header_timeout,
                send_task,
            ))
        } else {
            if !send_done {
                send_task.abort();
            }
            None
        };
        (head, request_body_send_task)
    }
}

pub(super) async fn race_request_body_and_response_head<Head, SendBodyFuture, RecvHeadFuture>(
    config: ResponseHeadRaceConfig,
    body: Body,
    send_body: impl FnOnce(Body) -> SendBodyFuture + Send + 'static,
    recv_head: impl FnOnce(tokio::time::Instant) -> RecvHeadFuture,
) -> Result<ResponseHeadRaceOutcome<Head>>
where
    SendBodyFuture: Future<Output = Result<()>> + Send + 'static,
    RecvHeadFuture: Future<Output = Result<Head>> + Send,
{
    let response_header_deadline = tokio::time::Instant::now() + config.response_header_timeout;
    let upload_label = config.upload_label;
    let response_header_timeout = config.response_header_timeout;
    let mut send_task = tokio::spawn(async move {
        let result = send_body(body).await;
        if let Err(error) = &result {
            error!(error = %error, upload_label, "failed to send request body");
        }
        result
    });
    let mut send_done = false;
    let response_head = recv_head(response_header_deadline);
    tokio::pin!(response_head);

    let head = match config.bias {
        ResponseHeadRaceBias::SendFirst => {
            tokio::select! {
                biased;
                send_result = &mut send_task => {
                    send_done = true;
                    match send_result.context(config.upload_panic_context)? {
                        Ok(()) => response_head.await?,
                        Err(error) => return Err(error.context(config.upload_error_context)),
                    }
                },
                response_head = &mut response_head => match response_head {
                    Ok(response_head) => response_head,
                    Err(error) => {
                        send_task.abort();
                        return Err(error);
                    }
                },
            }
        }
        ResponseHeadRaceBias::ResponseFirst => {
            tokio::select! {
                biased;
                response_head = &mut response_head => match response_head {
                    Ok(response_head) => response_head,
                    Err(error) => {
                        send_task.abort();
                        return Err(error);
                    }
                },
                send_result = &mut send_task => {
                    send_done = true;
                    match send_result.context(config.upload_panic_context)? {
                        Ok(()) => response_head.await?,
                        Err(error) => return Err(error.context(config.upload_error_context)),
                    }
                },
            }
        }
    };

    Ok(ResponseHeadRaceOutcome {
        head,
        send_done,
        send_task,
        upload_label,
        response_header_timeout,
    })
}
