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

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Result, ensure};
use tokio::sync::watch;

use crate::routing_state::RegistrationGeneration;

use super::body::OpenStreamingRequest;
use super::custom::CustomConnectionHandle;
use super::http3::Http3ConnectionHandle;
use super::request::OpenTunnelRequest;
use super::webtransport::WebTransportConnectionHandle;

#[derive(Clone)]
pub(super) enum TunnelConnection {
    Custom(CustomConnectionHandle),
    Http3(Http3ConnectionHandle),
    WebTransport(WebTransportConnectionHandle),
}

pub(crate) struct RegistrationConnections {
    state: watch::Sender<RegistrationConnectionState>,
}

#[derive(Debug)]
enum RegistrationConnectionState {
    Disconnected,
    Connected(TunnelConnectionSet),
    ReverseInstalling,
    ReverseReplacing(TunnelConnectionSet),
    Retired,
}

enum ConnectionReadiness {
    Pending,
    Healthy,
    Retired,
}

pub(super) struct ReverseConnectionInstall {
    registration: Arc<RegistrationGeneration>,
}

impl TunnelConnection {
    pub(super) fn is_healthy(&self) -> bool {
        match self {
            Self::Custom(handle) => handle.is_healthy(),
            Self::Http3(handle) => handle.is_healthy(),
            Self::WebTransport(handle) => handle.is_healthy(),
        }
    }

    fn stable_id(&self) -> usize {
        match self {
            Self::Custom(handle) => handle.stable_id(),
            Self::Http3(handle) => handle.stable_id(),
            Self::WebTransport(handle) => handle.stable_id(),
        }
    }

    pub(super) async fn open_streaming_request(
        self,
        request: OpenTunnelRequest<'_>,
    ) -> Result<OpenStreamingRequest> {
        match self {
            Self::Custom(handle) => handle.open_streaming_request(request).await,
            Self::Http3(handle) => handle.open_streaming_request(request).await,
            Self::WebTransport(handle) => handle.open_streaming_request(request).await,
        }
    }
}

impl RegistrationConnections {
    pub(crate) fn new() -> Self {
        let (state, _) = watch::channel(RegistrationConnectionState::Disconnected);
        Self { state }
    }

    pub(super) fn retire(&self) -> bool {
        self.state.send_if_modified(|state| {
            if matches!(state, RegistrationConnectionState::Retired) {
                false
            } else {
                *state = RegistrationConnectionState::Retired;
                true
            }
        })
    }

    pub(super) fn is_active(&self) -> bool {
        !matches!(&*self.state.borrow(), RegistrationConnectionState::Retired)
    }

    pub(super) fn connection_set(&self) -> Option<TunnelConnectionSet> {
        match &*self.state.borrow() {
            RegistrationConnectionState::Connected(connections) => Some(connections.clone()),
            RegistrationConnectionState::ReverseReplacing(previous) => Some(previous.clone()),
            RegistrationConnectionState::Disconnected
            | RegistrationConnectionState::ReverseInstalling
            | RegistrationConnectionState::Retired => None,
        }
    }

    pub(super) fn has_healthy_connection(&self) -> bool {
        self.connection_set()
            .is_some_and(|connections| connections.is_healthy())
    }

    pub(super) fn needs_replenishment(&self) -> bool {
        match &*self.state.borrow() {
            RegistrationConnectionState::Disconnected => true,
            RegistrationConnectionState::Connected(connections) => {
                connections.needs_replenishment()
            }
            RegistrationConnectionState::ReverseReplacing(previous) => {
                previous.needs_replenishment()
            }
            RegistrationConnectionState::ReverseInstalling => true,
            RegistrationConnectionState::Retired => false,
        }
    }

    pub(super) fn install_direct(&self, connections: TunnelConnectionSet) -> bool {
        self.state.send_if_modified(move |state| {
            if matches!(state, RegistrationConnectionState::Retired) {
                false
            } else {
                *state = RegistrationConnectionState::Connected(connections);
                true
            }
        })
    }

    fn begin_reverse_install(&self) -> bool {
        self.state.send_if_modified(|state| {
            let installing = match &*state {
                RegistrationConnectionState::Disconnected => {
                    RegistrationConnectionState::ReverseInstalling
                }
                RegistrationConnectionState::Connected(connections)
                    if !connections.is_healthy() =>
                {
                    RegistrationConnectionState::ReverseReplacing(connections.clone())
                }
                RegistrationConnectionState::Connected(_)
                | RegistrationConnectionState::ReverseInstalling
                | RegistrationConnectionState::ReverseReplacing(_)
                | RegistrationConnectionState::Retired => return false,
            };
            *state = installing;
            true
        })
    }

    fn finish_reverse_install(&self, connection: TunnelConnection) -> bool {
        self.state.send_if_modified(move |state| {
            match &*state {
                RegistrationConnectionState::ReverseInstalling => {}
                RegistrationConnectionState::ReverseReplacing(previous)
                    if !previous.is_healthy() => {}
                RegistrationConnectionState::Disconnected
                | RegistrationConnectionState::Connected(_)
                | RegistrationConnectionState::ReverseReplacing(_)
                | RegistrationConnectionState::Retired => return false,
            }
            *state =
                RegistrationConnectionState::Connected(TunnelConnectionSet::single(connection));
            true
        })
    }

    pub(super) fn remove_connection(&self, stable_id: usize) -> bool {
        self.state.send_if_modified(|state| match state {
            RegistrationConnectionState::Connected(connections)
                if connections.contains_stable_id(stable_id) =>
            {
                *state = RegistrationConnectionState::Disconnected;
                true
            }
            RegistrationConnectionState::ReverseReplacing(previous)
                if previous.contains_stable_id(stable_id) =>
            {
                *state = RegistrationConnectionState::ReverseInstalling;
                true
            }
            RegistrationConnectionState::Disconnected
            | RegistrationConnectionState::Connected(_)
            | RegistrationConnectionState::ReverseInstalling
            | RegistrationConnectionState::ReverseReplacing(_)
            | RegistrationConnectionState::Retired => false,
        })
    }

    pub(super) async fn wait_for_healthy(&self, timeout: Duration) -> bool {
        let mut state = self.state.subscribe();
        tokio::time::timeout(timeout, async {
            loop {
                let readiness = state.borrow_and_update().readiness();
                match readiness {
                    ConnectionReadiness::Healthy => return true,
                    ConnectionReadiness::Retired => return false,
                    ConnectionReadiness::Pending => {}
                }
                if state.changed().await.is_err() {
                    return false;
                }
            }
        })
        .await
        .unwrap_or(false)
    }

    fn cancel_reverse_install(&self) {
        self.state.send_if_modified(|state| match &*state {
            RegistrationConnectionState::ReverseInstalling => {
                *state = RegistrationConnectionState::Disconnected;
                true
            }
            RegistrationConnectionState::ReverseReplacing(previous) => {
                *state = RegistrationConnectionState::Connected(previous.clone());
                true
            }
            RegistrationConnectionState::Disconnected
            | RegistrationConnectionState::Connected(_)
            | RegistrationConnectionState::Retired => false,
        });
    }
}

impl RegistrationConnectionState {
    fn readiness(&self) -> ConnectionReadiness {
        match self {
            RegistrationConnectionState::Connected(connections) if connections.is_healthy() => {
                ConnectionReadiness::Healthy
            }
            RegistrationConnectionState::ReverseReplacing(previous) if previous.is_healthy() => {
                ConnectionReadiness::Healthy
            }
            RegistrationConnectionState::Disconnected
            | RegistrationConnectionState::Connected(_)
            | RegistrationConnectionState::ReverseInstalling
            | RegistrationConnectionState::ReverseReplacing(_) => ConnectionReadiness::Pending,
            RegistrationConnectionState::Retired => ConnectionReadiness::Retired,
        }
    }
}

impl fmt::Debug for RegistrationConnections {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegistrationConnections")
            .field("state", &*self.state.borrow())
            .finish()
    }
}

impl RegistrationGeneration {
    pub(super) fn begin_reverse_connection_install(
        self: &Arc<Self>,
    ) -> Option<ReverseConnectionInstall> {
        self.tunnel_connections()
            .begin_reverse_install()
            .then(|| ReverseConnectionInstall {
                registration: Arc::clone(self),
            })
    }
}

impl ReverseConnectionInstall {
    pub(super) fn finish(self, connection: TunnelConnection) -> bool {
        self.registration
            .tunnel_connections()
            .finish_reverse_install(connection)
    }
}

impl Drop for ReverseConnectionInstall {
    fn drop(&mut self) {
        self.registration
            .tunnel_connections()
            .cancel_reverse_install();
    }
}

#[derive(Clone)]
pub(super) struct TunnelConnectionSet {
    inner: Arc<TunnelConnectionSetInner>,
}

struct TunnelConnectionSetInner {
    connections: Vec<TunnelConnection>,
    cursor: AtomicUsize,
}

impl fmt::Debug for TunnelConnectionSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TunnelConnectionSet")
            .field("connection_count", &self.inner.connections.len())
            .field("healthy", &self.is_healthy())
            .finish()
    }
}

impl TunnelConnectionSet {
    pub(super) fn new(connections: Vec<TunnelConnection>) -> Result<Self> {
        ensure!(!connections.is_empty(), "tunnel connection set is empty");
        Ok(Self {
            inner: Arc::new(TunnelConnectionSetInner {
                connections,
                cursor: AtomicUsize::new(0),
            }),
        })
    }

    pub(super) fn single(connection: TunnelConnection) -> Self {
        Self {
            inner: Arc::new(TunnelConnectionSetInner {
                connections: vec![connection],
                cursor: AtomicUsize::new(0),
            }),
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.inner.connections.len()
    }

    #[cfg(test)]
    pub(super) fn connection(&self, index: usize) -> &TunnelConnection {
        &self.inner.connections[index]
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.inner
            .connections
            .iter()
            .any(TunnelConnection::is_healthy)
    }

    pub(super) fn needs_replenishment(&self) -> bool {
        !self
            .inner
            .connections
            .iter()
            .all(TunnelConnection::is_healthy)
    }

    pub(super) fn choose_healthy(&self) -> Option<TunnelConnection> {
        let len = self.inner.connections.len();
        // The cursor only spreads load across equivalent live connections, so
        // relaxed ordering is enough; the health check below owns correctness.
        let start = self.inner.cursor.fetch_add(1, Ordering::Relaxed) % len;
        for offset in 0..len {
            let index = (start + offset) % len;
            let connection = &self.inner.connections[index];
            if connection.is_healthy() {
                return Some(connection.clone());
            }
        }
        None
    }

    pub(super) fn contains_stable_id(&self, stable_id: usize) -> bool {
        self.inner
            .connections
            .iter()
            .any(|connection| connection.stable_id() == stable_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_install_begin_and_cancel_publish_state_changes() {
        let connections = RegistrationConnections::new();
        let mut updates = connections.state.subscribe();

        assert!(connections.begin_reverse_install());
        assert!(
            updates.has_changed().expect("update sender should remain"),
            "reverse install acquisition must publish its lifecycle transition"
        );
        updates.borrow_and_update();

        connections.cancel_reverse_install();
        assert!(
            updates.has_changed().expect("update sender should remain"),
            "reverse install cancellation must publish its lifecycle transition"
        );
    }
}
