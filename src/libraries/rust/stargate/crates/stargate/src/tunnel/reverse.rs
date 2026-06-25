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

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use quinn::{Connection, Endpoint, EndpointConfig};
use tokio_util::task::TaskTracker;
use tracing::{Instrument, info, info_span, warn};

use stargate_forwarding::{self as forwarding, ForwardingResolver, PeerResolution};
use stargate_protocol::TunnelTransportProtocol;

use crate::routing_state::{RegistrationGeneration, StargateState};
use stargate_runtime::CriticalTaskGroup;

use super::QuicHttpProxy;
use super::connection::TunnelConnection;
use super::custom::CustomConnectionHandle;
use super::endpoint::{build_client_config, build_server_config};
use super::http3::build_h3_client_connection;

mod handshake;
mod webtransport;

use self::handshake::handle_reverse_handshake;
use self::webtransport::handle_reverse_webtransport_connect;

#[derive(Clone, Copy)]
pub(super) enum ReverseConnectionCleanupKind {
    Tunnel,
    WebTransport,
}

pub(super) fn spawn_reverse_connection_cleanup(
    task_tracker: &TaskTracker,
    registration: Arc<RegistrationGeneration>,
    connection: Connection,
    kind: ReverseConnectionCleanupKind,
) {
    let inference_server_id = registration.inference_server_id().to_string();
    let closed_id = connection.stable_id();
    let cleanup_span = match kind {
        ReverseConnectionCleanupKind::Tunnel => info_span!(
            "reverse_tunnel_connection_cleanup",
            inference_server_id = %inference_server_id,
            stable_id = closed_id,
        ),
        ReverseConnectionCleanupKind::WebTransport => info_span!(
            "reverse_webtransport_connection_cleanup",
            inference_server_id = %inference_server_id,
            stable_id = closed_id,
        ),
    };
    task_tracker.spawn(
        async move {
            connection.closed().await;
            if registration
                .tunnel_connections()
                .remove_connection(closed_id)
            {
                match kind {
                    ReverseConnectionCleanupKind::Tunnel => {
                        warn!(inference_server_id = %inference_server_id, "reverse tunnel connection closed, removed from pool");
                    }
                    ReverseConnectionCleanupKind::WebTransport => {
                        warn!(inference_server_id = %inference_server_id, "reverse WebTransport connection closed, removed from pool");
                    }
                }
            }
        }
        .instrument(cleanup_span),
    );
}

impl QuicHttpProxy {
    pub async fn await_reverse_connection(
        &self,
        registration: Arc<RegistrationGeneration>,
        timeout: Duration,
    ) -> bool {
        registration
            .tunnel_connections()
            .wait_for_healthy(timeout)
            .await
    }

    pub async fn store_reverse_connection(
        &self,
        registration: Arc<RegistrationGeneration>,
        connection: Connection,
    ) -> bool {
        let Some(install) = registration.begin_reverse_connection_install() else {
            return false;
        };

        let tunnel_connection = match self.build_reverse_tunnel_connection(connection).await {
            Ok(connection) => connection,
            Err(error) => {
                warn!(
                    inference_server_id = %registration.inference_server_id(),
                    error = %error,
                    "failed to initialize tunnel connection"
                );
                return false;
            }
        };
        install.finish(tunnel_connection)
    }

    async fn build_reverse_tunnel_connection(
        &self,
        connection: Connection,
    ) -> Result<TunnelConnection> {
        match self.config.tunnel_protocol {
            TunnelTransportProtocol::Custom => Ok(TunnelConnection::Custom(
                CustomConnectionHandle::new(connection),
            )),
            TunnelTransportProtocol::Http3 => Ok(TunnelConnection::Http3(
                build_h3_client_connection(connection).await?,
            )),
            TunnelTransportProtocol::WebTransport => {
                bail!("reverse WebTransport connections are established by CONNECT handshake")
            }
        }
    }

    pub async fn start_reverse_listener(
        self: &Arc<Self>,
        listen_addr: SocketAddr,
        state: Arc<StargateState>,
        tasks: CriticalTaskGroup,
        forwarding: Option<Arc<dyn ForwardingResolver>>,
        pre_bound_socket: Option<std::net::UdpSocket>,
    ) -> Result<SocketAddr> {
        let server_config = build_server_config(
            &self.config.server_tls_identity,
            self.config.tunnel_protocol,
        )?;
        let endpoint = match pre_bound_socket {
            Some(socket) => {
                socket
                    .set_nonblocking(true)
                    .context("set reverse listener socket to non-blocking")?;
                let runtime =
                    quinn::default_runtime().context("no async runtime for quinn endpoint")?;
                Endpoint::new(
                    EndpointConfig::default(),
                    Some(server_config),
                    socket,
                    runtime,
                )
                .context("create reverse listener from pre-bound socket")?
            }
            None => {
                Endpoint::server(server_config, listen_addr).context("bind reverse listener")?
            }
        };
        let bound_addr = endpoint
            .local_addr()
            .context("reverse listener local addr")?;

        let relay_client_config = build_client_config(
            self.config.tls_cert_pem.as_deref(),
            self.config.quic_insecure,
            self.config.tunnel_protocol,
        )?;
        let relay_endpoints = Arc::new(
            forwarding::build_relay_endpoints(
                forwarding::RelayEndpointConfig::default(),
                relay_client_config,
            )
            .context("build relay endpoints")?,
        );

        let proxy = self.clone();
        let listener_tasks = tasks.task_tracker();
        let listener_span = info_span!("reverse_tunnel_listener", addr = %bound_addr);
        tasks.spawn_critical("reverse tunnel listener", move |stop| {
            async move {
                loop {
                    tokio::select! {
                        _ = stop.cancelled() => break,
                        incoming = endpoint.accept() => {
                            let Some(incoming) = incoming else { break };
                            let proxy = proxy.clone();
                            let state = state.clone();
                            let forwarding = forwarding.clone();
                            let relay_endpoints = relay_endpoints.clone();
                            let port = bound_addr.port();
                            let peer_connect_timeout = proxy.config.connect_timeout;
                            let connection_tasks = listener_tasks.clone();
                            let connection_span = info_span!("reverse_tunnel_connection", port);
                            listener_tasks.spawn(async move {
                                let dispatch = ReverseDispatchContext {
                                    proxy: &proxy,
                                    state: &state,
                                    forwarding: forwarding.as_deref(),
                                    relay_endpoints: &relay_endpoints,
                                    listen_port: port,
                                    peer_connect_timeout,
                                    task_tracker: &connection_tasks,
                                };
                                if let Err(e) = dispatch_incoming(incoming, dispatch).await {
                                    warn!(error = %e, "reverse tunnel connection failed");
                                }
                            }.instrument(connection_span));
                        }
                    }
                }
                endpoint.close(0u32.into(), b"shutdown");
                Ok(())
            }
            .instrument(listener_span)
        });

        info!(addr = %bound_addr, "reverse tunnel listener started");
        Ok(bound_addr)
    }
}

struct ReverseDispatchContext<'a> {
    proxy: &'a QuicHttpProxy,
    state: &'a StargateState,
    forwarding: Option<&'a dyn ForwardingResolver>,
    relay_endpoints: &'a forwarding::RelayEndpoints,
    listen_port: u16,
    peer_connect_timeout: Duration,
    task_tracker: &'a TaskTracker,
}

async fn dispatch_incoming(
    incoming: quinn::Incoming,
    dispatch: ReverseDispatchContext<'_>,
) -> Result<()> {
    let connection = incoming.await.context("accept reverse connection")?;

    if let Some(fwd) = dispatch.forwarding {
        let sni = connection
            .handshake_data()
            .and_then(|data| data.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
            .and_then(|hd| hd.server_name);

        if let Some(sni) = sni {
            match fwd.resolve_peer(&sni, dispatch.listen_port) {
                PeerResolution::Peer(peer) => {
                    info!(
                        peer = %peer.dial_addr,
                        server_name = %peer.server_name,
                        sni = %sni,
                        "relaying QUIC connection to peer"
                    );
                    return forwarding::forward_quic_connection(
                        connection,
                        &peer,
                        dispatch.relay_endpoints,
                        dispatch.peer_connect_timeout,
                    )
                    .await;
                }
                PeerResolution::Local | PeerResolution::NotPeer => {}
            }
        }
    }

    match dispatch.proxy.config.tunnel_protocol {
        TunnelTransportProtocol::WebTransport => {
            handle_reverse_webtransport_connect(
                connection,
                dispatch.proxy,
                dispatch.state,
                dispatch.task_tracker,
            )
            .await
        }
        TunnelTransportProtocol::Custom | TunnelTransportProtocol::Http3 => {
            handle_reverse_handshake(
                connection,
                dispatch.proxy,
                dispatch.state,
                dispatch.task_tracker,
            )
            .await
        }
    }
}
