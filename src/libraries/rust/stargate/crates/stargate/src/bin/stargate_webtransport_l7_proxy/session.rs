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
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use bytes::Bytes;
use http::{HeaderMap, Method, Request, Response, StatusCode};
use quinn::{ClientConfig, Endpoint};
use stargate_protocol::TunnelTransportProtocol;
use stargate_protocol::tunnel_contract::WEBTRANSPORT_TUNNEL_PATH;
use tracing::{info, warn};

use super::network::{connect_first_upstream_candidate, resolve_upstream_addrs};

const WEBTRANSPORT_STREAM_HEADER_TIMEOUT: Duration = Duration::from_secs(5);

type DownstreamH3Connection = h3::server::Connection<h3_quinn::Connection, Bytes>;
type DownstreamConnectStream = h3::server::RequestStream<
    <h3_quinn::Connection as h3::quic::OpenStreams<Bytes>>::BidiStream,
    Bytes,
>;
type UpstreamH3Connection = h3::client::Connection<h3_quinn::Connection, Bytes>;
type UpstreamConnectStream = h3::client::RequestStream<
    <h3_quinn::OpenStreams as h3::quic::OpenStreams<Bytes>>::BidiStream,
    Bytes,
>;
type UpstreamSendRequest = h3::client::SendRequest<h3_quinn::OpenStreams, Bytes>;

#[derive(Clone, Debug)]
pub(super) struct UpstreamTlsConfig {
    pub(super) cert_pem: Option<Vec<u8>>,
    pub(super) quic_insecure: bool,
}

impl Default for UpstreamTlsConfig {
    fn default() -> Self {
        Self {
            cert_pem: None,
            quic_insecure: true,
        }
    }
}

pub(super) async fn handle_connection(
    incoming: quinn::Incoming,
    upstream_template: String,
    upstream_tls: UpstreamTlsConfig,
) -> Result<()> {
    let Some(downstream) = DownstreamSession::accept(incoming, &upstream_template).await? else {
        return Ok(());
    };
    let upstream = downstream.connect_upstream(&upstream_tls).await?;
    downstream.forward(upstream).await
}

async fn accept_downstream_quic(incoming: quinn::Incoming) -> Result<quinn::Connection> {
    incoming.await.context("accept downstream QUIC connection")
}

fn downstream_target_for_connection(
    connection: &quinn::Connection,
    upstream_template: &str,
) -> Result<(String, String)> {
    let server_name = downstream_server_name(connection).unwrap_or_else(|| "stargate".to_string());
    let upstream_addr = upstream_addr_for_sni(upstream_template, &server_name)?;
    Ok((server_name, upstream_addr))
}

async fn build_downstream_h3(connection: quinn::Connection) -> Result<DownstreamH3Connection> {
    let mut builder = h3::server::builder();
    builder
        .enable_webtransport(true)
        .enable_extended_connect(true)
        .enable_datagram(true)
        .max_webtransport_sessions(1);
    builder
        .build(h3_quinn::Connection::new(connection))
        .await
        .map_err(|error| anyhow!("create downstream h3 server: {error:?}"))
}

async fn accept_downstream_connect(
    downstream_h3: &mut DownstreamH3Connection,
) -> Result<Option<(Request<()>, DownstreamConnectStream)>> {
    let Some(resolver) = downstream_h3
        .accept()
        .await
        .map_err(|error| anyhow!("accept downstream WebTransport CONNECT: {error:?}"))?
    else {
        return Ok(None);
    };
    resolver
        .resolve_request()
        .await
        .map(Some)
        .map_err(|error| anyhow!("resolve downstream WebTransport CONNECT: {error:?}"))
}

struct DownstreamSession {
    connection: quinn::Connection,
    server_name: String,
    upstream_addr: String,
    request_headers: HeaderMap,
    h3_connection: DownstreamH3Connection,
    connect_stream: DownstreamConnectStream,
}

impl DownstreamSession {
    async fn accept(incoming: quinn::Incoming, upstream_template: &str) -> Result<Option<Self>> {
        let connection = accept_downstream_quic(incoming).await?;
        let (server_name, upstream_addr) =
            downstream_target_for_connection(&connection, upstream_template)?;
        info!(%server_name, %upstream_addr, "accepted downstream WebTransport connection");
        Self::from_connection(connection, server_name, upstream_addr).await
    }

    async fn from_connection(
        connection: quinn::Connection,
        server_name: String,
        upstream_addr: String,
    ) -> Result<Option<Self>> {
        let mut h3_connection = build_downstream_h3(connection.clone()).await?;
        let Some((request, connect_stream)) = accept_downstream_connect(&mut h3_connection).await?
        else {
            return Ok(None);
        };
        Self::from_connect(
            connection,
            server_name,
            upstream_addr,
            h3_connection,
            request,
            connect_stream,
        )
        .map(Some)
    }

    fn from_connect(
        connection: quinn::Connection,
        server_name: String,
        upstream_addr: String,
        h3_connection: DownstreamH3Connection,
        request: Request<()>,
        connect_stream: DownstreamConnectStream,
    ) -> Result<Self> {
        validate_webtransport_connect(&request)?;
        let (parts, _) = request.into_parts();
        Ok(Self {
            connection,
            server_name,
            upstream_addr,
            request_headers: parts.headers,
            h3_connection,
            connect_stream,
        })
    }

    async fn connect_upstream(&self, tls: &UpstreamTlsConfig) -> Result<UpstreamSession> {
        connect_upstream(
            &self.upstream_addr,
            &self.server_name,
            &self.request_headers,
            tls,
        )
        .await
    }

    async fn forward(mut self, upstream: UpstreamSession) -> Result<()> {
        if upstream.response_status.is_success() {
            self.accept_and_bridge(upstream).await
        } else {
            self.reject(upstream.response_status).await
        }
    }

    async fn reject(&mut self, status: StatusCode) -> Result<()> {
        send_downstream_rejection(&mut self.connect_stream, status).await
    }

    async fn accept_and_bridge(mut self, upstream: UpstreamSession) -> Result<()> {
        let downstream_session_id = self.connect_stream.id().into_inner();
        send_downstream_success(&mut self.connect_stream).await?;
        // Keep the accepted CONNECT and h3 session handles alive while raw QUIC streams bridge.
        let _downstream_h3 = self.h3_connection;
        let _downstream_connect = self.connect_stream;
        let _upstream_endpoint = upstream.endpoint;
        let _upstream_h3 = upstream.h3_connection;
        let _upstream_connect = upstream.connect_stream;
        bridge_upstream_webtransport_streams(
            self.connection,
            upstream.connection,
            upstream.session_id,
            downstream_session_id,
        )
        .await;
        Ok(())
    }
}

async fn send_downstream_rejection(
    connect_stream: &mut DownstreamConnectStream,
    status: StatusCode,
) -> Result<()> {
    let response = Response::builder()
        .status(status)
        .body(())
        .context("build downstream rejection response")?;
    connect_stream
        .send_response(response)
        .await
        .map_err(|error| anyhow!("send downstream rejection response: {error:?}"))?;
    connect_stream
        .finish()
        .await
        .map_err(|error| anyhow!("finish downstream rejection response: {error:?}"))?;
    Ok(())
}

async fn send_downstream_success(connect_stream: &mut DownstreamConnectStream) -> Result<()> {
    let response = Response::builder()
        .status(StatusCode::OK)
        .body(())
        .context("build downstream WebTransport response")?;
    connect_stream
        .send_response(response)
        .await
        .map_err(|error| anyhow!("send downstream WebTransport response: {error:?}"))?;
    Ok(())
}

struct UpstreamSession {
    endpoint: Endpoint,
    connection: quinn::Connection,
    h3_connection: UpstreamH3Connection,
    connect_stream: UpstreamConnectStream,
    response_status: StatusCode,
    session_id: u64,
}

async fn connect_upstream(
    upstream_addr: &str,
    server_name: &str,
    headers: &HeaderMap,
    tls: &UpstreamTlsConfig,
) -> Result<UpstreamSession> {
    let upstream = prepare_upstream_quic(upstream_addr, server_name, tls).await?;
    open_upstream_session(upstream, server_name, headers).await
}

struct UpstreamQuic {
    endpoint: Endpoint,
    connection: quinn::Connection,
}

async fn prepare_upstream_quic(
    upstream_addr: &str,
    server_name: &str,
    tls: &UpstreamTlsConfig,
) -> Result<UpstreamQuic> {
    let candidates = resolve_upstream_addrs(upstream_addr).await?;
    let client_config = client_config(tls)?;
    let (_, upstream) = connect_first_upstream_candidate(&candidates, |dial_target| {
        let client_config = client_config.clone();
        async move {
            let endpoint = bind_upstream_endpoint(&client_config, dial_target)?;
            let connection = connect_upstream_quic(&endpoint, dial_target, server_name).await?;
            Ok(UpstreamQuic {
                endpoint,
                connection,
            })
        }
    })
    .await
    .with_context(|| format!("connect upstream QUIC to {upstream_addr}"))?;
    Ok(upstream)
}

async fn open_upstream_session(
    upstream: UpstreamQuic,
    server_name: &str,
    headers: &HeaderMap,
) -> Result<UpstreamSession> {
    let (h3_connection, send_request) = build_upstream_h3(&upstream.connection).await?;
    let (connect_stream, response_status, session_id) =
        open_upstream_webtransport(send_request, server_name, headers).await?;

    Ok(UpstreamSession {
        endpoint: upstream.endpoint,
        connection: upstream.connection,
        h3_connection,
        connect_stream,
        response_status,
        session_id,
    })
}

fn bind_upstream_endpoint(
    client_config: &ClientConfig,
    dial_target: SocketAddr,
) -> Result<Endpoint> {
    let mut endpoint = Endpoint::client(stargate_tls::quic_client_bind_addr(dial_target))
        .context("bind upstream QUIC client")?;
    endpoint.set_default_client_config(client_config.clone());
    Ok(endpoint)
}

async fn connect_upstream_quic(
    endpoint: &Endpoint,
    dial_target: SocketAddr,
    server_name: &str,
) -> Result<quinn::Connection> {
    endpoint
        .connect(dial_target, server_name)
        .context("start upstream QUIC connection")?
        .await
        .context("establish upstream QUIC connection")
}

async fn build_upstream_h3(
    connection: &quinn::Connection,
) -> Result<(UpstreamH3Connection, UpstreamSendRequest)> {
    let mut builder = h3::client::builder();
    builder.enable_extended_connect(true).enable_datagram(true);
    builder
        .build(h3_quinn::Connection::new(connection.clone()))
        .await
        .map_err(|error| anyhow!("create upstream h3 client: {error:?}"))
}

async fn open_upstream_webtransport(
    mut send_request: UpstreamSendRequest,
    server_name: &str,
    headers: &HeaderMap,
) -> Result<(UpstreamConnectStream, StatusCode, u64)> {
    let mut connect_stream =
        send_upstream_webtransport_connect(&mut send_request, server_name, headers).await?;
    let session_id = connect_stream.id().into_inner();
    finish_upstream_webtransport_connect(&mut connect_stream).await?;
    let response_status = read_upstream_webtransport_status(&mut connect_stream).await?;
    Ok((connect_stream, response_status, session_id))
}

async fn send_upstream_webtransport_connect(
    send_request: &mut UpstreamSendRequest,
    server_name: &str,
    headers: &HeaderMap,
) -> Result<UpstreamConnectStream> {
    let request = build_upstream_webtransport_request(server_name, headers)?;
    send_request
        .send_request(request)
        .await
        .map_err(|error| anyhow!("send upstream WebTransport CONNECT: {error:?}"))
}

async fn finish_upstream_webtransport_connect(
    connect_stream: &mut UpstreamConnectStream,
) -> Result<()> {
    connect_stream
        .finish()
        .await
        .map_err(|error| anyhow!("finish upstream WebTransport CONNECT: {error:?}"))
}

async fn read_upstream_webtransport_status(
    connect_stream: &mut UpstreamConnectStream,
) -> Result<StatusCode> {
    connect_stream
        .recv_response()
        .await
        .map(|response| response.status())
        .map_err(|error| anyhow!("read upstream WebTransport CONNECT response: {error:?}"))
}

fn build_upstream_webtransport_request(
    server_name: &str,
    headers: &HeaderMap,
) -> Result<Request<()>> {
    let mut request = Request::builder()
        .method(Method::CONNECT)
        .uri(format!("https://{server_name}{WEBTRANSPORT_TUNNEL_PATH}"))
        .body(())
        .context("build upstream WebTransport CONNECT")?;
    for (name, value) in headers {
        request.headers_mut().append(name, value.clone());
    }
    request
        .extensions_mut()
        .insert(h3::ext::Protocol::WEB_TRANSPORT);
    Ok(request)
}

async fn bridge_upstream_webtransport_streams(
    downstream_connection: quinn::Connection,
    upstream_connection: quinn::Connection,
    upstream_session_id: u64,
    downstream_session_id: u64,
) {
    loop {
        tokio::select! {
            _ = downstream_connection.closed() => {
                upstream_connection.close(0u32.into(), b"downstream webtransport closed");
                break;
            }
            stream = upstream_connection.accept_bi() => {
                let Ok((upstream_send, upstream_recv)) = stream else {
                    downstream_connection.close(0u32.into(), b"upstream webtransport closed");
                    break;
                };
                let downstream_connection = downstream_connection.clone();
                tokio::spawn(async move {
                    if let Err(error) = bridge_upstream_webtransport_stream(
                        downstream_connection,
                        upstream_send,
                        upstream_recv,
                        upstream_session_id,
                        downstream_session_id,
                    )
                    .await
                    {
                        warn!(error = %error, "WebTransport L7 stream bridge failed");
                    }
                });
            }
        }
    }
}

async fn bridge_upstream_webtransport_stream(
    downstream_connection: quinn::Connection,
    mut upstream_send: quinn::SendStream,
    mut upstream_recv: quinn::RecvStream,
    upstream_session_id: u64,
    downstream_session_id: u64,
) -> Result<()> {
    let stream_session_id = match tokio::time::timeout(
        WEBTRANSPORT_STREAM_HEADER_TIMEOUT,
        stargate_protocol::read_webtransport_bidi_header(&mut upstream_recv),
    )
    .await
    {
        Ok(Ok(session_id)) => session_id,
        Ok(Err(error)) => {
            reset_webtransport_stream(&mut upstream_send, &mut upstream_recv);
            return Err(error).context("read upstream WebTransport stream header");
        }
        Err(_) => {
            reset_webtransport_stream(&mut upstream_send, &mut upstream_recv);
            return Err(anyhow!(
                "timed out waiting for upstream WebTransport stream header"
            ));
        }
    };
    if stream_session_id != upstream_session_id {
        reset_webtransport_stream(&mut upstream_send, &mut upstream_recv);
        ensure!(
            stream_session_id == upstream_session_id,
            "upstream WebTransport session id mismatch: got {stream_session_id}, expected {upstream_session_id}"
        );
    }

    let (mut downstream_send, downstream_recv) = downstream_connection
        .open_bi()
        .await
        .context("open downstream WebTransport stream")?;
    stargate_protocol::write_webtransport_bidi_header(&mut downstream_send, downstream_session_id)
        .await
        .context("write downstream WebTransport stream header")?;

    bridge_bidirectional(
        upstream_send,
        upstream_recv,
        downstream_send,
        downstream_recv,
    )
    .await
}

fn reset_webtransport_stream(
    quinn_send: &mut quinn::SendStream,
    quinn_recv: &mut quinn::RecvStream,
) {
    let _ = quinn_send.reset(0u32.into());
    let _ = quinn_recv.stop(0u32.into());
}

async fn bridge_bidirectional(
    upstream_send: quinn::SendStream,
    upstream_recv: quinn::RecvStream,
    downstream_send: quinn::SendStream,
    downstream_recv: quinn::RecvStream,
) -> Result<()> {
    let downstream = copy_stream(upstream_recv, downstream_send);
    let upstream = copy_stream(downstream_recv, upstream_send);
    let (downstream_result, upstream_result) = tokio::join!(downstream, upstream);
    downstream_result.context("copy upstream to downstream")?;
    upstream_result.context("copy downstream to upstream")?;
    Ok(())
}

async fn copy_stream(mut recv: quinn::RecvStream, mut send: quinn::SendStream) -> Result<()> {
    while let Some(chunk) = recv
        .read_chunk(usize::MAX, true)
        .await
        .context("read QUIC stream chunk")?
    {
        send.write_all(&chunk.bytes)
            .await
            .context("write QUIC stream chunk")?;
    }
    send.finish().context("finish QUIC send stream")?;
    Ok(())
}

fn validate_webtransport_connect<B>(request: &Request<B>) -> Result<()> {
    let is_webtransport = request
        .extensions()
        .get::<h3::ext::Protocol>()
        .is_some_and(|protocol| *protocol == h3::ext::Protocol::WEB_TRANSPORT);
    ensure!(
        request.method() == Method::CONNECT
            && request.uri().path() == WEBTRANSPORT_TUNNEL_PATH
            && is_webtransport,
        "invalid downstream WebTransport CONNECT"
    );
    Ok(())
}

fn downstream_server_name(connection: &quinn::Connection) -> Option<String> {
    connection
        .handshake_data()
        .and_then(|data| data.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
        .and_then(|data| data.server_name)
}

fn upstream_addr_for_sni(template: &str, server_name: &str) -> Result<String> {
    let pod_name = server_name
        .split('.')
        .next()
        .filter(|value| !value.is_empty())
        .context("server name does not include a pod hostname")?;
    Ok(template
        .replace("{pod_name}", pod_name)
        .replace("{server_name}", server_name))
}

fn client_config(tls: &UpstreamTlsConfig) -> Result<ClientConfig> {
    if tls.quic_insecure {
        return stargate_tls::build_insecure_quic_client_config_with_alpn(
            TunnelTransportProtocol::WebTransport.alpn_protocols(),
        );
    }
    let cert_pem = tls
        .cert_pem
        .as_deref()
        .context("upstream TLS cert required when verification is enabled")?;
    stargate_tls::build_trusted_quic_client_config_with_alpn(
        cert_pem,
        TunnelTransportProtocol::WebTransport.alpn_protocols(),
    )
}

#[cfg(test)]
mod tests {
    use http::{HeaderMap, HeaderValue, Method};
    use quinn::Endpoint;

    use super::*;
    use crate::server_config;

    #[test]
    fn upstream_addr_template_uses_pod_name_from_sni() {
        let addr = upstream_addr_for_sni(
            "{pod_name}.stargate-headless.ns.svc.cluster.local:50072",
            "stargate-1.stargate.external",
        )
        .unwrap();

        assert_eq!(
            addr,
            "stargate-1.stargate-headless.ns.svc.cluster.local:50072"
        );
    }

    #[test]
    fn upstream_addr_template_can_use_full_server_name() {
        let addr =
            upstream_addr_for_sni("{server_name}:50072", "stargate-1.stargate.example").unwrap();

        assert_eq!(addr, "stargate-1.stargate.example:50072");
    }

    #[test]
    fn upstream_connect_request_preserves_downstream_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("request-1"));
        headers.append("x-routing-key", HeaderValue::from_static("alpha"));
        headers.append("x-routing-key", HeaderValue::from_static("beta"));

        let request = build_upstream_webtransport_request("stargate-1.example", &headers).unwrap();

        assert_eq!(request.method(), Method::CONNECT);
        assert_eq!(
            request.uri().to_string(),
            format!("https://stargate-1.example{WEBTRANSPORT_TUNNEL_PATH}")
        );
        assert!(
            request
                .extensions()
                .get::<h3::ext::Protocol>()
                .is_some_and(|protocol| *protocol == h3::ext::Protocol::WEB_TRANSPORT)
        );
        assert_eq!(
            request.headers().get("x-request-id").unwrap(),
            &HeaderValue::from_static("request-1")
        );
        let routing_values: Vec<_> = request
            .headers()
            .get_all("x-routing-key")
            .iter()
            .map(|value| value.to_str().unwrap())
            .collect();
        assert_eq!(routing_values, ["alpha", "beta"]);
    }

    #[test]
    fn upstream_client_config_requires_trust_when_verification_is_enabled() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let secure_without_trust = UpstreamTlsConfig {
            cert_pem: None,
            quic_insecure: false,
        };
        assert!(client_config(&secure_without_trust).is_err());

        let (cert_pem, _) = stargate_tls::generate_self_signed_cert().unwrap();
        let secure_with_trust = UpstreamTlsConfig {
            cert_pem: Some(cert_pem),
            quic_insecure: false,
        };
        assert!(client_config(&secure_with_trust).is_ok());

        let insecure = UpstreamTlsConfig {
            cert_pem: None,
            quic_insecure: true,
        };
        assert!(client_config(&insecure).is_ok());
    }

    #[tokio::test]
    async fn prepare_upstream_quic_resolves_and_connects_to_a_local_server() -> Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let server_endpoint = Endpoint::server(
            server_config(&stargate_tls::ServerTlsIdentity::SelfSigned)
                .expect("local QUIC server config should build"),
            "127.0.0.1:0"
                .parse()
                .expect("local QUIC server address should parse"),
        )
        .expect("local QUIC server should bind");
        let server_addr = server_endpoint.local_addr()?;
        let server_task = tokio::spawn(async move {
            let incoming = server_endpoint
                .accept()
                .await
                .expect("server should accept the proxy's QUIC connection");
            let connection = incoming
                .await
                .expect("proxy connection should complete its QUIC handshake");
            tokio::time::timeout(Duration::from_secs(1), connection.closed())
                .await
                .expect("closing the proxy connection should reach the server");
        });

        let upstream = prepare_upstream_quic(
            &server_addr.to_string(),
            "stargate",
            &UpstreamTlsConfig::default(),
        )
        .await?;

        assert_eq!(upstream.connection.remote_address(), server_addr);
        upstream.connection.close(0u32.into(), b"test complete");
        server_task
            .await
            .expect("local QUIC server task should not panic");
        Ok(())
    }

    #[test]
    fn downstream_connect_validation_rejects_non_webtransport_requests() {
        let request = Request::builder()
            .method(Method::GET)
            .uri(WEBTRANSPORT_TUNNEL_PATH)
            .body(())
            .unwrap();
        assert!(validate_webtransport_connect(&request).is_err());

        let mut request = Request::builder()
            .method(Method::CONNECT)
            .uri("/not-webtransport")
            .body(())
            .unwrap();
        request
            .extensions_mut()
            .insert(h3::ext::Protocol::WEB_TRANSPORT);
        assert!(validate_webtransport_connect(&request).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn downstream_session_accepts_h3_connect_and_can_reject() -> Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let (_server_endpoint, _client_endpoint, server_connection, client_connection) =
            connect_quic_pair().await;
        let server_task = tokio::spawn(DownstreamSession::from_connection(
            server_connection,
            "stargate-1.example".to_string(),
            "127.0.0.1:50072".to_string(),
        ));
        let (_client_h3, mut send_request) = build_upstream_h3(&client_connection).await?;
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", HeaderValue::from_static("request-1"));
        let request = build_upstream_webtransport_request("stargate-1.example", &headers)?;

        let mut client_stream = send_request
            .send_request(request)
            .await
            .map_err(|error| anyhow!("send downstream test CONNECT: {error:?}"))?;
        client_stream
            .finish()
            .await
            .map_err(|error| anyhow!("finish downstream test CONNECT: {error:?}"))?;
        let mut session = server_task
            .await
            .expect("downstream session task should not panic")?
            .expect("valid CONNECT should produce a session");

        assert_eq!(session.server_name, "stargate-1.example");
        assert_eq!(session.upstream_addr, "127.0.0.1:50072");
        assert_eq!(
            session.request_headers.get("x-request-id").unwrap(),
            &HeaderValue::from_static("request-1")
        );

        session.reject(StatusCode::BAD_GATEWAY).await?;
        let response = client_stream
            .recv_response()
            .await
            .map_err(|error| anyhow!("read downstream rejection response: {error:?}"))?;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upstream_session_open_sends_webtransport_connect_and_reads_status() -> Result<()> {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let (_server_endpoint, client_endpoint, server_connection, client_connection) =
            connect_quic_pair().await;
        let (release_server_tx, release_server_rx) = tokio::sync::oneshot::channel();
        let server_task = tokio::spawn(async move {
            let mut builder = h3::server::builder();
            builder
                .enable_webtransport(true)
                .enable_extended_connect(true)
                .enable_datagram(true)
                .max_webtransport_sessions(1);
            let mut server_h3: DownstreamH3Connection = builder
                .build(h3_quinn::Connection::new(server_connection))
                .await
                .map_err(|error| anyhow!("create test upstream h3 server: {error:?}"))?;
            let resolver = server_h3
                .accept()
                .await
                .map_err(|error| anyhow!("accept test upstream CONNECT: {error:?}"))?
                .expect("upstream server should receive CONNECT");
            let (request, mut stream) = resolver
                .resolve_request()
                .await
                .map_err(|error| anyhow!("resolve test upstream CONNECT: {error:?}"))?;
            assert_eq!(request.method(), Method::CONNECT);
            assert_eq!(
                request.uri().path(),
                WEBTRANSPORT_TUNNEL_PATH,
                "upstream CONNECT should target the tunnel path"
            );
            assert_eq!(
                request.headers().get("x-routing-key").unwrap(),
                &HeaderValue::from_static("tenant-a")
            );
            let response = Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(())
                .context("build test upstream response")?;
            stream
                .send_response(response)
                .await
                .map_err(|error| anyhow!("send test upstream response: {error:?}"))?;
            stream
                .finish()
                .await
                .map_err(|error| anyhow!("finish test upstream response: {error:?}"))?;
            let _ = release_server_rx.await;
            Result::<()>::Ok(())
        });
        let mut headers = HeaderMap::new();
        headers.insert("x-routing-key", HeaderValue::from_static("tenant-a"));

        let upstream = UpstreamQuic {
            endpoint: client_endpoint,
            connection: client_connection,
        };
        let session = open_upstream_session(upstream, "stargate-1.example", &headers).await?;

        assert_eq!(session.response_status, StatusCode::BAD_GATEWAY);
        let _ = release_server_tx.send(());
        server_task
            .await
            .expect("upstream server task should not panic")?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_upstream_stream_header_does_not_block_later_streams() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let fixture = connect_bridge_fixture().await;

        let bridge_task = tokio::spawn(bridge_upstream_webtransport_streams(
            fixture.downstream_proxy_connection,
            fixture.upstream_proxy_connection,
            41,
            77,
        ));
        let (_stalled_send, _stalled_recv) = fixture
            .upstream_server_connection
            .open_bi()
            .await
            .expect("stalled stream");

        let (mut upstream_send, _upstream_recv) = fixture
            .upstream_server_connection
            .open_bi()
            .await
            .expect("second upstream stream");
        stargate_protocol::write_webtransport_bidi_header(&mut upstream_send, 41)
            .await
            .expect("write upstream WebTransport header");
        upstream_send
            .write_all(b"later stream")
            .await
            .expect("write upstream payload");
        upstream_send.finish().expect("finish upstream stream");

        let (mut downstream_send, mut downstream_recv) = tokio::time::timeout(
            Duration::from_secs(1),
            fixture.downstream_client_connection.accept_bi(),
        )
        .await
        .expect("later downstream stream should not be blocked")
        .expect("downstream stream");
        let downstream_session_id =
            stargate_protocol::read_webtransport_bidi_header(&mut downstream_recv)
                .await
                .expect("read downstream WebTransport header");
        assert_eq!(downstream_session_id, 77);
        downstream_send.finish().expect("finish downstream stream");
        let payload = downstream_recv
            .read_to_end(1024)
            .await
            .expect("read payload");
        assert_eq!(payload, b"later stream");

        bridge_task.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upstream_stream_header_error_resets_stream() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let fixture = connect_bridge_fixture().await;

        let (mut server_send, _server_recv) = fixture
            .upstream_server_connection
            .open_bi()
            .await
            .expect("upstream stream");
        server_send.finish().expect("finish empty upstream stream");
        let (proxy_send, proxy_recv) = fixture
            .upstream_proxy_connection
            .accept_bi()
            .await
            .expect("proxy should accept upstream stream");

        let error = bridge_upstream_webtransport_stream(
            fixture.downstream_proxy_connection,
            proxy_send,
            proxy_recv,
            41,
            77,
        )
        .await
        .expect_err("empty stream should fail header decoding");

        assert!(
            error
                .to_string()
                .contains("read upstream WebTransport stream header"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn upstream_stream_session_mismatch_resets_stream() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let fixture = connect_bridge_fixture().await;

        let (mut server_send, _server_recv) = fixture
            .upstream_server_connection
            .open_bi()
            .await
            .expect("upstream stream");
        stargate_protocol::write_webtransport_bidi_header(&mut server_send, 99)
            .await
            .expect("write mismatched upstream WebTransport header");
        let (proxy_send, proxy_recv) = fixture
            .upstream_proxy_connection
            .accept_bi()
            .await
            .expect("proxy should accept upstream stream");

        let error = bridge_upstream_webtransport_stream(
            fixture.downstream_proxy_connection,
            proxy_send,
            proxy_recv,
            41,
            77,
        )
        .await
        .expect_err("mismatched session id should fail");

        assert!(
            error
                .to_string()
                .contains("upstream WebTransport session id mismatch"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn matching_upstream_stream_bridges_bytes_in_both_directions() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let fixture = connect_bridge_fixture().await;
        let (mut server_send, mut server_recv) = fixture
            .upstream_server_connection
            .open_bi()
            .await
            .expect("upstream stream");
        stargate_protocol::write_webtransport_bidi_header(&mut server_send, 41)
            .await
            .expect("write upstream WebTransport header");
        server_send
            .write_all(b"upstream payload")
            .await
            .expect("write upstream payload");
        server_send.finish().expect("finish upstream send");
        let (proxy_send, proxy_recv) = fixture
            .upstream_proxy_connection
            .accept_bi()
            .await
            .expect("proxy should accept upstream stream");

        let bridge_task = tokio::spawn(bridge_upstream_webtransport_stream(
            fixture.downstream_proxy_connection,
            proxy_send,
            proxy_recv,
            41,
            77,
        ));
        let (mut downstream_send, mut downstream_recv) = tokio::time::timeout(
            Duration::from_secs(1),
            fixture.downstream_client_connection.accept_bi(),
        )
        .await
        .expect("downstream stream should open")
        .expect("downstream stream");
        let downstream_session_id =
            stargate_protocol::read_webtransport_bidi_header(&mut downstream_recv)
                .await
                .expect("read downstream WebTransport header");
        assert_eq!(downstream_session_id, 77);
        let downstream_payload = downstream_recv
            .read_to_end(1024)
            .await
            .expect("read downstream payload");
        assert_eq!(downstream_payload, b"upstream payload");

        downstream_send
            .write_all(b"downstream payload")
            .await
            .expect("write downstream payload");
        downstream_send.finish().expect("finish downstream send");
        let upstream_payload = server_recv
            .read_to_end(1024)
            .await
            .expect("read upstream payload");
        assert_eq!(upstream_payload, b"downstream payload");
        bridge_task
            .await
            .expect("bridge task should not panic")
            .expect("matching stream should bridge cleanly");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn downstream_close_closes_upstream_session() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let fixture = connect_bridge_fixture().await;

        let bridge_task = tokio::spawn(bridge_upstream_webtransport_streams(
            fixture.downstream_proxy_connection,
            fixture.upstream_proxy_connection,
            41,
            77,
        ));

        fixture
            .downstream_client_connection
            .close(0u32.into(), b"client restart");
        tokio::time::timeout(
            Duration::from_secs(1),
            fixture.upstream_server_connection.closed(),
        )
        .await
        .expect("upstream session should close after downstream disconnects");

        bridge_task.abort();
    }

    struct BridgeFixture {
        _endpoints: [Endpoint; 4],
        upstream_server_connection: quinn::Connection,
        upstream_proxy_connection: quinn::Connection,
        downstream_proxy_connection: quinn::Connection,
        downstream_client_connection: quinn::Connection,
    }

    async fn connect_bridge_fixture() -> BridgeFixture {
        let (
            upstream_server_endpoint,
            upstream_proxy_endpoint,
            upstream_server_connection,
            upstream_proxy_connection,
        ) = connect_quic_pair().await;
        let (
            downstream_proxy_endpoint,
            downstream_client_endpoint,
            downstream_proxy_connection,
            downstream_client_connection,
        ) = connect_quic_pair().await;
        BridgeFixture {
            _endpoints: [
                upstream_server_endpoint,
                upstream_proxy_endpoint,
                downstream_proxy_endpoint,
                downstream_client_endpoint,
            ],
            upstream_server_connection,
            upstream_proxy_connection,
            downstream_proxy_connection,
            downstream_client_connection,
        }
    }

    async fn connect_quic_pair() -> (Endpoint, Endpoint, quinn::Connection, quinn::Connection) {
        let server_endpoint = Endpoint::server(
            server_config(&stargate_tls::ServerTlsIdentity::SelfSigned).unwrap(),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        let server_addr = server_endpoint.local_addr().unwrap();
        let server_task = tokio::spawn(async move {
            let incoming = server_endpoint
                .accept()
                .await
                .expect("server should accept");
            let server_connection = incoming.await.expect("server connection should complete");
            (server_endpoint, server_connection)
        });

        let mut client_endpoint = Endpoint::client("127.0.0.1:0".parse().unwrap()).unwrap();
        client_endpoint
            .set_default_client_config(client_config(&UpstreamTlsConfig::default()).unwrap());
        let client_connection = client_endpoint
            .connect(server_addr, "stargate")
            .unwrap()
            .await
            .unwrap();
        let (server_endpoint, server_connection) = server_task.await.unwrap();

        (
            server_endpoint,
            client_endpoint,
            server_connection,
            client_connection,
        )
    }
}
