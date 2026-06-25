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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use quinn::{Endpoint, ServerConfig};
use stargate_protocol::TunnelTransportProtocol;
use stargate_tls::ServerTlsIdentity;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

#[path = "stargate_webtransport_l7_proxy/network.rs"]
mod network;
#[path = "stargate_webtransport_l7_proxy/session.rs"]
mod session;

use network::{connect_first_upstream_candidate, resolve_upstream_addrs};

#[derive(Parser, Debug)]
#[command(name = "stargate-webtransport-l7-proxy")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:50072", value_name = "ADDR")]
    listen_addr: SocketAddr,
    #[arg(long, value_name = "TEMPLATE")]
    upstream_template: String,
    #[arg(long, default_value = "0.0.0.0:50071", value_name = "ADDR")]
    control_plane_listen_addr: SocketAddr,
    #[arg(long, value_name = "ADDR")]
    control_plane_upstream_addr: Option<String>,
    #[arg(long, value_name = "PATH")]
    tls_cert_path: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    tls_key_path: Option<PathBuf>,
    #[arg(long, value_name = "PATH")]
    upstream_tls_cert_path: Option<PathBuf>,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    upstream_quic_insecure: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    run(args).await
}

async fn run(args: Args) -> Result<()> {
    let downstream_identity = ServerTlsIdentity::from_optional_pem(
        read_optional_file(args.tls_cert_path.as_deref())?,
        read_optional_file(args.tls_key_path.as_deref())?,
    )?;
    let upstream_tls = session::UpstreamTlsConfig {
        cert_pem: read_optional_file(args.upstream_tls_cert_path.as_deref())?,
        quic_insecure: args.upstream_quic_insecure,
    };
    let endpoint = bind_webtransport_listener(args.listen_addr, &downstream_identity)?;
    let listen_addr = endpoint.local_addr().context("read l7 proxy listen addr")?;
    let shutdown = CancellationToken::new();
    let control_plane_task = spawn_control_plane_tcp_proxy_if_configured(&args, &shutdown).await?;
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(error = %error, "failed to wait for shutdown signal");
        }
        signal_shutdown.cancel();
    });
    info!(%listen_addr, "WebTransport L7 proxy listening");

    accept_webtransport_connections(&endpoint, &args.upstream_template, &upstream_tls, &shutdown)
        .await;
    close_webtransport_listener(endpoint).await;
    shutdown.cancel();
    join_control_plane_tcp_proxy(control_plane_task).await
}

fn read_optional_file(path: Option<&Path>) -> Result<Option<Vec<u8>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    std::fs::read(path)
        .map(Some)
        .with_context(|| format!("read TLS file {}", path.display()))
}

fn bind_webtransport_listener(
    listen_addr: SocketAddr,
    identity: &ServerTlsIdentity,
) -> Result<Endpoint> {
    Endpoint::server(server_config(identity)?, listen_addr).context("bind l7 proxy QUIC server")
}

async fn spawn_control_plane_tcp_proxy_if_configured(
    args: &Args,
    shutdown: &CancellationToken,
) -> Result<Option<JoinHandle<Result<()>>>> {
    let Some(upstream_addr) = args.control_plane_upstream_addr.clone() else {
        return Ok(None);
    };
    let control_plane_listener = TcpListener::bind(args.control_plane_listen_addr)
        .await
        .with_context(|| {
            format!(
                "bind control-plane TCP proxy on {}",
                args.control_plane_listen_addr
            )
        })?;
    Ok(Some(tokio::spawn(run_control_plane_tcp_proxy(
        control_plane_listener,
        upstream_addr,
        shutdown.clone(),
    ))))
}

async fn accept_webtransport_connections(
    endpoint: &Endpoint,
    upstream_template: &str,
    upstream_tls: &session::UpstreamTlsConfig,
    shutdown: &CancellationToken,
) {
    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else { break };
                let upstream_template = upstream_template.to_string();
                let upstream_tls = upstream_tls.clone();
                tokio::spawn(async move {
                    if let Err(error) =
                        session::handle_connection(incoming, upstream_template, upstream_tls).await
                    {
                        warn!(error = %error, "WebTransport L7 proxy connection failed");
                    }
                });
            }
        }
    }
}

async fn close_webtransport_listener(endpoint: Endpoint) {
    endpoint.close(0u32.into(), b"shutdown");
    endpoint.wait_idle().await;
}

async fn join_control_plane_tcp_proxy(
    control_plane_task: Option<JoinHandle<Result<()>>>,
) -> Result<()> {
    if let Some(task) = control_plane_task {
        task.await
            .context("join control-plane TCP proxy task")?
            .context("run control-plane TCP proxy")?;
    }
    Ok(())
}

async fn run_control_plane_tcp_proxy(
    listener: TcpListener,
    upstream_addr: String,
    shutdown: CancellationToken,
) -> Result<()> {
    let listen_addr = listener
        .local_addr()
        .context("read control-plane TCP proxy listen addr")?;
    info!(%listen_addr, %upstream_addr, "control-plane TCP proxy listening");

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            accepted = listener.accept() => {
                let (downstream, peer_addr) = accepted.context("accept control-plane TCP connection")?;
                let upstream_addr = upstream_addr.clone();
                tokio::spawn(async move {
                    if let Err(error) = handle_control_plane_tcp_connection(downstream, &upstream_addr).await {
                        warn!(%peer_addr, error = %error, "control-plane TCP proxy connection failed");
                    }
                });
            }
        }
    }

    Ok(())
}

async fn handle_control_plane_tcp_connection(
    downstream: TcpStream,
    upstream_addr: &str,
) -> Result<()> {
    let upstream = connect_control_plane_upstream(upstream_addr).await?;
    proxy_control_plane_streams(downstream, upstream).await
}

async fn connect_control_plane_upstream(upstream_addr: &str) -> Result<TcpStream> {
    let candidates = resolve_upstream_addrs(upstream_addr).await?;
    let (_, stream) = connect_first_upstream_candidate(&candidates, |candidate| async move {
        TcpStream::connect(candidate)
            .await
            .with_context(|| format!("connect control-plane upstream {upstream_addr}"))
    })
    .await
    .with_context(|| format!("connect control-plane upstream {upstream_addr}"))?;
    Ok(stream)
}

async fn proxy_control_plane_streams(downstream: TcpStream, upstream: TcpStream) -> Result<()> {
    let (downstream_recv, downstream_send) = downstream.into_split();
    let (upstream_recv, upstream_send) = upstream.into_split();
    let downstream_to_upstream = copy_tcp_half(downstream_recv, upstream_send);
    let upstream_to_downstream = copy_tcp_half(upstream_recv, downstream_send);
    let (downstream_result, upstream_result) =
        tokio::join!(downstream_to_upstream, upstream_to_downstream);
    downstream_result.context("copy downstream control-plane bytes upstream")?;
    upstream_result.context("copy upstream control-plane bytes downstream")?;
    Ok(())
}

async fn copy_tcp_half<R, W>(mut recv: R, mut send: W) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    tokio::io::copy(&mut recv, &mut send)
        .await
        .context("copy TCP stream")?;
    send.shutdown().await.context("shutdown TCP stream")?;
    Ok(())
}

fn server_config(identity: &ServerTlsIdentity) -> Result<ServerConfig> {
    let (cert_pem, key_pem) = identity.pem_pair()?;
    let cert_chain: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut &*cert_pem)
            .collect::<std::result::Result<_, _>>()
            .context("parse l7 proxy cert")?;
    let key = rustls_pemfile::private_key(&mut &*key_pem)
        .context("parse l7 proxy key")?
        .context("missing l7 proxy key")?;
    let mut tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .context("build l7 proxy TLS config")?;
    tls_config.alpn_protocols = TunnelTransportProtocol::WebTransport.alpn_protocols();
    Ok(ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls_config)?,
    )))
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    fn args(control_plane_upstream_addr: Option<String>) -> Args {
        Args {
            listen_addr: "127.0.0.1:0".parse().unwrap(),
            upstream_template: "{server_name}:50072".to_string(),
            control_plane_listen_addr: "127.0.0.1:0".parse().unwrap(),
            control_plane_upstream_addr,
            tls_cert_path: None,
            tls_key_path: None,
            upstream_tls_cert_path: None,
            upstream_quic_insecure: true,
        }
    }

    #[test]
    fn cli_parses_downstream_identity_and_upstream_trust_settings() {
        let args = Args::try_parse_from([
            "stargate-webtransport-l7-proxy",
            "--upstream-template={server_name}:50072",
            "--tls-cert-path=/tls/downstream.crt",
            "--tls-key-path=/tls/downstream.key",
            "--upstream-tls-cert-path=/tls/upstream.crt",
            "--upstream-quic-insecure=false",
        ])
        .expect("TLS CLI should parse");

        assert_eq!(
            args.tls_cert_path.as_deref(),
            Some(std::path::Path::new("/tls/downstream.crt"))
        );
        assert_eq!(
            args.tls_key_path.as_deref(),
            Some(std::path::Path::new("/tls/downstream.key"))
        );
        assert_eq!(
            args.upstream_tls_cert_path.as_deref(),
            Some(std::path::Path::new("/tls/upstream.crt"))
        );
        assert!(!args.upstream_quic_insecure);
    }

    #[test]
    fn downstream_server_config_accepts_provided_identity() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let (cert_pem, key_pem) = stargate_tls::generate_self_signed_cert().unwrap();
        let identity = stargate_tls::ServerTlsIdentity::Provided { cert_pem, key_pem };

        server_config(&identity).expect("provided downstream identity should build");
    }

    #[tokio::test]
    async fn webtransport_listener_binds_and_closes() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let endpoint = bind_webtransport_listener(
            "127.0.0.1:0".parse().unwrap(),
            &ServerTlsIdentity::SelfSigned,
        )
        .expect("listener should bind");

        assert!(endpoint.local_addr().unwrap().port() > 0);

        close_webtransport_listener(endpoint).await;
    }

    #[tokio::test]
    async fn webtransport_accept_loop_runs_until_shutdown() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let endpoint = bind_webtransport_listener(
            "127.0.0.1:0".parse().unwrap(),
            &ServerTlsIdentity::SelfSigned,
        )
        .expect("listener should bind");
        let shutdown = CancellationToken::new();
        let upstream_tls = session::UpstreamTlsConfig::default();
        {
            let accept_loop = accept_webtransport_connections(
                &endpoint,
                "{server_name}:50072",
                &upstream_tls,
                &shutdown,
            );
            tokio::pin!(accept_loop);

            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(20), &mut accept_loop)
                    .await
                    .is_err(),
                "accept loop should wait for connections or shutdown"
            );

            shutdown.cancel();
            tokio::time::timeout(std::time::Duration::from_secs(1), &mut accept_loop)
                .await
                .expect("accept loop should stop after cancellation");
        }
        close_webtransport_listener(endpoint).await;
    }

    #[tokio::test]
    async fn control_plane_tcp_proxy_forwards_bytes_and_shutdown() -> Result<()> {
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await?;
        let upstream_addr = upstream_listener.local_addr()?;
        let upstream_task = tokio::spawn(async move {
            let (mut stream, _) = upstream_listener.accept().await?;
            let mut request = [0; 4];
            stream.read_exact(&mut request).await?;
            assert_eq!(&request, b"ping");
            stream.write_all(b"pong").await?;
            stream.shutdown().await?;
            Result::<()>::Ok(())
        });

        let proxy_listener = TcpListener::bind("127.0.0.1:0").await?;
        let proxy_addr = proxy_listener.local_addr()?;
        let shutdown = CancellationToken::new();
        let proxy_task = tokio::spawn(run_control_plane_tcp_proxy(
            proxy_listener,
            upstream_addr.to_string(),
            shutdown.clone(),
        ));

        let mut client = TcpStream::connect(proxy_addr).await?;
        client.write_all(b"ping").await?;
        let mut response = [0; 4];
        client.read_exact(&mut response).await?;
        assert_eq!(&response, b"pong");
        client.shutdown().await?;

        upstream_task.await??;
        shutdown.cancel();
        proxy_task.await??;
        Ok(())
    }

    #[tokio::test]
    async fn control_plane_proxy_task_helpers_handle_absent_and_present_tasks() -> Result<()> {
        let shutdown = CancellationToken::new();
        assert!(
            spawn_control_plane_tcp_proxy_if_configured(&args(None), &shutdown)
                .await?
                .is_none()
        );

        let reserved_listener = TcpListener::bind("127.0.0.1:0").await?;
        let listen_addr = reserved_listener.local_addr()?;
        // Release the reserved port so the helper under test can prove it binds that address.
        drop(reserved_listener);
        let mut present_args = args(Some("127.0.0.1:1".to_string()));
        present_args.control_plane_listen_addr = listen_addr;
        let task = spawn_control_plane_tcp_proxy_if_configured(&present_args, &shutdown)
            .await?
            .expect("configured control-plane proxy should spawn a task");
        shutdown.cancel();
        join_control_plane_tcp_proxy(Some(task)).await?;

        let task = tokio::spawn(async { Result::<()>::Ok(()) });
        join_control_plane_tcp_proxy(Some(task)).await?;
        join_control_plane_tcp_proxy(None).await?;
        Ok(())
    }
}
