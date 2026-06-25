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

use anyhow::{Context, Result};
use quinn::ClientConfig;
use rustls::RootCertStore;
use tracing::info;

use stargate_protocol::TunnelTransportProtocol;
use stargate_tls::ServerTlsIdentity;

pub(super) fn build_client_config(
    cert_pem: Option<&[u8]>,
    insecure: bool,
    tunnel_protocol: TunnelTransportProtocol,
) -> Result<ClientConfig> {
    if insecure {
        return stargate_tls::build_insecure_quic_client_config_with_alpn(
            tunnel_protocol.alpn_protocols(),
        );
    }
    let cert_data = cert_pem.context("TLS cert required when --quic-insecure is not set")?;
    let mut roots = RootCertStore::empty();
    for cert in rustls_pemfile::certs(&mut &*cert_data) {
        roots
            .add(cert.context("failed to parse tunnel cert PEM")?)
            .context("failed to add tunnel cert to root store")?;
    }

    let mut tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    tls_config.alpn_protocols = tunnel_protocol.alpn_protocols();

    Ok(ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(tls_config)?,
    )))
}

pub(super) fn build_server_config(
    tls_identity: &ServerTlsIdentity,
    tunnel_protocol: TunnelTransportProtocol,
) -> Result<quinn::ServerConfig> {
    if matches!(tls_identity, ServerTlsIdentity::SelfSigned) {
        info!("no TLS cert/key provided, generating self-signed certificate");
    }
    let (cert_data, key_data) = tls_identity.pem_pair()?;
    let mut cert_reader = cert_data.as_ref();
    let cert_chain: Vec<rustls::pki_types::CertificateDer<'static>> =
        rustls_pemfile::certs(&mut cert_reader)
            .collect::<std::result::Result<_, _>>()
            .context("failed to parse reverse tunnel cert PEM")?;
    let mut key_reader = key_data.as_ref();
    let key = rustls_pemfile::private_key(&mut key_reader)
        .context("failed to parse reverse tunnel key PEM")?
        .context("no private key found in reverse tunnel PEM")?;
    let mut tls_config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .context("build reverse tunnel TLS server config failed")?;
    tls_config.alpn_protocols = tunnel_protocol.alpn_protocols();
    Ok(quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls_config)
            .context("build reverse tunnel QUIC server config failed")?,
    )))
}
