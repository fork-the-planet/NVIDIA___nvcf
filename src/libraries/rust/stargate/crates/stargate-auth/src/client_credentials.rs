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

//! OAuth2 client-credentials token source.
//!
//! Exchanges a client id/secret for a short-lived bearer token, for
//! deployments that have no static token signer.

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use sonic_rs::JsonValueTrait;
use tracing::debug;

/// Expiry used when the token endpoint omits `expires_in`.
const DEFAULT_EXPIRES_IN: Duration = Duration::from_secs(300);

/// Refresh a cached token this far before its stated expiry so in-flight calls
/// never present a token that expires mid-request.
const EXPIRY_REFRESH_BUFFER: Duration = Duration::from_secs(60);

/// Upper bound on a token lifetime. Clamps an absurd `expires_in` from the token
/// endpoint so adding it to `Instant::now()` cannot overflow and panic.
const MAX_EXPIRES_IN: Duration = Duration::from_secs(24 * 60 * 60);

/// Mints bearer tokens via the OAuth2 client-credentials grant.
///
/// Re-reads the id/secret per refresh so they can rotate; caches the token
/// until it nears expiry.
pub struct ClientCredentialsProvider {
    /// Token endpoint, formed as `<provider_host>/token`.
    token_url: String,
    /// Secrets file holding the `id` and `secret`.
    credentials_path: PathBuf,
    /// OAuth2 scope requested for the token.
    scope: String,
    http: reqwest::Client,
    cache: tokio::sync::Mutex<Option<CachedToken>>,
}

struct CachedToken {
    token: String,
    expires_at: Instant,
}

impl ClientCredentialsProvider {
    /// Mints tokens with `scope` at `<provider_host>/token`, reading the
    /// id/secret from `credentials_path`.
    pub fn new(provider_host: &str, credentials_path: PathBuf, scope: impl Into<String>) -> Self {
        Self {
            token_url: format!("{}/token", provider_host.trim_end_matches('/')),
            credentials_path,
            scope: scope.into(),
            http: reqwest::Client::new(),
            cache: tokio::sync::Mutex::new(None),
        }
    }

    /// Returns a cached token, minting a fresh one when none is cached or it is
    /// near expiry. The lock is held across the refresh so callers coalesce onto
    /// one token request.
    pub async fn resolve(&self) -> anyhow::Result<String> {
        let mut cache = self.cache.lock().await;
        if let Some(cached) = cache.as_ref()
            && Instant::now() + EXPIRY_REFRESH_BUFFER < cached.expires_at
        {
            return Ok(cached.token.clone());
        }

        let minted = self.mint().await?;
        let token = minted.token.clone();
        *cache = Some(minted);
        Ok(token)
    }

    async fn mint(&self) -> anyhow::Result<CachedToken> {
        let (client_id, client_secret) = self.read_credentials().await?;

        let response = self
            .http
            .post(&self.token_url)
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", client_id.as_str()),
                ("client_secret", client_secret.as_str()),
                ("scope", self.scope.as_str()),
            ])
            .send()
            .await
            .context("oauth2 token request failed")?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .context("failed to read oauth2 token response")?;
        if !status.is_success() {
            return Err(anyhow!("oauth2 token endpoint returned status {status}"));
        }

        let access_token = sonic_rs::get(&body, ["access_token"])
            .context("oauth2 token response missing access_token")?
            .as_str()
            .ok_or_else(|| anyhow!("oauth2 access_token is not a string"))?
            .to_owned();
        let expires_in = sonic_rs::get(&body, ["expires_in"])
            .ok()
            .and_then(|value| value.as_u64())
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_EXPIRES_IN)
            .min(MAX_EXPIRES_IN);

        debug!(
            scope = %self.scope,
            expires_in_secs = expires_in.as_secs(),
            "minted worker-auth client-credentials token"
        );
        Ok(CachedToken {
            token: access_token,
            expires_at: Instant::now() + expires_in,
        })
    }

    async fn read_credentials(&self) -> anyhow::Result<(String, String)> {
        let bytes = tokio::fs::read(&self.credentials_path)
            .await
            .with_context(|| format!("failed to read {}", self.credentials_path.display()))?;
        let id = json_string_field(&bytes, "id")?;
        let secret = json_string_field(&bytes, "secret")?;
        Ok((id, secret))
    }
}

fn json_string_field(bytes: &[u8], field: &str) -> anyhow::Result<String> {
    let value =
        sonic_rs::get(bytes, [field]).with_context(|| format!("secrets file missing {field}"))?;
    let s = value
        .as_str()
        .ok_or_else(|| anyhow!("secrets file value at {field} is not a string"))?;
    Ok(s.to_owned())
}

// Manual Debug so the cached token and the credentials path's contents never
// leak into logs.
impl fmt::Debug for ClientCredentialsProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientCredentialsProvider")
            .field("token_url", &self.token_url)
            .field("scope", &self.scope)
            .field("credentials_path", &self.credentials_path)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::extract::State;
    use axum::routing::post;

    struct TokenEndpoint {
        scope: String,
        client_id: String,
        client_secret: String,
        hits: Arc<AtomicUsize>,
        expires_in: u64,
    }

    /// Starts a fake OAuth2 token endpoint on a random port. It asserts the
    /// client-credentials form fields and returns a token, counting each hit.
    async fn serve_token_endpoint(endpoint: TokenEndpoint) -> (String, Arc<AtomicUsize>) {
        let hits = endpoint.hits.clone();
        let state = Arc::new(endpoint);
        let app = Router::new()
            .route("/token", post(handle_token))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), hits)
    }

    async fn handle_token(State(endpoint): State<Arc<TokenEndpoint>>, body: String) -> String {
        endpoint.hits.fetch_add(1, Ordering::SeqCst);
        let form: std::collections::HashMap<String, String> = url_decode_form(&body);
        assert_eq!(
            form.get("grant_type").map(String::as_str),
            Some("client_credentials")
        );
        assert_eq!(form.get("scope"), Some(&endpoint.scope));
        assert_eq!(form.get("client_id"), Some(&endpoint.client_id));
        assert_eq!(form.get("client_secret"), Some(&endpoint.client_secret));
        format!(
            r#"{{"access_token":"minted-token","token_type":"Bearer","expires_in":{}}}"#,
            endpoint.expires_in
        )
    }

    fn url_decode_form(body: &str) -> std::collections::HashMap<String, String> {
        body.split('&')
            .filter_map(|pair| pair.split_once('='))
            .map(|(k, v)| (percent_decode(k), percent_decode(v)))
            .collect()
    }

    fn percent_decode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut bytes = s.bytes();
        while let Some(b) = bytes.next() {
            match b {
                b'+' => out.push(' '),
                b'%' => {
                    let hi = bytes.next().unwrap();
                    let lo = bytes.next().unwrap();
                    let decoded =
                        u8::from_str_radix(&format!("{}{}", hi as char, lo as char), 16).unwrap();
                    out.push(decoded as char);
                }
                other => out.push(other as char),
            }
        }
        out
    }

    fn credentials_file(id: &str, secret: &str) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, r#"{{"id": "{id}", "secret": "{secret}"}}"#).unwrap();
        tmp
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mints_token_with_client_credentials_grant() {
        let (host, _hits) = serve_token_endpoint(TokenEndpoint {
            scope: "llm:check_worker".to_string(),
            client_id: "client-id".to_string(),
            client_secret: "client-secret".to_string(),
            hits: Arc::new(AtomicUsize::new(0)),
            expires_in: 300,
        })
        .await;
        let creds = credentials_file("client-id", "client-secret");

        let provider =
            ClientCredentialsProvider::new(&host, creds.path().to_path_buf(), "llm:check_worker");

        assert_eq!(provider.resolve().await.unwrap(), "minted-token");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn caches_token_within_validity_window() {
        let (host, hits) = serve_token_endpoint(TokenEndpoint {
            scope: "llm:check_worker".to_string(),
            client_id: "client-id".to_string(),
            client_secret: "client-secret".to_string(),
            hits: Arc::new(AtomicUsize::new(0)),
            expires_in: 300,
        })
        .await;
        let creds = credentials_file("client-id", "client-secret");
        let provider =
            ClientCredentialsProvider::new(&host, creds.path().to_path_buf(), "llm:check_worker");

        provider.resolve().await.unwrap();
        provider.resolve().await.unwrap();

        assert_eq!(
            hits.load(Ordering::SeqCst),
            1,
            "second resolve should hit the cache"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn refetches_when_token_is_near_expiry() {
        // expires_in below the refresh buffer means the cached token is always
        // considered stale, so every resolve mints a fresh one.
        let (host, hits) = serve_token_endpoint(TokenEndpoint {
            scope: "llm:check_worker".to_string(),
            client_id: "client-id".to_string(),
            client_secret: "client-secret".to_string(),
            hits: Arc::new(AtomicUsize::new(0)),
            expires_in: 1,
        })
        .await;
        let creds = credentials_file("client-id", "client-secret");
        let provider =
            ClientCredentialsProvider::new(&host, creds.path().to_path_buf(), "llm:check_worker");

        provider.resolve().await.unwrap();
        provider.resolve().await.unwrap();

        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "near-expiry token should be refetched"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clamps_oversized_expires_in() {
        // A huge expires_in must not panic when added to Instant::now(); the
        // clamped lifetime is far enough out that the token caches.
        let (host, hits) = serve_token_endpoint(TokenEndpoint {
            scope: "llm:check_worker".to_string(),
            client_id: "client-id".to_string(),
            client_secret: "client-secret".to_string(),
            hits: Arc::new(AtomicUsize::new(0)),
            expires_in: u64::MAX,
        })
        .await;
        let creds = credentials_file("client-id", "client-secret");
        let provider =
            ClientCredentialsProvider::new(&host, creds.path().to_path_buf(), "llm:check_worker");

        assert_eq!(provider.resolve().await.unwrap(), "minted-token");
        provider.resolve().await.unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn errors_when_credentials_file_lacks_id() {
        let (host, _hits) = serve_token_endpoint(TokenEndpoint {
            scope: "llm:check_worker".to_string(),
            client_id: "client-id".to_string(),
            client_secret: "client-secret".to_string(),
            hits: Arc::new(AtomicUsize::new(0)),
            expires_in: 300,
        })
        .await;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        write!(tmp, r#"{{"secret": "client-secret"}}"#).unwrap();
        let provider =
            ClientCredentialsProvider::new(&host, tmp.path().to_path_buf(), "llm:check_worker");

        let error = provider.resolve().await.unwrap_err();
        assert!(error.to_string().contains("missing id"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn errors_on_non_success_status() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/token",
            post(|| async { (axum::http::StatusCode::UNAUTHORIZED, "denied") }),
        );
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let creds = credentials_file("client-id", "client-secret");
        let provider = ClientCredentialsProvider::new(
            &format!("http://{addr}"),
            creds.path().to_path_buf(),
            "llm:check_worker",
        );

        let error = provider.resolve().await.unwrap_err();
        assert!(error.to_string().contains("401"));
    }
}
