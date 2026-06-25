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

use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::Status;
use tracing::{debug, warn};

use crate::auth::AuthResult;
use crate::routing_state::{RunningRegistration, StargateState};
use crate::tunnel::{EnsureConnectedResult, QuicHttpProxy, RegistrationTunnel};

use stargate_proto::REGISTRATION_HEARTBEAT_MS_METADATA;
use stargate_proto::pb::{
    InferenceServerAck, InferenceServerRegistration, ModelCalibrationDirective,
};

mod health;

use self::admission::{admit_initial_registration, validate_running_update};
use self::health::HealthCheckHandle;

mod admission;

#[derive(Clone)]
pub(crate) struct RegistrationConnectionConfig {
    pub(crate) quic_proxy: Arc<QuicHttpProxy>,
    pub(crate) reverse_tunnel: Option<ReverseTunnelRegistrationConfig>,
}

#[derive(Clone)]
pub(crate) struct ReverseTunnelRegistrationConfig {
    pub(crate) target: String,
    pub(crate) pylon_dial_addr: String,
    pub(crate) connect_timeout: Duration,
}

pub const DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT: Duration = Duration::from_secs(60);
pub const DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

struct RegistrationSession {
    state: Arc<StargateState>,
    connection: RegistrationConnectionConfig,
    registration: RunningRegistration,
    tunnel: RegistrationTunnel,
    health_check: HealthCheckHandle,
}

enum ApplyUpdateOutcome {
    Ack(InferenceServerAck),
    Skip,
    Shutdown,
}

struct RegistrationStreamProcessorContext {
    state: Arc<StargateState>,
    connection: RegistrationConnectionConfig,
    responses: flume::Sender<Result<InferenceServerAck, Status>>,
    auth_result: AuthResult,
    idle_timeout: Option<Duration>,
    stop: CancellationToken,
}

impl RegistrationStreamProcessorContext {
    fn new(
        state: Arc<StargateState>,
        connection: RegistrationConnectionConfig,
        responses: flume::Sender<Result<InferenceServerAck, Status>>,
        auth_result: AuthResult,
        idle_timeout: Option<Duration>,
        stop: CancellationToken,
    ) -> Self {
        Self {
            state,
            connection,
            responses,
            auth_result,
            idle_timeout,
            stop,
        }
    }
}

pub(super) async fn process_registration_stream(
    stream: impl Stream<Item = Result<InferenceServerRegistration, Status>> + Unpin,
    state: Arc<StargateState>,
    registration_connection_config: RegistrationConnectionConfig,
    tx: flume::Sender<Result<InferenceServerAck, Status>>,
    auth_result: AuthResult,
    idle_timeout: Option<Duration>,
    stop: CancellationToken,
) {
    process_registration_stream_with_context(
        stream,
        RegistrationStreamProcessorContext::new(
            state,
            registration_connection_config,
            tx,
            auth_result,
            idle_timeout,
            stop,
        ),
    )
    .await;
}

async fn process_registration_stream_with_context(
    mut stream: impl Stream<Item = Result<InferenceServerRegistration, Status>> + Unpin,
    context: RegistrationStreamProcessorContext,
) {
    let RegistrationStreamProcessorContext {
        state,
        connection: registration_connection_config,
        responses: tx,
        auth_result,
        idle_timeout,
        stop,
    } = context;

    let Some(first_update) = next_registration_update(&mut stream, idle_timeout, &stop).await
    else {
        debug!("register inference servers stream exited before admission");
        return;
    };
    let mut session = match RegistrationSession::start(
        &first_update,
        state,
        registration_connection_config,
        auth_result.routing_key.as_deref(),
    ) {
        Ok(session) => session,
        Err(status) => {
            let _ = send_registration_response(&tx, &stop, Err(status)).await;
            return;
        }
    };

    let mut update = first_update;
    loop {
        match session.apply_update(&update, &stop).await {
            Ok(ApplyUpdateOutcome::Ack(ack)) => {
                if !send_registration_response(&tx, &stop, Ok(ack)).await {
                    break;
                }
            }
            Ok(ApplyUpdateOutcome::Skip) => {}
            Ok(ApplyUpdateOutcome::Shutdown) => break,
            Err(status) => {
                let _ = send_registration_response(&tx, &stop, Err(status)).await;
                break;
            }
        }

        let Some(next_update) = next_registration_update(&mut stream, idle_timeout, &stop).await
        else {
            break;
        };
        update = next_update;
    }

    session.close().await;
}

async fn next_registration_update(
    stream: &mut (impl Stream<Item = Result<InferenceServerRegistration, Status>> + Unpin),
    idle_timeout: Option<Duration>,
    stop: &CancellationToken,
) -> Option<InferenceServerRegistration> {
    // Once shutdown begins, cleanup must win over another ready stream update.
    let next = if let Some(idle_timeout) = idle_timeout {
        let next = tokio::select! {
            biased;
            _ = stop.cancelled() => return None,
            next = tokio::time::timeout(idle_timeout, stream.next()) => next,
        };
        match next {
            Ok(Some(next)) => next,
            Ok(None) => return None,
            Err(_elapsed) => {
                warn!(
                    idle_timeout_ms = idle_timeout.as_millis(),
                    "registration stream idle timeout; closing registration"
                );
                return None;
            }
        }
    } else {
        let next = tokio::select! {
            biased;
            _ = stop.cancelled() => return None,
            next = stream.next() => next,
        };
        match next {
            Some(next) => next,
            None => return None,
        }
    };

    let update = match next {
        Ok(update) => update,
        Err(error) => {
            warn!(error = %error, "inference servers stream read failed or closed");
            return None;
        }
    };
    debug!(
        inference_server_id = %update.inference_server_id,
        model_ids = ?update.models.keys().collect::<Vec<_>>(),
        "received inference servers update"
    );
    Some(update)
}

async fn send_registration_response(
    tx: &flume::Sender<Result<InferenceServerAck, Status>>,
    stop: &CancellationToken,
    response: Result<InferenceServerAck, Status>,
) -> bool {
    tokio::select! {
        biased;
        _ = stop.cancelled() => false,
        result = tx.send_async(response) => result.is_ok(),
    }
}

pub(super) fn negotiated_registration_update_idle_timeout(
    metadata: &tonic::metadata::MetadataMap,
    configured_idle_timeout: Duration,
    configured_max_idle_timeout: Duration,
) -> Option<Duration> {
    if configured_idle_timeout.is_zero() || configured_max_idle_timeout.is_zero() {
        return None;
    }
    let Some(heartbeat_ms) = metadata.get(REGISTRATION_HEARTBEAT_MS_METADATA) else {
        return Some(configured_max_idle_timeout);
    };
    let Ok(heartbeat_ms) = heartbeat_ms.to_str() else {
        warn!(
            "{REGISTRATION_HEARTBEAT_MS_METADATA} must be ascii milliseconds; using configured registration max idle timeout"
        );
        return Some(configured_max_idle_timeout);
    };
    let Ok(heartbeat_ms) = heartbeat_ms.parse::<u64>() else {
        warn!(
            "{REGISTRATION_HEARTBEAT_MS_METADATA} must be integer milliseconds; using configured registration max idle timeout"
        );
        return Some(configured_max_idle_timeout);
    };
    // Untrusted heartbeat metadata must not overflow timeout math; cap through saturation before
    // applying the configured maximum.
    let negotiated_timeout = heartbeat_ms.saturating_mul(3);
    Some(
        Duration::from_millis(negotiated_timeout)
            .max(configured_idle_timeout)
            .min(configured_max_idle_timeout),
    )
}

impl RegistrationSession {
    fn start(
        update: &InferenceServerRegistration,
        state: Arc<StargateState>,
        connection: RegistrationConnectionConfig,
        routing_key: Option<&str>,
    ) -> Result<Self, Status> {
        let identity =
            admit_initial_registration(update, connection.reverse_tunnel.is_some(), routing_key)?;
        let registration = state.begin_registration(&identity)?;
        let generation = registration.generation();
        let tunnel = if identity.reverse_tunnel {
            RegistrationTunnel::reverse(
                connection.quic_proxy.clone(),
                generation.clone(),
                connection
                    .reverse_tunnel
                    .as_ref()
                    .expect("admitted reverse registration requires reverse endpoint config")
                    .connect_timeout,
            )
        } else {
            RegistrationTunnel::direct(connection.quic_proxy.clone(), generation.clone())
        };
        let health_check = HealthCheckHandle::start(connection.quic_proxy.clone(), generation);

        Ok(Self {
            state,
            connection,
            registration,
            tunnel,
            health_check,
        })
    }

    async fn apply_update(
        &mut self,
        update: &InferenceServerRegistration,
        stop: &CancellationToken,
    ) -> Result<ApplyUpdateOutcome, Status> {
        if let Some(status) = validate_running_update(self.registration.identity(), update) {
            return Err(status);
        }
        let connection = tokio::select! {
            biased;
            _ = stop.cancelled() => return Ok(ApplyUpdateOutcome::Shutdown),
            connection = self.tunnel.ensure_connected() => connection,
        };
        let reverse_connected = match connection {
            EnsureConnectedResult::Connected => true,
            EnsureConnectedResult::ReverseDisconnected => false,
            EnsureConnectedResult::Unavailable => {
                // Keep the no-ack retry behavior while clearing any stale route
                // that depended on the now-unavailable direct connection.
                self.state
                    .apply_registration_update(&self.registration, update, false, None)
                    .await;
                return Ok(ApplyUpdateOutcome::Skip);
            }
        };
        let rtt = tokio::select! {
            biased;
            _ = stop.cancelled() => return Ok(ApplyUpdateOutcome::Shutdown),
            rtt = self.health_check.latest_ready_rtt_or_probe() => rtt,
        };
        if stop.is_cancelled() {
            return Ok(ApplyUpdateOutcome::Shutdown);
        }

        let model_calibration_directives = self
            .state
            .apply_registration_update(&self.registration, update, reverse_connected, rtt)
            .await;

        Ok(ApplyUpdateOutcome::Ack(build_registration_ack(
            &self.connection,
            model_calibration_directives,
        )))
    }

    async fn close(self) {
        let Self {
            state,
            registration,
            tunnel,
            health_check,
            ..
        } = self;
        health_check.shutdown().await;
        state.end_registration(registration).await;
        // Routing teardown must finish before the exact tunnel generation is released.
        drop(tunnel);
    }
}

fn build_registration_ack(
    registration_connection_config: &RegistrationConnectionConfig,
    model_calibration_directives: Vec<ModelCalibrationDirective>,
) -> InferenceServerAck {
    let (reverse_tunnel_target, reverse_tunnel_pylon_dial_addr) =
        match &registration_connection_config.reverse_tunnel {
            Some(reverse_tunnel) => (
                reverse_tunnel.target.clone(),
                reverse_tunnel.pylon_dial_addr.clone(),
            ),
            None => (String::new(), String::new()),
        };
    InferenceServerAck {
        reverse_tunnel_target,
        reverse_tunnel_pylon_dial_addr,
        model_calibration_directives,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::auth::OpenAuthenticator;
    use crate::routing_state::RegistrationIdentity;
    use crate::tunnel::QuicTunnelConfig;
    use stargate_proto::pb::{InferenceServerModelRegistration, InferenceServerStatus, ModelStats};

    fn make_identity() -> RegistrationIdentity {
        RegistrationIdentity {
            inference_server_id: "server-1".to_string(),
            cluster_id: "server-1".to_string(),
            inference_server_url: "quic://10.0.0.1:8080".to_string(),
            routing_key: None,
            reverse_tunnel: false,
            coordinated_calibration: false,
        }
    }

    fn test_registration_connection_config() -> RegistrationConnectionConfig {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        RegistrationConnectionConfig {
            quic_proxy: Arc::new(
                QuicHttpProxy::new(
                    QuicTunnelConfig {
                        connect_timeout: Duration::from_millis(10),
                        request_timeout: Duration::from_millis(10),
                        direct_quic_connections: 1,
                        tls_cert_pem: None,
                        server_tls_identity: stargate_tls::ServerTlsIdentity::SelfSigned,
                        quic_insecure: true,
                        tunnel_protocol: Default::default(),
                    },
                    Arc::new(OpenAuthenticator),
                )
                .expect("quic proxy should initialize"),
            ),
            reverse_tunnel: None,
        }
    }

    #[tokio::test]
    async fn registration_ack_defaults_empty_reverse_tunnel_fields() {
        let ack = build_registration_ack(&test_registration_connection_config(), Vec::new());

        assert!(ack.reverse_tunnel_target.is_empty());
        assert!(ack.reverse_tunnel_pylon_dial_addr.is_empty());
        assert!(ack.model_calibration_directives.is_empty());
    }

    #[tokio::test]
    async fn registration_ack_includes_reverse_tunnel_target_and_pylon_dial_addr() {
        let config = RegistrationConnectionConfig {
            reverse_tunnel: Some(ReverseTunnelRegistrationConfig {
                target: "stargate-0.stargate-headless.stargate.svc.cluster.local:50072".to_string(),
                pylon_dial_addr: "stargate-quic-lb.stargate.svc.cluster.local:50072".to_string(),
                connect_timeout: Duration::from_millis(10),
            }),
            ..test_registration_connection_config()
        };

        let ack = build_registration_ack(&config, Vec::new());

        assert_eq!(
            ack.reverse_tunnel_target,
            "stargate-0.stargate-headless.stargate.svc.cluster.local:50072"
        );
        assert_eq!(
            ack.reverse_tunnel_pylon_dial_addr,
            "stargate-quic-lb.stargate.svc.cluster.local:50072"
        );
    }

    #[tokio::test]
    async fn registration_session_close_removes_routable_model_and_releases_identity() {
        let state = Arc::new(StargateState::default());
        let identity = make_identity();
        let update = InferenceServerRegistration {
            inference_server_id: identity.inference_server_id.clone(),
            cluster_id: String::new(),
            inference_server_url: identity.inference_server_url.clone(),
            reverse_tunnel: false,
            coordinated_calibration: false,
            models: HashMap::from([(
                "model-idle".to_string(),
                InferenceServerModelRegistration {
                    stats: Some(ModelStats::default()),
                    status: InferenceServerStatus::Active as i32,
                },
            )]),
        };
        let session = RegistrationSession::start(
            &update,
            state.clone(),
            test_registration_connection_config(),
            None,
        )
        .expect("registration session should start");
        state
            .apply_registration_update(
                &session.registration,
                &update,
                true,
                Some(Duration::from_millis(5)),
            )
            .await;

        let target = crate::routing_state::RoutingTargetKey {
            routing_key: None,
            model_id: "model-idle".to_string(),
        };
        assert_eq!(state.candidates_for_target(&target).await.len(), 1);

        session.close().await;

        assert!(state.candidates_for_target(&target).await.is_empty());
        let replacement = state
            .begin_registration(&identity)
            .expect("closing the session should release the registration identity");
        state.end_registration(replacement).await;
    }

    #[tokio::test]
    async fn registration_stream_idle_timeout_closes_admitted_session() {
        let state = Arc::new(StargateState::default());
        let identity = make_identity();
        let update = InferenceServerRegistration {
            inference_server_id: identity.inference_server_id.clone(),
            cluster_id: String::new(),
            inference_server_url: identity.inference_server_url.clone(),
            reverse_tunnel: false,
            coordinated_calibration: false,
            models: HashMap::new(),
        };
        let stream = futures::stream::pending::<Result<InferenceServerRegistration, Status>>();
        let stream = futures::stream::iter([Ok(update)]).chain(stream);
        let (tx, _rx) = flume::bounded(1);
        tokio::time::timeout(
            Duration::from_secs(2),
            process_registration_stream_with_context(
                stream,
                RegistrationStreamProcessorContext::new(
                    state.clone(),
                    test_registration_connection_config(),
                    tx,
                    AuthResult { routing_key: None },
                    Some(Duration::from_millis(1)),
                    CancellationToken::new(),
                ),
            ),
        )
        .await
        .expect("registration processor should exit after idle timeout");

        let replacement = state
            .begin_registration(&identity)
            .expect("idle timeout should release the registration identity");
        state.end_registration(replacement).await;
    }

    #[tokio::test]
    async fn registration_stream_skips_update_when_direct_connection_unavailable() {
        let state = Arc::new(StargateState::default());
        let update = InferenceServerRegistration {
            inference_server_id: "unavailable-direct".to_string(),
            cluster_id: String::new(),
            inference_server_url: "quic://127.0.0.1:1".to_string(),
            reverse_tunnel: false,
            coordinated_calibration: false,
            models: HashMap::from([(
                "model-unavailable".to_string(),
                InferenceServerModelRegistration {
                    stats: Some(ModelStats::default()),
                    status: InferenceServerStatus::Active as i32,
                },
            )]),
        };
        let target = crate::routing_state::RoutingTargetKey {
            routing_key: None,
            model_id: "model-unavailable".to_string(),
        };
        let stream = futures::stream::iter([Ok(update)]);
        let (tx, rx) = flume::bounded(1);

        process_registration_stream_with_context(
            stream,
            RegistrationStreamProcessorContext::new(
                state.clone(),
                test_registration_connection_config(),
                tx,
                AuthResult { routing_key: None },
                None,
                CancellationToken::new(),
            ),
        )
        .await;

        assert!(matches!(
            rx.try_recv(),
            Err(flume::TryRecvError::Disconnected)
        ));
        assert!(state.candidates_for_target(&target).await.is_empty());
    }

    #[tokio::test]
    async fn unavailable_direct_update_removes_existing_route_before_stream_cleanup() {
        let state = Arc::new(StargateState::default());
        let identity = RegistrationIdentity {
            inference_server_id: "lost-direct".to_string(),
            cluster_id: "lost-direct".to_string(),
            inference_server_url: "quic://127.0.0.1:1".to_string(),
            routing_key: None,
            reverse_tunnel: false,
            coordinated_calibration: false,
        };
        let update = InferenceServerRegistration {
            inference_server_id: identity.inference_server_id.clone(),
            cluster_id: identity.cluster_id.clone(),
            inference_server_url: identity.inference_server_url.clone(),
            reverse_tunnel: false,
            coordinated_calibration: false,
            models: HashMap::from([(
                "model-lost-direct".to_string(),
                InferenceServerModelRegistration {
                    stats: Some(ModelStats::default()),
                    status: InferenceServerStatus::Active as i32,
                },
            )]),
        };
        let mut session = RegistrationSession::start(
            &update,
            state.clone(),
            test_registration_connection_config(),
            None,
        )
        .expect("registration session should start");
        let target = crate::routing_state::RoutingTargetKey {
            routing_key: None,
            model_id: "model-lost-direct".to_string(),
        };
        state
            .apply_registration_update(
                &session.registration,
                &update,
                true,
                Some(Duration::from_millis(5)),
            )
            .await;
        assert_eq!(state.candidates_for_target(&target).await.len(), 1);

        let outcome = session
            .apply_update(&update, &CancellationToken::new())
            .await
            .expect("unavailable direct update should be handled");

        assert!(matches!(outcome, ApplyUpdateOutcome::Skip));
        assert!(state.candidates_for_target(&target).await.is_empty());
        session.close().await;
    }

    #[tokio::test]
    async fn admitted_registration_session_owns_health_check_until_close() {
        let state = Arc::new(StargateState::default());
        let identity = make_identity();
        let update = InferenceServerRegistration {
            inference_server_id: identity.inference_server_id.clone(),
            cluster_id: identity.cluster_id.clone(),
            inference_server_url: identity.inference_server_url.clone(),
            reverse_tunnel: identity.reverse_tunnel,
            coordinated_calibration: identity.coordinated_calibration,
            models: HashMap::new(),
        };
        let mut session =
            RegistrationSession::start(&update, state, test_registration_connection_config(), None)
                .expect("registration session should start");

        tokio::time::timeout(Duration::from_secs(1), session.health_check.changed())
            .await
            .expect("admitted session health check should publish initial pending status")
            .expect("health check sender should remain open");

        tokio::time::timeout(Duration::from_millis(200), session.close())
            .await
            .expect("session close should cancel the health-check interval wait");
    }

    #[tokio::test]
    async fn registration_stream_shutdown_interrupts_pending_stream_and_cleans_state() {
        let state = Arc::new(StargateState::default());
        let identity = make_identity();
        let update = InferenceServerRegistration {
            inference_server_id: identity.inference_server_id.clone(),
            cluster_id: String::new(),
            inference_server_url: identity.inference_server_url.clone(),
            reverse_tunnel: false,
            coordinated_calibration: false,
            models: HashMap::from([(
                "model-idle-timeout".to_string(),
                InferenceServerModelRegistration {
                    stats: Some(ModelStats::default()),
                    status: InferenceServerStatus::Active as i32,
                },
            )]),
        };

        let (polled_tx, polled_rx) = tokio::sync::oneshot::channel();
        let mut polled_tx = Some(polled_tx);
        let pending_stream = futures::stream::poll_fn(move |_cx| {
            if let Some(polled_tx) = polled_tx.take() {
                let _ = polled_tx.send(());
            }
            std::task::Poll::<Option<Result<InferenceServerRegistration, Status>>>::Pending
        });
        let stream = futures::stream::iter([Ok(update)]).chain(pending_stream);
        let (tx, _rx) = flume::bounded(1);
        let stop = CancellationToken::new();
        let processor = tokio::spawn(process_registration_stream_with_context(
            stream,
            RegistrationStreamProcessorContext::new(
                state.clone(),
                test_registration_connection_config(),
                tx,
                AuthResult { routing_key: None },
                None,
                stop.clone(),
            ),
        ));
        tokio::time::timeout(Duration::from_secs(1), polled_rx)
            .await
            .expect("registration processor should poll the stream")
            .expect("poll marker sender should be alive");

        assert!(!processor.is_finished());
        assert!(
            state.begin_registration(&identity).is_err(),
            "the admitted session should still own the registration identity"
        );

        stop.cancel();
        tokio::time::timeout(Duration::from_secs(1), processor)
            .await
            .expect("registration processor should stop after cancellation")
            .expect("registration processor should finish cleanly");
        let replacement = state
            .begin_registration(&identity)
            .expect("shutdown should release the registration identity");
        state.end_registration(replacement).await;
    }

    #[test]
    fn registration_idle_timeout_is_negotiated_from_heartbeat_metadata() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(
            REGISTRATION_HEARTBEAT_MS_METADATA,
            "120000".parse().unwrap(),
        );

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(600),
        );

        assert_eq!(timeout, Some(Duration::from_secs(360)));
    }

    #[test]
    fn registration_idle_timeout_uses_configured_floor() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(REGISTRATION_HEARTBEAT_MS_METADATA, "1000".parse().unwrap());

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(600),
        );

        assert_eq!(timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn registration_idle_timeout_zero_heartbeat_uses_configured_floor() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(REGISTRATION_HEARTBEAT_MS_METADATA, "0".parse().unwrap());

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(300),
        );

        assert_eq!(timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn registration_idle_timeout_uses_configured_cap_for_large_heartbeat() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(
            REGISTRATION_HEARTBEAT_MS_METADATA,
            "120000".parse().unwrap(),
        );

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(300),
        );

        assert_eq!(timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn registration_idle_timeout_uses_configured_cap_without_heartbeat_metadata() {
        let metadata = tonic::metadata::MetadataMap::new();

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(300),
        );

        assert_eq!(timeout, Some(Duration::from_secs(300)));
    }

    #[test]
    fn registration_idle_timeout_honors_configured_cap_below_floor() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(REGISTRATION_HEARTBEAT_MS_METADATA, "1000".parse().unwrap());

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(10),
        );

        assert_eq!(timeout, Some(Duration::from_secs(10)));

        let timeout = negotiated_registration_update_idle_timeout(
            &tonic::metadata::MetadataMap::new(),
            Duration::from_secs(60),
            Duration::from_secs(10),
        );

        assert_eq!(timeout, Some(Duration::from_secs(10)));
    }

    #[test]
    fn registration_idle_timeout_can_be_disabled() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(REGISTRATION_HEARTBEAT_MS_METADATA, "1000".parse().unwrap());

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::ZERO,
            Duration::from_secs(300),
        );

        assert_eq!(timeout, None);
    }

    #[test]
    fn registration_idle_timeout_max_zero_disables_enforcement() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(REGISTRATION_HEARTBEAT_MS_METADATA, "1000".parse().unwrap());

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::ZERO,
        );

        assert_eq!(timeout, None);
    }

    #[test]
    fn malformed_registration_heartbeat_metadata_uses_configured_cap() {
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert(
            REGISTRATION_HEARTBEAT_MS_METADATA,
            "not-a-number".parse().unwrap(),
        );

        let timeout = negotiated_registration_update_idle_timeout(
            &metadata,
            Duration::from_secs(60),
            Duration::from_secs(300),
        );

        assert_eq!(timeout, Some(Duration::from_secs(300)));
    }
}
