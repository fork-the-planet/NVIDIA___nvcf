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
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt, stream};
use tokio::sync::{mpsc, watch};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

use stargate_forwarding::{ForwardingResolver, PeerResolution, PeerTarget};

use crate::auth::WorkerAuthenticator;
use crate::discovery::Discovery;
use crate::routing_state::StargateState;
use stargate_runtime::CriticalTaskGroup;

use stargate_proto::pb::stargate_control_plane_client::StargateControlPlaneClient;
use stargate_proto::pb::stargate_control_plane_server::StargateControlPlane;
use stargate_proto::pb::stargate_model_discovery_server::StargateModelDiscovery;
use stargate_proto::pb::{
    CalibrationState, InferenceServerAck, InferenceServerRegistration, ListModelsRequest,
    ListModelsResponse, SubmitClusterCalibrationRequest, SubmitClusterCalibrationResponse,
    WatchStargatesRequest, WatchStargatesResponse,
};

mod registration;
mod watch_stargates;

pub use self::registration::{
    DEFAULT_REGISTRATION_UPDATE_IDLE_TIMEOUT, DEFAULT_REGISTRATION_UPDATE_MAX_IDLE_TIMEOUT,
};
pub(crate) use self::registration::{
    RegistrationConnectionConfig, ReverseTunnelRegistrationConfig,
};
use self::registration::{
    negotiated_registration_update_idle_timeout, process_registration_stream,
};
use self::watch_stargates::{
    WatchStargatesPublisherConfig, spawn_watch_stargates_publisher,
    watch_stargates_stream_from_receiver,
};

#[derive(Clone)]
pub struct StargateService {
    stargate_id: String,
    advertise_addr: SocketAddr,
    discovery_dns_name: String,
    watch_stargates_rx: watch::Receiver<WatchStargatesResponse>,
    state: Arc<StargateState>,
    registration_connection_config: RegistrationConnectionConfig,
    registration_update_idle_timeout: Duration,
    registration_update_max_idle_timeout: Duration,
    forwarding: Option<Arc<dyn ForwardingResolver>>,
    authenticator: Arc<dyn WorkerAuthenticator>,
    tasks: CriticalTaskGroup,
}

pub(crate) struct StargateServiceConfig {
    pub(crate) stargate_id: String,
    pub(crate) advertise_addr: SocketAddr,
    pub(crate) discovery_dns_name: String,
    pub(crate) discovery: Box<dyn Discovery>,
    pub(crate) remote_watch_stargate_urls: Vec<String>,
    pub(crate) grpc_pylon_dial_addr: Option<String>,
    pub(crate) discovery_poll_interval: Duration,
    pub(crate) watch_heartbeat_interval: Duration,
    pub(crate) tasks: CriticalTaskGroup,
    pub(crate) registration_update_idle_timeout: Duration,
    pub(crate) registration_update_max_idle_timeout: Duration,
    pub(crate) state: Arc<StargateState>,
    pub(crate) registration_connection_config: RegistrationConnectionConfig,
    pub(crate) forwarding: Option<Arc<dyn ForwardingResolver>>,
    pub(crate) authenticator: Arc<dyn WorkerAuthenticator>,
}

impl StargateService {
    pub(crate) fn new(config: StargateServiceConfig) -> Self {
        if !config.registration_update_idle_timeout.is_zero()
            && !config.registration_update_max_idle_timeout.is_zero()
            && config.registration_update_max_idle_timeout < config.registration_update_idle_timeout
        {
            warn!(
                registration_update_idle_timeout_ms =
                    config.registration_update_idle_timeout.as_millis(),
                registration_update_max_idle_timeout_ms =
                    config.registration_update_max_idle_timeout.as_millis(),
                "registration update max idle timeout is below the idle-timeout floor; max cap wins"
            );
        }

        let tasks = config.tasks.clone();
        let watch_stargates_rx = spawn_watch_stargates_publisher(WatchStargatesPublisherConfig {
            advertise_addr: config.advertise_addr,
            discovery_dns_name: config.discovery_dns_name.clone(),
            discovery: config.discovery,
            remote_watch_stargate_urls: config.remote_watch_stargate_urls,
            grpc_pylon_dial_addr: config.grpc_pylon_dial_addr,
            discovery_poll_interval: config.discovery_poll_interval,
            watch_heartbeat_interval: config.watch_heartbeat_interval,
            tasks: config.tasks,
        });

        Self {
            stargate_id: config.stargate_id,
            advertise_addr: config.advertise_addr,
            discovery_dns_name: config.discovery_dns_name,
            watch_stargates_rx,
            state: config.state,
            registration_connection_config: config.registration_connection_config,
            registration_update_idle_timeout: config.registration_update_idle_timeout,
            registration_update_max_idle_timeout: config.registration_update_max_idle_timeout,
            forwarding: config.forwarding,
            authenticator: config.authenticator,
            tasks,
        }
    }

    fn peer_grpc_target<T>(&self, request: &Request<T>) -> PeerGrpcTarget {
        let Some(fwd) = self.forwarding.as_ref() else {
            return PeerGrpcTarget::Local;
        };
        let Some(authority) = request.extensions().get::<http::uri::Authority>() else {
            return PeerGrpcTarget::Local;
        };
        let host = authority.host();
        match fwd.resolve_peer(host, self.advertise_addr.port()) {
            PeerResolution::Peer(target) => PeerGrpcTarget::Peer(target),
            PeerResolution::Local | PeerResolution::NotPeer => PeerGrpcTarget::Local,
        }
    }

    async fn connect_peer(
        addr: &str,
    ) -> Result<StargateControlPlaneClient<tonic::transport::Channel>, Status> {
        let endpoint = if addr.starts_with("http") {
            addr.to_string()
        } else {
            format!("http://{addr}")
        };
        let channel = tonic::transport::Channel::from_shared(endpoint)
            .map_err(|e| Status::internal(format!("invalid peer address: {e}")))?
            .connect()
            .await
            .map_err(|e| Status::unavailable(format!("failed to connect to peer: {e}")))?;
        Ok(StargateControlPlaneClient::new(channel))
    }

    pub fn state(&self) -> Arc<StargateState> {
        self.state.clone()
    }
}

fn registration_message_stream<S>(
    inbound: S,
) -> (
    impl Stream<Item = InferenceServerRegistration>,
    RegistrationStreamErrorRx,
)
where
    S: Stream<Item = Result<InferenceServerRegistration, Status>> + Send + 'static,
{
    let (stream_error_tx, stream_error_rx) = mpsc::channel(1);
    let messages = inbound.filter_map(move |result| {
        let stream_error_tx = stream_error_tx.clone();
        async move {
            match result {
                Ok(message) => Some(message),
                Err(error) => {
                    warn!(
                        error = %error,
                        "forwarded registration stream read error, forwarding stream error"
                    );
                    let _ = stream_error_tx.send(error).await;
                    None
                }
            }
        }
    });
    (messages, stream_error_rx)
}

async fn pending_registration_stream_error(
    stream_error_rx: &mut RegistrationStreamErrorRx,
) -> Option<Status> {
    match stream_error_rx.try_recv() {
        Ok(error) => Some(error),
        Err(mpsc::error::TryRecvError::Empty) => stream_error_rx.recv().await,
        Err(mpsc::error::TryRecvError::Disconnected) => None,
    }
}

enum PeerGrpcTarget {
    Local,
    Peer(PeerTarget),
}

type WatchStargatesStream =
    Pin<Box<dyn Stream<Item = Result<WatchStargatesResponse, Status>> + Send + 'static>>;
type RegisterInferenceServerStream =
    Pin<Box<dyn Stream<Item = Result<InferenceServerAck, Status>> + Send + 'static>>;
type RegistrationStreamErrorRx = mpsc::Receiver<Status>;

#[tonic::async_trait]
impl StargateControlPlane for StargateService {
    type WatchStargatesStream = WatchStargatesStream;
    type RegisterInferenceServerStream = RegisterInferenceServerStream;

    async fn watch_stargates(
        &self,
        request: Request<WatchStargatesRequest>,
    ) -> Result<Response<Self::WatchStargatesStream>, Status> {
        match self.peer_grpc_target(&request) {
            PeerGrpcTarget::Peer(peer) => {
                info!(
                    peer = %peer.dial_addr,
                    server_name = %peer.server_name,
                    "forwarding watch_stargates to peer"
                );
                let mut peer_client = Self::connect_peer(&peer.dial_addr).await?;
                let resp = peer_client
                    .watch_stargates(WatchStargatesRequest {})
                    .await?;
                let mut inner = resp.into_inner();
                let stream = async_stream::stream! {
                    let _client = peer_client;
                    while let Some(msg) = inner.message().await.transpose() {
                        yield msg;
                    }
                };
                return Ok(Response::new(Box::pin(stream)));
            }
            PeerGrpcTarget::Local => {}
        }

        info!(
            stargate_id = %self.stargate_id,
            advertise_addr = %self.advertise_addr,
            dns_name = %self.discovery_dns_name,
            "watch stargates stream opened"
        );

        let out = watch_stargates_stream_from_receiver(self.watch_stargates_rx.clone());

        Ok(Response::new(Box::pin(out)))
    }

    async fn register_inference_server(
        &self,
        request: Request<tonic::Streaming<InferenceServerRegistration>>,
    ) -> Result<Response<Self::RegisterInferenceServerStream>, Status> {
        match self.peer_grpc_target(&request) {
            PeerGrpcTarget::Peer(peer) => {
                info!(
                    peer = %peer.dial_addr,
                    server_name = %peer.server_name,
                    "forwarding register_inference_server to peer"
                );
                let mut peer_client = Self::connect_peer(&peer.dial_addr).await?;
                let metadata = request.metadata().clone();
                let (inbound, mut stream_error_rx) =
                    registration_message_stream(request.into_inner());
                let mut forwarded = Request::new(inbound);
                *forwarded.metadata_mut() = metadata;
                let resp = peer_client.register_inference_server(forwarded).await?;
                let mut inner = resp.into_inner();
                let stream = async_stream::stream! {
                    let _client = peer_client;
                    let mut stream_error_rx_open = true;
                    loop {
                        tokio::select! {
                            error = stream_error_rx.recv(), if stream_error_rx_open => {
                                match error {
                                    Some(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                    None => stream_error_rx_open = false,
                                }
                            }
                            message = inner.message() => {
                                match message {
                                    Ok(Some(message)) => yield Ok(message),
                                    Ok(None) => {
                                        if let Some(error) =
                                            pending_registration_stream_error(&mut stream_error_rx).await
                                        {
                                            yield Err(error);
                                        }
                                        break;
                                    }
                                    Err(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                };
                return Ok(Response::new(Box::pin(stream)));
            }
            PeerGrpcTarget::Local => {}
        }

        let token = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        let auth_result = self.authenticator.authenticate(token).await.map_err(|e| {
            warn!(error = %e, "gRPC registration authentication failed");
            Status::unauthenticated("authentication failed")
        })?;

        info!(
            stargate_id = %self.stargate_id,
            "register inference servers stream opened"
        );

        let (tx, rx) = flume::bounded::<Result<InferenceServerAck, Status>>(16);
        let state = self.state.clone();
        let registration_connection_config = self.registration_connection_config.clone();
        let idle_timeout = negotiated_registration_update_idle_timeout(
            request.metadata(),
            self.registration_update_idle_timeout,
            self.registration_update_max_idle_timeout,
        );
        let stop = self.tasks.shutdown_signal();

        self.tasks.task_tracker().spawn(async move {
            process_registration_stream(
                request.into_inner(),
                state,
                registration_connection_config,
                tx,
                auth_result,
                idle_timeout,
                stop,
            )
            .await;
            info!("register inference servers stream closed");
        });

        let out = stream::unfold(rx, |rx| async move {
            match rx.recv_async().await {
                Ok(item) => Some((item, rx)),
                Err(_) => None,
            }
        });

        Ok(Response::new(Box::pin(out)))
    }

    async fn submit_cluster_calibration(
        &self,
        request: Request<SubmitClusterCalibrationRequest>,
    ) -> Result<Response<SubmitClusterCalibrationResponse>, Status> {
        match self.peer_grpc_target(&request) {
            PeerGrpcTarget::Peer(peer) => {
                info!(
                    peer = %peer.dial_addr,
                    server_name = %peer.server_name,
                    "forwarding submit_cluster_calibration to peer"
                );
                let mut peer_client = Self::connect_peer(&peer.dial_addr).await?;
                let metadata = request.metadata().clone();
                let mut forwarded = Request::new(request.into_inner());
                *forwarded.metadata_mut() = metadata;
                return peer_client.submit_cluster_calibration(forwarded).await;
            }
            PeerGrpcTarget::Local => {}
        }

        let token = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        let auth_result = self
            .authenticator
            .authenticate(token)
            .await
            .map_err(|error| {
                warn!(error = %error, "gRPC calibration submission authentication failed");
                Status::unauthenticated("authentication failed")
            })?;

        self.state
            .submit_cluster_calibration(auth_result.routing_key, request.get_ref())
            .await?;

        Ok(Response::new(SubmitClusterCalibrationResponse {
            state: CalibrationState::Complete as i32,
        }))
    }
}

#[tonic::async_trait]
impl StargateModelDiscovery for StargateService {
    async fn list_models(
        &self,
        request: Request<ListModelsRequest>,
    ) -> Result<Response<ListModelsResponse>, Status> {
        let requested = normalize_list_models_request(request.into_inner())
            .map_err(Status::invalid_argument)?;
        let model_id_filter_count = requested.model_ids.len();
        let model_ids = self
            .state
            .list_active_models(requested.routing_key.as_deref(), &requested.model_ids)
            .await;

        debug!(
            routing_key = ?requested.routing_key,
            model_id_filter_count,
            return_all_models = model_id_filter_count == 0,
            returned_model_count = model_ids.len(),
            "list_models completed"
        );

        Ok(Response::new(ListModelsResponse { model_ids }))
    }
}

#[derive(Debug)]
struct NormalizedListModelsRequest {
    routing_key: Option<String>,
    model_ids: Vec<String>,
}

fn normalize_list_models_request(
    request: ListModelsRequest,
) -> Result<NormalizedListModelsRequest, &'static str> {
    let ListModelsRequest {
        routing_key,
        model_ids,
    } = request;
    let routing_key = routing_key
        .as_deref()
        .map(str::trim)
        .filter(|routing_key| !routing_key.is_empty())
        .map(ToOwned::to_owned);

    let mut normalized_model_ids = Vec::with_capacity(model_ids.len());
    for model_id in model_ids {
        let model_id = model_id.trim();
        if model_id.is_empty() {
            return Err("model_ids must not contain empty values");
        }
        normalized_model_ids.push(model_id.to_string());
    }

    Ok(NormalizedListModelsRequest {
        routing_key,
        model_ids: normalized_model_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalize_list_models_request_trims_model_filters() {
        let request = normalize_list_models_request(ListModelsRequest {
            routing_key: None,
            model_ids: vec![" model-a ".to_string(), "model-b".to_string()],
        })
        .expect("valid filters should normalize");

        assert_eq!(request.model_ids, vec!["model-a", "model-b"]);
        assert_eq!(request.routing_key, None);
    }

    #[test]
    fn normalize_list_models_request_trims_routing_key() {
        let request = normalize_list_models_request(ListModelsRequest {
            routing_key: Some(" rk-a ".to_string()),
            model_ids: Vec::new(),
        })
        .expect("valid routing key should normalize");

        assert_eq!(request.routing_key.as_deref(), Some("rk-a"));
    }

    #[test]
    fn normalize_list_models_request_treats_blank_routing_key_as_none() {
        let request = normalize_list_models_request(ListModelsRequest {
            routing_key: Some(" ".to_string()),
            model_ids: Vec::new(),
        })
        .expect("blank routing key should normalize to unscoped");

        assert_eq!(request.routing_key, None);
    }

    #[test]
    fn normalize_list_models_request_allows_empty_filter() {
        let request = normalize_list_models_request(ListModelsRequest {
            routing_key: None,
            model_ids: Vec::new(),
        })
        .expect("empty model filter should request all models");

        assert!(request.model_ids.is_empty());
    }

    #[test]
    fn normalize_list_models_request_rejects_blank_model_filter() {
        let error = normalize_list_models_request(ListModelsRequest {
            routing_key: None,
            model_ids: vec![" ".to_string()],
        })
        .expect_err("blank model filter should be rejected");

        assert_eq!(error, "model_ids must not contain empty values");
    }

    #[tokio::test]
    async fn registration_message_stream_reports_inbound_stream_errors() {
        let (messages, mut errors) = registration_message_stream(futures::stream::iter([Err(
            Status::cancelled("client cancelled registration stream"),
        )]));
        futures::pin_mut!(messages);

        assert!(
            messages.next().await.is_none(),
            "stream errors must not be converted into registration messages"
        );
        let error = errors
            .recv()
            .await
            .expect("inbound stream error should be retained for response termination");
        assert_eq!(error.code(), tonic::Code::Cancelled);
        assert_eq!(error.message(), "client cancelled registration stream");
    }

    #[tokio::test]
    async fn pending_registration_stream_error_returns_buffered_error() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.send(Status::cancelled("buffered registration stream error"))
            .await
            .expect("receiver should be alive");

        let error = pending_registration_stream_error(&mut rx)
            .await
            .expect("buffered stream error should be returned");

        assert_eq!(error.code(), tonic::Code::Cancelled);
        assert_eq!(error.message(), "buffered registration stream error");
    }

    #[tokio::test]
    async fn pending_registration_stream_error_waits_for_delayed_error() {
        let (tx, mut rx) = mpsc::channel(1);
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            tx.send(Status::internal("delayed registration stream error"))
                .await
                .expect("receiver should be alive");
        });

        let error = tokio::time::timeout(
            Duration::from_secs(1),
            pending_registration_stream_error(&mut rx),
        )
        .await
        .expect("pending stream error should arrive")
        .expect("delayed stream error should be returned");

        assert_eq!(error.code(), tonic::Code::Internal);
        assert_eq!(error.message(), "delayed registration stream error");
    }

    #[tokio::test]
    async fn pending_registration_stream_error_returns_none_after_disconnect() {
        let (tx, mut rx) = mpsc::channel(1);
        drop(tx);

        let error = pending_registration_stream_error(&mut rx).await;

        assert!(error.is_none());
    }
}
