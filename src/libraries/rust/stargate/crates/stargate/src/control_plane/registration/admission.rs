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

use std::net::IpAddr;

use tonic::Status;
use tracing::warn;
use url::Url;

use crate::routing_state::RegistrationIdentity;

use stargate_proto::pb::InferenceServerRegistration;

pub(super) fn admit_initial_registration(
    update: &InferenceServerRegistration,
    reverse_tunnel_configured: bool,
    routing_key: Option<&str>,
) -> Result<RegistrationIdentity, Status> {
    if update.inference_server_id.is_empty() {
        warn!("inference_server_id is empty; denying registration");
        return Err(Status::invalid_argument("inference_server_id is empty"));
    }

    if update.inference_server_url.is_empty() {
        warn!(
            inference_server_id = %update.inference_server_id,
            "inference_server_url is empty; denying registration"
        );
        return Err(Status::invalid_argument("inference_server_url is empty"));
    }

    let url_validation = if update.reverse_tunnel {
        validate_reverse_tunnel_inference_server_url(&update.inference_server_url)
    } else {
        validate_inference_server_url(&update.inference_server_url)
    };
    if let Err(error) = url_validation {
        warn!(
            inference_server_id = %update.inference_server_id,
            inference_server_url = %update.inference_server_url,
            reverse_tunnel = update.reverse_tunnel,
            error = %error,
            "invalid inference_server_url; denying registration"
        );
        return Err(Status::invalid_argument(format!(
            "invalid inference_server_url: {error}"
        )));
    }

    if update.reverse_tunnel && !reverse_tunnel_configured {
        warn!(
            inference_server_id = %update.inference_server_id,
            "reverse tunnel flag set but no reverse tunnel config; denying registration"
        );
        return Err(Status::invalid_argument(
            "reverse tunnel flag set but no reverse tunnel config",
        ));
    }

    Ok(RegistrationIdentity {
        inference_server_id: update.inference_server_id.clone(),
        cluster_id: effective_cluster_id(update),
        inference_server_url: update.inference_server_url.clone(),
        routing_key: routing_key.map(ToOwned::to_owned),
        reverse_tunnel: update.reverse_tunnel,
        coordinated_calibration: update.coordinated_calibration,
    })
}

pub(super) fn validate_running_update(
    identity: &RegistrationIdentity,
    update: &InferenceServerRegistration,
) -> Option<Status> {
    if update.reverse_tunnel != identity.reverse_tunnel {
        warn!(
            inference_server_id = %update.inference_server_id,
            reverse_tunnel = %update.reverse_tunnel,
            "reverse tunnel flag changed; denying registration"
        );
        return Some(Status::invalid_argument("reverse tunnel flag changed"));
    }
    if update.inference_server_url != identity.inference_server_url {
        warn!(
            inference_server_id = %update.inference_server_id,
            inference_server_url = %update.inference_server_url,
            "inference_server_url changed; denying registration"
        );
        return Some(Status::invalid_argument("inference_server_url changed"));
    }
    if update.inference_server_id != identity.inference_server_id {
        warn!(
            inference_server_id = %update.inference_server_id,
            "inference_server_id changed; denying registration"
        );
        return Some(Status::invalid_argument("inference_server_id changed"));
    }
    if effective_cluster_id(update) != identity.cluster_id {
        warn!(
            inference_server_id = %update.inference_server_id,
            cluster_id = %update.cluster_id,
            "cluster_id changed; denying registration"
        );
        return Some(Status::invalid_argument("cluster_id changed"));
    }
    if update.coordinated_calibration != identity.coordinated_calibration {
        warn!(
            inference_server_id = %update.inference_server_id,
            coordinated_calibration = update.coordinated_calibration,
            "coordinated_calibration changed; denying registration"
        );
        return Some(Status::invalid_argument("coordinated_calibration changed"));
    }
    None
}

fn effective_cluster_id(update: &InferenceServerRegistration) -> String {
    if update.cluster_id.is_empty() {
        update.inference_server_id.clone()
    } else {
        update.cluster_id.clone()
    }
}

fn validate_inference_server_url(url: &str) -> Result<(), anyhow::Error> {
    use anyhow::Context;
    let parsed = Url::parse(url).context("inference_server_url must be a valid URL")?;
    if parsed.scheme() != "quic" {
        anyhow::bail!("inference_server_url scheme must be quic");
    }
    if parsed.host_str().is_none() {
        anyhow::bail!("inference_server_url must include host");
    }
    if parsed
        .host_str()
        .and_then(|host| host.parse::<IpAddr>().ok())
        .is_none()
    {
        anyhow::bail!("inference_server_url host must be an IP address");
    }
    if parsed.port_or_known_default().is_none() {
        anyhow::bail!("inference_server_url must include port");
    }
    Ok(())
}

fn validate_reverse_tunnel_inference_server_url(url: &str) -> Result<(), anyhow::Error> {
    use anyhow::Context;
    let parsed = Url::parse(url).context("inference_server_url must be a valid URL")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("inference_server_url scheme must be http or https");
    }
    if parsed.host_str().is_none() {
        anyhow::bail!("inference_server_url must include host");
    }
    if parsed.port_or_known_default().is_none() {
        anyhow::bail!("inference_server_url must include port");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(id: &str, url: &str, reverse_tunnel: bool) -> InferenceServerRegistration {
        InferenceServerRegistration {
            inference_server_id: id.to_string(),
            cluster_id: String::new(),
            inference_server_url: url.to_string(),
            reverse_tunnel,
            models: Default::default(),
            coordinated_calibration: false,
        }
    }

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

    fn assert_invalid_argument(status: Status, expected_message: &str) {
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains(expected_message), "got: {status}");
    }

    #[test]
    fn initial_registration_rejects_empty_id() {
        let update = make_update("", "quic://10.0.0.1:8080", false);

        let status = admit_initial_registration(&update, false, None)
            .expect_err("empty inference_server_id should be rejected");

        assert_invalid_argument(status, "inference_server_id is empty");
    }

    #[test]
    fn initial_registration_rejects_empty_url() {
        let update = make_update("server-1", "", true);

        let status = admit_initial_registration(&update, true, None)
            .expect_err("empty inference_server_url should be rejected");

        assert_invalid_argument(status, "inference_server_url is empty");
    }

    #[test]
    fn direct_registration_accepts_ip_quic_url_and_derives_identity() {
        let update = InferenceServerRegistration {
            cluster_id: "cluster-a".to_string(),
            coordinated_calibration: true,
            ..make_update("server-1", "quic://10.0.0.1:8080", false)
        };

        let identity = admit_initial_registration(&update, false, Some("tenant-a"))
            .expect("valid direct registration should be admitted");

        assert_eq!(identity.inference_server_id, "server-1");
        assert_eq!(identity.cluster_id, "cluster-a");
        assert_eq!(identity.inference_server_url, "quic://10.0.0.1:8080");
        assert_eq!(identity.routing_key.as_deref(), Some("tenant-a"));
        assert!(!identity.reverse_tunnel);
        assert!(identity.coordinated_calibration);
    }

    #[test]
    fn direct_registration_defaults_empty_cluster_id_to_server_id() {
        let update = make_update("server-1", "quic://10.0.0.1:8080", false);

        let identity = admit_initial_registration(&update, false, None)
            .expect("valid direct registration should be admitted");

        assert_eq!(identity.cluster_id, "server-1");
    }

    #[test]
    fn validate_inference_server_url_rejects_http() {
        let result = validate_inference_server_url("http://10.0.0.1:8080");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("scheme must be quic"), "got: {msg}");
    }

    #[test]
    fn validate_inference_server_url_rejects_missing_port() {
        let result = validate_inference_server_url("quic://10.0.0.1");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("must include port"), "got: {msg}");
    }

    #[test]
    fn validate_inference_server_url_accepts_ip_host() {
        validate_inference_server_url("quic://10.0.0.1:8080")
            .expect("direct quic URL with IP host and port should be valid");
    }

    #[test]
    fn validate_inference_server_url_rejects_hostname_host() {
        let result = validate_inference_server_url("quic://backend.default.svc:8080");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("host must be an IP address"), "got: {msg}");
    }

    #[test]
    fn validate_inference_server_url_rejects_garbage() {
        let result = validate_inference_server_url("not a url at all");
        assert!(result.is_err());
    }

    #[test]
    fn reverse_tunnel_registration_rejects_non_http_url() {
        let update = make_update("server-1", "quic://10.0.0.1:8080", true);

        let status = admit_initial_registration(&update, true, None)
            .expect_err("reverse-tunnel registration with non-HTTP URL should be rejected");

        assert_invalid_argument(status, "inference_server_url scheme must be http or https");
    }

    #[test]
    fn reverse_tunnel_registration_requires_reverse_tunnel_config() {
        let update = make_update("server-1", "http://backend.default.svc:8080", true);

        let status = admit_initial_registration(&update, false, None)
            .expect_err("reverse-tunnel registration without config should be rejected");

        assert_invalid_argument(
            status,
            "reverse tunnel flag set but no reverse tunnel config",
        );
    }

    #[test]
    fn reverse_tunnel_registration_accepts_http_url_when_configured() {
        let update = make_update("server-1", "http://backend.default.svc:8080", true);

        let identity = admit_initial_registration(&update, true, None)
            .expect("valid reverse tunnel registration should be admitted");

        assert!(identity.reverse_tunnel);
        assert_eq!(
            identity.inference_server_url,
            "http://backend.default.svc:8080"
        );
    }

    #[test]
    fn validate_running_update_rejects_changed_url() {
        let identity = make_identity();
        let update = make_update("server-1", "quic://10.0.0.2:9090", false);
        let status = validate_running_update(&identity, &update);
        assert!(status.is_some());
        assert_eq!(status.unwrap().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn validate_running_update_rejects_changed_id() {
        let identity = make_identity();
        let update = make_update("server-2", "quic://10.0.0.1:8080", false);
        let status = validate_running_update(&identity, &update);
        assert!(status.is_some());
        assert_eq!(status.unwrap().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn validate_running_update_rejects_toggled_reverse_tunnel() {
        let identity = make_identity();
        let update = make_update("server-1", "quic://10.0.0.1:8080", true);
        let status = validate_running_update(&identity, &update);
        assert!(status.is_some());
        assert_eq!(status.unwrap().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn validate_running_update_rejects_changed_cluster_id() {
        let identity = make_identity();
        let update = InferenceServerRegistration {
            cluster_id: "different-cluster".to_string(),
            ..make_update("server-1", "quic://10.0.0.1:8080", false)
        };

        let status = validate_running_update(&identity, &update);

        assert!(status.is_some());
        assert_eq!(status.unwrap().code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn validate_running_update_rejects_changed_coordinated_calibration() {
        let identity = make_identity();
        let update = InferenceServerRegistration {
            coordinated_calibration: true,
            ..make_update("server-1", "quic://10.0.0.1:8080", false)
        };

        let status = validate_running_update(&identity, &update);

        assert!(status.is_some());
        assert_eq!(status.unwrap().code(), tonic::Code::InvalidArgument);
    }
}
