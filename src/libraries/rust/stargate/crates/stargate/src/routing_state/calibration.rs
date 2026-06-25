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

use super::keys::RegistrationIdentity;
use super::*;

#[derive(Debug, Default)]
pub(super) struct ClusterCalibrations {
    assignments: SccHashMap<String, ClusterCalibrationState>,
}

#[derive(Debug)]
enum ClusterCalibrationState {
    Assigned {
        owner_inference_server_id: String,
        assignment_token: String,
    },
    Complete {
        owner_inference_server_id: String,
        assignment_token: String,
        last_mean_input_tps: f64,
    },
}

pub(super) enum RegistrationCalibrationDecision {
    Disabled,
    Pending(ModelCalibrationDirective),
    Complete(ModelCalibrationDirective),
}

impl RegistrationCalibrationDecision {
    pub(super) fn into_parts(self) -> (Option<ModelCalibrationDirective>, bool) {
        match self {
            Self::Disabled => (None, false),
            Self::Pending(directive) => (Some(directive), true),
            Self::Complete(directive) => (Some(directive), false),
        }
    }
}

impl ClusterCalibrationState {
    fn registration_decision(
        &self,
        identity: &RegistrationIdentity,
        model_id: &str,
    ) -> RegistrationCalibrationDecision {
        match self {
            Self::Assigned {
                owner_inference_server_id,
                assignment_token,
            } if owner_inference_server_id == &identity.inference_server_id => {
                RegistrationCalibrationDecision::Pending(calibration_directive(
                    model_id,
                    CalibrationState::Run,
                    assignment_token.clone(),
                ))
            }
            Self::Assigned { .. } => RegistrationCalibrationDecision::Pending(
                calibration_directive(model_id, CalibrationState::Waiting, String::new()),
            ),
            Self::Complete { .. } => RegistrationCalibrationDecision::Complete(
                calibration_directive(model_id, CalibrationState::Complete, String::new()),
            ),
        }
    }

    fn submit_validated(
        &mut self,
        request: &SubmitClusterCalibrationRequest,
    ) -> Result<(), Status> {
        match self {
            Self::Assigned {
                owner_inference_server_id,
                assignment_token,
            } if owner_inference_server_id == &request.inference_server_id
                && assignment_token == &request.assignment_token =>
            {
                *self = Self::Complete {
                    owner_inference_server_id: request.inference_server_id.clone(),
                    assignment_token: request.assignment_token.clone(),
                    last_mean_input_tps: request.measured_last_mean_input_tps,
                };
                Ok(())
            }
            Self::Complete {
                owner_inference_server_id,
                assignment_token,
                last_mean_input_tps,
            } if owner_inference_server_id == &request.inference_server_id
                && assignment_token == &request.assignment_token
                && last_mean_input_tps.to_bits()
                    == request.measured_last_mean_input_tps.to_bits() =>
            {
                Ok(())
            }
            Self::Assigned { .. } => Err(Status::failed_precondition(
                "cluster calibration submission does not own the local assignment",
            )),
            Self::Complete { .. } => Err(Status::failed_precondition(
                "cluster calibration was already completed by another submission",
            )),
        }
    }

    fn completed_last_mean_input_tps(&self) -> Option<f64> {
        match self {
            Self::Complete {
                last_mean_input_tps,
                ..
            } => Some(*last_mean_input_tps),
            Self::Assigned { .. } => None,
        }
    }
}

impl ClusterCalibrations {
    pub(super) async fn registration_decision(
        &self,
        identity: &RegistrationIdentity,
        model_id: &str,
    ) -> RegistrationCalibrationDecision {
        if !identity.coordinated_calibration {
            return RegistrationCalibrationDecision::Disabled;
        }

        loop {
            if let Some(existing) = self
                .assignments
                .read_async(model_id, |_model_id, state| {
                    state.registration_decision(identity, model_id)
                })
                .await
            {
                return existing;
            }

            let assignment_token = new_cluster_calibration_assignment_token();
            let initial_state = ClusterCalibrationState::Assigned {
                owner_inference_server_id: identity.inference_server_id.clone(),
                assignment_token: assignment_token.clone(),
            };
            if self
                .assignments
                .insert_async(model_id.to_string(), initial_state)
                .await
                .is_ok()
            {
                return RegistrationCalibrationDecision::Pending(calibration_directive(
                    model_id,
                    CalibrationState::Run,
                    assignment_token,
                ));
            }
        }
    }

    pub(super) async fn submit_validated(
        &self,
        request: &SubmitClusterCalibrationRequest,
    ) -> Result<(), Status> {
        self.assignments
            .update_async(request.model_id.as_str(), |_model_id, state| {
                state.submit_validated(request)
            })
            .await
            .unwrap_or_else(|| {
                Err(Status::failed_precondition(
                    "cluster calibration has no active local assignment",
                ))
            })
    }

    pub(super) async fn completed_last_mean_input_tps(&self, model_id: &str) -> Option<f64> {
        self.assignments
            .read_async(model_id, |_model_id, state| {
                state.completed_last_mean_input_tps()
            })
            .await
            .flatten()
    }

    pub(super) async fn release_owned_assignments(
        &self,
        identity: &RegistrationIdentity,
        model_ids: &BTreeSet<String>,
    ) {
        for model_id in model_ids {
            let owner_inference_server_id = identity.inference_server_id.clone();
            let _ = self
                .assignments
                .remove_if_async(model_id.as_str(), move |state| {
                    matches!(
                        state,
                        ClusterCalibrationState::Assigned {
                            owner_inference_server_id: owner,
                            ..
                        } if owner == &owner_inference_server_id
                    )
                })
                .await;
        }
    }
}

pub(super) fn validate_cluster_calibration_submission(
    request: &SubmitClusterCalibrationRequest,
) -> Result<(), Status> {
    if request.inference_server_id.is_empty()
        || request.cluster_id.is_empty()
        || request.model_id.is_empty()
        || request.assignment_token.is_empty()
    {
        return Err(Status::invalid_argument(
            "cluster calibration submission identity and assignment token must be non-empty",
        ));
    }
    if !valid_last_mean_input_tps(request.measured_last_mean_input_tps) {
        return Err(Status::invalid_argument(
            "measured_last_mean_input_tps must be positive and finite",
        ));
    }
    Ok(())
}

pub(super) fn valid_last_mean_input_tps(last_mean_input_tps: f64) -> bool {
    last_mean_input_tps > 0.0 && last_mean_input_tps.is_finite()
}

fn new_cluster_calibration_assignment_token() -> String {
    format!("{:032x}", rand::random::<u128>())
}

fn calibration_directive(
    model_id: &str,
    state: CalibrationState,
    assignment_token: String,
) -> ModelCalibrationDirective {
    ModelCalibrationDirective {
        model_id: model_id.to_string(),
        state: state as i32,
        assignment_token,
    }
}
