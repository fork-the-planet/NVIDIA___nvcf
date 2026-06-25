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

use reqwest::header::HeaderMap;

use crate::runtime_state::PylonRuntimeState;

use super::embeddings::{EmbeddingsRequestObserver, embedding_items_from_request_body};
use super::{RequestObservationEndpoint, RequestObserver, RequiredTunnelHeaders};

pub(crate) struct TunnelRequestObserver {
    kind: TunnelRequestObserverKind,
}

enum TunnelRequestObserverKind {
    Generation(RequestObserver),
    Embeddings(EmbeddingsRequestObserver),
}

impl TunnelRequestObserver {
    pub(crate) fn accepted(
        endpoint: RequestObservationEndpoint,
        required: RequiredTunnelHeaders,
        runtime_state: PylonRuntimeState,
    ) -> Self {
        let kind =
            match endpoint {
                RequestObservationEndpoint::ChatCompletions
                | RequestObservationEndpoint::Responses => TunnelRequestObserverKind::Generation(
                    RequestObserver::from_required(endpoint, required, runtime_state),
                ),
                RequestObservationEndpoint::Embeddings => TunnelRequestObserverKind::Embeddings(
                    EmbeddingsRequestObserver::accepted(required, runtime_state),
                ),
            };
        Self { kind }
    }

    pub(crate) fn is_streaming(&self) -> bool {
        matches!(self.kind, TunnelRequestObserverKind::Generation(_))
    }

    pub(crate) fn generation_mut(&mut self) -> Option<&mut RequestObserver> {
        match &mut self.kind {
            TunnelRequestObserverKind::Generation(observer) => Some(observer),
            TunnelRequestObserverKind::Embeddings(_) => None,
        }
    }

    pub(crate) fn observe_request_body(&mut self, body_bytes: &[u8]) {
        if let TunnelRequestObserverKind::Embeddings(observer) = &mut self.kind {
            observer.update_embedding_items(embedding_items_from_request_body(body_bytes));
        }
    }

    pub(crate) fn on_upstream_response_headers(
        &mut self,
        response_headers: &HeaderMap,
        status: u16,
    ) {
        match &mut self.kind {
            TunnelRequestObserverKind::Generation(observer) => {
                observer.on_upstream_response_headers(response_headers, status);
            }
            TunnelRequestObserverKind::Embeddings(observer) => {
                observer.on_upstream_response_headers(status);
            }
        }
    }

    pub(crate) fn finish(&mut self) {
        match &mut self.kind {
            TunnelRequestObserverKind::Generation(observer) => observer.finish(),
            TunnelRequestObserverKind::Embeddings(observer) => observer.finish(),
        }
    }

    pub(crate) fn fail(&mut self) {
        match &mut self.kind {
            TunnelRequestObserverKind::Generation(observer) => observer.fail(),
            TunnelRequestObserverKind::Embeddings(observer) => observer.fail(),
        }
    }

    fn is_terminal(&self) -> bool {
        match &self.kind {
            TunnelRequestObserverKind::Generation(observer) => observer.is_terminal(),
            TunnelRequestObserverKind::Embeddings(observer) => observer.is_terminal(),
        }
    }
}

impl Drop for TunnelRequestObserver {
    fn drop(&mut self) {
        if !self.is_terminal() {
            self.fail();
        }
    }
}
