/*
SPDX-FileCopyrightText: Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package clusteragent

import (
	"context"
	"time"
)

// AgentMaintainer performs maintenance mutations against a compute-plane
// cluster's NVCA. It is the write-side counterpart to AgentInspector: drain and
// undrain toggle CordonAndDrain maintenance on the NVCA agent-config ConfigMap
// (which the operator picks up on a rollout restart).
//
// The Kubernetes implementation (k8s_maintainer.go) mirrors the proven operator
// logic in nvca/pkg/operator/cleanup/cleanup.go. The interface is the seam where
// a future NVCA HTTP maintenance endpoint can be swapped in without touching the
// cobra handlers.
type AgentMaintainer interface {
	// ResolveCluster reads the NVCFBackend CR in backendNS and returns the
	// cluster identity and the system/requests namespaces, applying defaults for
	// any field the CR leaves empty. It is the common preamble for every
	// maintenance operation.
	ResolveCluster(ctx context.Context, backendNS string) (*ClusterTarget, error)
	// Drain puts the cluster into CordonAndDrain maintenance.
	Drain(ctx context.Context, opts DrainOptions) (*DrainResult, error)
	// Undrain reverses Drain, returning the cluster to normal operation.
	Undrain(ctx context.Context, opts DrainOptions) (*DrainResult, error)
}

// ClusterTarget is the NVCFBackend-derived identity and namespace layout of a
// compute-plane cluster.
type ClusterTarget struct {
	ClusterID         string `json:"clusterId,omitempty"`
	ClusterName       string `json:"clusterName,omitempty"`
	SystemNamespace   string `json:"systemNamespace"`
	RequestsNamespace string `json:"requestsNamespace"`
}

// DrainOptions controls Drain and Undrain.
type DrainOptions struct {
	// BackendNS is the namespace holding the NVCFBackend CR.
	BackendNS string
	// ExpectClusterID, when non-empty, must match the live cluster's ID or name
	// or the operation is refused. Empty means trust the selected context.
	ExpectClusterID string
	// DryRun reports the intended change without mutating the cluster.
	DryRun bool
	// Force skips waiting for the NVCA rollout to complete.
	Force bool
	// Timeout bounds the rollout wait. Zero (with Force false) skips the wait.
	Timeout time.Duration
}

// DrainResult is the outcome of a Drain or Undrain.
type DrainResult struct {
	ClusterID        string `json:"clusterId,omitempty"`
	ClusterName      string `json:"clusterName,omitempty"`
	SystemNamespace  string `json:"systemNamespace"`
	Mode             string `json:"mode"`
	ConfigChanged    bool   `json:"configChanged"`
	RolloutTriggered bool   `json:"rolloutTriggered"`
	RolloutComplete  bool   `json:"rolloutComplete"`
	DryRun           bool   `json:"dryRun"`
	Message          string `json:"message,omitempty"`
}
