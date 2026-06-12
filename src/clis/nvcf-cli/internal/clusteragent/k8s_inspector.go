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
	"fmt"
	"sort"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// GroupVersionResources for the CRDs we read. Defined upstream in
// nvca/pkg/apis. Resource is the lowercase plural from the CRD spec.names.
var (
	nvcfBackendGVR = schema.GroupVersionResource{
		Group: "nvcf.nvidia.io", Version: "v1", Resource: "nvcfbackends",
	}
	icmsRequestGVR = schema.GroupVersionResource{
		Group: "nvca.nvcf.nvidia.io", Version: "v2beta1", Resource: "icmsrequests",
	}
)

// k8sInspector reads NVCA state from a compute-plane cluster's Kubernetes API
// using the dynamic client. It is the only AgentInspector implementation today.
type k8sInspector struct {
	dc dynamic.Interface
}

// NewK8sInspector returns an AgentInspector backed by the Kubernetes dynamic
// client.
func NewK8sInspector(dc dynamic.Interface) AgentInspector {
	return &k8sInspector{dc: dc}
}

// Status reads the NVCFBackend CR from namespace and maps it to AgentStatus.
func (k *k8sInspector) Status(ctx context.Context, namespace string) (*AgentStatus, error) {
	list, err := k.dc.Resource(nvcfBackendGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, wrapCRDError(err, "NVCFBackend", namespace)
	}
	if len(list.Items) == 0 {
		return nil, fmt.Errorf("no NVCFBackend resource found in namespace %q; is this context pointed at an NVCF compute-plane cluster (try --namespace)?", namespace)
	}

	obj := list.Items[0].Object
	status := &AgentStatus{
		Namespace:         namespace,
		ClusterID:         firstNonEmpty(nestedString(obj, "spec", "clusterConfig", "clusterId"), nestedString(obj, "spec", "clusterConfig", "clusterID")),
		ClusterName:       nestedString(obj, "spec", "clusterConfig", "clusterName"),
		NVCAVersion:       nestedString(obj, "spec", "version"),
		AgentHealth:       firstNonEmpty(nestedString(obj, "status", "agentStatus"), "Unknown"),
		KubernetesVersion: nestedString(obj, "status", "kubernetesVersion"),
		LastUpdated:       nestedString(obj, "status", "lastUpdated"),
		GPU:               extractGPUUsage(obj),
	}
	return status, nil
}

func extractGPUUsage(obj map[string]interface{}) []GPUUsage {
	raw, found, err := unstructured.NestedMap(obj, "status", "gpuUsage")
	if !found || err != nil {
		return nil
	}
	gpus := make([]GPUUsage, 0, len(raw))
	for name, v := range raw {
		m, ok := v.(map[string]interface{})
		if !ok {
			continue
		}
		gpus = append(gpus, GPUUsage{
			Name:      name,
			Capacity:  nestedInt(m, "capacity"),
			Available: nestedInt(m, "available"),
			Allocated: nestedInt(m, "allocated"),
		})
	}
	sort.Slice(gpus, func(i, j int) bool { return gpus[i].Name < gpus[j].Name })
	return gpus
}

// wrapCRDError turns common dynamic-client failures into actionable messages.
func wrapCRDError(err error, kind, namespace string) error {
	switch {
	case apierrors.IsForbidden(err):
		return fmt.Errorf("not permitted to list %s resources in %s: %w", kind, namespaceLabel(namespace), err)
	case apierrors.IsNotFound(err):
		return fmt.Errorf("%s CRD is not installed on this cluster; is this context pointed at an NVCF compute-plane cluster? %w", kind, err)
	default:
		return fmt.Errorf("failed to list %s resources in %s: %w", kind, namespaceLabel(namespace), err)
	}
}

func namespaceLabel(namespace string) string {
	if namespace == "" {
		return "all namespaces"
	}
	return "namespace " + namespace
}

// --- unstructured field helpers (missing fields degrade to zero values) ---

func nestedString(obj map[string]interface{}, fields ...string) string {
	v, _, _ := unstructured.NestedString(obj, fields...)
	return v
}

func nestedInt(obj map[string]interface{}, fields ...string) int64 {
	v, found, err := unstructured.NestedInt64(obj, fields...)
	if found && err == nil {
		return v
	}
	// gpuUsage/instances counts may decode as float64 from JSON.
	if f, found, err := unstructured.NestedFloat64(obj, fields...); found && err == nil {
		return int64(f)
	}
	return 0
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
