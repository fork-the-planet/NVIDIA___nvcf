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
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
)

func newFakeInspector(objs ...runtime.Object) AgentInspector {
	scheme := runtime.NewScheme()
	gvrToListKind := map[schema.GroupVersionResource]string{
		nvcfBackendGVR: "NVCFBackendList",
		icmsRequestGVR: "ICMSRequestList",
	}
	dc := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, gvrToListKind, objs...)
	return NewK8sInspector(dc)
}

func nvcfBackend(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "nvcf.nvidia.io/v1",
		"kind":       "NVCFBackend",
		"metadata":   map[string]interface{}{"namespace": namespace, "name": name},
		"spec": map[string]interface{}{
			"version": "2.30.4",
			"clusterConfig": map[string]interface{}{
				"clusterId":   "cluster-uuid-1",
				"clusterName": "edge-1",
			},
		},
		"status": map[string]interface{}{
			"agentStatus":       "Healthy",
			"kubernetesVersion": "v1.30.2",
			"lastUpdated":       "2026-05-30T10:00:00Z",
			"gpuUsage": map[string]interface{}{
				"A100": map[string]interface{}{
					"capacity":  int64(8),
					"available": int64(3),
					"allocated": int64(5),
				},
			},
		},
	}}
}

// icmsRequest builds a fake ICMSRequest object. It is also used by
// k8s_maintainer_test.go in this package.
func icmsRequest(namespace, name, functionID, versionID, action, status string, useLegacy bool) *unstructured.Unstructured {
	spec := map[string]interface{}{"action": action}
	if useLegacy {
		spec["functionId"] = functionID
		spec["functionVersionId"] = versionID
	} else {
		spec["functionDetails"] = map[string]interface{}{
			"functionId":        functionID,
			"functionVersionId": versionID,
		}
	}
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "nvca.nvcf.nvidia.io/v2beta1",
		"kind":       "ICMSRequest",
		"metadata":   map[string]interface{}{"namespace": namespace, "name": name},
		"spec":       spec,
		"status": map[string]interface{}{
			"requestStatus": status,
			"instances": map[string]interface{}{
				"inst-a": map[string]interface{}{
					"id":     "inst-a",
					"type":   "Pod",
					"status": "Running",
				},
			},
		},
	}}
}

func TestStatusExtractsBackendFields(t *testing.T) {
	insp := newFakeInspector(nvcfBackend("nvca-operator", "backend"))

	st, err := insp.Status(context.Background(), "nvca-operator")
	if err != nil {
		t.Fatalf("Status returned error: %v", err)
	}
	if st.NVCAVersion != "2.30.4" {
		t.Errorf("NVCAVersion = %q, want 2.30.4", st.NVCAVersion)
	}
	if st.AgentHealth != "Healthy" {
		t.Errorf("AgentHealth = %q, want Healthy", st.AgentHealth)
	}
	if st.ClusterID != "cluster-uuid-1" || st.ClusterName != "edge-1" {
		t.Errorf("cluster identity = %q/%q, want cluster-uuid-1/edge-1", st.ClusterID, st.ClusterName)
	}
	if len(st.GPU) != 1 || st.GPU[0].Name != "A100" || st.GPU[0].Capacity != 8 || st.GPU[0].Allocated != 5 {
		t.Errorf("GPU = %+v, want one A100 with capacity 8 allocated 5", st.GPU)
	}
}

func TestStatusErrorsWhenNoBackend(t *testing.T) {
	insp := newFakeInspector()
	if _, err := insp.Status(context.Background(), "nvca-operator"); err == nil {
		t.Fatal("expected error when no NVCFBackend exists")
	}
}
