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

package nvca

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func marshalSelfManaged(t *testing.T, sm SelfManagedValues) string {
	t.Helper()
	body, err := yaml.Marshal(Values{SelfManaged: sm})
	require.NoError(t, err)
	return string(body)
}

// TestTransportTLSOmittedWhenNil verifies that an absent transportTls renders
// no selfManaged.transportTls block, so the nvca-operator chart applies its
// default (trustMode: system).
func TestTransportTLSOmittedWhenNil(t *testing.T) {
	out := marshalSelfManaged(t, SelfManagedValues{IdentitySource: "psat", TransportTLS: nil})
	require.NotContains(t, out, "transportTls")
}

// TestTransportTLSSystemRendersModeOnly verifies system mode renders only the
// trustMode (no fingerprint/PEM).
func TestTransportTLSSystemRendersModeOnly(t *testing.T) {
	out := marshalSelfManaged(t, SelfManagedValues{
		IdentitySource: "psat",
		TransportTLS:   &TransportTLSValues{TrustMode: "system"},
	})
	require.Contains(t, out, "trustMode: system")
	require.NotContains(t, out, "trustBundleFingerprint")
	require.NotContains(t, out, "trustBundlePem")
}

// TestTransportTLSBundleRendersFingerprintAndPEM verifies bundle mode renders
// the fingerprint and PEM unchanged for the chart to carry to the worker pods.
func TestTransportTLSBundleRendersFingerprintAndPEM(t *testing.T) {
	pem := "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
	out := marshalSelfManaged(t, SelfManagedValues{
		IdentitySource: "psat",
		TransportTLS: &TransportTLSValues{
			TrustMode:              "bundle",
			TrustBundleFingerprint: "sha256:" + "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
			TrustBundlePem:         pem,
		},
	})
	require.Contains(t, out, "trustMode: bundle")
	require.Contains(t, out, "trustBundleFingerprint: sha256:0123456789abcdef")
	require.Contains(t, out, "-----BEGIN CERTIFICATE-----")
}

// TestAgentRequestRouterAddressRendered verifies agent.llm.requestRouterAddress
// is rendered when set and the agent section is omitted entirely when nil.
func TestAgentRequestRouterAddressRendered(t *testing.T) {
	// Nil agent section is omitted.
	require.NotContains(t, marshalSelfManaged(t, SelfManagedValues{IdentitySource: "psat"}), "agent:")

	body, err := yaml.Marshal(Values{
		Agent: &AgentValues{LLM: &AgentLLMValues{RequestRouterAddress: "llm-request-router.nvcf-cp.internal:50071"}},
	})
	require.NoError(t, err)
	out := string(body)
	require.Contains(t, out, "agent:")
	require.Contains(t, out, "llm:")
	require.Contains(t, out, "requestRouterAddress: llm-request-router.nvcf-cp.internal:50071")
}

// TestWriteFileRoundTrip verifies WriteFile emits parseable YAML carrying the
// transport trust block.
func TestWriteFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nvca-values.yaml")
	require.NoError(t, WriteFile(path, Values{
		ClusterID:      "c-1",
		ClusterGroupID: "g-1",
		SelfManaged: SelfManagedValues{
			IdentitySource: "psat",
			TransportTLS:   &TransportTLSValues{TrustMode: "system"},
		},
	}))
	body, err := os.ReadFile(path)
	require.NoError(t, err)
	var got Values
	require.NoError(t, yaml.Unmarshal(body, &got))
	require.NotNil(t, got.SelfManaged.TransportTLS)
	require.Equal(t, "system", got.SelfManaged.TransportTLS.TrustMode)
}
