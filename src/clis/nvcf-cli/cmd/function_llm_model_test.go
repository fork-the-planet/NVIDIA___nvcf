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

package cmd

import (
	"encoding/json"
	"testing"

	"github.com/spf13/cobra"
)

func TestCreateConfigParsesLLMConfigFromJSON(t *testing.T) {
	t.Parallel()

	var config CreateConfig
	err := json.Unmarshal([]byte(`{
		"models": [{
			"name": "dummy-model",
			"llmConfig": {
				"uris": ["/v1/chat/completions", "/v1/responses", "/v1/embeddings"],
				"routingMethod": "round_robin",
				"tokenRateLimit": "1000-M"
			}
		}]
	}`), &config)
	if err != nil {
		t.Fatalf("unmarshal create config: %v", err)
	}

	if len(config.Models) != 1 {
		t.Fatalf("models length = %d, want 1", len(config.Models))
	}
	model := config.Models[0]
	if model.LLMConfig == nil {
		t.Fatal("llmConfig is nil")
	}
	assertStringSlice(t, model.LLMConfig.URIs, []string{"/v1/chat/completions", "/v1/responses", "/v1/embeddings"})
	if got := stringValue(model.LLMConfig.RoutingMethod); got != "round_robin" {
		t.Fatalf("routingMethod = %q, want round_robin", got)
	}
	if got := stringValue(model.LLMConfig.TokenRateLimit); got != "1000-M" {
		t.Fatalf("tokenRateLimit = %q, want 1000-M", got)
	}
}

func TestParseLLMModelString(t *testing.T) {
	t.Parallel()

	model, err := parseLLMModelString("name=dummy-model,uris=/v1/chat/completions|/v1/responses|/v1/embeddings,routingMethod=power_of_two,tokenRateLimit=1000-M")
	if err != nil {
		t.Fatalf("parse llm model: %v", err)
	}

	if model.Name != "dummy-model" {
		t.Fatalf("name = %q, want dummy-model", model.Name)
	}
	if model.LLMConfig == nil {
		t.Fatal("llmConfig is nil")
	}
	assertStringSlice(t, model.LLMConfig.URIs, []string{"/v1/chat/completions", "/v1/responses", "/v1/embeddings"})
	if got := stringValue(model.LLMConfig.RoutingMethod); got != "power_of_two" {
		t.Fatalf("routingMethod = %q, want power_of_two", got)
	}
}

func TestParseLLMModelStringRejectsInvalidRoutingMethod(t *testing.T) {
	t.Parallel()

	_, err := parseLLMModelString("name=dummy-model,uris=/v1/chat/completions,routingMethod=sticky")
	if err == nil {
		t.Fatal("expected invalid routing method error")
	}
}

func TestLoadCreateConfigAppendsLLMModelFlag(t *testing.T) {
	originalFlags := createFlags
	t.Cleanup(func() {
		createFlags = originalFlags
	})

	cmd := &cobra.Command{}
	cmd.Flags().StringArray("llm-model", nil, "")
	if err := cmd.Flags().Set("llm-model", "name=dummy-model,uris=/v1/chat/completions|/v1/embeddings,routingMethod=round_robin"); err != nil {
		t.Fatalf("set llm-model flag: %v", err)
	}
	createFlags = struct {
		inputFile string

		name           string
		containerImage string
		inferenceURL   string
		inferencePort  int

		description string
		tags        []string

		healthURI            string
		healthProtocol       string
		healthPort           int
		healthTimeout        string
		healthExpectedStatus int

		functionType         string
		apiBodyFormat        string
		containerArgs        string
		containerEnvironment []string

		helmChart            string
		helmChartServiceName string

		secrets []string

		models    []string
		llmModels []string
		resources []string

		rateLimit         string
		rateLimitExempted []string
		rateLimitSync     bool

		logsTelemetryId    string
		metricsTelemetryId string
		tracesTelemetryId  string
	}{
		llmModels: []string{"name=dummy-model,uris=/v1/chat/completions|/v1/embeddings,routingMethod=round_robin"},
	}

	config, err := loadCreateConfig(cmd)
	if err != nil {
		t.Fatalf("load create config: %v", err)
	}

	if len(config.Models) != 1 {
		t.Fatalf("models length = %d, want 1", len(config.Models))
	}
	model := config.Models[0]
	if model.LLMConfig == nil {
		t.Fatal("llmConfig is nil")
	}
	if got := stringValue(model.LLMConfig.RoutingMethod); got != "round_robin" {
		t.Fatalf("routingMethod = %q, want round_robin", got)
	}
	assertStringSlice(t, model.LLMConfig.URIs, []string{"/v1/chat/completions", "/v1/embeddings"})
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func assertStringSlice(t *testing.T, got []string, want []string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("uris = %#v, want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("uris = %#v, want %#v", got, want)
		}
	}
}
