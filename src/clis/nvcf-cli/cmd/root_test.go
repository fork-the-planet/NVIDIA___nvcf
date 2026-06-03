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
	"os"
	"path/filepath"
	"testing"
	"time"

	"nvcf-cli/internal/client"
	"nvcf-cli/internal/state"

	"github.com/spf13/viper"
)

func TestStateForCurrentCommandUsesAutoDiscoveredConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("NVCF_TOKEN", "")
	t.Setenv("NVCF_API_KEY", "")

	previousCfgFile := cfgFile
	previousManager := configStateManager
	previousManagerKey := configStateManagerKey
	previousDefaultManager := state.DefaultStateManager
	t.Cleanup(func() {
		cfgFile = previousCfgFile
		configStateManager = previousManager
		configStateManagerKey = previousManagerKey
		state.DefaultStateManager = previousDefaultManager
		viper.Reset()
	})

	cfgFile = ""
	configStateManager = nil
	configStateManagerKey = ""
	state.DefaultStateManager = state.NewStateManager()
	viper.Reset()
	viper.SetEnvPrefix("NVCF")
	viper.AutomaticEnv()

	configPath := filepath.Join(home, ".nvcf-cli.yaml")
	configBody := []byte(`
base_http_url: "http://api.localhost:8080"
invoke_url: "http://invocation.localhost:8080"
client_id: "nvcf-default"
`)
	if err := os.WriteFile(configPath, configBody, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	viper.AddConfigPath(home)
	viper.SetConfigType("yaml")
	viper.SetConfigName(".nvcf-cli")
	if err := viper.ReadInConfig(); err != nil {
		t.Fatalf("read config: %v", err)
	}
	if used := viper.ConfigFileUsed(); used != configPath {
		t.Fatalf("ConfigFileUsed = %q, want %q", used, configPath)
	}

	if err := LoadStateForCurrentCommand(); err != nil {
		t.Fatalf("load state: %v", err)
	}
	SetCurrentTokens("admin-token", "", time.Now().Add(time.Hour), time.Time{})
	if err := SaveStateForCurrentCommand(); err != nil {
		t.Fatalf("save state: %v", err)
	}

	config, err := client.LoadConfig()
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if config.Token != "admin-token" {
		t.Fatalf("Token = %q, want admin-token", config.Token)
	}
}
