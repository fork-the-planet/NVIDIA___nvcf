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
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"nvcf-cli/internal/selfhosted"
)

func resetComputePlaneFlags(t *testing.T) {
	t.Helper()
	selfHostedEnv = "local"
	selfHostedStack = ""
	selfHostedNoApply = false
	selfHostedControlPlaneContext = ""
	selfHostedComputePlaneContext = ""
	prevRuntimeResolver := resolveSelfHostedHelmRuntimeMode
	resolveSelfHostedHelmRuntimeMode = func(context.Context) (selfhosted.HelmRuntimeMode, error) {
		return selfhosted.HelmRuntimeHelm3Legacy, nil
	}
	t.Cleanup(func() {
		selfHostedEnv = "local"
		selfHostedStack = ""
		selfHostedNoApply = false
		selfHostedControlPlaneContext = ""
		selfHostedComputePlaneContext = ""
		computePlaneInstallValues = ""
		computePlaneInstallKubeContext = ""
		computePlaneInstallClusterName = ""
		computePlaneInstallNCAID = "nvcf-default"
		resolveSelfHostedHelmRuntimeMode = prevRuntimeResolver
	})
}

func TestComputePlaneInstallRequiresValues(t *testing.T) {
	resetComputePlaneFlags(t)

	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetErr(&bytes.Buffer{})
	rootCmd.SetArgs([]string{
		"self-hosted", "compute-plane", "install",
		"--stack", t.TempDir(),
	})

	err := rootCmd.Execute()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "--values is required")
}

func TestComputePlaneInstallTemplatesUserValuesFile(t *testing.T) {
	resetComputePlaneFlags(t)

	stackDir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(stackDir, "helmfile.d"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(stackDir, "helmfile-nvca-operator.yaml.gotmpl"), []byte("releases: []\n"), 0o644))

	valuesFile := filepath.Join(t.TempDir(), "custom-nvca-values.yaml")
	require.NoError(t, os.WriteFile(valuesFile, []byte("clusterName: gpu-from-values\nncaId: nca-from-values\n"), 0o644))

	fakeBin := installFakeComputePlaneHelmfile(t)

	var stdout, stderr bytes.Buffer
	rootCmd.SetOut(&stdout)
	rootCmd.SetErr(&stderr)
	rootCmd.SetArgs([]string{
		"self-hosted", "compute-plane", "install",
		"--stack", stackDir,
		"--values", valuesFile,
		"--kube-context", "gpu-context",
		"--no-apply",
	})

	require.NoError(t, rootCmd.Execute())
	out := stdout.String()
	assert.Contains(t, out, "verb=template")
	assert.Contains(t, out, "arg=--kube-context=gpu-context")
	assert.Contains(t, out, "env:NVCF_NVCA_VALUES_FILE="+valuesFile)
	assert.Contains(t, out, "env:CLUSTER_NAME=gpu-from-values")
	assert.Contains(t, out, "env:NCA_ID=nca-from-values")
	assert.Contains(t, out, "arg="+filepath.Join(stackDir, "helmfile-nvca-operator.yaml.gotmpl"))
	assert.FileExists(t, fakeBin)
}

func TestComputePlaneInstallAppliesByDefault(t *testing.T) {
	resetComputePlaneFlags(t)

	stackDir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(stackDir, "helmfile.d"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(stackDir, "helmfile-nvca-operator.yaml.gotmpl"), []byte("releases: []\n"), 0o644))

	valuesFile := filepath.Join(t.TempDir(), "gpu-a-nvca-values.yaml")
	require.NoError(t, os.WriteFile(valuesFile, []byte("clusterID: id-a\nclusterGroupID: group-a\n"), 0o644))

	installFakeComputePlaneHelmfile(t)

	var stdout bytes.Buffer
	rootCmd.SetOut(&stdout)
	rootCmd.SetErr(&bytes.Buffer{})
	rootCmd.SetArgs([]string{
		"self-hosted", "compute-plane", "install",
		"--stack", stackDir,
		"--values", valuesFile,
	})

	require.NoError(t, rootCmd.Execute())
	out := stdout.String()
	assert.Contains(t, out, "verb=apply")
	assert.Contains(t, out, "env:NVCF_NVCA_VALUES_FILE="+valuesFile)
	assert.Contains(t, out, "env:CLUSTER_NAME=gpu-a")
}

func installFakeComputePlaneHelmfile(t *testing.T) string {
	t.Helper()
	fakeBin := filepath.Join(t.TempDir(), "helmfile")
	body := `#!/bin/sh
last=
for arg in "$@"; do
  printf 'arg=%s\n' "$arg"
  last="$arg"
done
printf 'verb=%s\n' "$last"
printf 'env:NVCF_NVCA_VALUES_FILE=%s\n' "$NVCF_NVCA_VALUES_FILE"
printf 'env:CLUSTER_NAME=%s\n' "$CLUSTER_NAME"
printf 'env:NCA_ID=%s\n' "$NCA_ID"
`
	require.NoError(t, os.WriteFile(fakeBin, []byte(body), 0o755))
	t.Setenv("PATH", filepath.Dir(fakeBin)+":"+os.Getenv("PATH"))
	return fakeBin
}
