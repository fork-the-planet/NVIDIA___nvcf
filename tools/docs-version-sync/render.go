// SPDX-FileCopyrightText: Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var ErrCheckFailed = errors.New("docs version catalog is out of sync")

func Render(renderer string, catalog *Catalog) (string, error) {
	if err := ValidateCatalog(catalog); err != nil {
		return "", err
	}
	switch renderer {
	case "manifest-artifact-registry-paths":
		return renderManifestArtifactRegistryPaths(catalog)
	case "manifest-deployment-resources":
		return renderManifestDeploymentResources(catalog)
	case "image-mirroring-resource-examples":
		return renderImageMirroringResourceExamples(catalog)
	case "image-mirroring-stack-snippet":
		return renderImageMirroringStackSnippet(catalog)
	case "image-mirroring-compute-stack-snippet":
		return renderImageMirroringComputeStackSnippet(catalog)
	case "image-mirroring-cli-snippet":
		return renderImageMirroringCLISnippet(catalog)
	default:
		return "", fmt.Errorf("unknown renderer %q", renderer)
	}
}

type manifestCategory struct {
	Heading     string
	Description string
	Names       []string
	StaticRows  []manifestStaticRow
}

type manifestStaticRow struct {
	Type string
	Name string
	Path string
}

func renderManifestArtifactRegistryPaths(catalog *Catalog) (string, error) {
	used := map[string]struct{}{}
	var b strings.Builder

	categories := []manifestCategory{
		{
			Heading:     "Infrastructure Components",
			Description: "Core infrastructure services including NATS for messaging, NATS auth callout support, Cassandra for data storage, and OpenBao for secret management.",
			Names: []string{
				"nats-box",
				"nats-server",
				"nats-server-config-reloader",
				"helm-nvcf-nats",
				"nvcf-nats-auth-callout-service",
				"helm-nvcf-nats-auth-callout-service",
				"bitnami-cassandra",
				"nvcf-cassandra-migrations",
				"helm-nvcf-cassandra",
				"nvcf-openbao",
				"nvcf-openbao-migrations",
				"helm-nvcf-openbao-server",
				"oss-vault-k8s",
			},
		},
		{
			Heading:     "Control Plane Components",
			Description: "Services that manage the NVCF platform including API gateway, deployment orchestration, invocation handling, LLM routing, and security services.",
			Names: []string{
				"spot",
				"strap",
				"helm-nvcf-api",
				"helm-nvcf-sis",
				"nvcf-grpc-proxy",
				"helm-nvcf-grpc-proxy",
				"nvcf-invocation-service",
				"helm-nvcf-invocation-service",
				"ess-api",
				"helm-nvcf-ess-api",
				"notary-service",
				"helm-nvcf-notary-service",
				"reval-server",
				"helm-reval",
				"nv-api-keys",
				"helm-nvcf-api-keys",
				"nvct-service-oss",
				"helm-nvct-api",
				"llm-api-gateway",
				"llm-request-router",
				"helm-nvcf-llm-api-gateway",
				"helm-nvcf-llm-request-router",
			},
		},
		{
			Heading:     "GPU Workload Components",
			Description: "Components that run on GPU nodes to manage function execution, including the NVCA operator and supporting containers.",
			Names: []string{
				"nvca",
				"nvca-operator",
				"helm-nvca-operator",
				"nvcf_worker_utils",
				"nvcf_worker_init",
				"nvcf_worker_niclls",
				"ess-agent",
				"nvcf-image-credential-helper",
			},
		},
		{
			Heading:     "Supporting Components",
			Description: "Additional utilities and helper services required for the platform, including the NVIDIA GPU Operator for GPU node management.",
			Names: []string{
				"alpine-k8s",
				"load_tester_supreme",
			},
			StaticRows: []manifestStaticRow{
				{Type: "Chart (HTTP)", Name: "gpu-operator", Path: "[Public NGC Helm repo](https://helm.ngc.nvidia.com/nvidia)"},
				{Type: "Image", Name: "gpu-operator-validator", Path: "[Public NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cloud-native/containers/gpu-operator-validator)"},
				{Type: "Image", Name: "k8s-device-plugin", Path: "[Public NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/k8s/containers/device-plugin)"},
				{Type: "Chart (HTTP)", Name: "ebs-csi-driver", Path: "https://kubernetes-sigs.github.io/aws-ebs-csi-driver"},
				{Type: "Chart (HTTP)", Name: "csi-driver-smb", Path: "https://raw.githubusercontent.com/kubernetes-csi/csi-driver-smb/master/charts"},
			},
		},
		{
			Heading:     "Reference Architecture Components",
			Description: "Optional components for the reference deployment architecture.",
			Names: []string{
				"nvcf-gateway-routes",
				"admin-token-issuer-proxy",
				"helm-admin-token-issuer-proxy",
			},
		},
		{
			Heading:     "Observability Components",
			Description: "Optional example components for monitoring and observability. These are provided as reference implementations only and are not intended for production use. See [self-hosted-example-dashboards](./example-dashboards.md) for deployment instructions.",
			Names: []string{
				"nvcf-observability-reference-stack",
				"nvcf-example-dashboards",
				"helm-nvcf-state-metrics",
			},
		},
		{
			Heading:     "Container Caching Components",
			Description: "Optional components for accelerating container image pulls across all workload types.",
			Names: []string{
				"nvcf-container-cache",
				"helm-nvcf-container-cache",
				"nvcf-proxy-tls-certs",
			},
		},
		{
			Heading:     "Simulation Caching Components",
			Description: "Optional caching components for Low Latency Streaming (LLS) and simulation workloads, including shader caching, derived data caching, and USD content caching.",
			Names: []string{
				"gxcache-webhook",
				"gxcache-init",
				"gxcache-service",
				"helm-gxcache",
				"ddcs-dist-kv",
				"usd-content-cache",
			},
			StaticRows: []manifestStaticRow{
				{Type: "Chart (HTTP)", Name: "ddcs", Path: "https://helm.ngc.nvidia.com/nvidia/omniverse/ddcs:5.0.0"},
				{Type: "Chart (HTTP)", Name: "usd-content-cache", Path: "https://helm.ngc.nvidia.com/nvidia/omniverse/usd-content-cache:3.0.3"},
			},
		},
		{
			Heading:     "Storage API Components",
			Description: "Optional components for USD Storage API functionality used in simulation workloads.",
			Names: []string{
				"storage-service",
				"simple-nginx",
			},
			StaticRows: []manifestStaticRow{
				{Type: "Chart (HTTP)", Name: "storage-service", Path: "https://helm.ngc.nvidia.com/nvidia/omniverse/storage-service:1.0.2"},
				{Type: "Chart (HTTP)", Name: "discovery-service", Path: "https://helm.ngc.nvidia.com/nvidia/omniverse/discovery-service:2.3.8"},
			},
		},
		{
			Heading:     "Low Latency Streaming (LLS) Components",
			Description: "Components for Low Latency Streaming functionality.",
			Names: []string{
				"streaming-proxy",
				"gdn-streaming",
			},
		},
	}

	for _, category := range categories {
		b.WriteString(fmt.Sprintf("#### %s\n\n", category.Heading))
		b.WriteString(category.Description)
		b.WriteString("\n\n")
		if err := renderManifestCategoryTable(&b, catalog, category, used); err != nil {
			return "", err
		}
		b.WriteString("\n")
	}

	otherArtifacts := catalog.uncategorizedArtifacts(used)
	if len(otherArtifacts) > 0 {
		b.WriteString("#### Other Published Components\n\n")
		b.WriteString("Additional components present in the current stack artifact manifest.\n\n")
		if err := renderArtifactTable(&b, catalog, otherArtifacts); err != nil {
			return "", err
		}
		b.WriteString("\n")
	}

	b.WriteString("#### Deployment Resources\n\n")
	b.WriteString("Helmfile and CLI resources for deployment.\n\n")
	deployment := manifestCategory{
		Names: []string{defaultStackResourceName, "nvcf-cli"},
	}
	for _, artifact := range catalog.resourceArtifacts() {
		deployment.Names = appendIfMissing(deployment.Names, artifact.Name)
	}
	if err := renderManifestCategoryTable(&b, catalog, deployment, used); err != nil {
		return "", err
	}
	return strings.TrimRight(b.String(), "\n") + "\n", nil
}

func renderManifestCategoryTable(b *strings.Builder, catalog *Catalog, category manifestCategory, used map[string]struct{}) error {
	var artifacts []Artifact
	for _, name := range category.Names {
		for _, artifact := range catalog.findArtifacts(name) {
			if catalog.isDenied(artifact.Name) {
				continue
			}
			artifacts = append(artifacts, artifact)
			used[artifact.catalogKey()] = struct{}{}
		}
	}
	if len(artifacts) == 0 && len(category.StaticRows) == 0 {
		b.WriteString("No artifacts are listed for this category in the current catalog.\n")
		return nil
	}
	if err := renderArtifactTable(b, catalog, artifacts); err != nil {
		return err
	}
	for _, row := range category.StaticRows {
		b.WriteString(fmt.Sprintf("| %s | %s | %s |\n", row.Type, row.Name, formatStaticPath(row.Path)))
	}
	return nil
}

func formatStaticPath(path string) string {
	if strings.HasPrefix(path, "[") {
		return path
	}
	return fmt.Sprintf("`%s`", path)
}

func renderArtifactTable(b *strings.Builder, catalog *Catalog, artifacts []Artifact) error {
	b.WriteString("| Type | Component Name | Full Path |\n")
	b.WriteString("| --- | --- | --- |\n")
	for _, artifact := range artifacts {
		path, err := catalog.artifactPath(artifact)
		if err != nil {
			return err
		}
		b.WriteString(fmt.Sprintf("| %s | %s | `%s` |\n", artifactTypeLabel(artifact), artifact.Name, path))
	}
	return nil
}

func renderManifestDeploymentResources(catalog *Catalog) (string, error) {
	resources := catalog.resourceArtifacts()
	var b strings.Builder
	b.WriteString("| Type | Component Name | Full Path |\n")
	b.WriteString("| --- | --- | --- |\n")
	for _, artifact := range resources {
		path, err := catalog.artifactPath(artifact)
		if err != nil {
			return "", err
		}
		b.WriteString(fmt.Sprintf("| Resource | %s | `%s` |\n", artifact.Name, path))
	}
	return b.String(), nil
}

func renderImageMirroringResourceExamples(catalog *Catalog) (string, error) {
	stack := catalog.stackArtifact()
	ref, err := catalog.resourceRef(stack)
	if err != nil {
		return "", err
	}

	compute, hasComputeStack := catalog.findArtifact(computeStackResourceName)
	computeRef := ""
	if hasComputeStack {
		if compute.Type != ArtifactTypeResource {
			return "", fmt.Errorf("%s must be a resource artifact", computeStackResourceName)
		}
		computeRef, err = catalog.resourceRef(compute)
		if err != nil {
			return "", err
		}
	}

	refWithVersion := strings.Replace(ref, stack.Version, "${STACK_VERSION}", 1)
	refWithoutVersion := strings.TrimSuffix(ref, ":"+stack.Version)

	var b strings.Builder
	b.WriteString("```bash\n")
	b.WriteString("# Set stack versions\n")
	b.WriteString(fmt.Sprintf("export STACK_VERSION=%q\n", stack.Version))
	if hasComputeStack {
		b.WriteString(fmt.Sprintf("export COMPUTE_STACK_VERSION=%q\n", compute.Version))
	}
	b.WriteString("\n")
	b.WriteString("# Download a specific control-plane stack version\n")
	b.WriteString("ngc registry resource download-version \\\n")
	b.WriteString(fmt.Sprintf("  %q\n\n", refWithVersion))
	b.WriteString("# List all control-plane stack versions\n")
	b.WriteString("ngc registry resource list \\\n")
	b.WriteString(fmt.Sprintf("  %q\n\n", refWithoutVersion+":*"))
	b.WriteString("# Download latest control-plane stack version (omit version)\n")
	b.WriteString("ngc registry resource download-version \\\n")
	b.WriteString(fmt.Sprintf("  %q\n", refWithoutVersion))

	if hasComputeStack {
		computeRefWithVersion := strings.Replace(computeRef, compute.Version, "${COMPUTE_STACK_VERSION}", 1)
		computeRefWithoutVersion := strings.TrimSuffix(computeRef, ":"+compute.Version)
		b.WriteString("\n# Download a specific compute-plane stack version\n")
		b.WriteString("ngc registry resource download-version \\\n")
		b.WriteString(fmt.Sprintf("  %q\n\n", computeRefWithVersion))
		b.WriteString("# List all compute-plane stack versions\n")
		b.WriteString("ngc registry resource list \\\n")
		b.WriteString(fmt.Sprintf("  %q\n\n", computeRefWithoutVersion+":*"))
		b.WriteString("# Download latest compute-plane stack version (omit version)\n")
		b.WriteString("ngc registry resource download-version \\\n")
		b.WriteString(fmt.Sprintf("  %q\n", computeRefWithoutVersion))
	}

	b.WriteString("```\n")
	return b.String(), nil
}

func renderImageMirroringStackSnippet(catalog *Catalog) (string, error) {
	stack := catalog.stackArtifact()
	ref, err := catalog.resourceRef(stack)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("```bash\n# Set the version\nexport VERSION=%q\n\nngc registry resource download-version %q && \\\n   mkdir -p %s && \\\n   tar -xzf %s_v${VERSION}/%s-${VERSION}.tar.gz -C %s && \\\n   rm -rf %s_v${VERSION}\n```\n",
		stack.Version,
		strings.Replace(ref, stack.Version, "${VERSION}", 1),
		stack.Name,
		stack.Name,
		stack.Name,
		stack.Name,
		stack.Name,
	), nil
}

func renderImageMirroringComputeStackSnippet(catalog *Catalog) (string, error) {
	compute, ok := catalog.findArtifact(computeStackResourceName)
	if !ok {
		return "", fmt.Errorf("supplemental artifact %s is required", computeStackResourceName)
	}
	if compute.Type != ArtifactTypeResource {
		return "", fmt.Errorf("%s must be a resource artifact", computeStackResourceName)
	}
	ref, err := catalog.resourceRef(compute)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("```bash\n# Set the version\nexport COMPUTE_VERSION=%q\n\nngc registry resource download-version %q && \\\n   mkdir -p %s && \\\n   tar -xzf %s_v${COMPUTE_VERSION}/%s-${COMPUTE_VERSION}.tar.gz -C %s && \\\n   rm -rf %s_v${COMPUTE_VERSION}\n```\n",
		compute.Version,
		strings.Replace(ref, compute.Version, "${COMPUTE_VERSION}", 1),
		compute.Name,
		compute.Name,
		compute.Name,
		compute.Name,
		compute.Name,
	), nil
}

func renderImageMirroringCLISnippet(catalog *Catalog) (string, error) {
	cli, ok := catalog.findArtifact("nvcf-cli")
	if !ok {
		return "", fmt.Errorf("supplemental artifact nvcf-cli is required")
	}
	ref, err := catalog.resourceRef(cli)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("```bash\n# Set the version\nexport VERSION=%q\n\n# Set your platform (linux-amd64, linux-arm64, darwin-amd64, darwin-arm64, windows-amd64)\nexport PLATFORM=\"linux-amd64\"\n\nngc registry resource download-version %q\n\ntar -xzf nvcf-cli_v${VERSION}/${PLATFORM}/nvcf-cli-${PLATFORM}-${VERSION}.tar.gz\nmv nvcf-cli-${PLATFORM}-${VERSION} nvcf-cli\nchmod +x nvcf-cli/nvcf-cli\n```\n",
		cli.Version,
		strings.Replace(ref, cli.Version, "${VERSION}", 1),
	), nil
}

func (catalog *Catalog) stackArtifact() Artifact {
	return Artifact{
		Name:     catalog.Stack.Name,
		Type:     ArtifactTypeResource,
		Registry: catalog.Stack.Registry,
		Version:  catalog.Stack.Version,
	}
}

func artifactTypeLabel(artifact Artifact) string {
	switch artifact.Type {
	case ArtifactTypeChart:
		return "Chart (OCI)"
	case ArtifactTypeResource:
		return "Resource"
	default:
		return "Image"
	}
}

func (catalog *Catalog) resourceArtifacts() []Artifact {
	denylist := catalog.DenylistMap()
	var resources []Artifact
	if _, denied := denylist[catalog.Stack.Name]; !denied {
		resources = append(resources, catalog.stackArtifact())
	}
	for _, artifact := range catalog.Artifacts {
		if artifact.Type != ArtifactTypeResource {
			continue
		}
		if _, denied := denylist[artifact.Name]; denied {
			continue
		}
		if artifact.Name == catalog.Stack.Name {
			continue
		}
		resources = append(resources, artifact)
	}
	for _, artifact := range catalog.SupplementalArtifacts {
		if artifact.Type != ArtifactTypeResource {
			continue
		}
		if _, denied := denylist[artifact.Name]; denied {
			continue
		}
		resources = append(resources, artifact)
	}
	return resources
}

func (catalog *Catalog) findArtifact(name string) (Artifact, bool) {
	if catalog.Stack.Name == name {
		return catalog.stackArtifact(), true
	}
	for _, artifact := range catalog.Artifacts {
		if artifact.Name == name {
			return artifact, true
		}
	}
	for _, artifact := range catalog.SupplementalArtifacts {
		if artifact.Name == name {
			return artifact, true
		}
	}
	return Artifact{}, false
}

func (catalog *Catalog) findArtifacts(name string) []Artifact {
	var artifacts []Artifact
	if catalog.Stack.Name == name {
		artifacts = append(artifacts, catalog.stackArtifact())
	}
	for _, artifact := range catalog.Artifacts {
		if artifact.Name == name {
			artifacts = append(artifacts, artifact)
		}
	}
	for _, artifact := range catalog.SupplementalArtifacts {
		if artifact.Name == name {
			artifacts = append(artifacts, artifact)
		}
	}
	return artifacts
}

func (catalog *Catalog) uncategorizedArtifacts(used map[string]struct{}) []Artifact {
	denylist := catalog.DenylistMap()
	var artifacts []Artifact
	for _, artifact := range catalog.Artifacts {
		if artifact.Type == ArtifactTypeResource {
			continue
		}
		if _, denied := denylist[artifact.Name]; denied {
			continue
		}
		if _, ok := used[artifact.catalogKey()]; ok {
			continue
		}
		artifacts = append(artifacts, artifact)
	}
	for _, artifact := range catalog.SupplementalArtifacts {
		if artifact.Type == ArtifactTypeResource {
			continue
		}
		if _, denied := denylist[artifact.Name]; denied {
			continue
		}
		if _, ok := used[artifact.catalogKey()]; ok {
			continue
		}
		artifacts = append(artifacts, artifact)
	}
	return artifacts
}

func (catalog *Catalog) isDenied(name string) bool {
	_, denied := catalog.DenylistMap()[name]
	return denied
}

func appendIfMissing(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func SyncDocs(repoRoot string, catalog *Catalog, check bool) error {
	if err := ValidateCatalog(catalog); err != nil {
		return err
	}
	var drifted []string
	for _, output := range catalog.Outputs {
		fullPath := filepath.Join(repoRoot, filepath.FromSlash(output.Path))
		data, err := os.ReadFile(fullPath)
		if err != nil {
			return fmt.Errorf("read %s: %w", output.Path, err)
		}
		updated := string(data)
		updated, changed, err := SyncInlineVersions(output.Path, updated, catalog)
		if err != nil {
			return fmt.Errorf("%s: %w", output.Path, err)
		}
		for _, block := range output.Blocks {
			rendered, err := Render(block.Renderer, catalog)
			if err != nil {
				return err
			}
			next, blockChanged, err := ReplaceMarkedBlock(updated, block.Marker, rendered)
			if err != nil {
				return fmt.Errorf("%s: %w", output.Path, err)
			}
			updated = next
			changed = changed || blockChanged
		}
		if !changed {
			continue
		}
		if check {
			drifted = append(drifted, output.Path)
			continue
		}
		if err := os.WriteFile(fullPath, []byte(updated), 0o644); err != nil {
			return fmt.Errorf("write %s: %w", output.Path, err)
		}
	}
	if len(drifted) > 0 {
		return fmt.Errorf("%w: %s", ErrCheckFailed, strings.Join(drifted, ", "))
	}
	return nil
}

func ReplaceMarkedBlock(content, marker, rendered string) (string, bool, error) {
	syntax, beginIndex, searchStart, endIndex, err := findMarkedBlock(content, marker)
	if err != nil {
		return "", false, err
	}
	replacement := "\n\n" + strings.TrimRight(rendered, "\n") + "\n\n"
	if syntax.Legacy {
		current := markerSyntaxes(marker)[0]
		updated := content[:beginIndex] + current.Begin + replacement + current.End + content[endIndex+len(syntax.End):]
		return updated, updated != content, nil
	}
	updated := content[:searchStart] + replacement + content[endIndex:]
	return updated, updated != content, nil
}

type markerSyntax struct {
	Begin  string
	End    string
	Legacy bool
}

func findMarkedBlock(content, marker string) (markerSyntax, int, int, int, error) {
	var selected markerSyntax
	beginIndex := -1
	for _, syntax := range markerSyntaxes(marker) {
		index := strings.Index(content, syntax.Begin)
		if index < 0 {
			continue
		}
		if beginIndex >= 0 {
			return markerSyntax{}, 0, 0, 0, fmt.Errorf("duplicate marker %q", marker)
		}
		if nextBegin := strings.Index(content[index+len(syntax.Begin):], syntax.Begin); nextBegin >= 0 {
			return markerSyntax{}, 0, 0, 0, fmt.Errorf("duplicate marker %q", marker)
		}
		selected = syntax
		beginIndex = index
	}
	if beginIndex < 0 {
		return markerSyntax{}, 0, 0, 0, fmt.Errorf("missing begin marker for %q", marker)
	}
	searchStart := beginIndex + len(selected.Begin)
	relativeEnd := strings.Index(content[searchStart:], selected.End)
	if relativeEnd < 0 {
		return markerSyntax{}, 0, 0, 0, fmt.Errorf("missing end marker for %q", marker)
	}
	endIndex := searchStart + relativeEnd
	return selected, beginIndex, searchStart, endIndex, nil
}

func markerSyntaxes(marker string) []markerSyntax {
	return []markerSyntax{
		{
			Begin: fmt.Sprintf("{/* docs-version-sync:BEGIN %s */}", marker),
			End:   fmt.Sprintf("{/* docs-version-sync:END %s */}", marker),
		},
		{
			Begin:  fmt.Sprintf("<!-- docs-version-sync:BEGIN %s -->", marker),
			End:    fmt.Sprintf("<!-- docs-version-sync:END %s -->", marker),
			Legacy: true,
		},
	}
}
