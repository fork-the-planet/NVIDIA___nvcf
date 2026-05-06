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
	"fmt"
	"regexp"
	"strings"
)

func SyncInlineVersions(path, content string, catalog *Catalog) (string, bool, error) {
	switch path {
	case "docs/user/standalone-infrastructure.md":
		return syncStandaloneInfrastructure(content, catalog)
	case "docs/user/standalone-core-services.md":
		return syncStandaloneCoreServices(content, catalog)
	case "docs/user/standalone-gateway.md":
		return syncStandaloneGateway(content, catalog)
	case "docs/user/image-mirroring.md":
		return syncImageMirroring(content, catalog)
	case "docs/user/cluster-management/self-managed.md":
		return syncClusterManagementSelfManaged(content, catalog)
	case "docs/user/cluster-management/reference.md":
		return syncClusterManagementReference(content, catalog)
	default:
		return content, false, nil
	}
}

func syncStandaloneInfrastructure(content string, catalog *Catalog) (string, bool, error) {
	replacements := []string{
		"helm-nvcf-nats",
		"helm-nvcf-openbao-server",
		"helm-nvcf-cassandra",
	}
	updated, changed, err := replaceChartVersions(content, catalog, replacements)
	if err != nil {
		return "", false, err
	}
	openbao, ok := catalog.latestArtifact("nvcf-openbao")
	if !ok {
		return "", false, fmt.Errorf("artifact nvcf-openbao is required")
	}
	updated, imageChanged := replaceImageTag(updated, "nvcf-openbao", openbao.Version)
	return updated, changed || imageChanged, nil
}

func syncStandaloneCoreServices(content string, catalog *Catalog) (string, bool, error) {
	return replaceChartVersions(content, catalog, []string{
		"helm-nvcf-api-keys",
		"helm-nvcf-sis",
		"helm-nvcf-ess-api",
		"helm-nvcf-api",
		"helm-nvcf-invocation-service",
		"helm-nvcf-grpc-proxy",
		"helm-nvcf-notary-service",
		"helm-reval",
		"helm-admin-token-issuer-proxy",
	})
}

func syncStandaloneGateway(content string, catalog *Catalog) (string, bool, error) {
	updated, changed, err := replaceChartVersions(content, catalog, []string{"nvcf-gateway-routes"})
	if err != nil {
		return "", false, err
	}
	admin, ok := catalog.findArtifact("helm-admin-token-issuer-proxy")
	if !ok {
		return "", false, fmt.Errorf("artifact helm-admin-token-issuer-proxy is required")
	}
	var count int
	updated, count = replaceHelmVersionArgument(updated, admin.Name, admin.Version)
	if count == 0 {
		return "", false, fmt.Errorf("helm --version argument for %s not found", admin.Name)
	}
	return updated, changed || updated != content, nil
}

func syncImageMirroring(content string, catalog *Catalog) (string, bool, error) {
	chart, ok := catalog.findArtifact("helm-nvca-operator")
	if !ok {
		return "", false, fmt.Errorf("artifact helm-nvca-operator is required")
	}
	updated := content
	var total int
	var count int
	updated, count = replaceNVCAOperatorChartPull(updated, chart.Version)
	total += count
	updated, count = replaceNVCAOperatorChartArchiveComments(updated, chart.Version)
	total += count
	updated, count = replaceNVCAOperatorChartPush(updated, chart.Version)
	total += count
	if total == 0 {
		return "", false, fmt.Errorf("nvca operator chart mirroring examples not found")
	}
	return updated, updated != content, nil
}

func syncClusterManagementSelfManaged(content string, catalog *Catalog) (string, bool, error) {
	updated, changed, err := replaceChartVersions(content, catalog, []string{"helm-nvca-operator"})
	if err != nil {
		return "", false, err
	}
	nvca, ok := catalog.findArtifact("nvca")
	if !ok {
		return "", false, fmt.Errorf("artifact nvca is required")
	}
	var count int
	updated, count = replaceYAMLStringValue(updated, "nvcaVersion", nvca.Version)
	if count == 0 {
		return "", false, fmt.Errorf("selfManaged.nvcaVersion not found")
	}
	return updated, changed || updated != content, nil
}

func syncClusterManagementReference(content string, catalog *Catalog) (string, bool, error) {
	helper, ok := catalog.findArtifact("nvcf-image-credential-helper")
	if !ok {
		return "", false, fmt.Errorf("artifact nvcf-image-credential-helper is required")
	}
	re := regexp.MustCompile(`(?m)(imageCredHelper:\n\s+imageRepository: ""\n\s+imageTag: )[^\n]+`)
	matches := re.FindAllStringIndex(content, -1)
	if len(matches) == 0 {
		return "", false, fmt.Errorf("imageCredHelper imageTag block not found")
	}
	updated := re.ReplaceAllString(content, "${1}"+helper.Version)
	return updated, updated != content, nil
}

func replaceChartVersions(content string, catalog *Catalog, names []string) (string, bool, error) {
	updated := content
	for _, name := range names {
		artifact, ok := catalog.findArtifact(name)
		if !ok {
			return "", false, fmt.Errorf("artifact %s is required", name)
		}
		var count int
		updated, count = replaceVersionTable(updated, name, artifact.Version)
		if count == 0 {
			return "", false, fmt.Errorf("version table for %s not found", name)
		}
		updated, count = replaceHelmVersionArgument(updated, name, artifact.Version)
		if count == 0 {
			return "", false, fmt.Errorf("helm --version argument for %s not found", name)
		}
	}
	return updated, updated != content, nil
}

func replaceVersionTable(content, chartName, version string) (string, int) {
	pattern := fmt.Sprintf(`(?s)(\| \*\*Chart\*\* \| %s \|\n\| --- \| --- \|\n\| \*\*Version\*\* \| )`+"`[^`]+`"+`( \|)`, regexp.QuoteMeta("`"+chartName+"`"))
	re := regexp.MustCompile(pattern)
	count := len(re.FindAllStringIndex(content, -1))
	return re.ReplaceAllString(content, "${1}`"+version+"`${2}"), count
}

func replaceHelmVersionArgument(content, chartName, version string) (string, int) {
	pattern := fmt.Sprintf(`(?ms)(oci://[^\s]+/%s\s+\\\n(?:[^\n]*\\\n)*?\s+--version )[^\s\\]+`, regexp.QuoteMeta(chartName))
	re := regexp.MustCompile(pattern)
	count := len(re.FindAllStringIndex(content, -1))
	return re.ReplaceAllString(content, "${1}"+version), count
}

func replaceYAMLStringValue(content, key, value string) (string, int) {
	pattern := fmt.Sprintf(`(?m)^(\s+%s:\s*)"[^"]+"`, regexp.QuoteMeta(key))
	re := regexp.MustCompile(pattern)
	count := len(re.FindAllStringIndex(content, -1))
	return re.ReplaceAllString(content, "${1}\""+value+"\""), count
}

func replaceNVCAOperatorChartPull(content, version string) (string, int) {
	re := regexp.MustCompile(`oci://nvcr\.io/0833294136851237/nvcf-ncp-staging/(?:helm-nvca-operator|nvca-operator) --version [^\s]+`)
	count := len(re.FindAllStringIndex(content, -1))
	replacement := "oci://nvcr.io/0833294136851237/nvcf-ncp-staging/helm-nvca-operator --version " + version
	return re.ReplaceAllString(content, replacement), count
}

func replaceNVCAOperatorChartArchiveComments(content, version string) (string, int) {
	re := regexp.MustCompile(`# This creates: (?:helm-nvca-operator|nvca-operator)-[^\s]+\.tgz`)
	count := len(re.FindAllStringIndex(content, -1))
	return re.ReplaceAllString(content, "# This creates: helm-nvca-operator-"+version+".tgz"), count
}

func replaceNVCAOperatorChartPush(content, version string) (string, int) {
	re := regexp.MustCompile(`helm push (?:helm-nvca-operator|nvca-operator)-[^\s]+\.tgz`)
	count := len(re.FindAllStringIndex(content, -1))
	return re.ReplaceAllString(content, "helm push helm-nvca-operator-"+version+".tgz"), count
}

func replaceImageTag(content, imageName, version string) (string, bool) {
	re := regexp.MustCompile(regexp.QuoteMeta(imageName) + `:[^\s"']+`)
	updated := re.ReplaceAllStringFunc(content, func(match string) string {
		return imageName + ":" + strings.TrimRight(version, "\\")
	})
	return updated, updated != content
}

func (catalog *Catalog) latestArtifact(name string) (Artifact, bool) {
	// Duplicate artifact names keep catalog order; generated docs intentionally use
	// the last matching entry rather than comparing version strings.
	var found Artifact
	ok := false
	for _, artifact := range catalog.findArtifacts(name) {
		found = artifact
		ok = true
	}
	return found, ok
}
