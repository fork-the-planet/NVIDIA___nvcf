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
	"context"
	"fmt"
	"time"

	"nvcf-cli/internal/client"
	"nvcf-cli/internal/clusteragent"
	"nvcf-cli/internal/k8s"

	"github.com/spf13/cobra"
)

const (
	flagComputePlaneContext = "compute-plane-context"
	flagKubeconfig          = "kubeconfig"
	flagNamespace           = "namespace"

	defaultBackendNamespace = "nvca-operator"
)

var clusterAgentCmd = &cobra.Command{
	Use:          "agent",
	Short:        "Inspect the NVCA compute-plane agent",
	SilenceUsage: true,
	Long: `Inspect the NVCF cluster agent (NVCA) running on a compute-plane cluster.

These are read-only commands that read the NVCFBackend and ICMSRequest custom
resources from the target cluster. Select the cluster with --compute-plane-context
(a kube context); --kubeconfig and the standard KUBECONFIG resolution apply.`,
}

var clusterAgentStatusCmd = &cobra.Command{
	Use:          "status",
	Short:        "Show NVCA version, health, and GPU usage for a cluster",
	SilenceUsage: true,
	Long: `Show the NVCA agent version, health, and GPU usage for a compute-plane cluster.

Version and health come from the NVCFBackend CR. When --nca-id is supplied and
NVCF_TOKEN is set, the output is enriched with the control-plane (ICMS) view of
the cluster. ICMS enrichment is additive: it is skipped, with a note, when not
available, and never fails the command.`,
	RunE: runClusterAgentStatus,
}

var clusterAgentFlags struct {
	ncaID string
}

func initClusterAgentCmds() {
	clusterCmd.AddCommand(clusterAgentCmd)
	clusterAgentCmd.AddCommand(clusterAgentStatusCmd)

	clusterAgentStatusCmd.Flags().String(flagComputePlaneContext, "", "Kube context for the target compute-plane cluster")
	clusterAgentStatusCmd.Flags().String(flagKubeconfig, "", "Path to kubeconfig for the target cluster")
	clusterAgentStatusCmd.Flags().String(flagNamespace, defaultBackendNamespace, "Namespace of the NVCFBackend resource")
	clusterAgentStatusCmd.Flags().StringVar(&clusterAgentFlags.ncaID, clusterFlagNcaID, "", "NCA/tenant ID for control-plane (ICMS) enrichment")
	addClusterICMSURLFlags(clusterAgentStatusCmd)
}

// loadAgentInspector builds the AgentInspector from the command's
// --kubeconfig/--compute-plane-context flags. It is the common preamble for
// every cluster agent handler.
func loadAgentInspector(cmd *cobra.Command) (clusteragent.AgentInspector, error) {
	kubeconfig, _ := cmd.Flags().GetString(flagKubeconfig)
	ctxOverride, _ := cmd.Flags().GetString(flagComputePlaneContext)

	kc, err := k8s.NewClient(&k8s.ClientConfig{
		KubeconfigPath:  kubeconfig,
		ContextOverride: ctxOverride,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to compute-plane cluster: %w", err)
	}
	return clusteragent.NewK8sInspector(kc.Dynamic()), nil
}

func runClusterAgentStatus(cmd *cobra.Command, args []string) error {
	inspector, err := loadAgentInspector(cmd)
	if err != nil {
		return err
	}

	namespace, _ := cmd.Flags().GetString(flagNamespace)
	ctxOverride, _ := cmd.Flags().GetString(flagComputePlaneContext)

	cfg, _ := client.LoadConfig()
	timeout := 30 * time.Second
	if cfg != nil {
		timeout = cfg.DefaultTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	status, err := inspector.Status(ctx, namespace)
	if err != nil {
		return err
	}
	status.ComputePlaneContext = ctxOverride
	status.ControlPlane = enrichStatusFromICMS(cmd, ctx, clusterAgentFlags.ncaID, status)

	if IsJSONOutput() {
		return OutputJSON(status)
	}
	printAgentStatus(status)
	return nil
}

// enrichStatusFromICMS attempts to add the control-plane (ICMS) view to the
// CR-derived status. It always returns a non-nil ICMSInfo; on any miss it sets
// Available=false with a Note and the caller still prints CR data.
func enrichStatusFromICMS(cmd *cobra.Command, ctx context.Context, ncaID string, status *clusteragent.AgentStatus) *clusteragent.ICMSInfo {
	if ncaID == "" {
		return &clusteragent.ICMSInfo{Available: false, Note: "ICMS enrichment skipped: --nca-id not provided"}
	}

	cfg, err := client.LoadConfig()
	if err != nil {
		return &clusteragent.ICMSInfo{Available: false, Note: fmt.Sprintf("ICMS enrichment skipped: %v", err)}
	}
	if cfg.Token == "" {
		return &clusteragent.ICMSInfo{Available: false, Note: "ICMS enrichment skipped: requires NVCF_TOKEN with cluster-management scope"}
	}

	c, err := client.NewClient(cfg)
	if err != nil {
		return &clusteragent.ICMSInfo{Available: false, Note: fmt.Sprintf("ICMS enrichment skipped: %v", err)}
	}
	defer c.Close()

	icmsURL := getICMSURL(cmd, cfg)
	clusters, err := c.ListClusters(ctx, icmsURL, ncaID)
	if err != nil {
		return &clusteragent.ICMSInfo{Available: false, Note: fmt.Sprintf("ICMS enrichment failed: %v", err)}
	}

	match := matchICMSCluster(clusters, status.ClusterID, status.ClusterName)
	if match == nil {
		return &clusteragent.ICMSInfo{Available: false, Note: "cluster not found in ICMS listing for this account"}
	}

	info := &clusteragent.ICMSInfo{
		Available:         true,
		ClusterName:       match.ClusterName,
		NVCAVersion:       match.NVCAVersion,
		ClusterStatus:     match.ClusterStatus,
		NVCALastConnected: match.NVCALastConnected,
	}
	if info.NVCAVersion == "" && info.ClusterStatus == "" && info.NVCALastConnected == "" {
		info.Available = false
		info.Note = "ICMS returned no enrichment fields (the list endpoint may not serialize cluster detail yet)"
	}
	return info
}

// matchICMSCluster finds the cluster by ID when clusterID is non-empty, or by
// name when clusterID is empty. When an ID is provided but not found the
// function returns nil; it does not fall through to name matching, which would
// silently return a different cluster sharing the same name.
func matchICMSCluster(clusters []client.ICMSCluster, clusterID, clusterName string) *client.ICMSCluster {
	if clusterID != "" {
		for i := range clusters {
			if clusters[i].ClusterID == clusterID {
				return &clusters[i]
			}
		}
		return nil
	}
	if clusterName != "" {
		for i := range clusters {
			if clusters[i].ClusterName == clusterName {
				return &clusters[i]
			}
		}
	}
	return nil
}

func printAgentStatus(s *clusteragent.AgentStatus) {
	fmt.Println("NVCA Status")
	if s.ClusterName != "" || s.ClusterID != "" {
		fmt.Printf("  Cluster:             %s\n", joinNameID(s.ClusterName, s.ClusterID))
	}
	fmt.Printf("  Namespace:           %s\n", s.Namespace)
	fmt.Printf("  NVCA Version:        %s\n", orDash(s.NVCAVersion))
	fmt.Printf("  Agent Health:        %s\n", orDash(s.AgentHealth))
	fmt.Printf("  Kubernetes Version:  %s\n", orDash(s.KubernetesVersion))
	fmt.Printf("  Last Updated:        %s\n", orDash(s.LastUpdated))

	fmt.Println("\nGPU")
	if len(s.GPU) == 0 {
		fmt.Println("  (none reported)")
	} else {
		fmt.Printf("  %-24s %-10s %-10s %-10s\n", "NAME", "CAPACITY", "AVAILABLE", "ALLOCATED")
		for _, g := range s.GPU {
			fmt.Printf("  %-24s %-10d %-10d %-10d\n", g.Name, g.Capacity, g.Available, g.Allocated)
		}
	}

	fmt.Println("\nControl Plane (ICMS)")
	if s.ControlPlane == nil || !s.ControlPlane.Available {
		note := "unavailable"
		if s.ControlPlane != nil && s.ControlPlane.Note != "" {
			note = s.ControlPlane.Note
		}
		fmt.Printf("  %s\n", note)
		return
	}
	fmt.Printf("  Cluster Status:      %s\n", orDash(s.ControlPlane.ClusterStatus))
	fmt.Printf("  NVCA Version:        %s\n", orDash(s.ControlPlane.NVCAVersion))
	fmt.Printf("  NVCA Last Connected: %s\n", orDash(s.ControlPlane.NVCALastConnected))
}

func joinNameID(name, id string) string {
	switch {
	case name != "" && id != "":
		return fmt.Sprintf("%s (%s)", name, id)
	case name != "":
		return name
	default:
		return id
	}
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
