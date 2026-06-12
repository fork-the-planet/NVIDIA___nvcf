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
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"nvcf-cli/internal/clusteragent"
	"nvcf-cli/internal/k8s"
	"nvcf-cli/internal/logging"

	"github.com/spf13/cobra"
)

const (
	flagBackendNamespace = "backend-namespace"
	flagDryRun           = "dry-run"
	flagYes              = "yes"
	flagForce            = "force"
	flagExpectClusterID  = "expect-cluster-id"
	flagTimeout          = "timeout"

	defaultRolloutTimeout = 5 * time.Minute
)

var clusterAgentCordonDrainCmd = &cobra.Command{
	Use:          "cordon-and-drain",
	Aliases:      []string{"drain"},
	Short:        "Drain a cluster for maintenance (cordon, then drain to zero)",
	SilenceUsage: true,
	Args:         cobra.NoArgs,
	Long: `Put the cluster into CordonAndDrain maintenance: stop accepting new
deployments, let in-flight requests complete, and scale all function instances
to zero.

This sets the CordonAndDrainMaintenance feature flag and maintenanceMode on the
NVCA agent-config ConfigMap and restarts the NVCA deployment. The command returns
once NVCA has been told to drain (and, by default, once the restart rolls out);
use "cluster agent list-functions --phase DRAINING" to watch instances wind down.

Select the cluster with --compute-plane-context, as with the inspection commands.`,
	RunE: runClusterAgentCordonDrain,
}

var clusterAgentUncordonCmd = &cobra.Command{
	Use:          "uncordon",
	Aliases:      []string{"undrain"},
	Short:        "Reverse a drain and re-enable the cluster",
	SilenceUsage: true,
	Args:         cobra.NoArgs,
	Long: `Reverse a cordon-and-drain: remove the CordonAndDrainMaintenance feature
flag and maintenanceMode from the NVCA agent-config ConfigMap and restart NVCA so
the cluster accepts new deployments again.`,
	RunE: runClusterAgentUncordon,
}

// newAgentMaintainer builds the AgentMaintainer for a command. It is a package
// var so tests can swap in a fake (mirroring newClusterDeleterForDown).
var newAgentMaintainer = loadAgentMaintainer

func loadAgentMaintainer(cmd *cobra.Command) (clusteragent.AgentMaintainer, error) {
	kubeconfig, _ := cmd.Flags().GetString(flagKubeconfig)
	ctxOverride, _ := cmd.Flags().GetString(flagComputePlaneContext)

	kc, err := k8s.NewClient(&k8s.ClientConfig{
		KubeconfigPath:  kubeconfig,
		ContextOverride: ctxOverride,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to compute-plane cluster: %w", err)
	}
	return clusteragent.NewK8sMaintainer(kc.Dynamic(), kc.Clientset()), nil
}

func initClusterAgentMaintenanceCmds() {
	maintenanceCmds := []*cobra.Command{
		clusterAgentCordonDrainCmd,
		clusterAgentUncordonCmd,
	}
	for _, c := range maintenanceCmds {
		clusterAgentCmd.AddCommand(c)
		c.Flags().String(flagComputePlaneContext, "", "Kube context for the target compute-plane cluster")
		c.Flags().String(flagKubeconfig, "", "Path to kubeconfig for the target cluster")
		c.Flags().String(flagBackendNamespace, defaultBackendNamespace, "Namespace of the NVCFBackend resource")
		c.Flags().Bool(flagDryRun, false, "Print intended actions without mutating the cluster")
		c.Flags().Bool(flagYes, false, "Skip the confirmation prompt")
		c.Flags().String(flagExpectClusterID, "", "Refuse to act unless the connected cluster's id or name matches this value")
	}

	for _, c := range []*cobra.Command{clusterAgentCordonDrainCmd, clusterAgentUncordonCmd} {
		c.Flags().Duration(flagTimeout, defaultRolloutTimeout, "How long to wait for the NVCA rollout to complete")
		c.Flags().Bool(flagForce, false, "Skip waiting for the NVCA rollout to complete")
	}
}

func runClusterAgentCordonDrain(cmd *cobra.Command, _ []string) error {
	return runDrainCommon(cmd, true)
}

func runClusterAgentUncordon(cmd *cobra.Command, _ []string) error {
	return runDrainCommon(cmd, false)
}

func runDrainCommon(cmd *cobra.Command, drain bool) error {
	m, err := newAgentMaintainer(cmd)
	if err != nil {
		return err
	}

	backendNS, _ := cmd.Flags().GetString(flagBackendNamespace)
	dryRun, _ := cmd.Flags().GetBool(flagDryRun)
	yes, _ := cmd.Flags().GetBool(flagYes)
	force, _ := cmd.Flags().GetBool(flagForce)
	timeout, _ := cmd.Flags().GetDuration(flagTimeout)
	expect, _ := cmd.Flags().GetString(flagExpectClusterID)

	ctx := context.Background()
	verb := "drain"
	if !drain {
		verb = "undrain"
	}

	if !dryRun {
		target, err := m.ResolveCluster(ctx, backendNS)
		if err != nil {
			return err
		}
		if err := checkExpectedCluster(target, expect); err != nil {
			return err
		}
		ok, err := confirmSimple(cmd, fmt.Sprintf("This will %s cluster %s.", verb, joinNameID(target.ClusterName, target.ClusterID)), yes)
		if err != nil {
			return err
		}
		if !ok {
			logging.Info("%s cancelled", strings.ToUpper(verb[:1])+verb[1:])
			return nil
		}
	}

	opts := clusteragent.DrainOptions{
		BackendNS:       backendNS,
		ExpectClusterID: expect,
		DryRun:          dryRun,
		Force:           force,
		Timeout:         timeout,
	}

	var res *clusteragent.DrainResult
	if drain {
		res, err = m.Drain(ctx, opts)
	} else {
		res, err = m.Undrain(ctx, opts)
	}
	return finishDrain(cmd, res, drain, err)
}

// finishDrain renders a drain/undrain result and returns the (possibly non-nil)
// error. A partial result (e.g. ConfigChanged but the rollout trigger failed) is
// still emitted so --json callers can see the partial state.
func finishDrain(cmd *cobra.Command, res *clusteragent.DrainResult, drain bool, err error) error {
	if res == nil {
		return err
	}
	if IsJSONOutput() {
		if jsonErr := OutputJSON(res); jsonErr != nil {
			return jsonErr
		}
		return err
	}
	printDrainResult(cmd, res, drain)
	return err
}

// checkExpectedCluster enforces the optional --expect-cluster-id guard in the
// cmd layer so a mismatch aborts before prompting. The maintainer verifies again
// for defense in depth.
func checkExpectedCluster(target *clusteragent.ClusterTarget, expect string) error {
	if expect == "" {
		return nil
	}
	if expect == target.ClusterID || expect == target.ClusterName {
		return nil
	}
	return fmt.Errorf("refusing to proceed: --expect-cluster-id %q does not match the connected cluster %s; check --compute-plane-context", expect, joinNameID(target.ClusterName, target.ClusterID))
}

// confirmSimple prompts for a y/N confirmation, returning true when the operator
// agrees or --yes was passed. It reads from cmd.InOrStdin so it is testable.
func confirmSimple(cmd *cobra.Command, message string, yes bool) (bool, error) {
	if yes {
		return true, nil
	}
	fmt.Fprintf(cmd.OutOrStdout(), "%s Proceed? [y/N]: ", message)
	resp, err := readLine(cmd)
	if err != nil {
		return false, err
	}
	resp = strings.ToLower(strings.TrimSpace(resp))
	return resp == "y" || resp == "yes", nil
}

func readLine(cmd *cobra.Command) (string, error) {
	reader := bufio.NewReader(cmd.InOrStdin())
	line, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		return "", fmt.Errorf("failed to read confirmation: %w", err)
	}
	return line, nil
}

func printDrainResult(cmd *cobra.Command, res *clusteragent.DrainResult, drain bool) {
	w := cmd.OutOrStdout()
	verb := "Drain"
	if !drain {
		verb = "Undrain"
	}
	prefix := ""
	if res.DryRun {
		prefix = "[dry-run] "
	}
	fmt.Fprintf(w, "%s%s cluster %s\n", prefix, verb, joinNameID(res.ClusterName, res.ClusterID))
	if !res.ConfigChanged {
		fmt.Fprintln(w, "  already in the requested state; no change")
		return
	}
	if res.DryRun {
		fmt.Fprintln(w, "  would update agent-config and restart NVCA")
		return
	}
	if drain {
		fmt.Fprintf(w, "  agent-config updated (maintenanceMode=%s); NVCA restart triggered\n", orDash(res.Mode))
	} else {
		fmt.Fprintln(w, "  agent-config updated (maintenance cleared); NVCA restart triggered")
	}
	switch {
	case res.RolloutComplete:
		fmt.Fprintln(w, "  rollout complete")
	case res.Message != "":
		fmt.Fprintf(w, "  %s\n", res.Message)
	}
}
