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
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	apiextclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"nvcf-cli/internal/client"
	"nvcf-cli/internal/selfhosted"
	"nvcf-cli/internal/selfhosted/auth"
	"nvcf-cli/internal/selfhosted/installlock"
	"nvcf-cli/internal/selfhosted/kubectx"
	"nvcf-cli/internal/selfhosted/progress"
	"nvcf-cli/internal/state"
)

var (
	upClusterName string
	upNCAID       string
	upRegion      string
	upPlanOnly    bool
)

// Phase numbers per SRD/SDD §6.4. Phases 3, 5 and 8 are synthetic — they emit
// started+completed back-to-back because the underlying work is fused into
// helmfile apply (phase 4 / 7) or hasn't been split out yet.
const (
	phaseControlPlaneNamespacesEnv = "NVCF_CLI_CP_NAMESPACES"
	phaseComputePlaneNamespacesEnv = "NVCF_CLI_COMPUTE_NAMESPACES"

	upPhasePreflight         = string(selfhosted.PhasePreflight)
	upPhaseResolve           = string(selfhosted.PhaseResolve)
	upPhaseRender            = string(selfhosted.PhaseRender)
	upPhaseApplyCP           = string(selfhosted.PhaseApplyCP)
	upPhaseCheckCP           = string(selfhosted.PhaseCheckCP)
	upPhaseRegister          = string(selfhosted.PhaseRegister)
	upPhaseApplyComputePlane = string(selfhosted.PhaseApplyCompute)
	upPhaseFinalHealth       = string(selfhosted.PhaseFinalCheck)
)

// defaultControlPlaneNamespaces is the watcher's filter for phase 4 (apply-cp).
// Best-effort enumeration of the namespaces that helmfile creates; mismatches
// just degrade the sub-progress to "0/0" without breaking the install.
var defaultControlPlaneNamespaces = []string{
	"sis", "nvcf", "cassandra-system", "openbao-system",
	"api-keys", "ess", "nats-system", "nvcf-backend",
}

// defaultComputePlaneNamespaces is the watcher's filter for phase 7
// (apply-compute-plane). The compute plane lives in nvca-system + nvca-operator
// per the multi-cluster topology.
var defaultComputePlaneNamespaces = []string{
	"nvca-system", "nvca-operator",
}

var selfHostedUpCmd = &cobra.Command{
	Use:          "up",
	Short:        "One-shot install: pre-flight + control plane + register + compute plane",
	SilenceUsage: true,
	RunE:         runSelfHostedUp,
}

func init() {
	selfHostedCmd.AddCommand(selfHostedUpCmd)
	selfHostedUpCmd.Flags().StringVar(&upClusterName, "cluster-name", "", "Cluster name (required)")
	_ = selfHostedUpCmd.MarkFlagRequired("cluster-name")
	selfHostedUpCmd.Flags().StringVar(&upNCAID, "nca-id", "nvcf-default", "NCA ID (account) the cluster registers under")
	selfHostedUpCmd.Flags().StringVar(&upRegion, "region", "us-west-1", "Cluster region (ICMS requires non-empty)")
	selfHostedUpCmd.Flags().BoolVar(&upPlanOnly, "plan-only", false, "Dry-run: emit planned phase sequence + ETAs without changing cluster state")
}

func runSelfHostedUp(c *cobra.Command, _ []string) error {
	ctx, stop := signal.NotifyContext(c.Context(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	started := time.Now().UTC()

	// Construct EventSink + selected renderer from --json/--plain/--accessible
	// + ambient state (NO_COLOR / TERM / CI / TTY size).
	// Cluster/Target/Stack populate the bubbletea Model header for TTY modes;
	// they are ignored by the plain, JSONL, and accessible renderers.
	// selfHostedStack is the flag value (before resolution); resolved.Path is
	// not yet available here, so the flag string is the best we have.
	sink, kind, err := progress.SelectRenderer(c.ErrOrStderr(), progress.RenderOpts{
		JSON:                selfHostedJSON,
		Plain:               selfHostedPlain,
		Accessible:          selfHostedAccessible,
		Cluster:             upClusterName,
		Target:              upRegion,
		Stack:               selfHostedStack,
		ControlPlaneContext: selfHostedControlPlaneContext, // M+9.E: split-cluster header
		ComputePlaneContext: selfHostedComputePlaneContext, // M+9.E: split-cluster header
	})
	if err != nil {
		return err
	}
	defer func() { _ = sink.Close() }()

	helmfileStdout, helmfileStderr, restoreOutput := configureUpOutput(c, sink, kind)
	defer restoreOutput()
	startEventSink(sink)

	r := &selfHostedUpRun{
		c:              c,
		ctx:            ctx,
		sink:           sink,
		started:        started,
		helmfileStdout: helmfileStdout,
		helmfileStderr: helmfileStderr,
	}
	return r.run()
}

type selfHostedUpRun struct {
	c              *cobra.Command
	ctx            context.Context
	sink           progress.EventSink
	started        time.Time
	helmfileStdout io.Writer
	helmfileStderr io.Writer
}

type upClusterRegistration struct {
	ClusterID      string
	ClusterGroupID string
	NCAID          string
	Region         string
	IdentitySource string
}

func configureUpOutput(c *cobra.Command, sink progress.EventSink, kind progress.RendererKind) (io.Writer, io.Writer, func()) {
	helmfileStdout := c.OutOrStdout()
	helmfileStderr := c.ErrOrStderr()
	if !isUpTTYRenderer(kind) {
		return helmfileStdout, helmfileStderr, noopUpCleanup
	}

	prev := log.Writer()
	log.SetOutput(io.Discard)
	stdoutW := progress.NewLogLineWriter(sink, "stdout", "helmfile")
	stderrW := progress.NewLogLineWriter(sink, "stderr", "helmfile")
	restore := func() {
		_ = stdoutW.Close()
		_ = stderrW.Close()
		log.SetOutput(prev)
	}
	return stdoutW, stderrW, restore
}

func isUpTTYRenderer(kind progress.RendererKind) bool {
	return kind == progress.RendererTTYFull || kind == progress.RendererTTYCompact
}

func startEventSink(sink progress.EventSink) {
	if starter, ok := sink.(interface{ Start() }); ok {
		starter.Start()
	}
}

func noopUpCleanup() {
	// Nothing was acquired or wrapped, so there is nothing to release.
}

func (r *selfHostedUpRun) run() error {
	if err := requireExplicitICMSForSplitMode(); err != nil {
		return err
	}
	releaseLock, err := r.runPreflightAndLock()
	if err != nil {
		return err
	}
	defer releaseLock()

	resolved, err := r.resolveStack()
	if err != nil {
		return err
	}
	if upPlanOnly {
		return r.emitPlanOnly(resolved.Path)
	}

	r.emitSyntheticPhase(3, upPhaseRender, kubectxFor(3))
	if err := r.applyControlPlane(resolved.Path); err != nil {
		return err
	}
	if err := r.checkControlPlane(); err != nil {
		return err
	}
	registration, err := r.registerCluster(resolved.Path)
	if err != nil {
		return err
	}
	if err := r.applyComputePlane(resolved.Path, registration); err != nil {
		return err
	}
	return r.emitFinalHealth(registration)
}

func requireExplicitICMSForSplitMode() error {
	if kubectx.SelectMode(selfHostedControlPlaneContext, selfHostedComputePlaneContext) != kubectx.ModeSplit {
		return nil
	}
	if selfHostedICMSURL != "" {
		return nil
	}
	return &ExitCodeError{
		Code: 3,
		Msg:  "split-cluster mode requires explicit --icms-url=<https://sis.…>; the control-plane ICMS endpoint won't be reachable from the compute-plane context",
	}
}

func (r *selfHostedUpRun) runPreflightAndLock() (func(), error) {
	p1Start, err := r.runPreflight()
	if err != nil {
		return noopUpCleanup, err
	}
	return r.acquireInstallLock(p1Start)
}

func (r *selfHostedUpRun) runPreflight() (time.Time, error) {
	if r.ctx.Err() != nil {
		return time.Time{}, r.emitCancellation(1, upPhasePreflight)
	}
	p1Start := r.emitPhase(1, upPhasePreflight)
	results := runUpPreflight(r.ctx, selfhosted.PreflightConfig{Tools: selfhosted.DefaultTools()})
	selfhosted.RenderText(r.c.ErrOrStderr(), results)
	if anyFailed(results) {
		r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhasePreflight, Err: fmt.Errorf("pre-flight checks failed")}, p1Start)
		return p1Start, &ExitCodeError{Code: 2, Msg: "pre-flight checks failed"}
	}
	r.emitPhaseDone(1, upPhasePreflight, p1Start)
	if r.ctx.Err() != nil {
		return p1Start, r.emitCancellation(2, upPhaseResolve)
	}
	return p1Start, nil
}

func (r *selfHostedUpRun) acquireInstallLock(p1Start time.Time) (func(), error) {
	kubeForLock, _, kerr := buildKubeClientsForWatcher()
	if kerr != nil {
		r.emitInstallLockWarning("install lock unavailable (kubeconfig: " + kerr.Error() + ")")
		return noopUpCleanup, nil
	}

	lk := installlock.NewLock(kubeForLock, lockKey(upNCAID, upClusterName), installlock.Options{})
	if lerr := lk.Acquire(r.ctx); lerr != nil {
		return r.handleInstallLockError(lerr, p1Start)
	}
	return func() { _ = lk.Release(context.Background()) }, nil
}

func (r *selfHostedUpRun) handleInstallLockError(lerr error, p1Start time.Time) (func(), error) {
	if errors.Is(lerr, installlock.ErrAlreadyHeld) {
		r.emitFailure(selfhosted.Failure{
			Phase: selfhosted.PhasePreflight,
			Err:   fmt.Errorf("another nvcf-cli self-hosted up is in progress: %w", lerr),
		}, p1Start)
		return noopUpCleanup, &ExitCodeError{Code: 1, Msg: lerr.Error()}
	}
	r.emitInstallLockWarning("install lock unavailable (proceeding without): " + lerr.Error())
	return noopUpCleanup, nil
}

func (r *selfHostedUpRun) emitInstallLockWarning(detail string) {
	_ = r.sink.Emit(r.ctx, progress.LastProgress{
		Num:    1,
		Detail: detail,
		At:     time.Now().UTC(),
	})
}

func (r *selfHostedUpRun) resolveStack() (*selfhosted.ResolvedStack, error) {
	p2Start := r.emitPhase(2, upPhaseResolve)
	resolved, err := selfhosted.ResolveStack(r.ctx, selfhosted.StackOptions{
		Source:        selfHostedStack,
		BuiltInOCIRef: builtInStackOCI(),
	})
	if err != nil {
		r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhaseResolve, Err: err}, p2Start)
		return nil, err
	}
	r.emitPhaseDone(2, upPhaseResolve, p2Start)
	if r.ctx.Err() != nil {
		return nil, r.emitCancellation(3, upPhaseRender)
	}
	return resolved, nil
}

func (r *selfHostedUpRun) emitPlanOnly(stackPath string) error {
	phases := progress.AllPhaseETAs()
	var totalETA int
	for _, p := range phases {
		totalETA += p.ETASec
	}
	_ = r.sink.Emit(r.ctx, progress.Planned{
		Cluster:     upClusterName,
		Target:      upRegion,
		Stack:       stackPath,
		Phases:      phases,
		TotalETASec: totalETA,
	})
	_ = r.sink.Emit(r.ctx, progress.Final{
		PlanOnly: true,
		Success:  true,
		Duration: time.Since(r.started),
	})
	return nil
}

func (r *selfHostedUpRun) applyControlPlane(stackPath string) error {
	p4Start := r.emitPhaseCtx(4, upPhaseApplyCP, kubectxFor(4))
	cancelWatcher, watcherDone := startWatcher(r.ctx, r.sink, 4, upPhaseApplyCP, controlPlaneNamespaces())
	err := selfhosted.Render(selfhosted.RenderOptions{
		StackPath:   stackPath,
		Env:         selfHostedEnv,
		Apply:       true,
		KubeContext: kubectxFor(4), // M+9.E: control-plane context in split mode
		Stdout:      r.helmfileStdout,
		Stderr:      r.helmfileStderr,
		Ctx:         r.ctx,
	})
	stopWatcher(cancelWatcher, watcherDone)
	if err != nil {
		return r.handleHelmfilePhaseError(4, upPhaseApplyCP, selfhosted.PhaseApplyCP, "control-plane install", err, p4Start)
	}
	r.emitPhaseDoneCtx(4, upPhaseApplyCP, kubectxFor(4), p4Start)
	if r.ctx.Err() != nil {
		return r.emitCancellation(5, upPhaseCheckCP)
	}
	return nil
}

func (r *selfHostedUpRun) checkControlPlane() error {
	p5Start := r.emitPhaseCtx(5, upPhaseCheckCP, kubectxFor(5))
	if selfHostedToken == "" {
		if err := authGatePhase5(r.ctx, r.sink, p5Start); err != nil {
			r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhaseCheckCP, Err: err}, p5Start)
			return fmt.Errorf("auth-gate: %w", err)
		}
	}
	r.emitPhaseDoneCtx(5, upPhaseCheckCP, kubectxFor(5), p5Start)
	if r.ctx.Err() != nil {
		return r.emitCancellation(6, upPhaseRegister)
	}
	return nil
}

func (r *selfHostedUpRun) registerCluster(stackPath string) (upClusterRegistration, error) {
	p6Start := r.emitPhaseCtx(6, upPhaseRegister, kubectxFor(6))
	registration, err := r.createClusterRegistration(stackPath, p6Start)
	if err != nil {
		return upClusterRegistration{}, err
	}
	r.emitPhaseDoneCtx(6, upPhaseRegister, kubectxFor(6), p6Start)
	_ = r.sink.Emit(r.ctx, progress.LastProgress{
		Num:    6,
		Detail: "cluster registered: " + registration.ClusterID,
		At:     time.Now().UTC(),
	})
	if r.ctx.Err() != nil {
		return upClusterRegistration{}, r.emitCancellation(7, upPhaseApplyComputePlane)
	}
	return registration, nil
}

func (r *selfHostedUpRun) createClusterRegistration(stackPath string, p6Start time.Time) (upClusterRegistration, error) {
	icmsURL := resolveICMSURL(selfHostedICMSURL)
	cc, err := newClusterClientForSelfHosted(icmsURL)
	if err != nil {
		r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhaseRegister, Err: err}, p6Start)
		return upClusterRegistration{}, err
	}
	defer cc.Close()

	oidcIssuer, jwks, identitySource, err := fetchClusterIdentity(r.ctx, kubectxFor(6))
	if err != nil {
		wrapped := fmt.Errorf("fetching cluster JWKS: %w", err)
		r.emitFailure(selfhosted.Failure{
			Phase:            selfhosted.PhaseRegister,
			Err:              wrapped,
			KubernetesReason: "JWKSFetchFailed",
		}, p6Start)
		return upClusterRegistration{}, wrapped
	}
	registration := upClusterRegistration{
		NCAID:          getenvDefault("NCA_ID", upNCAID),
		Region:         getenvDefault("CLUSTER_REGION", upRegion),
		IdentitySource: defaultString(identitySource, "psat"),
	}
	resp, err := cc.RegisterCluster(r.ctx, selfhosted.RegisterRequest{
		ClusterName:    upClusterName,
		NCAID:          registration.NCAID,
		Region:         registration.Region,
		JWKS:           jwks,
		OIDCIssuer:     oidcIssuer,
		IdentitySource: registration.IdentitySource,
	})
	if err != nil {
		wrapped := fmt.Errorf("cluster register: %w", err)
		r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhaseRegister, Err: wrapped, HTTPStatus: httpStatusFromErr(err)}, p6Start)
		return upClusterRegistration{}, wrapped
	}
	registration.ClusterID = resp.ClusterID
	registration.ClusterGroupID = resp.ClusterGroupID
	return registration, r.writeRegistrationValues(stackPath, icmsURL, registration, p6Start)
}

func (r *selfHostedUpRun) writeRegistrationValues(stackPath, icmsURL string, registration upClusterRegistration, p6Start time.Time) error {
	endpoints := resolveRegisterEndpointValues(selfHostedEnv, selfHostedControlPlaneContext, selfHostedComputePlaneContext, icmsURL, selfHostedNATSURL)
	if err := writeRegisterValuesYAML(registerValuesWriteRequest{
		StackPath:      stackPath,
		ClusterName:    upClusterName,
		NCAID:          registration.NCAID,
		Region:         registration.Region,
		IdentitySource: registration.IdentitySource,
		ClusterID:      registration.ClusterID,
		ClusterGroupID: registration.ClusterGroupID,
		Endpoints:      endpoints,
	}); err != nil {
		wrapped := fmt.Errorf("writing register-values: %w", err)
		r.emitFailure(selfhosted.Failure{Phase: selfhosted.PhaseRegister, Err: wrapped}, p6Start)
		return wrapped
	}
	return nil
}

func (r *selfHostedUpRun) applyComputePlane(stackPath string, registration upClusterRegistration) error {
	p7Start := r.emitPhaseCtx(7, upPhaseApplyComputePlane, kubectxFor(7))
	cancelWatcher, watcherDone := startWatcher(r.ctx, r.sink, 7, upPhaseApplyComputePlane, computePlaneNamespaces())
	helmfileFile, selector := computePlaneTarget(stackPath)
	err := selfhosted.Render(selfhosted.RenderOptions{
		StackPath:    stackPath,
		HelmfileFile: helmfileFile,
		Env:          selfHostedEnv,
		Apply:        true,
		Selector:     selector,
		KubeContext:  kubectxFor(7), // M+9.E: compute-plane context in split mode
		Stdout:       r.helmfileStdout,
		Stderr:       r.helmfileStderr,
		Ctx:          r.ctx,
		ExtraEnv:     registration.computePlaneEnv(),
	})
	stopWatcher(cancelWatcher, watcherDone)
	if err != nil {
		return r.handleHelmfilePhaseError(7, upPhaseApplyComputePlane, selfhosted.PhaseApplyCompute, "compute-plane install", err, p7Start)
	}
	r.emitPhaseDoneCtx(7, upPhaseApplyComputePlane, kubectxFor(7), p7Start)
	if r.ctx.Err() != nil {
		return r.emitCancellation(8, upPhaseFinalHealth)
	}
	return nil
}

func (r *selfHostedUpRun) emitFinalHealth(registration upClusterRegistration) error {
	p8Start := r.emitPhaseCtx(8, upPhaseFinalHealth, kubectxFor(8))
	r.emitPhaseDoneCtx(8, upPhaseFinalHealth, kubectxFor(8), p8Start)
	_ = r.sink.Emit(r.ctx, progress.Final{
		Success:           true,
		ClusterID:         registration.ClusterID,
		ClusterGroupID:    registration.ClusterGroupID,
		NVCFBackendHealth: "healthy", // placeholder; real probe is M+8
		Duration:          time.Since(r.started),
	})
	return nil
}

func (r *selfHostedUpRun) handleHelmfilePhaseError(num int, name string, phase selfhosted.PhaseID, prefix string, err error, started time.Time) error {
	if errors.Is(err, context.Canceled) || r.ctx.Err() != nil {
		return r.emitCancellation(num, name)
	}
	r.emitFailure(selfhosted.Failure{
		Phase:      phase,
		Err:        err,
		Subprocess: "helmfile",
		ExitCode:   exitCodeFromErr(err),
	}, started)
	return fmt.Errorf("%s: %w", prefix, err)
}

func (r *selfHostedUpRun) emitCancellation(phaseNum int, phaseName string) error {
	_ = r.sink.Emit(context.Background(), progress.PhaseCancelled{
		Num:    phaseNum,
		Name:   phaseName,
		Reason: "sigint",
	})
	_ = r.sink.Emit(context.Background(), progress.Final{
		Cancelled: true,
		Duration:  time.Since(r.started),
	})
	return &ExitCodeError{Code: 130, Msg: "interrupted"}
}

func (r *selfHostedUpRun) emitPhase(num int, name string) time.Time {
	t := time.Now().UTC()
	_ = r.sink.Emit(r.ctx, progress.PhaseStarted{Num: num, Name: name, StartedAt: t})
	return t
}

func (r *selfHostedUpRun) emitPhaseDone(num int, name string, start time.Time) {
	_ = r.sink.Emit(r.ctx, progress.PhaseCompleted{
		Num:      num,
		Name:     name,
		Duration: time.Since(start),
	})
}

func (r *selfHostedUpRun) emitPhaseCtx(num int, name, kctx string) time.Time {
	t := time.Now().UTC()
	_ = r.sink.Emit(r.ctx, progress.PhaseStarted{Num: num, Name: name, StartedAt: t, Context: kctx})
	return t
}

func (r *selfHostedUpRun) emitPhaseDoneCtx(num int, name, kctx string, start time.Time) {
	_ = r.sink.Emit(r.ctx, progress.PhaseCompleted{
		Num:      num,
		Name:     name,
		Duration: time.Since(start),
		Context:  kctx,
	})
}

func (r *selfHostedUpRun) emitSyntheticPhase(num int, name, kctx string) {
	started := r.emitPhaseCtx(num, name, kctx)
	r.emitPhaseDoneCtx(num, name, kctx, started)
}

func (r *selfHostedUpRun) emitFailure(f selfhosted.Failure, started time.Time) {
	_ = emitStructuredFailure(r.ctx, r.sink, f, started)
	r.emitFinalFailure()
}

func (r *selfHostedUpRun) emitFinalFailure() {
	_ = r.sink.Emit(r.ctx, progress.Final{
		Success:  false,
		Duration: time.Since(r.started),
	})
}

func (r upClusterRegistration) computePlaneEnv() []string {
	return []string{
		"CLUSTER_NAME=" + upClusterName,
		"CLUSTER_ID=" + r.ClusterID,
		"CLUSTER_GROUP_ID=" + r.ClusterGroupID,
		"IDENTITY_SOURCE=" + r.IdentitySource,
		"NCA_ID=" + r.NCAID,
		"CLUSTER_REGION=" + r.Region,
	}
}

func getenvDefault(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func defaultString(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}

// emitStructuredFailure categorizes a Failure (via selfhosted.Categorize) and
// emits a fully-populated PhaseFailed event with duration, category, retry
// class, remediation strings, and Raw sub-fields all filled.
func emitStructuredFailure(
	ctx context.Context,
	sink progress.EventSink,
	f selfhosted.Failure,
	started time.Time,
) error {
	pf := selfhosted.Categorize(f)
	pf.Duration = time.Since(started)
	return sink.Emit(ctx, pf)
}

// exitCodeFromErr extracts a subprocess exit code from a wrapped *exec.ExitError
// if present. Returns 1 as a reasonable non-zero default when the error doesn't
// carry an explicit exit code.
func exitCodeFromErr(err error) int {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	if err != nil {
		return 1
	}
	return 0
}

// httpStatusFromErr performs a best-effort extraction of an HTTP status code
// embedded in the error message (e.g. "status 409: ..."). Returns 0 when no
// status is parseable — callers treat 0 as "not an HTTP error".
func httpStatusFromErr(err error) int {
	if err == nil {
		return 0
	}
	msg := err.Error()
	// Look for patterns like "status 409", "HTTP 503", "503 service unavailable",
	// or the client package's "unexpected status: 422" convention.
	for _, prefix := range []string{"status ", "HTTP ", "status: "} {
		lower := strings.ToLower(msg)
		idx := strings.Index(lower, strings.ToLower(prefix))
		if idx == -1 {
			continue
		}
		rest := msg[idx+len(prefix):]
		code := parseLeadingInt(rest)
		if code >= 100 && code <= 599 {
			return code
		}
	}
	return 0
}

// parseLeadingInt reads a non-negative integer from the start of s and returns
// it. Returns 0 on any parse failure.
func parseLeadingInt(s string) int {
	val := 0
	found := false
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
			val = val*10 + int(ch-'0')
			found = true
		} else {
			break
		}
	}
	if !found {
		return 0
	}
	return val
}

// startWatcher launches WatchResources in a goroutine. Returns a cancel func
// and a done channel; pair them with stopWatcher to join cleanly. Watcher
// errors (kubeconfig unloadable, malformed) are swallowed — the install
// proceeds without sub-progress, which is graceful degradation rather than
// a failure mode.
func startWatcher(
	parent context.Context,
	sink progress.EventSink,
	phaseNum int,
	phaseName string,
	namespaces []string,
) (context.CancelFunc, <-chan struct{}) {
	ctx, cancel := context.WithCancel(parent)
	done := make(chan struct{})
	go func() {
		defer close(done)
		kubeClient, apiextClient, err := buildKubeClientsForWatcher()
		if err != nil {
			// Kubeconfig unavailable; renderer just won't see PhaseProgress
			// events for this phase. The install still works.
			return
		}
		_ = progress.WatchResources(ctx, kubeClient, apiextClient, namespaces, phaseNum, phaseName, sink)
	}()
	return cancel, done
}

// stopWatcher cancels the watcher context and waits for the goroutine to drain.
// Idempotent in the cancel sense (CancelFunc is) and required even on the
// success path — the watcher otherwise outlives the function until parent ctx
// is cancelled, which can leak informers in tests.
func stopWatcher(cancel context.CancelFunc, done <-chan struct{}) {
	cancel()
	<-done
}

// buildKubeClientsForWatcher constructs a kubernetes.Interface and an
// apiextensionsclientset.Interface from the user's kubeconfig (KUBECONFIG env
// or ~/.kube/config). Used to power the sub-progress watcher during phases 4
// and 7. Errors are non-fatal at the call site; callers degrade gracefully
// and skip the watcher.
//
// Exposed as a package-level var so tests can swap in a fake client without
// requiring a kubeconfig in CI.
var buildKubeClientsForWatcher = func() (kubernetes.Interface, apiextclientset.Interface, error) {
	cfg, err := loadKubeConfigForWatcher()
	if err != nil {
		return nil, nil, err
	}
	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}
	apiextClient, err := apiextclientset.NewForConfig(cfg)
	if err != nil {
		return nil, nil, err
	}
	return kubeClient, apiextClient, nil
}

// loadKubeConfigForWatcher resolves the user's kubeconfig using the standard
// client-go loading rules (--kubeconfig flag absent here, so it falls back to
// KUBECONFIG env then ~/.kube/config).
func loadKubeConfigForWatcher() (*rest.Config, error) {
	rules := clientcmd.NewDefaultClientConfigLoadingRules()
	overrides := &clientcmd.ConfigOverrides{}
	return clientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, overrides).ClientConfig()
}

// controlPlaneNamespaces returns the namespace filter for the phase-4 watcher.
// NVCF_CLI_CP_NAMESPACES (comma-separated) overrides the default list — useful
// when a custom helmfile lays things out differently.
func controlPlaneNamespaces() []string {
	if v := os.Getenv(phaseControlPlaneNamespacesEnv); v != "" {
		return splitAndTrim(v)
	}
	return defaultControlPlaneNamespaces
}

// computePlaneNamespaces returns the namespace filter for the phase-7 watcher.
// NVCF_CLI_COMPUTE_NAMESPACES overrides the default for parity with the
// control-plane env-var.
func computePlaneNamespaces() []string {
	if v := os.Getenv(phaseComputePlaneNamespacesEnv); v != "" {
		return splitAndTrim(v)
	}
	return defaultComputePlaneNamespaces
}

func splitAndTrim(csv string) []string {
	out := make([]string, 0, 4)
	start := 0
	for i := 0; i <= len(csv); i++ {
		if i == len(csv) || csv[i] == ',' {
			s := csv[start:i]
			// trim whitespace
			for len(s) > 0 && (s[0] == ' ' || s[0] == '\t') {
				s = s[1:]
			}
			for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
				s = s[:len(s)-1]
			}
			if s != "" {
				out = append(out, s)
			}
			start = i + 1
		}
	}
	return out
}

// runUpPreflight is a package-level seam so unit tests can short-circuit the
// pre-flight binary probes (kubectl/helmfile/helm version) without staging a
// fully-functional fake of each tool. Production callers run the real check.
var runUpPreflight = func(ctx context.Context, cfg selfhosted.PreflightConfig) []selfhosted.CheckResult {
	return selfhosted.RunPreflight(ctx, cfg)
}

// runSelfHostedInit shells out to the existing nvcf-cli init command — the
// runtime equivalent of the user typing `nvcf-cli init` themselves. Exposed
// as a package-level var so tests can swap it out without spawning subprocesses.
var runSelfHostedInit = func(ctx context.Context) error {
	c := exec.CommandContext(ctx, os.Args[0], selfHostedInitArgs()...)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	return c.Run()
}

func selfHostedInitArgs() []string {
	if cfgFile == "" {
		return []string{"init"}
	}
	return []string{"--config", cfgFile, "init"}
}

// authProbe is a package-level seam so tests can override auth.Probe without
// requiring a real HTTP server. Production callers use auth.Probe directly.
var authProbe = auth.Probe

// authGatePhase5 implements the M+8.G decision tree:
//  1. Probe control-plane fingerprint from config.BaseHTTPURL.
//  2. Load cached SelfHostedAuth from state.
//  3. If cached.Valid(now, fp) && !--refresh-token → use cached token; skip init.
//  4. Else: run init (which writes Token+TokenExpiration to state).
//     After init returns, re-load state, capture Token+TokenExpiration,
//     and save SelfHostedAuth{Token, ExpiresAt, Fingerprint} → State.Save().
//
// Errors from Probe are non-fatal in "fall-through" mode: cache_corruption
// or network errors → fall through to re-mint via init. Only init errors
// are reported as phase_failed.
func authGatePhase5(ctx context.Context, sink progress.EventSink, p5Start time.Time) error {
	// Step 1: probe fingerprint (non-fatal: probe failures fall through to re-mint).
	// LoadConfigWithoutAuth is used so missing NVCF_API_KEY / NVCF_TOKEN don't
	// cause a hard failure here — this path runs before init has minted a token.
	cfg, err := client.LoadConfigWithoutAuth()
	if err != nil {
		return fmt.Errorf("load client config: %w", err)
	}
	var fp *auth.Fingerprint
	if cfg.BaseHTTPURL != "" {
		probedFP, probeErr := authProbe(ctx, cfg.BaseHTTPURL)
		if probeErr == nil {
			fp = probedFP
		} else {
			_ = sink.Emit(ctx, progress.LastProgress{
				Num:    5,
				Detail: fmt.Sprintf("fingerprint probe failed (%v); falling through to re-mint", probeErr),
				At:     time.Now().UTC(),
			})
		}
	}

	// Step 2: try cached auth (skipped if --refresh-token).
	if !selfHostedRefreshToken && fp != nil {
		sm := state.NewStateManager()
		if err := sm.Load(); err == nil {
			if cached := sm.GetState().SelfHostedAuth; cached != nil && cached.Token != "" {
				cachedFP := fingerprintFromRef(cached.Fingerprint)
				cache := &auth.Cache{
					Token:       cached.Token,
					ExpiresAt:   cached.ExpiresAt,
					Fingerprint: cachedFP,
				}
				if cache.Valid(time.Now().UTC(), fp) {
					_ = sink.Emit(ctx, progress.LastProgress{
						Num:    5,
						Detail: "using cached admin token (fingerprint matches)",
						At:     time.Now().UTC(),
					})
					selfHostedToken = cached.Token
					return nil
				}
			}
		}
	}

	// Step 3: re-mint via init.
	_ = sink.Emit(ctx, progress.LastProgress{
		Num:    5,
		Detail: "minting admin token via API Keys service",
		At:     time.Now().UTC(),
	})
	if err := runSelfHostedInit(ctx); err != nil {
		return fmt.Errorf("init: %w", err)
	}

	// Step 4: persist Token+TokenExpiration that init wrote, alongside fingerprint.
	if fp != nil {
		sm := state.NewStateManager()
		if err := sm.Load(); err != nil {
			// Init succeeded but we can't read state — log and continue.
			// The token is in the state file; we just couldn't add the fingerprint.
			_ = sink.Emit(ctx, progress.LastProgress{
				Num:    5,
				Detail: fmt.Sprintf("warn: load state after init failed (%v); fingerprint not persisted", err),
				At:     time.Now().UTC(),
			})
			return nil
		}
		s := sm.GetState()
		if s.Token != "" {
			s.SelfHostedAuth = &state.SelfHostedAuth{
				Token:       s.Token,
				ExpiresAt:   s.TokenExpiration,
				Fingerprint: fingerprintRefFromAuth(fp),
			}
			if err := sm.Save(); err != nil {
				_ = sink.Emit(ctx, progress.LastProgress{
					Num:    5,
					Detail: fmt.Sprintf("warn: save SelfHostedAuth failed (%v)", err),
					At:     time.Now().UTC(),
				})
			}
		}
	}
	return nil
}

// fingerprintFromRef converts the persisted state.FingerprintRef to the
// runtime auth.Fingerprint type.
func fingerprintFromRef(ref *state.FingerprintRef) *auth.Fingerprint {
	if ref == nil {
		return nil
	}
	return &auth.Fingerprint{
		IssuerURL:       ref.IssuerURL,
		JWKSKid:         ref.JWKSKid,
		APIKeysEndpoint: ref.APIKeysEndpoint,
	}
}

// fingerprintRefFromAuth converts the runtime auth.Fingerprint back to
// the persisted state.FingerprintRef.
func fingerprintRefFromAuth(fp *auth.Fingerprint) *state.FingerprintRef {
	if fp == nil {
		return nil
	}
	return &state.FingerprintRef{
		IssuerURL:       fp.IssuerURL,
		JWKSKid:         fp.JWKSKid,
		APIKeysEndpoint: fp.APIKeysEndpoint,
	}
}

// lockKey returns the unique conflict surface for the install lock: NCA ID +
// cluster name identify the specific cluster being installed, which is the
// minimal key that would cause two concurrent `up` invocations to race.
func lockKey(ncaID, clusterName string) string {
	return ncaID + "--" + clusterName
}

// kubectxFor returns the kubeconfig context that phase phaseNum's kubectl /
// helmfile invocations should target. Returns "" in single-cluster mode (both
// context flags empty) so callers pass "" to Render/fetchClusterIdentity and
// the existing behaviour is preserved.
//
// SRD/SDD §4.1 split-cluster phase routing:
//
//	1 preflight        — fan out, both contexts (handled separately; returns "")
//	2 resolve-stack    — no cluster contact (returns "")
//	3 render-cp        — control plane
//	4 apply-cp         — control plane
//	5 check-cp         — control plane (auth-gate)
//	6 register         — compute plane (reads compute-plane JWKS)
//	7 apply-compute    — compute plane
//	8 final-health     — control plane (NVCF API on control plane)
func kubectxFor(phaseNum int) string {
	if selfHostedControlPlaneContext == "" || selfHostedComputePlaneContext == "" {
		return ""
	}
	switch phaseNum {
	case 6, 7:
		return selfHostedComputePlaneContext
	case 3, 4, 5, 8:
		return selfHostedControlPlaneContext
	default:
		return ""
	}
}

// writeRegisterValuesYAML persists the helm values handoff file the worker-layer
// helmfile expects at <stack>/out/<cluster>-register-values.yaml. Schema mirrors
// `printRegistrationOutput` (cmd/cluster_registration.go, fixed in `ff9aaf7`):
// top-level `clusterID`/`clusterGroupID`/`ncaID` with the mixed-case "ID" suffix
// to match the nvca-operator chart's `## @param` annotations in
// `nvca-operator/values.yaml`. `selfManaged.identitySource` and endpoint
// values stay nested because they scope self-managed-only chart inputs alongside
// the chart's other `selfManaged:` keys (`nvcaVersion`, `region`, etc.).
//
// The pre-`ff9aaf7` shape (nested `selfManaged.clusterId`, lowercase-d) left
// the chart's `.Values.clusterID` empty → cluster-dto.yaml rendered with empty
// IDs → operator failed to fetch backend identity. (Iter #12 from dev-VM E2E.)
type registerValuesWriteRequest struct {
	StackPath      string
	ClusterName    string
	NCAID          string
	Region         string
	IdentitySource string
	ClusterID      string
	ClusterGroupID string
	Endpoints      registerEndpointValues
}

func writeRegisterValuesYAML(req registerValuesWriteRequest) error {
	outDir := filepath.Join(req.StackPath, "out")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", outDir, err)
	}
	vals := helmValues{
		ClusterID:      req.ClusterID,
		ClusterGroupID: req.ClusterGroupID,
		NcaID:          req.NCAID,
		Region:         req.Region,
		SelfManaged:    newSelfManagedValuesFromEndpoints(req.IdentitySource, req.Endpoints),
	}
	body, err := yaml.Marshal(vals)
	if err != nil {
		return fmt.Errorf("marshal register values: %w", err)
	}
	path := filepath.Join(outDir, req.ClusterName+"-register-values.yaml")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}
