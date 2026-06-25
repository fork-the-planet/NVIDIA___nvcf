// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::{Duration, Instant};

use anyhow::Context;

use crate::config::{BenchmarkConfig, DegradationActionConfig, DegradationActionKind};
use crate::driver::{DriveConfig, RequestResult, drive_manifest, load_manifest};
use crate::k8s::{
    BenchmarkK8sRun, apply as apply_k8s, collect_logs, delete as delete_k8s, delete_backend_pod,
    prepare_benchmark_k8s_run, scale_backend, stargate_metrics_endpoints, wait_ready,
};
use crate::manifest::{Manifest, ManifestRequest, write_manifest_json};
use crate::metadata::{
    BenchmarkTier, DriverMode, ReliabilityMode, collect_run_metadata, write_run_metadata,
};
use crate::report::{ReportContext, ReportEntry, write_benchmark_report_artifacts};
use crate::score::{
    RoutingTopology, RunSummary, queue_admission_summary_delta_from_prometheus,
    routing_selection_summary_delta_from_prometheus, summarize_with_topology, topology_for,
};

const COLLECTOR_SCRAPE_SETTLE_DELAY: Duration = Duration::from_millis(1_100);
const K8S_READINESS_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum K8sLogCollection {
    Collect,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum K8sResourceDisposition {
    Delete,
    KeepFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct K8sRunFinalization {
    logs: K8sLogCollection,
    resources: K8sResourceDisposition,
}

impl K8sRunFinalization {
    fn for_result(run_failed: bool, keep_resources_on_failure: bool) -> Self {
        let logs = if run_failed || keep_resources_on_failure {
            K8sLogCollection::Collect
        } else {
            K8sLogCollection::Skip
        };
        let resources = if run_failed && keep_resources_on_failure {
            K8sResourceDisposition::KeepFailed
        } else {
            K8sResourceDisposition::Delete
        };
        Self { logs, resources }
    }
}

#[derive(Debug, Clone)]
struct K8sRunArtifacts {
    collector_baseline_metrics: PathBuf,
    requests: PathBuf,
    stargate_metrics: PathBuf,
    collector_metrics: PathBuf,
    summary: PathBuf,
}

impl K8sRunArtifacts {
    fn for_run_dir(run_dir: &Path) -> Self {
        Self {
            collector_baseline_metrics: run_dir.join("collector-baseline-metrics.prom"),
            requests: run_dir.join("requests.jsonl"),
            stargate_metrics: run_dir.join("metrics.prom"),
            collector_metrics: run_dir.join("collector-metrics.prom"),
            summary: run_dir.join("summary.json"),
        }
    }
}

#[derive(Debug, Clone)]
struct K8sReplayPlan {
    manifest: Manifest,
    routing_probe_request: ManifestRequest,
}

impl K8sReplayPlan {
    fn load(manifest_path: &Path) -> anyhow::Result<Self> {
        let manifest = load_manifest(manifest_path)?;
        let routing_probe_request = manifest.requests.first().cloned().ok_or_else(|| {
            anyhow::anyhow!("benchmark manifest must contain at least one request")
        })?;
        Ok(Self {
            manifest,
            routing_probe_request,
        })
    }
}

struct K8sCollectorBaseline {
    metrics: String,
    request_totals: ScrapedRequestTotals,
}

#[derive(Debug, Clone)]
struct K8sReadinessGate {
    health_url: String,
    metrics_endpoints: Vec<String>,
    routing_url: String,
    model: String,
    expected_backend_count: usize,
    routing_probe_request: ManifestRequest,
    timeout: Duration,
}

impl K8sReadinessGate {
    fn for_run(
        run: &crate::k8s::BenchmarkK8sRun,
        backend_count: usize,
        replay_plan: &K8sReplayPlan,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            health_url: format!("{}/healthz", run.stargate_http_endpoint),
            metrics_endpoints: stargate_metrics_endpoints(run)?,
            routing_url: format!("{}/v1/chat/completions", run.stargate_http_endpoint),
            model: replay_plan.manifest.model.clone(),
            expected_backend_count: backend_count,
            routing_probe_request: replay_plan.routing_probe_request.clone(),
            timeout: K8S_READINESS_TIMEOUT,
        })
    }

    fn wait(&self, runtime: &tokio::runtime::Runtime) -> anyhow::Result<()> {
        runtime.block_on(self.wait_async())
    }

    async fn wait_async(&self) -> anyhow::Result<()> {
        wait_for_http_ok(&self.health_url, self.timeout).await?;
        wait_for_active_backend_counts(
            &self.metrics_endpoints,
            &self.model,
            self.routing_probe_request.routing_key.as_deref(),
            self.expected_backend_count,
            self.timeout,
        )
        .await?;
        wait_for_routing(
            &self.routing_url,
            &self.model,
            &self.routing_probe_request,
            self.timeout,
        )
        .await
    }
}

struct K8sSingleRun<'a> {
    benchmark_run: &'a BenchmarkK8sRun,
    concurrency_limit: usize,
    backend_count: usize,
    topology: &'a RoutingTopology,
    degradation_actions: &'a [DegradationActionConfig],
}

impl<'a> K8sSingleRun<'a> {
    fn new(
        run: &'a BenchmarkK8sRun,
        concurrency_limit: usize,
        backend_count: usize,
        topology: &'a RoutingTopology,
        degradation_actions: &'a [DegradationActionConfig],
    ) -> Self {
        Self {
            benchmark_run: run,
            concurrency_limit,
            backend_count,
            topology,
            degradation_actions,
        }
    }

    fn execute(&self) -> anyhow::Result<RunSummary> {
        wait_ready(self.benchmark_run, self.backend_count)?;
        let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
        let replay_plan = K8sReplayPlan::load(&self.benchmark_run.manifest_path)?;
        let artifacts = K8sRunArtifacts::for_run_dir(&self.benchmark_run.run_dir);
        wait_for_k8s_probe_readiness(
            &runtime,
            self.benchmark_run,
            self.backend_count,
            &replay_plan,
        )?;
        let baseline = capture_collector_baseline(&runtime, self.benchmark_run, &artifacts)?;
        let results = drive_k8s_replay(
            &runtime,
            self.benchmark_run,
            replay_plan.manifest,
            self.concurrency_limit,
            self.degradation_actions,
            &artifacts,
        )?;
        write_k8s_run_summary(
            &runtime,
            self.benchmark_run,
            self.topology,
            &results,
            baseline,
            &artifacts,
        )
    }
}

pub fn run_k8s_benchmark(
    config: BenchmarkConfig,
    manifest: Manifest,
    output_dir: PathBuf,
    keep_resources_on_failure: bool,
    reliability_mode: ReliabilityMode,
) -> anyhow::Result<()> {
    let metadata = collect_run_metadata(
        BenchmarkTier::LocalK8sSmoke,
        reliability_mode,
        DriverMode::ExternalNodePort,
    );
    K8sBenchmarkCommand::from_metadata(
        config,
        manifest,
        output_dir,
        keep_resources_on_failure,
        metadata,
    )
    .run_to_completion()
}

struct K8sBenchmarkCommand {
    config: BenchmarkConfig,
    manifest: Manifest,
    output_dir: PathBuf,
    keep_resources_on_failure: bool,
    metadata: crate::metadata::RunMetadata,
}

impl K8sBenchmarkCommand {
    fn from_metadata(
        config: BenchmarkConfig,
        manifest: Manifest,
        output_dir: PathBuf,
        keep_resources_on_failure: bool,
        metadata: crate::metadata::RunMetadata,
    ) -> Self {
        Self {
            config,
            manifest,
            output_dir,
            keep_resources_on_failure,
            metadata,
        }
    }

    fn run_to_completion(&self) -> anyhow::Result<()> {
        self.run_with_context_check(ensure_k8s_context)
    }

    fn run_with_context_check(
        &self,
        ensure_context: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.prepare_to_run(ensure_context)?;
        let report_entries = self.run_algorithms()?;
        self.write_report(&report_entries)?;
        println!("completed {} algorithm runs", report_entries.len());
        Ok(())
    }

    fn prepare_to_run(
        &self,
        ensure_context: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.create_output_dir()?;
        self.write_metadata()?;
        self.ensure_preflight()?;
        ensure_context()?;
        self.write_manifest()?;
        self.print_run_header();
        Ok(())
    }

    fn create_output_dir(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("failed to create output dir {}", self.output_dir.display()))
    }

    fn write_metadata(&self) -> anyhow::Result<()> {
        write_run_metadata(&self.metadata_path(), &self.metadata)
    }

    fn ensure_preflight(&self) -> anyhow::Result<()> {
        if self.metadata.preflight.should_fail {
            anyhow::bail!(
                "strict reliability preflight failed with {} failure(s); inspect {}",
                self.metadata.preflight.failure_count,
                self.metadata_path().display()
            );
        }
        Ok(())
    }

    fn write_manifest(&self) -> anyhow::Result<()> {
        write_manifest_json(&self.manifest_path(), &self.manifest)
    }

    fn print_run_header(&self) {
        println!(
            "running benchmark '{}' with {} request(s), {} backend(s), {} stargate(s)",
            self.config.name,
            self.config.request_count,
            self.config.backends.count,
            self.config.stargates.count
        );
        println!("output directory: {}", self.output_dir.display());
        println!(
            "algorithms: {}",
            self.config
                .algorithms
                .iter()
                .map(|algorithm| algorithm.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    fn run_algorithms(&self) -> anyhow::Result<Vec<ReportEntry>> {
        let mut report_entries = Vec::with_capacity(self.config.algorithms.len());
        let topology = topology_for(&self.config.backends);
        for (run_index, algorithm) in self.config.algorithms.iter().enumerate() {
            println!(
                "starting algorithm {}/{}: {}",
                run_index + 1,
                self.config.algorithms.len(),
                algorithm.name
            );
            let run = prepare_benchmark_k8s_run(
                &self.config,
                algorithm,
                &self.manifest_path(),
                &self.output_dir,
                run_index,
            )?;
            apply_k8s(&run)?;
            let run_result = run_single_k8s(
                &run,
                self.manifest.max_concurrency,
                self.config.backends.count,
                &topology,
                &self.config.degradation.actions,
            );
            finalize_k8s_run(&run, run_result.is_err(), self.keep_resources_on_failure);
            let summary = run_result?;
            println!(
                "finished {}: success_rate={:.3}, avg_ttlt_ms={:.1}, run_dir={}",
                run.algorithm_name,
                summary.success_rate,
                summary.avg_ttlt_ms,
                run.run_dir.display()
            );
            report_entries.push(ReportEntry {
                algorithm_name: run.algorithm_name.clone(),
                pylon_queue_admission: algorithm.pylon_queue_admission.clone(),
                summary,
            });
        }
        Ok(report_entries)
    }

    fn write_report(&self, report_entries: &[ReportEntry]) -> anyhow::Result<()> {
        write_benchmark_report_artifacts(
            &self.output_dir,
            &ReportContext::from_config(&self.config),
            report_entries,
        )?;
        Ok(())
    }

    fn metadata_path(&self) -> PathBuf {
        self.output_dir.join("run-metadata.json")
    }

    fn manifest_path(&self) -> PathBuf {
        self.output_dir.join("manifest.json")
    }
}

fn finalize_k8s_run(run: &BenchmarkK8sRun, run_failed: bool, keep_resources_on_failure: bool) {
    let finalization = K8sRunFinalization::for_result(run_failed, keep_resources_on_failure);
    finalize_k8s_run_with_actions(
        &run.algorithm_name,
        finalization,
        || collect_logs(run),
        || delete_k8s(run),
    );
}

fn finalize_k8s_run_with_actions(
    algorithm_name: &str,
    finalization: K8sRunFinalization,
    mut collect_logs_action: impl FnMut() -> anyhow::Result<()>,
    mut delete_resources_action: impl FnMut() -> anyhow::Result<()>,
) {
    match finalization.logs {
        K8sLogCollection::Collect => {
            if let Err(error) = collect_logs_action() {
                eprintln!(
                    "warning: failed to collect k8s benchmark logs for {algorithm_name}: {error}"
                );
            }
        }
        K8sLogCollection::Skip => {}
    }

    match finalization.resources {
        K8sResourceDisposition::KeepFailed => {
            eprintln!("keeping k8s benchmark resources for failed run {algorithm_name}");
        }
        K8sResourceDisposition::Delete => {
            if let Err(error) = delete_resources_action() {
                eprintln!(
                    "warning: failed to delete k8s benchmark resources for {algorithm_name}: {error}"
                );
            }
        }
    }
}

fn ensure_k8s_context() -> anyhow::Result<()> {
    let output = ProcessCommand::new("kubectl")
        .arg("config")
        .arg("current-context")
        .output()
        .context("failed to query current kubectl context")?;
    if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim().is_empty() {
        anyhow::bail!(
            "no active kubectl context; configure access to a Kubernetes cluster before running Kubernetes benchmarks"
        );
    }
    Ok(())
}

fn run_single_k8s(
    run: &BenchmarkK8sRun,
    concurrency_limit: usize,
    backend_count: usize,
    topology: &RoutingTopology,
    degradation_actions: &[DegradationActionConfig],
) -> anyhow::Result<RunSummary> {
    K8sSingleRun::new(
        run,
        concurrency_limit,
        backend_count,
        topology,
        degradation_actions,
    )
    .execute()
}

fn wait_for_k8s_probe_readiness(
    runtime: &tokio::runtime::Runtime,
    run: &crate::k8s::BenchmarkK8sRun,
    backend_count: usize,
    replay_plan: &K8sReplayPlan,
) -> anyhow::Result<()> {
    K8sReadinessGate::for_run(run, backend_count, replay_plan)?.wait(runtime)
}

fn capture_collector_baseline(
    runtime: &tokio::runtime::Runtime,
    run: &crate::k8s::BenchmarkK8sRun,
    artifacts: &K8sRunArtifacts,
) -> anyhow::Result<K8sCollectorBaseline> {
    let collector_baseline = runtime.block_on(wait_for_scraped_benchmark_metrics(
        &run.collector_metrics_endpoint,
        Duration::from_secs(60),
    ))?;
    let baseline_request_totals = scraped_request_totals(&collector_baseline)
        .context("collector baseline did not expose Stargate and Pylon request counters")?;
    std::fs::write(&artifacts.collector_baseline_metrics, &collector_baseline).with_context(
        || {
            format!(
                "failed to write {}",
                artifacts.collector_baseline_metrics.display()
            )
        },
    )?;
    Ok(K8sCollectorBaseline {
        metrics: collector_baseline,
        request_totals: baseline_request_totals,
    })
}

fn drive_k8s_replay(
    runtime: &tokio::runtime::Runtime,
    run: &crate::k8s::BenchmarkK8sRun,
    manifest: Manifest,
    concurrency_limit: usize,
    degradation_actions: &[DegradationActionConfig],
    artifacts: &K8sRunArtifacts,
) -> anyhow::Result<Vec<RequestResult>> {
    let degradation_handles =
        start_degradation_actions(run, &manifest.requests, degradation_actions);
    let results = runtime.block_on(drive_manifest(
        DriveConfig {
            endpoint: format!("{}/v1/chat/completions", run.stargate_http_endpoint),
            output_path: artifacts.requests.clone(),
            concurrency_limit,
        },
        manifest,
    ));
    join_degradation_actions(degradation_handles);
    results
}

fn write_k8s_run_summary(
    runtime: &tokio::runtime::Runtime,
    run: &crate::k8s::BenchmarkK8sRun,
    topology: &RoutingTopology,
    results: &[RequestResult],
    collector_baseline: K8sCollectorBaseline,
    artifacts: &K8sRunArtifacts,
) -> anyhow::Result<RunSummary> {
    let successful_request_count = results.iter().filter(|result| result.ok).count();
    let mut summary = summarize_with_topology(results, topology);

    if let Ok(metrics) = runtime.block_on(fetch_text(&run.stargate_metrics_endpoint)) {
        std::fs::write(&artifacts.stargate_metrics, metrics)
            .with_context(|| format!("failed to write {}", artifacts.stargate_metrics.display()))?;
    }

    let collector_metrics = runtime.block_on(wait_for_post_replay_scraped_benchmark_metrics(
        &run.collector_metrics_endpoint,
        collector_baseline.request_totals,
        results.len(),
        successful_request_count,
        Duration::from_secs(60),
    ))?;
    std::fs::write(&artifacts.collector_metrics, &collector_metrics)
        .with_context(|| format!("failed to write {}", artifacts.collector_metrics.display()))?;
    summary.queue_admission_summary = queue_admission_summary_delta_from_prometheus(
        &collector_baseline.metrics,
        &collector_metrics,
    );
    summary.routing_selection_summary = routing_selection_summary_delta_from_prometheus(
        &collector_baseline.metrics,
        &collector_metrics,
    );
    let summary_bytes =
        serde_json::to_vec_pretty(&summary).context("failed to serialize run summary")?;
    std::fs::write(&artifacts.summary, summary_bytes)
        .with_context(|| format!("failed to write {}", artifacts.summary.display()))?;

    Ok(summary)
}

fn start_degradation_actions(
    run: &crate::k8s::BenchmarkK8sRun,
    requests: &[ManifestRequest],
    actions: &[DegradationActionConfig],
) -> Vec<std::thread::JoinHandle<()>> {
    actions
        .iter()
        .map(|action| {
            let action = action.clone();
            let run = run.clone();
            let delay = requests
                .get(action.at_request)
                .map(|request| Duration::from_millis(request.scheduled_offset_ms))
                .unwrap_or_default();
            std::thread::spawn(move || {
                std::thread::sleep(delay);
                let result = match action.action {
                    DegradationActionKind::DeleteBackendPod => {
                        delete_backend_pod(&run, action.backend_index)
                    }
                    DegradationActionKind::ScaleBackend { replicas } => {
                        scale_backend(&run, action.backend_index, replicas)
                    }
                };
                if let Err(error) = result {
                    eprintln!(
                        "warning: degradation action failed for backend-{} in {}: {error}",
                        action.backend_index, run.algorithm_name
                    );
                }
            })
        })
        .collect()
}

fn join_degradation_actions(handles: Vec<std::thread::JoinHandle<()>>) {
    for handle in handles {
        let _ = handle.join();
    }
}

async fn wait_for_http_ok(url: &str, timeout: Duration) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    let client = reqwest::Client::new();
    loop {
        if let Ok(response) = client.get(url).send().await
            && response.status().is_success()
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for {}", url);
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn fetch_text(url: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to fetch {}", url))?;
    response
        .text()
        .await
        .with_context(|| format!("failed to read response body from {}", url))
}

async fn wait_for_scraped_benchmark_metrics(
    collector_metrics_endpoint: &str,
    timeout: Duration,
) -> anyhow::Result<String> {
    let deadline = Instant::now() + timeout;
    let mut last_metrics_len = 0usize;
    loop {
        if let Ok(metrics) = fetch_text(collector_metrics_endpoint).await {
            last_metrics_len = metrics.len();
            if has_scraped_benchmark_metrics(&metrics) {
                return Ok(metrics);
            }
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for OTel collector to scrape benchmark metrics from {} (last_response_bytes={})",
                collector_metrics_endpoint,
                last_metrics_len
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn wait_for_post_replay_scraped_benchmark_metrics(
    collector_metrics_endpoint: &str,
    baseline: ScrapedRequestTotals,
    replay_request_count: usize,
    replay_success_count: usize,
    timeout: Duration,
) -> anyhow::Result<String> {
    let started_at = Instant::now();
    let deadline = started_at + timeout;
    let mut last_metrics_len = 0usize;
    loop {
        if let Ok(metrics) = fetch_text(collector_metrics_endpoint).await {
            last_metrics_len = metrics.len();
            if started_at.elapsed() >= COLLECTOR_SCRAPE_SETTLE_DELAY
                && has_post_replay_scraped_benchmark_metrics(
                    &metrics,
                    baseline,
                    replay_request_count,
                    replay_success_count,
                )
            {
                return Ok(metrics);
            }
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for post-replay OTel collector metrics from {} (last_response_bytes={})",
                collector_metrics_endpoint,
                last_metrics_len
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub(crate) fn has_scraped_benchmark_metrics(metrics: &str) -> bool {
    has_any_metric(
        metrics,
        &["stargate_requests_total", "stargate_requests_total_total"],
    ) && has_any_metric(
        metrics,
        &["pylon_requests_total", "pylon_requests_total_total"],
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScrapedRequestTotals {
    stargate: f64,
    pylon: f64,
}

pub(crate) fn scraped_request_totals(metrics: &str) -> Option<ScrapedRequestTotals> {
    if !has_scraped_benchmark_metrics(metrics) {
        return None;
    }
    Some(ScrapedRequestTotals {
        stargate: metric_total(
            metrics,
            &["stargate_requests_total", "stargate_requests_total_total"],
        ),
        pylon: metric_total(
            metrics,
            &["pylon_requests_total", "pylon_requests_total_total"],
        ),
    })
}

pub(crate) fn has_post_replay_scraped_benchmark_metrics(
    metrics: &str,
    baseline: ScrapedRequestTotals,
    replay_request_count: usize,
    replay_success_count: usize,
) -> bool {
    let Some(current) = scraped_request_totals(metrics) else {
        return false;
    };
    current.stargate >= baseline.stargate + replay_request_count as f64
        && current.pylon >= baseline.pylon + replay_success_count as f64
}

fn metric_total(metrics: &str, names: &[&str]) -> f64 {
    metrics
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let series = fields.next()?;
            if !names
                .iter()
                .any(|name| series.starts_with(&format!("{name}{{")) || series == *name)
            {
                return None;
            }
            fields.next()?.parse::<f64>().ok()
        })
        .sum()
}

fn has_any_metric(metrics: &str, names: &[&str]) -> bool {
    metrics.lines().any(|line| {
        names.iter().any(|name| {
            line.starts_with(&format!("{name}{{")) || line.starts_with(&format!("{name} "))
        })
    })
}

async fn wait_for_active_backend_counts(
    metrics_endpoints: &[String],
    model: &str,
    routing_key: Option<&str>,
    expected_count: usize,
    timeout: Duration,
) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    let mut last_counts = Vec::new();
    loop {
        last_counts.clear();
        for metrics_endpoint in metrics_endpoints {
            let count = match fetch_text(metrics_endpoint).await {
                Ok(metrics) => active_backend_count(&metrics, model, routing_key),
                Err(_) => None,
            };
            last_counts.push(count);
        }
        if active_backend_counts_ready(&last_counts, expected_count) {
            return Ok(());
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for {expected_count} active benchmark backends on every stargate metrics endpoint {:?} (model={}, routing_key={:?}, last_counts={:?})",
                metrics_endpoints,
                model,
                routing_key,
                last_counts
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub(crate) fn active_backend_counts_ready(counts: &[Option<usize>], expected_count: usize) -> bool {
    !counts.is_empty()
        && counts
            .iter()
            .all(|count| count.is_some_and(|count| count >= expected_count))
}

pub(crate) fn active_backend_count(
    metrics: &str,
    model: &str,
    routing_key: Option<&str>,
) -> Option<usize> {
    metrics.lines().find_map(|line| {
        if !line.starts_with("stargate_active_inference_servers{") {
            return None;
        }
        let (metric, value) = line.rsplit_once(' ')?;
        let metric_model = prometheus_label_value(metric, "model")?;
        let metric_routing_key = prometheus_label_value(metric, "routing_key").unwrap_or("");
        if metric_model != model || metric_routing_key != routing_key.unwrap_or("") {
            return None;
        }
        value.parse::<f64>().ok().map(|value| value as usize)
    })
}

fn prometheus_label_value<'a>(metric: &'a str, label: &str) -> Option<&'a str> {
    let needle = format!(r#"{label}=""#);
    let start = metric.find(&needle)? + needle.len();
    let rest = &metric[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

async fn wait_for_routing(
    endpoint: &str,
    model: &str,
    request: &ManifestRequest,
    timeout: Duration,
) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "benchmark-ready"}],
        "max_tokens": 1,
        "stream": true,
    });
    let mut last_status = None;
    let probe_cache_affinity_key = routing_probe_cache_affinity_key(request);
    loop {
        let mut builder = client
            .post(endpoint)
            .header("content-type", "application/json")
            .header("x-model", model)
            .header(
                "x-request-id",
                format!("benchmark-ready-probe-{}", request.request_index),
            )
            .header("x-input-tokens", "1")
            .header("x-output-tokens", "1");
        if let Some(routing_key) = &request.routing_key {
            builder = builder.header("x-routing-key", routing_key);
        }
        if let Some(cache_affinity_key) = &probe_cache_affinity_key {
            builder = builder.header("x-cache-affinity-key", cache_affinity_key);
        }
        match builder.json(&body).send().await {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(response) => {
                last_status = Some(response.status());
            }
            Err(_) => {}
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "timed out waiting for routable benchmark traffic on {} (model={}, routing_key={:?}, cache_affinity_key={:?}, last_status={:?})",
                endpoint,
                model,
                request.routing_key,
                probe_cache_affinity_key,
                last_status
            );
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub(crate) fn routing_probe_cache_affinity_key(request: &ManifestRequest) -> Option<String> {
    request.cache_affinity_key.as_ref().map(|_| {
        format!(
            "__stargate_bench_benchmark-ready-probe-{}",
            request.request_index
        )
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::path::Path;
    use std::sync::mpsc;

    use super::*;
    use crate::config::{
        AlgorithmConfig, ArrivalPatternConfig, BackendConfig, BackendProfile, RegistrationConfig,
        ScenarioMetadata, ServiceTimeConfig, StargateConfig, TokenDistributionConfig,
        TrafficPatternConfig, UniformTrafficConfig,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn k8s_run_finalization_policy_matches_existing_cleanup_truth_table() {
        assert_eq!(
            K8sRunFinalization::for_result(true, false),
            K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::Delete,
            }
        );
        assert_eq!(
            K8sRunFinalization::for_result(true, true),
            K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::KeepFailed,
            }
        );
        assert_eq!(
            K8sRunFinalization::for_result(false, true),
            K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::Delete,
            }
        );
        assert_eq!(
            K8sRunFinalization::for_result(false, false),
            K8sRunFinalization {
                logs: K8sLogCollection::Skip,
                resources: K8sResourceDisposition::Delete,
            }
        );
    }

    #[test]
    fn k8s_run_finalization_executes_selected_cleanup_actions() {
        assert_eq!(
            recorded_finalization_actions(K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::Delete,
            }),
            vec!["logs", "delete"]
        );
        assert_eq!(
            recorded_finalization_actions(K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::KeepFailed,
            }),
            vec!["logs"]
        );
        assert_eq!(
            recorded_finalization_actions(K8sRunFinalization {
                logs: K8sLogCollection::Skip,
                resources: K8sResourceDisposition::Delete,
            }),
            vec!["delete"]
        );
    }

    #[test]
    fn k8s_run_finalization_keeps_cleanup_errors_non_fatal() {
        let actions = RefCell::new(Vec::new());

        finalize_k8s_run_with_actions(
            "power-of-two",
            K8sRunFinalization {
                logs: K8sLogCollection::Collect,
                resources: K8sResourceDisposition::Delete,
            },
            || {
                actions.borrow_mut().push("logs");
                Err(anyhow::anyhow!("log collection failed"))
            },
            || {
                actions.borrow_mut().push("delete");
                Err(anyhow::anyhow!("resource deletion failed"))
            },
        );

        assert_eq!(actions.into_inner(), vec!["logs", "delete"]);
    }

    fn recorded_finalization_actions(finalization: K8sRunFinalization) -> Vec<&'static str> {
        let actions = RefCell::new(Vec::new());
        finalize_k8s_run_with_actions(
            "power-of-two",
            finalization,
            || {
                actions.borrow_mut().push("logs");
                Ok(())
            },
            || {
                actions.borrow_mut().push("delete");
                Ok(())
            },
        );
        actions.into_inner()
    }

    #[test]
    fn k8s_benchmark_command_writes_metadata_before_strict_preflight_failure() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let command = super::K8sBenchmarkCommand::from_metadata(
            benchmark_config_for_test(),
            manifest_with_requests(vec![manifest_request(0, None)]),
            tempdir.path().to_path_buf(),
            false,
            strict_failing_metadata(),
        );

        let error = command
            .run_to_completion()
            .expect_err("strict preflight should fail before kubectl context checks");

        let metadata_path = tempdir.path().join("run-metadata.json");
        assert!(metadata_path.exists());
        assert!(!tempdir.path().join("manifest.json").exists());
        let metadata = serde_json::from_slice::<crate::metadata::RunMetadata>(
            &std::fs::read(&metadata_path).expect("metadata should read"),
        )
        .expect("metadata should parse");
        assert_eq!(metadata.preflight.failure_count, 1);
        assert!(
            error
                .to_string()
                .contains("strict reliability preflight failed with 1 failure(s)")
        );
    }

    #[test]
    fn k8s_benchmark_command_success_writes_manifest_and_empty_report_without_algorithms() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let mut config = benchmark_config_for_test();
        config.algorithms.clear();
        let command = super::K8sBenchmarkCommand::from_metadata(
            config,
            manifest_with_requests(vec![manifest_request(0, None)]),
            tempdir.path().to_path_buf(),
            false,
            passing_metadata(),
        );

        command
            .run_with_context_check(|| Ok(()))
            .expect("empty algorithm run should write command artifacts");

        assert!(tempdir.path().join("run-metadata.json").exists());
        assert!(tempdir.path().join("manifest.json").exists());
        let comparison: serde_json::Value = serde_json::from_slice(
            &std::fs::read(tempdir.path().join("comparison.json")).expect("comparison should read"),
        )
        .expect("comparison should parse");
        assert_eq!(comparison.as_array().expect("comparison is array").len(), 0);
        assert!(tempdir.path().join("report.md").exists());
    }

    #[test]
    fn k8s_replay_plan_loads_manifest_and_clones_first_request_for_probe() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let manifest_path = tempdir.path().join("manifest.json");
        let first_request = manifest_request(0, Some("tenant-a"));
        let second_request = manifest_request(1, Some("tenant-b"));
        write_manifest_json(
            &manifest_path,
            &manifest_with_requests(vec![first_request.clone(), second_request]),
        )
        .expect("manifest should write");

        let replay_plan = K8sReplayPlan::load(&manifest_path).expect("replay plan should load");

        assert_eq!(replay_plan.manifest.requests.len(), 2);
        assert_eq!(
            replay_plan.routing_probe_request.request_id,
            first_request.request_id
        );
        assert_eq!(
            replay_plan.routing_probe_request.routing_key.as_deref(),
            Some("tenant-a")
        );
    }

    #[test]
    fn k8s_replay_plan_rejects_empty_manifest() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let manifest_path = tempdir.path().join("empty-manifest.json");
        write_manifest_json(&manifest_path, &manifest_with_requests(Vec::new()))
            .expect("manifest should write");

        let error = K8sReplayPlan::load(&manifest_path)
            .expect_err("empty manifests should not produce a replay plan");

        assert!(
            error
                .to_string()
                .contains("benchmark manifest must contain at least one request")
        );
    }

    #[test]
    fn k8s_replay_artifacts_use_canonical_run_filenames() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let run_dir = tempdir.path().join("run-power-of-two");

        let artifacts = K8sRunArtifacts::for_run_dir(&run_dir);

        assert_eq!(
            artifacts.collector_baseline_metrics,
            run_dir.join("collector-baseline-metrics.prom")
        );
        assert_eq!(artifacts.requests, run_dir.join("requests.jsonl"));
        assert_eq!(artifacts.stargate_metrics, run_dir.join("metrics.prom"));
        assert_eq!(
            artifacts.collector_metrics,
            run_dir.join("collector-metrics.prom")
        );
        assert_eq!(artifacts.summary, run_dir.join("summary.json"));
    }

    #[test]
    fn k8s_readiness_gate_checks_health_metrics_and_routing_with_local_servers() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime should create");
        let metrics = r#"
stargate_active_inference_servers{model="dummy-model",routing_key="tenant-a"} 1
"#;
        let (stargate_endpoint, stargate_server) = spawn_fixed_text_server(&runtime, "", 2);
        let stargate_base = stargate_endpoint
            .strip_suffix("/metrics")
            .expect("test endpoint should include metrics suffix");
        let (metrics_endpoint, metrics_server) = spawn_fixed_text_server(&runtime, metrics, 1);
        let gate = K8sReadinessGate {
            health_url: format!("{stargate_base}/healthz"),
            metrics_endpoints: vec![metrics_endpoint],
            routing_url: format!("{stargate_base}/v1/chat/completions"),
            model: "dummy-model".to_string(),
            expected_backend_count: 1,
            routing_probe_request: manifest_request(7, Some("tenant-a")),
            timeout: Duration::from_secs(1),
        };

        gate.wait(&runtime).expect("readiness gate should pass");

        runtime
            .block_on(stargate_server)
            .expect("stargate readiness server should finish");
        runtime
            .block_on(metrics_server)
            .expect("metrics readiness server should finish");
    }

    #[test]
    fn k8s_replay_collector_baseline_writes_metrics_and_totals() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime should create");
        let metrics = r#"
stargate_requests_total_total{model="dummy-model",status="ok"} 2
pylon_requests_total_total{model="dummy-model",status="complete"} 3
"#;
        let (collector_endpoint, server) = spawn_fixed_text_server(&runtime, metrics, 1);
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let run = benchmark_run_for_test(
            tempdir.path(),
            "http://127.0.0.1:9/metrics",
            &collector_endpoint,
        );
        let artifacts = K8sRunArtifacts::for_run_dir(&run.run_dir);

        let baseline = capture_collector_baseline(&runtime, &run, &artifacts)
            .expect("baseline should capture");

        assert_eq!(
            baseline.request_totals,
            ScrapedRequestTotals {
                stargate: 2.0,
                pylon: 3.0,
            }
        );
        assert_eq!(
            std::fs::read_to_string(&artifacts.collector_baseline_metrics)
                .expect("baseline artifact should read"),
            metrics
        );
        runtime
            .block_on(server)
            .expect("metrics server should finish");
    }

    #[test]
    fn k8s_run_summary_writes_post_replay_artifacts_and_metric_deltas() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime should create");
        let stargate_metrics = r#"
stargate_active_inference_servers{model="dummy-model",routing_key=""} 1
"#;
        let collector_baseline_metrics = r#"
stargate_requests_total_total{model="dummy-model",status="ok"} 2
pylon_requests_total_total{model="dummy-model",status="complete"} 2
pylon_queue_admission_decisions_total_total{model_id="dummy-model",result="accepted"} 1
stargate_routing_selections_total_total{model="dummy-model",selection="primary"} 1
stargate_routing_selections_total_total{model="dummy-model",selection="fallback"} 1
"#;
        let collector_post_metrics = r#"
stargate_requests_total_total{model="dummy-model",status="ok"} 3
pylon_requests_total_total{model="dummy-model",status="complete"} 3
pylon_queue_admission_decisions_total_total{model_id="dummy-model",result="accepted"} 4
stargate_routing_selections_total_total{model="dummy-model",selection="primary"} 6
stargate_routing_selections_total_total{model="dummy-model",selection="fallback"} 3
"#;
        let (stargate_endpoint, stargate_server) =
            spawn_fixed_text_server(&runtime, stargate_metrics, 1);
        let (collector_endpoint, collector_stop, collector_server) =
            spawn_fixed_text_server_until_stop(&runtime, collector_post_metrics);
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let run = benchmark_run_for_test(tempdir.path(), &stargate_endpoint, &collector_endpoint);
        let artifacts = K8sRunArtifacts::for_run_dir(&run.run_dir);
        let config = benchmark_config_for_test();
        let topology = topology_for(&config.backends);

        let summary = write_k8s_run_summary(
            &runtime,
            &run,
            &topology,
            &[successful_request_result()],
            K8sCollectorBaseline {
                metrics: collector_baseline_metrics.to_string(),
                request_totals: ScrapedRequestTotals {
                    stargate: 2.0,
                    pylon: 2.0,
                },
            },
            &artifacts,
        )
        .expect("summary artifacts should write");

        assert_eq!(summary.request_count, 1);
        assert_eq!(summary.queue_admission_summary.pylon_accepted_count, 3.0);
        assert_eq!(summary.routing_selection_summary.primary_count, 5.0);
        assert_eq!(summary.routing_selection_summary.fallback_count, 2.0);
        assert_eq!(
            std::fs::read_to_string(&artifacts.stargate_metrics)
                .expect("stargate metrics artifact should read"),
            stargate_metrics
        );
        assert_eq!(
            std::fs::read_to_string(&artifacts.collector_metrics)
                .expect("collector metrics artifact should read"),
            collector_post_metrics
        );
        let summary_artifact = serde_json::from_slice::<RunSummary>(
            &std::fs::read(&artifacts.summary).expect("summary artifact should read"),
        )
        .expect("summary artifact should parse");
        assert_eq!(
            summary_artifact
                .queue_admission_summary
                .pylon_accepted_count,
            3.0
        );
        runtime
            .block_on(stargate_server)
            .expect("stargate metrics server should finish");
        collector_stop
            .send(())
            .expect("collector metrics server should receive stop signal");
        runtime
            .block_on(collector_server)
            .expect("collector metrics server should finish");
    }

    #[test]
    fn k8s_replay_drives_empty_manifest_to_requests_artifact_without_network() {
        let runtime = tokio::runtime::Runtime::new().expect("runtime should create");
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let run = benchmark_run_for_test(
            tempdir.path(),
            "http://127.0.0.1:9/metrics",
            "http://127.0.0.1:9/metrics",
        );
        let artifacts = K8sRunArtifacts::for_run_dir(&run.run_dir);

        let results = drive_k8s_replay(
            &runtime,
            &run,
            manifest_with_requests(Vec::new()),
            1,
            &[],
            &artifacts,
        )
        .expect("empty replay should complete without requests");

        assert!(results.is_empty());
        assert_eq!(
            std::fs::read_to_string(&artifacts.requests).expect("requests artifact should read"),
            ""
        );
    }

    fn manifest_with_requests(requests: Vec<ManifestRequest>) -> Manifest {
        Manifest {
            manifest_version: 1,
            benchmark_name: "k8s-replay-test".to_string(),
            metadata: crate::config::ScenarioMetadata::default(),
            model: "dummy-model".to_string(),
            seed: 7,
            request_count: requests.len(),
            max_concurrency: 1,
            stargate_count: 1,
            backend_count: 1,
            cluster_count: 1,
            pylons_per_cluster: 1,
            requests,
        }
    }

    fn benchmark_config_for_test() -> BenchmarkConfig {
        BenchmarkConfig {
            name: "k8s-command-test".to_string(),
            metadata: ScenarioMetadata::default(),
            model: "dummy-model".to_string(),
            seed: Some(7),
            request_count: 1,
            max_concurrency: 1,
            tunnel_protocol: stargate_protocol::TunnelTransportProtocol::Custom,
            stargates: StargateConfig { count: 1 },
            backends: BackendConfig {
                count: 1,
                cluster_id_template: None,
                pylons_per_cluster: 1,
                profiles: Vec::new(),
                profile: BackendProfile {
                    name: "balanced".to_string(),
                    weight: 1.0,
                    max_concurrent_requests: None,
                    kv_cache_capacity_tokens: 0,
                    service_time_ms: ServiceTimeConfig {
                        ttft_mean: 150,
                        ttft_jitter_ms: 10,
                        decode_tokens_per_s: 50,
                        decode_jitter_ms: 0,
                        prefill_tokens_per_s: None,
                    },
                    registration: RegistrationConfig {
                        last_mean_input_tps: 100.0,
                    },
                },
            },
            traffic_pattern: TrafficPatternConfig::Uniform(UniformTrafficConfig {
                routing_keys: 1,
                cache_affinity_keys: 1,
                input_tokens: TokenDistributionConfig::Constant { value: 100 },
                output_tokens: TokenDistributionConfig::Constant { value: 20 },
                arrival: ArrivalPatternConfig::Constant { interval_ms: 10 },
            }),
            degradation: crate::config::DegradationConfig::default(),
            algorithms: vec![AlgorithmConfig {
                name: "power-of-two".to_string(),
                config: serde_json::json!({"default": "power-of-two"}),
                pylon_queue_admission: None,
            }],
        }
    }

    fn passing_metadata() -> crate::metadata::RunMetadata {
        metadata_with_preflight(crate::metadata::PreflightReport {
            checks: Vec::new(),
            warning_count: 0,
            failure_count: 0,
            should_fail: false,
        })
    }

    fn strict_failing_metadata() -> crate::metadata::RunMetadata {
        metadata_with_preflight(crate::metadata::PreflightReport {
            checks: vec![crate::metadata::PreflightCheck {
                name: "release-binary".to_string(),
                level: crate::metadata::PreflightLevel::Failure,
                message: "debug build".to_string(),
            }],
            warning_count: 0,
            failure_count: 1,
            should_fail: true,
        })
    }

    fn metadata_with_preflight(
        preflight: crate::metadata::PreflightReport,
    ) -> crate::metadata::RunMetadata {
        crate::metadata::RunMetadata {
            schema_version: 3,
            benchmark_tier: BenchmarkTier::LocalK8sSmoke,
            reliability_mode: ReliabilityMode::Strict,
            driver_mode: DriverMode::ExternalNodePort,
            command_line: vec!["stargate-bench".to_string(), "run".to_string()],
            started_at_unix_seconds: 0,
            current_exe: None,
            working_dir: None,
            git: crate::metadata::GitMetadata::default(),
            rust: crate::metadata::RustMetadata::default(),
            host: crate::metadata::HostMetadata::default(),
            kubernetes: crate::metadata::KubernetesMetadata::default(),
            preflight,
            known_limitations: Vec::new(),
        }
    }

    fn manifest_request(request_index: usize, routing_key: Option<&str>) -> ManifestRequest {
        ManifestRequest {
            request_index,
            request_id: format!("request-{request_index}"),
            scheduled_offset_ms: request_index as u64,
            routing_key: routing_key.map(str::to_string),
            cache_affinity_key: Some(format!("cache-key-{request_index}")),
            input_tokens: 128,
            output_tokens: 16,
            backend_behavior_class: "default".to_string(),
        }
    }

    fn benchmark_run_for_test(
        run_dir: &Path,
        stargate_metrics_endpoint: &str,
        collector_metrics_endpoint: &str,
    ) -> crate::k8s::BenchmarkK8sRun {
        std::fs::create_dir_all(run_dir).expect("run dir should create");
        crate::k8s::BenchmarkK8sRun {
            algorithm_name: "power-of-two".to_string(),
            manifest_path: run_dir.join("manifest.json"),
            run_dir: run_dir.to_path_buf(),
            stargate_ns: "sgbench-sg-power-of-two".to_string(),
            backends_ns: "sgbench-be-power-of-two".to_string(),
            stargate_count: 1,
            nodeport_host: "node.test".to_string(),
            stargate_http_endpoint: "http://node.test:30080".to_string(),
            stargate_metrics_endpoint: stargate_metrics_endpoint.to_string(),
            collector_metrics_endpoint: collector_metrics_endpoint.to_string(),
            backend_upstream_indices: vec![0],
            upstream_backend_indices: vec![0],
        }
    }

    fn successful_request_result() -> RequestResult {
        RequestResult {
            request_index: 0,
            request_id: "request-0".to_string(),
            routing_key: None,
            cache_affinity_key: Some("cache-key-0".to_string()),
            input_tokens: 128,
            output_tokens: 16,
            scheduled_offset_ms: 0,
            status_code: 200,
            selected_backend_id: Some("backend-0".to_string()),
            dispatch_offset_ms: 0,
            response_headers_ms: Some(10),
            first_output_ms: Some(20),
            completion_ms: 40,
            kv_cache_hit: Some(false),
            kv_cache_reused_input_tokens: Some(0),
            kv_cache_uncached_input_tokens: Some(128),
            kv_cache_evicted_entries: Some(0),
            kv_cache_evicted_tokens: Some(0),
            ok: true,
            error: None,
        }
    }

    fn spawn_fixed_text_server(
        runtime: &tokio::runtime::Runtime,
        body: &'static str,
        request_count: usize,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let (addr_tx, addr_rx) = mpsc::channel();
        let server = runtime.spawn(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test metrics server should bind");
            addr_tx
                .send(listener.local_addr().expect("local addr should exist"))
                .expect("server addr should send");
            for _ in 0..request_count {
                let (mut socket, _) = listener.accept().await.expect("request should connect");
                tokio::spawn(async move {
                    let mut bytes = Vec::new();
                    let mut buffer = [0u8; 1024];
                    loop {
                        let read = socket.read(&mut buffer).await.expect("request should read");
                        if read == 0 {
                            break;
                        }
                        bytes.extend_from_slice(&buffer[..read]);
                        if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket
                        .write_all(response.as_bytes())
                        .await
                        .expect("response should write");
                });
            }
        });
        let addr = addr_rx.recv().expect("server addr should receive");
        (format!("http://{addr}/metrics"), server)
    }

    fn spawn_fixed_text_server_until_stop(
        runtime: &tokio::runtime::Runtime,
        body: &'static str,
    ) -> (
        String,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let (addr_tx, addr_rx) = mpsc::channel();
        let (stop_tx, mut stop_rx) = tokio::sync::oneshot::channel();
        let server = runtime.spawn(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test metrics server should bind");
            addr_tx
                .send(listener.local_addr().expect("local addr should exist"))
                .expect("server addr should send");
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    accepted = listener.accept() => {
                        let (mut socket, _) = accepted.expect("request should connect");
                        tokio::spawn(async move {
                            let mut bytes = Vec::new();
                            let mut buffer = [0u8; 1024];
                            loop {
                                let read = socket.read(&mut buffer).await.expect("request should read");
                                if read == 0 {
                                    break;
                                }
                                bytes.extend_from_slice(&buffer[..read]);
                                if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                                    break;
                                }
                            }
                            let response = format!(
                                "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            socket
                                .write_all(response.as_bytes())
                                .await
                                .expect("response should write");
                        });
                    }
                }
            }
        });
        let addr = addr_rx.recv().expect("server addr should receive");
        (format!("http://{addr}/metrics"), stop_tx, server)
    }
}
