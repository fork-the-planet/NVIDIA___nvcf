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

use anyhow::Context;

use crate::metadata::{
    BenchmarkTier, DriverMode, ReliabilityMode, RunMetadata, collect_run_metadata,
    write_run_metadata,
};

use super::{
    TransportBenchConfig, TransportBenchmarkOutcome, render_transport_benchmark_report,
    run_transport_benchmark, write_transport_benchmark_artifacts,
};

pub fn run_transport_benchmark_command(
    config: TransportBenchConfig,
    reliability_mode: ReliabilityMode,
    output_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    TransportBenchmarkCommand::collect(config, reliability_mode, output_dir).run_to_completion()
}

struct TransportBenchmarkCommand {
    config: TransportBenchConfig,
    metadata: RunMetadata,
    output_dir: Option<PathBuf>,
}

impl TransportBenchmarkCommand {
    fn collect(
        config: TransportBenchConfig,
        reliability_mode: ReliabilityMode,
        output_dir: Option<PathBuf>,
    ) -> Self {
        let metadata = collect_run_metadata(
            BenchmarkTier::TransportLoopback,
            reliability_mode,
            DriverMode::LocalProcess,
        );
        Self {
            config,
            metadata,
            output_dir,
        }
    }

    #[cfg(test)]
    fn from_metadata(
        config: TransportBenchConfig,
        metadata: RunMetadata,
        output_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            config,
            metadata,
            output_dir,
        }
    }

    fn run_to_completion(&self) -> anyhow::Result<()> {
        self.prepare_to_run()?;
        self.run_and_publish()
    }

    fn prepare_to_run(&self) -> anyhow::Result<()> {
        self.write_metadata()?;
        self.ensure_preflight()
    }

    fn run_and_publish(&self) -> anyhow::Result<()> {
        let outcome = self.run_benchmark()?;
        self.publish_outcome(&outcome)
    }

    fn publish_outcome(&self, outcome: &TransportBenchmarkOutcome) -> anyhow::Result<()> {
        self.print_report(outcome);
        self.write_artifacts(outcome)
    }

    fn write_metadata(&self) -> anyhow::Result<()> {
        let Some(output_dir) = self.output_dir() else {
            return Ok(());
        };
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("failed to create {}", output_dir.display()))?;
        write_run_metadata(&self.metadata_path(output_dir), &self.metadata)
    }

    fn ensure_preflight(&self) -> anyhow::Result<()> {
        if self.metadata.preflight.should_fail {
            anyhow::bail!(
                "strict reliability preflight failed with {} failure(s); inspect run-metadata.json when --output-dir is set",
                self.metadata.preflight.failure_count
            );
        }
        Ok(())
    }

    fn run_benchmark(&self) -> anyhow::Result<TransportBenchmarkOutcome> {
        let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
        runtime.block_on(run_transport_benchmark(self.config))
    }

    fn print_report(&self, outcome: &TransportBenchmarkOutcome) {
        let report = render_transport_benchmark_report(outcome);
        println!("{report}");
    }

    fn write_artifacts(&self, outcome: &TransportBenchmarkOutcome) -> anyhow::Result<()> {
        let Some(output_dir) = self.output_dir() else {
            return Ok(());
        };
        write_transport_benchmark_artifacts(output_dir, outcome)?;
        println!(
            "wrote transport benchmark artifacts to {}",
            output_dir.display()
        );
        Ok(())
    }

    fn output_dir(&self) -> Option<&Path> {
        self.output_dir.as_deref()
    }

    fn metadata_path(&self, output_dir: &Path) -> PathBuf {
        output_dir.join("run-metadata.json")
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::metadata::{
        BenchmarkTier, DriverMode, GitMetadata, HostMetadata, KubernetesMetadata, PreflightCheck,
        PreflightLevel, PreflightReport, ReliabilityMode, RunMetadata, RustMetadata,
    };
    use crate::transport::TransportBenchConfig;

    fn tiny_transport_config() -> TransportBenchConfig {
        TransportBenchConfig {
            request_count: 1,
            concurrency: 1,
            quic_connections: 1,
            warmup_requests: 0,
            request_body_bytes: 1,
            response_body_bytes: 1,
            request_chunk_bytes: 1,
            response_chunk_bytes: 1,
            quic_send_fairness: true,
            http3_send_grease: true,
            trials: 1,
            warmup_trials: 0,
            cooldown_ms: 0,
            randomize_order: false,
            noise_threshold_cv: 0.02,
            min_effect_size_percent: 1.0,
        }
    }

    fn strict_failing_metadata() -> RunMetadata {
        RunMetadata {
            schema_version: 3,
            benchmark_tier: BenchmarkTier::TransportLoopback,
            reliability_mode: ReliabilityMode::Strict,
            driver_mode: DriverMode::LocalProcess,
            command_line: vec!["stargate-bench".to_string(), "transport-bench".to_string()],
            started_at_unix_seconds: 0,
            current_exe: None,
            working_dir: None,
            git: GitMetadata::default(),
            rust: RustMetadata::default(),
            host: HostMetadata::default(),
            kubernetes: KubernetesMetadata::default(),
            preflight: PreflightReport {
                checks: vec![PreflightCheck {
                    name: "release-binary".to_string(),
                    level: PreflightLevel::Failure,
                    message: "debug build".to_string(),
                }],
                warning_count: 0,
                failure_count: 1,
                should_fail: true,
            },
            known_limitations: Vec::new(),
        }
    }

    #[test]
    fn transport_command_writes_metadata_before_strict_preflight_failure() {
        let tempdir = tempfile::tempdir().expect("tempdir should create");
        let command = super::TransportBenchmarkCommand::from_metadata(
            tiny_transport_config(),
            strict_failing_metadata(),
            Some(PathBuf::from(tempdir.path())),
        );

        let error = command
            .run_to_completion()
            .expect_err("strict preflight should fail before benchmark execution");

        let metadata_path = tempdir.path().join("run-metadata.json");
        assert!(metadata_path.exists());
        let metadata = serde_json::from_slice::<RunMetadata>(
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
}
