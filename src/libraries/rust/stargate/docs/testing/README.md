# Testing Docs

> Type: Index. Verification entrypoint for behavior, coverage, and CI work.

Use these docs to choose the smallest verification set that proves a change.

## Current Contracts

- [Feature and behavior test matrix](../feature-behavior-test-matrix.md):
  externally visible Stargate/Pylon behavior, including registration, bringup,
  routing, and Kubernetes lifecycle coverage.
- [Code coverage and test quality](../code-coverage.md): local and CI
  coverage gates, patch coverage, CRAP baselines, standalone mutation testing,
  and report interpretation.
- [Docs health check](../reference/README.md): docs references are validated by
  `scripts/check_docs.sh`.

## Local Kubernetes Integration Execution

`python3 scripts/run_k8s_integ.py` prepares the local cluster and host-side
probes once, then runs the full suite in conflict-safe waves:

1. smoke and transport lanes in parallel;
2. alpha and beta lifecycle lanes in parallel;
3. router lifecycle lane by itself.

Use `--suite smoke`, `--suite transport`, or an exact `--case` for focused
serial debugging.
`python3 scripts/run_tilt.py ci --context kind-kind --timeout 30m` also runs the
namespace-isolated global-watch integration resource concurrently with the
local suite.

## Related Context

- [Architecture docs](../architecture/README.md): behavior contracts that tests
  should prove.
- [Local benchmark runner](../local-benchmark-runner.md): benchmark checks and
  benchmark-specific validation.
