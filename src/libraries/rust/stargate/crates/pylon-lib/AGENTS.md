<coding_guidelines>
# crates/pylon-lib

`pylon-lib` owns pylon backend sidecar behavior. A pylon is the former `stargate-client`: it manages local input-TPS bootstrap, registration streams, reverse tunnels, local upstream forwarding, request observation, metrics, health monitoring, and active canaries.

## Local Invariants

- The QUIC tunnel requires `x-request-id`, `x-model`, and `x-input-tokens`; missing required headers return HTTP `400` and must not be forwarded upstream.
- For proxied gateway traffic, treat `x-request-id` as the canonical globally unique request identity and do not synthesize replacement local request IDs.
- `x-stargate-retryable`, `x-stargate-retry-reason`, and `x-stargate-retry-after-ms` are response metadata emitted by the pylon, not caller request headers.
- `x-stargate-expected-queue-ms` is internal request metadata emitted by Stargate only. Strip it before forwarding upstream and never trust a caller-supplied value.
- Queue mismatch retry responses are local pylon admission decisions and must use pylon-emitted retry response metadata.
- Request observation assumes streamed responses. Derive output progress from SSE `data:` events, not non-streaming JSON bodies.
- Terminal request-observer transitions are invariants. Calling terminalization from a terminal state should fail loudly.
- Model advertisement is gated by both caller-provided status and bringup lifecycle state.
- Keep current registration advertisement inputs in one authoritative shared snapshot; do not split status, stats, and bringup state across private watch channels.
- A registration client is idle or owns one complete running session root; discovery, registration, router, and reverse-tunnel tasks derive child cancellation from that root. Finite startup bootstrap and ongoing bringup are separate Pylon-owned lifecycles established before registration starts.
- Every production pylon background task must be owned by the shared abort-on-drop task owner or a `JoinSet`; do not store or return bare `JoinHandle`s.
- Keep cooperative cancellation inside physical task ownership: `OwnedTask` owns the cancellation token, nested tasks derive child tokens, sibling shutdown signals every task before awaiting any one, and `watch` channels carry state only.
- Treat nested `OwnedTask` work as long-lived and failure-supervised: unexpected return or panic cancels the physical parent, while finite work is applied synchronously or modeled explicitly instead of being spawned as a critical child.
- The Pylon process supervisor observes registration, stats, metrics, required engine-stats, and direct-tunnel roots; unexpected critical-root exit fails the process, successful auto-mode engine-stats fallback remains nonfatal, and SIGINT/SIGTERM stop top-level siblings together.
- Keep registration-router topology in one awaiting-initial-or-published watch
  generation shared directly by discovery and registration workers. After the
  first publication, registration targets every discovered Stargate.
- Keep each cumulative request-counter identity in one live-or-finalized lifecycle map; a counted live total may mirror membership only to preserve O(1) metrics.
- Keep queue-admission request identity and per-model aggregate state under one locked owner. Queue phases only advance, and valid queue-time overflow must saturate instead of appearing idle or unknown.
- Keep request lifecycle transitions source-owned. Queue projection, lifecycle metrics, and stats publication must derive from one transition event; do not replay observations or reconstruct request identity in metrics.
- Keep input-throughput distributions, threshold decisions, sticky throughput, and runtime queue updates inside `StatsAggregator`; do not relay fallback samples through a parallel task, channel, or per-model map.
- Reverse-tunnel connectivity gates advertisement per Stargate after local startup has completed.
- CLI deployment topology is explicit: direct/Edge pylon owns a reachable QUIC
  listener advertised to Stargate; reverse/Cloud pylon owns outbound reverse
  connections to Stargate. Do not infer topology from unrelated TLS or address
  settings.

## Bringup And Canaries

- Every Pylon must select exactly one input-TPS bootstrap source:
  `--do-calibration` or `--initial-input-tps`. Calibration runs against the
  local upstream and is valid only when the process is the cluster's sole
  Pylon; shared-Pylon clusters use an operator-provided per-Pylon initial TPS.
- Bootstrap every configured model before any Stargate RPC. Calibration model
  sweeps are sequential, while request concurrency within one model's plan is
  allowed. A failure aborts startup without registering.
- Initialize input-throughput distribution, runtime stats, and queue admission
  through the `StatsAggregator` bootstrap path. Later samples update the value
  unless the benchmark-only pin policy is selected.
- Start local stats, metrics, direct tunnel, and ongoing bringup ownership
  before starting registration. Registration never owns or retries bootstrap.
- Active canaries begin only after active advertisement.
- The built-in canary is the deterministic `1+1=` chat request. Completion exactly at `canary_max_generation_threshold` is treated as runaway generation and demotes the model.

## Metrics

- Keep pylon metrics in `src/metrics.rs` using the per-process `PylonMetrics` registry.
- Model gauges should be emitted for request-observation updates and KV-cache poll-only updates.
- Preserve terminal request counters and token histograms for all terminal outcomes.
</coding_guidelines>
