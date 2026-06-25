# Tokio Async Rust Audit

> Type: Historical. Current async rules live in `AGENTS.md`, scoped `AGENTS.md`, and `.memory/lessons.md`.

Date: 2026-05-19

Static audit of task ownership, shutdown, cancellation, blocking work,
channels, lock lifetimes, and async tests.

## Summary

No production deadlock or unbounded load-bearing queue stood out. The main risk
was task ownership: several background tasks were spawned outside their owning
runtime tracker or were stopped by `abort()` without cooperative shutdown.

The implementation branch that followed addressed the high/medium findings:

- track reverse listener tasks under Stargate runtime shutdown
- add cooperative pylon registration shutdown
- retain/finalize request-body sender tasks
- add cancellable health-check shutdown
- replace sleep-based propagation waits with observable polling helpers

## Findings

| Severity | Finding | Current expectation |
| --- | --- | --- |
| High | Reverse tunnel listener tasks must be owned by runtime shutdown. | Track listener, accepted connection, and cleanup tasks. |
| Medium | Pylon registration shutdown should be joinable. | Prefer `TaskTracker` plus cancellation over raw aborts. |
| Medium | Per-request body sender tasks must not detach silently. | Retain, await, or abort them through the response body lifecycle. |
| Low | Retry/backoff loops should be cancellation-aware. | Select on stop/cancel tokens. |
| Low | Tests should not sleep for propagation. | Poll observable state or add deterministic seams. |

## Positive Patterns

- Stargate runtime uses `TaskTracker` and `CancellationToken`.
- Routing state uses `scc` APIs with short closures.
- Load-bearing channels are bounded.
- The Kubernetes router has clean task ownership.
- Pylon tunnel handles cancel on drop as a defensive fallback.

## Checks For Async Changes

```bash
cargo test -p stargate reverse_tunnel
cargo test -p stargate quic_forwarding
cargo test -p pylon-lib
cargo test -p stargate-k8s-router
```
