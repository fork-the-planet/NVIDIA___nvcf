# Agent Architecture Reference

> Type: Reference. Source: `crates/stargate`, `crates/pylon-lib`, Kubernetes manifests, and current behavior tests.

Read this for routing, proxying, registration, Kubernetes, pylon, and
observability changes.

## System Shape

```text
gateway -> stargate proxy -> QUIC tunnel -> pylon -> local inference server
backend/pylon -> stargate gRPC registration -> local routing state
```

`--backend-connectivity=direct|reverse` selects who establishes the QUIC
connection. Direct is the Edge topology: pylon advertises a reachable
`quic://` listener and Stargate dials it. Reverse is the Cloud topology: pylon
dials the reverse target returned by Stargate. The transport protocol remains
an independent `raw-quic|http3|webtransport` choice.

Stargate is the control plane and routing entrypoint. Pylon is the backend
sidecar.

## Stargate Process Contract

Stargate owns one failure-supervised runtime tree:

- process roots use the workspace `stargate-runtime` critical-task owner, which
  gives Stargate and the Kubernetes router the same cancellation, panic, and
  named-failure semantics;
- one runtime cancellation token defines proxy draining and graceful shutdown;
- control-plane gRPC, model-discovery gRPC, HTTP proxy, WatchStargates
  publication, optional reverse-tunnel listening, and metrics serving are
  critical process roots;
- unexpected critical-root return, error, or panic stops the whole runtime and
  fails the process so the pod or process supervisor restarts one coherent
  Stargate;
- dynamic registration-stream, request, and reverse-connection work is tracked
  for bounded shutdown but does not fail the process when one task completes;
  long-lived tracked work observes a downward-only shutdown signal at blocking
  boundaries and runs cleanup before completing;
- dropping the runtime handle begins shutdown, and SIGINT/SIGTERM use the same
  bounded drain path.

## Routing Identity

- Routing key type: `RoutingTargetKey { routing_key: Option<String>, model_id: String }`.
- `routing_key` comes from `WorkerAuthenticator`, not the registration proto.
- HTTP callers may provide trusted `x-routing-key`; omitted means `None`.
- `inference_server_id` identifies one live backend registration.
- `cluster_id` groups backend registrations that share one hardware/cache
  domain.

## Registration Lifetime

Before the first update is admitted, the registration stream processor owns
only stream-level context. Admission constructs one complete
`RegistrationSession` that owns the running registration, connection config,
exact-generation tunnel lifecycle, and one immediately started periodic
health-check task. The session
validates and applies the first and subsequent updates through the same path,
then shuts down health work and removes the registration when the stream ends,
times out, fails, loses its response channel, or observes runtime shutdown.
The health handle owns its exact registration and proxy probe inputs; one
cancellation-safe task owner aborts on dropped cleanup and joins cooperatively
during normal session close.

One short-held local registration registry owns exact-id records, active
cluster generations, each registration's `Stable`/`Applying` advertised-model
lifecycle, and registered-target advertiser counts. Membership transitions
update those indexes atomically without holding a guard across asynchronous
upstream, routing, or network work. Cluster and registered-target lookups
are average `O(1)` rather than full-registration scans.

Overlapping registrations in the same `(routing_key, cluster_id)` scope share
one `RegistrationClusterGeneration`; the generation retires permanently at the
true zero-registration boundary. Cleanup consumes the exact running
registration, so a stale session cannot remove a replacement that reused its
`inference_server_id`. On a proxy miss, the registered-target index
distinguishes a registered but unavailable target from an unknown target
without scanning registrations.

Each admitted session also owns one exact `RegistrationGeneration` whose
direct or reverse connection set and reverse-install state live directly on
that generation. Direct and reverse connection installation, forwarded health
checks, routed snapshots, proxy stream opens, and reverse-connection cleanup
all carry that exact generation. A stale route, delayed handshake, or delayed
cleanup therefore cannot borrow or remove a same-id replacement tunnel.
Connection availability and reverse installation are explicit lifecycle
variants, and an acquired reverse-install capability owns both commit and
cancellation on that exact generation.
Connection state retires irreversibly when the session tunnel ends, so stale
owners cannot reclaim an ended generation. Its short-held lock protects only
in-memory connection-handle transitions and is never held across network
awaits; there is no proxy-global tunnel registry.

## Proxy Contract

The canonical frontend/proxy contract is [api-gateway-contract.md](api-gateway-contract.md).
Keep endpoint lists, required headers, error codes, and retry rules there.

Architecture invariants:

- Stargate never parses proxy request bodies.
- Request bodies are buffered only for retry replay.
- One replay-buffer lifecycle owns retained bytes and readiness. Overflow,
  stream failure, or abandonment releases non-replayable bytes; completed
  immutable bytes are shared across retries.
- Proxy requests use an already-established tunnel to a selected pylon.
- Stargate strips caller-supplied internal queue headers.
- Pylon strips the private retry-control header family before forwarding
  upstream and before returning upstream headers through the tunnel. Pylon
  alone translates `x-stargate-upstream-retryable` into the
  `x-stargate-retryable`, reason, and retry-delay response fields consumed by
  Stargate; callers and inference servers cannot inject that control exchange.

## Tunnel Transports

`--tunnel-protocol=raw-quic|http3|webtransport` must match on Stargate and pylon.

- `raw-quic`: raw QUIC bidirectional stream with Stargate framing.
- `http3`: HTTP/3 request stream.
- `webtransport`: HTTP/3 extended CONNECT session plus WebTransport streams.

Direct backends advertise `quic://...`. Reverse-tunnel backends advertise their
upstream HTTP URL and set `reverse_tunnel=true`.

Read [tunnel-transports.md](tunnel-transports.md) before changing transport
selection or load-balancer requirements.

## Routability

A backend is routable only after:

- registration is active for the model
- the QUIC path exists
- a forwarded `/health` RTT sample succeeds

Backend RTT comes from the registration-scoped forwarded `/health` loop, not
QUIC transport stats.

Stargates do not share routing state. HTTP proxy and `ListModels` requests use
only local state.

## Pylon Contract

Pylon:

- uses the workspace `stargate-runtime` abort-on-drop task owner for its
  registration, tunnel, stats, metrics, bringup, and canary task trees;
- completes input-TPS bootstrap and local startup before its first Stargate
  RPC, then keeps recursive discovery live and registers with every discovered
  Stargate
- validates tunneled request headers and endpoint body shape
- forwards to the local HTTP upstream
- converts local upstream retry hints into Stargate retry metadata
- observes request lifecycle and runtime stats
- optionally runs one local startup calibration plan per model and runs active
  canaries after startup
- opens reverse tunnels when configured
- treats registration, stats collection, metrics serving, required engine
  stats, and the direct tunnel accept loop as critical process roots
- exits when a critical root or nested long-lived task fails unexpectedly, so
  the process supervisor restarts one coherent sidecar
- handles SIGINT and SIGTERM through one graceful sibling-shutdown path

Streaming chat and Responses requests must set `"stream": true`.
Embeddings requests must be valid JSON and do not need `stream`.
Local upstreams can mark retryable admission failures with
`x-stargate-upstream-retryable: true`; pylon translates that to internal
Stargate retry headers.

Successful auto-mode engine-stats completion after publishing the OpenAI
fallback control update is intentionally nonfatal. Recoverable registration,
reverse-tunnel, and stats-stream connection failures remain inside their retry
loops.

Request-observer terminal transitions are invariants. Terminalizing an already
terminal request is a bug.

## Input-TPS Bootstrap

Every Pylon must choose exactly one bootstrap source before startup:

- `--do-calibration` runs a health check and one local calibration sweep per
  configured model. Use it only when that process is the cluster's sole Pylon.
- `--initial-input-tps <TPS>` installs a finite positive operator-selected
  per-Pylon value. Shared-hardware clusters must use this source.

Bootstrap initializes the input-throughput estimator, runtime publication, and
queue admission before Pylon starts `WatchStargates`. Model calibration sweeps
are sequential, though requests within one model's plan may run concurrently.
Any bootstrap failure terminates startup without a Stargate RPC. Later runtime
samples update an unpinned bootstrap normally.

Stargate has no calibration protocol or state. It receives capacity only in
ordinary registration `ModelStats`, and cluster input capacity is the sum of
active backend reports.

## Load Balancing

Built-ins:

- `power-of-two`
- `groq-multiregion`
- `round-robin`
- `random`
- `pulsar`
- `pulsar-multiregion`

`LoadBalancerRequest` carries request inputs. Do not grow trait methods with
positional arguments.

All algorithms evaluate cluster candidates. Backend selection inside a chosen
cluster is a state-owned round robin. PULSAR ranks by stable capacity and keeps
transient live load in feasibility gates.
Each cluster candidate's representative RTT is the unweighted arithmetic mean
of the latest forwarded `/health` RTT publication from every current active
backend in that cluster. Publication insertion, heartbeat replacement, and
removal refresh that shared snapshot value at nanosecond precision, truncating
fractional nanoseconds toward zero.
Groq multiregion shuffles each sampled prefix; `min_by` retains the first
equal-score candidate, so that shuffled order is the intentional tie-break.

Stateful cluster-selection instances are owned by the authoritative
`RoutingTargetState`. `LoadBalancerRouter` owns immutable configuration and
algorithm resolution, not target lifecycle. A routed target snapshot retains
the exact target generation while candidate selection is in flight; removing
and recreating a target therefore reclaims and resets its counters and caches.
The same snapshot captures the exact routed cluster owner beside each
load-balancer candidate. Once a candidate is chosen, backend selection and
optimistic reservation operate directly on that owner rather than looking up
the mutable current target and cluster again.

Each `RoutingTargetState` also owns one active-or-retired membership generation.
Cluster owners and the active-backend count change together under that
generation. Final empty-target retirement occurs while the top-level target-map
entry is exclusively owned, so a stale registration update must retry against
the replacement target instead of publishing into detached state.

Each routed cluster additionally retains the exact
`RegistrationClusterGeneration`, and each retained
`RoutedInferenceServerSnapshot` owns its exact `RegistrationGeneration` once
alongside exported routing data. Target membership, cluster membership,
cleanup, reservation, selection, and proxying all derive from that one backend
record. The snapshot constructor derives exported identity from the exact
registration, routing lifecycle creates one shared immutable publication, and
cluster insertion validates the identity boundary before storage. That exact
publication owner is retained through retired-target retry, cluster storage,
selection, and same-backend retry; a later heartbeat replaces the stored
publication while an in-flight request retains the snapshot it selected.
Public candidate inspection materializes owned values, but request routing
does not deep-clone complete backend snapshots. Fresh active cluster work may
replace retired routed state; retired cleanup cannot mutate a current active
cluster, and same-ID cleanup removes only its exact registered state.
Each pending optimistic reservation owns an atomic active state shared with its
one-shot `RoutingReservation` cancellation handle. Queue-mismatch release marks
only that exact reservation inactive without acquiring the cluster-generation
lock; routing snapshots and heartbeats compact inactive reservations while
performing their existing linear reconciliation.

## Kubernetes

- `stargate`: backend-facing gRPC/QUIC service.
- `stargate-headless`: peer discovery and pod identity.
- `stargate-model-discovery`: frontend `ListModels`.
- `stargate-proxy`: frontend OpenAI-compatible HTTP proxy.
- `stargate-k8s-router`: optional backend-facing router for `raw-quic` or
  `webtransport`; one transport mode is selected per UDP listener.

Gateway traffic uses only `stargate-model-discovery` and `stargate-proxy`.
Backend traffic always uses `WatchStargates` and registration. Edge/direct
deployments then have Stargate connect to each pylon's advertised pod URL;
Cloud/reverse deployments have pylon connect to Stargate's reverse listener.

Raw Stargate pod IPs are not a pylon discovery contract. Use advertised
per-pod hostnames and headless DNS for Stargate discovery and identity. In an
Edge/direct deployment, a pylon may advertise its own live pod IP as the
registration-scoped tunnel URL because that URL retires with the exact
registration generation.

### Development-Only Built-In Peer Relay

The built-in relay for backend `WatchStargates`, registration, and reverse-QUIC
traffic is disabled by default. It is enabled only by the CLI-only
`--enable-dev-peer-forwarding` flag, which requires both `--pod-name` and
`--pod-namespace` with DNS discovery enabled and emits a structured startup
warning. It must not run in production.

Headless DNS and pod identity alone never enable the relay. Normal production
backend traffic uses `stargate-k8s-router` or a supported load-balancer path.
HTTP proxy and `ListModels` stay local in every configuration.

`ListModels` is local-only and reads the selected Stargate's current routable
target generations at call time. It remains a hint rather than a reservation:
routing can change after the response.

Base mock backend manifests run pylon as a sidecar in each mock-dynamo
Deployment. Pylon forwards to the colocated inference container on loopback,
keeps the pod labeled `role=inference-engine`, and adds `pylon-sidecar=true`
for pylon metrics scraping.

The `kustomize/overlays/edge` example is router-free. Its backend-facing
`stargate` Service selects Stargate pods directly, Stargates publish concrete
headless-Service pod hostnames to pylons, and pylons bind their direct QUIC
listeners to pod IPs. The cloud-oriented base and local overlays retain reverse
listeners and `stargate-k8s-router`.

The active-development GKE overlay keeps the base `stargate` ClusterIP Service
for in-cluster backend traffic and also exposes split internal L4
LoadBalancer Services: `stargate-grpc-lb` for TCP `443` registration/watch
traffic and `stargate-quic-lb` for UDP `8080` Raw QUIC reverse tunnels. The
split is required because GKE internal LoadBalancer Services cannot mix TCP and
UDP ports. Both Services use the Terraform-managed shared internal VIP
`ip-us-central1-stargate-backend` (`10.69.170.115`) while LB DNS names are not
available.

Cross-cluster backend overlays seed pylon `--stargate-address` with the
backend-facing gRPC endpoint. If Stargate sets `--grpc-pylon-dial-addr`,
`WatchStargates` tells pylon to dial that gRPC endpoint while keeping each
`advertise_addr` as the per-pod HTTP/2 authority/SNI routing identity. If
Stargate sets `--reverse-tunnel-pylon-dial-addr`, ACKs tell pylon to dial that
QUIC endpoint while keeping `reverse_tunnel_target` as the per-pod QUIC
SNI/routing identity.

The base manifests currently leave NetworkPolicy enforcement to overlays or
cluster policy.

The local overlay mirrors the split backend-facing LB shape with ClusterIP
Services on `443` and `8080` that target router pod ports `50071` and `50072`.

## Observability

- Stargate metrics: `--metrics-port`, default `9090`.
- Pylon metrics: default `9089`.
- Stargate's HTTP listener has a local `GET /v1/models` mirror of `ListModels`
  and trusted-operator `GET /debug/state` inspection. The latter exposes a
  small, safe listener/tunnel configuration and the local active-model set;
  it does not mirror routing topology or serialize credential-bearing fields.
- Router metrics: router health listener.
- OTel export is opt-in with `--otel-endpoint`.
- Main proxy span: `proxy_openai_request`.
- Use `x-request-id` as the request correlation id.
- Pylon-generated calibration and canary requests set `x-request-id` as
  `calibration-<pylon-uuid>-<counter>` or `canary-<pylon-uuid>-<counter>`.
- Kubernetes base manifests expose VictoriaMetrics `VMPodScrape` resources for
  Stargate, `stargate-k8s-router`, and pods labeled `pylon-sidecar=true`.

## Important Files

- `crates/stargate/src/http_proxy.rs`
- `crates/stargate/src/runtime.rs` and `runtime/task_group.rs`: owned critical
  roots, process failure supervision, draining, and graceful shutdown.
- `crates/stargate/src/load_balancer/`
- `crates/stargate/src/routing_state/`
  - `mod.rs`: `StargateState` facade and cross-subsystem coordination.
  - `keys.rs`: routing identities, delivery targets, and registration identity.
  - `registration.rs`: authoritative registration membership registry, exact
    running registrations and their tunnel connection state,
    registered-target presence, and registration-cluster generation
    ownership.
  - `clusters.rs`: target lifecycle, cluster state, active models, and metrics.
  - `snapshots.rs`: routable backend and cluster snapshot types plus exact
    selected-cluster ownership.
  - `reservations.rs`: exact-owner routing reservation tokens, accounting, and
    queue estimates.
- `crates/stargate/src/metrics.rs`
- `crates/stargate/src/lib.rs` (`telemetry` module)
- `crates/pylon-lib/src/`
- `crates/protocol/`
- `crates/proto/`
