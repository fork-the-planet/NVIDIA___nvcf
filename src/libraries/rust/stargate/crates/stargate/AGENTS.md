<coding_guidelines>
# crates/stargate

Server-side stargate code owns registration state, load-balancer routing, HTTP proxying, QUIC connection use, development-only peer relay, metrics, and tracing.

## Local Invariants

- Keep routing state keyed by `RoutingTargetKey { routing_key, model_id }`.
- `routing_key` comes from `WorkerAuthenticator`; never trust a registration proto field for routing identity.
- Do not parse HTTP proxy request bodies. Treat bodies as opaque bytes and buffer only for replay.
- Keep replay bytes and readiness in one replay-buffer lifecycle owner. Terminal
  overflow, stream failure, and abandonment release retained bytes; completed
  immutable bytes are shared across retries. Do not reintroduce parallel
  completion or overflow flags.
- Direct backends use the exact live registration generation's QUIC
  connection set; `--direct-quic-connections` controls the set size and
  defaults to `1`.
- Keep `--backend-connectivity=direct|reverse` consistent with listener
  ownership: direct rejects reverse-listener settings, while reverse requires
  the Stargate reverse listener.
- Only route to backends with an open QUIC path and successful forwarded `/health` RTT sample.
- Keep HTTP proxy requests local to the serving stargate. Do not forward HTTP proxy requests between peer stargates.
- Keep built-in backend gRPC/reverse-QUIC peer relay behind the explicit, default-off `--enable-dev-peer-forwarding` development flag. Production deployment paths use `stargate-k8s-router` or supported load balancers, never this relay.
- Preserve retry/failover accounting so final request metrics emit once and attempt metrics emit per upstream attempt.
- Keep tracing fields on `proxy_openai_request` useful for backend selection, retry, replay, upstream status, and TTFT debugging.
- Keep Stargate process lifecycle under the shared runtime task owner. One
  cancellation token defines proxy draining and graceful shutdown; critical
  HTTP/gRPC, WatchStargates, reverse-listener, and metrics roots fail the
  process, while dynamic request/connection work remains tracker-owned and
  noncritical.
- Any tracker-owned task that can block must observe a downward-only runtime
  shutdown signal at its blocking boundaries and still run its cleanup path.
- After the first control-plane registration update is admitted, one
  `RegistrationSession` owns the running registration, connection watcher,
  connection config, eager periodic health-check lifecycle, update
  application, and cleanup. Do not reintroduce partial running states or
  external cleanup choreography.
- Overlapping registrations in one `(routing_key, cluster_id)` scope share one
  `RegistrationClusterGeneration`. That generation owns Stargate's local
  cluster lifetime and retires only after its final exact registration is
  removed. It must not own calibration or capacity-floor state.
- One short-held registration registry owns exact-id membership, active
  cluster generations, model-advertisement lifecycle, and registered-target
  advertiser counts. Keep its indexes atomic, keep its guards out of
  `.await`s, and do not reintroduce full-registration membership scans or
  per-registration model locks.
- Registration and routing cleanup must be conditional on exact registration
  and cluster-generation identity. Do not reintroduce id-only cleanup or scans
  that reconstruct cluster lifetime.
- Keep tunnel connection state on the exact `RegistrationGeneration` and make
  retirement downward-only. Do not reintroduce a proxy-global tunnel registry,
  id-keyed tunnel ownership, or an ended-generation reclaim path.
- Represent registration tunnel availability and reverse installation as
  explicit lifecycle variants. The acquired reverse-install capability owns
  both exact-generation commit and cancellation; do not reintroduce an active
  connection option/boolean bag or a separately supplied finish owner.

## Load Balancing

- New algorithms implement `LoadBalancer` in `src/load_balancer/` and are registered in `create_load_balancer()`.
- Pass request-specific inputs through `LoadBalancerRequest`; do not grow trait methods with positional arguments.
- Candidate lookup is scoped by `RoutingTargetKey`.
- Keep stateful load-balancer instances owned by the authoritative `RoutingTargetState`; do not add router-global per-target maps or cleanup callbacks.
- Keep cluster membership, active-backend count, and final retirement in one
  `RoutingTargetState` generation. Do not reintroduce a nested concurrent
  cluster map or mutate target-owned backend membership outside that owner.
- Keep active `RoutedInferenceServerSnapshot` values as the sole retained
  per-backend routing records. Each snapshot retains its exact registration
  generation once alongside exported routing data; derive cluster-scoped
  source values from those snapshots instead of wrapping or mirroring them.
  Construct snapshots from the exact registration, publish them directly, and
  validate exported identity before storage; do not add a shadow publication
  input. Store each immutable publication behind one shared owner and carry
  that exact publication through backend selection and same-backend retries;
  do not deep-clone a complete snapshot on the request hot path.
- Keep one chosen routing decision attached to its exact `RoutedClusterState`
  owner through backend selection and optimistic reservation insertion. Once
  accepted, let `RoutingReservation` own the exact pending reservation's
  cancellation state. Do not reintroduce target/cluster relookups, id-only
  reservation insertion, or owner/map-based reservation release.
- `pulsar` ranking must use effective candidate capacity (`ModelStats.last_mean_input_tps` after cluster aggregation) and keep transient live load in feasibility gates.
- Any algorithm that needs request headers must define missing-header behavior explicitly and return client errors where configured.

## Concurrency

- Use approved `scc` single-shot APIs. Avoid entry-style APIs and `contains_*` followed by dependent mutation.
- In async code, prefer `*_async` operations and never hold bucket-level locks across external `.await`s.
- Keep lock closures short, non-blocking, and side-effect free.
</coding_guidelines>
