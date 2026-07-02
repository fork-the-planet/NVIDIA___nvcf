# NvSnap

**Application-transparent GPU checkpoint/restore for Kubernetes.**

NvSnap captures a running GPU inference pod — process state *and* GPU
memory — and restores it on another node with a fully warm engine: no
model reload, no compilation-cache rebuild, no warm-up queries. It works
on **unmodified** inference containers (NVIDIA NIM, vLLM, SGLang,
TensorRT-LLM) — no application changes, no custom images.

The goal is faster cold starts for large GPU workloads. Restoring a warm
checkpoint avoids the model download, weight load, CUDA-graph capture, and
JIT/compile costs that dominate a cold start.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Kubernetes](https://img.shields.io/badge/kubernetes-1.25%2B-326ce5.svg)

---

## How it works

NvSnap picks one of two capture paths automatically, based on the
workload:

- **Single-GPU — full process checkpoint/restore.** CRIU plus NVIDIA
  `cuda-checkpoint` snapshot the whole process (CPU state, open FDs,
  io_uring/epoll, and GPU memory). The restored pod resumes a live,
  already-serving engine.
- **Multi-GPU / large models — cache distribution.** Process-level
  multi-GPU restore is blocked upstream (a `libcudart` per-process
  GPU-state replay limitation). Instead NvSnap captures the warm pod's
  model + cache directory and distributes it to other nodes, so restored
  pods skip the model fetch and first-time compile even though the engine
  starts fresh.

A node-level **agent** (DaemonSet) performs capture/restore using
`/proc` + cgroup inspection — runtime-agnostic, not tied to any container
runtime API. A **server** provides a REST API, web UI, and checkpoint
catalog. An optional **admission webhook** auto-injects restore plumbing
into workload pods.

Deeper design docs:
[architecture](docs/architecture/NVSNAP-ARCHITECTURE.md) ·
[quiescence / io_uring](docs/architecture/QUIESCENCE-ARCHITECTURE.md) ·
[runtime-agnostic discovery](docs/architecture/04-RUNTIME-AGNOSTIC.md) ·
[cuda-checkpoint call flow](docs/architecture/06-CUDA-CHECKPOINT-CALL-FLOW.md) ·
[vLLM multi-process](docs/architecture/VLLM-DEEP-DIVE.md).

---

## Benchmarks

Measured on H100 80GB, GKE A3-Mega. All workloads pass end-to-end
(checkpoint + restore + verified inference). "Speedup" is
`cold_start / restore`.

### Single-GPU (CRIU + cuda-checkpoint)

| Engine | Model | Ckpt size | Cold start | Capture | Restore | Speedup |
|---|---|---|---|---|---|---|
| NIM (TRT-LLM) | Qwen3-32B | 111 GB | 4m 03s | 3m 27s | 1m 40s | 2.4× |
| vLLM | TinyLlama 1.1B | 28 GB | 2m 11s | 1m 32s | 46 s | 2.9× |
| vLLM | Llama-3.1-8B | 76 GB | 1m 50s | 2m 41s | 1m 41s | 1.1× |
| TRT-LLM | TinyLlama 1.1B | 29 GB | 1m 21s | 1m 54s | 49 s | 1.7× |
| SGLang | TinyLlama 1.1B | 57 GB | 1m 01s | 6m 36s | 2m 51s | — |
| SGLang | Llama-3.1-8B | 88 GB | 1m 40s | 6m 52s | 3m 42s | — |

"—" where restore is not faster than a cold start (SGLang's init is fast
enough that CRIU overhead doesn't pay back at these sizes).

### Multi-GPU (rootfs cache distribution — Llama-3.1-70B, TP=4)

| Stage | Value |
|---|---|
| Cold start | 332 s |
| Rootfs capture size | 133 GiB (weights + HF cache + engine cache) |
| Restore (cache-warm) | 180 s — 1.8× vs cold start |

### Cross-node distribution (peer cascade)

Every node that fetches a checkpoint becomes a source for the next, so
aggregate bandwidth scales with peer count instead of bottlenecking on the
origin node.

| Workload | Size | Per receiver |
|---|---|---|
| vllm-small CRIU dump | 30 GiB | 25 s (1.20 GB/s) |
| vllm-70b rootfs | 133 GiB | 415 s (0.32 GB/s, many-files HTTP overhead) |

See [docs/BENCHMARK.md](docs/BENCHMARK.md) for methodology and the full
measured matrix.

---

## Quickstart

Prerequisites: a Kubernetes 1.25+ cluster with GPU nodes, NVIDIA driver
**555+** (required by `cuda-checkpoint`), containerd 1.7+/2.x or CRI-O, and
`kubectl` + `helm` 3.13+.

```bash
# 1. Namespace + image pull secret for the project registry
kubectl create namespace nvsnap-system
kubectl create secret docker-registry nvsnap-pull-secret \
  --namespace=nvsnap-system \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password='<NGC_API_KEY>'

# 2. Install (agent + server + blobstore)
helm install nvsnap deploy/helm/nvsnap --namespace nvsnap-system

# 3. Verify
kubectl -n nvsnap-system rollout status ds/nvsnap-agent
kubectl -n nvsnap-system get svc nvsnap-server
```

`scripts/install-nvsnap.sh` wraps the above (prereq checks, pull secret,
optional webhook). Full guide, options, and troubleshooting:
[docs/INSTALL.md](docs/INSTALL.md). Pull-secret details:
[docs/PULL-SECRET-SETUP.md](docs/PULL-SECRET-SETUP.md).

### Try it end-to-end

```bash
./scripts/test-e2e.sh vllm-small   # TinyLlama 1.1B on vLLM (single-GPU)
./scripts/test-e2e.sh vllm-8b      # Llama-3.1-8B on vLLM
./scripts/test-e2e.sh vllm-70b     # Llama-3.1-70B TP=4 (rootfs path)
```

---

## Architecture

```
┌──────────────────┐     ┌───────────────────────────────────────┐
│  nvsnap-server   │     │            GPU Node                   │
│  REST API + UI   │◄───►│  nvsnap-agent (DaemonSet)             │
│  catalog (SQLite)│     │   ├─ process discovery (/proc+cgroup) │
│  metrics + audit │     │   ├─ CRIU orchestration (go-criu RPC) │
└──────────────────┘     │   ├─ cuda-checkpoint integration      │
                         │   ├─ rootfs capture watcher           │
                         │   └─ peer-cascade HTTP server         │
                         │                                       │
                         │  GPU Pod (unmodified image)           │
                         │   ├─ NIM / vLLM / SGLang / TRT-LLM    │
                         │   └─ libnvsnap_intercept.so (PRELOAD) │
                         └───────────────────────────────────────┘
                                        │
              L1 same-node ─► L2 shared PVC ─► L3 object store
```

Checkpoints are fetched through a tiered path (same-node → shared
filesystem → peer cascade → object store); cluster config decides which
tiers are present. See
[docs/TRANSPORT-ARCHITECTURE.md](docs/TRANSPORT-ARCHITECTURE.md).

**Why the moving parts exist** (details in
[docs/architecture/](docs/architecture/)):

- **go-criu RPC, not CLI CRIU** — engines like vLLM run 900+ threads; CLI
  CRIU's per-thread ptrace detach takes minutes and often hangs, while RPC
  is seconds regardless of thread count.
- **`libnvsnap_intercept.so` (LD_PRELOAD)** — `io_uring` (via uvloop/libuv)
  and `libzmq` epoll don't survive CRIU restore cleanly. Rather than fork
  every engine, the library reinitializes them on a restore marker.
- **Forked CRIU** — 26 patches for Kubernetes container support, io_uring,
  and the CUDA plugin. See [docs/THIRD-PARTY-FORKS.md](docs/THIRD-PARTY-FORKS.md).

---

## Supported platforms

| Component | Supported | Notes |
|---|---|---|
| Linux kernel | 5.15.x | Kernel 6.x in progress (io_uring restore + cuda-checkpoint interaction). |
| NVIDIA driver | R555+ | Required by `cuda-checkpoint`. Installed via the NVIDIA GPU Operator. |
| Container runtime | containerd 1.7+/2.x, CRI-O | Discovery is at the Linux process level, not via runtime APIs. |
| Kubernetes | 1.25+ | |
| Engines tested | NIM, vLLM, SGLang, TensorRT-LLM | Unmodified stock images. |

Checkpointing operates at the Linux process level, not at the engine or
framework level — so any GPU process is a candidate. The engines above are
the ones validated end-to-end; others should work without modification.

---

## Security requirements

CRIU checkpoint/restore needs elevated privileges, configured per-pod (not
cluster-wide):

| Requirement | Why |
|---|---|
| `privileged: true` | CRIU needs `/proc/<pid>`, ptrace, device files |
| `hostPID: true` (restore) | Host PID namespace for CRIU mount ops |
| Host path volumes | Checkpoint data on node-local storage |
| NVIDIA device access | `/dev/nvidia*` for GPU checkpoint |

The agent DaemonSet runs privileged with `hostPID`, `hostNetwork`, and
host-path mounts for `/proc`, the container runtime state dir, and
checkpoint storage.

---

## REST API

`nvsnap-server` exposes a REST API and web UI on port 8080. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/checkpoints` | Checkpoint a running GPU pod |
| GET | `/api/v1/checkpoints` | List checkpoints (cursor pagination, filters) |
| GET | `/api/v1/checkpoints/{id}` | Checkpoint detail |
| DELETE | `/api/v1/checkpoints/{id}` | Delete checkpoint |
| POST | `/api/v1/restores` | Restore a checkpoint |
| GET/POST | `/api/v1/retention-policies` | Retention policies (CRUD) |
| GET | `/api/v1/audit` | Audit trail |
| GET | `/metrics` | Prometheus metrics |

The full endpoint list (including agent-side cascade endpoints) is in
[`internal/server/openapi.yaml`](internal/server/openapi.yaml).

---

## Project layout

```
cmd/                    binary entry points (agent, server, restore-entrypoint, gpu-restore, CLI)
internal/               agent, server, webhook, CRIU, checkpointstore (Go)
lib/nvsnap_intercept/   LD_PRELOAD interception library (C)
deploy/helm/nvsnap/     Helm chart
deploy/k8s/             manifests + sample workloads
docker/                 Dockerfiles
scripts/                build / test / deploy / benchmark automation
docs/                   architecture, design, reference
terraform/              example GKE cluster provisioning
```

---

## Documentation

- [docs/](docs/README.md) — documentation index (operator + developer)
- [docs/dev/getting-started.md](docs/dev/getting-started.md) — build → capture → validate → merge
- [docs/INSTALL.md](docs/INSTALL.md) — full install guide + troubleshooting
- [docs/architecture/NVSNAP-ARCHITECTURE.md](docs/architecture/NVSNAP-ARCHITECTURE.md) — system architecture
- [docs/BENCHMARK.md](docs/BENCHMARK.md) — benchmark methodology + results
- [docs/THIRD-PARTY-FORKS.md](docs/THIRD-PARTY-FORKS.md) — forked dependencies + patch policy
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer setup + build-from-source

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer setup, building from
source, and the forked-dependency layout. Issues and merge requests are
welcome.

## License

Apache 2.0 — see [LICENSE](LICENSE). Forked third-party components retain
their original licenses; see [NOTICE](NOTICE).
