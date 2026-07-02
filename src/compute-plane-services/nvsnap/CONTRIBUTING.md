# Contributing to NvSnap

Thanks for your interest in NvSnap — a GPU checkpoint/restore system for Kubernetes.

## Project location

The canonical NvSnap repo lives at `github.com/NVIDIA/nvcf/src/compute-plane-services/nvsnap`. Clone with:

```bash
git clone git@github.com/NVIDIA:nvcf/nvcf-cache/nvsnap.git
```

(Pre-OSS the project is hosted on internal GitLab. A public-GitHub mirror will be set up at OSS release time.)

## External repositories

NvSnap depends on several forks of upstream projects, each in its own repository:

| Component | Purpose | Branch |
|---|---|---|
| Forked CRIU | Checkpoint/restore patches (io_uring, NCCL, CUDA plugin, chardev) | `criu-dev` |
| Forked uvloop | CRIU-compatible event loop | `checkpoint-restore-v1` |
| Forked libuv | io_uring backend reinit after restore | `main` |
| Forked libzmq | epoll reinitialization after restore | `main` |
| Forked go-criu | RPC binding fixes | `main` |

Each is maintained as a separate repository with NvSnap-specific patches; see `docs/THIRD-PARTY-FORKS.md` for the fork-maintenance policy. All forks are required to build NvSnap from source. Pre-built images are the fast path for most users; only contributors hacking on the C library or CRIU itself need to build the forks from source.

## Developer setup

The conventional layout puts each fork as a sibling of this repo:

```
~/work/
├── nvsnap/                     # this repository
├── criu/                       # NVIDIA CRIU fork
├── uvloop/                     # uvloop fork
├── libuv/                      # libuv fork
├── libzmq/                     # libzmq fork
└── go-criu/                    # go-criu fork
```

With this layout, `scripts/versions.sh` auto-discovers the CRIU fork at `../criu` and you don't need to set any environment variables.

If your forks are elsewhere, override:

```bash
export NVSNAP_CRIU_SRC=/path/to/your/criu
```

## Build

```bash
# Pulls the prebuilt agent base image — fast path.
make build

# Rebuild the agent base image from scratch (needed when CRIU source changes).
./scripts/build-agent.sh base
./scripts/build-agent.sh push-base

# Rebuild the agent app image (Go code, intercept library).
./scripts/build-agent.sh app
./scripts/build-agent.sh push-app

# Build a dependency image (one of: libzmq, uvloop, pyzmq, libuv).
./scripts/build-deps.sh <name>
```

See [`scripts/versions.sh`](scripts/versions.sh) for the canonical version + registry config.

## Container registry

Built images go to `nvcr.io/0651155215864979/ncp-dev/<component>:vX.Y.Z`. To push, you need an NGC API key:

```bash
docker login nvcr.io
  Username: $oauthtoken
  Password: <your NGC API key>
```

## Code style

- **Go**: `gofmt`, idiomatic Go. Errors wrapped with `fmt.Errorf("...: %w", err)`. Use the structured logger.
- **C** (`lib/nvsnap_intercept/`): two-space indent, snake_case for functions, ALLCAPS for macros. Match the existing surrounding style.
- **Bash**: `set -euo pipefail` at the top of new scripts.
- **YAML manifests**: kubectl-applyable from `deploy/k8s/`. Run `./scripts/sync-versions.sh` after a version bump.

## Pull requests

1. Branch from `main`.
2. Per-component image tag bumps follow `CLAUDE.md` rule 19 (never reuse a tag on rebuild — `:v0.0.1` → `:v0.0.2`).
3. Run `./scripts/test-e2e.sh vllm-small` against a real GPU cluster before merging anything touching `lib/nvsnap_intercept/`, `cmd/restore-entrypoint/`, the patched libraries, or K8s manifests.
4. Squash commits per logical change. Reference the GitHub/GitLab issue number if one exists.

## Reporting bugs / requesting features

File an issue. Include:
- NvSnap agent image tag (`kubectl -n nvsnap-system get ds nvsnap-agent -o jsonpath='{.spec.template.spec.containers[0].image}'`)
- Workload (vLLM, SGLang, TRT-LLM, etc.) + model + GPU count
- Full agent log: `kubectl -n nvsnap-system logs ds/nvsnap-agent`
- Full restore-entrypoint log if relevant: `kubectl logs <pod> -c restore-entrypoint`
- CRIU log if relevant: at `/var/log/criu/<checkpoint-id>/dump.log` on the GPU node

## License

Apache 2.0 (see `LICENSE`).
