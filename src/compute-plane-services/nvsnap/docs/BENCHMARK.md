# NVSNAP Checkpoint/Restore Benchmarks

## Environment

- **Cluster**: example-gpu-cluster (GKE)
- **Kernel**: 5.15.0-1081-gke
- **GPUs**: NVIDIA H100 80GB HBM3 (8 per node, 3 nodes)
- **Node storage**: 5.9TB `/dev/md0` at `/var/lib/containerd`
- **Checkpoint path**: `/var/lib/containerd/nvsnap-checkpoints`
- **Container runtime**: containerd 1.7.26
- **Agent**: v0.9.46 (build at time of measurement; see `scripts/versions.sh` for the current pinned version)
- **CRIU**: forked, with io_uring C/R, CUDA plugin, binfmt_misc fixes
- **Patched libs**: uvloop v0.22.1-gpucr16, libzmq v4.3.6-criu-epoll-v12, pyzmq v27.2.0-gpucr3, libuv v1.48.0-criu-v3
- **vLLM**: v0.11.2
- **SGLang**: latest (lmsysorg/sglang:latest)

## Summary (2026-02-25)

| Workload | Model | GPUs | Cold Start | Checkpoint | Ckpt Size | Restore | Speedup | Result |
|----------|-------|------|-----------|------------|-----------|---------|---------|--------|
| vLLM | TinyLlama 1.1B | 1x H100 | 2m 30s | 1m 29s | 28 GB | **1m 02s** | 2.4x | PASS |
| SGLang | TinyLlama 1.1B | 1x H100 | 1m 00s | 5m 22s | 39 GB | **2m 55s** | — | PASS |
| vLLM | Llama-3.1-8B | 1x H100 | 1m 50s | 2m 41s | 76 GB | **1m 41s** | 1.1x | PASS |
| SGLang | Llama-3.1-8B | 1x H100 | 1m 40s | 6m 52s | 88 GB | **3m 42s** | — | PASS |

- **Speedup** = Cold Start / Restore (restore includes CRIU process restore + GPU memory restore)
- SGLang cold starts are faster (smaller base image, no torch.compile), so restore speedup is less pronounced
- SGLang checkpoints are larger due to higher default GPU memory allocation
- All tests run sequentially on a clean cluster with fresh checkpoints

## Detailed Results

### vLLM + TinyLlama 1.1B (single GPU)

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **GPUs**: 1x H100
- **max_model_len**: 2048, **gpu_memory_utilization**: 0.3

| Step | Duration |
|------|----------|
| Cold start (pod ready) | 2m 30s |
| /v1/models ready | 0m 01s |
| Pre-checkpoint inference | 0m 01s |
| **Checkpoint** | **1m 29s** |
| Checkpoint size | 28 GB |
| **Restore (pod ready)** | **1m 02s** |
| Post-restore /v1/models | 0m 01s |
| Post-restore inference | 0m 01s |
| **Total E2E** | **5m 18s** |

### SGLang + TinyLlama 1.1B (single GPU)

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **GPUs**: 1x H100
- **Image**: lmsysorg/sglang:latest
- **mem_fraction_static**: 0.3

| Step | Duration |
|------|----------|
| Cold start (pod ready) | 1m 00s |
| /v1/models ready | 0m 01s |
| Pre-checkpoint inference | 0m 01s |
| **Checkpoint** | **5m 22s** |
| Checkpoint size | 39 GB |
| **Restore (pod ready)** | **2m 55s** |
| Post-restore /v1/models | 0m 01s |
| Post-restore inference | 0m 01s |
| **Total E2E** | **9m 35s** |

### vLLM + Llama-3.1-8B-Instruct (single GPU)

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **GPUs**: 1x H100
- **max_model_len**: 4096, **gpu_memory_utilization**: 0.9

| Step | Duration |
|------|----------|
| Cold start (pod ready) | 1m 50s |
| /v1/models ready | 0m 01s |
| Pre-checkpoint inference | 0m 01s |
| **Checkpoint** | **2m 41s** |
| Checkpoint size | 76 GB |
| **Restore (pod ready)** | **1m 41s** |
| Post-restore /v1/models | 0m 01s |
| Post-restore inference | 0m 01s |
| **Total E2E** | **6m 31s** |

### SGLang + Llama-3.1-8B-Instruct (single GPU)

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **GPUs**: 1x H100
- **Image**: lmsysorg/sglang:latest
- **mem_fraction_static**: 0.85

| Step | Duration |
|------|----------|
| Cold start (pod ready) | 1m 40s |
| /v1/models ready | 0m 08s |
| Pre-checkpoint inference | 0m 01s |
| **Checkpoint** | **6m 52s** |
| Checkpoint size | 88 GB |
| **Restore (pod ready)** | **3m 42s** |
| Post-restore /v1/models | 0m 02s |
| Post-restore inference | 0m 01s |
| **Total E2E** | **12m 42s** |

## Multi-GPU Results

### Llama-3.1-70B-Instruct (TP=4, 4x H100)

Multi-GPU uses the rootfs cache-distribution path (process-level multi-GPU
restore is blocked upstream). Measured: cold start 332 s; rootfs capture
133 GiB; cache-warm restore 180 s (~1.8× vs cold). See the
[README benchmarks](../README.md#benchmarks) and
[PDF-BENCH-RESULTS.md](PDF-BENCH-RESULTS.md) for the current matrix.
