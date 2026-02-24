# Local Docker Build Guide

**When to use this:** Only needed if your RunPod region has CUDA 12.8 drivers and no 12.9+ region is available. Pre-built RunPod images (v2.13.0+) use CUDA 12.9 and won't work on 12.8 drivers.

## Why Source Build is Needed

- RTX 5090 (sm_120) is too new — no pre-built vLLM wheel includes it
- Pre-built wheels use PTX JIT fallback, which fails on CUDA 12.8 drivers
- Building from source with CUDA 12.8 toolkit produces native sm_120 binaries

## Build Steps

```bash
# 1. Clone the repo
git clone https://github.com/indra-sapyoung/runpod-gptoss-worker.git
cd runpod-gptoss-worker

# 2. Build the image (~15-20 min on a powerful PC)
docker build -t indrasapyoung/worker-vllm-gptoss:v4-cu128 .

# 3. Login to DockerHub
docker login

# 4. Push to DockerHub
docker push indrasapyoung/worker-vllm-gptoss:v4-cu128
```

## Dockerfile Overview

The Dockerfile (`Dockerfile` in repo root):

| Step | What it does | Time |
|------|-------------|------|
| Base image | `nvidia/cuda:12.8.1-devel-ubuntu24.04` (~4GB) | ~2 min |
| Python | Install Python 3.12 from Ubuntu 24.04 | ~1 min |
| PyTorch | Nightly cu128 (need 2.7+ for MXFP4 support) | ~3 min |
| Build tools | setuptools, cmake, ninja, wheel | ~30 sec |
| vLLM v0.15.1 | Source compilation for sm_86 (A40) + sm_120 (RTX 5090) | ~10-15 min |
| RunPod SDK | runpod, pydantic | ~30 sec |

## Key Settings

| Setting | Value | Why |
|---------|-------|-----|
| `TORCH_CUDA_ARCH_LIST` | `"8.6;12.0"` | A40 (sm_86) + RTX 5090 (sm_120) |
| `MAX_JOBS` | `8` | Parallel compilation (adjust to your CPU cores) |
| PyTorch | Nightly cu128 | Stable cu128 only has 2.6, need 2.7+ for Float8_e8m0fnu |

## Adding More GPU Support

To support additional GPUs, add their compute capability to `TORCH_CUDA_ARCH_LIST` in the Dockerfile:

| GPU | Compute Capability | TORCH_CUDA_ARCH_LIST value |
|-----|-------------------|---------------------------|
| A100 | sm_80 | `8.0` |
| A40 | sm_86 | `8.6` |
| L40, L40S | sm_89 | `8.9` |
| H100 | sm_90 | `9.0` |
| RTX 5090 | sm_120 | `12.0` |

Each architecture adds ~10 min of compile time. Only include GPUs you actually deploy on.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Float8_e8m0fnu is not a member` | PyTorch too old (< 2.7) | Ensure nightly torch is installed |
| OOM during compilation | Not enough RAM | Reduce `MAX_JOBS` (e.g., 4 or 2) |
| Disk space error during triton clone | Full disk | The `--depth 1` clone + sed injection should handle this |
| `libmpi.so.40` missing | Wrong base image | Use `nvidia/cuda`, not NGC PyTorch image |
| PTX unsupported toolchain (at runtime) | CUDA driver < toolkit version | Driver must be >= 12.8 |

## When You DON'T Need This

- RunPod region has CUDA **12.9+** drivers → use RunPod's pre-built `v2.13.0` image
- Running on A40 only (no 5090) → pre-built vLLM wheel works fine
- Running locally → just `pip install vllm` and `vllm serve`
