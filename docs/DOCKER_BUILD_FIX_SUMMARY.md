# Docker Build Fix Summary

**Date:** 2026-02-12 | **Target:** vLLM v0.15.1 + CUDA 12.8 + RTX 5090 (sm_120)

## Issues & Fixes

| # | Error | Root Cause | Fix | Commit |
|---|-------|-----------|-----|--------|
| 1 | `setuptools` can't parse license field | Base image setuptools 70.3 doesn't support PEP 639 | `pip install "setuptools>=75.0" "packaging>=24.2"` | `25dfb9d` |
| 2 | Disk full downloading PyTorch | `requirements/build.txt` re-downloads torch (~5GB) | Use `--no-build-isolation` to reuse base image's torch | `25dfb9d` |
| 3 | `No space left on device` cloning triton | cmake FetchContent clones full triton repo (460MB+) | Pre-clone with `--depth 1`, inject path via `sed` into cmake file | `f2afc90` |
| 4 | `libmpi.so.40: cannot open shared object` | Cleanup deleted `/opt/hpcx/` which PyTorch links against | Remove `/opt/hpcx/` from cleanup list | `596925a` |
| 5 | `"Float8_e8m0fnu" is not a member` | PyTorch 2.6 lacks Blackwell FP8 types needed by vLLM v0.15.1 | `pip install torch --index-url .../whl/cu128` before build | `9c17f82` |

## Wrong Turns (Reverted)

| Commit | What was done | Why it was wrong |
|--------|--------------|-----------------|
| `20c99f8` | Changed arch to `9.0+PTX` | Causes 30-60s JIT delay on cold start |
| `e21538c` | Changed arch to `10.0` (sm_100) | sm_100 = datacenter Blackwell, won't run on RTX 5090 (sm_120) |

## CUDA Architecture Reference

| TORCH_CUDA_ARCH_LIST | sm_ | GPU | Notes |
|---------------------|-----|-----|-------|
| 8.0 | sm_80 | A100 | Ampere datacenter |
| 8.6 | sm_86 | A40 | Ampere workstation |
| 8.9 | sm_89 | L40, L40S | Ada Lovelace |
| 9.0 | sm_90 | H100 | Hopper |
| **12.0** | **sm_120** | **RTX 5090** | **Blackwell consumer** |
| 10.0 | sm_100 | B100/B200 | Blackwell datacenter (NOT RTX 5090) |

## Build Requirements

| Component | Required Version | Base Image Has | Action |
|-----------|-----------------|---------------|--------|
| CUDA toolkit | 12.8+ (for sm_120) | 12.8.1 | None |
| PyTorch | 2.7+ (for Float8_e8m0fnu) | 2.6 | Upgrade via cu128 wheel |
| setuptools | 75.0+ (for PEP 639) | 70.3 | Upgrade |
| Disk space | ~15GB free for build | ~14GB (GitHub Actions) | Cleanup + shallow triton |
