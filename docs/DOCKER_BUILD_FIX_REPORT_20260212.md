# RunPod Docker Build Fix Report

**Date:** 2026-02-12
**Repo:** `runpod-gptoss-worker`
**Branch:** `main`
**Goal:** Build vLLM v0.15.1 from source on CUDA 12.8 for RTX 5090 (Blackwell)

---

## Background

The RunPod region with network storage only has CUDA 12.8 driver. The previous working image used CUDA 12.9, but that region lacks network storage. We need to rebuild using the `nvcr.io/nvidia/pytorch:25.02-py3` base image (CUDA 12.8.1) to deploy in the correct region.

**Target GPUs:** A40 (sm_86), A100 (sm_80), L40/L40S (sm_89), H100 (sm_90), RTX 5090 (sm_120)

---

## Issues Encountered & Fixes

### Issue 1: PEP 639 License Format (Fixed previously)

**Error:** `setuptools` in the base image (v70.3) couldn't parse vLLM's `pyproject.toml` license field.

**Fix:** Upgrade setuptools before building:
```dockerfile
RUN pip install --no-cache-dir "setuptools>=75.0" "packaging>=24.2" setuptools_scm cmake ninja wheel
```

### Issue 2: `requirements/build.txt` Re-downloads PyTorch (Fixed previously)

**Error:** vLLM's `requirements/build.txt` pulls a fresh PyTorch (~5GB), exhausting disk space.

**Fix:** Use `--no-build-isolation` to build against the base image's existing PyTorch:
```dockerfile
pip install --no-cache-dir --no-build-isolation .
```

### Issue 3: Triton Clone Exhausts Disk Space

**Error:**
```
Receiving objects: 52% (126557/243378), 460.61 MiB | 44.32 MiB/s
fatal: write error: No space left on device
Failed to clone repository: 'https://github.com/triton-lang/triton.git'
```

**Root cause:** During cmake configure, vLLM's `cmake/external_projects/triton_kernels.cmake` uses `FetchContent_MakeAvailable()` to clone the full `triton-lang/triton` repo (243K objects, 460MB+). GitHub Actions' disk fills up.

**Failed attempt:** `pip install -Ccmake.define.FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS=/tmp/triton` — the `-C` config setting syntax only works with scikit-build, not setuptools. vLLM uses setuptools, so the flag was silently ignored.

**Fix:** Pre-clone triton with `--depth 1` (~50MB), then inject cmake variable via `sed` directly into vLLM's cmake file:
```dockerfile
RUN git clone --depth 1 https://github.com/triton-lang/triton.git /tmp/triton && \
    git clone --depth 1 --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    sed -i '1i set(FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS "/tmp/triton" CACHE PATH "" FORCE)' \
        cmake/external_projects/triton_kernels.cmake && \
    pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/vllm /tmp/triton
```

**Commit:** `f2afc90`

### Issue 4: `libmpi.so.40` Missing After Cleanup

**Error:**
```
OSError: libmpi.so.40: cannot open shared object file: No such file or directory
```

**Root cause:** Aggressive disk cleanup deleted `/opt/hpcx/` which contains HPC-X MPI libraries. PyTorch in the base image is linked against `libmpi.so.40`.

**Fix:** Removed `/opt/hpcx/` from the cleanup list.

**Commit:** `596925a`

### Issue 5: PyTorch 2.6 Lacks Blackwell FP8 Types

**Error:**
```
/tmp/vllm/csrc/quantization/marlin/marlin.cu(587): error: enum "c10::ScalarType" has no member "Float8_e8m0fnu"
```

**Root cause:** vLLM v0.15.1 has Blackwell-specific quantization code (Marlin kernels) that uses `at::ScalarType::Float8_e8m0fnu` — an FP8 data type added in PyTorch 2.7+. The base image's PyTorch 2.6 doesn't have it.

The cmake correctly detected sm_120 (Blackwell) and enabled `-DENABLE_FP8`, `-DENABLE_SCALED_MM_SM120=1`, `-DENABLE_NVFP4_SM120=1`, `-DENABLE_CUTLASS_MOE_SM120=1`. But the PyTorch headers don't define the required types.

**Fix:** Upgrade PyTorch to the latest stable cu128 build before building vLLM:
```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
```

**Commit:** `9c17f82`

---

## TORCH_CUDA_ARCH_LIST Clarification

The original value `"8.0;8.6;8.9;9.0;12.0"` was **correct** all along:

| Value | sm_ | GPU | Status |
|-------|-----|-----|--------|
| 8.0 | sm_80 | A100 | Native SASS |
| 8.6 | sm_86 | A40 | Native SASS |
| 8.9 | sm_89 | L40, L40S | Native SASS |
| 9.0 | sm_90 | H100 | Native SASS |
| 12.0 | sm_120 | RTX 5090 | Native SASS |

**Note:** `12.0` is NOT the same as `10.0`:
- `sm_100` (10.0) = datacenter Blackwell (B100/B200) — does NOT run on RTX 5090
- `sm_120` (12.0) = consumer Blackwell (RTX 5090/5080) — correct target

Two wrong intermediate commits changed this to `9.0+PTX` and `10.0` before being corrected back to `12.0`.

---

## Final Dockerfile (as of `9c17f82`)

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV MAX_JOBS=2
ENV VLLM_TARGET_DEVICE=cuda

# Cleanup base image (~2-3GB freed)
RUN rm -rf /opt/nvidia/nsight-systems* /opt/nvidia/nsight-compute* \
    /opt/nvidia/entitlement* \
    /usr/local/cuda/samples /usr/local/cuda/extras/CUPTI/samples \
    /usr/local/cuda/compute-sanitizer \
    /usr/local/lib/python3.12/dist-packages/torch/test/ \
    /usr/local/lib/python3.12/dist-packages/tensorrt* \
    /usr/share/doc /usr/share/man /var/cache/apt/* \
    && pip cache purge 2>/dev/null || true \
    && apt-get clean 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages/ -name '*.pyc' -delete 2>/dev/null || true \
    && find /usr/local/lib/python3.12/dist-packages/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Upgrade PyTorch for Blackwell FP8 types (Float8_e8m0fnu)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Build tools
RUN pip install --no-cache-dir "setuptools>=75.0" "packaging>=24.2" setuptools_scm cmake ninja wheel

# Build vLLM v0.15.1 from source
# - Shallow triton clone via sed (saves 400MB+ disk)
# - --no-build-isolation uses existing PyTorch
RUN git clone --depth 1 https://github.com/triton-lang/triton.git /tmp/triton && \
    git clone --depth 1 --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    sed -i '1i set(FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS "/tmp/triton" CACHE PATH "" FORCE)' \
        cmake/external_projects/triton_kernels.cmake && \
    pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/vllm /tmp/triton

# RunPod handler dependencies
RUN pip install --no-cache-dir "runpod>=1.8,<2.0" pydantic pydantic-settings

# ... (environment vars, COPY src, CMD)
```

---

## Commit History

| Commit | Description |
|--------|-------------|
| `25dfb9d` | Fix build: pin packaging>=24.2 required by setuptools>=77 |
| `20c99f8` | ~~Fix CUDA arch: 9.0+PTX~~ (wrong — reverted) |
| `e21538c` | ~~Use native sm_100~~ (wrong — sm_100 ≠ RTX 5090) |
| `d73619b` | Fix disk space: shallow triton + aggressive cleanup |
| `596925a` | Fix: don't delete /opt/hpcx/ (PyTorch needs libmpi.so.40) |
| `f2afc90` | Fix triton fetch: sed cmake var instead of pip -C flag |
| `9c17f82` | Upgrade PyTorch to cu128 stable for Blackwell FP8 types |

---

## Key Learnings

1. **RTX 5090 = sm_120 = compute capability 12.0** (not 10.0, not 12.0 as "doesn't exist")
2. **sm_100 ≠ sm_120** — datacenter vs consumer Blackwell, no forward compatibility
3. **pip `-Ccmake.define.*` only works with scikit-build**, not setuptools — use `sed` to inject cmake variables
4. **Docker layer cleanup doesn't reclaim base image space** — `rm -rf` in a new layer adds whiteout entries but the base layer data persists
5. **vLLM v0.15.1 + Blackwell requires both**: CUDA 12.8 nvcc (for sm_120 compilation) AND PyTorch 2.7+ (for FP8 data types)
6. **Don't delete `/opt/hpcx/`** from NGC containers — PyTorch is linked against `libmpi.so.40`

---

## Status

**Build `9c17f82`:** In progress — waiting for GitHub Actions result.

**Potential remaining risks:**
- PyTorch cu128 stable wheel may have additional version conflicts with other NGC packages
- Disk space may still be tight with PyTorch upgrade (~2GB) added to the layer
- There may be additional PyTorch 2.6→2.7+ incompatibilities beyond `Float8_e8m0fnu`
