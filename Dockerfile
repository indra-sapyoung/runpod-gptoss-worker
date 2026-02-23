# Custom RunPod Serverless Worker for GPT-OSS-20B
# Base: NVIDIA PyTorch container with CUDA 12.8.1
# Builds vLLM v0.15.1 from source for CUDA 12.8 compatibility
# Works on ALL RunPod GPUs: A40, A100, RTX 5090, L40, L40S (driver 570.x+)

FROM nvcr.io/nvidia/pytorch:25.02-py3

# Build settings for vLLM source compilation
# A40 (sm_86) + RTX 5090 (sm_120) only — add more archs if deploying on other GPUs
# Each arch adds ~10 min compile time, so keep this minimal
ENV TORCH_CUDA_ARCH_LIST="8.6;12.0"
ENV MAX_JOBS=4
ENV VLLM_TARGET_DEVICE=cuda

# Free up space: base image is ~15GB, GitHub Actions has limited disk
# Must free enough for vLLM build + triton clone (~500MB) + build artifacts
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

# Upgrade PyTorch to nightly cu128 (base image's 2.6 lacks Blackwell FP8 types)
# vLLM v0.15.1 requires at::ScalarType::Float8_e8m0fnu (added in PyTorch 2.7+)
# --force-reinstall: NGC's torch 2.6.0a0 satisfies pip's check, so without --force it's a no-op
# --no-deps: keep NGC's existing CUDA libs (nvidia-cublas, nccl, etc.) — saves ~4GB disk
# nvidia-nvshmem-cu12: new dep in torch 2.12 not in NGC base (libnvshmem_host.so.3)
RUN pip install --no-cache-dir --force-reinstall --no-deps --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir nvidia-nvshmem-cu12

# Upgrade setuptools (base image's version doesn't support PEP 639 license format)
# and install minimal build tools (NOT requirements/build.txt which
# re-downloads torch + all CUDA libs already in the base image)
RUN pip install --no-cache-dir "setuptools>=75.0" "packaging>=24.2" setuptools_scm cmake ninja wheel

# Build vLLM v0.15.1 from source using existing PyTorch + CUDA 12.8
# --no-build-isolation: use base image's torch instead of downloading a new one
# Pre-clone triton with --depth 1 to avoid disk space exhaustion (full clone is 460MB+)
# Then inject cmake variable via sed (pip -C flag doesn't work with setuptools)
RUN git clone --depth 1 https://github.com/triton-lang/triton.git /tmp/triton && \
    git clone --depth 1 --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    sed -i '1i set(FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS "/tmp/triton" CACHE PATH "" FORCE)' \
        cmake/external_projects/triton_kernels.cmake && \
    pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/vllm /tmp/triton

# Install RunPod and other dependencies
RUN pip install --no-cache-dir \
    "runpod>=1.8,<2.0" \
    pydantic \
    pydantic-settings

# Setup environment
ARG MODEL_NAME="openai/gpt-oss-20b"
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
    BASE_PATH=$BASE_PATH \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    PYTHONPATH="/" \
    RAY_metrics_report_interval_ms=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_DISABLE_DOCKER_CPU_WARNING=1 \
    ENFORCE_EAGER=true \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    TENSOR_PARALLEL_SIZE=1 \
    DISTRIBUTED_EXECUTOR_BACKEND=mp

# Copy handler code
COPY src /src

# Override the base image's entrypoint
ENTRYPOINT []

# Start the handler (raise thread limit to prevent thread exhaustion)
CMD ["bash", "-c", "ulimit -u 65535 2>/dev/null; exec python3 /src/handler.py"]
