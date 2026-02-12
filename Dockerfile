# Custom RunPod Serverless Worker for GPT-OSS-20B
# Base: NVIDIA PyTorch container with CUDA 12.8.1
# Builds vLLM v0.15.1 from source for CUDA 12.8 compatibility
# Works on ALL RunPod GPUs: A40, A100, RTX 5090, L40, L40S (driver 570.x+)

FROM nvcr.io/nvidia/pytorch:25.02-py3

# Build settings for vLLM source compilation
# Include Ampere (8.0, 8.6, 8.9), Hopper (9.0), and Blackwell (12.0)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV MAX_JOBS=2
ENV VLLM_TARGET_DEVICE=cuda

# Free up space: base image is ~15GB, GitHub Actions has limited disk
RUN rm -rf /opt/nvidia/nsight-systems* /opt/nvidia/nsight-compute* \
    /opt/nvidia/entitlement* \
    /usr/local/cuda/samples /usr/local/cuda/extras/CUPTI/samples \
    /usr/share/doc /usr/share/man /var/cache/apt/* \
    && pip cache purge 2>/dev/null || true \
    && apt-get clean 2>/dev/null || true

# Install minimal build tools only (NOT requirements/build.txt which
# re-downloads torch + all CUDA libs that are already in the base image)
RUN pip install --no-cache-dir setuptools_scm cmake ninja packaging wheel

# Build vLLM v0.15.1 from source using existing PyTorch + CUDA 12.8
# --no-build-isolation: use base image's torch instead of downloading a new one
RUN git clone --depth 1 --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/vllm

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
