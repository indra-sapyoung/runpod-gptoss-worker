# Custom RunPod Serverless Worker for GPT-OSS-20B
# Clean build: minimal CUDA base + PyTorch nightly + vLLM from source
# No NGC bloat — just what we need for CUDA 12.8 + RTX 5090 (sm_120)

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 + build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git curl && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build settings for vLLM source compilation
# A40 (sm_86) + RTX 5090 (sm_120) only — add more archs if deploying on other GPUs
ENV TORCH_CUDA_ARCH_LIST="8.6;12.0"
ENV MAX_JOBS=2
ENV VLLM_TARGET_DEVICE=cuda

# Install PyTorch nightly cu128 (need 2.7+ for Float8_e8m0fnu / MXFP4 support)
# GPT-OSS-20B uses native MXFP4 weights, so FP8/FP4 quantization kernels are required
RUN pip install --no-cache-dir --break-system-packages \
    --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Install build tools
RUN pip install --no-cache-dir --break-system-packages \
    "setuptools>=75.0,<82" "packaging==24.2" setuptools_scm cmake ninja wheel

# Build vLLM v0.15.1 from source
# Pre-clone triton with --depth 1 to save disk (full clone is 460MB+)
# Inject cmake variable via sed (pip -C flag doesn't work with setuptools)
RUN git clone --depth 1 https://github.com/triton-lang/triton.git /tmp/triton && \
    git clone --depth 1 --branch v0.15.1 https://github.com/vllm-project/vllm.git /tmp/vllm && \
    cd /tmp/vllm && \
    sed -i '1i set(FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS "/tmp/triton" CACHE PATH "" FORCE)' \
        cmake/external_projects/triton_kernels.cmake && \
    pip install --no-cache-dir --break-system-packages --no-build-isolation . && \
    cd / && rm -rf /tmp/vllm /tmp/triton

# Install RunPod and other dependencies
RUN pip install --no-cache-dir --break-system-packages \
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
    ENFORCE_EAGER=true \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    TENSOR_PARALLEL_SIZE=1 \
    DISTRIBUTED_EXECUTOR_BACKEND=mp

# Copy handler code
COPY src /src

# Start the handler (raise thread limit to prevent thread exhaustion)
CMD ["bash", "-c", "ulimit -u 65535 2>/dev/null; exec python3 /src/handler.py"]
