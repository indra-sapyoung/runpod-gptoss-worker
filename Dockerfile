# Custom RunPod Serverless Worker for GPT-OSS-20B
# Pre-built vLLM wheel — no source compilation, builds in ~5 minutes
# Works on RunPod GPUs: A40, RTX 5090 (via PTX JIT), L40S, etc.

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 (ships with Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install vLLM v0.15.1 with pre-built cu128 wheel (no compilation!)
# RTX 5090 (sm_120) works via PTX JIT — same as local pip install
RUN pip install --no-cache-dir --break-system-packages \
    "vllm==0.15.1" --extra-index-url https://download.pytorch.org/whl/cu128

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
