# Custom RunPod Serverless Worker for GPT-OSS-20B
# v0.15.1-cu130: CUDA 13.0 for Blackwell (RTX 5090) + backward compat (A40, A100, H100)

FROM vllm/vllm-openai:v0.15.1-cu130

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
