# Custom RunPod Serverless Worker for GPT-OSS-20B
# Uses latest vLLM image (v0.15.1+ has GPT-OSS support built-in)

FROM vllm/vllm-openai:latest

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
    PYTHONPATH="/"

# Copy handler code
COPY src /src

# Override the base image's entrypoint
ENTRYPOINT []

# Start the handler
CMD ["python3", "/src/handler.py"]
