# Custom RunPod Serverless Worker for GPT-OSS-20B
# Uses official vLLM gptoss image as base

FROM vllm/vllm-openai:gptoss

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

# Start the handler
CMD ["python3", "/src/handler.py"]
