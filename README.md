# Custom RunPod Serverless Worker for GPT-OSS-20B

This is a custom RunPod serverless worker that uses the official `vllm/vllm-openai:gptoss` image
with RunPod's handler code. This fixes the compatibility issues with RunPod's outdated worker images.

## Prerequisites

- Docker installed
- Docker Hub account (free at https://hub.docker.com)
- ~50GB disk space for building

## Quick Start

### Option A: Build Locally

1. **Build and push the image:**

```bash
chmod +x build_and_push.sh
./build_and_push.sh YOUR_DOCKERHUB_USERNAME
```

2. **Use in RunPod Serverless:**
   - Image: `YOUR_DOCKERHUB_USERNAME/worker-vllm-gptoss:latest`
   - GPU: H100 or RTX 5090 (Blackwell)
   - Environment Variables:
     ```
     MODEL_NAME=openai/gpt-oss-20b
     TOKENIZER_MODE=slow
     MAX_MODEL_LEN=16384
     GPU_MEMORY_UTILIZATION=0.85
     ```

### Option B: Build with GitHub Actions (Easier)

1. Fork this folder to a new GitHub repo

2. Add Docker Hub credentials as GitHub Secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`

3. Create `.github/workflows/build.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ secrets.DOCKERHUB_USERNAME }}/worker-vllm-gptoss:latest
```

4. Push to GitHub - it will auto-build

## RunPod Configuration

| Setting | Value |
|---------|-------|
| Container Image | `YOUR_USERNAME/worker-vllm-gptoss:latest` |
| GPU | H100 80GB or RTX 5090 32GB |
| Start Command | *(leave empty)* |

### Environment Variables

```
MODEL_NAME=openai/gpt-oss-20b
TOKENIZER_MODE=slow
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.85
ENABLE_PREFIX_CACHING=false
```

## Testing

After deployment, test with:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Image Tags

| Tag | Base | GPU Support |
|-----|------|------------|
| `latest` / `v2-cu130` | vllm v0.15.1 + CUDA 13.0 | A40, A100, H100, **RTX 5090** |
| `v1-cu124` | vllm v0.15.1 + CUDA 12.4 | A40, A100, H100 (no 5090) |

To rollback: change RunPod container image to `YOUR_USERNAME/worker-vllm-gptoss:v1-cu124`

## Troubleshooting

- **Tokenizer error**: Make sure `TOKENIZER_MODE=slow` is set
- **FlashAttention error on RTX 5090**: Make sure you're using the `:latest` or `:v2-cu130` tag (CUDA 13.0)
- **Out of memory**: Reduce `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION`
