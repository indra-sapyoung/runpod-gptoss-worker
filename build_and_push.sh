#!/bin/bash

# Build and push custom GPT-OSS RunPod worker
# Usage: ./build_and_push.sh <dockerhub_username>

set -e

if [ -z "$1" ]; then
    echo "Usage: ./build_and_push.sh <dockerhub_username>"
    echo "Example: ./build_and_push.sh myusername"
    exit 1
fi

DOCKER_USERNAME=$1
IMAGE_NAME="worker-vllm-gptoss"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo "=========================================="
echo "Building new image (vLLM v0.15.1, universal GPU support)"
echo "=========================================="

docker build -t "${FULL_IMAGE}:latest" -t "${FULL_IMAGE}:v3-universal" .

echo "=========================================="
echo "Build complete!"
echo "=========================================="

# Ask to push
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to Docker Hub..."
    docker login

    echo "Pushing tags..."
    docker push "${FULL_IMAGE}:v3-universal"
    docker push "${FULL_IMAGE}:latest"

    echo "=========================================="
    echo "Done! Images pushed:"
    echo "  ${FULL_IMAGE}:latest        ← current (all GPUs)"
    echo "  ${FULL_IMAGE}:v3-universal  ← same as latest"
    echo ""
    echo "Old images still available for rollback:"
    echo "  ${FULL_IMAGE}:v1-cu124      ← vLLM v0.15.1 (no CUDA override)"
    echo "  ${FULL_IMAGE}:v2-cu130      ← vLLM v0.15.1-cu130"
    echo "=========================================="
else
    echo "Skipped push. To push later, run:"
    echo "  docker push ${FULL_IMAGE}:v3-universal"
    echo "  docker push ${FULL_IMAGE}:latest"
fi
