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
echo "Step 1: Backup current :latest as :v1-cu124"
echo "=========================================="

# Pull current latest and retag as backup
if docker pull "${FULL_IMAGE}:latest" 2>/dev/null; then
    docker tag "${FULL_IMAGE}:latest" "${FULL_IMAGE}:v1-cu124"
    echo "✅ Tagged backup: ${FULL_IMAGE}:v1-cu124"
else
    echo "⚠️ No existing :latest to backup (first build?)"
fi

echo "=========================================="
echo "Step 2: Building new image (cu130 for RTX 5090)"
echo "=========================================="

docker build -t "${FULL_IMAGE}:latest" -t "${FULL_IMAGE}:v2-cu130" .

echo "=========================================="
echo "Build complete!"
echo "=========================================="

# Ask to push
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to Docker Hub..."
    docker login

    echo "Pushing backup tag..."
    docker push "${FULL_IMAGE}:v1-cu124" 2>/dev/null || echo "(backup push skipped)"

    echo "Pushing new tags..."
    docker push "${FULL_IMAGE}:v2-cu130"
    docker push "${FULL_IMAGE}:latest"

    echo "=========================================="
    echo "Done! Images pushed:"
    echo "  ${FULL_IMAGE}:latest    ← new (cu130, RTX 5090 ready)"
    echo "  ${FULL_IMAGE}:v2-cu130  ← same as latest"
    echo "  ${FULL_IMAGE}:v1-cu124  ← backup (rollback)"
    echo ""
    echo "To rollback: change RunPod image to ${FULL_IMAGE}:v1-cu124"
    echo "=========================================="
else
    echo "Skipped push. To push later, run:"
    echo "  docker push ${FULL_IMAGE}:v1-cu124"
    echo "  docker push ${FULL_IMAGE}:v2-cu130"
    echo "  docker push ${FULL_IMAGE}:latest"
fi
