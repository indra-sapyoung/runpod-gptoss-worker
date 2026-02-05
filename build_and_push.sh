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
TAG="latest"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "=========================================="
echo "Building: ${FULL_IMAGE}"
echo "=========================================="

# Build the image
docker build -t ${FULL_IMAGE} .

echo "=========================================="
echo "Build complete!"
echo "=========================================="

# Ask to push
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to Docker Hub..."
    docker login

    echo "Pushing ${FULL_IMAGE}..."
    docker push ${FULL_IMAGE}

    echo "=========================================="
    echo "Done! Use this image in RunPod:"
    echo "${FULL_IMAGE}"
    echo "=========================================="
else
    echo "Skipped push. To push later, run:"
    echo "docker push ${FULL_IMAGE}"
fi
