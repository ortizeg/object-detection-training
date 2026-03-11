#!/bin/bash
set -e

# Docker build and push script for object-detection-training
#
# Usage:
#   ./scripts/build-docker.sh [--registry <name>] [--local]
#
# Registries:
#   dockerhub  - Docker Hub (default, works with vast.ai and all providers)
#   ghcr       - GitHub Container Registry
#   gcp        - Google Artifact Registry
#
# Examples:
#   ./scripts/build-docker.sh                        # Build + push to Docker Hub
#   ./scripts/build-docker.sh --registry ghcr        # Build + push to GHCR
#   ./scripts/build-docker.sh --registry gcp         # Build + push to GCP AR
#   ./scripts/build-docker.sh --local                # Build only, no push
#
# Environment variables (override defaults):
#   DOCKERHUB_USER   - Docker Hub username    (default: ortizeg)
#   GHCR_USER        - GitHub username/org    (default: ortizeg)
#   GCP_PROJECT_ID   - GCP project ID         (default: from gcloud config)
#   GCP_REGION       - GCP AR region          (default: us)

LOCAL_ONLY=false
REGISTRY="dockerhub"

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_ONLY=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--registry <dockerhub|ghcr|gcp>] [--local]"
            exit 1
            ;;
    esac
done

IMAGE_NAME="object-detection-training"
SHORT_SHA=$(git rev-parse --short HEAD)

# --- Build ---
echo "Building Docker image: $IMAGE_NAME"
echo "  Platform:  linux/amd64"
echo "  SHORT_SHA: $SHORT_SHA"

docker build --platform linux/amd64 \
    -t "$IMAGE_NAME:$SHORT_SHA" \
    -t "$IMAGE_NAME:latest" \
    .

if [ "$LOCAL_ONLY" = true ]; then
    echo ""
    echo "Local build complete:"
    echo "  $IMAGE_NAME:$SHORT_SHA"
    echo "  $IMAGE_NAME:latest"
    exit 0
fi

# --- Registry-specific login, tag, and push ---
case "$REGISTRY" in
    dockerhub)
        DOCKERHUB_USER="${DOCKERHUB_USER:-ortizeg}"
        REMOTE_SHA="docker.io/$DOCKERHUB_USER/$IMAGE_NAME:$SHORT_SHA"
        REMOTE_LATEST="docker.io/$DOCKERHUB_USER/$IMAGE_NAME:latest"

        echo ""
        echo "Pushing to Docker Hub ($DOCKERHUB_USER/$IMAGE_NAME)..."
        # Requires: docker login (or DOCKER_USERNAME + DOCKER_PASSWORD env vars)
        if ! grep -q "index.docker.io\|docker.io\|auth" ~/.docker/config.json 2>/dev/null; then
            echo "Not logged in to Docker Hub. Run: docker login"
            exit 1
        fi
        ;;

    ghcr)
        GHCR_USER="${GHCR_USER:-ortizeg}"
        REMOTE_SHA="ghcr.io/$GHCR_USER/$IMAGE_NAME:$SHORT_SHA"
        REMOTE_LATEST="ghcr.io/$GHCR_USER/$IMAGE_NAME:latest"

        echo ""
        echo "Pushing to GHCR (ghcr.io/$GHCR_USER/$IMAGE_NAME)..."
        # Requires: echo $GITHUB_TOKEN | docker login ghcr.io -u $GHCR_USER --password-stdin
        if ! docker info 2>/dev/null | grep -q "ghcr.io"; then
            echo "Attempting GHCR login..."
            if [ -n "$GITHUB_TOKEN" ]; then
                echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
            else
                echo "Not logged in to GHCR. Run:"
                echo "  echo \$GITHUB_TOKEN | docker login ghcr.io -u $GHCR_USER --password-stdin"
                exit 1
            fi
        fi
        ;;

    gcp)
        GCP_REGION="${GCP_REGION:-us}"
        GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
        REPO_NAME="object-detection-training"
        ARTIFACT_HOST="$GCP_REGION-docker.pkg.dev"

        if [ -z "$GCP_PROJECT_ID" ]; then
            echo "Error: No GCP project configured."
            echo "  Set GCP_PROJECT_ID env var or run: gcloud config set project <PROJECT_ID>"
            exit 1
        fi

        REMOTE_SHA="$ARTIFACT_HOST/$GCP_PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$SHORT_SHA"
        REMOTE_LATEST="$ARTIFACT_HOST/$GCP_PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

        echo ""
        echo "Pushing to GCP Artifact Registry ($ARTIFACT_HOST/$GCP_PROJECT_ID/$REPO_NAME)..."
        gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://$ARTIFACT_HOST"
        ;;

    *)
        echo "Unknown registry: $REGISTRY"
        echo "Supported: dockerhub, ghcr, gcp"
        exit 1
        ;;
esac

docker tag "$IMAGE_NAME:$SHORT_SHA" "$REMOTE_SHA"
docker tag "$IMAGE_NAME:latest" "$REMOTE_LATEST"
docker push "$REMOTE_SHA"
docker push "$REMOTE_LATEST"

echo ""
echo "Successfully pushed:"
echo "  $REMOTE_SHA"
echo "  $REMOTE_LATEST"
echo ""
echo "Pull command:"
echo "  docker pull $REMOTE_LATEST"
