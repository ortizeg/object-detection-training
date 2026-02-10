#!/bin/bash
set -e

# Local Docker build script for object-detection-training
# Usage: ./scripts/build-docker.sh [--local]
#   --local: Build only, do not push to Artifact Registry

LOCAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --local)
            LOCAL_ONLY=true
            shift
            ;;
    esac
done

IMAGE_NAME="object-detection-training"
SHORT_SHA=$(git rev-parse --short HEAD)

# Derive GCP project from gcloud config
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] && [ "$LOCAL_ONLY" = false ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>"
    echo "       Or use --local to build without pushing."
    exit 1
fi

REGION="us"
REPO_NAME="object-detection-training"
ARTIFACT_HOST="$REGION-docker.pkg.dev"
ARTIFACT_URL_SHA="$ARTIFACT_HOST/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$SHORT_SHA"
ARTIFACT_URL_LATEST="$ARTIFACT_HOST/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest"

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

echo ""
echo "Tagging and pushing to Artifact Registry..."
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$ARTIFACT_HOST

docker tag "$IMAGE_NAME:$SHORT_SHA" "$ARTIFACT_URL_SHA"
docker tag "$IMAGE_NAME:latest" "$ARTIFACT_URL_LATEST"
docker push "$ARTIFACT_URL_SHA"
docker push "$ARTIFACT_URL_LATEST"

echo ""
echo "Successfully pushed:"
echo "  $ARTIFACT_URL_SHA"
echo "  $ARTIFACT_URL_LATEST"
