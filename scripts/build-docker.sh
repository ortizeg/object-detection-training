#!/bin/bash
set -e

IMAGE_NAME="object-detection-training"
TAG="latest"
GCP_PROJECT_ID="api-project-562713517696"
REGION="us"
REPO_NAME="object-detection-training"
ARTIFACT_HOST="$REGION-docker.pkg.dev"
ARTIFACT_URL="$ARTIFACT_HOST/$GCP_PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"

if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY is not set."
    exit 1
fi

echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build --platform linux/amd64 --build-arg WANDB_API_KEY=$WANDB_API_KEY -t $IMAGE_NAME:$TAG .

echo "Tagging and pushing to Artifact Registry ($ARTIFACT_URL)..."
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$ARTIFACT_HOST
docker tag $IMAGE_NAME:$TAG $ARTIFACT_URL
docker push $ARTIFACT_URL
echo "Successfully pushed to $ARTIFACT_URL"
