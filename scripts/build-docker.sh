#!/bin/bash
set -e

IMAGE_NAME="object-detection-training"
TAG="latest"
GCP_PROJECT_ID="api-project-562713517696"
REGION="us-central1"
REPO_NAME="object-detection-training"
ARTIFACT_HOST="$REGION-docker.pkg.dev"
ARTIFACT_URL="$ARTIFACT_HOST/$GCP_PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"

echo "Building Docker image: $IMAGE_NAME:$TAG"
docker build --platform linux/amd64 -t $IMAGE_NAME:$TAG .

echo "Tagging and pushing to Artifact Registry ($ARTIFACT_URL)..."
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://$ARTIFACT_HOST
docker tag $IMAGE_NAME:$TAG $ARTIFACT_URL
docker push $ARTIFACT_URL
echo "Successfully pushed to $ARTIFACT_URL"
