#!/bin/bash
set -e

# Cloud Build submission script for object-detection-training
# Usage: ./scripts/cloud-build.sh [--dry-run]

DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

# Compute SHORT_SHA locally (Cloud Build doesn't auto-populate for manual submissions)
SHORT_SHA=$(git rev-parse --short HEAD)

# Derive GCP project from gcloud config
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Warning: You have uncommitted changes. The build will use the last committed state."
    echo "         SHORT_SHA: $SHORT_SHA"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

REGION="us"
REPO_NAME="object-detection-training"
IMAGE_NAME="object-detection-training"

echo "Building image with Cloud Build..."
echo "  Project:   $PROJECT_ID"
echo "  SHORT_SHA: $SHORT_SHA"
echo "  Image:     ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

CMD="gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions=SHORT_SHA=${SHORT_SHA} \
    --project=${PROJECT_ID} \
    ."

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Dry run - would execute:"
    echo "$CMD"
else
    echo ""
    eval $CMD
    echo ""
    echo "Successfully built and pushed:"
    echo "  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${SHORT_SHA}"
    echo "  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
fi
