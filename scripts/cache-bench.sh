#!/bin/bash
set -euo pipefail

# Cache Benchmark — submits 3 Vertex AI jobs (disk, ram, auto) on 2x T4
# Measures dataloader throughput in each cache mode under DDP.
#
# Usage:
#   ./scripts/cache-bench.sh [--image TAG] [--region REGION] [--dry-run]

# ── Defaults ──────────────────────────────────────────────────────────────────

IMAGE_TAG=""
REGION="us-east1"
DRY_RUN=false

# GCP constants
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SERVICE_ACCOUNT="562713517696-compute@developer.gserviceaccount.com"
ARTIFACT_REGISTRY="us-docker.pkg.dev/${PROJECT_ID}/object-detection-training/object-detection-training"

# Machine config — cheapest multi-GPU option
MACHINE_TYPE="n1-standard-16"
GPU_TYPE="NVIDIA_TESLA_T4"
GPU_COUNT=2
DISK_SIZE_GB=200

# Data paths — use val2017 as "train" for cheapness (~0.8GB)
COCO_BASE="/gcs/deep-ego-model-training/ego-training-data/coco/coco2017"
TRAIN_PATH="${COCO_BASE}/val2017"
VAL_PATH="${COCO_BASE}/val2017"

CONFIG="train_cache_bench_t4_2x"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# ── Parse Arguments ───────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --image)   IMAGE_TAG="$2"; shift 2 ;;
        --region)  REGION="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>"
    exit 1
fi

if [ -z "$IMAGE_TAG" ]; then
    IMAGE_TAG=$(git rev-parse --short HEAD)
fi

IMAGE_URI="${ARTIFACT_REGISTRY}:${IMAGE_TAG}"

# ── W&B API Key ───────────────────────────────────────────────────────────────

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    if command -v wandb &>/dev/null; then
        WANDB_API_KEY=$(python3 -c "import netrc; n=netrc.netrc(); print(n.authenticators('api.wandb.ai')[2])" 2>/dev/null || true)
    fi
fi

WANDB_ENV=""
if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ENV="
        - name: WANDB_API_KEY
          value: \"${WANDB_API_KEY}\""
else
    echo "Warning: WANDB_API_KEY not set. W&B logging disabled."
    WANDB_ENV="
        - name: WANDB_MODE
          value: \"disabled\""
fi

# ── Functions ─────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

submit_cache_job() {
    local label="$1"        # e.g. "disk", "ram", "auto", "staged"
    local cache_type="$2"   # Hydra data.cache_type value
    local extra_args="$3"   # Additional Hydra overrides (space-separated)
    local run_name="cache-bench-${label}-${TIMESTAMP}"
    local output_dir="/gcs/deep-ego-model-training/training-outputs/${run_name}"

    # Build extra args YAML entries
    local extra_yaml=""
    if [ -n "$extra_args" ]; then
        for arg in $extra_args; do
            extra_yaml="${extra_yaml}
        - \"${arg}\""
        done
    fi

    local job_spec=$(cat <<YAML
serviceAccount: ${SERVICE_ACCOUNT}
scheduling:
  strategy: SPOT
workerPoolSpecs:
  - machineSpec:
      machineType: ${MACHINE_TYPE}
      acceleratorType: ${GPU_TYPE}
      acceleratorCount: ${GPU_COUNT}
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: ${DISK_SIZE_GB}
    containerSpec:
      imageUri: ${IMAGE_URI}
      command: [pixi, run, task_manager]
      args:
        - "--config-name"
        - "${CONFIG}"
        - "data.cache_type=${cache_type}"
        - "data.train_path=${TRAIN_PATH}"
        - "data.val_path=${VAL_PATH}"
        - "data.test_path=${VAL_PATH}"
        - "data.num_workers=4"
        - "models.download_pretrained=false"
        - "hydra.run.dir=${output_dir}"${extra_yaml}
      env:
        - name: NCCL_IB_DISABLE
          value: "1"
        - name: HYDRA_FULL_ERROR
          value: "1"${WANDB_ENV}
YAML
)

    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN — ${label} job spec:"
        echo "$job_spec"
        echo "---"
        return
    fi

    local spec_file
    spec_file=$(mktemp /tmp/cache-bench-XXXXXXXX)
    mv "$spec_file" "${spec_file}.yaml"
    spec_file="${spec_file}.yaml"
    echo "$job_spec" > "$spec_file"

    log "Submitting ${label} job: ${run_name}"
    local output
    output=$(gcloud ai custom-jobs create \
        --display-name="${run_name}" \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --config="$spec_file" \
        2>&1)

    rm -f "$spec_file"

    local job_id
    job_id=$(echo "$output" | sed -n 's/.*customJobs\/\([0-9]*\).*/\1/p' | head -1)
    if [ -z "$job_id" ]; then
        job_id=$(echo "$output" | grep -oE '[0-9]{15,}' | head -1)
    fi

    if [ -z "$job_id" ]; then
        log "ERROR: Failed to extract job ID for ${cache_type}:"
        echo "$output"
        return
    fi

    log "  Job ID: ${job_id}"
    log "  Dashboard: https://console.cloud.google.com/vertex-ai/training/custom-jobs/${job_id}?project=${PROJECT_ID}"
    log "  Output: ${output_dir}"
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════════════"
log "Cache Benchmark — 4 jobs on 2x T4"
log "═══════════════════════════════════════════════════════════"
log "  Image:    ${IMAGE_URI}"
log "  Machine:  ${MACHINE_TYPE} (${GPU_COUNT}x ${GPU_TYPE})"
log "  Region:   ${REGION}"
log "  Data:     val2017 as train (~0.8GB)"
log "  Epochs:   3 x 100 batches"
log "  Modes:    disk, ram, auto, staged+disk"
log "═══════════════════════════════════════════════════════════"
echo ""

# Cache-only modes
submit_cache_job "disk"   "disk" ""
submit_cache_job "ram"    "ram"  ""
submit_cache_job "auto"   "auto" ""

# Staged: bulk-copy to local SSD first, then disk cache on top
submit_cache_job "staged" "disk" "data.stage_data=true"

log "═══════════════════════════════════════════════════════════"
log "All 4 jobs submitted. Check W&B or logs for throughput."
log "  Search W&B: project=DINOX, name contains 'cache-bench'"
log "  Or check GCS: gs://deep-ego-model-training/training-outputs/cache-bench-*"
log "═══════════════════════════════════════════════════════════"
