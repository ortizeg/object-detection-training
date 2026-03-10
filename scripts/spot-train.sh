#!/bin/bash
set -euo pipefail

# Spot Instance Training Runner for Vertex AI
# Submits a training job with spot/preemptible VMs, monitors for preemption,
# and auto-restarts from the last checkpoint.
#
# Usage:
#   ./scripts/spot-train.sh --config train_dinox_s_coco_l4_8x [OPTIONS]
#
# Required:
#   --config NAME          Hydra config name (e.g. train_dinox_s_coco_l4_8x)
#
# Optional:
#   --image TAG            Docker image tag (default: latest git SHA)
#   --run-name NAME        Logical run name (default: config name + timestamp)
#   --region REGION        GCP region (default: us-central1)
#   --machine MACHINE      Machine type (default: g2-standard-96)
#   --gpu-type TYPE        GPU type (default: NVIDIA_L4)
#   --gpu-count N          Number of GPUs (default: 8)
#   --disk-size GB         Boot disk size in GB (default: 500)
#   --max-restarts N       Max preemption restarts (default: 10)
#   --poll-interval SEC    Seconds between status checks (default: 300)
#   --num-workers N        DataLoader workers (default: 4)
#   --extra-args "..."     Extra Hydra overrides (quoted string)
#   --output-dir PATH      GCS output directory for checkpoints
#   --ckpt-path PATH       Resume from specific checkpoint path
#   --dry-run              Print job spec without submitting
#
# Examples:
#   # Basic spot training on 8x L4
#   ./scripts/spot-train.sh --config train_dinox_s_coco_l4_8x
#
#   # A100 training with custom run name
#   ./scripts/spot-train.sh --config train_dinox_s_coco_a100_1x \
#     --machine a2-highgpu-1g --gpu-type NVIDIA_TESLA_A100 --gpu-count 1 \
#     --run-name "baseline-a100-v1"
#
#   # Resume from checkpoint after manual stop
#   ./scripts/spot-train.sh --config train_dinox_s_coco_l4_8x \
#     --ckpt-path /app/outputs/my-run/checkpoints/last.ckpt

# ── Defaults ────────────────────────────────────────────────────────────────

CONFIG=""
IMAGE_TAG=""
RUN_NAME=""
REGION="us-central1"
MACHINE_TYPE="g2-standard-96"
GPU_TYPE="NVIDIA_L4"
GPU_COUNT=8
DISK_SIZE_GB=500
MAX_RESTARTS=10
POLL_INTERVAL=300
NUM_WORKERS=4
EXTRA_ARGS=""
OUTPUT_DIR=""
CKPT_PATH=""
DRY_RUN=false

# GCP constants (derived from existing scripts)
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
SERVICE_ACCOUNT="562713517696-compute@developer.gserviceaccount.com"
ARTIFACT_REGISTRY="us-docker.pkg.dev/${PROJECT_ID}/object-detection-training/object-detection-training"

# ── Parse Arguments ─────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2"; shift 2 ;;
        --image)        IMAGE_TAG="$2"; shift 2 ;;
        --run-name)     RUN_NAME="$2"; shift 2 ;;
        --region)       REGION="$2"; shift 2 ;;
        --machine)      MACHINE_TYPE="$2"; shift 2 ;;
        --gpu-type)     GPU_TYPE="$2"; shift 2 ;;
        --gpu-count)    GPU_COUNT="$2"; shift 2 ;;
        --disk-size)    DISK_SIZE_GB="$2"; shift 2 ;;
        --max-restarts) MAX_RESTARTS="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
        --extra-args)   EXTRA_ARGS="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --ckpt-path)    CKPT_PATH="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    echo "Usage: ./scripts/spot-train.sh --config <hydra-config-name> [OPTIONS]"
    exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project <PROJECT_ID>"
    exit 1
fi

# Default image tag to current git SHA
if [ -z "$IMAGE_TAG" ]; then
    IMAGE_TAG=$(git rev-parse --short HEAD)
fi

# Default run name
if [ -z "$RUN_NAME" ]; then
    RUN_NAME="${CONFIG}-$(date +%Y%m%d-%H%M%S)"
fi

# Default output dir — use GCS FUSE so checkpoints survive preemption
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="/gcs/deep-ego-model-training/training-outputs/${RUN_NAME}"
fi

IMAGE_URI="${ARTIFACT_REGISTRY}:${IMAGE_TAG}"

# ── W&B API Key ─────────────────────────────────────────────────────────────

WANDB_API_KEY="${WANDB_API_KEY:-}"
if [ -z "$WANDB_API_KEY" ]; then
    # Try to read from wandb config
    if command -v wandb &>/dev/null; then
        WANDB_API_KEY=$(python3 -c "import netrc; n=netrc.netrc(); print(n.authenticators('api.wandb.ai')[2])" 2>/dev/null || true)
    fi
    if [ -z "$WANDB_API_KEY" ]; then
        echo "Warning: WANDB_API_KEY not set. W&B logging will be disabled."
        echo "  Set it via: export WANDB_API_KEY=<your-key>"
    fi
fi

# ── Functions ───────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

submit_job() {
    local job_name="$1"
    local ckpt_arg="$2"

    # Build args list
    local args=(
        "--config-name" "$CONFIG"
        "trainer.max_epochs=300"
        "models.download_pretrained=false"
        "data.num_workers=${NUM_WORKERS}"
        "data.train_path=/gcs/deep-ego-model-training/ego-training-data/coco/coco2017/train2017"
        "data.val_path=/gcs/deep-ego-model-training/ego-training-data/coco/coco2017/val2017"
        "data.test_path=/gcs/deep-ego-model-training/ego-training-data/coco/coco2017/val2017"
        "hydra.run.dir=${OUTPUT_DIR}"
    )

    # Add checkpoint resume if provided
    if [ -n "$ckpt_arg" ]; then
        args+=("ckpt_path=${ckpt_arg}")
    fi

    # Add any extra Hydra overrides
    if [ -n "$EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        args+=($EXTRA_ARGS)
    fi

    # Build env vars section
    local env_section=""
    if [ -n "$WANDB_API_KEY" ]; then
        env_section="
        env:
          - name: WANDB_API_KEY
            value: \"${WANDB_API_KEY}\"
          - name: NCCL_IB_DISABLE
            value: \"1\"
          - name: HYDRA_FULL_ERROR
            value: \"1\""
    else
        env_section="
        env:
          - name: NCCL_IB_DISABLE
            value: \"1\"
          - name: HYDRA_FULL_ERROR
            value: \"1\"
          - name: WANDB_MODE
            value: \"disabled\""
    fi

    # Build args YAML list
    local args_yaml=""
    for arg in "${args[@]}"; do
        args_yaml="${args_yaml}
          - \"${arg}\""
    done

    # Create job spec
    local job_spec=$(cat <<YAML
displayName: ${job_name}
jobSpec:
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
        args:${args_yaml}${env_section}
YAML
)

    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN - Job spec:"
        echo "$job_spec"
        echo "---"
        return
    fi

    # Write spec to temp file and submit
    local spec_file=$(mktemp /tmp/spot-train-XXXXXX.yaml)
    echo "$job_spec" > "$spec_file"

    log "Submitting job: ${job_name}"
    local output
    output=$(gcloud ai custom-jobs create \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --config="$spec_file" \
        2>&1)

    rm -f "$spec_file"

    # Extract job ID from output (macOS-compatible — no grep -P)
    local job_id
    job_id=$(echo "$output" | sed -n 's/.*customJobs\/\([0-9]*\).*/\1/p' | head -1)
    if [ -z "$job_id" ]; then
        # Try alternative: longest numeric string (Vertex AI IDs are 19 digits)
        job_id=$(echo "$output" | grep -oE '[0-9]{15,}' | head -1)
    fi

    if [ -z "$job_id" ]; then
        log "ERROR: Failed to extract job ID from output:"
        echo "$output"
        exit 1
    fi

    echo "$job_id"
}

get_job_state() {
    local job_id="$1"
    gcloud ai custom-jobs describe "$job_id" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(state)" 2>/dev/null
}

get_job_error() {
    local job_id="$1"
    gcloud ai custom-jobs describe "$job_id" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format="value(error.message)" 2>/dev/null
}

wait_for_job() {
    local job_id="$1"
    local restart_num="$2"

    log "Monitoring job ${job_id} (restart #${restart_num})..."
    log "  Poll interval: ${POLL_INTERVAL}s"
    log "  Dashboard: https://console.cloud.google.com/vertex-ai/training/custom-jobs/${job_id}?project=${PROJECT_ID}"

    while true; do
        sleep "$POLL_INTERVAL"

        local state
        state=$(get_job_state "$job_id")

        case "$state" in
            JOB_STATE_SUCCEEDED)
                log "Job ${job_id} completed SUCCESSFULLY"
                return 0
                ;;
            JOB_STATE_FAILED)
                local error_msg
                error_msg=$(get_job_error "$job_id")
                if echo "$error_msg" | grep -qi "preempt\|spot\|stockout"; then
                    log "Job ${job_id} was PREEMPTED: ${error_msg}"
                    return 2  # Preempted — should restart
                else
                    log "Job ${job_id} FAILED: ${error_msg}"
                    return 1  # Real failure — don't restart
                fi
                ;;
            JOB_STATE_CANCELLED)
                log "Job ${job_id} was CANCELLED"
                return 1
                ;;
            JOB_STATE_QUEUED|JOB_STATE_PENDING|JOB_STATE_RUNNING|JOB_STATE_PREPARING)
                # Still running — print status every 10 polls (50 min at 5 min interval)
                log "  Status: ${state}"
                ;;
            "")
                log "  Warning: Empty state returned, retrying..."
                ;;
            *)
                log "  Unknown state: ${state}"
                ;;
        esac
    done
}

# ── Main Loop ───────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════════════"
log "Spot Training Runner"
log "═══════════════════════════════════════════════════════════"
log "  Config:      ${CONFIG}"
log "  Image:       ${IMAGE_URI}"
log "  Run Name:    ${RUN_NAME}"
log "  Machine:     ${MACHINE_TYPE} (${GPU_COUNT}x ${GPU_TYPE})"
log "  Disk:        ${DISK_SIZE_GB}GB SSD"
log "  Output:      ${OUTPUT_DIR}"
log "  Max Restarts: ${MAX_RESTARTS}"
log "  Poll Every:  ${POLL_INTERVAL}s"
if [ -n "$CKPT_PATH" ]; then
    log "  Resume From: ${CKPT_PATH}"
fi
log "═══════════════════════════════════════════════════════════"

current_ckpt="$CKPT_PATH"
restart_count=0
total_start=$(date +%s)

while true; do
    job_name="${RUN_NAME}-r${restart_count}"

    if [ "$DRY_RUN" = true ]; then
        submit_job "$job_name" "$current_ckpt"
        log "Dry run complete."
        exit 0
    fi

    job_id=$(submit_job "$job_name" "$current_ckpt")
    log "Submitted job: ${job_id} (${job_name})"

    wait_for_job "$job_id" "$restart_count"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # Success
        total_elapsed=$(( $(date +%s) - total_start ))
        log "═══════════════════════════════════════════════════════════"
        log "Training COMPLETE after ${restart_count} restarts"
        log "Total wall time: $(( total_elapsed / 3600 ))h $(( (total_elapsed % 3600) / 60 ))m"
        log "═══════════════════════════════════════════════════════════"
        exit 0
    elif [ $exit_code -eq 2 ]; then
        # Preempted — restart from last checkpoint
        restart_count=$((restart_count + 1))
        if [ $restart_count -gt "$MAX_RESTARTS" ]; then
            log "ERROR: Max restarts (${MAX_RESTARTS}) reached. Giving up."
            exit 1
        fi

        # Resume from last.ckpt in the output directory
        current_ckpt="${OUTPUT_DIR}/checkpoints/last.ckpt"
        log "Restarting (${restart_count}/${MAX_RESTARTS}) from: ${current_ckpt}"
        log "Waiting 30s before resubmitting..."
        sleep 30
    else
        # Real failure
        total_elapsed=$(( $(date +%s) - total_start ))
        log "═══════════════════════════════════════════════════════════"
        log "Training FAILED after ${restart_count} restarts"
        log "Total wall time: $(( total_elapsed / 3600 ))h $(( (total_elapsed % 3600) / 60 ))m"
        log "Check logs: gcloud ai custom-jobs describe ${job_id} --region=${REGION}"
        log "═══════════════════════════════════════════════════════════"
        exit 1
    fi
done
