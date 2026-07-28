#!/bin/bash
set -euo pipefail

# Usage:
#   TRAIN_ROOT_DIRS="$HOME/FoundationTrain" \
#   OUTPUT_DIR="$HOME/FoundationTrain/distilled_cpsam_encoder_fpn_three_stage" \
#   qsub jobs/train_cpsam_student_fpn_stage1.pbs
#
# Or submit all three with PBS dependencies:
#   bash jobs/submit_cpsam_student_fpn_three_stage.pbs.sh

: "${REPO_DIR:=$HOME/MultiCellPose}"
: "${TRAIN_ROOT_DIRS:=$HOME/FoundationTrain}"
: "${OUTPUT_DIR:=$HOME/FoundationTrain/distilled_cpsam_encoder_fpn_three_stage}"
: "${CONDA_ENV:=cpsam_foundation310}"
: "${CUDA_MODULE:=cuda/12.1}"
: "${STAGE3_FLOW_CACHE_DIR:=${EPHEMERAL:-$OUTPUT_DIR}/cpsam_fpn_supervised_flow_cache}"
: "${STAGE3_LR:=1e-5}"
: "${STAGE3_TRAIN_MODE:=head-only}"
: "${STAGE3_OUTPUT_DISTILL_WEIGHT:=0.1}"
: "${STEPS_PER_EPOCH:=200}"
: "${BATCH_SIZE:=8}"
: "${NUM_WORKERS:=8}"

mkdir -p "${OUTPUT_DIR}"

stage1_job=$(
    qsub \
        -v "REPO_DIR=${REPO_DIR},TRAIN_ROOT_DIRS=${TRAIN_ROOT_DIRS},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},CUDA_MODULE=${CUDA_MODULE},STAGE3_FLOW_CACHE_DIR=${STAGE3_FLOW_CACHE_DIR},STAGE3_LR=${STAGE3_LR},STAGE3_TRAIN_MODE=${STAGE3_TRAIN_MODE},STAGE3_OUTPUT_DISTILL_WEIGHT=${STAGE3_OUTPUT_DISTILL_WEIGHT},STEPS_PER_EPOCH=${STEPS_PER_EPOCH},BATCH_SIZE=${BATCH_SIZE},NUM_WORKERS=${NUM_WORKERS}" \
        jobs/train_cpsam_student_fpn_stage1.pbs
)
stage1_id="${stage1_job%%.*}"
echo "submitted stage 1: ${stage1_job}"

stage2_job=$(
    qsub \
        -W "depend=afterok:${stage1_id}" \
        -v "REPO_DIR=${REPO_DIR},TRAIN_ROOT_DIRS=${TRAIN_ROOT_DIRS},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},CUDA_MODULE=${CUDA_MODULE},STAGE3_FLOW_CACHE_DIR=${STAGE3_FLOW_CACHE_DIR},STAGE3_LR=${STAGE3_LR},STAGE3_TRAIN_MODE=${STAGE3_TRAIN_MODE},STAGE3_OUTPUT_DISTILL_WEIGHT=${STAGE3_OUTPUT_DISTILL_WEIGHT},STEPS_PER_EPOCH=${STEPS_PER_EPOCH},BATCH_SIZE=${BATCH_SIZE},NUM_WORKERS=${NUM_WORKERS}" \
        jobs/train_cpsam_student_fpn_stage2.pbs
)
stage2_id="${stage2_job%%.*}"
echo "submitted stage 2: ${stage2_job}"

stage3_job=$(
    qsub \
        -W "depend=afterok:${stage2_id}" \
        -v "REPO_DIR=${REPO_DIR},TRAIN_ROOT_DIRS=${TRAIN_ROOT_DIRS},OUTPUT_DIR=${OUTPUT_DIR},CONDA_ENV=${CONDA_ENV},CUDA_MODULE=${CUDA_MODULE},STAGE3_FLOW_CACHE_DIR=${STAGE3_FLOW_CACHE_DIR},STAGE3_LR=${STAGE3_LR},STAGE3_TRAIN_MODE=${STAGE3_TRAIN_MODE},STAGE3_OUTPUT_DISTILL_WEIGHT=${STAGE3_OUTPUT_DISTILL_WEIGHT},STEPS_PER_EPOCH=${STEPS_PER_EPOCH},BATCH_SIZE=${BATCH_SIZE},NUM_WORKERS=${NUM_WORKERS}" \
        jobs/train_cpsam_student_fpn_stage3.pbs
)
echo "submitted stage 3: ${stage3_job}"
