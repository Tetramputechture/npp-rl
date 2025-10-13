#!/bin/bash
# Example: Training with S3 artifact upload

set -e

# Check AWS credentials
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Error: AWS credentials not set!"
    echo "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
    exit 1
fi

EXPERIMENT_NAME="production_run"
ARCHITECTURES="full_hgt vision_free"
TRAIN_DATASET="../nclone/datasets/train"
TEST_DATASET="../nclone/datasets/test"
TOTAL_TIMESTEPS=10000000
NUM_ENVS=256
NUM_GPUS=4
S3_BUCKET="npp-rl-experiments"
S3_PREFIX="experiments/"
OUTPUT_DIR="experiments"

python scripts/train_and_compare.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --architectures ${ARCHITECTURES} \
    --no-pretraining \
    --train-dataset "${TRAIN_DATASET}" \
    --test-dataset "${TEST_DATASET}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-gpus ${NUM_GPUS} \
    --mixed-precision \
    --s3-bucket "${S3_BUCKET}" \
    --s3-prefix "${S3_PREFIX}" \
    --s3-sync-freq 500000 \
    --output-dir "${OUTPUT_DIR}"

echo "Training complete!"
echo "Artifacts uploaded to: s3://${S3_BUCKET}/${S3_PREFIX}${EXPERIMENT_NAME}"
echo "Local results: ${OUTPUT_DIR}/${EXPERIMENT_NAME}_*/"
