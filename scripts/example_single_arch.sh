#!/bin/bash
# Example: Train single architecture without pretraining

set -e

EXPERIMENT_NAME="single_arch_test"
ARCHITECTURE="vision_free"
TRAIN_DATASET="../nclone/datasets/train"
TEST_DATASET="../nclone/datasets/test"
TOTAL_TIMESTEPS=1000000  # 1M for quick test
NUM_ENVS=16
OUTPUT_DIR="experiments"

python scripts/train_and_compare.py \
    --experiment-name "${EXPERIMENT_NAME}" \
    --architectures ${ARCHITECTURE} \
    --no-pretraining \
    --train-dataset "${TRAIN_DATASET}" \
    --test-dataset "${TEST_DATASET}" \
    --total-timesteps ${TOTAL_TIMESTEPS} \
    --num-envs ${NUM_ENVS} \
    --num-gpus 1 \
    --output-dir "${OUTPUT_DIR}" \
    --debug

echo "Training complete!"
echo "View logs with: tensorboard --logdir ${OUTPUT_DIR}/${EXPERIMENT_NAME}_*/"
