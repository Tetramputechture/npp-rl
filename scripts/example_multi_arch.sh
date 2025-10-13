#!/bin/bash
# Example: Compare multiple architectures

set -e

EXPERIMENT_NAME="arch_comparison"
ARCHITECTURES="full_hgt vision_free gat"
TRAIN_DATASET="../nclone/datasets/train"
TEST_DATASET="../nclone/datasets/test"
TOTAL_TIMESTEPS=10000000  # 10M for full training
NUM_ENVS=64
NUM_GPUS=4
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
    --eval-freq 100000 \
    --save-freq 500000 \
    --output-dir "${OUTPUT_DIR}"

echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}/${EXPERIMENT_NAME}_*/"
echo "View comparison with: tensorboard --logdir ${OUTPUT_DIR}/${EXPERIMENT_NAME}_*/"
