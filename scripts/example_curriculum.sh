#!/bin/bash
# Example: Curriculum learning with hierarchical PPO

echo "===================================="
echo "Curriculum Learning Example"
echo "===================================="
echo ""
echo "This trains with:"
echo "- Hierarchical PPO"
echo "- Curriculum learning (simple -> mine_heavy)"
echo "- Automatic progression through difficulty stages"
echo ""

python scripts/train_and_compare.py \
    --experiment-name "curriculum_hierarchical" \
    --architectures full_hgt \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 20000000 \
    --num-envs 64 \
    --num-gpus 1 \
    --output-dir experiments/ \
    --use-hierarchical-ppo \
    --high-level-update-freq 50 \
    --use-curriculum \
    --curriculum-start-stage simple \
    --curriculum-threshold 0.5 \
    --curriculum-min-episodes 50 \
    --enable-pbrs \
    --eval-freq 100000 \
    --save-freq 500000

echo ""
echo "Training complete!"
echo "View results:"
echo "  tensorboard --logdir experiments/curriculum_hierarchical_*/"
