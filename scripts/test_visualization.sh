#!/bin/bash
# Test script for training visualization feature
# This runs a quick training session with visualization enabled

set -e

echo "========================================"
echo "NPP-RL Training Visualization Test"
echo "========================================"
echo ""
echo "This script will run a quick training session with visualization enabled."
echo "You should see a pygame window showing the agent playing N++."
echo ""
echo "Controls while training:"
echo "  SPACE - Pause/unpause visualization"
echo "  ESC/Q - Close visualization window (training continues)"
echo ""
echo "Press Ctrl+C to stop training at any time."
echo ""

# Default values
EXPERIMENT_NAME="viz_test_$(date +%Y%m%d_%H%M%S)"
ARCHITECTURE="simple_mlp"
NUM_ENVS=2
TOTAL_TIMESTEPS=50000
VIS_RENDER_FREQ=50
VIS_FPS=60

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --timesteps)
            TOTAL_TIMESTEPS="$2"
            shift 2
            ;;
        --render-freq)
            VIS_RENDER_FREQ="$2"
            shift 2
            ;;
        --fps)
            VIS_FPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--architecture ARCH] [--num-envs N] [--timesteps N] [--render-freq N] [--fps N]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Architecture: $ARCHITECTURE"
echo "  Environments: $NUM_ENVS"
echo "  Total timesteps: $TOTAL_TIMESTEPS"
echo "  Render frequency: every $VIS_RENDER_FREQ timesteps"
echo "  Target FPS: $VIS_FPS"
echo ""

# Check if train and test datasets exist
if [ ! -d "data/train" ] && [ ! -d "../data/train" ]; then
    echo "ERROR: Training dataset not found!"
    echo "Please ensure data/train/ exists with map files."
    echo ""
    echo "You can generate test data using:"
    echo "  python -m nclone.map_generation.generate_test_suite_maps"
    exit 1
fi

# Determine dataset paths
if [ -d "data/train" ]; then
    TRAIN_DATASET="data/train"
    TEST_DATASET="data/test"
else
    TRAIN_DATASET="../data/train"
    TEST_DATASET="../data/test"
fi

echo "Using datasets:"
echo "  Train: $TRAIN_DATASET"
echo "  Test: $TEST_DATASET"
echo ""

echo "Starting training with visualization..."
echo "========================================"
echo ""

# Run training with visualization
python scripts/train_and_compare.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --architectures "$ARCHITECTURE" \
    --train-dataset "$TRAIN_DATASET" \
    --test-dataset "$TEST_DATASET" \
    --num-envs "$NUM_ENVS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --visualize-training \
    --vis-render-freq "$VIS_RENDER_FREQ" \
    --vis-env-idx 0 \
    --vis-fps "$VIS_FPS" \
    --skip-final-eval \
    --no-pretraining

echo ""
echo "========================================"
echo "Visualization test complete!"
echo "========================================"

