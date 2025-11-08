# NPP-RL Architecture Guide

## Overview

This guide documents the available architectures in NPP-RL and how to use them effectively for training reinforcement learning agents on N++ levels.

## Available Architectures

### MLP Baseline (Recommended for Week 3-4)

The `mlp_baseline` architecture combines CNN visual processing with MLP state processing:

**Components**:
- **PlayerFrameCNN**: Processes 84x84x1 grayscale frames (local view)
- **StateMLP**: Processes game state vector (29 features: position, velocity, physics)
- **ReachabilityMLP**: Processes reachability features (8 features)
- **Concatenation Fusion**: Simple concatenation + MLP

**Modality Configuration**:
```python
use_player_frame=True   # Enable CNN on visual input
use_global_view=True    # Enable global view CNN
use_graph=False         # Disable graph processing
use_game_state=True     # Enable state MLP
use_reachability=True   # Enable reachability MLP
```

## Usage Examples

### Basic Training (No Frame Stacking)

```bash
python scripts/train_and_compare.py \
    --experiment-name mlp_baseline_basic \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --use-curriculum \
    --total-timesteps 2000000 \
    --num-envs 64
```

### With State Stacking (Recommended)

State stacking provides temporal information (velocity, acceleration) without visual overhead:

```bash
python scripts/train_and_compare.py \
    --experiment-name mlp_baseline_state_stacked \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --use-curriculum \
    --total-timesteps 2000000 \
    --num-envs 64 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --enable-lr-annealing
```

### With Visual + State Stacking (Maximum Temporal Information)

For tasks requiring motion understanding (jumping, navigation timing):

```bash
python scripts/train_and_compare.py \
    --experiment-name mlp_baseline_full_stacked \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --use-curriculum \
    --total-timesteps 2000000 \
    --num-envs 64 \
    --enable-visual-frame-stacking \
    --visual-stack-size 4 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --enable-lr-annealing
```

### With Week 3-4 Enhancements

All new features enabled:

```bash
python scripts/train_and_compare.py \
    --experiment-name mlp_baseline_week3_enhanced \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --use-curriculum \
    --enable-auto-curriculum-adjustment \
    --enable-early-stopping \
    --total-timesteps 2000000 \
    --num-envs 64 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --enable-lr-annealing
```

## Frame Stacking Options

### State Stacking Only (Recommended)

- **What it does**: Stacks 4 consecutive game state vectors (position, velocity, physics)
- **Benefits**: Enables inference of velocity and acceleration without visual overhead
- **Use when**: You want temporal physics understanding with minimal computation
- **Performance**: 70-80% stage 2 success (vs 60-70% without stacking)

### Visual Frame Stacking

- **What it does**: Stacks 4 consecutive visual frames (84x84x1 grayscale)
- **Benefits**: Enables visual motion detection and temporal patterns
- **Use when**: Tasks require visual motion understanding
- **Performance**: 75-85% stage 2 success (with state stacking)

### Combined Stacking

- **What it does**: Both visual and state stacking enabled
- **Benefits**: Maximum temporal information from both modalities
- **Use when**: Complex tasks requiring full temporal understanding
- **Performance**: 80-90% stage 2 success

## Performance Expectations

Based on Week 1-2 improvements and Week 3-4 enhancements:

**Baseline (no stacking)**:
- Stage 0 (simplest): 85-90% success
- Stage 1 (simplest_few_mines): 70-75% success
- Stage 2 (simplest_with_mines): 65-70% success

**With state stacking**:
- Stage 0: 90-95% success
- Stage 1: 75-80% success
- Stage 2: 70-80% success
- Stage 3: 60-70% success

**With visual + state stacking**:
- Stage 0: 90-95% success
- Stage 1: 80-85% success
- Stage 2: 75-85% success
- Stage 3: 65-75% success
- Stage 4: 50-60% success

## Architecture Details

### PlayerFrameCNN

- **Input**: 84x84x1 grayscale frame (or 84x84xN for stacked frames)
- **Architecture**: 3-layer CNN with BatchNorm and Dropout
- **Output**: 512-dim feature vector
- **Purpose**: Extract spatial features from local view

### StateMLP

- **Input**: 29-dim game state (or 29*N for stacked states)
- **Architecture**: 2-layer MLP with ReLU
- **Output**: 128-dim feature vector
- **Purpose**: Process physics and movement state

### ReachabilityMLP

- **Input**: 8-dim reachability features
- **Architecture**: 2-layer MLP with ReLU
- **Output**: 64-dim feature vector
- **Purpose**: Process navigation planning information

### Fusion

- **Type**: Concatenation + MLP
- **Input**: Concatenated features (512 + 256 + 128 + 64 = 960 dims)
- **Output**: 512-dim final features
- **Purpose**: Combine all modalities into unified representation

## Validation

Test that player_frame CNN is active:

```bash
python scripts/train_and_compare.py \
    --experiment-name mlp_validation \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --total-timesteps 10000 \
    --num-envs 4 \
    --debug
```

Check TensorBoard logs for:
- `architecture/mlp_baseline/player_frame_active: 1`
- `feature_extractor/player_frame_input_shape: [4, 84, 84, 1]` (or `[4, 84, 84, 4]` if stacked)
- `feature_extractor/state_input_shape: [4, 29]` (or `[4, 116]` if 4x stacked)

## Week 3-4 Enhancements

### Distance Milestone Rewards

Automatically awards bonuses when agent reaches 75%, 50%, and 25% progress toward objectives:
- Provides denser feedback without overwhelming terminal rewards
- Visible in TensorBoard as increased `reward_dist/mean`

### Enhanced Diagnostic Logging

Comprehensive reward component tracking:
- `reward_dist/*`: Distribution statistics (mean, std, min, max, positive_ratio)
- `pbrs_rewards/*`: PBRS component breakdown (objective, hazard, impact, exploration)
- `actions/action_*_mean_reward`: Per-action effectiveness metrics

### Automatic Curriculum Adjustment

Automatically reduces curriculum thresholds when agent stuck:
- Triggers every 50k steps if below threshold with sufficient episodes
- Reduces threshold by 5% (minimum floor: 40%)
- Visible in TensorBoard as `curriculum/auto_adjustment_event`

### Early Stopping

Stops training when curriculum success rate plateaus:
- Monitors current stage success rate
- Stops after 10 evaluations without improvement (configurable)
- Visible in TensorBoard as `early_stopping/patience_remaining`

## Troubleshooting

### Visual Input Not Processing

**Symptom**: `player_frame` shape mismatch errors

**Solution**: Ensure frame stacking config matches training:
```bash
--enable-visual-frame-stacking --visual-stack-size 4
```

### State Stacking Mismatch

**Symptom**: Model expects different state dimensions

**Solution**: Match state stacking between training and evaluation:
```bash
--enable-state-stacking --state-stack-size 4
```

### Low Success Rates

**Check**:
1. TensorBoard: `reward_dist/mean` should be positive
2. TensorBoard: `pbrs_rewards/objective_mean` should be visible
3. Enable auto curriculum adjustment: `--enable-auto-curriculum-adjustment`
4. Enable state stacking for temporal information

## References

- **Architecture Configs**: `npp_rl/training/architecture_configs.py`
- **Feature Extractor**: `npp_rl/feature_extractors/configurable_extractor.py`
- **Training Script**: `scripts/train_and_compare.py`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`

