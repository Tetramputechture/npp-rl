# Frame Stacking Pretraining Pipeline - Implementation Summary

## Overview

This document summarizes the comprehensive implementation of frame stacking support in the BC (Behavioral Cloning) pretraining pipeline for NPP-RL.

## Problem Statement

The original BC pretraining pipeline did not support frame stacking, causing incompatibility when:
1. BC checkpoint was trained without frame stacking (1 input channel)
2. Target RL policy was configured with frame stacking enabled (4 input channels)

This led to weight loading errors during pretraining transfer, specifically:
```
RuntimeError: Error(s) in loading state_dict for ActorCriticPolicy:
size mismatch for features_extractor.player_frame_cnn.conv_layers.0.weight: 
copying a param with shape torch.Size([32, 1, 8, 8]) from checkpoint, 
the shape in current model is torch.Size([32, 4, 8, 8]).
```

## Solution Architecture

### 1. BC Dataset Frame Stacking Support

**File**: `npp_rl/training/bc_dataset.py`

**Changes**:
- Added `frame_stack_config` parameter to `BCReplayDataset.__init__()`
- Implemented frame buffer management using `collections.deque`
- Added visual frame stacking: stacks frames along first dimension as (stack_size, H, W, C)
- Added state stacking: concatenates game states along first dimension
- Implemented two padding strategies:
  - `zero`: Pad initial frames with zeros
  - `repeat`: Pad by repeating the first frame
- Updated `_simulate_replay()` to populate buffers and create stacked observations
- Updated `_simulate_replay_with_env()` fallback to support frame stacking

**Key Methods**:
- `_reset_frame_buffers()`: Initialize/reset frame buffers for new replay
- `_add_to_visual_buffer()`: Add visual frame to buffer
- `_add_to_state_buffer()`: Add game state to buffer
- `_buffers_ready()`: Check if buffers are full
- `_stack_observations()`: Create stacked observation from buffers

### 2. Pretraining Pipeline Updates

**File**: `npp_rl/training/pretraining_pipeline.py`

**Changes**:
- Added `frame_stack_config` parameter to `PretrainingPipeline.__init__()`
- Pass frame_stack_config to BCReplayDataset during dataset creation
- Added logging to display frame stacking configuration
- Updated `run_bc_pretraining_if_available()` to accept and propagate frame_stack_config

### 3. BC Trainer Updates

**File**: `npp_rl/training/bc_trainer.py`

**Changes**:
- Added `frame_stack_config` parameter to `BCTrainer.__init__()`
- Pass frame_stack_config to `save_policy_checkpoint()` during checkpoint saving
- Frame stacking config is now included in all checkpoint saves (best, periodic, final)

### 4. Policy Checkpoint Metadata

**File**: `npp_rl/training/policy_utils.py`

**Changes**:
- Added `frame_stack_config` parameter to `save_policy_checkpoint()`
- Save frame stacking configuration in checkpoint under 'frame_stacking' key
- Added logging to display frame stacking info when saving checkpoints

**Checkpoint Structure**:
```python
{
    'policy_state_dict': {...},
    'epoch': 10,
    'metrics': {'loss': 0.5, 'accuracy': 0.9},
    'architecture': 'mlp_baseline',
    'frame_stacking': {
        'enable_visual_frame_stacking': True,
        'visual_stack_size': 4,
        'enable_state_stacking': True,
        'state_stack_size': 4,
        'padding_type': 'zero'
    }
}
```

### 5. Train and Compare Script Updates

**File**: `scripts/train_and_compare.py`

**Changes**:
- Build `bc_frame_stack_config` from command-line arguments in both:
  - Multi-GPU worker function (`train_worker()`)
  - Single GPU/CPU training path
- Pass frame_stack_config to all `run_bc_pretraining_if_available()` calls
- Config is built from args:
  - `enable_visual_frame_stacking`
  - `visual_stack_size`
  - `enable_state_stacking`
  - `state_stack_size`
  - `frame_stack_padding`

## Data Flow

```
Command-line Args (train_and_compare.py)
    ↓
bc_frame_stack_config = {
    'enable_visual_frame_stacking': args.enable_visual_frame_stacking,
    'visual_stack_size': args.visual_stack_size,
    'enable_state_stacking': args.enable_state_stacking,
    'state_stack_size': args.state_stack_size,
    'padding_type': args.frame_stack_padding,
}
    ↓
run_bc_pretraining_if_available(frame_stack_config=bc_frame_stack_config)
    ↓
PretrainingPipeline(frame_stack_config=bc_frame_stack_config)
    ↓
BCReplayDataset(frame_stack_config=bc_frame_stack_config)
    [Processes replays with frame stacking]
    ↓
BCTrainer(frame_stack_config=bc_frame_stack_config)
    [Trains policy with stacked observations]
    ↓
save_policy_checkpoint(frame_stack_config=bc_frame_stack_config)
    [Saves checkpoint with frame stacking metadata]
```

## Frame Stacking Implementation Details

### Visual Frame Stacking

When enabled:
1. Maintains a deque buffer of size `visual_stack_size`
2. For each step, adds current visual frame to buffer
3. When buffer is full, stacks frames: `np.stack(buffer, axis=0)`
4. Output shape: `(stack_size, H, W, C)` e.g., `(4, 96, 96, 3)`
5. Padding for initial frames (when buffer not full):
   - **Zero padding**: Pads with `np.zeros((H, W, C))`
   - **Repeat padding**: Repeats first frame

### State Stacking

When enabled:
1. Maintains a deque buffer of size `state_stack_size`
2. For each step, adds current game state to buffer
3. When buffer is full, concatenates states: `np.concatenate(buffer, axis=0)`
4. Output shape: `(stack_size * state_dim,)` e.g., `(400,)` for state_dim=100, stack_size=4
5. Padding for initial frames:
   - **Zero padding**: Pads with `np.zeros(state_dim)`
   - **Repeat padding**: Repeats first state

### Observation Structure

With frame stacking enabled, observations returned by BC dataset:
```python
{
    'player_frame': np.array(shape=(4, 96, 96, 3)),  # Stacked visual frames
    'game_state': np.array(shape=(400,)),            # Concatenated states
    # ... other observations based on architecture config
}
```

## Testing and Validation

### Unit Tests

**File**: `tests/test_frame_stacking_bc.py`

Tests covering:
- Frame buffer initialization
- Visual stacking shape validation
- State stacking concatenation
- Zero padding logic
- Repeat padding logic
- Checkpoint metadata structure
- Architecture compatibility
- Config propagation through pipeline
- Backward compatibility (without frame stacking)

### Integration Test

**File**: `scripts/test_frame_stacking_implementation.py`

Validates:
- Visual frame stacking produces correct shapes
- State stacking produces correct shapes
- Zero padding works correctly
- Repeat padding works correctly
- Checkpoint structure with frame stacking
- Config propagation through pipeline

**All tests passing**: ✓

### Validation Scripts

1. **`scripts/validate_checkpoint_simple.py`**
   - Quick checkpoint validation
   - Compares checkpoint input channels with target
   - Provides compatibility recommendations

2. **`scripts/validate_frame_stacking_pretraining.py`**
   - Comprehensive pipeline validation
   - Tests checkpoint structure
   - Tests BC dataset with frame stacking
   - Tests full pretraining flow

## Usage Example

### Training BC with Frame Stacking

```bash
python scripts/train_and_compare.py \
    --experiment-name "bc_frame_stacking_test" \
    --architectures mlp_baseline \
    --replay-data-dir ../nclone/bc_replays \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 4 \
    --num-gpus 1 \
    --use-curriculum \
    --use-hierarchical-ppo \
    --output-dir experiments/ \
    --enable-visual-frame-stacking \
    --visual-stack-size 4 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --frame-stack-padding zero
```

### Expected Logs

During BC pretraining:
```
2025-10-23 XX:XX:XX [INFO] Initialized pretraining pipeline for mlp_baseline
2025-10-23 XX:XX:XX [INFO] Frame stacking configuration:
2025-10-23 XX:XX:XX [INFO]   Visual: True (size: 4)
2025-10-23 XX:XX:XX [INFO]   State: True (size: 4)
2025-10-23 XX:XX:XX [INFO]   Padding: zero
```

During checkpoint saving:
```
2025-10-23 XX:XX:XX [INFO] Saved policy checkpoint to experiments/.../bc_best.pth
2025-10-23 XX:XX:XX [INFO]   Frame stacking config saved in checkpoint:
2025-10-23 XX:XX:XX [INFO]     Visual: True (size: 4)
2025-10-23 XX:XX:XX [INFO]     State: True (size: 4)
```

During RL training with pretrained checkpoint:
```
2025-10-23 XX:XX:XX [INFO] ✓ Loaded BC pretrained feature extractor weights
2025-10-23 XX:XX:XX [INFO]   Loaded 58 weight tensors (BC → hierarchical)
2025-10-23 XX:XX:XX [INFO]   Missing keys (will use random init): 74
```

**Key difference**: Now the input channels will match (4 vs 4) instead of mismatching (1 vs 4)!

## Backward Compatibility

The implementation maintains full backward compatibility:

1. **Without frame stacking**: When `frame_stack_config` is `None` or empty dict:
   - BC dataset returns single frames/states (no stacking)
   - Checkpoints saved without 'frame_stacking' key
   - Existing checkpoints can still be loaded

2. **Checkpoint loading**: The weight loading logic in `architecture_trainer.py` already handles:
   - Missing keys (new parameters in target model)
   - Unexpected keys (parameters only in checkpoint)
   - Shape mismatches (logs warning, skips incompatible weights)

## Files Modified

1. `npp_rl/training/bc_dataset.py` - Frame stacking in dataset
2. `npp_rl/training/pretraining_pipeline.py` - Pipeline config propagation
3. `npp_rl/training/bc_trainer.py` - Trainer config storage
4. `npp_rl/training/policy_utils.py` - Checkpoint metadata
5. `scripts/train_and_compare.py` - Config building and passing

## Files Created

1. `scripts/validate_checkpoint_simple.py` - Quick validation tool
2. `scripts/validate_frame_stacking_pretraining.py` - Comprehensive validation
3. `scripts/test_frame_stacking_implementation.py` - Integration tests
4. `tests/test_frame_stacking_bc.py` - Unit tests
5. `VALIDATION_REPORT.md` - Detailed validation findings
6. `docs/FRAME_STACKING_PRETRAINING_VALIDATION.md` - Technical documentation
7. `docs/FRAME_STACKING_BC_TODO.md` - Implementation guide
8. `FRAME_STACKING_IMPLEMENTATION_SUMMARY.md` - This document

## Verification Checklist

- [x] BC dataset accepts frame_stack_config
- [x] Frame buffers properly initialized and managed
- [x] Visual frames stacked correctly (4, H, W, C)
- [x] Game states concatenated correctly (stack_size * dim,)
- [x] Zero padding implemented
- [x] Repeat padding implemented
- [x] PretrainingPipeline accepts and propagates frame_stack_config
- [x] BCTrainer accepts and stores frame_stack_config
- [x] Checkpoints include frame_stacking metadata
- [x] train_and_compare.py builds and passes config (multi-GPU path)
- [x] train_and_compare.py builds and passes config (single GPU path)
- [x] Unit tests created and passing
- [x] Integration tests created and passing
- [x] Validation scripts created and working
- [x] Documentation created
- [x] Backward compatibility maintained

## Next Steps

### For End-to-End Testing

1. **Generate new BC checkpoint with frame stacking**:
   ```bash
   python scripts/train_and_compare.py \
       --experiment-name "bc_with_frame_stacking" \
       --architectures mlp_baseline \
       --replay-data-dir ../nclone/bc_replays \
       --train-dataset ../nclone/datasets/train \
       --test-dataset ../nclone/datasets/test \
       --total-timesteps 100000 \
       --num-envs 4 \
       --bc-epochs 20 \
       --use-hierarchical-ppo \
       --enable-visual-frame-stacking \
       --enable-state-stacking \
       --output-dir experiments/
   ```

2. **Validate checkpoint**:
   ```bash
   python scripts/validate_checkpoint_simple.py \
       --checkpoint experiments/.../bc_best.pth \
       --architecture mlp_baseline \
       --enable-visual-frame-stacking \
       --visual-stack-size 4
   ```

3. **Monitor weight loading**:
   - Check logs during RL training
   - Verify input channel shapes match (4 vs 4)
   - Confirm weights load without shape mismatches

### For Production Use

1. Regenerate BC checkpoints with frame stacking enabled
2. Update checkpoint naming to indicate frame stacking (e.g., `bc_best_fs4.pth`)
3. Add checkpoint validation step in CI/CD pipeline
4. Monitor training logs for weight loading success

## Conclusion

The frame stacking pretraining pipeline is now fully implemented and validated. The solution:

✓ **Comprehensive**: Covers entire pipeline from data ingestion to checkpoint saving
✓ **Correct**: Implements proper frame stacking semantics (stacking vs concatenation)
✓ **Compatible**: Maintains backward compatibility with existing code
✓ **Tested**: Unit tests, integration tests, and validation scripts all passing
✓ **Documented**: Comprehensive documentation and implementation guides

The pipeline is ready for end-to-end testing with real replay data.
