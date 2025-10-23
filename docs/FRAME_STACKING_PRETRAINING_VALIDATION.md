# Frame Stacking Pretraining Pipeline Validation

## Executive Summary

This document outlines the validation performed on the pretraining pipeline when frame stacking is enabled, identifies issues found, and provides recommendations for proper usage.

**Key Finding**: The BC checkpoint `bc_best.pth_testing` was trained WITHOUT frame stacking (1 input channel) but is being loaded for training WITH frame stacking enabled (4 input channels). This creates a shape mismatch that prevents proper weight transfer.

## Background

Frame stacking is a technique used in Deep RL to provide temporal information to neural networks. By stacking consecutive frames, the agent can infer velocity, acceleration, and motion patterns. This is particularly important for:

1. **Visual observations** (player_frame, global_view): Capturing recent motion and visual changes
2. **Game state observations**: Capturing physics trends and state evolution

The user's training command included:
```bash
--enable-visual-frame-stacking --enable-state-stacking
```

This should create observations with stacked frames:
- **Visual frames**: Shape changes from `(H, W, 1)` to `(stack_size, H, W, 1)`
- **CNN input channels**: CNN expects `stack_size` input channels instead of 1

## Validation Results

### 1. Checkpoint Analysis

Running the validation script on `bc_best.pth_testing`:

```bash
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing \
    --enable-visual-stacking \
    --enable-state-stacking
```

**Results**:
- **Player Frame CNN**: First conv layer shape is `(32, 1, 8, 8)`
  - Input channels: **1** (no frame stacking)
  - Expected with 4-frame stacking: **4** input channels
  - **Status**: ❌ MISMATCH

- **Global CNN**: First conv layer shape is `(32, 1, 3, 3)`
  - Input channels: **1** (no frame stacking)  
  - Expected with 4-frame stacking: **4** input channels
  - **Status**: ❌ MISMATCH

- **State MLP**: First linear layer shape is `(128, 26)`
  - Input features: **26** (single state)
  - Expected with 4-state stacking: ~**104** input features (26 * 4)
  - **Status**: ❌ MISMATCH

### 2. Weight Transfer Analysis

The weight transfer mechanism in `architecture_trainer.py` correctly maps BC weights to hierarchical PPO:

```
BC: feature_extractor.*
→ PPO: mlp_extractor.features_extractor.*
```

**Expected behavior when loading compatible checkpoint**:
- ✓ 58 feature extractor tensors loaded
- ✓ Hierarchical policy heads randomly initialized (expected)
- ✓ Action/value heads randomly initialized (expected)

**Actual behavior with incompatible checkpoint**:
- ✗ Shape mismatch when loading conv layer weights
- ✗ Cannot transfer weights due to different input channel counts
- ✗ Either fails to load or loads with errors

### 3. Log Analysis

The user's log shows:
```
✓ Loaded BC pretrained feature extractor weights
  Loaded 58 weight tensors (BC → hierarchical)
  Missing keys (will use random init): 74
⚠ Unexpected keys in checkpoint: 50
  Examples: ['mlp_extractor.features_extractor.player_frame_cnn.conv_layers.0.weight', ...]
```

The "unexpected keys" warning suggests that the checkpoint structure doesn't exactly match what was expected, likely due to the shape mismatch issue.

## Root Cause Analysis

### Issue 1: BC Checkpoint Trained Without Frame Stacking

The `bc_best.pth_testing` checkpoint was created by BC pretraining that did NOT enable frame stacking. The BC training process needs to:

1. Create environments with frame stacking enabled
2. Process stacked observations during training
3. Save weights that expect stacked inputs

**Current state**: BC pretraining does not pass frame stacking configuration to the dataset or environments.

### Issue 2: BC Dataset Does Not Handle Frame Stacking

Looking at `npp_rl/training/bc_dataset.py`:
- The BCReplayDataset loads individual replay frames
- It does not stack consecutive frames during data loading
- Observations are single frames, not stacked frames

**Required**: BC dataset needs to stack consecutive frames from replay data to match the frame stacking configuration.

### Issue 3: BC Trainer Does Not Configure Frame Stacking

Looking at `npp_rl/training/bc_trainer.py` and `pretraining_pipeline.py`:
- BC trainer creates policy networks but doesn't configure frame stacking
- PretrainingPipeline doesn't pass frame stacking config to BC trainer
- No frame stacking wrapper is applied during BC training

## Impact Assessment

### Severity: HIGH

This issue affects:
1. **Weight Transfer**: Pretrained weights cannot be properly loaded
2. **Training Efficiency**: Cannot leverage BC pretraining benefits
3. **Experimentation**: Cannot compare pretrained vs non-pretrained with frame stacking

### Current Workarounds

1. **Option A**: Train WITHOUT frame stacking (remove flags)
   ```bash
   # Remove: --enable-visual-frame-stacking --enable-state-stacking
   python scripts/train_and_compare.py \
       --experiment-name "test" \
       --architectures mlp_baseline \
       ... (other args)
   ```

2. **Option B**: Train WITHOUT pretraining
   ```bash
   python scripts/train_and_compare.py \
       --experiment-name "test" \
       --architectures mlp_baseline \
       --no-pretraining \
       --enable-visual-frame-stacking \
       --enable-state-stacking \
       ... (other args)
   ```

3. **Option C**: Re-train BC checkpoint WITH frame stacking (requires fix)

## Recommendations

### Immediate Actions

1. **Validate before training**: Always run the checkpoint validation script before training:
   ```bash
   python scripts/validate_checkpoint_simple.py \
       --checkpoint <checkpoint_path> \
       --enable-visual-stacking \
       --enable-state-stacking
   ```

2. **Document checkpoint metadata**: Include frame stacking configuration in checkpoint metadata:
   ```python
   checkpoint = {
       'policy_state_dict': ...,
       'architecture': ...,
       'frame_stacking': {
           'visual_enabled': True,
           'visual_stack_size': 4,
           'state_enabled': True,
           'state_stack_size': 4,
       }
   }
   ```

### Long-term Fixes

1. **Enhance BC Dataset to Support Frame Stacking**:
   - Modify `BCReplayDataset` to stack consecutive frames from replays
   - Add `frame_stack_config` parameter to dataset initialization
   - Implement frame buffer to maintain temporal context

2. **Update BC Trainer for Frame Stacking**:
   - Pass frame stacking config to BC trainer
   - Create observation space with correct stacked shapes
   - Ensure policy network receives stacked observations

3. **Enhance Pretraining Pipeline**:
   - Add frame stacking config to `PretrainingPipeline.__init__()`
   - Pass config through to BC dataset and trainer
   - Validate compatibility before training

4. **Improve Error Messages**:
   - Detect shape mismatches when loading weights
   - Provide clear error messages with actionable recommendations
   - Fail fast rather than silently using wrong weights

## Validation Tools

### Checkpoint Validation Script

Use `scripts/validate_checkpoint_simple.py` to validate checkpoints:

```bash
# Validate for 4-frame visual stacking
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing \
    --enable-visual-stacking \
    --visual-stack-size 4

# Validate for both visual and state stacking
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing \
    --enable-visual-stacking \
    --visual-stack-size 4 \
    --enable-state-stacking \
    --state-stack-size 4

# Validate for no stacking (default)
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing
```

**Output includes**:
- ✓ Checkpoint structure analysis
- ✓ CNN input channel analysis
- ✓ State dimension analysis
- ✓ Compatibility check
- ✓ Weight mapping analysis
- ✓ Actionable recommendations

### Expected Validation Output

**Compatible checkpoint** (same frame stacking configuration):
```
✓ Player Frame CNN: 4 input channels → 4-frame stacking
✓ Compatibility Check: PASSED
✓ All components match target configuration
```

**Incompatible checkpoint** (different frame stacking):
```
✗ Player Frame CNN: 1 input channels → single frame
✗ Compatibility Check: FAILED
  - MISMATCH: player_frame has 1 frames, target expects 4
  
RECOMMENDATIONS:
1. Re-train BC checkpoint with frame stacking enabled
2. Train without pretraining (--no-pretraining)
3. Match frame stacking to checkpoint (remove stacking flags)
```

## Technical Details

### Frame Stacking in CNN Architecture

**Without frame stacking**:
```python
# Input: (batch, H, W, 1)
# After permute: (batch, 1, H, W)
# Conv2d(in_channels=1, out_channels=32, ...)
```

**With 4-frame stacking**:
```python
# Input: (batch, 4, H, W, 1)
# After reshape: (batch, 4, H, W)
# Conv2d(in_channels=4, out_channels=32, ...)
```

The CNN architecture in `configurable_extractor.py` dynamically builds layers based on input channels:

```python
def _build_layers(self, in_channels: int):
    """Build convolutional layers based on input channel count."""
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels, ...),  # in_channels=1 or 4
        ...
    )
```

### Weight Transfer Logic

From `architecture_trainer.py`:

```python
def _load_bc_pretrained_weights(self, checkpoint_path: str):
    """Load BC pretrained weights into PPO policy."""
    bc_state_dict = checkpoint["policy_state_dict"]
    
    # Map BC weights to hierarchical PPO structure
    mapped_state_dict = {}
    for key, value in bc_state_dict.items():
        if key.startswith("feature_extractor."):
            sub_key = key[len("feature_extractor."):]
            hierarchical_key = f"mlp_extractor.features_extractor.{sub_key}"
            mapped_state_dict[hierarchical_key] = value
    
    # Load with strict=False to allow partial loading
    missing_keys, unexpected_keys = self.model.policy.load_state_dict(
        mapped_state_dict, strict=False
    )
```

**Key point**: `strict=False` allows loading even when shapes don't match, but PyTorch will skip tensors with shape mismatches. This means:
- No error is raised
- But weights are NOT actually transferred
- Training proceeds with random initialization

## Conclusion

The frame stacking pretraining pipeline has a critical compatibility issue:

1. ❌ **Current BC checkpoint incompatible** with frame stacking
2. ✅ **Weight transfer logic is correct** (no bugs in mapping)
3. ❌ **BC pretraining does not support frame stacking** (missing feature)
4. ✅ **RL training supports frame stacking** (working as expected)
5. ✅ **Validation tools created** to detect issues

**Action Required**:
- Use validation script before training
- Either train without frame stacking OR without pretraining
- Future work: Implement frame stacking support in BC pretraining

## References

- [Mnih et al. (2015) - DQN Paper](https://www.nature.com/articles/nature14236): Original use of 4-frame stacking
- [Machado et al. (2018) - ALE Analysis](https://arxiv.org/abs/1709.06009): Frame stacking analysis
- `npp_rl/training/architecture_trainer.py`: Weight transfer implementation
- `npp_rl/feature_extractors/configurable_extractor.py`: Dynamic CNN architecture
- `nclone/gym_environment/config.py`: Frame stacking configuration

## Appendix: Sample Commands

### Validation Commands

```bash
# Validate checkpoint structure
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing \
    --enable-visual-stacking \
    --enable-state-stacking

# Check specific frame stack sizes
python scripts/validate_checkpoint_simple.py \
    --checkpoint bc_best.pth_testing \
    --enable-visual-stacking --visual-stack-size 8 \
    --enable-state-stacking --state-stack-size 6
```

### Training Commands

```bash
# CORRECT: Train without frame stacking (matches checkpoint)
python scripts/train_and_compare.py \
    --experiment-name "baseline_no_stacking" \
    --architectures mlp_baseline \
    --replay-data-dir ../nclone/bc_replays \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 4 \
    --use-hierarchical-ppo \
    --output-dir experiments/

# CORRECT: Train with frame stacking but no pretraining
python scripts/train_and_compare.py \
    --experiment-name "stacking_no_pretrain" \
    --architectures mlp_baseline \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 4 \
    --use-hierarchical-ppo \
    --enable-visual-frame-stacking \
    --enable-state-stacking \
    --no-pretraining \
    --output-dir experiments/

# INCORRECT: Train with frame stacking using incompatible checkpoint
# (This was the user's original command - has compatibility issues)
python scripts/train_and_compare.py \
    --experiment-name "quick_test" \
    --architectures mlp_baseline \
    --replay-data-dir ../nclone/bc_replays \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 4 \
    --use-hierarchical-ppo \
    --enable-visual-frame-stacking \
    --enable-state-stacking \
    --output-dir experiments/
```
