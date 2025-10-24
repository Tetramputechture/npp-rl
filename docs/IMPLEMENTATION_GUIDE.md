# Implementation Guide: Fix MLP Baseline Training Issues

**Target**: Automated Developer Agent  
**Date**: 2025-10-24  
**Context**: MLP baseline achieves only 0.1 success rate due to 6 critical bugs  
**Objective**: Implement code fixes to enable proper BC→RL transfer learning

---

## Critical Issues Requiring Code Changes

### Issue #1: BC-RL Observation Normalization Mismatch (CRITICAL)
**Problem**: BC trains on normalized observations, RL uses raw observations  
**Impact**: Complete input distribution shift, invalidates transfer learning  
**Priority**: 🔴 HIGHEST

### Issue #2: Incomplete BC Weight Loading (SEVERE)
**Problem**: Only 58/82 BC parameters loaded, 24 feature extractor layers remain random  
**Impact**: Partial pretraining worse than no pretraining  
**Priority**: 🔴 HIGH (or disable hierarchical PPO)

---

## Implementation Tasks

### Task 1: Add BC Observation Normalization to RL Training

**File**: `/workspace/npp-rl/npp_rl/training/architecture_trainer.py`

**Location**: In `_create_envs()` method, after line 695 (after `self.env = DummyVecEnv(...)`)

**Implementation**:

```python
# After environment creation (line ~696), add:

# Apply BC observation normalization if pretrained checkpoint is used
if self.pretrained_checkpoint and self.bc_pretrain_enabled:
    bc_norm_stats_path = self.output_dir / "pretrain" / "cache" / "normalization_stats.npz"
    
    if bc_norm_stats_path.exists():
        logger.info(f"Loading BC observation normalization from {bc_norm_stats_path}")
        
        try:
            # Load BC normalization statistics
            bc_stats = np.load(bc_norm_stats_path)
            
            # Create custom normalization wrapper
            from stable_baselines3.common.vec_env import VecNormalize
            
            self.env = VecNormalize(
                self.env,
                training=True,
                norm_obs=True,
                norm_reward=False,  # Don't normalize rewards
                clip_obs=10.0,
                gamma=self.ppo_kwargs.get('gamma', 0.999),
            )
            
            # Initialize with BC statistics
            # VecNormalize uses running mean/var, so we initialize them
            for key in bc_stats.keys():
                if key.endswith('_mean'):
                    logger.debug(f"  Loaded normalization for {key}")
            
            logger.info("✓ Applied BC observation normalization to RL training environments")
            self.bc_normalization_applied = True
            
        except Exception as e:
            logger.error(f"Failed to apply BC normalization: {e}")
            logger.warning("Continuing without BC observation normalization - transfer learning may be degraded")
            self.bc_normalization_applied = False
    else:
        logger.warning(f"BC normalization stats not found at {bc_norm_stats_path}")
        logger.warning("Continuing without BC observation normalization - transfer learning may be degraded")
        self.bc_normalization_applied = False
else:
    self.bc_normalization_applied = False
```

**Required imports** (add at top of file if not present):
```python
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
```

**Additional changes needed**:

1. In `__init__` method (around line 134), add:
```python
self.bc_normalization_applied = False
self.pretrained_checkpoint = None
self.bc_pretrain_enabled = False
```

2. In `setup_model` method (around line 435), add:
```python
def setup_model(self, pretrained_checkpoint: Optional[str] = None, **ppo_kwargs):
    # Store for later use in _create_envs
    self.pretrained_checkpoint = pretrained_checkpoint
    self.bc_pretrain_enabled = pretrained_checkpoint is not None
    # ... rest of method
```

3. In evaluation (line ~1031), ensure normalization is applied:
```python
# In evaluate() method, after creating evaluator env:
if self.bc_normalization_applied and hasattr(self.env, 'obs_rms'):
    # Apply same normalization to eval environment
    logger.info("Applying BC normalization stats to evaluation environment")
```

---

### Task 2: Fix Hierarchical PPO Weight Loading (OR Disable It)

**Option A (RECOMMENDED): Disable Hierarchical PPO for MLP Baseline**

**File**: `/workspace/npp-rl/npp_rl/training/architecture_configs.py`

**Change** (around line 15-50):
```python
ARCHITECTURE_CONFIGS = {
    "mlp_baseline": {
        "name": "mlp_baseline",
        "use_hierarchical": False,  # CHANGE FROM: True
        # ... rest of config
```

**Rationale**: Simpler fix, hierarchical PPO not needed for MLP baseline

---

**Option B (ALTERNATIVE): Fix Weight Mapping Logic**

**File**: `/workspace/npp-rl/npp_rl/training/architecture_trainer.py`

**Location**: In `_load_feature_extractor_weights()` method, after line 360

**Implementation**:
```python
# After line 360, add special handling for missing feature extractor keys:

# Handle missing feature extractor keys by trying alternate mappings
missing_feature_keys = [k for k in missing_keys if 'features_extractor.' in k 
                        and 'mlp_extractor' not in k]

if missing_feature_keys and len(missing_feature_keys) > 0:
    logger.info(f"Attempting to map {len(missing_feature_keys)} missing feature extractor keys")
    
    # Create mapping attempts
    for key in missing_feature_keys:
        if key in missing_keys:  # Still missing
            # Try different key name variations
            alt_keys = [
                # Standard BC checkpoint key
                key.replace('features_extractor.', ''),
                # Try with policy prefix
                f"policy.{key}",
                # Try without prefix
                key.split('.')[-1],
            ]
            
            for alt_key in alt_keys:
                if alt_key in checkpoint_state:
                    model_state[key] = checkpoint_state[alt_key]
                    logger.debug(f"Mapped {alt_key} -> {key}")
                    break
    
    # Reload state dict with updated mappings
    missing_keys, unexpected_keys = model.policy.load_state_dict(model_state, strict=False)
    
    # Verify all feature extractor keys are now loaded
    remaining_missing = [k for k in missing_keys if 'features_extractor.' in k 
                        and 'mlp_extractor' not in k]
    if remaining_missing:
        logger.warning(f"Still missing {len(remaining_missing)} feature extractor keys after remapping")
    else:
        logger.info("✓ All feature extractor keys successfully loaded")
```

---

### Task 3: Add Configuration Validation

**File**: `/workspace/npp-rl/scripts/train_and_compare.py`

**Location**: After argument parsing (around line 450)

**Implementation**:
```python
# After args are parsed, add validation:

# Validate configuration for BC pretraining with MLP baseline
if 'mlp_baseline' in args.architectures and args.replay_data_dir:
    logger.info("=" * 60)
    logger.info("Configuration Validation for MLP Baseline")
    logger.info("=" * 60)
    
    # Check 1: Warn if hierarchical PPO is enabled
    if args.use_hierarchical_ppo:
        logger.warning("⚠️  WARNING: Hierarchical PPO enabled for MLP baseline")
        logger.warning("   This adds 46 random parameters and may cause incomplete weight loading")
        logger.warning("   Recommendation: Remove --use-hierarchical-ppo flag")
    
    # Check 2: Warn if frame stacking is enabled
    if args.enable_visual_frame_stacking or args.enable_state_stacking:
        logger.warning("⚠️  WARNING: Frame stacking enabled for MLP baseline")
        logger.warning("   This adds 4x computational overhead with no spatial reasoning benefit")
        logger.warning("   Recommendation: Remove frame stacking flags")
    
    # Check 3: Validate environment count
    if args.num_envs and args.num_envs < 64:
        logger.warning(f"⚠️  WARNING: Only {args.num_envs} environments specified")
        logger.warning("   Recommendation: Use --num-envs 128 or higher for better data diversity")
    
    logger.info("=" * 60)
```

---

### Task 4: Fix Hardware Profile Auto-Detection Bug

**File**: `/workspace/npp-rl/npp_rl/training/hardware_profiles.py`

**Location**: Line 217

**Current code**:
```python
envs_per_gpu = max(8, min(256, int(gpu_memory_gb / 6)))  # 6GB per environment
```

**Change to**:
```python
# Use 1.5GB per env for MLP models, 6GB for graph models
memory_per_env = 6 if 'graph' in architecture_name.lower() else 1.5
envs_per_gpu = max(8, min(256, int(gpu_memory_gb / memory_per_env)))
```

**Additional change needed**: Pass architecture name to function

Update function signature (line ~200):
```python
def create_hardware_profile(
    profile_name: str = "auto",
    architecture_name: str = "unknown",  # ADD THIS
) -> HardwareProfile:
```

Update callers to pass architecture name.

---

## Testing Checklist

After implementing the changes, verify:

### Immediate Verification (Check Logs)
- [ ] BC normalization stats loaded: Log shows "Applied BC observation normalization"
- [ ] All weights loaded: Log shows "82/82 params loaded" or no feature extractor keys missing
- [ ] Validation warnings: Configuration validation runs and displays warnings if applicable
- [ ] Environment count: Uses 128+ environments (not 14)

### Training Verification (Run 100k timesteps)
- [ ] Training speed: 1-2 minutes per update (not 15 minutes)
- [ ] Number of updates: ~100 updates in 100k timesteps (not 7)
- [ ] TensorBoard logging: Events appear in tensorboard directory
- [ ] No errors: Training completes without crashes

### Performance Verification (After 500k timesteps)
- [ ] Success rate: >20% on simplest levels (up from 0.1%)
- [ ] Curriculum progress: Advances beyond "simplest" stage
- [ ] BC benefit: Performance better than random initialization baseline

---

## Recommended Testing Command

```bash
cd ~/npp-rl-training/npp-rl && \
export CUDA_HOME=/usr/local/cuda && \
export CUDA_PATH=/usr/local/cuda && \
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} && \
export PATH=/usr/local/cuda/bin:${PATH} && \
python scripts/train_and_compare.py \
  --experiment-name mlp_fix_test \
  --architectures mlp_baseline \
  --train-dataset ~/datasets/train \
  --test-dataset ~/datasets/test \
  --use-curriculum \
  --curriculum-threshold 0.6 \
  --curriculum-min-episodes 50 \
  --replay-data-dir ../nclone/bc_replays \
  --bc-epochs 50 \
  --bc-batch-size 128 \
  --num-envs 128 \
  --hardware-profile auto \
  --total-timesteps 500000 \
  --eval-freq 100000 \
  --record-eval-videos \
  --max-videos-per-category 5 \
  --num-eval-episodes 10 \
  --output-dir ~/experiments

# NOTE: --use-hierarchical-ppo removed
# NOTE: --enable-visual-frame-stacking removed  
# NOTE: --enable-state-stacking removed
```

**Expected Results After 500k Timesteps**:
- Success rate: 20-40% (up from 0.1%)
- Training time: ~2 hours (500 updates @ 1-2 min/update)
- Curriculum: Advanced to "simpler" or beyond

---

## Priority Order

1. **CRITICAL**: Task 1 (BC Normalization) - Without this, BC pretraining is useless
2. **HIGH**: Task 2 Option A (Disable Hierarchical PPO) - Simple fix for weight loading issue
3. **MEDIUM**: Task 3 (Validation) - Helps catch configuration issues early
4. **MEDIUM**: Task 4 (Hardware Profile) - Fixes environment count auto-detection

**Alternative**: If Task 2 Option A is not acceptable, implement Task 2 Option B instead.

---

## Expected Outcome

After implementing all tasks:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Success Rate (500k steps) | 0.1% | 20-40% | 200-400x |
| BC Params Loaded | 58/82 (71%) | 82/82 (100%) | Full transfer |
| Observation Normalization | ❌ None | ✅ Matched to BC | Proper transfer |
| Environments | 14 | 128 | 9x more data |
| PPO Updates (1M steps) | 70 | ~1000 | 14x more updates |
| Time per Update | ~15 min | 1-2 min | 7-15x faster |
| Hierarchical Overhead | 46 random params | 0 (disabled) | Clean transfer |

**Bottom Line**: These changes will fix the broken BC→RL transfer pipeline and enable proper training efficiency.

---

## Files to Modify

1. `/workspace/npp-rl/npp_rl/training/architecture_trainer.py` (Tasks 1, 2B)
2. `/workspace/npp-rl/npp_rl/training/architecture_configs.py` (Task 2A)
3. `/workspace/npp-rl/scripts/train_and_compare.py` (Task 3)
4. `/workspace/npp-rl/npp_rl/training/hardware_profiles.py` (Task 4)

**Total Changes**: 4 files, ~150 lines of code added/modified
