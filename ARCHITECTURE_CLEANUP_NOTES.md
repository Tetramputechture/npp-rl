# Architecture System Cleanup - Aggressive Consolidation

**Date:** 2025-10-15  
**Status:** Aggressive cleanup completed - all legacy extractors removed

---

## Summary

Following the architecture validation audit, an aggressive cleanup was performed to eliminate all redundant and untested code. The codebase now uses a single, unified architecture system with all legacy extractors removed.

## What Changed

### ✅ Unified Architecture System (ONLY system now)

The **ConfigurableMultimodalExtractor** in `npp_rl/optimization/` is now the ONLY feature extractor:
- 8 validated architectures covering all use cases
- Full test coverage (167/167 tests pass including architecture tests)
- Used by ALL training scripts
- All components properly integrated and tested

### 🗑️ Removed Legacy Extractors

All legacy extractors have been **completely removed**:

#### Removed Files:
1. ❌ `npp_rl/feature_extractors/hgt_multimodal.py` - **REMOVED**
   - Replaced by: `get_architecture_config("full_hgt")`
   
2. ❌ `npp_rl/feature_extractors/vision_free_extractor.py` - **REMOVED**
   - Replaced by: `get_architecture_config("vision_free")` or `mlp_baseline`
   
3. ❌ `npp_rl/feature_extractors/minimal_state_extractor.py` - **REMOVED**
   - Replaced by: `get_architecture_config("mlp_baseline")`

#### Why These Were Removed:
- **No test coverage:** None of these extractors had any tests
- **Redundant:** ConfigurableMultimodalExtractor provides all their functionality
- **Untested assumptions:** "Special purpose" claims were not validated
- **Maintenance burden:** Multiple implementations of the same concepts

### 📝 Updated Files

All references to legacy extractors have been migrated:

#### 1. `npp_rl/agents/training.py`
- **Before:** Used `HGTMultimodalExtractor`, `VisionFreeExtractor`, `MinimalStateExtractor`
- **After:** Uses `ConfigurableMultimodalExtractor` with architecture configs
- **New feature:** Added `--architecture` flag for direct architecture selection
- **Legacy support:** `--extractor_type` still works, maps to architecture configs

#### 2. `train_hierarchical_stable.py`
- **Before:** Used `HGTMultimodalExtractor` directly
- **After:** Uses `ConfigurableMultimodalExtractor` with `"full_hgt"` config

#### 3. `npp_rl/training/training_utils.py`
- **Before:** Imported and used `HGTMultimodalExtractor`
- **After:** Uses `ConfigurableMultimodalExtractor` with `"full_hgt"` config

#### 4. `npp_rl/feature_extractors/__init__.py`
- **Before:** Exported 3 legacy extractors with usage examples
- **After:** Documents removal and provides migration guide to ConfigurableMultimodalExtractor

#### 5. `tests/training/test_architecture_trainer.py`
- **Before:** Mocked non-existent `create_graph_enhanced_env` function
- **After:** Correctly mocks `NppEnvironment` class

---

## How to Use the Unified System

### Method 1: Using npp_rl/agents/training.py (Recommended)

```bash
# Full HGT architecture (recommended for best performance)
python -m npp_rl.agents.training --architecture full_hgt --num_envs 64 --total_timesteps 10000000

# Simplified HGT (faster, still very capable)
python -m npp_rl.agents.training --architecture simplified_hgt --num_envs 32

# GAT (Graph Attention Networks)
python -m npp_rl.agents.training --architecture gat

# GCN (Graph Convolutional Networks)
python -m npp_rl.agents.training --architecture gcn

# MLP Baseline (no graph processing)
python -m npp_rl.agents.training --architecture mlp_baseline

# Vision-free (no visual processing, graph + state only)
python -m npp_rl.agents.training --architecture vision_free

# No global view (temporal + graph + state, no global frame)
python -m npp_rl.agents.training --architecture no_global_view

# Local frames only (same as no_global_view)
python -m npp_rl.agents.training --architecture local_frames_only

# Legacy flag support (maps to architecture configs)
python -m npp_rl.agents.training --extractor_type hgt  # → full_hgt
python -m npp_rl.agents.training --extractor_type vision_free  # → vision_free
python -m npp_rl.agents.training --extractor_type minimal  # → mlp_baseline
```

### Method 2: Using ArchitectureTrainer (For Systematic Comparison)

```python
from npp_rl.training.architecture_trainer import ArchitectureTrainer

# Create trainer for specific architecture
trainer = ArchitectureTrainer(config_name="full_hgt", env_id="NPP-v0")

# Train with custom parameters
trainer.train(
    total_timesteps=10_000_000,
    n_envs=64,
    save_freq=100_000,
    eval_freq=50_000
)

# Results saved to ./architecture_comparison_results/full_hgt/
```

### Method 3: Direct Usage in Custom Scripts

```python
from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
from npp_rl.optimization.architecture_configs import get_architecture_config
from stable_baselines3 import PPO

# Get architecture configuration
config = get_architecture_config("full_hgt")

# Create policy kwargs
policy_kwargs = {
    'features_extractor_class': ConfigurableMultimodalExtractor,
    'features_extractor_kwargs': {'config': config},
    'net_arch': {'pi': [256, 256], 'vf': [256, 256]},
}

# Create PPO model
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    # ... other PPO params
)

model.learn(total_timesteps=10_000_000)
```

### Architecture Selection Guide

| Architecture | Use Case | Speed | Performance |
|-------------|----------|-------|-------------|
| `full_hgt` | **Production training** | Slow | Best |
| `simplified_hgt` | Faster HGT variant | Medium | Very Good |
| `gat` | Graph attention, lighter than HGT | Medium-Fast | Good |
| `gcn` | Simple graph convolution | Fast | Good |
| `mlp_baseline` | **Baseline comparison** | Fastest | Fair |
| `vision_free` | No visual processing (CPU friendly) | Fast | Good (for non-visual) |
| `no_global_view` | Temporal + graph only | Medium-Fast | Good |
| `local_frames_only` | Same as no_global_view | Medium-Fast | Good |

---

## Verification Checklist

All completed ✅:

- [x] All references to HGTMultimodalExtractor updated to ConfigurableMultimodalExtractor
- [x] All references to VisionFreeExtractor migrated to architecture configs
- [x] All references to MinimalStateExtractor migrated to architecture configs
- [x] Training scripts tested with new extractors
- [x] Hierarchical RL system updated (train_hierarchical_stable.py)
- [x] Training utilities updated (npp_rl/training/training_utils.py)
- [x] Documentation updated (feature_extractors/__init__.py)
- [x] Legacy files removed
- [x] Full test suite passes (167/167 tests)
- [x] Test mocking issues fixed

---

## Benefits Realized

### Immediate Benefits ✅
- Single unified architecture system - no confusion about which extractor to use
- 8 validated architectures with comprehensive test coverage
- All training scripts use the same system
- Cleaner, more maintainable codebase

### Technical Benefits ✅
- Reduced code duplication
- Consistent interface across all architectures
- Easier to add new architectures (just add config, no new class)
- Better for systematic architecture comparison

### Developer Experience ✅
- Clear usage documentation
- Backward compatibility via flag mapping
- No deprecated code warnings
- Single source of truth for feature extraction

---

## Test Results

All tests passing after cleanup:

```
167 tests passed
  - 15 architecture integration tests (all 8 architectures validated)
  - 3 architecture trainer tests (fixed mocking issues)
  - 149 other tests (hierarchical RL, HRL, models, etc.)
```

Key architecture tests:
- ✅ All 8 architectures instantiate correctly
- ✅ All 8 architectures complete forward passes
- ✅ Batch size variations work (1, 4, 16)
- ✅ Output consistency validated
- ✅ Configuration validation tests pass

---

## Migration Summary

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| **Extractors** | 3 separate classes | 1 configurable class | ✅ Complete |
| **Training scripts** | Mixed usage | Unified ConfigurableMultimodalExtractor | ✅ Complete |
| **Tests** | Some mocking issues | All 167 passing | ✅ Complete |
| **Documentation** | Scattered | Centralized in this file | ✅ Complete |
| **Code maintenance** | High (3 implementations) | Low (1 implementation) | ✅ Improved |

---

## Conclusion

**Aggressive cleanup completed successfully.** The codebase now has:

1. **Single feature extraction system** - ConfigurableMultimodalExtractor
2. **8 validated architectures** - covering all use cases from baseline to full HGT
3. **Zero legacy code** - all redundant extractors removed
4. **Full test coverage** - 167/167 tests passing
5. **Clear documentation** - usage examples and migration guides

**All training scripts now use the unified system. No deprecated code remains.**
