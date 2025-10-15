# Architecture System Cleanup Notes

**Date:** 2025-10-15  
**Status:** Documentation completed, legacy extractors retained for compatibility

---

## Summary

After completing the architecture validation audit, this document provides guidance on the current state of the codebase and recommendations for future cleanup.

## Current State

### ✅ Production-Ready Architecture System

The **ConfigurableMultimodalExtractor** in `npp_rl/optimization/` is now the recommended system for all new training:
- 8 validated architectures ready for use
- Comprehensive test coverage (15/15 tests pass)
- Used by `ArchitectureTrainer` for systematic comparison
- All components properly integrated and tested

### 📦 Legacy Extractors (Retained for Compatibility)

Three legacy extractors in `npp_rl/feature_extractors/` are maintained for backward compatibility:

#### 1. HGTMultimodalExtractor (`hgt_multimodal.py`)
- **Status:** LEGACY - Superseded by ConfigurableMultimodalExtractor
- **Can be replaced with:** `get_architecture_config("full_hgt")`
- **Still used by:**
  - `train_hierarchical_stable.py` (line 57, 179)
  - `npp_rl/agents/training.py` (line 44, 416)
  - `npp_rl/training/training_utils.py` (line 13, 39)
- **Recommendation:** Update these files to use ConfigurableMultimodalExtractor, then deprecate

#### 2. VisionFreeExtractor (`vision_free_extractor.py`)
- **Status:** SPECIAL PURPOSE - Different from "vision_free" architecture config
- **Purpose:** For environments that provide `entity_positions` instead of `graph_obs`
- **Used by:** `npp_rl/agents/training.py` (line 44, 403) with `--extractor_type vision_free`
- **Note:** This is NOT the same as the "vision_free" architecture config
  - VisionFreeExtractor: Uses simple MLPs on entity_positions, reachability_features
  - vision_free config: Uses graph processing but no visual modalities
- **Recommendation:** Keep for CPU training and rapid prototyping use cases

#### 3. MinimalStateExtractor (`vision_free_extractor.py`)
- **Status:** SPECIAL PURPOSE - Minimal state-only extractor
- **Purpose:** Fastest option for debugging and CPU training
- **Used by:** `npp_rl/agents/training.py` (line 44, 410) with `--extractor_type minimal`
- **Recommendation:** Keep for debugging and baseline comparisons

### 📋 Documentation References

The following files contain documentation referencing legacy HGT implementations:
- `npp_rl/hrl/subtask_policies.py` (line mentioning HGTMultimodalExtractor)
- `npp_rl/agents/hierarchical_ppo.py` (documentation)
- `npp_rl/models/hierarchical_policy.py` (documentation)

**Recommendation:** Update documentation to reference ConfigurableMultimodalExtractor

---

## Migration Path (Future Work)

### Phase 1: Update Primary Training Scripts

**Priority:** HIGH  
**Effort:** Medium

1. **Update `npp_rl/agents/training.py`:**
   ```python
   # OLD (lines 400-425):
   if extractor_type == "vision_free":
       extractor_class = VisionFreeExtractor
   else:
       extractor_class = HGTMultimodalExtractor
   
   # NEW:
   from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
   from npp_rl.optimization.architecture_configs import get_architecture_config
   
   if extractor_type == "vision_free":
       extractor_class = VisionFreeExtractor  # Keep for special purpose
   elif extractor_type == "minimal":
       extractor_class = MinimalStateExtractor  # Keep for special purpose
   else:
       # Use configurable system
       config = get_architecture_config("full_hgt")
       extractor_class = ConfigurableMultimodalExtractor
       extractor_kwargs = {"config": config}
   ```

2. **Update `train_hierarchical_stable.py`:**
   ```python
   # OLD (lines 57, 179):
   from npp_rl.feature_extractors import HGTMultimodalExtractor
   policy_kwargs = {'features_extractor_class': HGTMultimodalExtractor, ...}
   
   # NEW:
   from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
   from npp_rl.optimization.architecture_configs import get_architecture_config
   
   config = get_architecture_config("full_hgt")
   policy_kwargs = {
       'features_extractor_class': ConfigurableMultimodalExtractor,
       'features_extractor_kwargs': {'config': config},
       ...
   }
   ```

3. **Update `npp_rl/training/training_utils.py`:**
   ```python
   # OLD (line 39):
   features_extractor = HGTMultimodalExtractor(observation_space, features_dim)
   
   # NEW:
   from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
   from npp_rl.optimization.architecture_configs import get_architecture_config
   
   config = get_architecture_config("full_hgt")
   features_extractor = ConfigurableMultimodalExtractor(observation_space, config)
   ```

### Phase 2: Add Deprecation Warnings

**Priority:** MEDIUM  
**Effort:** Low

Add deprecation warnings to legacy extractors:

```python
# In hgt_multimodal.py
import warnings

class HGTMultimodalExtractor(BaseFeaturesExtractor):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "HGTMultimodalExtractor is deprecated. "
            "Use ConfigurableMultimodalExtractor with 'full_hgt' config instead. "
            "See npp_rl.optimization.architecture_configs for details.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

### Phase 3: Remove After Migration

**Priority:** LOW  
**Effort:** Low  
**Timing:** After all references are updated and tested

Once all scripts are migrated:
1. Remove `npp_rl/feature_extractors/hgt_multimodal.py`
2. Remove HGTMultimodalExtractor from `__init__.py`
3. Keep VisionFreeExtractor and MinimalStateExtractor for special purposes

---

## Architecture Selection Guide

### For New Projects (RECOMMENDED)

Use **ArchitectureTrainer** with one of the 8 validated configs:

```python
from npp_rl.training.architecture_trainer import ArchitectureTrainer

# For full multimodal training
trainer = ArchitectureTrainer(config_name="full_hgt", env_id="NPP-v0")

# For faster training with simpler graphs
trainer = ArchitectureTrainer(config_name="gat", env_id="NPP-v0")

# For baseline without graphs
trainer = ArchitectureTrainer(config_name="mlp_baseline", env_id="NPP-v0")

trainer.train(total_timesteps=10_000_000)
```

### For CPU Training / Rapid Prototyping

Use **VisionFreeExtractor** or **MinimalStateExtractor**:

```python
# CPU training without graph processing
python -m npp_rl.agents.training --extractor_type vision_free --num_envs 4

# Minimal state-only for fastest debugging
python -m npp_rl.agents.training --extractor_type minimal --num_envs 1
```

### For Hierarchical RL (Current Implementation)

Currently uses `HGTMultimodalExtractor` in:
- `train_hierarchical_stable.py`
- `npp_rl/agents/hierarchical_ppo.py`

**Future:** Migrate to ConfigurableMultimodalExtractor once hierarchical system is updated

---

## Verification Checklist

Before removing legacy extractors:

- [ ] All references to HGTMultimodalExtractor updated to ConfigurableMultimodalExtractor
- [ ] Training scripts tested with new extractors
- [ ] Hierarchical RL system updated (if applicable)
- [ ] Behavioral cloning scripts updated
- [ ] Documentation updated
- [ ] Deprecation warnings added
- [ ] Full test suite passes
- [ ] Training runs produce comparable results

---

## Benefits of Migration

### Immediate Benefits
- ✅ Unified architecture system for systematic comparison
- ✅ 8 validated architectures ready to use
- ✅ Comprehensive test coverage
- ✅ Clearer code organization

### Long-term Benefits
- 📦 Reduced maintenance burden (single system vs. multiple extractors)
- 🔬 Easier to add new architectures (just add config)
- 📊 Better for research (systematic architecture comparison)
- 🧹 Cleaner codebase

---

## Notes

### Why Keep VisionFreeExtractor and MinimalStateExtractor?

These serve genuinely different purposes:
- **Different observation spaces:** They work with environments that don't provide graph observations
- **CPU training:** No graph processing overhead for rapid iteration
- **Debugging:** Minimal architecture for isolating issues
- **Baselines:** Simple architectures for performance comparison

### Why Not Remove HGTMultimodalExtractor Immediately?

- **Backward compatibility:** Used in existing training scripts
- **Hierarchical RL:** Integrated into hierarchical PPO system
- **Testing required:** Need to verify ConfigurableMultimodalExtractor produces equivalent results
- **Gradual migration:** Safer to migrate one script at a time

---

## Conclusion

The architecture system is now production-ready with clear migration paths. Legacy extractors are documented and can be phased out gradually as scripts are updated.

**Recommended Action:** Use ConfigurableMultimodalExtractor for all new training experiments. Update existing scripts as time permits.
