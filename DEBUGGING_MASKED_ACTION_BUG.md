# Debugging Masked Action Bug - Comprehensive Investigation Guide

## Current Status

The bug **still occurs at step 324** despite implementing:
1. ✅ Defensive copying of action_mask arrays in all wrappers
2. ✅ Shallow copy of observation dict + explicit action_mask copy in environment cache
3. ✅ Memory ownership validation
4. ❌ **Bug persists** - Root cause not yet fully identified

## Diagnostic Logging Added

We've added comprehensive diagnostic logging that will help identify where the mask corruption occurs. All logging is controlled by debug mode and only active when `sim_config.debug = True`.

### Logging Points

1. **[OBS_GET]** - Mask immediately when fetched from `_get_observation()`
   - Location: `base_environment.py` line 319
   - Shows: Fresh mask from environment before any caching

2. **[CACHE_SET]** - Mask when cached in `_prev_obs_cache`
   - Location: `base_environment.py` line 342
   - Shows: Mask after shallow dict copy + explicit mask copy

3. **[CACHE_GET]** - Mask retrieved from cache for validation
   - Location: `base_environment.py` line 286
   - Shows: Cached mask used for action validation in next step

4. **[WRAPPER_BEFORE]** - Mask before wrapper modification
   - Location: `curriculum_env.py` line 209
   - Shows: Mask as received by wrapper

5. **[WRAPPER_AFTER]** - Mask after wrapper creates new copy
   - Location: `curriculum_env.py` line 226
   - Shows: New mask created by wrapper

6. **[MASK_CREATE]** - Mask creation in environment
   - Location: `base_environment.py` line 867
   - Shows: When mask is generated from ninja state

7. **[MASK_TRACK]** - Mask lifecycle in PPO (if enabled)
   - Location: `masked_ppo.py` various lines
   - Shows: Mask as it moves through training pipeline

## How to Enable Diagnostic Mode

### Step 1: Enable Debug Mode in Environment

```python
from nclone.gym_environment.config import EnvironmentConfig

config = EnvironmentConfig.for_training()
config.graph.debug = True  # Enables sim_config.debug
```

Or in training script:

```python
# In your environment factory or config
environment_config.graph.debug = True
```

### Step 2: Set Logging Level to DEBUG

```python
import logging

# Set root logger to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Or set specific loggers
logging.getLogger('nclone.gym_environment.base_environment').setLevel(logging.DEBUG)
logging.getLogger('npp_rl.wrappers.curriculum_env').setLevel(logging.DEBUG)
logging.getLogger('npp_rl.agents.masked_ppo').setLevel(logging.DEBUG)
```

### Step 3: Run Training and Collect Logs

```bash
python your_training_script.py 2>&1 | tee debug_output.log
```

## What to Look For in Logs

When the bug occurs at step N, look for this sequence:

### Normal Case (No Bug)
```
[OBS_GET] pid=12345 frame=N-1 action=3 fresh_mask=[1 1 1 0 0 0] mask_id=... hash=...
[CACHE_SET] pid=12345 frame=N-1 action=3 cached_mask=[1 1 1 0 0 0] mask_id=... hash=...
[WRAPPER_BEFORE] CurriculumEnv pid=12345 action=3 mask_before=[1 1 1 0 0 0] mask_id=... hash=...
[WRAPPER_AFTER] CurriculumEnv pid=12345 action=3 mask_after=[1 1 1 0 0 0] mask_id=... hash=...
[CACHE_GET] pid=12345 frame=N action=5 retrieved_mask=[1 1 1 0 0 0] mask_id=... hash=...  ← CORRECT
[OBS_GET] pid=12345 frame=N action=5 fresh_mask=[1 1 1 1 1 1] mask_id=... hash=...
```

### Bug Case (What We Need to Find)
```
[OBS_GET] pid=12345 frame=N-1 action=3 fresh_mask=[1 1 1 0 0 0] mask_id=... hash=ABC
[CACHE_SET] pid=12345 frame=N-1 action=3 cached_mask=[1 1 1 0 0 0] mask_id=... hash=ABC
[WRAPPER_BEFORE] CurriculumEnv pid=12345 action=3 mask_before=[1 1 1 0 0 0] mask_id=... hash=ABC
[WRAPPER_AFTER] CurriculumEnv pid=12345 action=3 mask_after=[1 1 1 0 0 0] mask_id=XYZ hash=DEF
[CACHE_GET] pid=12345 frame=N action=5 retrieved_mask=[1 1 1 1 1 1] mask_id=??? hash=???  ← WRONG!
```

### Key Indicators of Bug

1. **Hash mismatch**: If hash at CACHE_SET differs from hash at CACHE_GET, the cached dict was mutated
2. **Mask ID match**: If mask_id at CACHE_GET matches a wrapper's mask_id, dict wasn't properly copied
3. **Process ID (pid)**: If PIDs differ, it's a SubprocVecEnv serialization issue
4. **Frame number**: Should increment by 1 between steps

## Potential Root Causes to Investigate

Based on the fact that shallow copy + explicit mask copy still fails:

### 1. SubprocVecEnv Serialization Issue
**Hypothesis**: Observations are serialized/deserialized when passed between processes, and this somehow shares memory.

**Evidence needed**: Different PIDs in logs showing same mask_id

**Fix if confirmed**: Force copy on the subprocess boundary (in SubprocVecEnv worker)

### 2. Observation Dictionary Reuse in _get_observation()
**Hypothesis**: `_get_observation()` might be reusing the same dictionary object

**Evidence needed**: Same dict ID across multiple calls to _get_observation

**Fix if confirmed**: Force dict recreation in `_get_observation()`

### 3. Numpy View Semantics
**Hypothesis**: Dict.copy() + mask.copy() might still create views in some edge case

**Evidence needed**: `owns_data=False` in CACHE_SET logs

**Fix if confirmed**: Use `np.array(mask, copy=True, order='C')` with explicit flags

### 4. Caching at Wrong Time
**Hypothesis**: We're caching BEFORE returning, but something modifies the dict between caching and return

**Evidence needed**: CACHE_SET shows correct mask, but WRAPPER_BEFORE shows different mask

**Fix if confirmed**: Cache AFTER all environment-side processing completes

### 5. Multiple Wrapper Layers
**Hypothesis**: Position tracking wrapper or another wrapper is modifying the dict

**Evidence needed**: Check if position_tracking_wrapper modifies action_mask

**Fix if confirmed**: Add defensive copying to all wrappers

## Immediate Actions

1. **Run with debug mode enabled** to collect logs
2. **Capture logs around the failure** (steps 320-330 in your case)
3. **Compare mask IDs and hashes** across the pipeline
4. **Check if PIDs differ** (indicates subprocess boundary issue)
5. **Report findings** so we can pinpoint the exact failure point

## Temporary Workaround

If debugging shows the issue is too deep, we can disable validation temporarily:

```python
# In base_environment.py, comment out validation (NOT RECOMMENDED)
# self._validate_action_against_mask(action, prev_obs)
```

But this allows the bug to propagate silently, which is worse than catching it early.

## Expected Timeline

- **With debug logs**: Should identify root cause within 1-2 runs
- **Without logs**: Continuing to guess will waste time

**Priority: Enable debug mode and capture the logs!**

