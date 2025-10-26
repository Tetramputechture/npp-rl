# Multi-Environment Curriculum Analysis

## Executive Summary

After thorough analysis of the curriculum manager code for multi-environment support (n_envs > 1), **ONE CRITICAL BUG** has been identified that prevents adaptive mixing from working correctly with subprocess-based vectorized environments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│     CurriculumVecEnvWrapper (Main Process)              │
│  - Master curriculum_manager with all episode data      │
│  - Global episode tracking and advancement              │
│  - Calls env_method() to sync stage to subprocesses     │
└─────────────────────────────────────────────────────────┘
                          │
                          │ env_method("set_curriculum_stage", stage)
                          │ env_method("set_adaptive_mixing_ratio", stage, ratio)  [NEEDED]
                          ▼
    ┌─────────────────────────────────────────────────────┐
    │  SubprocVecEnv / DummyVecEnv                        │
    └─────────────────────────────────────────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      ▼                   ▼                   ▼
┌───────────┐       ┌───────────┐       ┌───────────┐
│ Subprocess│       │ Subprocess│       │ Subprocess│
│ Env 0     │       │ Env 1     │       │ Env n-1   │
│           │       │           │       │           │
│ PICKLED   │       │ PICKLED   │       │ PICKLED   │
│ COPY of   │       │ COPY of   │       │ COPY of   │
│ curriculum│       │ curriculum│       │ curriculum│
│ _manager  │       │ _manager  │       │ _manager  │
└───────────┘       └───────────┘       └───────────┘
```

### Key Insight: Subprocess Isolation
When using `SubprocVecEnv`, Python pickles objects when sending them to subprocesses. Each subprocess receives a **COPY** of the curriculum_manager, not a reference to the original. This means:
- ✅ Each subprocess can independently call methods on its copy
- ❌ Updates to state in one subprocess don't affect others
- ❌ Updates to state in subprocess don't affect main process
- ❌ Subprocess copies don't receive updates from main process automatically

## Critical Bug: Adaptive Mixing with Stale Data

### The Problem

**File**: `npp_rl/training/curriculum_manager.py`  
**Method**: `_get_adaptive_mixing_ratio()` (lines 207-240)

```python
def _get_adaptive_mixing_ratio(self, stage: str) -> float:
    # Get current performance
    success_rate = self.get_stage_success_rate(stage)  # ⚠️ READS FROM self.stage_performance
    
    # Calculate adaptive ratio based on success_rate
    if success_rate < 0.50:
        adaptive_ratio = 0.40
    # ... etc
```

**Call Chain**:
1. Subprocess env calls `reset()`
2. → `curriculum_manager.sample_level()`
3. → `_get_adaptive_mixing_ratio(current_stage)`
4. → `get_stage_success_rate(stage)` - **reads from `self.stage_performance`**

**The Issue**:
- `self.stage_performance` in subprocess is a **PICKLED COPY** from initialization
- It contains performance data from when the subprocess was created
- New episodes are recorded in the **MAIN PROCESS** curriculum_manager
- Subprocess never receives these updates
- **Result**: Adaptive mixing uses STALE performance data (potentially empty or very old)

### Impact

**Severity**: 🔴 **CRITICAL** for adaptive mixing feature

When using `n_envs > 1` with `SubprocVecEnv`:
- Adaptive mixing ratios are calculated from stale/empty performance data
- Subprocesses may use wrong mixing ratios (e.g., 40% when should be 5%)
- This defeats the purpose of adaptive mixing
- Agent may receive too much easy content (over-mixing) or too much hard content (under-mixing)

**Note**: This bug does NOT affect:
- Stage-specific thresholds (checked in main process)
- Adaptive minimum episodes (checked in main process)
- Early advancement (checked in main process)
- Trend analysis (checked in main process)
- DummyVecEnv (runs in main process, shares curriculum_manager reference)

It ONLY affects adaptive mixing ratio calculation in subprocess environments.

## Other Potential Issues Analyzed

### ✅ Issue 2: Stage Synchronization (CORRECT)

**Status**: No bug found - working as designed

When curriculum advances:
1. Main process detects advancement
2. Calls `env_method("set_curriculum_stage", new_stage)`
3. Each subprocess's `CurriculumEnv.set_curriculum_stage()` executes
4. Updates that subprocess's `curriculum_manager.current_stage_idx`
5. Next `sample_level()` uses new stage

**Potential concern**: What if env resets between advancement and sync?
- **Answer**: `env_method()` is synchronous - waits for all envs to complete
- Next reset will use the updated stage

**Potential concern**: Episodes in progress when stage changes?
- **Answer**: In-progress episodes keep their original stage (level already loaded)
- Only NEW episodes (after reset) use new stage
- This is correct behavior

### ✅ Issue 3: Episode Tracking (CORRECT)

**Status**: No bug found - working as designed

Episode tracking flow:
1. Subprocess env completes episode
2. Returns `done=True` with `info["curriculum_stage"]`
3. VecEnvWrapper's `step_wait()` receives completion
4. Records episode in **main process** curriculum_manager
5. Stage info comes from subprocess, but recording happens in main process

**Design correctness**:
- ✅ All episodes recorded centrally in main process
- ✅ No duplicate recording (local tracking disabled in subprocesses)
- ✅ Stage info preserved from when level was loaded

### ✅ Issue 4: Local Tracking Disabled (CORRECT)

**Status**: No bug found - proper design pattern

When using VecEnvWrapper:
- Individual `CurriculumEnv` instances have `enable_local_tracking=False`
- This disables `record_episode()` and `check_advancement()` in subprocesses
- Only the main process VecEnvWrapper tracks and checks advancement

**Correctness verified**:
- ✅ No duplicate episode recording
- ✅ No conflicting advancement checks
- ✅ Single source of truth (main process)

### ✅ Issue 5: Advancement Frequency (CORRECT)

**Status**: No bug found - working as designed

```python
if (episodes_completed_this_step > 0 
    and self.total_episodes >= self.last_advancement_check + self.check_advancement_freq):
```

**Analysis**:
- Checks advancement every N **total** episodes across all envs
- If multiple envs complete in same step, still only checks once
- This is correct - we want to check after N episodes total, not per completion
- Prevents redundant checks in the same step

### ⚠️ Issue 6: Stage-Specific Thresholds with Global Override (DESIGN CONSIDERATION)

**Status**: Design works as intended, but documentation needed

In curriculum_manager.py:
```python
def __init__(self, advancement_threshold=None, ...):
    if advancement_threshold is not None:
        # Global override - all stages use this threshold
        self.stage_thresholds = {stage: advancement_threshold for stage in self.CURRICULUM_ORDER}
    else:
        # Use stage-specific thresholds
        self.stage_thresholds = self.STAGE_THRESHOLDS.copy()
```

**Observation**:
- If user passes `advancement_threshold`, it overrides ALL stage-specific thresholds
- This is correct for backwards compatibility
- But users might expect to override just one stage

**Recommendation**: Document this behavior clearly in docstring

## Summary of Findings

| Issue | Component | Severity | Status | Fix Needed |
|-------|-----------|----------|--------|------------|
| Adaptive mixing with stale data | sample_level() in subprocess | 🔴 CRITICAL | BUG | YES |
| Stage synchronization | set_curriculum_stage() | ✅ OK | CORRECT | NO |
| Episode tracking | VecEnvWrapper.step_wait() | ✅ OK | CORRECT | NO |
| Local tracking disabled | enable_local_tracking | ✅ OK | CORRECT | NO |
| Advancement frequency | check_advancement() timing | ✅ OK | CORRECT | NO |
| Global threshold override | __init__ parameter | ⚠️ DESIGN | DOC NEEDED | CLARIFY DOC |

## Recommended Fixes

### Fix 1: Sync Adaptive Mixing Ratio from Main Process

**Approach**: When syncing curriculum stage, also sync the current mixing ratio.

**Implementation**:

#### A. Add method to CurriculumEnv to set mixing ratio:

```python
# In npp_rl/wrappers/curriculum_env.py, CurriculumEnv class

def set_adaptive_mixing_ratio(self, stage: str, ratio: float):
    """Set the adaptive mixing ratio for a stage.
    
    Called by VecEnvWrapper to sync mixing ratios from main process.
    
    Args:
        stage: Stage name
        ratio: Mixing ratio (0.0 to 1.0)
    """
    if hasattr(self.curriculum_manager, 'stage_mixing_ratios'):
        self.curriculum_manager.stage_mixing_ratios[stage] = ratio
        logger.debug(f"Mixing ratio for stage '{stage}' set to {ratio:.1%}")
```

#### B. Modify _sync_curriculum_stage to also sync mixing ratio:

```python
# In npp_rl/wrappers/curriculum_env.py, CurriculumVecEnvWrapper class

def _sync_curriculum_stage(self, stage: str):
    """Synchronize curriculum stage and adaptive mixing ratio to all subprocess environments."""
    try:
        if hasattr(self.venv, "env_method"):
            # Sync stage
            self.venv.env_method("set_curriculum_stage", stage)
            
            # Calculate and sync adaptive mixing ratio from main process
            if self.curriculum_manager.enable_adaptive_mixing:
                mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(stage)
                self.venv.env_method("set_adaptive_mixing_ratio", stage, mixing_ratio)
                logger.info(
                    f"[VecEnv] Synced stage '{stage}' (mixing: {mixing_ratio:.1%}) "
                    f"to all {self.num_envs} environments"
                )
            else:
                logger.info(
                    f"[VecEnv] Synced stage '{stage}' to all {self.num_envs} environments"
                )
        else:
            logger.warning(
                "[VecEnv] VecEnv does not support env_method, cannot sync curriculum stage"
            )
    except Exception as e:
        logger.error(f"[VecEnv] Failed to sync curriculum: {e}", exc_info=True)
```

#### C. Modify _get_adaptive_mixing_ratio to use cached value in subprocesses:

```python
# In npp_rl/training/curriculum_manager.py

def _get_adaptive_mixing_ratio(self, stage: str) -> float:
    """Get adaptive mixing ratio for a stage based on current performance.
    
    For subprocess environments, this returns the last synced ratio.
    For main process, this calculates based on current performance.
    
    Args:
        stage: Stage name
        
    Returns:
        Adaptive mixing ratio (0.0 to 1.0)
    """
    if not self.enable_adaptive_mixing:
        return self.base_mixing_ratio
    
    # If we have a cached ratio and minimal performance data,
    # we're likely in a subprocess - use cached value
    if stage in self.stage_mixing_ratios and len(self.stage_performance.get(stage, [])) < 5:
        # Use cached ratio (synced from main process)
        return self.stage_mixing_ratios[stage]
    
    # Calculate fresh ratio (main process path)
    success_rate = self.get_stage_success_rate(stage)
    
    if success_rate < 0.50:
        adaptive_ratio = 0.40
    elif success_rate < 0.65:
        adaptive_ratio = 0.25
    elif success_rate < 0.80:
        adaptive_ratio = 0.15
    else:
        adaptive_ratio = 0.05
    
    # Cache for future calls
    self.stage_mixing_ratios[stage] = adaptive_ratio
    
    return adaptive_ratio
```

**Alternative simpler approach**: Always use cached ratio if available, fall back to calculation:

```python
def _get_adaptive_mixing_ratio(self, stage: str) -> float:
    """Get adaptive mixing ratio for a stage based on current performance."""
    if not self.enable_adaptive_mixing:
        return self.base_mixing_ratio
    
    # Return cached ratio if available (synced from main process in multi-env setup)
    if stage in self.stage_mixing_ratios:
        return self.stage_mixing_ratios[stage]
    
    # Calculate fresh ratio (first time or single-env setup)
    success_rate = self.get_stage_success_rate(stage)
    
    if success_rate < 0.50:
        adaptive_ratio = 0.40
    elif success_rate < 0.65:
        adaptive_ratio = 0.25
    elif success_rate < 0.80:
        adaptive_ratio = 0.15
    else:
        adaptive_ratio = 0.05
    
    self.stage_mixing_ratios[stage] = adaptive_ratio
    return adaptive_ratio
```

### Fix 2: Add Periodic Mixing Ratio Sync

Since mixing ratios should adapt during training (not just at stage changes), add periodic syncing:

```python
# In CurriculumVecEnvWrapper.step_wait(), after advancement check:

# Periodically sync mixing ratios even if no advancement
# This ensures subprocesses have current adaptive ratios
if self.total_episodes % 50 == 0:  # Every 50 episodes
    self._sync_mixing_ratios()

def _sync_mixing_ratios(self):
    """Sync current adaptive mixing ratios to all subprocesses."""
    if not self.curriculum_manager.enable_adaptive_mixing:
        return
    
    current_stage = self.curriculum_manager.get_current_stage()
    
    try:
        mixing_ratio = self.curriculum_manager._get_adaptive_mixing_ratio(current_stage)
        if hasattr(self.venv, "env_method"):
            self.venv.env_method("set_adaptive_mixing_ratio", current_stage, mixing_ratio)
            logger.debug(
                f"[VecEnv] Synced mixing ratio for '{current_stage}': {mixing_ratio:.1%}"
            )
    except Exception as e:
        logger.error(f"[VecEnv] Failed to sync mixing ratios: {e}", exc_info=True)
```

### Fix 3: Improve Documentation

Add clear warning in GRANULAR_CURRICULUM_PROGRESSION.md about the subprocess behavior and how the fix works.

## Testing Recommendations

### Test 1: Verify Mixing Ratio Sync

```python
def test_mixing_ratio_sync_multi_env():
    """Test that adaptive mixing ratios are synced to subprocesses."""
    curriculum_manager = CurriculumManager(
        dataset_path="data/test_suite",
        enable_adaptive_mixing=True,
    )
    
    # Record episodes to establish performance (low success rate)
    for i in range(30):
        curriculum_manager.record_episode("simplest", i < 12)  # 40% success
    
    # Get mixing ratio in main process
    main_ratio = curriculum_manager._get_adaptive_mixing_ratio("simplest")
    assert main_ratio == 0.40, "Main process should calculate 40% mixing for 40% success"
    
    # Create vectorized envs
    def make_env():
        env = MockNppEnvironment()
        return CurriculumEnv(env, curriculum_manager, enable_local_tracking=False)
    
    venv = DummyVecEnv([make_env for _ in range(4)])
    venv = CurriculumVecEnvWrapper(venv, curriculum_manager)
    
    # After initialization, check that ratio was synced
    # Sample from subprocess env and verify mixing ratio is used
    obs = venv.reset()
    
    # Check that all subprocess envs have the correct mixing ratio
    ratios = venv.env_method("get_mixing_ratio", "simplest")  # Need to add this method
    assert all(r == 0.40 for r in ratios), "All subprocesses should have synced ratio"
```

### Test 2: Verify Mixing Ratio Adapts During Training

```python
def test_adaptive_mixing_during_training():
    """Test that mixing ratio updates as performance improves."""
    # Start with low performance -> high mixing
    # Train to high performance -> low mixing
    # Verify that subprocess envs reflect the change
```

## Conclusion

**Status**: One critical bug identified in adaptive mixing with multi-environment support.

**Recommended Action**:
1. ✅ Implement Fix 1 (sync mixing ratios from main process)
2. ✅ Implement Fix 2 (periodic mixing ratio sync)
3. ✅ Add tests to verify fixes
4. ✅ Update documentation

**After fixes**, the multi-environment curriculum system will work correctly with all granular features:
- ✅ Stage-specific thresholds
- ✅ Adaptive minimum episodes
- ✅ Early advancement
- ✅ Trend analysis
- ✅ **Adaptive mixing (after fix)**
- ✅ Global progression tracking
- ✅ Synchronized stage advancement
