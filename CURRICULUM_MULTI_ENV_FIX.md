# Curriculum Learning Multi-Environment Fix

## Problem Statement

When using curriculum learning with multiple training environments (`n_envs > 1`), the curriculum progression was not being tracked globally. This caused several issues:

1. **Per-Environment Tracking**: Each environment in a `SubprocVecEnv` had its own copy of the curriculum manager, leading to independent tracking of performance.
2. **Inconsistent Stage Progression**: Different environments could advance to different stages independently.
3. **Lost Performance Data**: Episode results from subprocesses weren't being aggregated properly.
4. **No Synchronization**: When stage advancement occurred, it wasn't synchronized across all environments.

## Solution Overview

The fix implements **centralized curriculum tracking** in the main process, ensuring all environments share the same curriculum state and advance together.

### Key Changes

#### 1. CurriculumEnv - Disable Local Tracking for VecEnv

**File**: `npp_rl/wrappers/curriculum_env.py`

Added `enable_local_tracking` parameter to `CurriculumEnv`:

```python
def __init__(
    self,
    env: gym.Env,
    curriculum_manager,
    check_advancement_freq: int = 10,
    enable_local_tracking: bool = True,  # NEW
):
```

When `enable_local_tracking=False`:
- Episode recording is disabled in individual environments
- Advancement checking is disabled in individual environments
- The environment only samples levels based on the synced stage

This prevents duplicate tracking in subprocess environments.

#### 2. CurriculumVecEnvWrapper - Global Tracking

**File**: `npp_rl/wrappers/curriculum_env.py`

Enhanced `CurriculumVecEnvWrapper` to be the **single source of truth**:

- **Global Episode Tracking**: Records all episode completions from all environments
- **Centralized Performance**: Maintains global performance metrics in the main process
- **Advancement Checking**: Checks for stage advancement centrally
- **Stage Synchronization**: Syncs stage changes to ALL subprocess environments using `env_method()`

Key improvements:
- Better logging with `[VecEnv]` prefix for clarity
- Tracks `total_episodes` across all environments
- Checks advancement after every N global episodes
- Immediately syncs new stage to all environments after advancement

#### 3. Architecture Trainer - Proper Configuration

**File**: `npp_rl/training/architecture_trainer.py`

Updated environment factory to disable local tracking:

```python
if use_curr and curr_mgr:
    env = CurriculumEnv(
        env,
        curr_mgr,
        check_advancement_freq=10,
        enable_local_tracking=False,  # Disabled for VecEnv
    )
```

Added clearer logging:
```python
logger.info("Wrapping environments with global curriculum tracking...")
logger.info(
    f"CurriculumVecEnvWrapper will track progression across all {num_envs} environments"
)
```

## How It Works

### Architecture with n_envs > 1

```
Main Process:
├── CurriculumManager (shared instance)
├── CurriculumVecEnvWrapper (global tracker)
│   ├── Records ALL episodes from ALL envs
│   ├── Checks advancement centrally
│   └── Syncs stage changes to subprocesses
└── VecEnv (SubprocVecEnv or DummyVecEnv)
    ├── Subprocess 0: CurriculumEnv (local_tracking=False)
    ├── Subprocess 1: CurriculumEnv (local_tracking=False)
    ├── Subprocess 2: CurriculumEnv (local_tracking=False)
    └── Subprocess 3: CurriculumEnv (local_tracking=False)
```

### Episode Completion Flow

1. **Step Execution**: All environments step in parallel
2. **Episode Completion**: When any environment completes an episode:
   - Info dict contains `is_success` and `curriculum_stage`
   - `CurriculumVecEnvWrapper.step_wait()` detects completion
3. **Global Recording**: Main process records episode in curriculum manager:
   ```python
   self.curriculum_manager.record_episode(stage, success)
   ```
4. **Advancement Check**: After every N episodes (globally):
   ```python
   if self.total_episodes >= self.last_advancement_check + self.check_advancement_freq:
       advanced = self.curriculum_manager.check_advancement()
   ```
5. **Stage Synchronization**: If advanced:
   ```python
   self._sync_curriculum_stage(new_stage)
   # Calls set_curriculum_stage() on ALL subprocess environments
   ```

### Stage Synchronization

When curriculum advances:
1. Main process updates its curriculum manager
2. `_sync_curriculum_stage()` calls `env_method("set_curriculum_stage", stage)`
3. ALL subprocess environments update their stage index
4. Future level sampling uses the new stage

## Benefits

### ✅ Global Consistency
- All environments share the same curriculum state
- Stage advancement is synchronized across ALL environments
- No environment can "drift" to a different stage

### ✅ Accurate Performance Tracking
- All episode results are tracked in one place
- Success rates reflect global performance across all environments
- Advancement decisions based on complete data

### ✅ Scalability
- Works with any number of environments (n_envs)
- Works with both `SubprocVecEnv` (multi-process) and `DummyVecEnv` (single-process)
- No performance overhead from duplicate tracking

### ✅ Clear Logging
- `[VecEnv]` prefix for global tracking messages
- Advancement logs show which stage and global episode count
- Easy to debug and monitor curriculum progression

## Testing

### Manual Testing
Run training with curriculum learning and multiple environments:

```bash
python -m npp_rl.training.train_architecture \
    --use-curriculum \
    --curriculum-starting-stage simplest \
    --curriculum-advancement-threshold 0.7 \
    --num-envs 8
```

Watch logs for:
- `[VecEnv]` messages showing global tracking
- Stage advancement messages with episode counts
- Stage synchronization confirmations

### Unit Tests
Created test files:
- `tests/test_curriculum_multi_env.py` - Comprehensive unit tests (requires full environment)
- `tests/test_curriculum_tracking_simple.py` - Simple standalone tests

## Migration Guide

### For Existing Code

If you have custom curriculum environment setup:

**Before:**
```python
env = CurriculumEnv(base_env, curriculum_manager)
vec_env = SubprocVecEnv([lambda: env for _ in range(n_envs)])
```

**After:**
```python
# Create envs with local tracking disabled
def make_env():
    base_env = create_base_env()
    return CurriculumEnv(
        base_env,
        curriculum_manager,
        enable_local_tracking=False  # Important!
    )

vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])

# Wrap with global tracker
vec_env = CurriculumVecEnvWrapper(
    vec_env,
    curriculum_manager,
    check_advancement_freq=10
)
```

### For Single Environment (n_envs=1)

No changes needed! Local tracking still works:

```python
env = CurriculumEnv(
    base_env,
    curriculum_manager,
    enable_local_tracking=True  # Default
)
```

## Implementation Notes

### Why Disable Local Tracking?

With `SubprocVecEnv`, each subprocess gets a **pickled copy** of the curriculum manager. Changes in subprocesses don't affect the main process. Therefore:

- ❌ Recording episodes in subprocesses → lost data
- ❌ Checking advancement in subprocesses → inconsistent stages
- ✅ Recording episodes in main process → centralized tracking
- ✅ Checking advancement in main process → synchronized progression

### Performance Considerations

- **No Additional Overhead**: Removing duplicate tracking actually reduces overhead
- **Efficient Stage Sync**: `env_method()` is efficient for broadcasting to subprocesses
- **Scalable**: Works with 1 to 100+ environments

### Backward Compatibility

- Default behavior unchanged for single environments
- `enable_local_tracking=True` by default maintains compatibility
- Existing code without `CurriculumVecEnvWrapper` still works (but won't have global tracking)

## Verification

To verify the fix is working:

1. **Check Logs**: Look for `[VecEnv]` prefix in curriculum messages
2. **Monitor Advancement**: All environments should advance together
3. **Verify Success Rates**: Should reflect global performance across all envs
4. **Test with Different n_envs**: Try 1, 4, 8, 16 environments

Example log output:
```
[VecEnv] Env 0 completed episode 1: stage=simplest, success=True, total_episodes=1
[VecEnv] Env 2 completed episode 1: stage=simplest, success=False, total_episodes=2
[VecEnv] Advancement check at 10 episodes: stage=simplest, success_rate=70.00%, episodes=10, can_advance=True
[VecEnv] ✨ Curriculum advanced to: simpler (syncing to all 4 environments)
[VecEnv] Successfully synced stage 'simpler' to all 4 environments
```

## Summary

This fix ensures flawless curriculum progression when using multiple environments:
- ✅ Global tracking across all environments
- ✅ Synchronized stage advancement
- ✅ Accurate performance metrics
- ✅ Works with any n_envs value
- ✅ Clear logging and debugging
- ✅ Backward compatible

The curriculum system now properly scales to multi-environment training while maintaining consistent progression across all parallel environments.
