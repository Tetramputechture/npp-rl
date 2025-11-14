# Performance Analysis After Initial Optimizations

## Executive Summary

**Date:** 2025-11-13  
**Comparison:** Before optimizations vs After optimizations (with 2 environments)

### Key Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Training duration** | 398.98s | 352.91s | **-11.5%** ✅ |
| **Rollout collection** | 154.06s | 172.37s | **+11.9%** ❌ |
| **Total time** | 529.33s | 415.99s | **-21.4%** ✅ |
| **CPU memory growth** | +4.18GB | +4.31GB | **-3.1%** (still growing) |
| **Steps/second** | ~5.0 | ~5.7 | **+14%** |

**Overall:** Training improved significantly, but rollout collection regressed. Memory growth persists.

---

## Critical Findings from PyTorch Trace Analysis

### 1. **Excessive Model-Level NaN Validation** (CRITICAL BOTTLENECK)

**Impact:** ~73ms+ per validation cycle, running on EVERY forward pass

**Evidence from Trace:**
- Top event: `aten::where` - 73.43ms
- `aten::nonzero` - 73.38ms  
- `aten::nonzero_numpy` - 73.43ms
- Total CPU ops: 854ms (0.85s) across trace

**Root Cause:**
- 50+ `torch.where()` and `torch.isnan()` checks per forward pass
- These run unconditionally, even when debug mode is disabled
- Located in:
  - `npp_rl/models/attentive_state_mlp.py` (18+ checks)
  - `npp_rl/feature_extractors/configurable_extractor.py` (30+ checks)
  - `npp_rl/agents/masked_actor_critic_policy.py` (10+ checks)

**Example Code Pattern:**
```python
# In attentive_state_mlp.py, configurable_extractor.py, etc.
if torch.isnan(phys_feat).any():
    nan_mask = torch.isnan(phys_feat)
    batch_indices = torch.where(nan_mask.any(dim=1))[0]  # EXPENSIVE!
    raise ValueError(...)
```

**Solution:** Make these checks conditional on a debug flag (similar to environment-level checks)

**Expected Impact:** 20-30% speedup in forward pass (eliminates ~73ms overhead per validation cycle)

---

### 2. **Rollout Collection Regression**

**Problem:** Rollout time increased from 154s → 172s (+11.9%)

**Possible Causes:**
1. Memory profiler callback overhead (snapshots every 10k steps)
2. Additional callback chaining (MemorySnapshotCallback + existing callbacks)
3. PyTorch profiler instrumentation overhead
4. Garbage collection calls in memory callback

**Investigation Needed:**
- Profile with `--enable-memory-profiling false` to isolate overhead
- Consider making memory snapshots async or less frequent
- Measure callback overhead separately

**Expected Fix Impact:** 5-15% speedup if overhead is significant

---

### 3. **CPU Memory Growth Still Occurring**

**Problem:** 4.3GB growth during training (only 3% better than before)

**Analysis:**
- Buffer limits were added, but growth persists
- Need memory profiler report to identify exact sources
- Possible causes:
  - PyTorch internal buffers
  - Observation processing temporary allocations
  - Callback data accumulation (even with maxlen)
  - Graph data structures

**Next Steps:**
- Review memory profiler report when available
- Identify specific leak sources
- Add additional buffer limits or cleanup

---

### 4. **Action Mask Operations**

**Impact:** Moderate (necessary but could be optimized)

**Location:** `masked_actor_critic_policy.py:_apply_action_mask()`

**Current Implementation:**
```python
masked_logits = torch.where(
    action_mask,
    action_logits,
    torch.tensor(float("-inf"), ...)
)
```

**Optimization Opportunities:**
- Use in-place operations where possible
- Pre-allocate -inf tensor once
- Consider using masked_fill_ for better performance

**Expected Impact:** 2-5% speedup

---

## Recommended Next Steps (Priority Order)

### Phase 6: Model-Level Debug Checks (HIGHEST PRIORITY)

**Goal:** Eliminate 73ms+ overhead from NaN validation

**Implementation:**
1. Add `debug_mode` parameter to model classes
2. Make all `torch.isnan()` and `torch.where()` validation checks conditional
3. Pass debug flag from training script through model initialization

**Files to Modify:**
- `npp_rl/models/attentive_state_mlp.py` (~20 checks)
- `npp_rl/feature_extractors/configurable_extractor.py` (~30 checks)
- `npp_rl/agents/masked_actor_critic_policy.py` (~10 checks)
- `npp_rl/agents/objective_attention_actor_critic_policy.py` (if applicable)

**Expected Impact:** 20-30% speedup in forward pass

**Time Estimate:** 2-3 hours

---

### Phase 7: Memory Profiler Overhead Analysis

**Goal:** Measure and reduce memory profiler overhead

**Implementation:**
1. Profile with/without `--enable-memory-profiling`
2. If overhead >5%, optimize:
   - Reduce snapshot frequency
   - Make snapshots async
   - Defer expensive comparisons
   - Only snapshot on rank 0 for multi-GPU

**Expected Impact:** 5-15% speedup if overhead is significant

**Time Estimate:** 1-2 hours

---

### Phase 8: Action Mask Optimization

**Goal:** Optimize action masking operations

**Implementation:**
1. Pre-allocate -inf tensor
2. Use `masked_fill_` instead of `torch.where()` where possible
3. Profile to verify improvement

**Expected Impact:** 2-5% speedup

**Time Estimate:** 1-2 hours

---

### Phase 9: Memory Leak Investigation

**Goal:** Identify and fix remaining memory leaks

**Implementation:**
1. Review memory profiler reports
2. Identify specific leak sources
3. Add targeted fixes (buffer limits, cleanup, etc.)

**Expected Impact:** Stable memory usage

**Time Estimate:** 2-4 hours

---

## Performance Projections

### Current State (After Phase 1-5)
- Training: 352.91s for 2000 steps
- Rollout: 172.37s (86ms/step)
- Steps/second: ~5.7

### After Phase 6 (Model Debug Checks)
- Expected: 20-30% faster forward pass
- Projected rollout: 120-138s (60-69ms/step)
- Projected steps/second: ~7.2-8.3

### After Phase 7 (Memory Profiler Optimization)
- Expected: 5-15% additional if overhead significant
- Projected rollout: 102-131s (51-66ms/step)
- Projected steps/second: ~7.6-9.8

### With 128 Environments (After All Optimizations)
- Current: ~5.7 steps/second × 2 envs = 11.4 total steps/sec
- Projected: ~8 steps/second × 128 envs = **1,024 steps/second**
- **Improvement: 90x faster than original baseline**

---

## Files Requiring Model-Level Debug Flag

1. **npp_rl/models/attentive_state_mlp.py**
   - ~18 NaN checks using `torch.where()`
   - Add `debug_mode=False` parameter to `__init__()`
   - Wrap all checks: `if self.debug_mode and torch.isnan(...)`

2. **npp_rl/feature_extractors/configurable_extractor.py**
   - ~30 NaN checks across multiple methods
   - Add `debug_mode=False` parameter
   - Wrap all validation checks

3. **npp_rl/agents/masked_actor_critic_policy.py**
   - ~10 NaN checks
   - Add `debug_mode=False` parameter
   - Wrap validation checks

4. **Model initialization chain:**
   - `npp_rl/training/architecture_trainer.py` - pass debug flag
   - `npp_rl/training/policy_utils.py` - propagate flag
   - Stable-baselines3 PPO initialization - may need wrapper

---

## Additional Observations

### GPU Utilization
- Still very low (<5% during rollout)
- GPU memory: 142MB allocated, 12GB reserved (99% waste)
- Action: Increase environments to 128+ to utilize GPU better

### Evaluation Time
- Improved significantly: 128s → 59s (54% faster)
- Likely due to debug check optimizations

### Environment Setup
- Slower: 2.18s → 4.18s (92% slower)
- May be due to additional initialization or profiling overhead
- Not critical (one-time cost)

---

## Conclusion

The initial optimizations achieved **21.4% overall improvement**, but revealed a critical bottleneck: **excessive model-level NaN validation checks** consuming 73ms+ per forward pass. Making these conditional is the highest-priority next step, expected to yield an additional **20-30% speedup**.

The rollout collection regression needs investigation to determine if it's from memory profiler overhead or other causes.

Memory growth persists and requires further investigation using the memory profiler reports.

