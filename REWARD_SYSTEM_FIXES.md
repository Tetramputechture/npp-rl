# Reward System Fixes and Analysis

**Date:** 2025-10-26  
**Scope:** Comprehensive reward system review, accuracy analysis, and fixes

---

## Executive Summary

Conducted a comprehensive review of the NPP-RL reward system focusing on:
1. Reward propagation through training pipeline
2. Compliance with ML/RL best practices
3. TensorBoard logging completeness
4. Learning-hindering issues

**Critical Issues Found and Fixed:**
1. ⚠️ **PBRS gamma mismatch** - PBRS used γ=0.99 while PPO used γ=0.999 (violates policy invariance)
2. ⚠️ **Incomplete step-level reward logging** - PBRS components not tracked at step level
3. ℹ️ **Time penalty accumulation risk** - Documented potential issue for long episodes

---

## Critical Issues Fixed

### 1. PBRS Gamma Mismatch 🔴 CRITICAL

**Problem:**
- **PPO gamma:** 0.999 (in `architecture_trainer.py:527`)
- **PBRS gamma:** 0.99 (in `reward_constants.py:126`)

**Why This Matters:**
According to Ng et al. (1999), potential-based reward shaping (PBRS) maintains policy invariance ONLY when the discount factor γ in the shaping function matches the RL algorithm's discount factor:

```
F(s,s') = γ * Φ(s') - Φ(s)
```

If γ_PBRS ≠ γ_PPO, the optimal policy can change, defeating the purpose of reward shaping!

**Fix Applied:**
```python
# reward_constants.py:128
PBRS_GAMMA = 0.999  # Changed from 0.99 to match PPO
```

**Impact:**
- **Before:** Policy invariance NOT guaranteed (theoretical violation)
- **After:** Policy invariance guaranteed (theoretically sound)
- **Learning:** Should see more stable learning and better convergence

**File Modified:**
- `nclone/nclone/gym_environment/reward_calculation/reward_constants.py`

---

### 2. Incomplete Step-Level PBRS Logging 🟠 IMPORTANT

**Problem:**
PBRS reward components (navigation, exploration, shaping) were only available in episode-end info dict, not logged at step level. This made it impossible to:
- Track PBRS contribution during training
- Debug reward shaping issues
- Analyze potential function behavior over time

**Fix Applied:**
Enhanced `EnhancedTensorBoardCallback` to track and log PBRS components at every step:

**New Tracking Buffers:**
```python
# Step-level PBRS reward tracking
self.pbrs_navigation_rewards = deque(maxlen=1000)
self.pbrs_exploration_rewards = deque(maxlen=1000)
self.pbrs_shaping_rewards = deque(maxlen=1000)
self.pbrs_total_rewards = deque(maxlen=1000)

# PBRS potential tracking
self.pbrs_objective_potentials = deque(maxlen=1000)
self.pbrs_hazard_potentials = deque(maxlen=1000)
self.pbrs_impact_potentials = deque(maxlen=1000)
self.pbrs_exploration_potentials = deque(maxlen=1000)
```

**New Method:**
```python
def _track_pbrs_components(self, info: Dict[str, Any]):
    """Track PBRS reward components from step info."""
    # Extracts pbrs_components from info dict at every step
    # Tracks navigation, exploration, PBRS shaping, and potentials
```

**New TensorBoard Metrics:**
```
pbrs_rewards/
  ├── navigation_mean
  ├── navigation_std
  ├── exploration_mean
  ├── exploration_std
  ├── pbrs_mean          # The F(s,s') shaping reward
  ├── pbrs_std
  ├── pbrs_min           # Debugging
  ├── pbrs_max           # Debugging
  ├── total_mean
  └── total_std

pbrs_potentials/
  ├── objective_mean     # Distance to switch/exit
  ├── objective_std
  ├── hazard_mean
  ├── impact_mean
  └── exploration_mean

pbrs_summary/
  └── pbrs_contribution_ratio  # |PBRS| / |Total| (how much PBRS contributes)
```

**Impact:**
- Real-time monitoring of PBRS contribution
- Early detection of PBRS domination or ineffectiveness
- Better debugging of reward shaping issues
- Validation of potential function design

**File Modified:**
- `npp_rl/callbacks/enhanced_tensorboard_callback.py`

---

## Documented Issues (Not Fixed)

### Time Penalty Accumulation Risk ℹ️ INFO

**Observation:**
```python
TIME_PENALTY_PER_STEP = -0.01  # reward_constants.py:58
```

**Analysis:**
- **100-step episode:** -1.0 total time penalty (equals success reward!)
- **200-step episode:** -2.0 total time penalty (exceeds success reward)
- **Risk:** For difficult/long levels, time penalty can make successful completion unrewarding

**Example Scenario:**
```
Level requires 150 steps to complete perfectly
Time penalty: 150 * -0.01 = -1.5
Success reward: +1.0
Net reward: +1.0 - 1.5 = -0.5 (negative!)
```

**Why Not Fixed:**
This is a design decision, not a bug. The time penalty is intentional to encourage efficiency. However, users should be aware of this for:
1. Curriculum design (start with shorter levels)
2. Level selection (avoid extremely long levels early in training)
3. Hyperparameter tuning (consider adaptive time penalty or different scales for different level types)

**Recommendations:**
1. **Monitor:** Track `episode/length_mean` vs `episode/reward_mean` in TensorBoard
2. **Alert:** If successful episodes have negative net reward due to time penalty
3. **Consider:** Adaptive time penalty based on level difficulty/length:
   ```python
   time_penalty = -0.01 * (1.0 / difficulty_multiplier)
   ```
4. **Alternative:** Cap total time penalty per episode:
   ```python
   max_time_penalty = -0.5  # Never exceed 50% of success reward
   ```

---

## Additional Improvements

### RouteVisualizationCallback Validation ✅

**Reviewed:**
- Line 253: `'episode_reward': info.get('episode', {}).get('r', 0)`
- Line 346: `f"Reward: {route_data['episode_reward']:.2f}"`

**Verdict:** Correctly reads and displays reward from info dict. No issues found.

---

## Reward System Validation

### PBRS Implementation ✅ Verified Correct

**Formula (Ng et al., 1999):**
```
F(s,a,s') = γ * Φ(s') - Φ(s)
```

**Implementation (`main_reward_calculator.py:125`):**
```python
pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
reward += pbrs_reward
```

**Verification:**
- ✅ Formula matches theory exactly
- ✅ Gamma now matches PPO (0.999) ← **FIXED**
- ✅ Prev_potential initialized to None (correct for first step)
- ✅ Potentials normalized to [0, 1] range
- ✅ Policy invariance now guaranteed

### Reward Scales ✅ Appropriate

**Terminal Rewards:**
```python
LEVEL_COMPLETION_REWARD = 1.0
DEATH_PENALTY = -0.5
SWITCH_ACTIVATION_REWARD = 0.1
TIME_PENALTY_PER_STEP = -0.01
```

**PBRS Scales:**
```python
PBRS_SWITCH_DISTANCE_SCALE = 0.05
PBRS_EXIT_DISTANCE_SCALE = 0.05
```

**Analysis:**
- Potentials in [0, 1] → Scaled potentials in [0, 0.05]
- Max PBRS per step: `0.999 * 0.05 - 0 = 0.0495` (moving from worst to best position)
- **Ratio:** PBRS (~0.05) : Terminal (1.0) = 1:20 ✅
- **Verdict:** PBRS provides gradient without dominating terminal rewards

### VecNormalize Configuration ✅ Correct

**Configuration (`architecture_trainer.py:739`):**
```python
VecNormalize(
    env,
    norm_obs=True,        # ✅ Normalize observations
    norm_reward=False,    # ✅ Do NOT normalize rewards
    gamma=0.999
)
```

**Why This Is Correct:**
- Reward normalization can destroy sparse/shaped reward structure
- PBRS provides carefully designed gradients that shouldn't be normalized
- Follows OpenAI Spinning Up best practices
- Reward scales already appropriate for PPO

---

## Testing Recommendations

### 1. Verify PBRS Gamma Fix

**Test:**
```bash
cd /workspace/nclone
python -c "from nclone.gym_environment.reward_calculation.reward_constants import PBRS_GAMMA; print(f'PBRS Gamma: {PBRS_GAMMA}')"
# Should print: PBRS Gamma: 0.999
```

**Verify in training:**
```python
# Check that PPO and PBRS gammas match
ppo_gamma = model.gamma
pbrs_gamma = env.reward_calculator.pbrs_gamma
assert ppo_gamma == pbrs_gamma, f"Gamma mismatch! PPO: {ppo_gamma}, PBRS: {pbrs_gamma}"
```

### 2. Verify PBRS Logging

**Start training with PBRS:**
```bash
cd /workspace/npp-rl
python npp_rl/training/architecture_trainer.py --enable-pbrs
```

**Check TensorBoard:**
```bash
tensorboard --logdir runs/
```

**Verify these metrics appear:**
- `pbrs_rewards/pbrs_mean` - Should show small values (~0.01-0.05 range)
- `pbrs_rewards/pbrs_min` - Should show negative values (moving away from goal)
- `pbrs_rewards/pbrs_max` - Should show positive values (moving toward goal)
- `pbrs_potentials/objective_mean` - Should increase as agent learns to approach goals
- `pbrs_summary/pbrs_contribution_ratio` - Should be ~5-20% of total reward

**Expected Patterns:**
- **Early training:** PBRS contribution high (~20-30%), agent exploring
- **Mid training:** PBRS contribution moderate (~10-15%), agent learning path
- **Late training:** PBRS contribution low (~5-10%), agent using terminal rewards

### 3. Monitor Time Penalty Impact

**Check in TensorBoard:**
```
episode/length_mean vs episode/reward_mean
```

**Alert if:**
- Successful episodes (is_success=1) have negative net rewards
- Episode length > 100 steps consistently
- Reward trend decreases as episode length increases

**Fix if needed:**
- Adjust `TIME_PENALTY_PER_STEP` to smaller value (e.g., -0.005)
- Implement adaptive time penalty
- Filter training to shorter levels initially

---

## Comparison: Before vs After

### Before Fixes

**Issue 1: Gamma Mismatch**
```
PPO:  γ = 0.999
PBRS: γ = 0.99
Result: Policy invariance NOT guaranteed ❌
```

**Issue 2: Missing PBRS Logging**
```
TensorBoard Metrics:
- episode/reward_mean ✓
- pbrs_rewards/* ❌ (missing)
- pbrs_potentials/* ❌ (missing)

Debugging Capability: Poor
```

### After Fixes

**Issue 1: Gamma Fixed**
```
PPO:  γ = 0.999
PBRS: γ = 0.999
Result: Policy invariance guaranteed ✅
```

**Issue 2: Complete PBRS Logging**
```
TensorBoard Metrics:
- episode/reward_mean ✓
- pbrs_rewards/navigation_mean ✓
- pbrs_rewards/exploration_mean ✓
- pbrs_rewards/pbrs_mean ✓
- pbrs_rewards/pbrs_min/max ✓
- pbrs_potentials/objective_mean ✓
- pbrs_potentials/hazard/impact/exploration_mean ✓
- pbrs_summary/pbrs_contribution_ratio ✓

Debugging Capability: Excellent
```

---

## Learning Impact Analysis

### Impact of Gamma Fix

**Theoretical:**
- PBRS with mismatched gamma CAN change optimal policy
- With matched gamma, optimal policy provably unchanged
- Agent can learn faster without policy distortion

**Expected Improvements:**
1. **More stable learning** - No policy interference from PBRS
2. **Better convergence** - Clear gradient toward optimal policy
3. **Consistent behavior** - Same optimal policy with/without PBRS

**Measurement:**
- Compare learning curves before/after fix
- Check if agent reaches same final policy
- Verify convergence stability (less oscillation)

### Impact of Enhanced Logging

**Debugging Capability:**
- **Before:** Only episode-level rewards, hard to diagnose issues
- **After:** Step-level component tracking, easy to identify problems

**Example Debugging Scenarios:**

**Scenario 1: Agent not learning**
- **Check:** `pbrs_potentials/objective_mean`
- **If flat:** Potential function not providing gradient
- **If decreasing:** Agent moving away from goal (wrong learning signal)
- **If increasing:** Agent learning correctly, issue elsewhere

**Scenario 2: PBRS dominating**
- **Check:** `pbrs_summary/pbrs_contribution_ratio`
- **If > 50%:** PBRS scales too large, reduce PBRS_*_SCALE constants
- **If < 5%:** PBRS scales too small, increase for better gradient

**Scenario 3: Reward explosion**
- **Check:** `pbrs_rewards/pbrs_min` and `pbrs_rewards/pbrs_max`
- **If |values| > 0.5:** Potential function not normalized correctly
- **Expected:** Values in [-0.05, 0.05] range

---

## Best Practices Compliance

### ✅ Following Best Practices

1. **Correct PBRS Theory** (Ng et al., 1999)
   - ✅ Matched gamma (NOW FIXED)
   - ✅ Normalized potentials
   - ✅ Policy invariance guaranteed

2. **No Reward Normalization** (OpenAI Spinning Up)
   - ✅ `norm_reward=False` in VecNormalize
   - ✅ Preserves sparse reward structure
   - ✅ Maintains PBRS gradients

3. **Appropriate Reward Scales** (PPO best practices)
   - ✅ Terminal rewards in [-1, 1] range
   - ✅ PBRS rewards ~5% of terminal rewards
   - ✅ No extreme outliers

4. **Comprehensive Logging** (RL debugging best practice)
   - ✅ Step-level component tracking (NOW ADDED)
   - ✅ Statistical aggregation (mean, std, min, max)
   - ✅ Ratio analysis for balance checking

5. **Adaptive Intrinsic Rewards** (Pathak et al., 2017)
   - ✅ ICM-based curiosity
   - ✅ Decay over time
   - ✅ Separate tracking

---

## Files Modified

### nclone Repository

1. **`nclone/gym_environment/reward_calculation/reward_constants.py`**
   - Line 128: Changed `PBRS_GAMMA = 0.99` → `PBRS_GAMMA = 0.999`
   - Added critical warning comment about matching PPO gamma

### npp-rl Repository

2. **`npp_rl/callbacks/enhanced_tensorboard_callback.py`**
   - Lines 75-85: Added PBRS tracking buffers
   - Lines 153: Added PBRS tracking call in `_on_step()`
   - Lines 206-238: Added `_track_pbrs_components()` method
   - Lines 333-393: Added PBRS logging in `_log_scalar_metrics()`

---

## Monitoring Dashboard

When training with these fixes, monitor these key metrics in TensorBoard:

### Critical Metrics

1. **`pbrs_summary/pbrs_contribution_ratio`**
   - **Ideal:** 10-20% early, 5-10% late
   - **Alert if:** > 50% (PBRS dominating) or < 2% (PBRS ineffective)

2. **`pbrs_potentials/objective_mean`**
   - **Expected:** Increasing trend as agent learns
   - **Alert if:** Flat or decreasing

3. **`episode/reward_mean` vs `episode/length_mean`**
   - **Alert if:** Negative rewards for successful completions
   - **Alert if:** Strong negative correlation (time penalty dominating)

### Debugging Metrics

4. **`pbrs_rewards/pbrs_min` and `pbrs_rewards/pbrs_max`**
   - **Expected:** In [-0.05, 0.05] range
   - **Alert if:** Outside [-0.1, 0.1]

5. **`pbrs_rewards/pbrs_std`**
   - **Expected:** Decreases as agent learns optimal path
   - **Alert if:** Increases over time (instability)

---

## References

1. **Ng, A. Y., Harada, D., & Russell, S. (1999)**. "Policy invariance under reward transformations: Theory and application to reward shaping." *ICML 1999*.
   - Theoretical foundation for PBRS
   - Proves policy invariance when γ_PBRS = γ_RL

2. **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017)**. "Curiosity-driven exploration by self-supervised prediction." *ICML 2017*.
   - ICM-based intrinsic motivation
   - Adaptive reward weighting

3. **OpenAI Spinning Up**. "Introduction to RL." https://spinningup.openai.com/
   - Reward normalization best practices
   - PPO hyperparameter guidelines

4. **Lilian Weng**. "A (Long) Peek into Reinforcement Learning." https://lilianweng.github.io/posts/2018-02-19-rl-overview/
   - Reward shaping overview
   - Exploration strategies

---

## Conclusion

### Summary

Fixed 2 critical issues that could hinder learning:
1. ✅ **PBRS gamma mismatch** - Now matches PPO (0.999)
2. ✅ **Missing step-level PBRS logging** - Now comprehensive

Documented 1 design consideration:
3. ℹ️ **Time penalty accumulation** - Monitor for long episodes

### Expected Learning Improvements

1. **More stable convergence** - Policy invariance now guaranteed
2. **Better debugging** - Step-level component tracking enables rapid issue identification
3. **Clearer analysis** - TensorBoard metrics show exact reward contribution

### Next Steps

1. **Test the fixes** - Run training with PBRS enabled
2. **Monitor metrics** - Watch for expected patterns in TensorBoard
3. **Compare results** - Evaluate learning curves before/after fixes
4. **Adjust if needed** - Use new metrics to tune reward scales

**All changes maintain backward compatibility and preserve correct reward calculation logic.**

---

**Ready for training! 🚀**
