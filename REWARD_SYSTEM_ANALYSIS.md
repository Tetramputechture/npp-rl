# Reward System Review and Analysis

**Date:** 2025-10-26  
**Reviewer:** OpenHands AI  
**Scope:** Complete review of reward calculation, propagation, and logging in NPP-RL training system

---

## Executive Summary

This document provides a comprehensive analysis of the reward system in the NPP-RL project, covering reward calculation in the nclone simulator, reward wrappers in npp-rl, and TensorBoard logging. The review identified **one critical issue** (missing PBRS logging callback) and several opportunities for improvement, all of which have been addressed.

### Key Findings

✅ **Strengths:**
- Well-structured reward system with centralized constants
- Correct PBRS implementation following Ng et al. (1999) theory
- Proper reward normalization settings (norm_reward=False)
- Clean separation of reward components (terminal, PBRS, intrinsic, hierarchical)
- Good documentation in reward calculation files

🔧 **Issues Fixed:**
1. **Critical:** PBRSLoggingCallback was not being added to training callbacks
2. **Important:** Hierarchical stability callbacks were not added for hierarchical PPO training
3. **Moderate:** Reward component logging was incomplete in EnhancedTensorBoardCallback
4. **Minor:** Info dict keys inconsistent between wrappers and callbacks

---

## Reward System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   nclone: Base Environment                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Terminal Rewards: +1.0 (win), -0.5 (death), +0.1 (sw)│  │
│  │ Time Penalty: -0.01 per step                          │  │
│  │ Navigation Reward: Distance-based shaping             │  │
│  │ Exploration Reward: Multi-scale spatial coverage      │  │
│  │ PBRS: Potential-based reward shaping                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              npp-rl: Hierarchical Wrapper (Optional)         │
│  Adds subtask-specific dense rewards:                        │
│  - Switch navigation: Distance + velocity alignment          │
│  - Exit navigation: Distance + velocity alignment            │
│  - Hazard avoidance: Safety distance rewards                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              npp-rl: Intrinsic Reward Wrapper (Optional)     │
│  Adds ICM-based intrinsic motivation:                        │
│  - total_reward = extrinsic + α * intrinsic                  │
│  - α is adaptive based on training progress                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    PPO Training (SB3)                        │
│  VecNormalize: norm_obs=True, norm_reward=False             │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Analysis

### 1. Base Reward Calculation (nclone)

**File:** `nclone/gym_environment/reward_calculation/main_reward_calculator.py`

#### Terminal Rewards
```python
REWARD_LEVEL_COMPLETE = 1.0      # Mission success
REWARD_DEATH = -0.5              # Agent death penalty
REWARD_SWITCH_ACTIVATE = 0.1     # Subgoal completion
REWARD_STEP_PENALTY = -0.01      # Time penalty per step
```

**Analysis:**
- ✅ Reward scale is appropriate for PPO ([-1, 1] range roughly)
- ✅ Death penalty magnitude relative to success reward encourages survival
- ✅ Switch activation provides intermediate milestone reward
- ⚠️ Time penalty accumulates; for long episodes (>100 steps), it can become significant

**Best Practice Compliance:**
- ✅ Sparse rewards for key events (completion, death)
- ✅ Small step penalty to encourage efficiency
- ✅ No reward clipping (allows natural gradients)

#### PBRS Implementation

**File:** `nclone/gym_environment/reward_calculation/pbrs_potentials.py`

**Theory:** Potential-Based Reward Shaping (Ng et al., 1999)
```
F(s,a,s') = γ * Φ(s') - Φ(s)
```

**Implementation:**
```python
# In main_reward_calculator.py
if self.prev_potential is not None:
    pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
    reward += pbrs_reward
self.prev_potential = current_potential
```

**Analysis:**
- ✅ **Correct formula implementation**
- ✅ Gamma factor (0.999) matches PPO discount factor
- ✅ Prev_potential properly initialized to None (no shaping on first step)
- ✅ Potentials normalized to [0, 1] range for consistency
- ✅ Policy invariance guaranteed by correct formula

**Potential Functions:**

1. **objective_distance_potential**: Distance to switch/exit
   - Inverted normalized distance: 1.0 (at target) → 0.0 (far away)
   - Provides gradient toward current objective
   
2. **hazard_proximity_potential**: Proximity to mines
   - 1.0 (safe) → 0.0 (near hazard)
   - Encourages safe navigation
   
3. **impact_risk_potential**: Velocity-based collision risk
   - Lower potential for high downward velocity near surfaces
   - Discourages dangerous high-speed impacts
   
4. **exploration_potential**: State novelty
   - Based on visit counts in discretized state space
   - Encourages exploration of unvisited areas

**Weights:**
```python
PBRS_WEIGHT_OBJECTIVE = 0.7      # Primary: reaching objectives
PBRS_WEIGHT_HAZARD = 0.15        # Secondary: safety
PBRS_WEIGHT_IMPACT = 0.05        # Tertiary: impact avoidance
PBRS_WEIGHT_EXPLORATION = 0.1    # Tertiary: exploration
```

**Analysis:**
- ✅ Weights sum to 1.0 (proper convex combination)
- ✅ Objective heavily weighted (0.7) as primary goal
- ✅ Safety (hazard + impact = 0.2) balanced against exploration (0.1)

#### Navigation Reward

**File:** `nclone/gym_environment/reward_calculation/navigation_reward_calculator.py`

- Distance-based reward shaping to objectives
- Separate from PBRS (can be enabled/disabled independently)
- Uses exponential decay with distance

**Analysis:**
- ⚠️ Potential redundancy with PBRS objective_distance_potential
- ℹ️ This is intentional for backward compatibility and ablation studies

#### Exploration Reward

**File:** `nclone/gym_environment/reward_calculation/exploration_reward_calculator.py`

- Multi-scale spatial binning (coarse + fine grids)
- Visit count tracking with decay
- Novelty bonus for unvisited/rarely-visited states

**Analysis:**
- ✅ Multi-scale approach balances local and global exploration
- ✅ Count decay prevents over-exploration of previously visited areas
- ⚠️ May conflict with PBRS exploration_potential (different mechanisms)

---

### 2. Hierarchical Reward Wrapper (npp-rl)

**File:** `npp-rl/npp_rl/wrappers/hierarchical_reward_wrapper.py`

**Purpose:** Add dense subtask-specific rewards for hierarchical RL

**Subtasks:**
- NAVIGATE_TO_SWITCH
- NAVIGATE_TO_EXIT  
- AVOID_HAZARDS

**Reward Combination:**
```python
total_reward = base_reward + subtask_reward
```

**Subtask Reward Components:**

1. **Switch Navigation:**
   - Distance reward: Progress toward switch
   - Velocity alignment: Reward for moving in correct direction
   - Hazard penalty: Negative reward for proximity to mines

2. **Exit Navigation:**
   - Distance reward: Progress toward exit
   - Velocity alignment: Moving toward exit
   - Hazard penalty: Safety considerations

3. **Hazard Avoidance:**
   - Safety distance reward: Maintaining distance from mines
   - Survival bonus: Staying alive longer

**Analysis:**
- ✅ Clean separation of base and subtask rewards
- ✅ Subtask transitions tracked in info dict
- ✅ Detailed logging of reward components
- ⚠️ Subtask rewards should be scaled carefully to not overwhelm base rewards
- ℹ️ **Fixed:** Added `hierarchical_reward_episode` key for TensorBoard logging

**Best Practice Compliance:**
- ✅ Dense rewards for continuous feedback
- ✅ Subtask decomposition aids learning
- ⚠️ Need to monitor reward balance (base vs. subtask ratio)

---

### 3. Intrinsic Reward Wrapper (npp-rl)

**File:** `npp-rl/npp_rl/wrappers/intrinsic_reward_wrapper.py`

**Purpose:** Add curiosity-driven exploration via Intrinsic Curiosity Module (ICM)

**Reward Combination:**
```python
total_reward = extrinsic_reward + α * intrinsic_reward
```

**ICM Components:**
1. **Forward Model:** Predicts next state features from current state + action
2. **Inverse Model:** Predicts action from state transition
3. **Intrinsic Reward:** Forward model prediction error (novelty)

**Adaptive α Scaling:**
```python
α(t) = α_init * decay_factor^(timestep / decay_rate)
```
- Starts with high α for exploration
- Decays over time as agent learns
- Encourages early exploration, later exploitation

**Analysis:**
- ✅ ICM is well-established for exploration (Pathak et al., 2017)
- ✅ Adaptive α balances exploration vs. exploitation over training
- ✅ Separate tracking of extrinsic and intrinsic rewards
- ⚠️ Intrinsic rewards should be monitored to ensure they don't dominate
- ℹ️ **Fixed:** Added `r_ext_episode` and `r_int_episode` keys for logging

**Best Practice Compliance:**
- ✅ Intrinsic motivation for sparse reward environments
- ✅ Separate logging enables analysis of contribution
- ✅ Adaptive scaling prevents exploration from dominating late in training

---

### 4. Training Integration (architecture_trainer.py)

**File:** `npp-rl/npp_rl/training/architecture_trainer.py`

**VecNormalize Configuration:**
```python
VecNormalize(
    env,
    training=True,
    norm_obs=True,        # ✅ Normalize observations
    norm_reward=False,    # ✅ Do NOT normalize rewards
    clip_obs=10.0,
    gamma=0.999
)
```

**Analysis:**
- ✅ **Critical:** `norm_reward=False` is correct
- ✅ Reward normalization can destroy sparse/shaped reward structure
- ✅ Observation normalization aids neural network training
- ✅ Gamma (0.999) matches PBRS gamma for consistency

**Best Practice (OpenAI Spinning Up):**
> "Reward normalization is tricky for sparse or shaped rewards. 
> Normalizing can eliminate the carefully designed reward signal structure.
> Generally avoid unless rewards have extreme scale differences."

**Callback Registration:**

**Before Fix:**
```python
# Missing PBRS logging!
callbacks = [verbose_callback, enhanced_tb_callback, route_callback]
```

**After Fix:**
```python
callbacks = [
    verbose_callback,
    enhanced_tb_callback,
    pbrs_callback,           # ✅ Added
    route_callback,
    stability_callback,      # ✅ Added (if hierarchical)
    subtask_callback,        # ✅ Added (if hierarchical)
    curriculum_callback,     # (if curriculum)
]
```

**Analysis:**
- 🔧 **Fixed:** PBRSLoggingCallback now registered
- 🔧 **Fixed:** Hierarchical callbacks now registered when using hierarchical PPO
- ✅ All major reward components now logged to TensorBoard

---

### 5. TensorBoard Logging

#### PBRSLoggingCallback

**File:** `npp-rl/npp_rl/callbacks/pbrs_logging_callback.py`

**Logs:**
- `pbrs_rewards/navigation_reward_mean`
- `pbrs_rewards/exploration_reward_mean`
- `pbrs_rewards/pbrs_reward_mean`
- `pbrs_rewards/total_reward_mean`
- `pbrs_potentials/objective_mean`
- `pbrs_potentials/hazard_mean`
- `pbrs_potentials/impact_mean`
- `pbrs_potentials/exploration_mean`

**Analysis:**
- ✅ Comprehensive logging of all PBRS components
- ✅ Mean and std tracked for statistical analysis
- ℹ️ **Fixed:** Now properly registered in architecture_trainer.py

#### EnhancedTensorBoardCallback

**File:** `npp-rl/npp_rl/callbacks/enhanced_tensorboard_callback.py`

**Before Fix:**
- Episode rewards (total only)
- Episode lengths
- Success rates
- Action distributions

**After Fix (Added):**
- `rewards/intrinsic_mean` - Intrinsic reward statistics
- `rewards/extrinsic_mean` - Extrinsic reward statistics
- `rewards/hierarchical_mean` - Hierarchical reward statistics
- `rewards/intrinsic_ratio` - Ratio of intrinsic to total reward

**Analysis:**
- ✅ Complete reward decomposition for analysis
- ✅ Ratio tracking helps monitor intrinsic reward contribution
- ✅ Separate statistics enable debugging of each component

#### HierarchicalStabilityCallback

**File:** `npp-rl/npp_rl/callbacks/hierarchical_callbacks.py`

**Logs:**
- High-level and low-level gradient norms
- Policy and value losses for both levels
- Stability warnings for training divergence

**Analysis:**
- ℹ️ **Fixed:** Now registered when using hierarchical PPO
- ✅ Critical for monitoring hierarchical training stability
- ✅ Early warning system for gradient explosion/vanishing

---

## Best Practices Compliance

### ✅ Followed Best Practices

1. **No reward normalization in VecNormalize** ([Spinning Up](https://spinningup.openai.com/))
   - Preserves sparse reward structure
   - Maintains shaped reward gradients

2. **Correct PBRS implementation** ([Ng et al., 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf))
   - Formula: γ * Φ(s') - Φ(s)
   - Policy invariance guaranteed

3. **Reward decomposition and logging** (RL debugging best practice)
   - Separate tracking of each component
   - Statistical analysis (mean, std, ratio)
   - TensorBoard visualization

4. **Appropriate reward scales** (PPO best practices)
   - Terminal rewards in [-1, 1] range
   - Dense rewards scaled to not overwhelm sparse signals
   - No extreme outliers requiring clipping

5. **Adaptive intrinsic reward scaling** ([Curiosity-driven Exploration, Pathak et al. 2017](https://arxiv.org/abs/1705.05363))
   - High α early for exploration
   - Decay over time for exploitation
   - Prevents intrinsic rewards from dominating

### ⚠️ Monitoring Recommendations

1. **Time Penalty Accumulation**
   - For episodes >100 steps, time penalty can dominate
   - Monitor `pbrs_rewards/total_reward_mean` to ensure balance
   - Consider adaptive time penalty based on level difficulty

2. **Intrinsic vs. Extrinsic Balance**
   - Track `rewards/intrinsic_ratio` in TensorBoard
   - Ideal ratio: 10-30% intrinsic early, <5% late
   - Adjust α_decay_rate if ratio stays too high

3. **Hierarchical Reward Balance**
   - Track `rewards/hierarchical_mean` vs. base rewards
   - Subtask rewards should guide, not dominate
   - Adjust subtask_reward_weight if needed

4. **PBRS Potential Verification**
   - Monitor `pbrs_potentials/*_mean` trends
   - Potentials should increase as agent approaches goals
   - Sudden drops indicate potential calculation issues

---

## Issues Fixed

### 1. Critical: Missing PBRS Logging Callback

**Issue:** PBRSLoggingCallback was defined but never registered in architecture_trainer.py

**Impact:** No TensorBoard logs for PBRS components, making it impossible to debug reward shaping

**Fix:**
```python
# Added in architecture_trainer.py:1012-1017
from npp_rl.callbacks import PBRSLoggingCallback

pbrs_callback = PBRSLoggingCallback(verbose=1)
callbacks.append(pbrs_callback)
logger.info("Added PBRS logging callback for reward component tracking")
```

**Verification:**
- PBRSLoggingCallback now registered in callback list
- Will log to TensorBoard under `pbrs_rewards/` and `pbrs_potentials/`

---

### 2. Important: Missing Hierarchical Callbacks

**Issue:** When using hierarchical PPO, stability monitoring callbacks were not registered

**Impact:** No monitoring of gradient norms, training instability, or subtask transitions

**Fix:**
```python
# Added in architecture_trainer.py:1035-1060
if self.use_hierarchical_ppo:
    from npp_rl.callbacks.hierarchical_callbacks import (
        HierarchicalStabilityCallback,
        SubtaskTransitionCallback,
    )
    
    stability_callback = HierarchicalStabilityCallback(...)
    callbacks.append(stability_callback)
    
    subtask_callback = SubtaskTransitionCallback(...)
    callbacks.append(subtask_callback)
```

**Verification:**
- Callbacks now registered when `use_hierarchical_ppo=True`
- Will track gradient norms, stability, and subtask transitions

---

### 3. Moderate: Incomplete Reward Component Logging

**Issue:** EnhancedTensorBoardCallback didn't track intrinsic/hierarchical reward components

**Impact:** Missing detailed reward breakdown in TensorBoard for analysis

**Fix:**
```python
# Added in enhanced_tensorboard_callback.py:70-73
self.episode_intrinsic_rewards = deque(maxlen=100)
self.episode_extrinsic_rewards = deque(maxlen=100)
self.episode_hierarchical_rewards = deque(maxlen=100)

# Added logging in _log_scalar_metrics():252-281
if self.episode_intrinsic_rewards:
    self.tb_writer.add_scalar('rewards/intrinsic_mean', ...)
# ... etc for all components
```

**Verification:**
- Reward components now tracked and logged
- TensorBoard will show `rewards/intrinsic_mean`, `rewards/extrinsic_mean`, etc.

---

### 4. Minor: Inconsistent Info Dict Keys

**Issue:** Wrappers used different key names than callbacks expected

**Impact:** Episode statistics not captured by EnhancedTensorBoardCallback

**Fix:**
```python
# intrinsic_reward_wrapper.py:289-291
info.update({
    "r_ext_episode": self.episode_stats["r_ext_sum"],
    "r_int_episode": self.episode_stats["r_int_sum"],
})

# hierarchical_reward_wrapper.py:182
info["hierarchical_reward_episode"] = self.episode_subtask_reward
```

**Verification:**
- Info dict keys now match callback expectations
- Episode statistics will be captured correctly

---

## Testing Recommendations

### 1. Verify PBRS Logging
```bash
# Start training with PBRS enabled
python npp_rl/training/architecture_trainer.py --enable-pbrs

# Check TensorBoard for new logs
tensorboard --logdir runs/
# Look for: pbrs_rewards/* and pbrs_potentials/*
```

### 2. Verify Intrinsic Reward Logging
```bash
# Train with intrinsic rewards
python npp_rl/training/architecture_trainer.py --intrinsic-reward

# Check TensorBoard
# Look for: rewards/intrinsic_mean, rewards/extrinsic_mean, rewards/intrinsic_ratio
```

### 3. Verify Hierarchical Callbacks
```bash
# Train with hierarchical PPO
python npp_rl/training/architecture_trainer.py --use-hierarchical-ppo

# Check TensorBoard
# Look for: hierarchical/gradient_norms, hierarchical/stability_warnings
```

### 4. Reward Balance Check
```python
# After training, analyze reward components
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('runs/experiment_1')
ea.Reload()

# Compare reward magnitudes
intrinsic = ea.Scalars('rewards/intrinsic_mean')
extrinsic = ea.Scalars('rewards/extrinsic_mean')
pbrs = ea.Scalars('pbrs_rewards/pbrs_reward_mean')

# Verify balance: intrinsic should be 10-30% of total early, <5% late
```

---

## Reward System Configuration Guide

### Recommended Settings for Different Scenarios

#### 1. Standard Training (No Hierarchical, No Intrinsic)
```yaml
enable_pbrs: true
enable_navigation_reward: false  # Redundant with PBRS
enable_exploration_reward: true
pbrs_weights:
  objective: 0.7
  hazard: 0.15
  impact: 0.05
  exploration: 0.1
```

#### 2. Hierarchical RL Training
```yaml
use_hierarchical_ppo: true
enable_pbrs: true
hierarchical_subtask_weight: 0.3  # Balance with base rewards
enable_subtask_logging: true
```

#### 3. Exploration-Heavy (Intrinsic Motivation)
```yaml
enable_intrinsic_reward: true
intrinsic_alpha_init: 0.5       # High initial exploration
intrinsic_alpha_decay: 0.99999  # Slow decay
enable_icm_training: true
icm_update_freq: 4              # Update ICM every 4 steps
```

#### 4. Curriculum Learning
```yaml
use_curriculum: true
curriculum_start_stage: 0
enable_pbrs: true
# Gradually increase task difficulty
# PBRS provides consistent shaping across stages
```

---

## References

1. **Ng, A. Y., Harada, D., & Russell, S. (1999).** "Policy invariance under reward transformations: Theory and application to reward shaping." ICML 1999.

2. **Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017).** "Curiosity-driven exploration by self-supervised prediction." ICML 2017.

3. **OpenAI Spinning Up.** "Introduction to RL." https://spinningup.openai.com/

4. **Lilian Weng.** "A (Long) Peek into Reinforcement Learning." https://lilianweng.github.io/posts/2018-02-19-rl-overview/

5. **Stable Baselines3 Documentation.** "VecNormalize." https://stable-baselines3.readthedocs.io/

---

## Conclusion

The NPP-RL reward system is **well-designed and follows RL best practices**. The review identified and fixed one critical issue (missing PBRS logging) and several opportunities for improvement in monitoring and debugging capabilities.

### Summary of Changes:
1. ✅ Added PBRSLoggingCallback registration
2. ✅ Added hierarchical training callbacks
3. ✅ Enhanced reward component logging in TensorBoard
4. ✅ Fixed info dict key consistency

### Next Steps:
1. Run training with updated callbacks and verify TensorBoard logs
2. Monitor reward component balance (intrinsic vs. extrinsic ratios)
3. Adjust reward scaling if needed based on training performance
4. Consider adaptive time penalty for variable-length episodes

All changes preserve the correct reward calculation logic while significantly improving observability and debugging capabilities.
