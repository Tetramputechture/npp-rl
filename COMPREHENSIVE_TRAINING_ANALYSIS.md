# Comprehensive RL Training Analysis and Recommendations

**Analysis Date:** 2025-10-27  
**Training Run:** mlp-baseline-1026 (1M timesteps)  
**Configuration:** MLP baseline architecture, 14 envs, curriculum learning enabled

---

## Executive Summary

This analysis reveals **critical issues preventing effective learning** in the current RL setup. The agent has become stuck at curriculum stage 2 with only 4% success rate (down from 14.8% peak), while simultaneously experiencing severe value function collapse (value estimates degraded by -6966%). The root causes are:

1. **Catastrophic reward scaling mismatch** making level completion essentially worthless
2. **Value function collapse** indicating critic network failure
3. **Curriculum learning stagnation** with no recovery mechanism
4. **Insufficient reward density** for complex platformer navigation

**Priority Level: CRITICAL** - Current training setup will not produce a functional agent without significant changes.

---

## 1. Critical Findings from TensorBoard Analysis

### 1.1 Curriculum Learning Failure (CRITICAL ⚠️)

**Current State:**
- **Stage 0 (simplest):** 100% success rate ✓
- **Stage 1 (simpler):** 68% success rate (declining from 100%)
- **Stage 2 (simple):** 4% success rate ❌ (STUCK HERE for 435 episodes)
- **Stages 3-6:** Never reached (0 episodes)

**Problems Identified:**
```
Success Rate Trajectory on Stage 2:
Initial: 14.8% (peak)
Current: 4.0%
Trend: Declining (catastrophic forgetting)
Episodes in stage: 435
Advancement threshold: 70%
Gap to threshold: 66 percentage points!
```

**Root Causes:**
1. **No regression mechanism** - Agent cannot return to easier stages when failing
2. **Threshold too aggressive** - 70% success rate may be unrealistic for harder stages
3. **Catastrophic forgetting** - Agent losing skills learned on earlier stages
4. **Insufficient episodes** - May need more practice before declaring failure

### 1.2 Value Function Collapse (CRITICAL ⚠️)

**Metrics:**
```
Value Estimate Changes:
├─ Mean:  -0.06 → -4.33  (-6966% change!)
├─ Min:   -0.46 → -7.35  (-1505% change)
├─ Max:    0.39 → -0.31  (-178% change)
└─ Std:    0.19 →  1.76  (+804% change)

Value Loss: 0.38 → 0.60 (+56%)
```

**Interpretation:**
The value function has completely collapsed. The critic network has learned that:
- All states have negative expected returns (mean: -4.33)
- Worst case scenarios are catastrophically bad (min: -7.35)
- Even "best" states have negative value (max: -0.31)

**Why This Happens:**
Due to the reward scaling issue (see Section 2.1), most episodes result in highly negative returns:
```
Typical Failed Episode:
├─ Time penalty: -0.01 × 5000 steps = -50
├─ Death penalty: -0.5
└─ Total return: -50.5 (with no completion bonus)

Even Successful Episode (slow):
├─ Completion reward: +1.0
├─ Time penalty: -0.01 × 10000 steps = -100
└─ Total return: -99.0 (negative despite winning!)
```

The critic correctly learns that expected returns are very negative, which:
1. Demotivates exploration (everything looks bad anyway)
2. Makes policy updates unreliable (advantages become meaningless)
3. Causes training instability (large negative values amplify gradient noise)

### 1.3 Action Distribution Analysis (GOOD ✓)

**Metrics:**
```
Action Entropy: 1.79 (near theoretical maximum of log(6) ≈ 1.79)

Action Frequencies:
├─ NOOP:        15.6%
├─ Left:        17.0%
├─ Right:       16.7%
├─ Jump:        16.6%
├─ Jump+Left:   16.0%
└─ Jump+Right:  18.1%

Jump Behavior:
├─ Total jump frequency: 50.7%
├─ Directional jumps: 67.3%
└─ Vertical-only jumps: 32.7%
```

**Assessment:** ✓ **This is actually GOOD!**
- Exploration is healthy and maintained throughout training
- All actions are being tried with near-uniform distribution
- No action bias or collapse to degenerate policy
- Jump/movement balance is reasonable for platformer

**However:** Good exploration without learning indicates the problem is with the reward signal, not exploration.

### 1.4 PPO Training Metrics

**Metrics:**
```
Policy Gradient Loss: -0.004 → -0.008 (stable)
Entropy Loss: -1.79 → -1.71 (slight decrease, expected)
Clip Fraction: 0.06 → 0.16 (increasing)
Approx KL: 0.009 → 0.014 (within safe range)
Learning Rate: 3e-4 (constant)
```

**Assessment:** Mixed
- ✓ Policy updates are happening (clip fraction increasing)
- ✓ KL divergence in safe range (not too aggressive)
- ✓ Entropy maintained (exploration preserved)
- ❌ But learning is not translating to improved performance

### 1.5 Episode Statistics

**Metrics:**
```
Success Rate: 100% → 26% (declining)
Failure Rate: 0% → 74% (increasing)
```

**Interpretation:**
Clear evidence of negative learning - agent performing worse over time. This indicates:
1. The learning signal is counterproductive
2. Agent is overfitting to incorrect patterns
3. Value function collapse is causing policy degradation

---

## 2. Root Cause Analysis

### 2.1 Reward Scaling Catastrophe (CRITICAL ⚠️)

**Current Reward Structure:**
```python
LEVEL_COMPLETION_REWARD = 1.0
DEATH_PENALTY = -0.5
TIME_PENALTY_PER_STEP = -0.01
MAX_EPISODE_LENGTH = 20,000 steps
```

**The Problem:**
```
Scenario Analysis:

Fast Completion (1000 steps):
├─ Completion: +1.0
├─ Time: -0.01 × 1000 = -10.0
└─ TOTAL: -9.0 (NEGATIVE!)

Moderate Completion (5000 steps):
├─ Completion: +1.0
├─ Time: -0.01 × 5000 = -50.0
└─ TOTAL: -49.0 (VERY NEGATIVE!)

Death at mid-episode (10000 steps):
├─ Death: -0.5
├─ Time: -0.01 × 10000 = -100.0
└─ TOTAL: -100.5 (CATASTROPHIC!)

Maximum episode length:
└─ Time penalty alone: -200.0
```

**Mathematical Analysis:**
To achieve a positive return, the agent must complete the level in:
```
1.0 + (steps × -0.01) > 0
steps < 100

Agent must complete levels in under 100 steps!
```

Looking at the successful routes:
- Image 1: 395 pixels traveled → Return: -2.95 (negative despite success!)
- Image 2: 586 pixels traveled → Return: -4.86 (very negative despite success!)
- Image 3: 291 pixels traveled → Return: -1.91 (negative despite success!)

**Conclusion:** The reward structure makes level completion nearly impossible to learn as a positive outcome!

### 2.2 Sparse Reward Problem

**Current Reward Density:**
```
Dense Rewards Available (but likely disabled):
├─ Navigation shaping: DISTANCE_IMPROVEMENT_SCALE = 0.0001
├─ Exploration rewards: CELL_REWARD = 0.001
└─ PBRS potentials: Various scales

Terminal Rewards Only:
├─ Level completion: +1.0 (once per episode if successful)
├─ Switch activation: +0.1 (once per episode)
└─ Death: -0.5 (when dying)
```

For a platformer game with complex physics and long episodes, these rewards are far too sparse. The agent receives:
- Positive feedback: Maximum 2 times per episode (switch + exit)
- Negative feedback: Continuous every step (-0.01)

**Best Practice:** Platformer games need dense feedback on:
- Distance to objective
- Height changes
- Collision avoidance
- Movement efficiency
- Checkpoint progress

### 2.3 Curriculum Design Issues

**Current Curriculum:**
```
Stage Definitions (inferred):
0. simplest - Trivial levels (100% success ✓)
1. simpler - Easy levels (68% success)
2. simple - Basic challenges (4% success ❌ STUCK)
3. medium - Moderate difficulty (never reached)
4. complex - Hard levels (never reached)
5. mine_heavy - Specialized challenges (never reached)
6. exploration - Open-ended levels (never reached)

Advancement Rule:
├─ Threshold: 70% success rate
├─ Minimum episodes: 100
└─ No regression: Once advanced, cannot go back
```

**Problems:**
1. **No regression** - Cannot return to easier stages when struggling
2. **Fixed threshold** - 70% may be too high for some stages
3. **Difficulty spike** - Stage 1→2 transition may be too large (68% → 4%)
4. **No adaptive scheduling** - Cannot temporarily return to easier content

**Best Practice (from curriculum learning literature):**
- Use adaptive thresholds (e.g., 60-80% based on stage)
- Allow regression to previous stages
- Implement "mixed training" - occasionally sample from all mastered stages
- Use success rate trend, not just absolute value
- Consider confidence intervals (100 episodes may not be enough)

### 2.4 Value Function Architecture Issues

**Current Setup:**
```python
Architecture: MLP baseline
Policy Network: [512, 512, 512] (3 layers)
Value Network: [512, 512, 512] (3 layers)
Shared Feature Extractor: ConfigurableMultimodalExtractor
```

**Potential Issues:**
1. **Value head too small** - Complex state space may need larger value network
2. **Shared features** - Policy and value may need different representations
3. **No value clipping** - Large negative values not being clipped
4. **No value normalization** - Returns not normalized to manageable scale

**Evidence of Issues:**
```
Gradient Norms (Value Network):
├─ Layer 0 weights: 0.026 → 0.023 (decreasing)
├─ Layer 2 weights: 0.022 → 0.033 (increasing)
├─ Layer 4 weights: 0.030 → 0.028 (stable)
└─ Total gradient norm fluctuating widely
```

Inconsistent gradient magnitudes suggest value function is struggling to find stable solution.

### 2.5 PPO Hyperparameter Analysis

**Current Settings (from config):**
```python
learning_rate: 3e-4 (constant)
batch_size: 256
n_steps: 1024
num_envs: 14
clip_range: 0.2
gamma: 0.99 (assumed)
gae_lambda: 0.95 (assumed)
```

**Assessment:**
- ✓ Learning rate reasonable for PPO
- ✓ Batch size good
- ✓ N_steps adequate
- ⚠️ Num_envs quite small (14) - more parallelism could help
- ❌ No learning rate scheduling
- ❌ No clip range annealing
- ❌ No entropy coefficient specified/visible

**Concern:** Clip fraction of 0.16 is moderate but indicates policy changes may be too large given the poor value estimates.

---

## 3. Detailed Recommendations (Prioritized)

### TIER 1: Critical Fixes (Implement Immediately)

#### 3.1 Fix Reward Scaling (Priority: CRITICAL ⚠️)

**Problem:** Time penalty makes level completion worthless.

**Solution A: Reduce Time Penalty (Recommended)**
```python
# Current (BROKEN):
TIME_PENALTY_PER_STEP = -0.01

# Proposed Fix:
TIME_PENALTY_PER_STEP = -0.0001  # 100x smaller

# Impact Analysis:
# Fast completion (1000 steps): +1.0 - 0.1 = +0.9 (POSITIVE! ✓)
# Slow completion (10000 steps): +1.0 - 1.0 = 0.0 (NEUTRAL)
# Max episode: -2.0 (manageable)
```

**Solution B: Increase Completion Reward**
```python
# Current:
LEVEL_COMPLETION_REWARD = 1.0

# Proposed:
LEVEL_COMPLETION_REWARD = 100.0  # Scale up by 100x
# Keep time penalty at -0.01
# Fast completion: +100 - 10 = +90 (POSITIVE! ✓)
# Slow completion: +100 - 100 = 0 (NEUTRAL)
```

**Solution C: Hybrid Approach (BEST)**
```python
# Increase completion reward moderately
LEVEL_COMPLETION_REWARD = 10.0

# Reduce time penalty moderately  
TIME_PENALTY_PER_STEP = -0.001

# Fast completion (1000 steps): +10 - 1 = +9 (GOOD! ✓)
# Moderate (5000 steps): +10 - 5 = +5 (STILL POSITIVE! ✓)
# Slow (10000 steps): +10 - 10 = 0 (NEUTRAL)
# Death (5000 steps): -0.5 - 5 = -5.5 (CLEAR PENALTY)
```

**Recommendation:** Use Solution C (Hybrid) for balanced learning.

#### 3.2 Add Dense Reward Shaping (Priority: CRITICAL ⚠️)

**Problem:** Rewards too sparse for platformer learning.

**Solution: Enable and Tune PBRS Navigation Rewards**
```python
# Current (likely disabled or too weak):
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.0001
PBRS_SWITCH_DISTANCE_SCALE = 0.05
PBRS_EXIT_DISTANCE_SCALE = 0.05

# Proposed (make 10-50x stronger):
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001  # 10x increase
PBRS_SWITCH_DISTANCE_SCALE = 0.5  # 10x increase
PBRS_EXIT_DISTANCE_SCALE = 0.5  # 10x increase

# Enable PBRS in config:
pbrs_weights = {
    "objective_weight": 1.0,  # Enable objective distance shaping
    "hazard_weight": 0.1,     # Mild hazard avoidance
    "impact_weight": 0.0,     # Disable for now
    "exploration_weight": 0.2  # Mild exploration bonus
}
```

**Expected Impact:**
```
Dense Reward Example:
Step 1: Move 50 pixels toward switch
  └─ Reward: +0.05 (distance improvement) - 0.001 (time) = +0.049

Step 2: Move 30 pixels toward switch  
  └─ Reward: +0.03 - 0.001 = +0.029

Step 3: Hit switch
  └─ Reward: +10.0 (milestone) + shaping

Agent now receives positive feedback hundreds of times per episode!
```

#### 3.3 Implement Value Function Fixes (Priority: CRITICAL ⚠️)

**Problem:** Value estimates collapsing to large negative numbers.

**Solution A: Add Value Clipping**
```python
# In PPO config or training code:
vf_clip_param = 10.0  # Clip value predictions to [-10, 10]
clip_value_loss = True  # Enable value loss clipping
```

**Solution B: Return Normalization**
```python
# Use VecNormalize wrapper:
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,  # Normalize returns to zero mean, unit variance
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)
```

**Solution C: Increase Value Network Capacity**
```python
# Current:
policy_kwargs = dict(
    net_arch=[512, 512, 512]  # Shared for policy and value
)

# Proposed:
policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 512, 512],  # Policy network
        vf=[1024, 1024, 512, 256]  # Larger value network with more layers
    )
)
```

**Recommendation:** Implement all three solutions:
1. Add value clipping (immediate stability)
2. Use return normalization (scale management)
3. Increase value capacity (better approximation)

#### 3.4 Fix Curriculum Learning (Priority: HIGH ⚠️)

**Problem:** Agent stuck at stage 2 with no recovery mechanism.

**Solution A: Add Regression Capability**
```python
class AdaptiveCurriculum:
    def __init__(self):
        self.advance_threshold = 0.70
        self.regress_threshold = 0.30  # NEW: Regress if below 30%
        self.min_episodes_for_regression = 200  # Longer than advance
        
    def should_regress(self, stage_idx, success_rate, episodes):
        """Check if should return to easier stage."""
        if stage_idx == 0:  # Can't regress from first stage
            return False
        if episodes < self.min_episodes_for_regression:
            return False
        if success_rate < self.regress_threshold:
            return True
        return False
```

**Solution B: Adaptive Thresholds**
```python
# Different thresholds for different stages
STAGE_THRESHOLDS = {
    0: 0.80,  # Simplest - should master
    1: 0.70,  # Simpler - high success needed
    2: 0.60,  # Simple - moderate success (LOWER!)
    3: 0.55,  # Medium - harder, lower threshold
    4: 0.50,  # Complex - very hard
    5: 0.45,  # Mine heavy - extremely challenging
    6: 0.40,  # Exploration - different goal
}
```

**Solution C: Mixed Training**
```python
class MixedCurriculumSampler:
    def sample_stage(self, current_stage):
        """Sample from current and previous stages."""
        # 70% from current stage
        # 20% from previous mastered stages (prevent forgetting)
        # 10% from next stage (preview harder content)
        
        r = random.random()
        if r < 0.70:
            return current_stage
        elif r < 0.90 and current_stage > 0:
            return random.randint(0, current_stage - 1)
        elif current_stage < self.max_stage:
            return current_stage + 1
        return current_stage
```

**Recommendation:** Implement all three:
1. Add regression (immediate help for stuck training)
2. Adaptive thresholds (more realistic goals)
3. Mixed training (prevent catastrophic forgetting)

### TIER 2: Important Improvements (Implement Soon)

#### 3.5 Add Learning Rate Scheduling

**Current:** Constant learning rate of 3e-4

**Proposed:**
```python
# Option A: Linear decay
learning_rate = LinearSchedule(
    initial_value=3e-4,
    final_value=3e-5,  # 10x reduction
    total_timesteps=1_000_000
)

# Option B: Cosine annealing
from torch.optim.lr_scheduler import CosineAnnealingLR
# (Requires custom PPO modification)

# Option C: Adaptive based on curriculum
def get_learning_rate(stage_idx):
    # Higher LR for easier stages (faster learning)
    # Lower LR for harder stages (more careful)
    base_lr = 3e-4
    return base_lr * (0.8 ** stage_idx)
```

**Recommendation:** Use Option C (adaptive) - higher LR helps recover from stuck stages.

#### 3.6 Increase Environment Parallelism

**Current:** 14 environments

**Proposed:** 32-64 environments
```python
num_envs = 32  # Minimum
# or
num_envs = 64  # Better for sample efficiency

# Adjust batch size to match:
batch_size = 512  # 2x increase
n_steps = 2048  # 2x increase

# This maintains approximately same update frequency but with more diverse data
```

**Benefits:**
- More diverse experience per update
- Better gradient estimates
- Faster wall-clock training
- Reduced correlation between samples

**Caution:** Requires more GPU memory. May need to reduce batch size if OOM.

#### 3.7 Add Entropy Coefficient Annealing

**Current:** Entropy stable at 1.79 (maximum)

**Problem:** High entropy is good for exploration but may prevent convergence.

**Proposed:**
```python
# Start with high entropy, decay gradually
ent_coef = LinearSchedule(
    initial_value=0.01,  # Standard PPO entropy coefficient
    final_value=0.001,   # Lower for later stage convergence
    total_timesteps=1_000_000
)

# Or curriculum-adaptive:
def get_ent_coef(stage_idx, episodes_in_stage):
    # Higher entropy when first entering a stage (exploration)
    # Lower entropy when experienced with stage (exploitation)
    if episodes_in_stage < 100:
        return 0.02  # High exploration
    elif episodes_in_stage < 300:
        return 0.01  # Medium
    else:
        return 0.005  # Focus on exploitation
```

**Recommendation:** Use curriculum-adaptive approach to balance exploration/exploitation.

#### 3.8 Implement Checkpoint Reward Shaping

**Problem:** Levels may have natural "checkpoints" that aren't rewarded.

**Proposed:**
```python
class CheckpointRewardWrapper:
    """Add rewards for reaching new areas of the level."""
    
    def __init__(self, env):
        self.env = env
        self.max_x = float('-inf')
        self.max_y = float('-inf')
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Reward for reaching new rightward position
        current_x = info.get('player_x', 0)
        if current_x > self.max_x:
            reward += 0.01 * (current_x - self.max_x)  # Small bonus
            self.max_x = current_x
            
        # Similar for vertical exploration if levels have vertical components
        current_y = info.get('player_y', 0)
        if current_y > self.max_y:
            reward += 0.005 * (current_y - self.max_y)
            self.max_y = current_y
            
        return obs, reward, done, truncated, info
```

### TIER 3: Optimization & Experimentation (Future Work)

#### 3.9 Implement Hindsight Experience Replay (HER)

For sparse reward problems, HER can be very effective:
```python
# After failed episode, relabel some steps as "successful" with alternative goals
# E.g., "reach position X" instead of "complete level"
# This provides positive examples even from failed episodes
```

#### 3.10 Add Auxiliary Tasks

Help value function learn better representations:
```python
# Auxiliary prediction tasks:
# - Predict future positions
# - Predict switch state changes
# - Predict collision events
# - Predict remaining time to goal
```

#### 3.11 Experiment with Different Architectures

Current: MLP baseline
Alternatives:
- Full HGT (Heterogeneous Graph Transformer) - for spatial reasoning
- GAT (Graph Attention) - lighter than HGT
- Recurrent policies (LSTM/GRU) - for temporal dependencies

#### 3.12 Population-Based Training (PBT)

Run multiple agents with different hyperparameters:
- Automatic hyperparameter tuning
- Exploit best performers
- Explore variations of successful configs

---

## 4. Implementation Plan

### Phase 1: Emergency Fixes (Week 1)
**Goal:** Stop the bleeding - make training viable

**Actions:**
1. ✅ Fix reward scaling (Solution C: hybrid approach)
2. ✅ Add value clipping and normalization
3. ✅ Enable dense PBRS rewards
4. ✅ Implement curriculum regression
5. ✅ Adjust advancement thresholds

**Expected Outcome:** 
- Success rate stabilizes or improves
- Value estimates return to reasonable range
- Agent can progress through curriculum

### Phase 2: Important Improvements (Week 2)
**Goal:** Optimize learning efficiency

**Actions:**
1. ✅ Implement learning rate scheduling
2. ✅ Increase environment parallelism
3. ✅ Add entropy coefficient annealing
4. ✅ Implement mixed curriculum training
5. ✅ Add checkpoint reward shaping

**Expected Outcome:**
- Faster learning
- Better sample efficiency
- Reduced training time

### Phase 3: Advanced Optimizations (Week 3-4)
**Goal:** Push performance boundaries

**Actions:**
1. ⬜ Experiment with HER
2. ⬜ Add auxiliary tasks
3. ⬜ Test alternative architectures
4. ⬜ Run hyperparameter sweeps
5. ⬜ Implement PBT if resources allow

**Expected Outcome:**
- Achieve 70%+ success on complex stages
- Stable, reproducible training
- Publishable results

---

## 5. Monitoring & Validation

### Key Metrics to Track

**Primary Metrics:**
```
✓ Success rate by curriculum stage (most important!)
✓ Value estimate range (should be in [-10, 10] range)
✓ Episode returns (should be positive for completions)
✓ Curriculum advancement rate (stages per 100k steps)
```

**Secondary Metrics:**
```
- Policy gradient norm
- Value gradient norm  
- Clip fraction (target: 0.1-0.2)
- KL divergence (target: < 0.02)
- Explained variance (target: > 0.7)
```

**Diagnostic Metrics:**
```
- Action entropy (should decay from 1.79 to ~1.0)
- Episodes per stage (track for regression detection)
- Time to completion (for successful episodes)
- Distance to goal over time (should decrease)
```

### Success Criteria

**Minimum Viable (Phase 1):**
- ✓ Value estimates in [-10, 10] range
- ✓ Positive returns for successful episodes
- ✓ No decline in success rate over time
- ✓ Can progress past stage 2

**Good Performance (Phase 2):**
- ✓ 60%+ success on stages 0-3
- ✓ 40%+ success on stages 4-5
- ✓ Reaches stage 6 (exploration)
- ✓ Stable value function (std < 5)

**Excellent Performance (Phase 3):**
- ✓ 80%+ success on stages 0-3
- ✓ 60%+ success on stages 4-5  
- ✓ 40%+ success on stage 6
- ✓ Fast completion times (< 3000 steps)

---

## 6. Code Changes Required

### 6.1 Reward Constants Update

**File:** `nclone/nclone/gym_environment/reward_calculation/reward_constants.py`

```python
# CRITICAL FIXES:

# Reduce time penalty 100x
TIME_PENALTY_PER_STEP = -0.0001  # was -0.01

# Increase completion reward 10x
LEVEL_COMPLETION_REWARD = 10.0  # was 1.0

# Increase switch reward proportionally
SWITCH_ACTIVATION_REWARD = 1.0  # was 0.1

# Increase dense reward shaping 10x
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001  # was 0.0001
PBRS_SWITCH_DISTANCE_SCALE = 0.5  # was 0.05
PBRS_EXIT_DISTANCE_SCALE = 0.5  # was 0.05

# Enable exploration rewards
EXPLORATION_CELL_REWARD = 0.01  # was 0.001 (10x increase)
```

### 6.2 Curriculum Configuration Update

**File:** `npp_rl/wrappers/curriculum_env.py` (or relevant curriculum file)

```python
# Add adaptive thresholds
STAGE_ADVANCE_THRESHOLDS = {
    0: 0.80,  # simplest
    1: 0.70,  # simpler
    2: 0.60,  # simple (LOWERED from 0.70)
    3: 0.55,  # medium
    4: 0.50,  # complex
    5: 0.45,  # mine_heavy
    6: 0.40,  # exploration
}

# Add regression thresholds
STAGE_REGRESS_THRESHOLDS = {
    1: 0.30,  # If simpler stage drops below 30%, go back to simplest
    2: 0.30,  # If simple stage drops below 30%, go back to simpler
    3: 0.25,  # Harder stages can have lower regression thresholds
    4: 0.20,
    5: 0.15,
    6: 0.15,
}

# Require more episodes before regression (avoid thrashing)
MIN_EPISODES_FOR_REGRESSION = 200  # was: no regression at all
```

### 6.3 PPO Configuration Update

**File:** `config.json` or training script

```python
{
  // Increase parallelism
  "num_envs": 32,  // was 14
  
  // Adjust batch size
  "batch_size": 512,  // was 256
  "n_steps": 2048,  // was 1024
  
  // Enable PBRS
  "pbrs_weights": {
    "objective_weight": 1.0,
    "hazard_weight": 0.1,
    "impact_weight": 0.0,
    "exploration_weight": 0.2
  },
  
  // Add value clipping
  "vf_clip_param": 10.0,
  "clip_value_loss": true,
  
  // Adaptive learning rate
  "learning_rate": {
    "type": "linear",
    "initial": 3e-4,
    "final": 3e-5
  },
  
  // Entropy annealing
  "ent_coef": {
    "type": "linear",
    "initial": 0.02,
    "final": 0.005
  }
}
```

### 6.4 Add VecNormalize Wrapper

**File:** `npp_rl/agents/training.py`

```python
from stable_baselines3.common.vec_env import VecNormalize

# After creating vec_env:
env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,  # CRITICAL: Normalize returns
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
    epsilon=1e-8,
)
```

---

## 7. Alternative Approaches to Consider

### 7.1 Reset to Behavioral Cloning

**Rationale:** Current RL training is not working. BC pretraining might provide better initialization.

**Approach:**
```python
# Train pure BC model for longer
bc_epochs = 200  # was 50
bc_batch_size = 256
# Then use as initialization for RL

# Or: Mix BC loss with RL loss
total_loss = rl_loss + 0.1 * bc_loss  # Keep some BC signal throughout
```

**Pros:** Guaranteed sensible policy to start from
**Cons:** May not generalize beyond demonstrations

### 7.2 Shaped Reward from Demonstrations

**Rationale:** Use demonstrations to define reward shaping.

**Approach:**
```python
# Learn potential function from demonstrations
# Reward agent for matching demonstration state distributions
# Gradually reduce demo dependence

reward = base_reward + alpha * similarity_to_demo
# alpha decays from 1.0 to 0.0 over training
```

### 7.3 Curriculum from Demonstrations

**Rationale:** Order levels by demonstration success, not by arbitrary categories.

**Approach:**
```python
# Sort levels by:
# - Demonstration success rate
# - Average episode length  
# - Number of deaths in demos
# - Complexity metrics (entities, hazards, etc.)

# Start with levels that demos solve quickly with few deaths
```

### 7.4 Hierarchical RL

**Rationale:** Break task into subtasks (reach switch, reach exit).

**Approach:**
```python
# High-level policy: Choose subtask (REACH_SWITCH, REACH_EXIT, EXPLORE)
# Low-level policy: Execute motor actions to achieve subtask
# Separate reward/value functions for each level

# Benefits:
# - Clearer credit assignment
# - Reusable low-level skills
# - Natural curriculum (master subtasks individually)
```

---

## 8. Expected Outcomes After Fixes

### Quantitative Predictions

**After Phase 1 (Emergency Fixes):**
```
Week 1 Results:
├─ Value estimates: [-5, 5] range (vs. current [-7, 0])
├─ Success rate trend: Stable or improving (vs. declining)
├─ Stage progression: Reach stage 3-4 (vs. stuck at stage 2)
└─ Episode returns: Positive for wins (vs. negative even for wins)
```

**After Phase 2 (Important Improvements):**
```
Week 2 Results:
├─ Stage 0-2 success: 70%+ (vs. current 4% on stage 2)
├─ Stage 3-4 success: 40%+ (vs. never reached)
├─ Training speed: 2x faster (more envs)
└─ Sample efficiency: 50% improvement (better rewards)
```

**After Phase 3 (Advanced Optimizations):**
```
Week 3-4 Results:
├─ Stage 0-3 success: 80%+
├─ Stage 4-5 success: 60%+
├─ Stage 6 success: 40%+
└─ Ready for deployment/publication
```

### Qualitative Predictions

**Short-term (Week 1):**
- Training will no longer degrade performance
- Agent will show clear learning on easier stages
- Value function will stabilize
- Tensorboard curves will show upward trends

**Medium-term (Week 2-3):**
- Agent will exhibit competent platforming behavior
- Successful navigation through multi-step levels
- Consistent switch activation and exit reaching
- Emergent strategies (wall jumping, hazard avoidance)

**Long-term (Week 4+):**
- Near-human performance on easier stages
- Creative solution finding on harder stages
- Robust to environment variations
- Transferable skills across level types

---

## 9. Risk Assessment & Mitigation

### Risk 1: Reward Scaling Overcorrection

**Risk:** Making rewards too dense might cause local optima.

**Mitigation:**
- Start with 10x changes, not 100x
- Monitor: Agent should still seek exits, not just explore
- Fallback: Gradually reduce shaping weights over training

### Risk 2: Curriculum Still Too Hard

**Risk:** Even with fixes, stage 2→3 might be too difficult.

**Mitigation:**
- Add intermediate stages
- Implement better level difficulty metrics
- Allow longer training periods per stage (500+ episodes)

### Risk 3: Value Function Still Unstable

**Risk:** Clipping and normalization might not be enough.

**Mitigation:**
- Consider Huber loss for value function (more robust)
- Increase value network size significantly
- Try separate feature extractors for policy and value
- Consider V-trace or other off-policy corrections

### Risk 4: Increased Computation Requirements

**Risk:** More envs and larger batches need more GPU memory.

**Mitigation:**
- Use mixed precision training (already enabled)
- Gradient accumulation if needed
- Cloud compute scaling (already using EC2 with GPU)

### Risk 5: Hyperparameter Sensitivity

**Risk:** Fixes might work but be very sensitive to exact values.

**Mitigation:**
- Document all changes carefully
- Run ablation studies
- Use population-based training for robustness
- Create "conservative" and "aggressive" config variants

---

## 10. References & Best Practices

### Academic References

1. **Proximal Policy Optimization (PPO):**
   - Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
   - arXiv:1707.06347

2. **Curriculum Learning:**
   - Bengio et al. (2009) "Curriculum Learning"
   - Narvekar et al. (2020) "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey"

3. **Reward Shaping:**
   - Ng et al. (1999) "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"
   - Potential-based reward shaping (PBRS) theory

4. **Value Function Stability:**
   - Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
   - Hessel et al. (2018) "Rainbow: Combining Improvements in Deep Reinforcement Learning"

5. **Sparse Rewards:**
   - Andrychowicz et al. (2017) "Hindsight Experience Replay"
   - Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"

### Practical Resources

1. **OpenAI Spinning Up:**
   - https://spinningup.openai.com/
   - Excellent PPO implementation guide
   - Hyperparameter recommendations

2. **Stable-Baselines3 Documentation:**
   - https://stable-baselines3.readthedocs.io/
   - PPO best practices
   - Common pitfalls and solutions

3. **RL Tips and Tricks:**
   - 37 Implementation Details matter for PPO (Huang et al. 2022)
   - Deep RL Bootcamp lectures (Berkeley)

### Domain-Specific Insights

**Platformer Games:**
- Dense reward shaping essential (unlike Atari where sparse works)
- Checkpoint-based rewards highly effective
- Curriculum from easy to hard levels standard practice
- Temporal credit assignment critical (long episodes)

**Curriculum Learning:**
- Success-based advancement standard (60-80% threshold)
- Regression capability important but often overlooked
- Mixed training prevents catastrophic forgetting
- Adaptive thresholds better than fixed

**PPO Hyperparameters:**
- Clip range: 0.1-0.3 (standard: 0.2)
- Learning rate: 1e-4 to 5e-4 for vision, 3e-4 standard
- Batch size: 32-512 depending on problem
- GAE lambda: 0.9-0.99 (standard: 0.95)
- Entropy coefficient: 0.001-0.01 (decay over time)

---

## 11. Conclusion

The current training run exhibits **critical failure modes** that prevent learning:

1. ⚠️ **Reward scaling makes success impossible to learn**
2. ⚠️ **Value function has completely collapsed**
3. ⚠️ **Curriculum stuck with no recovery mechanism**

These are **not minor issues** - they represent fundamental problems that will prevent any meaningful learning regardless of training duration.

**The good news:** All identified issues are fixable with well-understood techniques. The agent maintains good exploration (high entropy), and the training infrastructure works correctly.

**Recommended immediate action:**
1. Stop current training (it's not productive)
2. Implement Phase 1 emergency fixes (1-2 days)
3. Restart training with fixed configuration
4. Monitor for improvement within first 100k steps

**Expected timeline to working agent:**
- Phase 1 fixes: 1-2 days implementation
- Initial validation: 3-5 days training
- Full curriculum mastery: 2-3 weeks

**Confidence level:**
- High confidence (>90%) that Phase 1 fixes will stop the degradation
- Moderate confidence (70%) that agent will reach stage 4-5 with Phase 1+2
- Lower confidence (50%) on final performance without hyperparameter tuning

The path forward is clear, and the fixes are standard RL best practices. With proper reward scaling, value function stability, and curriculum fixes, this agent should learn effectively.

---

## Appendix A: Detailed TensorBoard Metrics

### Complete Metrics Catalog (152 total)

**Curriculum Metrics (20):**
- Success rates per stage (7 stages)
- Episodes per stage
- Current stage index
- Advancement criteria
- Overall success rate

**Action Metrics (52):**
- Action frequencies (6 actions)
- Action transitions (6×6 = 36)
- Jump analysis (frequency, directional %)
- Movement analysis (active %, bias, stationary %)
- Entropy

**Loss Metrics (8):**
- Total loss
- Policy gradient loss
- Value loss
- Entropy loss
- (Duplicates in train/ and training/ namespaces)

**Training Metrics (7):**
- Learning rate
- Clip fraction
- Approx KL divergence
- Clip range
- Explained variance

**Value Metrics (4):**
- Estimate mean
- Estimate min
- Estimate max
- Estimate std

**Gradient Metrics (30):**
- Layer-wise gradient norms for:
  - Feature extractor (global CNN, player CNN, MLPs)
  - Policy network (3 layers)
  - Value network (3 layers)
  - Total gradient norm

**Episode Metrics (2):**
- Success rate
- Failure rate

**Performance Metrics (6):**
- FPS (instant and mean)
- Steps per second
- Rollout time
- Elapsed time

### Key Observation Patterns

**Value Collapse Pattern:**
```
Timestamp: Step 0k → 1000k
Value Mean: -0.06 → -4.33 (monotonic decrease)
Value Min: -0.46 → -7.35 (monotonic decrease)
Value Max: 0.39 → -0.31 (crossed zero around step 400k)
```

**Curriculum Stuck Pattern:**
```
Stage 0 (simplest): 122 episodes, 100% success - STABLE ✓
Stage 1 (simpler): 485 episodes, 68% success - DECLINING ⚠️
Stage 2 (simple): 435 episodes, 4% success - STUCK ❌
Stages 3-6: 0 episodes - NEVER REACHED ❌
```

**Action Entropy Pattern:**
```
Initial: 1.754 (random policy)
Training: 1.789-1.791 (stable)
Final: 1.791 (maximum maintained)

Interpretation: Agent maintains maximum exploration
This is GOOD for exploration, BAD if not learning
Indicates: Reward signal not strong enough to drive specialization
```

---

## Appendix B: Training Configuration Deep Dive

### Full Configuration Analysis

```json
{
  "experiment_name": "mlp-baseline-1026",
  "architectures": ["mlp_baseline"],  // Simplest architecture
  
  // Datasets
  "train_dataset": "/home/ubuntu/datasets/train",
  "test_dataset": "/home/ubuntu/datasets/test",
  "replay_data_dir": "../nclone/bc_replays",
  
  // Pretraining
  "no_pretraining": false,  // BC enabled
  "test_pretraining": false,  // But not testing-only
  "bc_epochs": 50,  // Standard BC training
  "bc_batch_size": 128,
  
  // RL Training
  "total_timesteps": 1000000,  // 1M steps (moderate)
  "num_envs": 14,  // ⚠️ LOW - should be 32-64
  "eval_freq": 100000,  // Every 100k steps
  "save_freq": 500000,  // Every 500k steps
  "num_eval_episodes": 10,
  
  // Hardware (looks good)
  "hardware_profile": "Auto-1xGPU-85GB",
  "num_gpus": 1,
  "mixed_precision": true,  // ✓ Good
  
  // Curriculum
  "use_curriculum": true,  // ✓ Enabled
  "curriculum_start_stage": "simplest",  // ✓ Good
  "curriculum_threshold": 0.7,  // ⚠️ Too high for all stages
  "curriculum_min_episodes": 100,  // ⚠️ Might be too few
  
  // Frame stacking (disabled - probably fine for MLP baseline)
  "enable_visual_frame_stacking": false,
  "enable_state_stacking": false,
  
  // Auto-detected hyperparameters
  "hardware_profile_settings": {
    "batch_size": 256,  // Standard
    "n_steps": 1024,  // Standard
    "learning_rate": 0.0003  // Standard
  }
}
```

### Missing Configurations (Defaults Assumed)

Important PPO parameters not visible in config:
- `gamma`: Likely 0.99 (standard)
- `gae_lambda`: Likely 0.95 (standard)
- `ent_coef`: Not specified (may default to 0.0!)
- `vf_coef`: Likely 0.5 (standard)
- `clip_range`: Likely 0.2 (standard)
- `max_grad_norm`: Likely 0.5 (standard)

### Reward Configuration (from code analysis)

Based on reward_constants.py:
- Using "completion_focused_config" (likely)
- PBRS enabled but with weak scaling
- Exploration rewards enabled but small
- Time penalty too large (-0.01)

---

**End of Analysis**

---

*Generated: 2025-10-27*  
*Author: OpenHands AI Assistant*  
*Training Run Analyzed: mlp-baseline-1026*  
*Analysis Duration: Comprehensive (~2 hours)*
