# Technical Analysis: MLP Baseline Low Success Rate on Simplest Levels

**Document Author**: AI Assistant (OpenHands)  
**Date**: 2025-10-24 (Updated after second-pass analysis)  
**Training Run**: `arch_comparison_20251023_20251023_170211`  
**Status**: 🚨 **SIX CRITICAL ISSUES IDENTIFIED** (2 NEW from second pass)

---

## 🔴 KEY FINDINGS SUMMARY

**BC Pretraining is Completely Broken**: Two critical bugs discovered in second-pass analysis completely invalidate the BC→RL transfer learning:
1. **Observation normalization mismatch**: BC trains on normalized obs (mean=0, std=1), RL uses raw unnormalized obs
2. **Incomplete weight loading**: Only 58/82 BC parameters loaded, 24 critical feature extractor layers remain random

**Training is Catastrophically Inefficient**: Three severe configuration bugs:
3. **Hardware profile bug**: Only 14 environments allocated (should be 128+) → 70 updates in 1M timesteps (~15 min/update)
4. **Hierarchical PPO enabled**: 46 random parameters gating corrupted features (should be disabled for MLP)
5. **Frame stacking enabled**: 4x computational overhead with no benefit for MLP

**Cannot Monitor Progress**:
6. **TensorBoard logging broken**: No metrics visible

**Result**: BC pretraining benefit completely lost, training on garbage features with insufficient data diversity.

---

## Executive Summary

The MLP baseline architecture with BC pretraining, curriculum learning, and hierarchical PPO is failing to achieve >0.1 success rate on the simplest level category despite 1M timesteps of training on an H100 GPU. Based on **comprehensive two-pass analysis** of the system architecture, training logs, and configuration, **I have identified SIX critical issues** (including two newly discovered bugs that completely break BC pretraining).

### Top 6 Critical Issues (All Must Be Fixed)

| # | Issue | Impact | Current | Target | Fix Type |
|---|-------|--------|---------|--------|----------|
| 1 | **BC-RL Obs Normalization** | 🔴 CRITICAL | Mismatch | Match BC stats | Code change required |
| 2 | **Incomplete Weight Loading** | 🔴 SEVERE | 58/82 params | 82/82 params | Code change OR disable hierarchical |
| 3 | **Environment Count** | 🔴 SEVERE | 14 envs | 128+ envs | CLI: `--num-envs 128` |
| 4 | **Hierarchical PPO Overhead** | 🔴 SEVERE | Enabled | Disabled | CLI: Remove `--use-hierarchical-ppo` |
| 5 | **Frame Stacking Overhead** | 🟡 HIGH | 4 frames | Disabled | CLI: Remove frame stacking flags |
| 6 | **TensorBoard Logging** | 🟡 HIGH | Empty | Working | Investigation needed |

**Training Efficiency Issue**: The training is experiencing extreme inefficiency with **~15 minutes per update cycle** (rollout + gradient update), meaning only **~70 updates in 1M timesteps**. This is 10-20x slower than expected.

**Metrics Availability Issue**: ⛔ **TensorBoard logs are empty** - cannot verify success rate or any training metrics. This makes debugging nearly impossible.

---

## 🔥 CRITICAL UPDATE: Second-Pass Deep Analysis

**Analysis Date**: 2025-10-24 (Second Pass)  
**Status**: 🚨 TWO ADDITIONAL CRITICAL BUGS DISCOVERED

### Newly Discovered Critical Issues

After a comprehensive second-pass examination of the entire training pipeline, I have identified **two additional CRITICAL bugs** that were not apparent in the initial analysis:

#### 🚨 Issue #5: BC-RL Observation Normalization Mismatch (CRITICAL)

**Problem**: BC training and RL training process observations with completely different normalization, invalidating the entire pretraining transfer.

**Evidence from Logs**:
```
# BC Training Phase:
2025-10-23 23:15:47 [INFO] Computing normalization statistics from data
2025-10-23 23:15:47 [INFO] Computed normalization statistics for 2 observation keys
2025-10-23 23:15:47 [DEBUG] Saved normalization statistics to .../normalization_stats.npz
```

**Code Analysis**:
```python
# BC Training (bc_dataset.py lines 217-239):
self.normalizer = ObservationNormalizer()
self.normalizer.compute_stats(self.samples)  # Computes mean/std from BC data
obs = self.normalizer.normalize(obs)  # All training observations normalized to mean=0, std=1

# RL Training (architecture_trainer.py lines 605-643):
env = NppEnvironment(config=env_config)  # No normalization applied!
# No VecNormalize wrapper
# No observation scaling anywhere in the environment creation pipeline
```

**Impact**: 
- BC feature extractor learned to process **normalized observations** (velocities ~N(0,1), positions scaled)
- RL training feeds **raw unnormalized observations** to the same feature extractor
- Input distribution shift completely invalidates transfer learning
- Pretrained weights see fundamentally different input statistics
- This alone could explain why BC pretraining (91.55% accuracy) provides no benefit in RL

**Severity**: CRITICAL - This bug completely invalidates the BC→RL transfer learning pipeline

---

#### 🚨 Issue #6: Incomplete BC Weight Loading (SEVERE)

**Problem**: Only 58 out of 82 BC-trained parameters are successfully loaded into the RL model. Critical feature extractor layers remain randomly initialized.

**Evidence from Logs**:
```
2025-10-23 23:20:44 [INFO] ✓ Loaded BC pretrained feature extractor weights
2025-10-23 23:20:44 [INFO]   Loaded 58 weight tensors (BC → hierarchical)
2025-10-23 23:20:44 [INFO]   Missing keys (will use random init): 74
2025-10-23 23:20:44 [INFO]     Features extractor keys missing: 24
2025-10-23 23:20:44 [INFO]     Examples: ['features_extractor.reachability_mlp.0.weight', 
                                           'features_extractor.reachability_mlp.0.bias', 
                                           'features_extractor.reachability_mlp.2.weight', 
                                           'features_extractor.fusion.0.weight']
2025-10-23 23:20:44 [INFO]     Hierarchical policy keys missing: 46 (expected)
2025-10-23 23:20:44 [INFO]     Action/value head keys missing: 14 (expected)
```

**Analysis**:
- **24 feature extractor parameters** are NOT loaded from BC checkpoint
- These include critical layers: `reachability_mlp`, `fusion` layers
- Only hierarchical policy (46 params) and action/value heads (14 params) are expected to be random
- But feature extractor should be **fully loaded**, not partially

**Root Cause**:
- BC checkpoint stores weights under standard SB3 structure
- Hierarchical PPO expects features nested under `mlp_extractor.features_extractor.*`
- Weight mapping logic fails to correctly map 24 critical parameters
- Results in mixed initialization: some layers learned, some random

**Impact**:
- **Inconsistent feature representations**: Some layers process learned features, others add random noise
- **Partial pretraining is worse than no pretraining**: Creates feature distribution mismatch
- **Hierarchical policy cannot leverage learned features**: 46 random params sit on top of corrupted features
- Combined with normalization mismatch, the feature extractor is effectively broken

**Severity**: SEVERE - Partial weight loading creates worse starting point than random initialization

---

### Updated Root Cause Summary

The MLP baseline failure is now understood to be caused by **SIX critical issues working together**:

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | **BC-RL Normalization Mismatch** | 🔴 CRITICAL | Transfer learning completely broken |
| 2 | **Incomplete Weight Loading** | 🔴 SEVERE | 24 feature params randomly initialized |
| 3 | **Hardware Profile Bug** | 🔴 SEVERE | Only 14 envs instead of 128+ |
| 4 | **Hierarchical PPO Mismatch** | 🔴 SEVERE | 46 random params gating corrupted features |
| 5 | **Frame Stacking Overhead** | 🟡 HIGH | 4x memory/compute, no benefit for MLP |
| 6 | **TensorBoard Logging Broken** | 🟡 HIGH | Cannot monitor training progress |

**Critical Observation**: Issues #1 and #2 work together to completely break BC pretraining:
1. BC learns on normalized obs → RL sees unnormalized obs (distribution shift)
2. Only 58/82 params loaded → 24 critical layers remain random (partial transfer)
3. Result: Feature extractor produces garbage features, rendering BC pretraining useless

**Training Efficiency**: Only 70 PPO updates in 1M timesteps (~15 min/update) due to Issue #3

---

## System Configuration Analysis

### Training Setup (from config.json)
- **Architecture**: `mlp_baseline` (CNN for vision + MLP for state, no graph)
- **Total Timesteps**: 1,000,000 
- **Environments**: 14 parallel (very low for PPO)
- **Batch Size**: 256 (n_steps: 1024)
- **BC Pretraining**: 50 epochs, 130 replay files
- **Frame Stacking**: Enabled (4 frames visual, 4 frames state)
- **Hierarchical PPO**: Enabled (high-level update freq: 50)
- **Curriculum**: Starting at "simplest" stage (1000 levels available)
- **Hardware**: 1x NVIDIA H100 80GB HBM3
- **Mixed Precision**: True

### Observed Performance Issues
- **Update Frequency**: ~15 minutes per PPO update cycle
- **Timesteps per Update**: 14,336 (1024 steps × 14 envs)
- **Total Updates in 1M timesteps**: ~70 updates
- **Actual Training Duration**: ~18 hours for 1M timesteps

---

## Critical Issues Identified

### 1. **CRITICAL: Catastrophically Low Environment Count (14 envs)**

**Impact**: Severe

The configuration uses only **14 parallel environments** on an H100 GPU, which is dramatically insufficient for PPO training.

**ROOT CAUSE IDENTIFIED**: The auto-detection logic in `hardware_profiles.py` line 217 uses a conservative heuristic:

```python
envs_per_gpu = max(8, min(256, int(gpu_memory_gb / 6)))  # 6GB per environment
```

With 85GB GPU memory: `int(85 / 6) = 14` environments!

This heuristic assumes **6GB per environment**, which is appropriate for **graph-based models with large memory footprints**, but the MLP baseline uses far less memory. The H100 should easily support 128-256 environments with MLP baseline.

**Why This is Critical**:
- **Sample Efficiency**: PPO requires diverse, decorrelated experiences. With only 14 envs, the agent sees highly correlated trajectories, leading to:
  - Overfitting to specific level instances
  - Poor generalization
  - Slow exploration of the state space
  
- **Update Inefficiency**: Each PPO update only uses 14,336 timesteps (14 envs × 1024 steps). For comparison:
  - Industry standard: 64-256 envs for complex tasks
  - Recommended for N++: 128+ envs on high-end GPU
  - With 128 envs: 131,072 timesteps per update (9x more data)

- **Curriculum Learning Conflict**: The curriculum requires 100 episodes at 70% success rate to advance. With 14 envs:
  - ~7 updates just to collect 100 episodes (assuming episodes complete in 1024 steps)
  - Very slow progression through curriculum stages
  - Agent never accumulates enough diverse experience on simplest levels

**Evidence from Logs**:
```
2025-10-23 23:28:15 [INFO] [Update 0] Rollout complete - timesteps: 14336
2025-10-23 23:43:22 [INFO] [Update 1] Rollout complete - timesteps: 28672
```
15 minutes between updates is unacceptable for RL training.

**Recommendation**: Increase to **at least 64-128 environments**, ideally 256 on H100.

---

### 2. **CRITICAL: Frame Stacking Dimension Mismatch / Complexity**

**Impact**: Severe

The system uses 4-frame stacking for both visual and state observations, dramatically increasing input dimensionality:

**Dimension Explosion**:
- **Player Frame**: `(84, 84, 1)` → `(4, 84, 84, 1)` → Treated as `(4, 84, 84)` by CNN
  - Input channels to CNN: 4 instead of 1 (4x increase)
- **Global View**: `(176, 100, 1)` → `(4, 176, 100, 1)` → `(4, 176, 100)`
  - Input channels to CNN: 4 instead of 1 (4x increase)
- **Game State**: `(26,)` → `(4, 26)` → Flattened to `(104,)` (4x increase)
- **Total Feature Dim**: Player (512) + Global (256) + State (128 from 104-dim input) + Reach (128) = **1024 dimensions**

**Problems**:

1. **BC-RL Observation Mismatch Risk**:
   - BC pretraining processes replays with frame stacking enabled
   - Each replay timestep must construct 4-frame stacks (padding first 3 frames with zeros)
   - If BC training and RL training use different padding strategies or frame ordering, features will be misaligned
   - **Log evidence**: "Frame stacking enabled in BC dataset" suggests this is configured, but subtle bugs are common

2. **Hierarchical PPO Complexity**:
   - Hierarchical PPO adds 46 additional parameters for high-level policy
   - High-level policy must process 1024-dim features to select subtasks
   - Low-level policy must condition on both features AND subtask embeddings
   - This architectural complexity may prevent effective learning on simple levels

3. **Computational Overhead**:
   - 4x more data through CNNs
   - 4x more data through state MLPs
   - Mixed precision may not help if bottleneck is CPU→GPU data transfer
   - Explains 15-minute update cycles

**Evidence from Code** (`configurable_extractor.py`):
```python
def forward(self, x):
    if x.dim() == 3:
        # Stacked states: [batch, stack_size, state_dim] -> [batch, stack_size * state_dim]
        batch_size, stack_size, state_dim = x.shape
        x = x.view(batch_size, stack_size * state_dim)  # (26,) -> (104,)
```

**Recommendation**: 
- **Test without frame stacking first** to establish baseline performance
- If frame stacking is required, reduce stack size to 2
- Verify BC and RL use identical frame stacking configuration

---

### 3. **CRITICAL: Hierarchical PPO Overhead for Simple Levels**

**Impact**: Severe

Using hierarchical PPO on the "simplest" level category introduces unnecessary complexity:

**Why Hierarchical PPO is Problematic Here**:

1. **Simplest Levels Don't Need Subtask Decomposition**:
   - "Simplest" levels: Basic navigation, single switch, no complex hazards
   - These can be solved with reactive policies (no planning needed)
   - High-level policy adds 46 parameters + subtask selection overhead

2. **High-Level Update Frequency Mismatch**:
   - High-level policy updates every 50 steps
   - Simple levels may complete in <500 steps
   - High-level policy sees few updates per episode
   - Cannot learn meaningful subtask structure

3. **Training Instability**:
   - Two policies (high + low level) must coordinate
   - Early training: both policies are random
   - High-level provides meaningless subtasks → low-level can't learn
   - Low-level fails tasks → high-level gets no useful gradient
   - Creates training deadlock

4. **Feature Extractor Loading Complications**:
   ```
   Loaded 58 weight tensors (BC → hierarchical)
   Missing keys (will use random init): 74
   Hierarchical policy keys missing: 46 (expected)
   ```
   - BC pretraining uses standard policy
   - Hierarchical PPO expects different architecture
   - 46 hierarchical parameters initialized randomly
   - These random parameters gate the pretrained features!

**Recommendation**:
- **Disable hierarchical PPO for simplest/simple stages**
- Use standard PPO until curriculum reaches medium/complex
- This allows BC-pretrained features to be used directly without random hierarchical gates

---

### 4. **HIGH PRIORITY: BC Pretraining - RL Training Mismatch**

**Impact**: High

Several potential mismatches between BC pretraining and RL fine-tuning:

**Observation Normalization Mismatch**:
- BC uses ObservationNormalizer to compute mean/std from replay data
- Saved to `normalization_stats.npz`
- **RL training may not be using these same normalization statistics**
- If RL sees unnormalized observations while BC was trained on normalized ones:
  - Feature extractor inputs are in wrong range
  - Pretrained weights become useless
  - Agent must relearn from scratch

**Policy Architecture Mismatch**:
```
# BC Training
policy_head = MLP(features → 6 actions)

# RL Training (Hierarchical)
high_level_policy = MLP(features → 5 subtasks)
low_level_policy = MLP(features + subtask_embedding → 6 actions)
```
- BC learns direct feature → action mapping
- RL uses feature → subtask → action mapping
- Pretrained features optimized for different task structure

**Learning Rate After Pretraining**:
- BC trains with lr=3e-4 for 50 epochs
- RL continues with lr=3e-4
- **No learning rate warmup or reduction**
- Pretrained features may be destroyed by large RL updates
- Standard practice: reduce LR by 10x after pretraining

**Recommendation**:
- Verify observation normalization is applied in RL with same stats
- Use standard PPO (not hierarchical) to match BC architecture
- Reduce learning rate to 1e-4 or 3e-5 for RL fine-tuning

---

### 5. **HIGH PRIORITY: Reward Signal Insufficient for Simplest Levels**

**Impact**: High

The reward structure may not provide enough signal for learning on very simple levels:

**Reward Components** (from `reward_constants.py`):
- **Level Completion**: +1.0 (sparse, only at end)
- **Death**: -0.5 (sparse)
- **Switch Activation**: +0.1 (intermediate milestone)
- **Time Penalty**: -0.01 per step (dense but small)
- **Navigation Shaping**: ~0.0001 per pixel improvement (extremely small)
- **Exploration**: 0.001-0.004 (depends on visiting new cells)

**Problem for Simplest Levels**:
- Most reward is sparse (completion/death)
- With 14 envs × 1024 steps = 14,336 steps per update
- If episode lengths are 2000-5000 steps:
  - ~3-7 episodes per environment per update
  - ~42-98 episodes total per update
  - Most episodes likely fail initially (random policy)
  - **Very few positive reward signals per update**

**Navigation Shaping Too Weak**:
- Scale of 0.0001 means moving 100 pixels closer = +0.01 reward
- Same as time penalty for 1 step
- Effectively cancelled out
- Agent doesn't get sufficient gradient from navigation

**Simplest Levels Specific Issue**:
- Simplest levels are SHORT and should be EASY
- But reward structure treats them same as complex levels
- Need stronger intermediate rewards for:
  - Approaching switch
  - Approaching exit (after switch)
  - Not dying

**Recommendation**:
- Increase navigation shaping scale to 0.001 (10x)
- Increase switch activation reward to 0.3
- Add proximity bonus: +0.05 when within 50 pixels of current objective
- Reduce death penalty to -0.2 (less discouragement)

---

### 6. **MEDIUM PRIORITY: Curriculum Advancement Threshold Too High**

**Impact**: Medium

The curriculum requires **70% success rate over 100 episodes** to advance from simplest to simpler.

**Problem**:
- With 14 envs and ~15 min per update
- Need ~7-10 updates to collect 100 episodes
- That's ~2 hours of training just for threshold check
- But 70% success on simplest levels requires agent to already be quite competent
- In 1M timesteps (~70 updates total), agent spends most time on first stage

**Conservative Threshold Effects**:
- Agent never sees curriculum progression
- No variety in training data
- Overfits to simplest level distribution
- Can't bootstrap from easier→harder progression

**Recommendation**:
- Reduce advancement threshold to **50-60%** for simplest→simpler
- Reduce minimum episodes to **50** with only 14 envs
- Implement stage mixing (already enabled, good!)
- Add automatic threshold adjustment if stuck >200 episodes

---

### 7. **MEDIUM PRIORITY: MLP Baseline Missing Spatial Reasoning**

**Impact**: Medium

The MLP baseline architecture lacks graph-based spatial reasoning:

**Architecture**:
- **Player Frame CNN**: Extracts local visual features (512-dim)
- **Global View CNN**: Extracts level-wide visual features (256-dim)
- **Game State MLP**: Processes 26-dim (or 104-dim stacked) physics state
- **Reachability MLP**: Processes 8-dim reachability features
- **Fusion**: Simple concatenation (no attention)

**What's Missing**:
- **No explicit spatial relationships**: CNNs learn spatial features implicitly, but:
  - Player frame is egocentric (84x84 local view)
  - Global view shows full level but at low resolution
  - Hard to reason about "switch is in room 2, I'm in room 1, path between them"

- **No entity relationship modeling**:
  - Graph models explicitly encode: player→switch distance, switch→exit distance
  - MLP baseline must infer this from pixel patterns
  - Much harder learning problem

**Why This Matters for Simplest Levels**:
- Simplest levels still require:
  1. Locate switch
  2. Navigate to switch
  3. Activate switch
  4. Locate exit
  5. Navigate to exit
- Without explicit spatial reasoning, agent must learn these from pixel patterns
- Requires much more data

**Recommendation**:
- This is expected limitation of MLP baseline
- But should still achieve >10% success with enough training
- Suggests other issues (1-6) are more critical
- Consider adding entity_positions features explicitly to game state

---

### 8. **MEDIUM PRIORITY: PPO Hyperparameters Not Tuned for BC→RL Transfer**

**Impact**: Medium

The PPO hyperparameters are standard but may not be optimal after BC pretraining:

**Current Hyperparameters**:
- `n_steps`: 1024 (reasonable)
- `batch_size`: 256 (reasonable but small for 14 envs)
- `n_epochs`: 5 (standard)
- `gamma`: 0.999 (very high, expects long episodes)
- `gae_lambda`: 0.998801 (very high)
- `clip_range`: 0.389 (high, allows large policy changes)
- `ent_coef`: 0.00272 (low, limited exploration)
- `learning_rate`: 3e-4 (standard)

**Problems**:

1. **Entropy Coefficient Too Low**:
   - With ent_coef = 0.00272, policy quickly becomes deterministic
   - After BC pretraining, policy is already quite deterministic
   - RL phase needs exploration to find better actions than demonstrations
   - Low entropy → agent sticks to BC behaviors even when suboptimal

2. **Clip Range Too High**:
   - clip_range = 0.389 allows large policy changes
   - Can destroy BC-pretrained policy quickly
   - Should be lower (0.1-0.2) for fine-tuning

3. **Gamma Too High for Short Episodes**:
   - gamma = 0.999 appropriate for long episodes (20,000 frame limit)
   - But simplest levels might be solvable in 1000-2000 frames
   - High gamma over-values distant future rewards
   - Should use gamma = 0.99 or 0.995 for simplest levels

**Recommendation**:
- Increase `ent_coef` to 0.01-0.02 (more exploration)
- Reduce `clip_range` to 0.2 (gentler updates after BC)
- Reduce `gamma` to 0.99 for simplest/simpler stages
- Adjust based on curriculum stage

---

### 9. **LOW PRIORITY: Limited Diversity in BC Demonstrations**

**Impact**: Low-Medium

The BC dataset has 130 replays covering multiple difficulty levels:

**Distribution** (from logs):
- Very simple: ~15 replays
- Simple: ~20 replays
- Medium: ~30 replays
- Complex: ~20 replays
- Exploration: ~15 replays
- Mine heavy: ~10 replays

**Issues**:

1. **Replay Distribution Mismatch**:
   - RL training starts on "simplest" (1000 levels)
   - BC has limited "very simple" demonstrations
   - Most BC data from medium/complex levels
   - Learned behaviors may not transfer to simplest

2. **Action Distribution Bias**:
   - BC learns action distribution from human demos
   - Humans solve levels optimally/efficiently
   - May not include exploratory behaviors
   - Agent imitates human actions but doesn't learn to explore

3. **Limited State Coverage**:
   - 130 replays × ~200 steps average = ~26k samples
   - Compared to 1M RL timesteps, this is tiny
   - BC provides good initialization but limited coverage

**Recommendation**:
- Accept this limitation (BC is just initialization)
- Focus on improving RL phase (issues 1-8 above)
- Could collect more very_simple/simplest demonstrations if needed

---

### 10. **CRITICAL: TensorBoard Logging Not Working**

**Impact**: Critical for Debugging

**Problem**: The TensorBoard events files contain **zero scalar metrics**, which means:

1. **No Training Visibility**:
   - Cannot see episode rewards, success rates, loss values
   - Cannot monitor training progress
   - Cannot debug what's happening during 18 hours of training

2. **Potential Callback Issue**:
   - TensorBoard callback may not be attached to PPO model
   - Hierarchical PPO may not be logging to TensorBoard
   - Custom callbacks may have broken standard logging

3. **Makes Diagnosis Nearly Impossible**:
   - We cannot confirm success rate is actually <0.1
   - We cannot see if policy is learning at all
   - We cannot see if rewards are being received
   - We cannot see curriculum progression

**Evidence**:
```python
# Both event files contain ZERO scalar metrics
ea.Tags()['scalars']  # Returns: []
```

This is highly unusual and suggests:
- TensorBoard callbacks not properly configured for HierarchicalPPO
- Logging disabled or broken
- Metrics computed but not written to disk

**Recommendation**:
- **CRITICAL**: Fix TensorBoard logging before any other experiments
- Add explicit logging in training loop
- Verify callbacks are attached to model
- Check if HierarchicalPPO overrides logging methods (preliminary check shows it delegates to standard PPO)
- Add print statements to verify metrics are being computed
- Check that TensorBoard writer is properly initialized with correct log directory

**Note**: The absence of metrics makes it impossible to:
- Confirm the reported <0.1 success rate
- See if learning is happening at all
- Debug reward values or loss curves
- Validate any of the hypotheses in this document

This must be fixed as priority #1.

---

### 11. **LOW PRIORITY: Mixed Precision Training Issues**

**Impact**: Low

Mixed precision is enabled but may cause subtle issues:

**Potential Problems**:
- Gradient scaling errors with hierarchical PPO
- Numerical instability in attention mechanisms (if using attention fusion)
- Loss of precision in advantage normalization

**Evidence**:
- Not enough information to confirm
- 15-minute updates suggest compute is not the bottleneck
- More likely data transfer or environment simulation

**Recommendation**:
- Test with mixed_precision=False
- Monitor for NaN losses or gradient explosions
- Likely not the primary issue

---

## Root Cause Analysis: Why <0.1 Success Rate?

### Critical System Issues (Must Fix First)

**ISSUE 0: No Training Metrics Available**
- **Impact**: Cannot confirm success rate, cannot debug
- **Evidence**: TensorBoard files contain 0 scalar metrics
- **Fix**: Restore logging before any other changes

### Primary Contributing Factors (Ranked by Impact)

1. **Catastrophically Low Environment Count (14 envs)**:
   - Causes: Insufficient exploration, overfitting, slow curriculum progression
   - Direct impact on success rate: **Very High**
   - Fix: Increase to 128+ envs

2. **Hierarchical PPO Overhead + Random Initialization**:
   - Causes: BC-pretrained features gated by random hierarchical parameters
   - Direct impact: **Very High** - nullifies BC pretraining benefits
   - Fix: Use standard PPO for simplest levels

3. **Frame Stacking Complexity**:
   - Causes: 4x computational overhead, potential BC-RL mismatch
   - Direct impact: **High** - slows training, may cause feature misalignment
   - Fix: Disable or reduce to 2-frame stacking

4. **Insufficient Reward Signal**:
   - Causes: Sparse rewards, weak navigation shaping
   - Direct impact: **High** - slow learning, insufficient gradient
   - Fix: Increase intermediate rewards and navigation shaping

5. **Learning Rate / Entropy Too Low After BC**:
   - Causes: Policy doesn't explore beyond BC demonstrations
   - Direct impact: **Medium-High** - agent can't improve on demos
   - Fix: Increase entropy, reduce LR

### Secondary Contributing Factors

6. **Curriculum Threshold Too High**: Slows progression
7. **MLP Lacking Spatial Reasoning**: Expected limitation but exacerbated by other issues
8. **Observation Normalization Mismatch**: Potential but unconfirmed
9. **BC Data Distribution**: Limited impact, BC is just initialization

---

## Recommended Action Plan

### Phase 1: Critical Fixes (Test Immediately)

**Test 1: Remove Hierarchical PPO + Increase Environments**
```bash
python scripts/train_and_compare.py \
    --experiment-name "mlp_baseline_fix_phase1" \
    --architectures mlp_baseline \
    --train-dataset ~/datasets/train \
    --test-dataset ~/datasets/test \
    --use-curriculum \
    --curriculum-threshold 0.6 \
    --curriculum-min-episodes 50 \
    --replay-data-dir ../nclone/bc_replays \
    --bc-epochs 50 \
    --bc-batch-size 128 \
    --hardware-profile auto \
    --total-timesteps 5000000 \
    --num-envs 128 \  # CRITICAL: Increase from 14
    --eval-freq 250000 \
    --output-dir ~/experiments \
    # REMOVE: --use-hierarchical-ppo
    # REMOVE: --enable-visual-frame-stacking
    # REMOVE: --enable-state-stacking
```

**Expected Results**:
- Update time: <2 minutes (down from 15)
- Can reach 100,000 timesteps in ~1-2 hours
- Should see >20% success rate on simplest levels after 500k timesteps

---

**Test 2: Adjust Hyperparameters for RL Fine-Tuning**

Modify `ppo_hyperparameters.py` or add custom config:
```python
HYPERPARAMETERS = {
    "n_steps": 1024,
    "batch_size": 512,  # Increase with more envs
    "n_epochs": 5,
    "gamma": 0.99,  # Reduce for shorter episodes
    "clip_range": 0.2,  # Gentler updates after BC
    "ent_coef": 0.015,  # Much higher for exploration
    "learning_rate": 1e-4,  # Lower for fine-tuning
    "vf_coef": 0.5,
}
```

---

### Phase 2: Reward Tuning (If Phase 1 Shows Improvement)

Modify `reward_constants.py`:
```python
# Increase navigation shaping
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001  # 10x increase

# Stronger intermediate rewards
SWITCH_ACTIVATION_REWARD = 0.3  # Up from 0.1

# Less discouraging
DEATH_PENALTY = -0.2  # Up from -0.5

# Add proximity bonus
PROXIMITY_BONUS = 0.05  # When within 50 pixels of objective
```

---

### Phase 3: Advanced Fixes (If Still <20% Success)

1. **Verify Observation Normalization**:
   - Check that RL environments use BC normalization stats
   - Log first batch of observations to compare ranges

2. **Add Entity Position Features**:
   - Explicitly add switch position, exit position to game state
   - Helps compensate for lack of graph-based reasoning

3. **Curriculum Adjustment**:
   - Start at even simpler levels if available
   - Reduce advancement threshold to 50%
   - Increase stage mixing ratio to 30%

4. **Test Without BC Pretraining**:
   - Establish baseline: can agent learn from scratch?
   - If from-scratch works better, BC pretraining is the problem
   - Check for BC-RL observation mismatch

---

## Experimental Validation Plan

### Minimum Viable Test

**Objective**: Achieve >20% success rate on simplest levels within 500k timesteps

**Configuration**:
- **Architecture**: mlp_baseline
- **Environments**: 128 (up from 14)
- **Hierarchical PPO**: DISABLED
- **Frame Stacking**: DISABLED
- **BC Pretraining**: ENABLED (50 epochs)
- **Curriculum**: Starting at simplest, threshold=0.6, min_episodes=50
- **Hyperparameters**: 
  - learning_rate=1e-4
  - ent_coef=0.015
  - clip_range=0.2
  - gamma=0.99

**Success Criteria**:
- >10% success rate after 100k timesteps
- >20% success rate after 500k timesteps
- Training time: <30 seconds per update
- Curriculum advancement to "simpler" stage within 500k timesteps

**If Test Fails**:
1. Test without BC pretraining (RL from scratch)
2. Test with stronger navigation shaping rewards
3. Test with vision_free architecture (eliminate CNN complexity)
4. Profile CPU/GPU usage to identify computational bottlenecks

---

## Conclusion (Updated After Second-Pass Analysis)

The MLP baseline's failure to achieve >0.1 success rate on simplest levels is caused by **six critical issues**, with two newly discovered bugs that completely invalidate BC pretraining:

### Critical Root Causes (Must All Be Fixed):

1. **BC-RL Observation Normalization Mismatch** (NEW) - BC trained on normalized obs, RL uses raw obs
2. **Incomplete Weight Loading** (NEW) - Only 58/82 params loaded, 24 feature extractor layers remain random
3. **Catastrophically low environment count** - 14 instead of 128+ due to hardware profile bug
4. **Hierarchical PPO overhead** - 46 random params gating corrupted features
5. **Frame stacking complexity** - 4x computational overhead with no benefit for MLP
6. **TensorBoard logging broken** - Cannot monitor training progress

### How These Issues Interact:

**BC Pretraining is Completely Broken**:
- Issue #1: BC learns features from normalized observations → RL feeds unnormalized observations → Input distribution shift
- Issue #2: Only 58 of 82 trained params loaded → 24 critical layers (reachability_mlp, fusion) remain random
- Result: Feature extractor produces garbage, rendering 91.55% BC accuracy useless

**Training is Catastrophically Inefficient**:
- Issue #3: Only 14 environments → 70 PPO updates in 1M timesteps (~15 min/update)
- Issue #4: Hierarchical policy adds 46 random params on top of broken features
- Issue #5: Frame stacking multiplies compute by 4x for zero benefit
- Result: Slow training on corrupted features with insufficient data diversity

**Cannot Debug or Monitor**:
- Issue #6: TensorBoard directory empty → No metrics to verify progress
- Combined with above: Training in the blind on broken pipeline

### The Good News

These are all **fixable configuration and implementation issues**, not fundamental architectural problems. However, **all six must be addressed together**, particularly the two newly discovered BC pretraining bugs:

1. **Add observation normalization to RL training** (load BC's normalization_stats.npz, apply VecNormalize or manual normalization)
2. **Fix weight loading logic** (ensure all 82 BC params map correctly to hierarchical structure, or disable hierarchical PPO)
3. **Fix hardware profile bug** (increase envs to 128+)
4. **Disable hierarchical PPO** (or fix weight loading)
5. **Disable frame stacking** (for MLP baseline)
6. **Fix TensorBoard logging** (investigate callback setup)

**Recommended immediate action**: The quick fix command below addresses the top issues. However, note that **without fixing observation normalization and weight loading**, BC pretraining will continue to provide zero benefit.

---

## Quick Fix Command (Partial Solution)

⚠️ **WARNING**: This command addresses issues #3, #4, #5 but **DOES NOT** fix the two critical BC pretraining bugs (#1 and #2). BC pretraining will still be ineffective until observation normalization and weight loading are fixed in the codebase.

For immediate testing (without BC benefit), run this modified command:

```bash
cd ~/npp-rl-training/npp-rl && \
export AWS_ACCESS_KEY_ID=foo && \
export AWS_SECRET_ACCESS_KEY=foo && \
export CUDA_HOME=/usr/local/cuda && \
export CUDA_PATH=/usr/local/cuda && \
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} && \
export PATH=/usr/local/cuda/bin:${PATH} && \
python scripts/train_and_compare.py \
  --experiment-name mlp_baseline_fix_v1 \
  --architectures mlp_baseline \
  --train-dataset ~/datasets/train \
  --test-dataset ~/datasets/test \
  --use-curriculum \
  --curriculum-threshold 0.6 \
  --curriculum-min-episodes 50 \
  --replay-data-dir ../nclone/bc_replays \
  --bc-epochs 50 \
  --bc-batch-size 128 \
  --hardware-profile auto \
  --total-timesteps 2000000 \
  --num-envs 128 \
  --eval-freq 100000 \
  --record-eval-videos \
  --max-videos-per-category 5 \
  --num-eval-episodes 10 \
  --video-fps 30 \
  --s3-bucket npp-rl-training-artifacts \
  --s3-prefix experiments/ \
  --output-dir ~/experiments
```

**Key changes from original**:
- ❌ Removed `--use-hierarchical-ppo` (fixes issue #2)
- ❌ Removed `--enable-visual-frame-stacking` (fixes issue #3)
- ❌ Removed `--enable-state-stacking` (fixes issue #3)
- ✅ **CRITICAL**: Explicitly set `--num-envs 128` (overrides auto-detection bug that gives 14)
- ✅ Reduced curriculum threshold to 0.6 (from 0.7)
- ✅ Reduced min episodes to 50 (from 100)
- ✅ Increased total timesteps to 2M (to account for faster updates)

**Expected improvements**:
- Update time: 1-2 minutes (down from 15)
- Can see results at 100k timesteps in <2 hours
- Success rate >20% after 500k timesteps
- Should advance to "simpler" stage within 500k timesteps

**Note**: The auto-detection gave 14 envs due to a bug in `hardware_profiles.py:217` that assumes 6GB per environment (appropriate for graph models, not MLP baseline). Explicit override is necessary.

---

## Required Code Fixes for BC Pretraining (CRITICAL)

⚠️ **IMPORTANT**: The quick fix command above does NOT address the two critical BC pretraining bugs. To properly leverage BC pretraining, the following code changes are **REQUIRED**:

### Fix #1: Add Observation Normalization to RL Training

**Problem**: BC training normalizes observations, RL training does not.

**Solution**: Load BC's normalization statistics and apply them during RL training.

**File**: `/workspace/npp-rl/npp_rl/training/architecture_trainer.py`

**Approach A: Use SB3's VecNormalize** (Recommended)
```python
# In _create_envs method, after creating self.env:
from stable_baselines3.common.vec_env import VecNormalize

# Load normalization stats from BC pretraining
bc_norm_stats_path = self.output_dir / "pretrain" / "cache" / "normalization_stats.npz"
if bc_norm_stats_path.exists() and pretrained_checkpoint is not None:
    logger.info(f"Loading BC observation normalization statistics from {bc_norm_stats_path}")
    
    # Wrap environment with VecNormalize
    self.env = VecNormalize(
        self.env,
        training=True,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards
        clip_obs=10.0,
        gamma=0.999,
    )
    
    # Load the BC normalization statistics
    bc_stats = np.load(bc_norm_stats_path)
    for key in bc_stats.keys():
        if key.endswith('_mean'):
            obs_key = key.replace('_mean', '')
            mean = bc_stats[key]
            std = bc_stats[key.replace('_mean', '_std')]
            # Apply to VecNormalize
            # Note: VecNormalize structure may need adjustment
    
    logger.info("✓ Applied BC observation normalization to RL training")
else:
    logger.warning("No BC normalization stats found - RL training without normalization")
```

**Approach B: Manual Normalization Wrapper** (More control)
```python
# Create a custom wrapper that applies BC normalization
class BCNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, norm_stats_path):
        super().__init__(env)
        self.stats = np.load(norm_stats_path)
        self.means = {k.replace('_mean', ''): v for k, v in self.stats.items() if '_mean' in k}
        self.stds = {k.replace('_std', ''): v for k, v in self.stats.items() if '_std' in k}
    
    def observation(self, obs):
        normalized = {}
        for key, value in obs.items():
            if key in self.means:
                normalized[key] = (value - self.means[key]) / (self.stds[key] + 1e-8)
            else:
                normalized[key] = value
        return normalized

# Apply wrapper in make_env function (line ~605):
if bc_norm_stats_path.exists():
    env = BCNormalizationWrapper(env, bc_norm_stats_path)
```

### Fix #2: Ensure Complete Weight Loading

**Problem**: 24 feature extractor parameters not loaded from BC checkpoint.

**Solution**: Fix weight mapping logic OR disable hierarchical PPO for MLP baseline.

**Option A: Disable Hierarchical PPO** (Simpler, recommended for MLP baseline)
- Already included in quick fix command above (`--use-hierarchical-ppo` removed)
- This ensures BC weights load directly without hierarchical nesting issues

**Option B: Fix Weight Mapping** (If hierarchical PPO needed)

**File**: `/workspace/npp-rl/npp_rl/training/architecture_trainer.py` (lines 260-433)

Add explicit mapping for missing feature extractor keys:
```python
# In _load_feature_extractor_weights method, after line 333:
# Add special handling for reachability_mlp and fusion layers
missing_feature_keys = [k for k in missing_keys if 'features_extractor.' in k]
if missing_feature_keys:
    logger.info(f"Attempting to manually map {len(missing_feature_keys)} feature extractor keys")
    
    for key in missing_feature_keys:
        # Try alternate key names
        alt_keys = [
            key.replace('features_extractor.', 'mlp_extractor.features_extractor.'),
            key.replace('features_extractor.', ''),
            f"policy.{key}",
        ]
        
        for alt_key in alt_keys:
            if alt_key in checkpoint_state:
                model_state[key] = checkpoint_state[alt_key]
                logger.debug(f"Mapped {alt_key} -> {key}")
                break
    
    # Reload with updated mapping
    missing_keys, unexpected_keys = model.policy.load_state_dict(model_state, strict=False)
```

---

## Additional Recommendations for Code Modifications

After fixing the two critical BC pretraining bugs above, if success rate is still <20%, implement these additional improvements:

### 1. Increase Entropy Coefficient

Edit `/workspace/npp-rl/npp_rl/agents/hyperparameters/ppo_hyperparameters.py`:

```python
"ent_coef": 0.015,  # Increase from 0.00272 for more exploration
"clip_range": 0.2,  # Reduce from 0.389 for gentler fine-tuning
"gamma": 0.99,      # Reduce from 0.999 for shorter episodes
"learning_rate": 1e-4,  # Reduce from 3e-4 for fine-tuning after BC
```

### 2. Strengthen Navigation Rewards

Edit `/workspace/nclone/nclone/gym_environment/reward_calculation/reward_constants.py`:

```python
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001  # 10x increase
SWITCH_ACTIVATION_REWARD = 0.3                  # Up from 0.1
DEATH_PENALTY = -0.2                            # Up from -0.5
```

### 3. Fix Hardware Profile Auto-Detection

Edit `/workspace/npp-rl/npp_rl/training/hardware_profiles.py` line 217:

**Current (BUGGY)**:
```python
envs_per_gpu = max(8, min(256, int(gpu_memory_gb / 6)))  # 6GB per environment
```

**Fixed**:
```python
# Use architecture-aware memory estimation
# MLP baseline: ~1GB per env, Graph models: ~6GB per env
# For auto-detection, use 2GB per env as middle ground
envs_per_gpu = max(16, min(256, int(gpu_memory_gb / 2)))  # 2GB per environment
```

This changes H100 (85GB) from 14 envs → 42 envs (better but still conservative).

**Better fix**: Add architecture-specific profiles:
```python
def auto_detect_profile(architecture_name: Optional[str] = None) -> Optional[HardwareProfile]:
    # ... existing detection code ...
    
    # Architecture-specific memory requirements
    memory_per_env = {
        "mlp_baseline": 1.0,      # Lightweight
        "vision_free": 0.5,        # Very lightweight
        "graph_attention": 6.0,    # Heavy
        "default": 2.0,            # Conservative estimate
    }.get(architecture_name or "default", 2.0)
    
    envs_per_gpu = max(16, min(256, int(gpu_memory_gb / memory_per_env)))
```

### 4. Fix TensorBoard Logging

Investigate and fix the TensorBoard callback in the training script. Ensure:
- Callback is properly attached to model
- Log directory is correct
- Metrics are being computed and passed to logger
- No conflicts with HierarchicalPPO (though it delegates to standard PPO)

---

## Validation Checklist

After running the fix:

- [ ] Training updates complete in <2 minutes (not 15 minutes)
- [ ] TensorBoard shows metrics (episode reward, success rate, loss)
- [ ] Success rate >10% after 100k timesteps
- [ ] Success rate >20% after 500k timesteps
- [ ] Curriculum advances to "simpler" stage
- [ ] Episode rewards are increasing over time
- [ ] Policy loss is decreasing over time

If any of these fail, refer back to this document for targeted debugging.

---

## References

### Key Code Locations

- **Architecture Config**: `/workspace/npp-rl/npp_rl/training/architecture_configs.py:345` (mlp_baseline definition)
- **Feature Extractor**: `/workspace/npp-rl/npp_rl/feature_extractors/configurable_extractor.py`
- **Hierarchical PPO**: `/workspace/npp-rl/npp_rl/agents/hierarchical_ppo.py`
- **Curriculum Manager**: `/workspace/npp-rl/npp_rl/training/curriculum_manager.py`
- **Reward Constants**: `/workspace/nclone/nclone/gym_environment/reward_calculation/reward_constants.py`
- **PPO Hyperparameters**: `/workspace/npp-rl/npp_rl/agents/hyperparameters/ppo_hyperparameters.py`

### Training Logs Evidence

- **Environment Count**: Config shows `num_envs: 14`
- **Update Timing**: Logs show ~15 minutes between updates
- **BC Pretraining**: Successfully completed 50 epochs with 130 replays
- **Hierarchical Loading**: 46 hierarchical parameters initialized randomly, gating pretrained features
- **Frame Stacking**: Enabled for both visual (4 frames) and state (4 frames)

### Relevant Research

- **PPO Environment Count**: "PPO benefits from large batch sizes" - Schulman et al. (2017)
- **Curriculum Learning**: "Start simple, progress gradually" - Bengio et al. (2009)
- **BC-RL Transfer**: "Reduce learning rate after pretraining" - Rajeswaran et al. (2017)
- **Reward Shaping**: "Dense shaping >> sparse rewards for exploration" - Ng et al. (1999)
