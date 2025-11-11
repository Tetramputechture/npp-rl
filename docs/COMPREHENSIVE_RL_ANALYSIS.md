# Comprehensive RL Analysis and Recommendations for NPP-RL

**Date**: November 8, 2025  
**Training Run**: arch_comparison_20251107 (2M steps, 21.43 hours)  
**Architecture**: MLP Baseline  
**Status**: Agent performance plateaued, curriculum advancement blocked

---

## Executive Summary

After comprehensive analysis of TensorBoard metrics, training configuration, reward structure, and codebase architecture, we've identified **critical bottlenecks** preventing the agent from learning effectively:

1. **Curriculum Learning Failure**: Agent stuck on stage 1 (simplest_with_mines) with 44% success rate
2. **Negative Reward Regime**: All mean action rewards are negative, indicating poor reward structure
3. **PBRS Underutilization**: Potential-based reward shaping provides insufficient guidance
4. **Missing Temporal Information**: No frame/state stacking despite physics-based environment
5. **Suboptimal PPO Configuration**: High variance advantage estimates, no LR annealing

**Critical Finding**: The agent never progressed beyond the first two curriculum stages in 2M steps, indicating fundamental learning failures that must be addressed before expecting generalization.

---

## 1. Curriculum Learning Analysis

### Current State
```
Stage 0 (simplest):              74-92% success ✓ (can advance)
Stage 1 (simplest_with_mines):   44% success   ✗ (stuck here)
Stage 2+ (all others):            0% success    ✗ (never trained)
```

### Issues Identified

#### 1.1 Premature Difficulty Increase
- **Problem**: Stage 1 introduces mines too early, before basic navigation is mastered
- **Evidence**: Success rate drops from 77% (simplest) to 44% (simplest_with_mines)
- **Impact**: Agent spends entire training budget (2M steps) on stage 1, never seeing later stages

#### 1.2 Rigid Advancement Thresholds
- **Problem**: 80% threshold for stage 1 is too high given reward structure
- **Evidence**: Agent plateaus at 44% and never improves further
- **Impact**: No curriculum progression = no generalization to diverse scenarios

#### 1.3 Lack of Intermediate Stages
- **Problem**: Jump from "no mines" to "with mines" is too large
- **Recommendation**: Add intermediate stages with gradual mine introduction

### Recommendations

**HIGH PRIORITY - Curriculum Restructuring**:

```python
NEW_CURRICULUM_ORDER = [
    "simplest",              # Basic navigation (80% threshold)
    "simplest_few_mines",    # 1-2 mines (70% threshold) - NEW
    "simplest_with_mines",   # 3-5 mines (65% threshold) - ADJUSTED
    "simpler",               # More complex layouts (60% threshold)
    "simple",                # Multi-room (55% threshold)
    "medium",                # Intermediate (50% threshold)
    "complex",               # Advanced (45% threshold)
    "exploration",           # Requires exploration (40% threshold)
    "mine_heavy",            # Hardest (35% threshold)
]
```

**Key Changes**:
1. Add "simplest_few_mines" intermediate stage
2. Lower threshold for "simplest_with_mines" from 80% to 65%
3. Implement progressive threshold reduction (80% → 35%)
4. Add automatic threshold adjustment based on training progress

---

## 2. Reward Structure Analysis

### Current State

#### 2.1 Per-Action Rewards (All Negative)
```
Action 0 (NOOP):        -0.0305 to -0.0183
Action 1 (Left):        -0.0138 to -0.0171
Action 2 (Right):       -0.0089 to -0.0156
Action 3 (Jump):        -0.0035 to -0.0161
Action 4 (Jump+Left):   -0.0065 to -0.0121
Action 5 (Jump+Right):  -0.0118 to -0.0134
```

#### 2.2 PBRS Rewards (Near Zero)
```
Initial: -0.0067
Final:   ~0.0000
Mean:    ~0.0000
```

### Issues Identified

#### 2.1 Time Penalty Dominates
- **Problem**: -0.0001 per step × 5000 steps = -0.5 total penalty
- **Impact**: Even with exploration bonuses, net reward is negative
- **Evidence**: All action rewards negative, agent receives punishment for existing

#### 2.2 PBRS Too Conservative
- **Problem**: PBRS weights too low (1.5 for objective, 0.04 for hazards)
- **Impact**: PBRS provides ~0 guidance, agent doesn't learn to approach objectives
- **Evidence**: PBRS rewards trend to zero, no clear gradient toward goals

#### 2.3 Terminal Rewards Too Sparse
- **Problem**: Completion reward (20.0) only received at episode end
- **Impact**: Agent must discover completion through random exploration
- **Evidence**: 44% success rate after 2M steps indicates poor discovery

#### 2.4 Momentum Bonus Ineffective
- **Problem**: 0.0002 per step only when speed > 80% × MAX_SPEED
- **Impact**: Bonus too small to influence behavior
- **Evidence**: No clear increase in movement efficiency over training

### Recommendations

**HIGH PRIORITY - Reward Rebalancing**:

1. **Reduce Time Penalty** (10x reduction):
   ```python
   TIME_PENALTY_PER_STEP = -0.00001  # Was -0.0001
   ```

2. **Increase PBRS Weights** (3-5x increase):
   ```python
   PBRS_OBJECTIVE_WEIGHT = 4.5  # Was 1.5
   PBRS_HAZARD_WEIGHT = 0.15    # Was 0.04
   PBRS_IMPACT_WEIGHT = 0.15    # Was 0.04
   PBRS_EXPLORATION_WEIGHT = 0.6 # Was 0.2
   ```

3. **Add Distance-Based Milestone Rewards**:
   ```python
   # Reward for reaching checkpoints toward switch/exit
   DISTANCE_MILESTONE_REWARDS = {
       "75%": 0.5,   # Reached 75% of distance to objective
       "50%": 1.0,   # Reached 50% of distance
       "25%": 1.5,   # Reached 25% of distance
   }
   ```

4. **Increase Momentum Bonus** (5x increase):
   ```python
   MOMENTUM_BONUS_PER_STEP = 0.001  # Was 0.0002
   ```

5. **Implement Reward Normalization**:
   - Track running statistics of rewards
   - Normalize to [-1, 1] range
   - Ensures stable learning regardless of reward magnitudes

---

## 3. Observation Space and Feature Engineering

### Current Configuration

**Modalities in MLP Baseline**:
- `game_state`: 29 features (velocity, movement state, inputs, buffers, physics)
- `reachability_features`: 8 features (area ratio, distances, connectivity)
- `entity_positions`: 6 features
- `switch_states`: 25 features
- **Total**: 68 scalar features → MLP [256, 256, 128]

**Unused Modalities** (in MLP baseline):
- `player_frame`: 84×84×1 grayscale centered view
- `global_view`: 176×100×1 grayscale full level
- `graph`: Node/edge features for pathfinding

### Issues Identified

#### 3.1 No Temporal Information
- **Problem**: Single-frame observations in physics-based game
- **Impact**: Agent cannot perceive velocity, acceleration, momentum trends
- **Evidence**: Config shows `enable_state_stacking: false`

#### 3.2 MLP May Be Too Simple
- **Problem**: 68 features → 3-layer MLP may lack capacity
- **Impact**: Cannot learn complex spatial relationships
- **Evidence**: Stuck at 44% success on simple levels

#### 3.3 Visual Information Unused
- **Problem**: MLP baseline ignores visual observations
- **Impact**: Missing spatial context that CNNs excel at capturing
- **Evidence**: player_frame and global_view not used

### Recommendations

**HIGH PRIORITY - Enable State Stacking**:

```python
"enable_state_stacking": true,
"state_stack_size": 4,
"frame_stack_padding": "repeat",  # Repeat first frame for initial padding
```

**Benefits**:
- Agent sees last 4 timesteps → understands velocity as position change
- Can infer acceleration from velocity changes
- Temporal patterns in physics become observable

**MEDIUM PRIORITY - Upgrade to CNN+MLP Architecture**:

```python
"architectures": ["cnn_mlp_fusion"],
"modality_config": {
    "use_player_frame": true,   # Enable visual
    "use_global_view": false,   # Keep computational cost down
    "use_graph": false,         # Keep it simple for now
    "use_game_state": true,
    "use_reachability": true
}
```

**Benefits**:
- CNN extracts spatial features from local view
- MLP processes game state and reachability
- Fusion layer combines modalities
- More powerful representation learning

---

## 4. PPO Hyperparameter Analysis

### Current Configuration

```python
n_steps = 1024         # Steps per environment per update
batch_size = 256       # Minibatch size
n_epochs = 5           # Optimization epochs per update
gamma = 0.995          # Discount factor (from PBRS)
gae_lambda = 0.97      # GAE advantage estimation
learning_rate = 3e-4   # Constant (no annealing)
ent_coef = 0.02        # Entropy coefficient
vf_coef = 0.5          # Value function coefficient
max_grad_norm = 2.0    # Gradient clipping
clip_range = 0.2       # PPO clipping range
```

### Issues Identified

#### 4.1 High Variance Advantages
- **Problem**: `gae_lambda = 0.97` with `gamma = 0.995` creates high variance
- **Impact**: Advantage estimates are noisy → unstable learning
- **Evidence**: High clip fraction (0.4) indicates policy wants large updates

#### 4.2 No Learning Rate Annealing
- **Problem**: LR stays constant at 3e-4 throughout training
- **Impact**: Cannot fine-tune in later stages, continues making large updates
- **Evidence**: Config shows `enable_lr_annealing: false`

#### 4.3 Potential Sample Inefficiency
- **Problem**: `n_epochs = 5` may be too few for complex task
- **Impact**: Not fully optimizing each batch of data
- **Evidence**: Agent improves slowly (2M steps to reach 44%)

#### 4.4 Low Entropy Coefficient
- **Problem**: `ent_coef = 0.02` may be too low for exploration
- **Impact**: Policy becomes deterministic too quickly
- **Evidence**: Entropy decreases but performance doesn't improve

### Recommendations

**HIGH PRIORITY - PPO Hyperparameter Tuning**:

1. **Reduce GAE Lambda** (for more stable advantages):
   ```python
   "gae_lambda": 0.92  # Was 0.97, reduces variance
   ```

2. **Enable Learning Rate Annealing**:
   ```python
   "enable_lr_annealing": true,
   "initial_lr": 3e-4,
   "final_lr": 3e-5,      # 10x reduction
   "lr_schedule": "linear"  # or "cosine"
   ```

3. **Increase Entropy Coefficient** (maintain exploration longer):
   ```python
   "ent_coef": 0.05  # Was 0.02, 2.5x increase
   ```

4. **Increase Optimization Epochs** (better sample efficiency):
   ```python
   "n_epochs": 10  # Was 5, double the optimization
   ```

5. **Adjust Value Function Clipping** (match reward scale):
   ```python
   "clip_range_vf": 1.0  # Was 0.1, allows larger value updates
   ```

---

## 5. Training Procedure Improvements

### Current Issues

1. **No Automatic Checkpoint Recovery**: If training crashes, starts from scratch
2. **Rigid Evaluation Schedule**: eval_freq = 100k may be too sparse
3. **Limited Diagnostic Logging**: Hard to debug reward components
4. **No Early Stopping**: Training continues even when plateaued

### Recommendations

**MEDIUM PRIORITY - Training Infrastructure**:

1. **Increase Evaluation Frequency**:
   ```python
   "eval_freq": 25000,  # Was 100000, 4x more frequent
   "num_eval_episodes": 10,  # Was 2, more robust
   ```

2. **Implement Early Stopping**:
   ```python
   "early_stopping": {
       "patience": 10,  # Stop if no improvement for 10 evals
       "min_delta": 0.01,  # Minimum improvement threshold
       "metric": "curriculum/success_rate"
   }
   ```

3. **Enhanced Reward Component Logging**:
   - Log individual PBRS components per episode
   - Track reward distribution statistics
   - Monitor action effectiveness metrics

4. **Automatic Curriculum Adjustment**:
   ```python
   "adaptive_curriculum": {
       "auto_lower_threshold": true,
       "threshold_reduction_rate": 0.95,  # 5% reduction per adjustment
       "adjustment_frequency": 50000,  # Check every 50k steps
       "min_threshold": 0.4  # Don't go below 40%
   }
   ```

---

## 6. Architecture Comparison Recommendations

### Current: MLP Baseline Only
- Simple but may lack capacity
- Good baseline but shouldn't be only architecture tested

### Recommended Architecture Progression

#### Phase 1: Fix Fundamentals (Current Priority)
```python
"architectures": ["mlp_cnn"]
```
- Focus on reward/curriculum fixes first
- Validate improvements with simple architecture
- Establish performance baseline

#### Phase 2: Add Visual Features
```python
"architectures": ["cnn_mlp_fusion"]
```
- CNN for player_frame + MLP for state
- Concat or attention-based fusion
- Should improve spatial understanding

#### Phase 3: Add Graph Neural Network
```python
"architectures": ["cnn_gnn_mlp_fusion"]
```
- GNN for graph-based pathfinding
- Multi-modal fusion with attention
- Full representation power

### Don't Use Yet (Too Complex):
- Full HGT (Heterogeneous Graph Transformer)
- Multi-head hierarchical attention
- These add complexity without fixing core issues

---

## 7. Implementation Priority Matrix

### Immediate (Blocking Issues) - Week 1

1. **Reduce Time Penalty 10x** → Makes rewards positive
2. **Increase PBRS Weights 3x** → Provides guidance
3. **Enable State Stacking (4 frames)** → Temporal information
4. **Lower Curriculum Thresholds** → Allows progression
5. **Add Intermediate Curriculum Stage** → Smoother difficulty curve

### High Priority - Week 2

6. **Enable LR Annealing** → Better convergence
7. **Reduce GAE Lambda** → More stable learning
8. **Increase Entropy Coefficient** → Better exploration
9. **Increase Eval Frequency** → Better monitoring
10. **Implement Reward Normalization** → Stable training

### Medium Priority - Week 3-4

11. **Upgrade to CNN+MLP Architecture** → Better representations
12. **Add Distance Milestone Rewards** → Denser feedback
13. **Increase Optimization Epochs** → Better sample efficiency
14. **Implement Early Stopping** → Save compute
15. **Enhanced Diagnostic Logging** → Better debugging

### Lower Priority - Week 5+

16. **Add Graph Neural Network** → Advanced pathfinding
17. **Implement ICM for Exploration** → Curiosity-driven
18. **Hyperparameter Search with Optuna** → Optimization
19. **Multi-GPU Distributed Training** → Faster iterations
20. **Hierarchical Policy (HRL)** → Advanced behaviors

---

## 8. Expected Performance Improvements

### After Immediate Fixes (Week 1)

**Baseline (Current)**:
- Stage 0: 77% → Stage 1: 44% (stuck)
- Mean reward: -0.017
- Training: 2M steps, 21 hours

**Expected After Fixes**:
- Stage 0: 85% → Stage 1: 70% → Stage 2: 50% (progressing)
- Mean reward: +0.05 to +0.15 (positive!)
- Training: 2M steps should reach stage 3-4

### After High Priority (Week 2)

**Expected**:
- Stage 0-2: >80% each
- Stage 3-4: 60-70%
- Mean reward: +0.2 to +0.3
- More stable training curves

### After Medium Priority (Week 3-4)

**Expected**:
- Complete curriculum progression through all 8 stages
- Final stages (6-7): 40-50% success
- Mean reward: +0.5+
- Agent completes most levels efficiently

---

## 9. Validation Experiments

### Experiment 1: Reward Structure Validation
**Goal**: Verify reward changes produce positive mean rewards

**Setup**:
- Reduce time penalty 10x
- Increase PBRS weights 3x
- Train 100k steps on stage 0

**Success Criteria**:
- Mean reward > 0.0
- Success rate > 85%
- PBRS rewards show clear gradient toward objectives

### Experiment 2: Curriculum Progression Validation
**Goal**: Verify agent can progress through curriculum

**Setup**:
- Lower stage 1 threshold to 65%
- Add intermediate stage
- Train 500k steps

**Success Criteria**:
- Advance to stage 2 within 300k steps
- Maintain >70% on stage 1
- Show improvement trend

### Experiment 3: State Stacking Impact
**Goal**: Quantify benefit of temporal information

**Setup**:
- A/B test: 4-frame stacking vs no stacking
- Same reward/curriculum config
- Train 200k steps on stage 1

**Success Criteria**:
- Stacking shows +10% success rate improvement
- Faster convergence (fewer steps to plateau)
- Better action selection (fewer wasted actions)

---

## 10. Monitoring and Evaluation

### Key Metrics to Track

#### Training Progress
- `curriculum/success_rate` (primary)
- `curriculum/current_stage_idx` (progression)
- `episode/success_rate_smoothed` (stability)

#### Reward Health
- `reward_dist/mean` (should be positive)
- `pbrs_rewards/pbrs_mean` (should provide gradient)
- `reward_dist/positive_ratio` (should be >0.5)

#### Policy Quality
- `train/clip_fraction` (should be 0.1-0.3)
- `train/approx_kl` (should be <0.05)
- `actions/entropy` (should decay slowly)

#### Efficiency
- `performance/fps_mean` (throughput)
- `rollout/success_rate` (sample efficiency)
- `value/estimate_mean` (value accuracy)

### Red Flags

1. **Mean reward stays negative** → Reward structure still broken
2. **No curriculum advancement in 500k steps** → Thresholds too high
3. **Clip fraction >0.4** → Policy diverging, reduce LR
4. **Success rate oscillates wildly** → Reduce GAE lambda
5. **Entropy drops to <1.0 quickly** → Increase ent_coef

---

## 11. Long-Term Roadmap

### Q1 2026: Foundation
- ✓ Fix reward structure
- ✓ Enable curriculum progression
- ✓ Add temporal information
- ✓ Optimize PPO hyperparameters
- Target: Complete curriculum, 50%+ final stage success

### Q2 2026: Architecture
- Upgrade to CNN+MLP
- Add GNN for pathfinding
- Implement attention fusion
- Target: 60-70% final stage success

### Q3 2026: Advanced Techniques
- Implement ICM curiosity
- Hierarchical RL for sub-goals
- Multi-task learning across level types
- Target: 70-80% final stage success

### Q4 2026: Generalization
- Transfer learning to new level sets
- Meta-learning for rapid adaptation
- Human-level performance metrics
- Target: Match or exceed human expert on test suite

---

## 12. References and Best Practices

### Reinforcement Learning
1. **Schulman et al. (2017)**: "Proximal Policy Optimization Algorithms"
   - PPO is stable but requires careful tuning
   - Clipping prevents destructive policy updates

2. **Mnih et al. (2016)**: "Asynchronous Methods for Deep RL"
   - Parallel environments improve sample efficiency
   - Entropy regularization maintains exploration

3. **Cobbe et al. (2020)**: "Leveraging Procedural Generation for RL"
   - Curriculum learning essential for generalization
   - Progressive difficulty prevents overfitting

### Reward Shaping
4. **Ng et al. (1999)**: "Policy Invariance Under Reward Transformations"
   - PBRS theory: F(s,s') = γ·Φ(s') - Φ(s)
   - Maintains optimal policy while adding dense rewards

5. **Burda et al. (2018)**: "Exploration by Random Network Distillation"
   - Intrinsic motivation for sparse reward environments
   - Curiosity-driven exploration

### Platformer-Specific
6. **OpenAI (2019)**: "Solving Montezuma's Revenge"
   - Hierarchical RL for complex navigation
   - Explicit memory for revisiting states

7. **Ecoffet et al. (2019)**: "Go-Explore: Hard Exploration Problems"
   - Remember promising states
   - Return and explore from checkpoints

### Curriculum Learning
8. **Bengio et al. (2009)**: "Curriculum Learning"
   - Start simple, gradually increase difficulty
   - Faster convergence, better generalization

9. **Narvekar et al. (2020)**: "Curriculum Learning for RL"
   - Automatic difficulty adjustment
   - Performance-based progression

---

## 13. Conclusion

The current training run demonstrates fundamental issues that prevent effective learning:

1. **Reward structure produces negative feedback** → Agent doesn't learn to complete levels
2. **Curriculum cannot progress** → Agent never sees diverse scenarios
3. **Missing temporal information** → Agent cannot understand physics
4. **Suboptimal hyperparameters** → Learning is unstable and slow

**The good news**: These are all fixable with targeted interventions. The infrastructure is solid (PBRS, curriculum, multimodal observations, PPO implementation). The issues are configuration and tuning, not fundamental algorithmic problems.

**Recommended approach**:
1. Implement Week 1 immediate fixes (reward + curriculum)
2. Validate with 100k-step experiment
3. If successful, continue with high-priority items
4. Monitor metrics closely and adjust based on data
5. Iterate quickly with shorter training runs (500k steps) until configuration stabilized

**Expected timeline to working agent**: 3-4 weeks with focused iteration and validation experiments.

**Final note**: The path to success is clear. The agent has the right observations and the infrastructure exists. We need to tune the learning signals (rewards) and progression (curriculum) to guide the agent effectively. Once these are fixed, the agent should progress through the curriculum and achieve strong performance on the test suite.

---

**Document Version**: 1.0  
**Last Updated**: November 8, 2025  
**Next Review**: After implementing Week 1 fixes
