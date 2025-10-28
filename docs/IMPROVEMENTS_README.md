# RL Training Improvements - October 28, 2025

## Overview

This document describes the improvements made to the RL training system based on comprehensive analysis of TensorBoard logs and training metrics. The changes address critical issues that were preventing effective learning.

## Summary of Issues Found

The original training run (1M timesteps) showed the agent stuck at curriculum stage 2 ("simple") with only 14% success rate. Analysis revealed five critical issues:

1. **PBRS Disabled** - No dense navigation rewards
2. **Low Entropy Coefficient** - Premature policy convergence  
3. **Overly Aggressive Curriculum** - 70% threshold unreachable
4. **Action Space Collapse** - Jump actions decreased by 50%
5. **Training Too Short** - Insufficient exploration time

See `TRAINING_ANALYSIS_2025-10-28.md` for full analysis.

## Changes Made

### 1. PPO Hyperparameters (`npp_rl/agents/hyperparameters/ppo_hyperparameters.py`)

**Critical Changes:**
- **Entropy Coefficient:** 0.00272 → 0.02 (7x increase)
  - Maintains exploration for longer duration
  - Prevents premature convergence to suboptimal policies
  - Addresses action space collapse (jump avoidance)

**Important Changes:**
- **Gamma:** 0.999 → 0.995
  - Better credit assignment in sparse reward setting
  - Reduces amplification of noise in rewards
  
- **GAE Lambda:** 0.9988 → 0.97
  - Lower variance advantage estimates
  - More stable learning signal

- **Clip Range:** 0.389 → 0.2
  - More conservative policy updates
  - Standard PPO value for better stability

- **VF Coefficient:** 0.469 → 0.5
  - Standard value for better balance

- **Max Grad Norm:** 2.566 → 2.0
  - Cleaner gradient clipping

**Rationale:** These changes address the core issue of premature convergence while improving training stability.

### 2. PBRS Gamma Sync (`nclone/gym_environment/reward_calculation/reward_constants.py`)

**Change:**
- **PBRS_GAMMA:** 0.999 → 0.995

**Rationale:** PBRS theory (Ng et al., 1999) requires gamma to match RL algorithm for policy invariance. Must stay synchronized with PPO gamma.

### 3. Curriculum Thresholds (`npp_rl/training/curriculum_manager.py`)

**Stage Threshold Reductions:**
```python
"simplest":    0.80 → 0.70  (-0.10)
"simpler":     0.70 → 0.60  (-0.10)
"simple":      0.60 → 0.50  (-0.10)  # CRITICAL - was stuck here
"medium":      0.55 → 0.45  (-0.10)
"complex":     0.50 → 0.40  (-0.10)
"exploration": 0.45 → 0.35  (-0.10)
"mine_heavy":  0.40 → 0.30  (-0.10)
```

**Minimum Episode Reductions:**
```python
"simplest":    200 → 50   (-75%)
"simpler":     200 → 50   (-75%)
"simple":      200 → 75   (-62%)
"medium":      250 → 100  (-60%)
"complex":     300 → 150  (-50%)
"exploration": 300 → 150  (-50%)
"mine_heavy":  300 → 200  (-33%)
```

**Rationale:** 
- Agent was stuck at "simple" stage with 14% success rate (far below 70% threshold)
- Original minimums (200-300 episodes) were never reached in 1M timesteps
- Lower thresholds allow progressive learning while maintaining competence
- With PBRS enabled and 10x longer training, these should be achievable

### 4. Improved Training Configuration (`configs/improved_training_config.json`)

**Major Changes:**
- **enable_pbrs:** false → true (CRITICAL)
- **total_timesteps:** 1,000,000 → 10,000,000 (10x increase)
- **enable_visual_frame_stacking:** false → true
- **visual_stack_size:** 4 (enables temporal awareness)
- **frame_stack_padding:** "replicate" (better than zero padding)
- **bc_epochs:** 50 → 30 (reduce BC overfitting risk)
- **curriculum_threshold:** 0.7 → 0.5 (top-level config)
- **curriculum_min_episodes:** 100 → 50
- **enable_lr_annealing:** true (progressive LR reduction)

**Rationale:** These config changes enable the core improvements and extend training duration.

## Expected Results

With these improvements, we expect:

### Phase 1 (0-2M timesteps)
- ✓ Agent progresses past Stage 2 ("simple")
- ✓ Reaches Stage 3 ("medium") with 45%+ success rate
- ✓ Jump actions stabilize or increase (not decrease)
- ✓ Value estimates become less pessimistic

### Phase 2 (2-5M timesteps)
- ✓ Agent reaches Stage 4 ("complex")
- ✓ Success rate on simple stages exceeds 60%
- ✓ Action entropy remains above 1.5 (exploration maintained)
- ✓ PBRS rewards provide clear navigation signal

### Phase 3 (5-10M timesteps)
- ✓ Agent reaches Stage 5+ (exploration, mine_heavy)
- ✓ Learns effective jumping mechanics
- ✓ Completes 30%+ of hardest stages
- ✓ Generalizes to unseen level variations

## How to Use

### Training with New Configuration

```bash
# Use the improved configuration
python scripts/train.py --config configs/improved_training_config.json

# Or override specific parameters
python scripts/train.py \
    --enable_pbrs true \
    --total_timesteps 10000000 \
    --curriculum_threshold 0.5
```

### Monitoring Progress

Key metrics to track:
1. **Curriculum Stage** - Should progress through all 7 stages
2. **Success Rate per Stage** - Should meet lowered thresholds
3. **Action Entropy** - Should stay above 1.5 for first 5M steps
4. **Jump Frequency** - Should stabilize around 35-40%
5. **Value Estimates** - Should become less negative over time
6. **PBRS Contribution** - Should provide 10-30% of total reward

### Comparing to Baseline

To validate improvements, compare these metrics to baseline:

| Metric | Baseline (1M steps) | Target (10M steps) |
|--------|---------------------|-------------------|
| Max Stage Reached | 2 (simple) | 6 (mine_heavy) |
| Stage 2 Success Rate | 14% | 50%+ |
| Jump+Right Frequency | 8.5% (collapsed) | 15%+ (stable) |
| Action Entropy | 1.729 (low) | 1.5-1.7 (maintained) |
| Value Estimate Mean | -2.76 (pessimistic) | -0.5 to +2.0 |
| Episodes Completed | 201 | 2000+ |

## Implementation Details

### PBRS Configuration

The improved config enables PBRS with these settings:

```python
"enable_pbrs": true,
"pbrs_gamma": 0.995,  # Matches PPO gamma
"pbrs_objective_weight": 1.0,  # Primary navigation signal
"pbrs_hazard_weight": 0.1,     # Safety awareness
"pbrs_exploration_weight": 0.2 # Spatial coverage
```

PBRS provides dense rewards for:
- **Navigation:** Progress toward switch/exit (-1.0 to +1.0 per step)
- **Exploration:** Visiting new areas (+0.01 to +0.04 per step)
- **Hazard Avoidance:** Staying away from mines (-0.1 penalty proximity)

### Frame Stacking

Enabling visual frame stacking allows the agent to infer:
- **Velocity:** Change in position across frames
- **Momentum:** Trajectory predictions
- **Action Effects:** Immediate feedback on action outcomes

With 4-frame stacking:
- Input changes from [B, 84, 84, 1] to [B, 4, 84, 84, 1]
- CNN processes with replicate padding (better than zero)
- Adds ~1-2 MB memory overhead per environment

### Learning Rate Annealing

With LR annealing enabled:
- Starts at 0.0003 (standard)
- Linearly decreases to 0.00003 over 10M steps
- Allows aggressive exploration early, fine-tuning later
- Compatible with high entropy coefficient

## Validation Plan

### Quick Validation (1-2M steps)

Run training with improved config for 2M steps:

```bash
python scripts/train.py \
    --config configs/improved_training_config.json \
    --total_timesteps 2000000
```

**Success Criteria:**
- Progresses past Stage 2
- Stage 2 success rate > 25% (improvement over 14%)
- Jump+Right frequency > 10% (recovery from 8.5%)
- No crashes or instabilities

### Full Validation (10M steps)

Run complete training:

```bash
python scripts/train.py \
    --config configs/improved_training_config.json
```

**Success Criteria:**
- Reaches at least Stage 5 (exploration)
- Stage 4 success rate > 40%
- Action entropy > 1.5 at 5M steps
- Value estimates > -1.0 at 10M steps

### A/B Testing

To scientifically validate improvements, run both configurations:

```bash
# Baseline (for comparison)
python scripts/train.py \
    --config training-results/config.json \
    --experiment_name baseline_rerun

# Improved
python scripts/train.py \
    --config configs/improved_training_config.json \
    --experiment_name improved_v1
```

Compare final metrics after 10M steps.

## Troubleshooting

### Issue: Agent Still Not Progressing

**Possible Causes:**
1. PBRS not actually enabled (check logs)
2. Reward scale mismatch
3. Feature extractor issues
4. Environment bugs

**Debug Steps:**
```python
# Check PBRS is working
grep "pbrs_rewards" logs/*.log

# Verify entropy is high
tensorboard --logdir experiments/

# Check action distribution
python scripts/analyze_actions.py --logdir experiments/improved_v1
```

### Issue: Training Unstable

**Possible Causes:**
1. Learning rate too high
2. Batch size too small
3. Gradient issues

**Solutions:**
- Reduce LR to 0.0001
- Increase batch_size to 512
- Check gradient norms in TensorBoard

### Issue: Memory/Speed Problems

**Solutions:**
- Reduce num_envs from 21 to 16
- Disable frame_stacking temporarily
- Use smaller network architecture
- Enable mixed_precision (should be on by default)

## Future Improvements

### Priority 1 (Next Sprint)
- [ ] Implement intrinsic motivation (ICM or RND)
- [ ] Add action regularization bonus
- [ ] Enable graph neural network (spatial reasoning)
- [ ] Implement adaptive entropy scheduling

### Priority 2 (Future Work)
- [ ] Hierarchical RL with high-level planner
- [ ] Hindsight Experience Replay (HER)
- [ ] Population-based training
- [ ] Multi-task learning across levels

### Priority 3 (Research)
- [ ] Transformer-based policy
- [ ] World model learning
- [ ] Meta-learning for fast adaptation
- [ ] Curriculum learning with difficulty estimation

## References

1. **Training Analysis:** `docs/TRAINING_ANALYSIS_2025-10-28.md`
2. **PPO Implementation Details:** Huang et al. (2022)
3. **PBRS Theory:** Ng et al. (1999) - "Policy Invariance Under Reward Transformations"
4. **Curriculum Learning:** Bengio et al. (2009)
5. **OpenAI Spinning Up:** Policy Gradient Methods

## Contact

For questions or issues:
- Review the comprehensive analysis document
- Check TensorBoard logs for diagnostic info
- Consult the codebase documentation
- Refer to academic papers for theoretical background

---

**Last Updated:** October 28, 2025  
**Analysis By:** OpenHands AI Agent  
**Project:** npp-rl (N++ Reinforcement Learning)
