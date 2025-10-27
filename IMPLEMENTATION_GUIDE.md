# Implementation Guide for Training Fixes

This guide provides step-by-step instructions to implement the critical fixes identified in the comprehensive training analysis.

## Quick Start (Emergency Fixes)

**Time Required:** 1-2 hours  
**Expected Impact:** Stop training degradation, enable learning  
**Priority:** CRITICAL ⚠️

### Step 1: Update Reward Constants (5 minutes)

**File:** `../nclone/nclone/gym_environment/reward_calculation/reward_constants.py`

Replace the following constants:

```python
# OLD (BROKEN):
LEVEL_COMPLETION_REWARD = 1.0
SWITCH_ACTIVATION_REWARD = 0.1
TIME_PENALTY_PER_STEP = -0.01
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.0001
PBRS_SWITCH_DISTANCE_SCALE = 0.05
PBRS_EXIT_DISTANCE_SCALE = 0.05
EXPLORATION_CELL_REWARD = 0.001
EXPLORATION_AREA_4X4_REWARD = 0.001
EXPLORATION_AREA_8X8_REWARD = 0.001
EXPLORATION_AREA_16X16_REWARD = 0.001

# NEW (FIXED):
LEVEL_COMPLETION_REWARD = 10.0           # 10x increase
SWITCH_ACTIVATION_REWARD = 1.0            # 10x increase
TIME_PENALTY_PER_STEP = -0.0001           # 100x decrease (CRITICAL!)
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001  # 10x increase
PBRS_SWITCH_DISTANCE_SCALE = 0.5          # 10x increase
PBRS_EXIT_DISTANCE_SCALE = 0.5            # 10x increase
EXPLORATION_CELL_REWARD = 0.01            # 10x increase
EXPLORATION_AREA_4X4_REWARD = 0.01        # 10x increase
EXPLORATION_AREA_8X8_REWARD = 0.01        # 10x increase
EXPLORATION_AREA_16X16_REWARD = 0.01      # 10x increase
```

**Verification:** Run `python REWARD_CONSTANTS_FIXED.py` to validate the changes.

### Step 2: Enable VecNormalize Wrapper (10 minutes)

**File:** `npp_rl/agents/training.py`

Find the section where environments are created (around line 200-300), and add VecNormalize:

```python
from stable_baselines3.common.vec_env import VecNormalize

# After creating vec_env (SubprocVecEnv or DummyVecEnv):
vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

# ADD THIS:
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=True,  # CRITICAL: Normalize returns
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
    epsilon=1e-8,
    training=True,
    norm_obs_keys=None,  # Normalize all observation keys
)
```

**Important:** Also wrap eval_env with VecNormalize (set `training=False`).

### Step 3: Add Value Function Clipping (5 minutes)

**File:** Where PPO is instantiated (likely `npp_rl/agents/training.py`)

Add to `policy_kwargs` or PPO arguments:

```python
# When creating PPO:
model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    # ... other args ...
    
    # ADD THESE:
    clip_range_vf=10.0,  # Clip value predictions to [-10, 10]
    # Note: SB3 doesn't have direct vf_clip_param, but clip_range_vf helps
)
```

### Step 4: Update Curriculum Thresholds (10 minutes)

**File:** Find curriculum configuration (likely `npp_rl/wrappers/curriculum_env.py` or similar)

```python
# OLD (TOO AGGRESSIVE):
ADVANCEMENT_THRESHOLD = 0.70  # Same for all stages

# NEW (ADAPTIVE):
ADVANCEMENT_THRESHOLDS = {
    "simplest": 0.80,
    "simpler": 0.70,
    "simple": 0.60,      # LOWERED from 0.70
    "medium": 0.55,      # LOWERED from 0.70
    "complex": 0.50,     # LOWERED from 0.70
    "mine_heavy": 0.45,  # LOWERED from 0.70
    "exploration": 0.40, # LOWERED from 0.70
}

# ADD REGRESSION CAPABILITY:
REGRESSION_THRESHOLDS = {
    "simpler": 0.30,
    "simple": 0.30,
    "medium": 0.25,
    "complex": 0.20,
    "mine_heavy": 0.15,
    "exploration": 0.15,
}

MIN_EPISODES_FOR_REGRESSION = 200
```

### Step 5: Increase Environment Parallelism (2 minutes)

**File:** Training config or command-line args

```python
# OLD:
num_envs = 14

# NEW:
num_envs = 32  # Minimum recommended
# or
num_envs = 64  # Better for sample efficiency

# Also increase batch size and n_steps proportionally:
batch_size = 512      # was 256
n_steps = 2048        # was 1024
```

### Step 6: Enable PBRS Weights (5 minutes)

**File:** Environment configuration or reward wrapper

```python
# OLD (LIKELY):
pbrs_weights = {
    "objective_weight": 1.0,
    "hazard_weight": 0.0,    # Disabled
    "impact_weight": 0.0,    # Disabled
    "exploration_weight": 0.0,  # Disabled
}

# NEW (ENABLE SHAPING):
pbrs_weights = {
    "objective_weight": 1.0,
    "hazard_weight": 0.1,       # Enable mild hazard avoidance
    "impact_weight": 0.0,       # Keep disabled for speed focus
    "exploration_weight": 0.2,  # Enable exploration bonus
}
```

---

## Validation After Emergency Fixes

### Run Short Test Training (30 minutes)

```bash
# Start training with fixed config
python -m npp_rl.agents.training \
    --num_envs 32 \
    --total_timesteps 100000 \
    --architecture mlp_baseline \
    --curriculum True \
    --config config_fixed.json

# Monitor tensorboard:
tensorboard --logdir=<experiment_dir>
```

### Check These Metrics After 50k Steps:

1. **Value Estimates** - Should be in [-5, 5] range (not [-7, 0])
2. **Success Rate** - Should be stable or increasing (not declining)
3. **Episode Returns** - Positive for completions (not negative!)
4. **Curriculum Stage** - Should advance past stage 2 or show improvement

### Red Flags (Stop if you see these):

- ❌ Value estimates still going below -10
- ❌ Success rate still declining
- ❌ Episode returns negative for completions
- ❌ No curriculum advancement after 100k steps

---

## Advanced Improvements (Optional)

### Add Learning Rate Scheduling

**File:** PPO initialization

```python
from stable_baselines3.common.utils import linear_schedule

model = PPO(
    # ...
    learning_rate=linear_schedule(3e-4, 3e-5),  # Decay 10x
    # ...
)
```

### Add Entropy Coefficient Annealing

```python
# Custom callback for entropy scheduling
class EntropyScheduler(BaseCallback):
    def __init__(self, initial_ent=0.02, final_ent=0.005, total_steps=1_000_000):
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_steps = total_steps
    
    def _on_step(self):
        progress = self.num_timesteps / self.total_steps
        current_ent = self.initial_ent + (self.final_ent - self.initial_ent) * progress
        self.model.ent_coef = current_ent
        return True
```

### Implement Curriculum Regression

**File:** Curriculum wrapper

```python
class AdaptiveCurriculum:
    def update(self, stage_idx, success_rate, episodes):
        # Check for regression
        if self.should_regress(stage_idx, success_rate, episodes):
            print(f"Regressing from stage {stage_idx} to {stage_idx-1}")
            return stage_idx - 1
        
        # Check for advancement
        if self.should_advance(stage_idx, success_rate, episodes):
            print(f"Advancing from stage {stage_idx} to {stage_idx+1}")
            return stage_idx + 1
        
        return stage_idx
    
    def should_regress(self, stage_idx, success_rate, episodes):
        if stage_idx == 0:  # Can't regress from first stage
            return False
        if episodes < MIN_EPISODES_FOR_REGRESSION:
            return False
        threshold = REGRESSION_THRESHOLDS.get(self.stage_names[stage_idx], 0.3)
        return success_rate < threshold
    
    def should_advance(self, stage_idx, success_rate, episodes):
        if episodes < MIN_EPISODES_FOR_ADVANCEMENT:
            return False
        threshold = ADVANCEMENT_THRESHOLDS.get(self.stage_names[stage_idx], 0.7)
        return success_rate >= threshold
```

### Implement Mixed Training

```python
class MixedCurriculumSampler:
    def sample_level(self, current_stage, mastered_stages):
        """Sample from current, previous, and next stages."""
        r = random.random()
        
        # 70% current stage
        if r < 0.70:
            return self.sample_from_stage(current_stage)
        
        # 20% from mastered stages (prevent forgetting)
        elif r < 0.90 and len(mastered_stages) > 0:
            stage = random.choice(mastered_stages)
            return self.sample_from_stage(stage)
        
        # 10% from next stage (preview)
        elif current_stage < self.max_stage:
            return self.sample_from_stage(current_stage + 1)
        
        return self.sample_from_stage(current_stage)
```

---

## Testing & Validation Checklist

### Before Starting Full Training:

- [ ] Reward constants updated and validated
- [ ] VecNormalize wrapper added
- [ ] Value clipping enabled
- [ ] Curriculum thresholds adjusted
- [ ] Environment parallelism increased
- [ ] PBRS weights enabled
- [ ] Short test run (100k steps) successful
- [ ] Tensorboard showing correct metrics

### During Training (Check Every 100k Steps):

- [ ] Value estimates in reasonable range ([-10, 10])
- [ ] Success rate stable or improving
- [ ] Episode returns positive for completions
- [ ] Curriculum advancing (should reach stage 3+ by 500k steps)
- [ ] No NaN values in any metric
- [ ] GPU utilization high (>80%)
- [ ] No memory leaks (RAM stable)

### Success Criteria (After 1M Steps):

- [ ] Reached curriculum stage 4+ (was stuck at stage 2)
- [ ] 60%+ success on stages 0-2
- [ ] 40%+ success on stage 3
- [ ] Value function stable (std < 5)
- [ ] Policy converging (entropy declining to ~1.0)

---

## Troubleshooting

### Issue: Value estimates still collapsing

**Solutions:**
1. Check VecNormalize is active: `print(env.ret_rms.mean)`
2. Increase value network size: `vf: [1024, 1024, 512, 256]`
3. Try Huber loss instead of MSE (requires PPO modification)
4. Reduce learning rate: `3e-5` instead of `3e-4`

### Issue: Curriculum still stuck

**Solutions:**
1. Lower advancement threshold further: `0.50` instead of `0.60`
2. Increase min episodes: `200` instead of `100`
3. Check level difficulty distribution
4. Try pure stage 2 training (disable curriculum temporarily)

### Issue: Training too slow

**Solutions:**
1. Increase num_envs to 64
2. Use smaller batch size updates more frequently
3. Reduce n_steps to 1024
4. Disable video recording during training
5. Use mixed precision (already enabled)

### Issue: Out of memory

**Solutions:**
1. Reduce num_envs to 24
2. Reduce batch size to 256
3. Reduce n_steps to 1024
4. Reduce network sizes by 25%
5. Use gradient accumulation

---

## File Checklist

Files that need modification:

1. ✅ **reward_constants.py** - Reward scaling fixes
2. ✅ **training.py** - VecNormalize wrapper
3. ✅ **curriculum_env.py** - Adaptive thresholds and regression
4. ⬜ **ppo_hyperparameters.py** - Learning rate scheduling
5. ⬜ **config.json** - Update all hyperparameters
6. ⬜ **environment creation** - Enable PBRS weights

Files in this directory:

- `COMPREHENSIVE_TRAINING_ANALYSIS.md` - Full analysis (70 pages)
- `REWARD_CONSTANTS_FIXED.py` - Fixed reward values with validation
- `config_fixed.json` - Complete fixed configuration
- `IMPLEMENTATION_GUIDE.md` - This file
- `analysis_tensorboard.py` - Analysis script used

---

## Emergency Contacts

If issues persist after implementing all fixes:

1. Review full analysis: `COMPREHENSIVE_TRAINING_ANALYSIS.md`
2. Check tensorboard logs in detail
3. Run diagnostics: `python analysis_tensorboard.py`
4. Consider alternative approaches (Section 7 of analysis)

---

## Expected Timeline

| Phase | Duration | Goal |
|-------|----------|------|
| Implementation | 1-2 hours | Apply all emergency fixes |
| Test run | 3-5 hours | 100k steps validation |
| Full training | 2-3 days | 1-2M steps to stage 4+ |
| Optimization | 1 week | Tune for best performance |

---

## Success Metrics Summary

| Metric | Current | Target (Phase 1) | Target (Final) |
|--------|---------|------------------|----------------|
| Stage 2 Success | 4% | 40%+ | 70%+ |
| Value Mean | -4.33 | [-2, 2] | [-1, 1] |
| Curriculum Stage | 2 (stuck) | 3-4 | 5-6 |
| Episode Return (win) | Negative! | Positive | +8 to +10 |
| Training Trend | Declining | Stable | Improving |

---

**Last Updated:** 2025-10-27  
**Version:** 1.0  
**See Also:** COMPREHENSIVE_TRAINING_ANALYSIS.md
