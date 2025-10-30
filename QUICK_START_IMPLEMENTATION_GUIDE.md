# Quick Start: Implementing Critical Fixes

## Priority 1: Immediate Actions (2-3 days implementation)

### Fix 1: Reward Structure (HIGHEST PRIORITY) 🔥

**File:** `../nclone/nclone/gym_environment/npp_environment.py`

**Current Problem:** Agent getting -40 average reward per episode

**Change:**
```python
# Find the reward calculation method (likely in step() or _calculate_reward())
# Current (approximate):
def _calculate_reward(self, info):
    if self.player_won:
        return 1000.0
    if self.ninja.state == NinjaState.DEAD:
        return -100.0
    return -0.1  # Per frame penalty

# Change to:
def _calculate_reward(self, info):
    reward = 0.0
    
    # Normalize completion reward
    if self.player_won:
        reward += 1.0  # Changed from 1000.0
        # Add time bonus
        time_bonus = max(0, 1.0 - self.frames / self.max_frames)
        reward += 0.5 * time_bonus
    
    # Reduce death penalty
    if self.ninja.state == NinjaState.DEAD:
        reward -= 0.1  # Changed from -100.0
    
    # Reduce time penalty 10x
    reward -= 0.0001  # Changed from -0.1
    
    # Add milestone reward
    if not self._switch_touched_flag and info.get('switch_touched', False):
        reward += 0.5
        self._switch_touched_flag = True
    
    return reward

# Don't forget to add in __init__:
def __init__(self, ...):
    # ... existing code ...
    self._switch_touched_flag = False

# And reset in reset():
def reset(self, **kwargs):
    # ... existing code ...
    self._switch_touched_flag = False
    # ... rest of reset ...
```

**Test:**
```bash
# Run short test
python -m npp_rl.training.train \
    --architectures mlp_baseline \
    --total-timesteps 100000 \
    --output-dir test_reward_fix \
    --experiment-name test_reward_v1

# Check TensorBoard - reward should be positive!
tensorboard --logdir test_reward_fix
# Look for: rewards/hierarchical_mean > 0
```

---

### Fix 2: PBRS Debugging 🔥

**File:** `npp_rl/hrl/subtask_rewards.py` or `npp_rl/wrappers/hierarchical_reward_wrapper.py`

**Current Problem:** PBRS contributing -0.0043 (negative!)

**Add Debugging:**
```python
# In the calculate_subtask_reward method:

def calculate_subtask_reward(self, curr_obs, prev_obs, subtask):
    # ... existing code to calculate potential_curr and potential_prev ...
    
    # IMPORTANT: Ensure potentials are non-negative
    potential_curr = max(0.0, min(1.0, potential_curr))
    potential_prev = max(0.0, min(1.0, potential_prev))
    
    # Calculate PBRS
    pbrs_reward = self.pbrs_gamma * potential_curr - potential_prev
    
    # DEBUG LOGGING - Add this temporarily
    if hasattr(self, '_pbrs_stats'):
        self._pbrs_stats['rewards'].append(pbrs_reward)
        self._pbrs_stats['potentials_curr'].append(potential_curr)
        self._pbrs_stats['potentials_prev'].append(potential_prev)
        
        if len(self._pbrs_stats['rewards']) >= 1000:
            import numpy as np
            print(f"PBRS Stats (last 1000 steps):")
            print(f"  Mean reward: {np.mean(self._pbrs_stats['rewards']):.6f}")
            print(f"  Mean potential_curr: {np.mean(self._pbrs_stats['potentials_curr']):.6f}")
            print(f"  Mean potential_prev: {np.mean(self._pbrs_stats['potentials_prev']):.6f}")
            print(f"  Positive %: {np.mean(np.array(self._pbrs_stats['rewards']) > 0):.1%}")
            self._pbrs_stats = {'rewards': [], 'potentials_curr': [], 'potentials_prev': []}
    
    return pbrs_reward

# Add in __init__:
def __init__(self, ...):
    # ... existing code ...
    self._pbrs_stats = {'rewards': [], 'potentials_curr': [], 'potentials_prev': []}
```

**Verify Gamma:**
```python
# Make sure gamma is reasonable
self.pbrs_gamma = 0.99  # Not 0.995 (too high)
```

---

### Fix 3: Configuration Changes 🔥

**File:** Create new config file `configs/fixed_config_v1.json`

```json
{
  "experiment_name": "mlp-fix-v1",
  "architectures": ["mlp_baseline"],
  
  "total_timesteps": 5000000,
  "num_envs": 128,
  "eval_freq": 500000,
  "save_freq": 1000000,
  
  "enable_visual_frame_stacking": false,
  "enable_state_stacking": false,
  
  "use_curriculum": true,
  "curriculum_threshold": 0.4,
  "curriculum_min_episodes": 100,
  
  "enable_pbrs": true,
  "pbrs_gamma": 0.99,
  "enable_mine_avoidance_reward": true,
  
  "no_pretraining": false,
  "bc_epochs": 20,
  
  "hardware_profile": "auto",
  "mixed_precision": true
}
```

**Or modify existing config:**
```bash
# Copy current config
cp latest-training-results/config.json configs/fixed_config_v1.json

# Edit with these changes:
# Line  9: "enable_visual_frame_stacking": false,
# Line 15: "num_envs": 128,
# Line 14: "total_timesteps": 5000000,
# Line 28: "curriculum_threshold": 0.4,
```

---

### Fix 4: Run Test Training

```bash
# Run 2M timestep test with all fixes
python -m npp_rl.training.train \
    --config configs/fixed_config_v1.json \
    --total-timesteps 2000000 \
    --num-envs 128 \
    --experiment-name mlp-fix-phase1-test \
    --output-dir experiments/phase1_fixes

# This should take ~18-24 hours on A100
```

---

## Success Criteria for Phase 1

After 2M timesteps, you should see:

### ✅ Rewards (MOST IMPORTANT)
- [ ] Average episode reward: **> 0** (currently -40)
- [ ] Reward trend: **Increasing**
- [ ] PBRS mean: **> -0.001** (currently -0.0043)

### ✅ Success Rates
- [ ] Simplest: **> 85%** (currently 78%)
- [ ] Simpler: **> 60%** (currently 45%)
- [ ] Simple: **> 40%** (currently 27%)
- [ ] Test set: **> 5%** (currently 0%)

### ✅ Curriculum
- [ ] Curriculum advances to **stage 3 (medium)**
- [ ] Not stuck at simple/simpler

### ✅ Actions
- [ ] NOOP: **< 12%** (currently 18%)
- [ ] Jump: **< 45%** (currently 52%)

### ✅ Learning
- [ ] Explained variance: **> 0.8** (currently 0.87, keep it up!)
- [ ] Clip fraction: **< 0.35** (currently 0.41)

---

## Monitoring During Training

### TensorBoard Commands
```bash
# Start TensorBoard
tensorboard --logdir experiments/phase1_fixes --port 6006

# Monitor these metrics:
# - rewards/hierarchical_mean (should go positive!)
# - episode/success_rate_smoothed (should increase)
# - curriculum/current_stage_idx (should reach 2+)
# - actions/frequency/NOOP (should decrease)
```

### Check Progress Every 500K Steps
```bash
# Script to check key metrics
python << 'EOF'
import json
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

exp_dir = Path("experiments/phase1_fixes")
event_file = list(exp_dir.glob("**/events.out.tfevents.*"))[0]

ea = event_accumulator.EventAccumulator(str(event_file))
ea.Reload()

# Get latest values
def get_latest(tag):
    events = ea.Scalars(tag)
    return events[-1].value if events else None

print("=" * 60)
print("PROGRESS CHECK")
print("=" * 60)
print(f"Avg Reward:    {get_latest('rewards/hierarchical_mean'):.2f}")
print(f"Success Rate:  {get_latest('episode/success_rate_smoothed'):.1%}")
print(f"Curriculum:    Stage {get_latest('curriculum/current_stage_idx'):.0f}")
print(f"NOOP:          {get_latest('actions/frequency/NOOP'):.1%}")
print(f"PBRS Mean:     {get_latest('pbrs_rewards/pbrs_mean'):.6f}")
print("=" * 60)

# Check if on track
reward = get_latest('rewards/hierarchical_mean')
if reward > 0:
    print("✅ REWARD IS POSITIVE - GOOD!")
else:
    print("❌ REWARD STILL NEGATIVE - INVESTIGATE")
EOF
```

---

## Troubleshooting

### If Reward Still Negative After 500K Steps

**Check 1: Are the code changes applied?**
```bash
cd ../nclone
git diff nclone/gym_environment/npp_environment.py
# Should show your reward changes
```

**Check 2: Is PBRS the culprit?**
```python
# Temporarily disable PBRS
# In config: "enable_pbrs": false
# Re-run and check if reward goes positive
```

**Check 3: Print raw rewards**
```python
# Add to environment step():
if self.steps % 100 == 0:
    print(f"Step {self.steps}: Reward={reward:.4f}")
```

### If Curriculum Not Advancing

**Check 1: Success rate on current stage**
```bash
# Look at curriculum/success_rate in TensorBoard
# Should be > 40% to advance with new threshold
```

**Check 2: Manually force advancement**
```python
# In curriculum_manager.py, temporarily:
def should_advance(self):
    return True  # Force advance to test next stage
```

### If Training Crashes

**Common Issues:**
1. **OOM (Out of Memory)**
   - Reduce num_envs to 64
   - Reduce batch_size to 256
   
2. **NaN Loss**
   - Reduce learning_rate to 0.0001
   - Check reward bounds (clip to [-10, 10])
   
3. **Slow Training**
   - Check frame stacking is OFF
   - Profile with: `python -m cProfile -o profile.stats train.py`

---

## Next Steps After Phase 1

Once Phase 1 works (positive rewards, curriculum advancing):

### Phase 2: Scale Up (Week 2)
1. Increase to 10M timesteps
2. Add dense reward shaping (distance-based)
3. Tune PPO hyperparameters
4. Reduce episode timeout to 2500 frames

### Phase 3: Polish (Week 3)
1. Add intrinsic motivation (ICM)
2. Implement reward normalization (VecNormalize)
3. Train for 20M+ timesteps
4. Comprehensive evaluation

---

## Quick Reference: File Locations

```
Reward calculation:
  ../nclone/nclone/gym_environment/npp_environment.py
  
PBRS implementation:
  npp_rl/hrl/subtask_rewards.py
  npp_rl/wrappers/hierarchical_reward_wrapper.py
  
Configuration:
  configs/*.json
  
Training script:
  npp_rl/training/train.py
  npp_rl/training/architecture_trainer.py
  
Curriculum:
  npp_rl/training/curriculum_manager.py
  
Callbacks:
  npp_rl/training/training_callbacks.py
  npp_rl/callbacks/enhanced_tensorboard_callback.py
```

---

## Code Review Checklist

Before running full training, verify:

- [ ] Reward calculation modified (10x smaller penalties)
- [ ] Switch-touched reward added (+0.5)
- [ ] PBRS potentials clamped to [0, 1]
- [ ] Frame stacking disabled in config
- [ ] num_envs increased to 128
- [ ] curriculum_threshold lowered to 0.4
- [ ] total_timesteps increased to 5M+
- [ ] Git branch created and committed
- [ ] Backup of old results
- [ ] TensorBoard ready to monitor

---

## Expected Timeline

```
Day 1:  Implement reward changes, test locally
Day 2:  Implement PBRS fixes, test locally
Day 3:  Update config, start 2M test run
Day 4-5: Monitor test run, adjust if needed
Day 6:  Analyze results, prepare 10M run
Day 7+: Full 10M+ training run
```

---

**Remember:** The goal of Phase 1 is to get positive average rewards. Once that's working, everything else becomes easier!

Good luck! 🚀
