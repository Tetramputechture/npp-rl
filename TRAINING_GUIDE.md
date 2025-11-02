# N++ RL Training Guide

**Version**: 1.0  
**Last Updated**: 2025-11-02  
**Based on**: Comprehensive analysis of 1M timestep baseline training

---

## Quick Start

### Option 1: Conservative Improvements (Recommended First)

**Use case**: Validate reward structure fixes quickly (12-18 hours)

```bash
# Train with improved reward scaling but same MLP architecture
python scripts/train_and_compare.py \
    --config config_improved_conservative.json \
    --output-dir ./experiments/conservative_v1
```

**Expected Results** (3M steps):
- Success on simplest_with_mines: 70-75% (baseline: 60%)
- Mean reward: Positive (baseline: negative)
- Curriculum: Stage 2-3 (baseline: stuck at stage 1)

---

### Option 2: Aggressive Improvements (Full Stack)

**Use case**: Maximum performance with architecture upgrade (2-3 days)

```bash
# Train with GAT architecture + all improvements
python scripts/train_and_compare.py \
    --config config_improved_aggressive.json \
    --output-dir ./experiments/aggressive_v1
```

**Expected Results** (10M steps):
- Success on simple: 75-85%
- Success on medium: 60-70%
- Curriculum: Stage 5+ completion
- Average completion time: < 6000 steps

---

## What Changed?

### Critical Reward Structure Fixes

**File Modified**: `nclone/gym_environment/reward_calculation/reward_constants.py`

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| `PBRS_SWITCH_DISTANCE_SCALE` | 1.0 | **5.0** | 5x larger PBRS rewards |
| `PBRS_EXIT_DISTANCE_SCALE` | 1.0 | **5.0** | 5x larger PBRS rewards |
| `PBRS_HAZARD_WEIGHT` | 0.1 | **0.5** | 5x stronger mine avoidance |
| `EXPLORATION_CELL_REWARD` | 0.001 | **0.005** | 5x more exploration incentive |
| `NOOP_ACTION_PENALTY` | -0.01 | **-0.02** | 2x penalty for standing still |

**Why?**
- Baseline PBRS rewards were ~0.009/step (should be ~0.1/step)
- Agent experienced 97.5% negative rewards → insufficient positive feedback
- Mine avoidance not learned (60% vs 82% success)

---

### Training Configuration Changes

#### Conservative Config:
```json
{
  "total_timesteps": 3000000,        // Was 1M → 3x longer
  "num_envs": 64,                    // Was 28 → 2.3x parallelism
  "curriculum_threshold": 0.7,       // Was 0.8 → easier advancement
  "curriculum_min_episodes": 100     // Was 50 → more stable
}
```

#### Aggressive Config:
```json
{
  "architectures": ["gat"],          // Was mlp_baseline → adds graph reasoning
  "total_timesteps": 10000000,       // 10x longer for full curriculum
  "num_envs": 128,                   // 4.5x parallelism
  "enable_lr_annealing": true,       // Cosine LR decay
  "bc_epochs": 75                    // 50% more pretraining
}
```

---

## Understanding the Fixes

### 1. PBRS Scaling Problem

**Before**:
```python
# Distance normalized by surface area → too small
area_scale = sqrt(3000) * 12 ≈ 660 pixels
potential = 1.0 - (distance / 660)
reward = 0.995 * potential' - potential ≈ ±0.009  ❌ TOO SMALL
```

**After**:
```python
# Scale multiplier increases magnitude
scale = area_scale * 5.0  # 5x multiplier
potential = 1.0 - (distance / (scale * 5.0))
reward = 0.995 * potential' - potential ≈ ±0.045  ✅ EFFECTIVE
```

**Impact**: PBRS now provides meaningful guidance toward objectives.

---

### 2. Exploration Rewards

**Before**: 0.001 per cell explored (negligible vs -0.0001 time penalty)  
**After**: 0.005 per cell explored (5x larger, balances time penalty)

**Why it matters**:
- Agent repeats same routes without exploration incentive
- Diverse exploration improves generalization
- Prevents premature convergence to suboptimal solutions

---

### 3. Hazard Avoidance

**Before**: PBRS_HAZARD_WEIGHT = 0.1 (too weak)  
**After**: PBRS_HAZARD_WEIGHT = 0.5 (5x stronger)

**Why it matters**:
- Agent struggled specifically with mines (60% vs 82% success)
- Weak hazard signal didn't influence behavior
- Stronger signal encourages safe navigation

---

## Monitoring Training

### Real-Time Health Monitoring

Run the monitoring script alongside training:

```bash
# In separate terminal
python tools/monitor_training.py \
    --logdir ./experiments/your_run_name \
    --check-interval 60
```

**What it checks**:
- ✅ Mean reward becoming positive
- ✅ PBRS rewards in target range (±0.05-0.2)
- ✅ Success rate improving
- ⚠️ Value loss not increasing
- ⚠️ KL divergence < 0.1
- ⚠️ Entropy staying > 1.0
- ⚠️ NOOP usage < 15%

**Red flags that trigger alerts**:
- 🔴 Mean reward negative after 2M steps → reward structure still broken
- 🔴 PBRS < 0.02 → scaling insufficient
- 🟡 Value loss increasing → training instability
- 🟡 KL > 0.1 → policy changing too fast
- 🟡 Entropy < 1.0 → premature convergence

---

### Key Metrics to Watch in Tensorboard

Launch Tensorboard:
```bash
tensorboard --logdir ./experiments --port 6006
```

**Critical Metrics** (check these first):

1. **`reward_dist/mean`**: Should become POSITIVE by 1M steps
   - Before: -0.018 ❌
   - Target: +0.05 to +0.2 ✅

2. **`pbrs_rewards/pbrs_mean`**: Should be in ±0.05-0.2 range
   - Before: -0.0088 ❌
   - Target: ±0.05 to ±0.2 ✅

3. **`curriculum/success_rate`**: Should improve steadily
   - Stage 1 baseline: 60%
   - Stage 1 target: 70%+ by 1M steps

4. **`actions/frequency/NOOP`**: Should decrease
   - Before: 17.9% ❌
   - Target: < 10% ✅

5. **`train/value_loss`**: Should decrease consistently
   - Should NOT increase after initial decrease

---

### Success Milestones

**@ 1M steps** (Conservative):
- [ ] Mean reward > 0
- [ ] PBRS rewards ≈ ±0.05
- [ ] Success on simplest_with_mines ≥ 70%
- [ ] NOOP usage < 15%

**@ 3M steps** (Conservative):
- [ ] Curriculum advanced to stage 2
- [ ] Success on simplest_with_mines ≥ 75%
- [ ] Mean reward > +0.1
- [ ] NOOP usage < 12%

**@ 10M steps** (Aggressive):
- [ ] Curriculum stage 4-5 reached
- [ ] Success on medium ≥ 60%
- [ ] Average completion time < 7000 steps
- [ ] NOOP usage < 10%

---

## Troubleshooting

### Issue: Mean Reward Still Negative After 1M Steps

**Symptoms**:
- `reward_dist/mean` staying around -0.015 to -0.020
- `reward_dist/negative_ratio` > 95%

**Diagnosis**:
```bash
# Check PBRS rewards
tensorboard --logdir=./experiments --tag pbrs_rewards/pbrs_mean
# Should see values around ±0.05, not ±0.01
```

**Solution**:
1. Verify reward_constants.py changes applied:
   ```bash
   grep "PBRS_SWITCH_DISTANCE_SCALE" nclone/nclone/gym_environment/reward_calculation/reward_constants.py
   # Should show 5.0, not 1.0
   ```

2. If still 1.0, rebuild/reinstall nclone:
   ```bash
   cd nclone
   pip install -e .
   ```

3. If PBRS rewards still small, increase scale further:
   ```python
   PBRS_SWITCH_DISTANCE_SCALE = 10.0  # Try 10x instead of 5x
   ```

---

### Issue: Agent Stuck at Curriculum Stage 1

**Symptoms**:
- `curriculum/current_stage_idx` = 1 for > 500k steps
- `curriculum/can_advance` = 0.0
- `curriculum_stages/simplest_with_mines_success_rate` plateaued < 70%

**Diagnosis**:
```bash
# Check success rate trend
tensorboard --logdir=./experiments --tag curriculum_stages/simplest_with_mines_success_rate
# If flat line for 500k steps → stuck
```

**Solutions**:

**Option A**: Lower curriculum threshold
```json
{
  "curriculum_threshold": 0.65  // Was 0.7 → even easier
}
```

**Option B**: Check mine avoidance learning
```bash
# Look at hazard potential
tensorboard --tag pbrs_potentials/hazard_mean
# Should show non-zero values indicating hazard awareness
```

If hazard_mean ≈ 0, increase hazard weight:
```python
PBRS_HAZARD_WEIGHT = 0.7  # Was 0.5 → stronger
```

**Option C**: Add intermediate curriculum stage
See "Advanced: Curriculum Micro-Stages" section below.

---

### Issue: Training Unstable (Value Loss Increasing)

**Symptoms**:
- `train/value_loss` increasing after initial decrease
- `train/approx_kl` > 0.1
- `train/clip_fraction` > 50%

**Diagnosis**:
- Policy changing too aggressively
- Learning rate too high
- Batch size too small

**Solutions**:

**Option A**: Reduce learning rate
```python
# Add to config or modify hyperparameters
learning_rate = 0.0001  # Was 0.0003 → 3x slower
```

**Option B**: Increase batch size
```python
batch_size = 512  # Was 256 → 2x larger
n_steps = 2048    # Was 1024 → 2x larger
```

**Option C**: Reduce clip range
```python
clip_range = 0.1  # Was 0.2 → more conservative
```

---

### Issue: Agent Too Deterministic (Low Entropy)

**Symptoms**:
- `actions/entropy` < 1.0
- Policy converged prematurely
- Success rate plateaued early

**Diagnosis**:
- Insufficient exploration
- Entropy coefficient too low

**Solutions**:

**Option A**: Increase entropy coefficient
```python
# In ppo_hyperparameters.py
ent_coef = 0.05  # Was 0.02 → 2.5x stronger exploration
```

**Option B**: Use entropy scheduling
```python
# Decay from high to low over training
initial_ent_coef = 0.05
final_ent_coef = 0.01
# Implement via custom callback
```

---

### Issue: NOOP Usage Still High

**Symptoms**:
- `actions/frequency/NOOP` > 15% after 1M steps
- Agent stands still frequently

**Diagnosis**:
- NOOP penalty too small
- Or agent learned standing still is safe

**Solutions**:

**Option A**: Increase NOOP penalty
```python
NOOP_ACTION_PENALTY = -0.05  # Was -0.02 → 2.5x stronger
```

**Option B**: Check if time penalty is too harsh
```python
# If agent avoiding all actions, reduce time penalty
TIME_PENALTY_PER_STEP = -0.00005  # Was -0.0001 → gentler
```

---

## Advanced Topics

### Curriculum Micro-Stages

If agent struggles with simplest → simplest_with_mines transition, add intermediate stages:

**File**: `npp-rl/npp_rl/training/curriculum_manager.py` (modify)

```python
stages = [
    "simplest",                     # No mines
    "simplest_with_mines_10",      # NEW: 10% mine density
    "simplest_with_mines_25",      # NEW: 25% mine density
    "simplest_with_mines_50",      # NEW: 50% mine density
    "simplest_with_mines",          # 100% mine density (original)
    "simpler",
    "simple",
    ...
]
```

**Implementation**: Requires modifying level generator to support density parameter.

---

### Learning Rate Scheduling

For fine-grained control over learning rate:

**Linear Annealing** (built-in):
```python
# In config
enable_lr_annealing = True
initial_lr = 0.0003
final_lr = 0.00003
# LR decays linearly from initial to final over training
```

**Cosine Annealing** (custom callback):
```python
import numpy as np

class CosineAnnealingCallback(BaseCallback):
    def __init__(self, initial_lr, final_lr, total_timesteps):
        super().__init__()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps
    
    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        cos_decay = 0.5 * (1 + np.cos(np.pi * progress))
        lr = self.final_lr + (self.initial_lr - self.final_lr) * cos_decay
        self.model.lr_schedule = lambda _: lr
        return True
```

---

### Progressive Time Penalty

Encourage early exploration with progressive penalty:

**File**: `reward_constants.py`

```python
# Already defined, just enable in config
TIME_PENALTY_EARLY = -0.00005    # Steps 0-30%: gentle
TIME_PENALTY_MIDDLE = -0.0002    # Steps 30-70%: moderate
TIME_PENALTY_LATE = -0.0005      # Steps 70-100%: strict
```

**Config**:
```json
{
  "time_penalty_mode": "progressive"  // Was "fixed"
}
```

**Benefit**: Agent explores freely early, learns urgency later.

---

### Architecture Comparison Study

To validate architecture choice, run ablation:

```bash
# Test all architectures systematically
for arch in mlp_baseline gcn gat simplified_hgt; do
    python scripts/train_and_compare.py \
        --config config_improved_conservative.json \
        --architectures $arch \
        --output-dir ./ablation/$arch \
        --total-timesteps 3000000
done

# Compare results
python scripts/compare_architectures.py --logdir ./ablation
```

**Expected Ranking** (by sample efficiency):
1. GAT / Simplified HGT (best)
2. GCN (good)
3. MLP Baseline (baseline)

---

## Best Practices

### 1. Always Run Monitor Script

Don't waste compute on broken training:
```bash
# Start monitoring immediately when training starts
python tools/monitor_training.py --logdir ./experiments/run_name &
```

### 2. Use Multiple Seeds

For reliable results, run 3 seeds per config:
```bash
for seed in 42 123 456; do
    python scripts/train_and_compare.py \
        --config config.json \
        --seed $seed \
        --output-dir ./experiments/run_seed_$seed
done
```

### 3. Save Checkpoints Frequently

Disk is cheap, compute is expensive:
```json
{
  "save_freq": 500000  // Save every 500k steps
}
```

### 4. Enable Video Recording

Visual debugging is invaluable:
```json
{
  "record_eval_videos": true,
  "max_videos_per_category": 5
}
```

### 5. Use Mixed Precision

2x speedup with minimal accuracy loss:
```json
{
  "mixed_precision": true
}
```

### 6. Log Everything

Storage is cheap:
```bash
# Pipe stdout to log file
python scripts/train_and_compare.py ... 2>&1 | tee training.log
```

### 7. Use Tmux/Screen for Long Training

Don't lose training due to SSH disconnect:
```bash
# Start tmux session
tmux new -s training

# Run training
python scripts/train_and_compare.py ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## Hardware Requirements

### Minimum (Conservative Config):
- **GPU**: 1x GPU with 12GB VRAM (e.g., RTX 3060)
- **RAM**: 32GB
- **Storage**: 50GB
- **Time**: 12-18 hours for 3M steps

### Recommended (Aggressive Config):
- **GPU**: 1x GPU with 24GB VRAM (e.g., RTX 3090, A5000)
- **RAM**: 64GB
- **Storage**: 100GB
- **Time**: 2-3 days for 10M steps

### Optimal (Multi-GPU):
- **GPU**: 2-4x GPUs with 24GB VRAM each
- **RAM**: 128GB
- **Storage**: 200GB SSD
- **Time**: 1-2 days for 10M steps

---

## Expected Training Timeline

### Conservative (3M steps, 64 envs):

| Time | Steps | Success Rate | Curriculum Stage |
|------|-------|--------------|------------------|
| 0h | 0 | ~50% | simplest (stage 0) |
| 4h | 1M | 65-70% | simplest_with_mines (stage 1) |
| 8h | 2M | 70-75% | stage 1 → advancing |
| 12h | 3M | 75-80% | simpler (stage 2) |

### Aggressive (10M steps, 128 envs):

| Time | Steps | Success Rate | Curriculum Stage |
|------|-------|--------------|------------------|
| 0h | 0 | ~50% | simplest (stage 0) |
| 6h | 2M | 70-75% | simplest_with_mines (stage 1) |
| 12h | 4M | 75-80% | simpler (stage 2) |
| 18h | 6M | 70-80% | simple (stage 3) |
| 24h | 8M | 65-75% | medium (stage 4) |
| 36h | 10M | 60-70% | complex (stage 5) |

---

## FAQ

### Q: Why not just increase training time without other fixes?

**A**: Training longer won't help if reward structure is broken. The agent experienced 97.5% negative rewards - more training with same setup = more negative reinforcement = worse learning.

**Must fix reward structure first, then extend training.**

---

### Q: Can I use the old config with just PBRS scaling changes?

**A**: Yes, but not recommended. Critical issues:
- 1M timesteps too short (need minimum 3M)
- 28 envs limits parallelism (use 64+)
- 0.8 curriculum threshold too strict (use 0.7)

Conservative config bundles all minimal necessary changes.

---

### Q: Do I need to use GAT architecture?

**A**: No. MLP baseline with reward fixes should reach 70-75% success. But:
- GAT enables relational reasoning → better generalization
- GAT handles complex stages better (medium, complex)
- For full curriculum mastery, graph architecture recommended

**Recommendation**: Start with MLP + reward fixes, upgrade to GAT if needed.

---

### Q: How do I know if my training is working?

**A**: Check these at 500k steps:
1. Mean reward approaching zero (was very negative)
2. PBRS rewards ≈ ±0.05 (was ±0.01)
3. Success rate > 65% on current stage
4. Value loss decreasing

If ANY of these failing, stop and debug.

---

### Q: Can I resume from old checkpoint?

**A**: Not recommended if old model trained with broken rewards. The policy learned wrong behaviors (e.g., avoid positive rewards). 

**Better to start fresh with fixed rewards.**

If you must resume:
```python
resume_from = "/path/to/checkpoint.zip"
# But expect initial performance drop as policy unlearns bad behaviors
```

---

### Q: What if I can't afford 10M steps?

**A**: Use conservative config (3M steps). Should reach stage 2-3, which is 2x better than baseline. Then:
1. Evaluate results
2. If promising, extend to 5M steps
3. If still improving, extend to 10M steps

**Incremental approach saves compute if issues arise.**

---

## Getting Help

### If Training Fails

1. **Check monitor output**: Look for red flags
2. **Inspect tensorboard**: Focus on critical metrics
3. **Review troubleshooting**: Common issues covered above
4. **Check logs**: Look for errors/warnings
5. **Compare to baseline**: Are metrics improving?

### Debugging Checklist

- [ ] Reward constants updated? (grep for "5.0")
- [ ] Config loaded correctly? (check experiment_name)
- [ ] Monitor script running? (check for alerts)
- [ ] Tensorboard accessible? (port 6006)
- [ ] Sufficient disk space? (need 50-100GB)
- [ ] GPU being used? (nvidia-smi)
- [ ] No OOM errors? (reduce batch_size if needed)

---

## Summary

**Critical Changes**:
1. ✅ PBRS scaling: 1.0 → 5.0 (rewards 5x larger)
2. ✅ Hazard weight: 0.1 → 0.5 (mine avoidance 5x stronger)
3. ✅ Exploration: 0.001 → 0.005 (5x more incentive)
4. ✅ Training: 1M → 3M steps minimum (3x longer)
5. ✅ Parallelism: 28 → 64 envs (2.3x faster)

**Expected Improvements**:
- Mean reward: negative → positive
- Success rate: 60% → 75% on simplest_with_mines
- Curriculum: stage 1 → stage 2-3
- NOOP usage: 17.9% → < 12%

**Next Steps**:
1. Run conservative config (12-18 hours)
2. Monitor training health
3. If successful, run aggressive config (2-3 days)
4. Compare architectures (MLP vs GCN vs GAT)
5. Push to production

---

**Good luck with training! 🚀**

For issues or questions, check troubleshooting section or review analysis document.
