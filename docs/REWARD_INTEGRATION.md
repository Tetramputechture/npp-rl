# Reward System Integration Guide for npp-rl

## Overview

This guide explains how to integrate npp-rl's intrinsic motivation system with nclone's production-ready reward constants.

## nclone Reward System

The nclone repository now provides a comprehensive, well-documented reward system with centralized constants. All reward parameters are defined in `nclone.gym_environment.reward_calculation.reward_constants`.

### Key Features

1. **No Magic Numbers** - All constants documented with research backing
2. **PBRS Theory** - Policy-invariant reward shaping (Ng et al. 1999)
3. **Multi-Scale Exploration** - Count-based spatial coverage (Bellemare et al. 2016)
4. **ICM Integration** - Constants for intrinsic curiosity (Pathak et al. 2017)
5. **Production Ready** - Validation, presets, comprehensive testing

## Using nclone Reward Constants in npp-rl

### Importing Constants

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    # Terminal rewards
    LEVEL_COMPLETION_REWARD,
    DEATH_PENALTY,
    SWITCH_ACTIVATION_REWARD,
    TIME_PENALTY_PER_STEP,
    
    # ICM constants
    ICM_ALPHA,
    ICM_REWARD_CLIP,
    ICM_FORWARD_LOSS_WEIGHT,
    ICM_INVERSE_LOSS_WEIGHT,
    ICM_LEARNING_RATE,
    
    # Configuration presets
    get_completion_focused_config,
    get_exploration_focused_config,
)
```

### Updating IntrinsicRewardWrapper

Replace hardcoded values in `npp_rl/wrappers/intrinsic_reward_wrapper.py`:

```python
# Before (hardcoded)
alpha: float = 0.1,
r_int_clip: float = 1.0,

# After (using constants)
from nclone.gym_environment.reward_calculation.reward_constants import (
    ICM_ALPHA, ICM_REWARD_CLIP
)

alpha: float = ICM_ALPHA,
r_int_clip: float = ICM_REWARD_CLIP,
```

### Updating ICM Training

In `npp_rl/intrinsic/icm.py`, use centralized constants:

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    ICM_FORWARD_LOSS_WEIGHT,
    ICM_INVERSE_LOSS_WEIGHT,
    ICM_LEARNING_RATE,
)

class ICMTrainer:
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        learning_rate: float = ICM_LEARNING_RATE,
        forward_loss_weight: float = ICM_FORWARD_LOSS_WEIGHT,
        inverse_loss_weight: float = ICM_INVERSE_LOSS_WEIGHT,
    ):
        # Implementation...
```

## ICM Constants Reference

### ICM_ALPHA (Intrinsic/Extrinsic Combination Weight)

**Value:** `0.1`

**Usage:** Weight for combining intrinsic and extrinsic rewards.

**Formula:** `total_reward = extrinsic + ICM_ALPHA * intrinsic`

**Rationale:**
- Set to 10% to provide exploration boost without overwhelming task rewards
- Based on Pathak et al. (2017) standard practices
- Higher values (e.g., 0.3) increase exploration but may distract from task
- Lower values (e.g., 0.05) focus more on extrinsic objectives

### ICM_REWARD_CLIP (Maximum Intrinsic Reward)

**Value:** `1.0`

**Usage:** Maximum intrinsic reward value to prevent instability.

**Rationale:**
- Prevents large prediction errors from dominating learning
- Especially important during early training when predictions are poor
- Comparable to terminal reward magnitude (LEVEL_COMPLETION_REWARD = 1.0)
- Can be adjusted based on observation space complexity

### ICM_FORWARD_LOSS_WEIGHT (Forward Model Weight)

**Value:** `0.9`

**Usage:** Weight for forward model loss in ICM training.

**Formula:** `total_icm_loss = ICM_FORWARD_LOSS_WEIGHT * forward_loss + ICM_INVERSE_LOSS_WEIGHT * inverse_loss`

**Rationale:**
- Forward model (0.9) dominates for curiosity signal generation
- Based on Pathak et al. (2017) original implementation
- Forward model prediction error drives exploration
- Higher weight ensures robust curiosity signals

### ICM_INVERSE_LOSS_WEIGHT (Inverse Model Weight)

**Value:** `0.1`

**Usage:** Weight for inverse model loss in ICM training.

**Rationale:**
- Inverse model (0.1) provides auxiliary learning objective
- Helps learn action-relevant features
- Lower weight prevents over-fitting to controllable features
- Balances with forward model for stable training

### ICM_LEARNING_RATE (ICM Network Learning Rate)

**Value:** `1e-3` (0.001)

**Usage:** Learning rate for ICM network optimizer.

**Rationale:**
- Moderate learning rate for auxiliary task learning
- Higher than policy network (typical 3e-4) as ICM learns faster
- ICM is a supervised learning task (easier than RL)
- Can be adjusted based on network architecture

## Integration Example

Complete example of using reward constants in npp-rl training:

```python
import torch
from stable_baselines3 import PPO
from nclone import NPPEnvironment
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,
    ICM_ALPHA,
    ICM_REWARD_CLIP,
    ICM_LEARNING_RATE,
)
from npp_rl.intrinsic.icm import ICMTrainer
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper

# Create base environment
env = NPPEnvironment(
    render_mode="rgb_array",
    dataset_dir="datasets/train",
)

# Get reward configuration
reward_config = get_completion_focused_config()
print(f"Using completion-focused rewards:")
print(f"  Completion: +{reward_config['level_completion_reward']}")
print(f"  Death: {reward_config['death_penalty']}")
print(f"  Time penalty: {reward_config['time_penalty']}")

# Create ICM with constants
feature_dim = 512  # From policy network
action_dim = 6     # N++ action space

icm_trainer = ICMTrainer(
    feature_dim=feature_dim,
    action_dim=action_dim,
    learning_rate=ICM_LEARNING_RATE,
)

# Wrap environment with ICM
env = IntrinsicRewardWrapper(
    env,
    icm_trainer=icm_trainer,
    alpha=ICM_ALPHA,
    r_int_clip=ICM_REWARD_CLIP,
)

# Create PPO agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",
)

# Train
model.learn(total_timesteps=1_000_000)
```

## Configuration Presets for npp-rl

Different reward configurations for different training phases:

### Phase 1: Exploration with ICM (0-1M steps)

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_exploration_focused_config,
    ICM_ALPHA,
)

# Use exploration-focused config with boosted ICM
config = get_exploration_focused_config()
icm_alpha = ICM_ALPHA * 2.0  # Double ICM influence (0.2)
```

**Characteristics:**
- High exploration rewards (3×)
- Low time pressure (0.1×)
- Strong ICM signal (0.2)
- Encourages discovery and map coverage

### Phase 2: Completion Focus (1M-5M steps)

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,
    ICM_ALPHA,
)

# Standard completion-focused config
config = get_completion_focused_config()
icm_alpha = ICM_ALPHA  # Standard ICM (0.1)
```

**Characteristics:**
- Fast level completion priority
- Standard exploration (1×)
- Moderate ICM signal (0.1)
- Efficient solution finding

### Phase 3: Fine-tuning (5M+ steps)

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,
    ICM_ALPHA,
)

# Completion config with reduced ICM
config = get_completion_focused_config()
icm_alpha = ICM_ALPHA * 0.5  # Reduced ICM (0.05)
```

**Characteristics:**
- Pure completion focus
- Minimal exploration
- Low ICM signal (0.05)
- Policy refinement

## Best Practices

### 1. Always Use Constants

❌ **Don't:**
```python
alpha = 0.1  # Why 0.1? What does this mean?
```

✅ **Do:**
```python
from nclone.gym_environment.reward_calculation.reward_constants import ICM_ALPHA
alpha = ICM_ALPHA  # Clearly documented constant
```

### 2. Document Deviations

If you deviate from default constants, document why:

```python
from nclone.gym_environment.reward_calculation.reward_constants import ICM_ALPHA

# Custom alpha for high-dimensional observation space
# Rationale: Complex visual observations require stronger exploration signal
icm_alpha_custom = ICM_ALPHA * 1.5  # 0.15 instead of 0.1
```

### 3. Validate Custom Configurations

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    validate_reward_config,
    print_reward_summary,
)

custom_config = {
    # Your custom settings...
}

# Validate before training
validate_reward_config(custom_config)
print_reward_summary(custom_config)
```

### 4. Monitor Reward Components

Track both extrinsic and intrinsic rewards during training:

```python
# In training loop
info = env.get_stats()
logger.log("extrinsic_reward", info["episode_stats"]["r_ext_sum"])
logger.log("intrinsic_reward", info["episode_stats"]["r_int_sum"])
logger.log("total_reward", info["episode_stats"]["r_total_sum"])
logger.log("icm_alpha", info["global_stats"]["alpha"])
```

### 5. Adjust Based on Environment Complexity

Different level difficulties may need different ICM settings:

```python
from nclone.gym_environment.reward_calculation.reward_constants import ICM_ALPHA

# Simple levels (curriculum level 0-1)
icm_alpha_simple = ICM_ALPHA * 0.5  # 0.05 - less exploration needed

# Complex levels (curriculum level 3-4)
icm_alpha_complex = ICM_ALPHA * 1.5  # 0.15 - more exploration needed
```

## Troubleshooting

### Problem: ICM overwhelming extrinsic rewards

**Symptoms:**
- Agent explores aimlessly
- Low completion rate
- High intrinsic rewards, low extrinsic rewards

**Solution:**
```python
# Reduce ICM_ALPHA
from nclone.gym_environment.reward_calculation.reward_constants import ICM_ALPHA
icm_alpha = ICM_ALPHA * 0.5  # Reduce to 0.05
```

### Problem: Insufficient exploration

**Symptoms:**
- Agent gets stuck in local optima
- Poor discovery of switches/exits
- Low intrinsic rewards

**Solution:**
```python
# Increase ICM_ALPHA or use exploration config
from nclone.gym_environment.reward_calculation.reward_constants import (
    ICM_ALPHA,
    get_exploration_focused_config,
)
icm_alpha = ICM_ALPHA * 2.0  # Increase to 0.2
config = get_exploration_focused_config()
```

### Problem: Unstable ICM training

**Symptoms:**
- ICM losses diverge
- Intrinsic rewards become very large
- Training crashes

**Solution:**
```python
# Reduce ICM_LEARNING_RATE or clip rewards more aggressively
from nclone.gym_environment.reward_calculation.reward_constants import (
    ICM_LEARNING_RATE,
    ICM_REWARD_CLIP,
)
icm_lr = ICM_LEARNING_RATE * 0.5  # Reduce to 5e-4
icm_clip = ICM_REWARD_CLIP * 0.5  # Clip to 0.5
```

## References

1. **Ng, A.Y., Harada, D., and Russell, S. (1999).** "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." ICML 1999.

2. **Pathak, D., Agrawal, P., Efros, A.A., and Darrell, T. (2017).** "Curiosity-driven Exploration by Self-supervised Prediction." ICML 2017.

3. **Bellemare, M. et al. (2016).** "Unifying Count-Based Exploration and Intrinsic Motivation." NIPS 2016.

4. **nclone Reward System Documentation:** See `nclone/docs/REWARD_SYSTEM.md` for comprehensive reward system documentation.

## Next Steps

1. Update `npp_rl/wrappers/intrinsic_reward_wrapper.py` to use constants
2. Update `npp_rl/intrinsic/icm.py` to use constants
3. Add validation in training scripts
4. Create example notebooks showing different configurations
5. Benchmark different ICM_ALPHA values on N++ levels

---

For questions or issues, refer to the main reward system documentation in nclone or open an issue on GitHub.
