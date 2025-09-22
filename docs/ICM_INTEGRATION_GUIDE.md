# ICM Integration Guide

This guide explains how to integrate the reachability-aware Intrinsic Curiosity Module (ICM) with PPO training and nclone's exploration reward calculator.

## Overview

The ICM system provides curiosity-driven exploration by predicting the consequences of actions and rewarding the agent for encountering novel or hard-to-predict situations. Our implementation integrates with nclone's reachability analysis to provide more informed exploration rewards.

## Architecture

```
PPO Training (training.py)
    ↓
ICM Module (npp_rl.intrinsic.icm)
    ↓
Reachability Exploration (npp_rl.intrinsic.reachability_exploration)
    ↓
nclone Systems (TieredReachabilitySystem, CompactReachabilityFeatures, etc.)
```

## Key Components

### 1. ICMNetwork
- **Purpose**: Neural network that learns to predict next states and actions
- **Input**: Current state, action, next state
- **Output**: Intrinsic reward based on prediction error

### 2. ICMTrainer
- **Purpose**: Handles training of the ICM network
- **Features**: Gradient clipping, loss tracking, performance monitoring

### 3. ReachabilityAwareExplorationCalculator
- **Purpose**: Enhances ICM rewards with reachability analysis
- **Integration**: Uses nclone's frontier detection and compact features

## Integration with PPO Training

### Basic Setup

```python
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.intrinsic.reachability_exploration import ReachabilityAwareExplorationCalculator

# Initialize ICM components
icm_network = ICMNetwork(
    state_dim=observation_space.shape[0],
    action_dim=action_space.n,
    hidden_dim=256,
    feature_dim=128
)

icm_trainer = ICMTrainer(
    icm_network=icm_network,
    learning_rate=1e-4,
    beta=0.2,  # Balance between forward and inverse loss
    eta=0.01   # Intrinsic reward scaling
)

# Initialize reachability-aware exploration
exploration_calc = ReachabilityAwareExplorationCalculator(
    debug=False,
    reachability_weight=0.5,
    frontier_bonus=0.3
)
```

### Training Loop Integration

```python
def training_step(observations, actions, next_observations, rewards, dones):
    # 1. Calculate ICM intrinsic rewards
    icm_rewards = icm_network.calculate_intrinsic_reward(
        observations, actions, next_observations
    )
    
    # 2. Get reachability information
    reachability_info = icm_network.get_reachability_info(observations)
    
    # 3. Calculate enhanced exploration rewards
    exploration_rewards = exploration_calc.calculate_reachability_aware_reward(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        base_icm_rewards=icm_rewards,
        reachability_info=reachability_info
    )
    
    # 4. Combine with extrinsic rewards
    total_rewards = rewards + exploration_rewards['total_reward']
    
    # 5. Train ICM
    icm_loss = icm_trainer.train_step(observations, actions, next_observations)
    
    # 6. Continue with PPO training using total_rewards
    # ... PPO update logic ...
    
    return {
        'icm_loss': icm_loss,
        'intrinsic_reward_mean': icm_rewards.mean().item(),
        'exploration_reward_mean': exploration_rewards['total_reward'].mean().item(),
        'reachability_available': reachability_info.get('available', False)
    }
```

### Configuration in training.py

Add these parameters to your training configuration:

```python
# ICM Configuration
ICM_CONFIG = {
    'enabled': True,
    'hidden_dim': 256,
    'feature_dim': 128,
    'learning_rate': 1e-4,
    'beta': 0.2,           # Forward vs inverse loss balance
    'eta': 0.01,           # Intrinsic reward scaling
    'gradient_clip': 1.0,
    'update_frequency': 1   # Train ICM every N steps
}

# Reachability Configuration
REACHABILITY_CONFIG = {
    'enabled': True,
    'weight': 0.5,         # How much to weight reachability vs ICM
    'frontier_bonus': 0.3, # Extra reward for frontier exploration
    'novelty_threshold': 0.1,
    'debug': False
}
```

## Integration with nclone ExplorationRewardCalculator

The ICM system is designed to work alongside nclone's existing exploration reward calculator:

```python
from nclone.gym_environment.reward_calculation.exploration_reward_calculator import ExplorationRewardCalculator

# Initialize both systems
nclone_exploration = ExplorationRewardCalculator(
    # ... nclone configuration ...
)

icm_exploration = ReachabilityAwareExplorationCalculator(
    # ... ICM configuration ...
)

def calculate_combined_exploration_rewards(observations, actions, next_observations):
    # Get nclone exploration rewards
    nclone_rewards = nclone_exploration.calculate_reward(
        observations, actions, next_observations
    )
    
    # Get ICM-based exploration rewards
    icm_rewards = icm_network.calculate_intrinsic_reward(
        observations, actions, next_observations
    )
    
    # Get reachability-enhanced rewards
    reachability_info = icm_network.get_reachability_info(observations)
    enhanced_rewards = icm_exploration.calculate_reachability_aware_reward(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        base_icm_rewards=icm_rewards,
        reachability_info=reachability_info
    )
    
    # Combine rewards (weights can be tuned)
    total_exploration_reward = (
        0.4 * nclone_rewards +
        0.6 * enhanced_rewards['total_reward']
    )
    
    return total_exploration_reward
```

## Performance Considerations

### Memory Usage
- ICM networks are lightweight (~1-5MB depending on configuration)
- Reachability analysis adds minimal overhead (<1ms per step)
- Consider batch processing for efficiency

### Hyperparameter Tuning
- **beta**: Higher values emphasize forward model accuracy
- **eta**: Controls intrinsic reward magnitude relative to extrinsic rewards
- **reachability_weight**: Balance between ICM and reachability rewards
- **frontier_bonus**: Extra reward for exploring new areas

### Monitoring
Track these metrics during training:
- `icm_loss`: Should decrease over time
- `intrinsic_reward_mean`: Should remain positive but not dominate
- `exploration_reward_mean`: Combined exploration signal
- `reachability_available`: Percentage of steps with reachability data

## Example Training Script Modifications

```python
# In your main training loop
def train_ppo_with_icm():
    # ... existing setup ...
    
    # Add ICM components
    icm_network = ICMNetwork(**ICM_CONFIG)
    icm_trainer = ICMTrainer(icm_network, **ICM_CONFIG)
    exploration_calc = ReachabilityAwareExplorationCalculator(**REACHABILITY_CONFIG)
    
    for episode in range(num_episodes):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        
        # Collect episode data
        for step in range(max_steps):
            action = policy.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            
            obs = next_obs
            if done:
                break
        
        # Convert to tensors
        obs_tensor = torch.stack(observations)
        act_tensor = torch.tensor(actions)
        next_obs_tensor = torch.stack(next_observations)
        reward_tensor = torch.tensor(rewards)
        
        # Calculate enhanced rewards
        enhanced_rewards = calculate_combined_exploration_rewards(
            obs_tensor, act_tensor, next_obs_tensor
        )
        
        # Train ICM
        icm_loss = icm_trainer.train_step(obs_tensor, act_tensor, next_obs_tensor)
        
        # Train PPO with enhanced rewards
        total_rewards = reward_tensor + enhanced_rewards
        ppo_loss = ppo_trainer.train_step(obs_tensor, act_tensor, total_rewards)
        
        # Log metrics
        logger.log({
            'episode': episode,
            'icm_loss': icm_loss,
            'ppo_loss': ppo_loss,
            'mean_reward': total_rewards.mean().item(),
            'mean_intrinsic_reward': enhanced_rewards.mean().item()
        })
```

## Troubleshooting

### Common Issues

1. **High ICM Loss**: 
   - Reduce learning rate
   - Increase gradient clipping
   - Check input normalization

2. **Low Intrinsic Rewards**:
   - Increase eta parameter
   - Check reachability integration
   - Verify observation preprocessing

3. **Performance Issues**:
   - Reduce ICM update frequency
   - Use smaller network dimensions
   - Batch reachability calculations

### Debug Mode

Enable debug mode for detailed logging:

```python
exploration_calc = ReachabilityAwareExplorationCalculator(debug=True)
```

This will log:
- Reachability analysis results
- Reward component breakdowns
- Performance timing information
- Integration status with nclone systems

## Best Practices

1. **Start Simple**: Begin with basic ICM, then add reachability features
2. **Monitor Balance**: Ensure intrinsic rewards don't overwhelm extrinsic ones
3. **Tune Gradually**: Adjust hyperparameters incrementally
4. **Use Logging**: Track all reward components for analysis
5. **Test Integration**: Verify nclone systems are working correctly

## API Reference

### ICMNetwork Methods
- `calculate_intrinsic_reward(obs, actions, next_obs)`: Calculate curiosity rewards
- `get_reachability_info(obs)`: Extract reachability analysis
- `forward(obs, actions, next_obs)`: Full forward pass

### ICMTrainer Methods
- `train_step(obs, actions, next_obs)`: Single training step
- `get_metrics()`: Training metrics and statistics

### ReachabilityAwareExplorationCalculator Methods
- `calculate_reachability_aware_reward(...)`: Enhanced exploration rewards
- `is_nclone_available()`: Check nclone integration status