# NPP-RL Agent

A Deep Reinforcement Learning Agent for the game N++, implementing PPO (Proximal Policy Optimization) using Stable Baselines3 with a custom N++ simulation environment.

## Project Overview

This project trains an agent to play the game [N++](https://en.wikipedia.org/wiki/N%2B%2B). The agent learns to navigate complex, physics-based levels, activate switches, and reach exits by interacting with a custom Gym-compatible environment derived from the `nclone` simulator.

The agent architecture incorporates several features informed by recent deep reinforcement learning research to enhance performance, generalization, and sample efficiency.

## Core Agent Features & Architecture

The PPO agent leverages a multi-input policy capable of processing visual and vector-based game state information. Key architectural components include:

### 1. Observation Space & Processing

The agent receives multi-modal observations:

*   **Player-Centric Visual Frames**:
    *   Dimensions: 84x84 pixels.
    *   Temporal Stacking: 12 consecutive frames are stacked to provide temporal context.
        *   `TEMPORAL_FRAMES = 12` (defined in `nclone/nclone/gym_environment/constants.py`).
    *   Preprocessing: Grayscale conversion, centering on the player, cropping, and normalization.
    *   Augmentation: Random cutout is applied to player frames with a 50% chance to improve generalization (inspired by DeVries & Taylor, 2017, "Improved Regularization of Convolutional Neural Networks with Cutout").

*   **Global View**:
    *   A downsampled 176x100 pixel grayscale view of the entire level.

*   **Game State Vector**: A low-dimensional vector containing:
    *   Ninja physics state (position, velocity, airborne/walled status, jump duration, applied forces).
    *   Status of critical entities (e.g., exit door, switches).
    *   Normalized vectors from the ninja to key objectives (e.g., active switch, exit).
    *   Time remaining in the episode.

### 2. Feature Extraction

*   **`HGTMultimodalExtractor`:
    *   Implements **Heterogeneous Graph Transformers** with type-specific attention mechanisms.
    *   **Type-aware processing**: Specialized handling for different node types (grid cells, entities, hazards, switches).
    *   **Edge-type specialization**: Distinct processing for movement edges (walk, jump, fall) vs functional relationships.
    *   **Multi-head attention**: Advanced attention mechanisms adapted for heterogeneous graph structures.
    *   **Entity-aware embeddings**: Specialized processing for different entity types with hazard-aware attention.
    *   **Advanced multimodal fusion**: Cross-modal attention with spatial awareness for optimal feature integration.

The extractor processes the game state vector through a dedicated Multi-Layer Perceptron (MLP). The features from visual inputs, graph representations, and the game state vector are then fused and passed to the policy and value networks.

### 3. Network Architecture & Hyperparameters

*   **Policy and Value Networks**:
    *   The PPO agent uses separate MLP heads for the policy (actor) and value (critic) functions, following the feature extraction stage.
    *   Network Size: Hidden layers are configured as `[256, 256, 128]` (`NET_ARCH_SIZE` in `agents/hyperparameters/ppo_hyperparameters.py`).
*   **Key PPO Hyperparameters** (tuned values in `agents/hyperparameters/ppo_hyperparameters.py`):
    *   `n_steps`: 1024 (Number of steps per environment per update)
    *   `batch_size`: 256 (Minibatch size for optimization)
    *   `gamma`: 0.999 (Discount factor)
    *   `learning_rate`: Linearly decayed, typically from `3e-4` to `1e-6`.
    *   Other parameters such as `gae_lambda`, `clip_range`, `ent_coef`, and `vf_coef` are also defined.

### 4. Adaptive Exploration Strategies

To encourage efficient exploration and improve learning in sparse reward environments, the agent can utilize an `AdaptiveExplorationManager` (from `agents/adaptive_exploration.py`). This system combines:

*   **Intrinsic Curiosity Module (ICM)**:
    *   Based on Pathak et al. (2017), "Curiosity-driven Exploration by Self-supervised Prediction."
    *   The module consists of a forward model (predicting the next state's feature representation given the current state and action) and an inverse model (predicting the action taken between two consecutive states).
    *   The prediction error of the forward model serves as an intrinsic reward signal, encouraging the agent to visit states where its understanding of the environment dynamics is poor.
*   **Novelty Detection**:
    *   Employs a count-based approach where states (discretized player positions) are tracked.
    *   A novelty bonus is awarded for visiting less frequently encountered states, decaying over time. This is inspired by classic count-based exploration algorithms.
*   **Adaptive Scaling**:
    *   The overall magnitude of the exploration bonus (combined from ICM and novelty) is dynamically adjusted based on the agent's training progress (e.g., rate of extrinsic reward improvement).

## ICM + PPO Integration Guide

This section provides detailed guidance on how to effectively combine the Intrinsic Curiosity Module (ICM) with the regular PPO agent for enhanced exploration and learning performance.

### Overview

The ICM integration provides intrinsic motivation to the PPO agent by generating curiosity-driven exploration bonuses. This is particularly valuable in sparse reward environments like N++ where the agent may need to explore extensively before finding successful strategies.

### Architecture Integration

The ICM system integrates with PPO through the `AdaptiveExplorationManager` class, which acts as a bridge between the curiosity mechanisms and the standard PPO training loop:

```python
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager

# Initialize the exploration manager
exploration_manager = AdaptiveExplorationManager()

# Initialize ICM with appropriate dimensions
exploration_manager.initialize_curiosity_module(
    feature_dim=128,  # Match your feature extractor output
    action_dim=6      # N++ action space size
)
```

### nclone Planning System Integration

The ICM system leverages nclone's planning and reachability analysis for strategic exploration guidance:

```python
# The AdaptiveExplorationManager automatically integrates with nclone components:
# - EntityInteractionSubgoal: Unified subgoal representation
# - LevelCompletionPlanner: Strategic level completion planning  
# - SubgoalPrioritizer: Intelligent subgoal ranking and selection
# - ReachabilitySystem: Spatial analysis for feasible paths

# These components work together to provide structured exploration targets
# that guide the ICM's curiosity-driven learning process
```

**Key Integration Points:**

1. **Subgoal Generation**: nclone's planning system generates `EntityInteractionSubgoal` instances that represent strategic targets (switches, exits, navigation points)

2. **Reachability Analysis**: nclone's spatial analysis determines which subgoals are feasible from the current state

3. **Strategic Planning**: The `LevelCompletionPlanner` provides structured completion strategies that guide long-term exploration

4. **Priority-Based Selection**: The `SubgoalPrioritizer` ranks subgoals based on strategic value, reachability, and current game state

5. **ICM Feature Integration**: Subgoal information is incorporated into ICM feature representations to bias curiosity toward strategically relevant exploration

### Integration Workflow

#### 1. Training Loop Integration

The ICM should be integrated into your PPO training loop as follows:

```python
# During environment step
obs, reward, done, info = env.step(action)

# Calculate intrinsic reward bonus
intrinsic_bonus = exploration_manager.get_exploration_bonus(
    state=previous_features,
    action=action,
    next_state=current_features
)

# Combine extrinsic and intrinsic rewards
total_reward = reward + intrinsic_bonus

# Update ICM networks (important!)
exploration_manager.update_curiosity_module(
    state=previous_features,
    action=action,
    next_state=current_features
)
```

#### 2. Feature Extraction Compatibility

The ICM requires feature representations of the game state. Ensure your feature extractor outputs are compatible:

```python
# Your feature extractor should output consistent feature dimensions
features = feature_extractor(observation)  # Shape: [batch_size, feature_dim]

# ICM expects these features for curiosity calculation
curiosity_bonus = exploration_manager.get_exploration_bonus(
    state=features,
    action=action_tensor,
    next_state=next_features
)
```

#### 3. Hierarchical Subgoal Integration

The system supports hierarchical planning through reachability-guided subgoals using the unified `EntityInteractionSubgoal` architecture from nclone:

```python
# Get strategic subgoals for the current state
subgoals = exploration_manager.get_available_subgoals(
    ninja_pos=(player_x, player_y),
    level_data=level_info,
    switch_states=current_switches
)

# Use subgoals for reward shaping
for subgoal in subgoals:
    reward_bonus = subgoal.get_reward_shaping(ninja_pos)
    total_reward += reward_bonus * subgoal.priority

# Get level completion strategy using nclone's planning system
completion_strategy = exploration_manager.get_completion_strategy(
    ninja_pos=(player_x, player_y),
    level_data=level_info,
    switch_states=current_switches
)

# The completion strategy provides structured steps for level completion
for step in completion_strategy.steps:
    print(f"Step: {step.action_type} -> {step.target_position}")
```

**Subgoal Architecture:**
- **EntityInteractionSubgoal**: Unified subgoal class handling both navigation and switch activation
- **Backward Compatibility**: Legacy `NavigationSubgoal` and `SwitchActivationSubgoal` classes are maintained as aliases
- **nclone Integration**: Subgoals are generated using nclone's reachability analysis and planning system
- **Strategic Planning**: Level completion strategies provide structured guidance for ICM exploration

### Configuration Parameters

#### ICM Hyperparameters

Key parameters for ICM integration:

```python
# ICM Network Architecture
FEATURE_DIM = 128           # Feature representation size
HIDDEN_DIM = 256           # ICM network hidden layer size
LEARNING_RATE_ICM = 1e-3   # ICM learning rate (separate from PPO)

# Curiosity Scaling
CURIOSITY_SCALE = 0.1      # Scale factor for intrinsic rewards
NOVELTY_SCALE = 0.05       # Scale factor for novelty bonuses
ADAPTIVE_SCALING = True    # Enable adaptive exploration scaling

# Update Frequencies
ICM_UPDATE_FREQ = 1        # Update ICM every N steps
SUBGOAL_UPDATE_FREQ = 100  # Update subgoals every N steps
```

#### PPO Hyperparameter Adjustments

When using ICM, consider adjusting these PPO parameters:

```python
# Recommended adjustments for ICM integration
PPO_CONFIG = {
    'learning_rate': 3e-4,      # Standard rate works well
    'gamma': 0.999,             # High discount for long-term planning
    'gae_lambda': 0.95,         # Standard GAE parameter
    'ent_coef': 0.01,           # Slightly higher entropy for exploration
    'vf_coef': 0.5,             # Standard value coefficient
    'max_grad_norm': 0.5,       # Gradient clipping
    'n_steps': 2048,            # Longer rollouts for better ICM training
    'batch_size': 64,           # Smaller batches for stability
}
```

### Best Practices

#### 1. Reward Balance

Carefully balance extrinsic and intrinsic rewards:

```python
# Adaptive reward scaling based on training progress
def get_reward_scaling(episode_count, success_rate):
    if success_rate < 0.1:
        # High exploration phase
        return {'extrinsic': 1.0, 'intrinsic': 0.5}
    elif success_rate < 0.5:
        # Balanced phase
        return {'extrinsic': 1.0, 'intrinsic': 0.2}
    else:
        # Exploitation phase
        return {'extrinsic': 1.0, 'intrinsic': 0.05}
```

#### 2. Feature Normalization

Ensure consistent feature scaling for ICM:

```python
# Normalize features before passing to ICM
features = torch.nn.functional.normalize(raw_features, dim=-1)
curiosity_bonus = exploration_manager.get_exploration_bonus(
    state=features, action=action, next_state=next_features
)
```

#### 3. Monitoring and Debugging

Track key metrics for ICM performance:

```python
# Log ICM statistics
stats = exploration_manager.get_statistics()
logger.log({
    'icm/forward_loss': stats.get('forward_loss', 0),
    'icm/inverse_loss': stats.get('inverse_loss', 0),
    'icm/curiosity_reward': stats.get('avg_curiosity_reward', 0),
    'icm/novelty_bonus': stats.get('avg_novelty_bonus', 0),
    'exploration/total_intrinsic_reward': stats['total_intrinsic_reward'],
    'exploration/exploration_scale': stats['exploration_scale'],
})
```

### Common Issues and Solutions

#### Issue 1: ICM Overpowering Extrinsic Rewards

**Symptoms:** Agent explores endlessly without completing objectives
**Solution:** Reduce curiosity scaling or implement adaptive scaling

```python
# Implement curiosity decay
curiosity_scale = initial_scale * (decay_rate ** episode_count)
```

#### Issue 2: Feature Dimension Mismatch

**Symptoms:** Runtime errors during ICM forward pass
**Solution:** Ensure feature extractor output matches ICM input dimensions

```python
# Add dimension checking
assert features.shape[-1] == exploration_manager.feature_dim, \
    f"Feature dim mismatch: {features.shape[-1]} vs {exploration_manager.feature_dim}"
```

#### Issue 3: Poor Subgoal Quality

**Symptoms:** Subgoals lead to suboptimal behavior
**Solution:** Tune subgoal prioritization and reachability analysis

```python
# Adjust subgoal filtering
filtered_subgoals = [s for s in subgoals if s.success_probability > 0.7]
```

### Performance Optimization

#### Memory Management

ICM can be memory-intensive. Consider these optimizations:

```python
# Use gradient checkpointing for ICM networks
exploration_manager.enable_gradient_checkpointing()

# Limit replay buffer size for novelty detection
exploration_manager.set_novelty_buffer_size(10000)
```

#### Computational Efficiency

Optimize ICM updates for better performance:

```python
# Batch ICM updates
if step_count % ICM_BATCH_SIZE == 0:
    exploration_manager.batch_update_curiosity_module(
        states_batch, actions_batch, next_states_batch
    )
```

### Example Training Script Integration

Here's a complete example of integrating ICM with PPO training:

```python
import torch
from stable_baselines3 import PPO
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager

def train_with_icm(env, total_timesteps=1000000):
    # Initialize exploration manager
    exploration_manager = AdaptiveExplorationManager()
    exploration_manager.initialize_curiosity_module(
        feature_dim=128, action_dim=env.action_space.n
    )
    
    # Initialize PPO agent
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Training loop with ICM integration
    obs = env.reset()
    for step in range(total_timesteps):
        # Get action from PPO
        action, _states = model.predict(obs, deterministic=False)
        
        # Environment step
        next_obs, reward, done, info = env.step(action)
        
        # Extract features (assuming custom feature extractor)
        features = model.policy.extract_features(obs)
        next_features = model.policy.extract_features(next_obs)
        
        # Calculate intrinsic reward
        intrinsic_reward = exploration_manager.get_exploration_bonus(
            state=features, action=torch.tensor([action]), 
            next_state=next_features
        )
        
        # Combine rewards
        total_reward = reward + 0.1 * intrinsic_reward
        
        # Update ICM
        exploration_manager.update_curiosity_module(
            state=features, action=torch.tensor([action]),
            next_state=next_features
        )
        
        # Store experience with modified reward
        model.replay_buffer.add(obs, next_obs, action, total_reward, done, info)
        
        obs = next_obs
        if done:
            obs = env.reset()
    
    return model
```

This integration approach ensures that the ICM enhances exploration while maintaining the stability and performance of the underlying PPO algorithm.

### 5. Action Space

The agent interacts with the environment using a discrete action set:
*   NOOP (No action)
*   Left
*   Right
*   Jump
*   Jump + Left
*   Jump + Right

### 6. Reward System

The extrinsic reward signal from the environment is designed to guide the agent towards completing levels efficiently. It typically includes:
*   Small penalty for each time step (encouraging speed).
*   Positive reward for activating switches.
*   Large positive reward for reaching the exit.
*   Large negative penalty for dying.
*   Exploration rewards at multiple spatial scales.

## Project Structure

Consolidated architecture focused on hierarchical multimodal processing:

- `npp_rl/`
  - `agents/`
    - `training.py`: **Primary training entrypoint** with hierarchical multimodal architecture, CLI interface, PPO, vectorized environments, and comprehensive logging.
    - `adaptive_exploration.py`: Optional curiosity/novelty exploration manager and helpers.
    - `hyperparameters/ppo_hyperparameters.py`: Tuned PPO defaults and `NET_ARCH_SIZE`.
  - `feature_extractors/`
    - `hierarchical_multimodal.py`: **Primary feature extractor** with multi-resolution graph processing, DiffPool GNNs, and adaptive fusion.
    - `__init__.py`: Unified interface with factory functions for hierarchical extractor.
  - (other subpackages may be added in later phases)
- Top-level scripts
  - `ppo_train.py`: Thin wrapper to launch PPO via enhanced training.
  - `tools/`: Small utilities (e.g., `convert_actions.py`, `rotate_videos.py`).

## Training the Agent

The primary script for training the agent with all features is `npp_rl/agents/training.py`.

### Prerequisites

#### System Requirements

Before starting, ensure you have:
- Python 3.8 or higher
- Git
- pip (Python package installer)
- System dependencies for PyCairo:
  ```bash
  sudo apt install libcairo2-dev pkg-config python3-dev
  ```

#### Setting up the Development Environment

1. Clone this repository:
   ```bash
   git clone https://github.com/tetramputechture/npp-rl.git
   cd npp-rl
   ```

2. (Optional but recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the `nclone` simulator:
   ```bash
   # Navigate to the directory containing npp-rl
   cd ..
   git clone https://github.com/tetramputechture/nclone.git
   cd nclone
   pip install -e .
   cd ../npp-rl
   ```

4. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Starting a Training Run

**Recommended Quick Start:**
This command starts training using the HGT-based multimodal extractor (PRIMARY), adaptive exploration, and optimized hyperparameters, utilizing 64 parallel environments.

```bash
python -m npp_rl.agents.training --num_envs 64 --total_timesteps 10000000 --extractor_type hgt
```

**Command-Line Options for `training.py`:**
The `training.py` script offers various options:

```bash
python -m npp_rl.agents.training --help
```

Key options include:
*   `--num_envs`: Number of parallel simulation environments (default: 64).
*   `--total_timesteps`: Total number of training steps (default: 10,000,000).
*   `--extractor_type`: Feature extractor type - `hgt` (recommended) or `hierarchical` (default: hgt).
*   `--load_model`: Path to a previously saved model checkpoint to resume training.
*   `--render_mode`: Set to `human` for visual rendering (forces `num_envs=1`). Default is `rgb_array`.
*   `--disable_exploration`: Turn off the adaptive exploration system.

**Example - Resuming Training with HGT:**
```bash
python -m npp_rl.agents.training --load_model ./training_logs/enhanced_ppo_training/session-MM-DD-YYYY-HH-MM-SS/best_model/best_model.zip --num_envs 32 --extractor_type hgt
```

**Example - Using Hierarchical Extractor (Secondary):**
```bash
python -m npp_rl.agents.training --num_envs 64 --total_timesteps 10000000 --extractor_type hierarchical
```

The original training utilities in `npp_rl/agents/npp_agent_ppo.py` remain for compatibility; prefer `training.py` going forward.

### Hyperparameter Tuning

Automated hyperparameter optimization is available via Optuna. Scripts `ppo_tune.py` (for standard PPO features) and `recurrent_ppo_tune.py` (if a recurrent version is being tested) can be used. These scripts typically optimize:
*   Learning rate and schedule
*   Network architecture choices (within predefined options)
*   LSTM hidden size (for recurrent policies)
*   PPO-specific parameters (`batch_size`, `n_steps`, GAE, clip ranges, coefficients).

Tuning results are saved in `training_logs/tune_logs/` and `training_logs/tune_results_<timestamp>/`.

## Monitoring and Logging

*   **Tensorboard**: Training progress, including rewards, losses, and exploration metrics, can be monitored using Tensorboard:
    ```bash
    tensorboard --logdir ./training_logs/enhanced_ppo_training/
    ```
    (Adjust log directory path based on the session timestamp).
*   **Log Files**: Detailed logs, training configurations, and model checkpoints are saved under:
    `./training_logs/enhanced_ppo_training/session-<timestamp>/`
    This includes:
    *   `training_config.json`: Hyperparameters and settings for the run.
    *   `eval/`: Logs from evaluation callbacks.
    *   `tensorboard/`: Tensorboard event files.
    *   `best_model/`: The best model saved during training based on evaluation performance.
    *   `final_model/`: The model saved at the very end of training.

## Example Agent Performance

(This section can be updated with new GIFs or performance metrics as the agent develops further)

This is an example of a trained agent completing a non-trivial level:
![Example Level Completion](example_completion.gif)
*This agent was trained on a specific level configuration.*

Work on a generalized agent capable of playing a wide variety of N++ levels is an ongoing focus.

## Key Research References

The architecture and training procedures are informed by principles and findings from various research papers, including:

*   Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms.
*   Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised Prediction.
*   Cobbe, K., Hesse, C., Hilton, J., & Schulman, J. (2020). Leveraging Procedural Generation to Benchmark Reinforcement Learning. (Influenced choices for network scaling and temporal modeling).
*   Ji, S., Xu, W., Yang, M., & Yu, K. (2013). 3D convolutional neural networks for human action recognition. (Early work on 3D CNNs relevant to spatiotemporal feature learning).
*   DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout.
*   Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J. (2019). Go-Explore: a New Approach for Hard-Exploration Problems. (Inspired adaptive novelty components).
*   Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. (Foundation for CNNs in RL).
*   Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. (General insights into model scaling).
*   Ying, R., et al. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling. (DiffPool implementation for hierarchical GNNs).
*   Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. (GraphSAGE foundation for graph neural networks).

### Multi-Resolution Graph Architecture

The hierarchical graph system processes N++ levels at three resolution levels:

*   **Sub-cell Level (6px resolution)**: Fine-grained movement precision with ~15,456 nodes
*   **Tile Level (24px resolution)**: Standard game mechanics with ~966 nodes  
*   **Region Level (96px resolution)**: Strategic planning with ~60 nodes

### Key Components

*   **Hierarchical Graph Builder** (`nclone/graph/hierarchical_builder.py`):
    *   Creates multi-resolution representations through graph coarsening
    *   Maintains cross-scale connectivity for information flow
    *   Aggregates features from fine to coarse levels with statistical summaries

*   **DiffPool GNN** (`npp_rl/models/diffpool_gnn.py`):
    *   Implements differentiable graph pooling with soft cluster assignments
    *   Enables end-to-end training of hierarchical representations
    *   Includes auxiliary losses (link prediction, entropy, orthogonality) for stable training

*   **Multi-Scale Fusion** (`npp_rl/models/multi_scale_fusion.py`):
    *   Context-aware attention mechanisms that adapt to ninja physics state
    *   Learned routing between resolution levels
    *   Dynamic scale selection based on current task requirements

*   **Hierarchical Multimodal Extractor** (`npp_rl/feature_extractors/hierarchical_multimodal.py`):
    *   Integrates hierarchical graph processing with existing CNN/MLP architectures
    *   Supports auxiliary loss training for improved representations
    *   Graceful fallback when hierarchical graph data is unavailable

### Usage Example

```python
from npp_rl.feature_extractors import create_hgt_multimodal_extractor, create_hierarchical_multimodal_extractor

# PRIMARY: Create HGT-based feature extractor (RECOMMENDED)
hgt_extractor = create_hgt_multimodal_extractor(
    observation_space=env.observation_space,
    features_dim=512,
    hgt_hidden_dim=256,
    hgt_num_layers=3
)

# SECONDARY: Create hierarchical feature extractor
hierarchical_extractor = create_hierarchical_multimodal_extractor(
    observation_space=env.observation_space,
    features_dim=512,
    use_hierarchical_graph=True
)

# Use in PPO training with auxiliary losses
policy_kwargs = {
    'features_extractor_class': type(extractor),
    'features_extractor_kwargs': {
        'enable_auxiliary_losses': True,
        'hierarchical_hidden_dim': 128,
        'fusion_dim': 256
    }
}

model = PPO(
    policy="MultiInputPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    # ... other parameters
)
```

For detailed implementation information, see `TASK_2_1_IMPLEMENTATION_SUMMARY.md`.

## Dependencies and Installation

### System Requirements

- Python 3.8 or higher
- System dependencies for PyCairo (a dependency of `nclone`):
  ```sh
  sudo apt install libcairo2-dev pkg-config python3-dev
  ```

### Installing Dependencies

1. First, ensure the `nclone` environment is installed from a local sibling directory:
   ```bash
   # Navigate to the directory containing npp-rl
   cd /path/to/parent/directory
   git clone https://github.com/tetramputechture/nclone.git
   cd nclone
   pip install -e .
   cd ../npp-rl
   ```

2. Install Python dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

The `requirements.txt` file includes all necessary Python packages:
- `numpy`: For numerical computations
- `torch`: Deep learning framework
- `opencv-python`: Image processing
- `pillow`: Image handling
- `gymnasium`: Environment interface
- `stable-baselines3` and `sb3-contrib`: RL algorithms
- `optuna`: Hyperparameter tuning
- `tensorboard`: Training visualization
- `imageio`: Video recording
- `albumentations`: Image augmentations
- `pytest`: Testing framework

### Coding Standards

Standards are documented in the `.cursor/rules` directory. When you are writing code, you should follow these standards.

### Linting

Linting is done using ruff. You can run the following command to lint the code:
```bash
make lint
```

You can also run the following command to fix linting issues:
```bash
make fix
```

You can also run the following command to remove unused imports:
```bash
make imports
```
