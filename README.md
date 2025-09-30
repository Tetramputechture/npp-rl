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
    *   Augmentation: Game-optimized augmentation pipeline including random translation, horizontal flipping, cutout, and brightness/contrast variations for improved generalization (implemented in `nclone/gym_environment/frame_augmentation.py`).

*   **Global View**:
    *   A downsampled 176x100 pixel grayscale view of the entire level.

*   **Game State Vector**: A low-dimensional vector containing:
    *   Ninja physics state (position, velocity, airborne/walled status, jump duration, applied forces).
    *   Status of critical entities (e.g., exit door, switches).
    *   Normalized vectors from the ninja to key objectives (e.g., active switch, exit).
    *   Time remaining in the episode.

### 2. Feature Extraction

The agent uses an **HGTMultimodalExtractor** that combines multiple neural architectures for comprehensive multimodal processing:

*   **Temporal Processing (3D CNN)**:
    *   Processes 12-frame temporal stacks using 3D convolutional networks
    *   Captures movement patterns and temporal dynamics essential for NPP gameplay
    *   Uses batch normalization and dropout for stable training
    *   Adaptive pooling ensures consistent output dimensions

*   **Spatial Processing (2D CNN with Attention)**:
    *   Processes global level view (176x100 pixels) through 2D CNN
    *   Integrates spatial attention mechanisms for enhanced spatial reasoning
    *   Multi-scale feature extraction captures level structure at different resolutions

*   **Graph Processing (Heterogeneous Graph Transformer)**:
    *   Full HGT implementation with type-specific attention mechanisms
    *   **Type-aware processing**: Specialized handling for different node types (tiles, entities, hazards, switches)
    *   **Edge-type specialization**: Distinct processing for movement edges (adjacent, reachable, functional)
    *   **Multi-head attention**: Attention mechanisms adapted for heterogeneous graph structures
    *   **Entity-aware embeddings**: Specialized processing with hazard-aware attention

*   **Cross-Modal Fusion**:
    *   Attention mechanisms for multimodal integration
    *   Layer normalization and residual connections for stable training
    *   Cross-modal attention between temporal, spatial, graph, and state features
    *   Designed for robust performance across diverse level configurations

*   **Architecture Design**:
    *   Modular design with clear separation of concerns
    *   Comprehensive error handling and fallback mechanisms
    *   Optimized for both accuracy and computational efficiency
    *   Extensive logging and debugging capabilities

The extractor combines all modalities through fusion mechanisms, producing feature representations that enable generalization across different NPP level designs and difficulty configurations.

#### Usage

```python
from npp_rl.feature_extractors import HGTMultimodalExtractor

extractor = HGTMultimodalExtractor(
    observation_space=env.observation_space,
    features_dim=512,
    debug=False  # Set to True for detailed logging
)
```

## System Architecture

### Environment Integration

The NPP-RL system integrates with the nclone N++ simulator through a layered architecture:

#### Core Components

**Base Environment (`nclone`)**:
- Provides physics simulation and basic observations
- Generates player-centric frames (84x84x12), global view (176x100x1), game state (16 features), and reachability features (64 features)
- Handles N++ physics constants and movement mechanics

**Environment Wrappers (`npp_rl/environments/`)**:
- **DynamicGraphWrapper**: Adds graph observations from nclone's hierarchical graph builder
- **ReachabilityWrapper**: Integrates nclone's tiered reachability analysis system  
- **VectorizationWrapper**: Enables parallel environment processing

#### Graph Processing Pipeline

The system uses nclone's graph construction capabilities without over-engineering:

1. **Graph Construction**: nclone's HierarchicalGraphBuilder creates multi-resolution spatial graphs
2. **Dynamic Updates**: Simple state-based updates when switch/door states change
3. **Graph Observations**: Fixed-size arrays compatible with Gym (N_MAX_NODES=18000, E_MAX_EDGES=144000)
4. **HGT Processing**: Type-specific attention for heterogeneous node/edge types

#### Model Architecture (`npp_rl/models/`)

**Core Models**:
- **hgt_gnn.py**: Complete Heterogeneous Graph Transformer implementation
- **entity_type_system.py**: Entity-specialized embeddings and hazard-aware attention
- **conditional_edges.py**: Conditional edge processing for dynamic graphs
- **spatial_attention.py**: Graph-guided spatial attention mechanisms
- **physics_state_extractor.py**: Physics state extraction with momentum features

**Intrinsic Curiosity Components (`npp_rl/intrinsic/`)**:
- **icm.py**: Reachability-aware ICM with forward/inverse prediction models
- **reachability_exploration.py**: Integration with nclone reachability systems for spatial modulation
- **IntrinsicRewardWrapper**: Environment wrapper for seamless ICM integration with PPO

**Configuration Management**:
- **hgt_config.py**: Centralized configuration with sensible defaults
- Modular design enables easy hyperparameter tuning

### Data Flow

```
nclone Environment
    ↓
Environment Wrappers (Graph, Reachability, Vectorization)
    ↓
Multimodal Observations:
├── Temporal: player_frames [84,84,12]
├── Spatial: global_view [176,100,1] 
├── Vector: game_state [16], reachability_features [8]
└── Graph: node_feats, edge_feats, connectivity, masks, types
    ↓
HGTMultimodalExtractor:
├── 3D CNN → Temporal Features [512]
├── 2D CNN + Attention → Spatial Features [256]
├── Full HGT → Graph Features [256] 
├── MLPs → State [128] + Reachability [128]
└── Cross-Modal Fusion → Combined Features [512]
    ↓
    ├── PPO Policy/Value Networks → Action Selection
    └── ICM Network (Intrinsic Curiosity):
        ├── Forward Model: (state_t, action) → predicted_state_{t+1}
        ├── Inverse Model: (state_t, state_{t+1}) → predicted_action
        └── Reachability Modulation → Scaled Intrinsic Rewards
    ↓
Reward Combination: Extrinsic + Intrinsic Rewards → Total Reward
    ↓
PPO Training Updates (Policy, Value, ICM parameters)
```

### Performance Characteristics

- **Graph Updates**: Sub-millisecond performance suitable for real-time RL training
- **Feature Extraction**: ~5.6ms average inference time (target: <10ms)
- **Memory Usage**: Efficient with fixed-size arrays for Gym compatibility
- **Batch Processing**: Supports vectorized environments for parallel training

### Integration Points

**nclone Integration**:
- Clean abstraction layer using nclone for physics simulation and basic graph construction
- Avoids over-engineering by letting HGT learn complex patterns through attention mechanisms
- Uses simple reachability metrics rather than expensive physics calculations

**Graph Processing**:
- Seamless flow from nclone → DynamicGraphWrapper → HGT processing
- Type-specific attention for different node types (tiles, entities, hazards, switches)
- Edge-type specialization for movement relationships (adjacent, reachable, functional)

**Multimodal Fusion**:
- Cross-modal attention enables optimal integration of temporal, spatial, graph, and state information
- Layer normalization and residual connections ensure stable training
- Designed for generalizability across diverse NPP level configurations

**ICM Integration with PPO**:
- **Feature Sharing**: ICM uses the same 512-dimensional features from HGTMultimodalExtractor
- **Reward Combination**: IntrinsicRewardWrapper seamlessly combines extrinsic and intrinsic rewards
- **Reachability Modulation**: ICM curiosity is spatially modulated using nclone's reachability analysis
- **Training Synchronization**: ICM parameters update alongside PPO policy/value networks
- **Performance Optimization**: <0.5ms ICM computation maintains real-time training requirements

### Network Architecture & Hyperparameters

#### Technical Specifications

*   **3D CNN Architecture**:
    *   Layer 1: Conv3D(1→32, kernel=(4,7,7), stride=(2,2,2)) + BatchNorm + ReLU + Dropout
    *   Layer 2: Conv3D(32→64, kernel=(3,5,5), stride=(1,2,2)) + BatchNorm + ReLU + Dropout  
    *   Layer 3: Conv3D(64→128, kernel=(2,3,3), stride=(1,2,2)) + BatchNorm + ReLU
    *   Adaptive pooling to (1,4,4) + Feature projection to 512D

*   **2D CNN Architecture**:
    *   Layer 1: Conv2D(1→32, kernel=7, stride=2) + BatchNorm + ReLU + Dropout
    *   Layer 2: Conv2D(32→64, kernel=5, stride=2) + BatchNorm + ReLU + Dropout
    *   Layer 3: Conv2D(64→128, kernel=3, stride=2) + BatchNorm + ReLU
    *   Spatial attention integration + Adaptive pooling + Feature projection to 256D

*   **HGT Configuration**:
    *   Node feature dimension: 8 (optimized for efficiency)
    *   Edge feature dimension: 4 (adjacent, reachable, functional)
    *   Hidden dimension: 128, Layers: 3, Attention heads: 8
    *   Node types: 6 (tile, ninja, hazard, collectible, switch, exit)
    *   Edge types: 3 (adjacent, reachable, functional)

*   **Cross-Modal Fusion**:
    *   Input dimensions: Temporal(512) + Spatial(256) + Graph(256) + State(128) + Reachability(128)
    *   Fusion network with layer normalization and residual connections
    *   Progressive dimension reduction with attention mechanisms

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

The system includes an `AdaptiveExplorationManager` (from `npp_rl/agents/adaptive_exploration.py`) that provides hierarchical exploration capabilities:

*   **Hierarchical Subgoal Generation**:
    *   Uses nclone's reachability analysis to generate strategic subgoals
    *   Integrates with `EntityInteractionSubgoal` for unified subgoal representation
    *   Provides level completion planning through `LevelCompletionPlanner`
*   **Reachability-Guided Exploration**:
    *   Leverages nclone's `ReachabilitySystem` for performance-optimized planning
    *   Uses compact reachability features for subgoal filtering
    *   Maintains compatibility with NPP physics constants and level objectives
*   **Strategic Planning**:
    *   Dynamic subgoal updates adapt to changing game state
    *   Performance-optimized caching for real-time subgoal management (<3ms target)
    *   Integrates with existing NPP physics and level completion heuristics

## Training and Usage

### Basic Training

The system provides multiple training entry points:

```bash
# Basic PPO training
python ppo_train.py --num-envs 64 --render-mode rgb_array

# Advanced training with hierarchical exploration
python -m npp_rl.agents.training --num_envs 64 --total_timesteps 10000000 --extractor_type hgt
```

### Hierarchical Exploration Integration

The system integrates hierarchical exploration through the `AdaptiveExplorationManager` class:

```python
from npp_rl.agents.adaptive_exploration import AdaptiveExplorationManager

# Initialize the exploration manager
exploration_manager = AdaptiveExplorationManager()

# Initialize ICM with appropriate dimensions
exploration_manager.initialize_curiosity_module(
    feature_dim=512,  # Match HGTMultimodalExtractor output
    action_dim=6      # N++ action space size
)

# Get exploration bonus during training
intrinsic_bonus = exploration_manager.get_exploration_bonus(
    state=previous_features,
    action=action,
    next_state=current_features
)
```

### nclone Planning System Integration

The exploration system integrates with nclone's planning components:

- **EntityInteractionSubgoal**: Unified subgoal representation for navigation and switch activation
- **LevelCompletionPlanner**: Strategic level completion planning using reachability analysis
- **SubgoalPrioritizer**: Intelligent subgoal ranking based on strategic value and feasibility
- **ReachabilitySystem**: Performance-optimized spatial analysis for feasible paths

### Configuration

Key parameters for hierarchical exploration:

```python
# ICM Configuration
FEATURE_DIM = 512           # Match feature extractor output
CURIOSITY_SCALE = 0.1       # Scale factor for intrinsic rewards
NOVELTY_SCALE = 0.05        # Scale factor for novelty bonuses

# Subgoal Configuration  
MAX_SUBGOALS_PER_STEP = 5   # Limit active subgoals
MIN_REACHABILITY_SCORE = 0.3 # Filter unreachable subgoals
SUBGOAL_UPDATE_FREQ = 100   # Update frequency (steps)
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

### 7. ICM Configuration and Usage

The **Reachability-Aware Intrinsic Curiosity Module** is integrated into the main architecture pipeline (see System Architecture → Data Flow above). This section covers configuration and usage details.

#### Basic Usage

```python
from npp_rl.agents.training import train_agent

# ICM is automatically integrated when using adaptive exploration
config = {
    "enable_adaptive_exploration": True,
    "icm_config": {
        "feature_dim": 512,        # Match HGTMultimodalExtractor output
        "enable_reachability_awareness": True,
        "alpha": 0.1,              # Intrinsic reward weight
        "eta": 0.01,               # ICM learning rate
        "lambda_inv": 0.1,         # Inverse model loss weight
        "lambda_fwd": 0.9,         # Forward model loss weight
    }
}

# Train with ICM-enhanced exploration
model = train_agent(config)
```

#### Manual Integration

For custom environments or advanced usage:

```python
from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper

# Create reachability-aware ICM
icm = ICMNetwork(
    feature_dim=512,           # Match HGT feature extractor output
    action_dim=6,              # N++ action space size
    enable_reachability_awareness=True,
    reachability_dim=8,        # nclone reachability features
)

# Wrap environment with intrinsic rewards
env = IntrinsicRewardWrapper(
    env=base_env,
    icm_trainer=ICMTrainer(icm),
    alpha=0.1,                 # Intrinsic reward weight
    r_int_clip=1.0,           # Maximum intrinsic reward
)
```

#### Training Phase Configuration

```python
# Early training: Conservative exploration
early_config = {
    "alpha": 0.05,             # Lower intrinsic weight
    "eta": 0.005,              # Slower ICM learning
    "reachability_scale_factor": 1.5,
}

# Mid training: Balanced exploration
mid_config = {
    "alpha": 0.1,              # Standard intrinsic weight
    "eta": 0.01,               # Standard ICM learning
    "frontier_boost_factor": 3.0,
}

# Late training: Focused exploration
late_config = {
    "alpha": 0.15,             # Higher intrinsic weight
    "strategic_weight_factor": 2.0,  # More goal-directed
    "unreachable_penalty": 0.05,     # Stronger penalty
}
```

#### Key Configuration Parameters

- **alpha**: Intrinsic reward weight (0.05-0.15 typical range)
- **eta**: ICM learning rate (0.005-0.01 typical range)  
- **lambda_inv/lambda_fwd**: Loss balancing for inverse/forward models
- **reachability_scale_factor**: Modulation strength for spatial accessibility
- **frontier_boost_factor**: Temporary boost for newly accessible areas
- **strategic_weight_factor**: Goal-directed exploration weighting

For detailed implementation and usage examples, see [`npp_rl/intrinsic/README.md`](npp_rl/intrinsic/README.md).

## Project Structure

Consolidated architecture focused on hierarchical multimodal processing:

- `npp_rl/`
  - `agents/`
    - `training.py`: **Primary training entrypoint** with hierarchical multimodal architecture, CLI interface, PPO, vectorized environments, and comprehensive logging.
    - `adaptive_exploration.py`: Optional curiosity/novelty exploration manager and helpers.
    - `hyperparameters/ppo_hyperparameters.py`: Tuned PPO defaults and `NET_ARCH_SIZE`.
  - `feature_extractors/`
    - `hgt_multimodal.py`: **Primary feature extractor** with HGT-based multimodal processing and graph neural networks.
    - `__init__.py`: Unified interface with factory functions for hierarchical extractor.
  - `intrinsic/`
    - `icm.py`: **Reachability-aware ICM implementation** with forward/inverse models and spatial modulation.
    - `reachability_exploration.py`: Integration with nclone reachability systems for enhanced exploration.
    - `utils.py`: Utility functions for feature extraction, reward combination, and ICM configuration.
    - `README.md`: Comprehensive documentation for ICM usage and integration.
  - `wrappers/`
    - `intrinsic_reward_wrapper.py`: Environment wrapper for seamless ICM integration with PPO training.
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

The architecture and training procedures are informed by principles and findings from various research papers:

*   **Reinforcement Learning**: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
*   **3D CNNs**: Ji, S., et al. (2013). 3D convolutional neural networks for human action recognition.
*   **Graph Neural Networks**: Hamilton, W., et al. (2017). Inductive Representation Learning on Large Graphs.
*   **Hierarchical RL**: Nachum, O., et al. (2018). Data-Efficient Hierarchical Reinforcement Learning with Goal-Conditioned Policies.
*   **Exploration**: Pathak, D., et al. (2017). Curiosity-driven Exploration by Self-supervised Prediction.
*   **Attention Mechanisms**: Vaswani, A., et al. (2017). Attention Is All You Need.
*   **Graph Transformers**: Dwivedi, V. P., & Bresson, X. (2020). A Generalization of Transformer Networks to Graphs.

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

*   **Multi-Scale Fusion** (`npp_rl/models/multi_scale_fusion.py`):
    *   Context-aware attention mechanisms that adapt to ninja physics state
    *   Learned routing between resolution levels
    *   Dynamic scale selection based on current task requirements

*   **HGT Multimodal Extractor** (`npp_rl/feature_extractors/hgt_multimodal.py`):
    *   Integrates HGT-based graph processing with CNN/MLP architectures
    *   Supports multimodal fusion of visual, graph, and state features
    *   Optimized for real-time RL training with efficient processing

### Usage Example

```python
from npp_rl.feature_extractors import create_hgt_multimodal_extractor

# Create HGT-based feature extractor (PRIMARY)
hgt_extractor = create_hgt_multimodal_extractor(
    observation_space=env.observation_space,
    features_dim=512,
    hgt_hidden_dim=256,
    hgt_num_layers=3
)

# Use in PPO training
policy_kwargs = {
    'features_extractor_class': type(hgt_extractor),
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
