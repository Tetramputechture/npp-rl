# NPP-RL Repository Guide

## Project Purpose

NPP-RL is a Deep Reinforcement Learning project that trains an AI agent to play the physics-based platformer game N++. The agent uses PPO (Proximal Policy Optimization) with advanced features including 3D convolutions for temporal modeling, graph neural networks for structural understanding, intrinsic curiosity for exploration, and behavioral cloning from human demonstrations.

## Core Architecture

The project implements a sophisticated multi-modal RL agent with:

- **3D Feature Extraction**: 12-frame temporal stacking with 3D convolutions for spatiotemporal learning
- **Multi-Input Observations**: Player-centric visual frames (84*84*12), global view (176*100), and physics state vectors
- **Advanced Exploration**: Intrinsic Curiosity Module (ICM) and novelty detection for sparse reward environments  
- **Graph Neural Networks**: Structural level understanding through graph representations of game levels
- **Behavioral Cloning**: Pretraining on human replay data for faster learning

## Repository Structure

```
npp-rl/
├── npp_rl/                    # Main package
│   ├── agents/                # Training scripts and agent implementations
│   │   ├── training.py     # PRIMARY training script (recommended)
│   │   ├── enhanced_feature_extractor.py  # 3D/2D CNN feature extractors
│   │   ├── adaptive_exploration.py        # ICM and exploration management
│   │   ├── hyperparameters/              # PPO hyperparameters
│   │   └── npp_agent_ppo.py             # Legacy training utilities
│   ├── models/                # Neural network models and physics calculations
│   │   ├── gnn.py                  # Graph Neural Network encoder
│   │   ├── feature_extractors.py  # Multimodal feature extraction
│   │   ├── movement_classifier.py # Physics-based movement classification
│   │   └── trajectory_calculator.py # Physics trajectory calculations
│   ├── intrinsic/             # Intrinsic motivation and curiosity
│   ├── data/                  # Data loading (behavioral cloning datasets)
│   ├── eval/                  # Evaluation metrics and exploration analysis
│   ├── environments/          # Environment wrappers and configurations
│   ├── wrappers/              # Custom environment wrappers
│   └── config/                # Configuration management
├── bc_pretrain.py             # Behavioral cloning pretraining script
├── train_phase2.py           # Enhanced training with all Phase 2 features
├── ppo_train.py              # Simple training wrapper (legacy)
├── tests/                    # Comprehensive test suite
├── docs/                     # Technical documentation and implementation plans
├── tools/                    # Utility scripts
└── archive/                  # Deprecated/experimental code (Phase 1 archive)
```

## Key Dependencies and Setup

### System Requirements
- Python 3.8+
- CUDA-capable GPU recommended for training
- System dependencies: `libcairo2-dev pkg-config python3-dev`

### Critical Dependency: nclone
The project REQUIRES the `nclone` N++ simulator to be installed as a sibling directory:

```bash
# From parent directory containing npp-rl/
git clone https://github.com/tetramputechture/nclone.git
cd nclone
pip install -e .
cd ../npp-rl
pip install -r requirements.txt
```

### Python Dependencies
- `torch>=2.0.0` - Deep learning framework
- `stable-baselines3>=2.1.0` - RL algorithms
- `gymnasium>=0.29.0` - Environment interface
- `optuna>=3.3.0` - Hyperparameter tuning
- `opencv-python>=4.8.0` - Image processing
- `tensorboard>=2.14.0` - Training visualization

## Training Quick Start

### Recommended Training Command
```bash
# Primary training with all enhancements (recommended)
python -m npp_rl.agents.training --num_envs 64 --total_timesteps 10000000

# Phase 2 training with ICM and graph observations
python train_phase2.py --preset full_phase2 --experiment_name full_experiment

# Behavioral cloning pretraining
python bc_pretrain.py --dataset_dir datasets/shards --epochs 20
```

### Training Configurations
- **num_envs**: 64 parallel environments (adjust based on CPU cores)
- **total_timesteps**: 10M for substantial training (adjust based on time/resources)
- **enable_exploration**: ICM and novelty detection enabled

## Development Guidelines

### Code Standards
- **File size limit**: NEVER exceed 500 lines per file
- **Physics constants**: ALWAYS import from `nclone.constants`, never redefine
- **Testing focus**: Test behavior, not static values or constants
- **Documentation**: Include research paper references for algorithmic choices

### Physics Integration
```python
# ALWAYS use nclone constants
from nclone.constants import (
    NINJA_RADIUS, GRAVITY_FALL, GRAVITY_JUMP,
    MAX_HOR_SPEED, JUMP_FLAT_GROUND_Y
)
```

### Import Organization
1. Standard library imports
2. Third-party library imports (torch, numpy, sb3)
3. nclone imports
4. npp_rl imports

## Linting and Quality

### Makefile Commands
```bash
make lint        # Check code quality with ruff
make fix         # Auto-fix linting issues
make imports     # Remove unused imports
make dev-setup   # Install development tools
```

### Testing
```bash
pytest tests/                    # Run all tests
pytest tests/test_phase2_basic.py   # Run specific tests
pytest -v tests/                # Verbose output
```

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir ./training_logs/enhanced_ppo_training/
```

### Log Structure
```
training_logs/enhanced_ppo_training/session-YYYY-MM-DD-HH-MM-SS/
├── training_config.json    # Hyperparameters and settings
├── eval/                   # Evaluation logs
├── tensorboard/            # TensorBoard event files
├── best_model/             # Best performing model
└── final_model/            # Final training checkpoint
```

## Key Performance Features

### Architectural Improvements
- **3D Convolutions**: 37.9% reduction in optimality gap vs 2D
- **Frame Stacking**: 12 temporal frames for motion understanding
- **Scaled Networks**: [256, 256, 128] hidden layers for complex observations
- **Adaptive Learning Rate**: Linear decay from 3e-4 to 1e-6

### Exploration Enhancements
- **ICM (Intrinsic Curiosity)**: Exploration bonus from prediction error
- **Novelty Detection**: Count-based state visitation rewards
- **Adaptive Scaling**: Dynamic exploration weight adjustment

### Training Optimizations
- **Vectorized Environments**: 64 parallel environments for sample efficiency
- **H100/GPU Optimization**: TF32 and memory management for modern GPUs
- **Hyperparameter Tuning**: Optuna-optimized parameters

## Research Foundation

The implementation is informed by key research papers:
- PPO: Schulman et al. (2017)
- ICM: Pathak et al. (2017) "Curiosity-driven Exploration"
- 3D CNNs: Ji et al. (2013) for spatiotemporal features
- Scaling Laws: Kaplan et al. (2020) for network architecture
- ProcGen: Cobbe et al. (2020) for procedural environments

## Phase Implementation Status

**Phase 1 (Complete)**: Basic PPO with enhanced feature extraction and exploration
**Phase 2 (Complete)**: ICM, Graph Neural Networks, Behavioral Cloning, advanced metrics

Each phase builds incrementally on robust foundations with comprehensive testing and clear architectural boundaries.
