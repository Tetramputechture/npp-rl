# NPP-RL Repository Guide

## Project Purpose

NPP-RL is a Deep Reinforcement Learning system that trains agents to play N++, a physics-based platformer game. The system supports multiple neural network architectures (HGT, GAT, GCN, Vision-Free, MLP) with PPO-based training, curriculum learning, hierarchical RL, and comprehensive evaluation frameworks.

## Core Features

The project implements an advanced RL training system with:

- **Multi-Architecture Support**: HGT (Heterogeneous Graph Transformer), GAT, GCN, Vision-Free, and MLP architectures
- **Hierarchical RL**: High-level task planning with low-level execution policies
- **Curriculum Learning**: Progressive difficulty scaling for efficient training
- **Graph Neural Networks**: Entity-aware level understanding through heterogeneous graphs
- **Intrinsic Curiosity**: ICM, mine-aware curiosity, and reachability-based exploration
- **Multi-GPU Training**: Distributed training with mixed precision support
- **Behavioral Cloning**: Pretraining pipeline from human replay data
- **Comprehensive Testing**: Standardized test suite with diverse level types

## Repository Structure

```
npp-rl/
├── npp_rl/                    # Main package
│   ├── agents/                # Core agent implementations
│   │   ├── training.py                    # Agent training orchestration
│   │   ├── adaptive_exploration.py        # Adaptive exploration strategies
│   │   ├── hierarchical_ppo.py           # Hierarchical PPO implementation
│   │   └── hyperparameters/              # Hyperparameter configurations
│   │       ├── ppo_hyperparameters.py
│   │       └── hierarchical_hyperparameters.py
│   ├── models/                # Neural network architectures
│   │   ├── hgt_encoder.py, hgt_layer.py, hgt_factory.py  # HGT implementation
│   │   ├── gat.py, gcn.py                                # Graph architectures
│   │   ├── simplified_hgt.py                             # Lightweight HGT
│   │   ├── hierarchical_policy.py                        # Hierarchical policies
│   │   ├── attention_mechanisms.py, spatial_attention.py # Attention layers
│   │   ├── entity_type_system.py                         # Entity typing
│   │   └── conditional_edges.py                          # Dynamic graph edges
│   ├── hrl/                   # Hierarchical RL components
│   │   ├── high_level_policy.py          # Task-level planning
│   │   ├── subtask_policies.py           # Low-level execution
│   │   ├── subtask_rewards.py            # Reward shaping
│   │   ├── completion_controller.py      # Task completion logic
│   │   ├── progress_trackers.py          # Progress monitoring
│   │   └── mine_aware_context.py         # Mine interaction context
│   ├── training/              # Training infrastructure
│   │   ├── architecture_trainer.py       # Multi-architecture training
│   │   ├── architecture_configs.py       # Architecture definitions
│   │   ├── pretraining_pipeline.py       # Behavioral cloning pipeline
│   │   ├── curriculum_manager.py         # Curriculum learning
│   │   ├── distributed_utils.py          # Multi-GPU utilities
│   │   └── training_utils.py             # Training helpers
│   ├── intrinsic/             # Intrinsic motivation modules
│   │   ├── icm.py                        # Intrinsic Curiosity Module
│   │   ├── mine_aware_curiosity.py       # Mine-specific exploration
│   │   └── reachability_exploration.py   # Reachability-based rewards
│   ├── feature_extractors/    # Feature extraction
│   │   └── configurable_extractor.py     # Multi-modal feature extractor
│   ├── wrappers/              # Environment wrappers
│   │   ├── intrinsic_reward_wrapper.py   # ICM integration
│   │   ├── hierarchical_reward_wrapper.py # HRL reward shaping
│   │   └── curriculum_env.py             # Curriculum wrapper
│   ├── callbacks/             # Training callbacks
│   │   ├── hierarchical_callbacks.py     # HRL-specific callbacks
│   │   └── pbrs_logging_callback.py      # PBRS metrics logging
│   ├── evaluation/            # Evaluation system
│   │   ├── comprehensive_evaluator.py    # Full evaluation pipeline
│   │   └── test_suite_loader.py          # Test suite management
│   ├── eval/                  # Evaluation metrics
│   │   └── exploration_metrics.py        # Exploration analysis
│   ├── optimization/          # Performance optimization
│   │   ├── h100_optimization.py          # GPU-specific tuning
│   │   ├── benchmarking.py               # Performance benchmarks
│   │   └── amp_exploration.py            # Mixed precision tools
│   └── utils/                 # Utility modules
│       ├── logging_utils.py              # Structured logging
│       ├── performance_monitor.py        # Performance tracking
│       └── s3_uploader.py                # Cloud artifact management
├── scripts/                   # Training and utility scripts
│   ├── train_and_compare.py              # PRIMARY training script
│   ├── list_architectures.py             # List available architectures
│   ├── example_single_arch.sh            # Single architecture example
│   ├── example_multi_arch.sh             # Multi-architecture comparison
│   └── example_with_s3.sh                # Production training with S3
├── tools/                     # Development utilities
│   ├── compare_architectures.py          # Architecture analysis
│   ├── data_quality.py                   # Dataset validation
│   ├── replay_ingest.py                  # Replay processing
│   └── rotate_videos.py                  # Video preprocessing
├── tests/                     # Comprehensive test suite
├── docs/                      # Technical documentation
│   ├── QUICK_START_TRAINING.md           # Getting started guide
│   ├── TRAINING_SYSTEM.md                # Full training system docs
│   ├── ARCHITECTURE_COMPARISON_GUIDE.md  # Architecture selection
│   ├── CURRICULUM_LEARNING.md            # Curriculum learning guide
│   ├── ICM_INTEGRATION_GUIDE.md          # Intrinsic motivation guide
│   ├── OBSERVATION_SPACE_GUIDE.md        # Observation space details
│   └── TEST_SUITE.md                     # Test suite documentation
├── experiments/               # Training output directory
├── datasets/                  # Training/evaluation datasets
├── Dockerfile                 # Container definition
├── Makefile                   # Development commands
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
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

### List Available Architectures
```bash
# See all available architectures with descriptions
python scripts/list_architectures.py

# Available architectures:
# - full_hgt: Full HGT with all modalities (temporal frames, global view, graph, game state, reachability)
# - vision_free: Vision-free architecture (graph, game state, reachability only)
# - gat: Graph Attention Network
# - gcn: Graph Convolutional Network
# - mlp: Simple MLP baseline
```

### Quick Validation (5-10 minutes)
```bash
# Quick test on single architecture
python scripts/train_and_compare.py \
    --experiment-name "quick_test" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 16 \
    --output-dir experiments/
```

### Local Single-GPU Training
```bash
# Standard training run (~30-60 min on RTX 3080)
python scripts/train_and_compare.py \
    --experiment-name "local_training" \
    --architectures vision_free \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --output-dir experiments/

# With curriculum learning
python scripts/train_and_compare.py \
    --experiment-name "curriculum" \
    --architectures full_hgt \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --output-dir experiments/
```

### Multi-Architecture Comparison
```bash
# Compare multiple architectures (for ablation studies)
python scripts/train_and_compare.py \
    --experiment-name "arch_comparison" \
    --architectures full_hgt vision_free gat gcn \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000000 \
    --num-envs 64 \
    --output-dir experiments/
```

### Production Multi-GPU Training
```bash
# Full-scale training with distributed multi-GPU
python scripts/train_and_compare.py \
    --experiment-name "production_run" \
    --architectures full_hgt \
    --use-curriculum \
    --use-hierarchical-ppo \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 50000000 \
    --num-envs 128 \
    --mixed-precision \
    --s3-bucket npp-rl-artifacts \
    --s3-prefix experiments/production \
    --output-dir experiments/

# Monitor with TensorBoard
tensorboard --logdir experiments/
```

### Training Parameters
- **--architectures**: Space-separated list (full_hgt, vision_free, gat, gcn, mlp)
- **--num-envs**: Parallel environments (16 for local, 64-128 for multi-GPU)
- **--total-timesteps**: Total training steps (100K-50M)
- **--use-curriculum**: Enable progressive difficulty scaling
- **--use-hierarchical-ppo**: Enable hierarchical RL with subtask decomposition
- **--mixed-precision**: Enable for faster training on modern GPUs
- **--s3-bucket/--s3-prefix**: Automatic cloud artifact upload

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

## Monitoring and Logging

### TensorBoard
```bash
# Monitor all experiments
tensorboard --logdir experiments/

# Monitor specific experiment
tensorboard --logdir experiments/my_experiment_*/
```

### Output Structure
```
experiments/
└── {experiment_name}_{timestamp}/
    ├── config.json                 # Experiment configuration
    ├── {experiment_name}.log       # Training logs
    ├── {architecture}/
    │   ├── checkpoints/            # Model checkpoints (every 500K steps)
    │   │   ├── checkpoint_500000.zip
    │   │   ├── checkpoint_1000000.zip
    │   │   └── ...
    │   ├── tensorboard/            # TensorBoard event files
    │   │   └── events.out.tfevents.*
    │   ├── eval_results.json       # Evaluation metrics
    │   │   └── {test_map_results, level_type_breakdown, architecture_info}
    │   ├── final_model.zip         # Final trained model
    │   └── training_config.json    # Architecture-specific config
    ├── all_results.json            # Aggregated comparison results
    └── s3_manifest.json            # S3 upload manifest (if enabled)
```

### Key Metrics Tracked
- **Training**: episode_reward, episode_length, success_rate, fps
- **Exploration**: intrinsic_reward, novelty_bonus, state_visitation
- **Hierarchical RL**: subtask_completion, high_level_decisions, subtask_transitions
- **Evaluation**: test_suite_success_rate, per_level_type_performance
- **Performance**: GPU_utilization, memory_usage, samples_per_second

## Key System Features

### Multi-Architecture Support
- **HGT (Heterogeneous Graph Transformer)**: Entity-aware level understanding with typed nodes/edges
- **GAT/GCN**: Graph attention and convolutional alternatives
- **Vision-Free**: Graph and game state only (no visual frames)
- **MLP Baseline**: Simple baseline for comparison
- **Modular Design**: Easy architecture swapping via configuration

### Hierarchical Reinforcement Learning
- **High-Level Policy**: Task-level planning (reach_exit, toggle_mine, collect_gold)
- **Low-Level Policies**: Subtask-specific execution strategies
- **Mine-Aware Context**: Specialized handling for mine interactions
- **Progress Tracking**: Completion monitoring and reward shaping
- **Potential-Based Reward Shaping (PBRS)**: Policy-invariant reward augmentation

### Curriculum Learning
- **Progressive Difficulty**: Start simple, gradually increase complexity
- **Success-Based Progression**: Advance when agent achieves thresholds
- **Automated Management**: Built-in curriculum scheduler
- **Entity-Aware**: Curriculum considers entity types (mines, switches, etc.)

### Exploration & Intrinsic Motivation
- **ICM (Intrinsic Curiosity Module)**: Forward/inverse model prediction error
- **Mine-Aware Curiosity**: Entity-specific exploration bonuses
- **Reachability Exploration**: Reward based on reachable state space
- **Adaptive Scaling**: Dynamic intrinsic reward balancing

### Training Infrastructure
- **Multi-GPU Support**: Distributed training with DDP (DistributedDataParallel)
- **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training
- **S3 Integration**: Automatic artifact upload to cloud storage
- **Comprehensive Evaluation**: Standardized test suite with diverse level types
- **Architecture Comparison**: Train and compare multiple architectures simultaneously

## Research Foundation

The implementation is informed by key research papers:
- **PPO**: Schulman et al. (2017) - Proximal Policy Optimization
- **ICM**: Pathak et al. (2017) - Curiosity-driven Exploration by Self-supervised Prediction
- **HGT**: Hu et al. (2020) - Heterogeneous Graph Transformer
- **GAT**: Veličković et al. (2018) - Graph Attention Networks
- **Hierarchical RL**: Dayan & Hinton (1993), Kulkarni et al. (2016) - Feudal RL and hierarchical DQN
- **PBRS**: Ng et al. (1999) - Policy Invariance Under Reward Shaping
- **Curriculum Learning**: Bengio et al. (2009), Narvekar et al. (2020)

## Entity Support

The system supports the following N++ entities:
- **Navigation**: Exit doors, switches, locked doors, locked door switches
- **Hazards**: Mines (toggled and active states)
- **Collectibles**: Gold pieces
- **Future**: Thwumps, drones, turrets, zap drones (planned)

## Documentation

Comprehensive documentation available in `docs/`:
- **QUICK_START_TRAINING.md**: Getting started with training
- **TRAINING_SYSTEM.md**: Complete training system documentation
- **ARCHITECTURE_COMPARISON_GUIDE.md**: Architecture selection and comparison
- **CURRICULUM_LEARNING.md**: Curriculum learning setup and configuration
- **ICM_INTEGRATION_GUIDE.md**: Intrinsic motivation and exploration
- **OBSERVATION_SPACE_GUIDE.md**: Observation space details and design
- **TEST_SUITE.md**: Test suite structure and usage

## Development Workflow

### Common Tasks

**List available architectures:**
```bash
python scripts/list_architectures.py
```

**Quick validation run:**
```bash
./scripts/example_single_arch.sh
```

**Architecture comparison:**
```bash
./scripts/example_multi_arch.sh
```

**Production training with S3:**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
./scripts/example_with_s3.sh
```

### Testing and Quality

```bash
# Run tests
pytest tests/

# Check code quality
make lint

# Auto-fix linting issues
make fix

# Remove unused imports
make imports
```

### Data Generation

Generate test suite datasets:
```bash
cd ../nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output-dir datasets \
    --train-count 250 \
    --test-count 250
```
