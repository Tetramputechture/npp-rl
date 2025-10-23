# NPP-RL: Deep RL Agent for N++

Deep Reinforcement Learning system for training agents to play N++, a physics-based platformer game.

## System Overview

**Core Features:**
- PPO-based training with Stable Baselines3
- Multi-architecture support (HGT, GAT, GCN, Vision-Free, MLP)
- Curriculum learning for progressive difficulty
- Hierarchical RL with high/low-level policies
- Behavioral cloning from human replays
- Intrinsic Curiosity Module (ICM) for exploration
- Multi-GPU distributed training support
- Comprehensive test suite with standardized levels

**Entity Support (Production v1):**
- Exit doors and switches
- Mines (toggled and active)
- Locked doors and locked door switches

## Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU training)
- System packages: `libcairo2-dev pkg-config python3-dev build-essential`

### Setup

#### Standard x86_64 Installation
```bash
# Clone and install nclone simulator (required dependency)
git clone https://github.com/Tetramputechture/nclone.git
cd nclone && pip install -e . && cd ..

# Clone and install npp-rl
git clone https://github.com/Tetramputechture/npp-rl.git
cd npp-rl && pip install -e .
```

#### ARM64/aarch64 Installation (NVIDIA Grace Hopper, etc.)

**Important**: Standard pip installation may install CPU-only PyTorch on ARM64 systems. Use one of these methods:

**Option 1 - PyTorch Nightly (Recommended for ARM64 CUDA)**
```bash
# Install PyTorch with CUDA support FIRST
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Then install npp-rl
cd npp-rl && pip install -e .
```

**Option 2 - PyTorch 2.4.0 with CUDA 12.1**
```bash
# Install PyTorch with CUDA support FIRST
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Then install npp-rl
cd npp-rl && pip install -e .
```

**Troubleshooting ARM64 CUDA Issues:**
If PyTorch cannot access CUDA:
1. Check your PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. If version has `+cpu` suffix, reinstall using options above
3. Verify NVIDIA driver: `nvidia-smi`
4. Check environment: `echo $CUDA_VISIBLE_DEVICES` (should be empty or show GPU IDs)

The orchestration script (`scripts/orchestrate_architecture_comparison.sh`) will automatically detect and fix CPU-only PyTorch on ARM64 systems.

## Training

### Local Single-GPU Validation

Quick validation run for testing on limited hardware:

```bash
# Basic validation (5M timesteps, ~30-60 min on RTX 3080)
python scripts/train_and_compare.py \
    --experiment-name "local_validation" \
    --architectures vision_free \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --output-dir experiments/

# With curriculum learning
python scripts/train_and_compare.py \
    --experiment-name "local_curriculum" \
    --architectures vision_free \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 16 \
    --output-dir experiments/
```

### Multi-GPU Production Training

Full-scale training with distributed multi-GPU support:

```bash
# Single architecture, multi-GPU
python scripts/train_and_compare.py \
    --experiment-name "production_hgt" \
    --architectures full_hgt \
    --use-curriculum \
    --use-hierarchical-ppo \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 50000000 \
    --num-envs 128 \
    --output-dir experiments/ \
    --s3-bucket npp-rl-artifacts \
    --s3-prefix experiments/production

# Architecture comparison (for ablation studies)
python scripts/train_and_compare.py \
    --experiment-name "arch_comparison" \
    --architectures full_hgt vision_free gat gcn \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 20000000 \
    --num-envs 128 \
    --output-dir experiments/arch_compare/
```

**Multi-GPU Training:**
```bash
# Multi-GPU training automatically uses DistributedDataParallel
# Spawns one process per GPU for optimal performance

# 4-GPU training (e.g., on p3.8xlarge with 4x V100)
python scripts/train_and_compare.py \
    --experiment-name "multi_gpu_4x" \
    --architectures full_hgt \
    --num-gpus 4 \
    --num-envs 256 \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000000 \
    --output-dir experiments/

# Hardware profile (automatically configures multi-GPU settings)
python scripts/train_and_compare.py \
    --experiment-name "multi_gpu_auto" \
    --architectures full_hgt \
    --hardware-profile 8xV100-32GB \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000000

# With S3 uploads and curriculum learning
python scripts/train_and_compare.py \
    --experiment-name "distributed_prod" \
    --architectures full_hgt \
    --use-curriculum \
    --num-gpus 8 \
    --num-envs 512 \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000000 \
    --s3-bucket npp-rl-production \
    --s3-prefix runs/$(date +%Y-%m-%d)
```

**Multi-GPU Requirements:**
- PyTorch with CUDA support
- NCCL backend (automatically used for GPU communication)
- One CUDA-capable GPU per process
- Sufficient GPU memory (see hardware profiles for recommendations)

**Expected Performance Scaling:**
- 2 GPUs: ~1.8x speedup (90% efficiency)
- 4 GPUs: ~3.4x speedup (85% efficiency)
- 8 GPUs: ~6.0x speedup (75% efficiency)

### Behavioral Cloning Pretraining

Pretrain policies using human replay data before RL fine-tuning:

```bash
# Step 1: Pretrain with behavioral cloning
python npp_rl/training/bc_trainer.py \
    --replay-dir bc_replays/ \
    --architecture full_hgt \
    --epochs 50 \
    --batch-size 128 \
    --output-dir models/bc_pretrained/

# Step 2: Fine-tune with RL
python scripts/train_and_compare.py \
    --experiment-name "bc_finetuned" \
    --architectures full_hgt \
    --pretrained-model models/bc_pretrained/best_model.zip \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 20000000 \
    --num-envs 64 \
    --output-dir experiments/
```

### Frame Stacking

Frame stacking provides temporal information by concatenating consecutive observations. This technique, from DQN (Mnih et al., 2015), enables the policy to infer velocity and motion dynamics.

**Enable in training:**

```python
from nclone.gym_environment import EnvironmentConfig, FrameStackConfig

# Configure frame stacking
config = EnvironmentConfig.for_training()
config.frame_stack = FrameStackConfig(
    enable_visual_frame_stacking=True,
    visual_stack_size=4,  # 4 frames for visual observations
    enable_state_stacking=True,
    state_stack_size=4,  # 4 frames for game state
    padding_type="zero"
)

env = create_training_env(config)
```

**Feature extractors automatically adapt** to stacked observations:
- CNNs handle `(batch, stack_size, H, W, C)` inputs by treating stack_size as additional input channels
- MLPs flatten `(batch, stack_size, state_dim)` to `(batch, stack_size * state_dim)`
- Augmentation is applied consistently across all frames in the stack

**TensorBoard visualization:**
```python
from npp_rl.utils import log_stacked_observations, log_frame_stack_config

# Log configuration
log_frame_stack_config(writer, config.to_dict(), global_step)

# Log frame visualizations during training
log_stacked_observations(writer, observations, global_step)
```

**References:**
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529-533.
- Machado et al. (2018). "Revisiting the Arcade Learning Environment." IJCAI 61, 523-562.

### Architecture Comparison

Compare multiple architectures systematically:

```bash
# List available architectures
python scripts/list_architectures.py

# Run comparison
python scripts/train_and_compare.py \
    --experiment-name "arch_ablation_$(date +%Y%m%d)" \
    --architectures full_hgt simplified_hgt gat gcn vision_free mlp_baseline \
    --use-curriculum \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 20000000 \
    --num-envs 64 \
    --eval-freq 50000 \
    --output-dir experiments/ablation/
```

**Architectures:**
- `full_hgt`: Full Heterogeneous Graph Transformer (best performance)
- `simplified_hgt`: Reduced HGT (faster, good performance)
- `gat`: Graph Attention Network
- `gcn`: Graph Convolutional Network
- `vision_free`: Physics-state only (baseline, fastest)
- `mlp_baseline`: Simple MLP (minimal baseline)

Results saved to `experiments/ablation/comparison_TIMESTAMP.md` with:
- Training curves
- Final performance metrics
- Computational cost
- Hyperparameters used

## Monitoring

### TensorBoard

```bash
# View training progress
tensorboard --logdir experiments/

# View specific experiment
tensorboard --logdir experiments/production_hgt/
```

**Key metrics:**
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/success_rate`: Level completion rate
- `train/policy_loss`: Policy network loss
- `train/value_loss`: Value network loss
- `eval/mean_reward`: Evaluation performance

### Checkpoints

Models automatically saved to:
- `{output_dir}/{experiment_name}/{architecture}/best_model/`: Best performing checkpoint
- `{output_dir}/{experiment_name}/{architecture}/final_model/`: Final training state
- `{output_dir}/{experiment_name}/{architecture}/checkpoints/`: Periodic saves (every 1M steps)

## Evaluation

### Test Suite

Run standardized evaluation:

```bash
# Evaluate trained model
python npp_rl/evaluation/evaluate_agent.py \
    --model-path experiments/production_hgt/full_hgt/best_model/best_model.zip \
    --test-dataset ../nclone/datasets/test \
    --num-episodes 100 \
    --render \
    --output-dir eval_results/

# Compare multiple models
python tools/compare_architectures.py \
    --model-dir experiments/arch_ablation/ \
    --test-dataset ../nclone/datasets/test \
    --num-episodes 100 \
    --output-file comparison_results.csv
```

Test suite includes:
- Simple navigation (3 levels)
- Switch activation (3 levels)
- Mine avoidance (3 levels)
- Combined challenges (3 levels)

## Project Structure

```
npp-rl/
├── npp_rl/
│   ├── training/           # Training pipelines
│   │   ├── architecture_trainer.py
│   │   ├── bc_trainer.py
│   │   ├── curriculum_manager.py
│   │   └── pretraining_pipeline.py
│   ├── models/             # Neural network architectures
│   │   ├── hgt_gnn.py           # HGT implementation
│   │   ├── hierarchical_policy.py
│   │   └── feature_extractors.py
│   ├── hrl/                # Hierarchical RL components
│   ├── intrinsic/          # ICM and exploration
│   ├── evaluation/         # Evaluation tools
│   └── wrappers/           # Environment wrappers
├── scripts/
│   ├── train_and_compare.py    # Main training script
│   ├── list_architectures.py   # List available architectures
│   └── example_*.sh            # Example workflows
├── tests/                  # Comprehensive test suite
├── docs/                   # Detailed documentation
└── bc_replays/            # Human replay data (optional)
```

## Development

### Running Tests

```bash
# Run full test suite
make test

# Run specific test module
pytest tests/training/test_curriculum_manager.py -xvs

# Run with coverage
make test-coverage
```

### Code Quality

```bash
# Lint code
make lint

# Auto-fix issues
make fix

# Format code
make format
```

## Documentation

- **Quick Start**: `docs/QUICK_START_TRAINING.md`
- **Curriculum Learning**: `docs/CURRICULUM_LEARNING.md`
- **Architecture Comparison**: `docs/ARCHITECTURE_COMPARISON_GUIDE.md`
- **Complete System**: `docs/TRAINING_SYSTEM.md`
- **Test Suite**: `docs/TEST_SUITE.md`

## Hardware Requirements

### Minimum (Local Validation)
- GPU: GTX 1080 Ti / RTX 2060 (8GB VRAM)
- RAM: 16GB
- Training time: ~2-4 hours for 5M steps

### Recommended (Production)
- GPU: RTX 3090 / A6000 (24GB VRAM) or better
- RAM: 32GB
- Training time: ~8-12 hours for 50M steps

### Multi-GPU (Distributed)
- GPUs: 2-4x V100 / A100
- RAM: 64GB+
- Training time: ~4-8 hours for 100M steps with 4 GPUs

### ARM64 Systems (NVIDIA Grace Hopper)
- GPU: NVIDIA GH200 (96-480GB HBM3)
- Architecture: aarch64 (ARM Neoverse)
- Special considerations:
  - Requires PyTorch nightly or 2.4.0+ with CUDA 12.1+
  - Standard pip may install CPU-only PyTorch (see Installation section)
  - Orchestration script auto-detects and fixes CUDA issues
- Performance: Excellent for large-scale training (unified memory architecture)
