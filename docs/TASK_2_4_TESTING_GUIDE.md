# Task 2.4: End-to-End Testing and Benchmarking Guide

**Phase 2 Task 2.4: Training Stability and Optimization**

This comprehensive guide provides instructions for testing, training, and benchmarking the hierarchical PPO framework with stability optimizations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation and Setup](#installation-and-setup)
4. [Training Procedures](#training-procedures)
5. [Monitoring and Evaluation](#monitoring-and-evaluation)
6. [Benchmarking](#benchmarking)
7. [Troubleshooting](#troubleshooting)
8. [Performance Metrics](#performance-metrics)

---

## Quick Start

For users familiar with the system, here's the fastest way to get started:

```bash
# 1. Setup environment
cd /path/to/npp-rl
pip install -e ../nclone  # Install nclone simulator
pip install -r requirements.txt

# 2. Start training with default settings (recommended)
python train_hierarchical_stable.py

# 3. Monitor training in real-time
tensorboard --logdir training_logs/
```

Training will run with optimal defaults:
- 64 parallel environments
- 10M total timesteps (~8-12 hours on H100)
- Warmup phase: 100k steps
- Adaptive learning rate enabled
- Full stability monitoring

---

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS 11+
- **Python**: 3.8 or higher
- **CPU**: 16+ cores (for parallel environments)
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 50GB+ free space

### Recommended Requirements (H100 Optimization)

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **CPU**: 32+ cores (AMD EPYC or Intel Xeon)
- **RAM**: 128GB+
- **GPU**: NVIDIA H100 (80GB), A100 (40/80GB), or RTX 4090
- **Storage**: 200GB+ NVMe SSD
- **CUDA**: 11.8+ or 12.0+

### Software Dependencies

```
torch>=2.0.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
numpy>=1.21.0
opencv-python>=4.8.0
tensorboard>=2.14.0
nclone (sibling repository)
```

---

## Installation and Setup

### Step 1: Clone Repositories

```bash
# Create project directory
mkdir npp-rl-project
cd npp-rl-project

# Clone npp-rl
git clone https://github.com/Tetramputechture/npp-rl.git
cd npp-rl
git checkout task-2.4-training-stability-optimization

# Clone nclone (required dependency)
cd ..
git clone https://github.com/Tetramputechture/nclone.git
cd nclone
git checkout task-2.4-training-stability-optimization  # If nclone changes exist
```

### Step 2: Setup Python Environment

**Option A: Using conda (recommended)**

```bash
# Create conda environment
conda create -n npp-rl python=3.10
conda activate npp-rl

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install nclone
cd nclone
pip install -e .

# Install npp-rl
cd ../npp-rl
pip install -r requirements.txt
```

**Option B: Using venv**

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install nclone
cd nclone
pip install -e .

# Install npp-rl
cd ../npp-rl
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Test nclone installation
python -c "import nclone; print('nclone version:', nclone.__version__)"

# Test GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run quick environment test
python -c "from nclone.gym_environment import create_hierarchical_env; env = create_hierarchical_env(); print('Environment created successfully'); env.close()"
```

Expected output:
```
nclone version: 0.1.0
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
Environment created successfully
```

### Step 4: Download Pre-trained Models (Optional)

If you want to start from pre-trained checkpoints:

```bash
# Download Phase 2 Task 2.3 checkpoint (if available)
mkdir -p models/pretrained
cd models/pretrained
# wget <checkpoint_url>  # Replace with actual URL
```

---

## Training Procedures

### Basic Training

**Default Configuration (Recommended)**

Start with default settings optimized for H100:

```bash
python train_hierarchical_stable.py
```

This runs:
- 64 parallel environments
- 10M timesteps (~8-12 hours on H100)
- Warmup: 100k steps
- Adaptive LR: Enabled
- ICM exploration: Enabled
- Checkpoints every 50k steps
- Evaluation every 10k steps

**Custom Configuration**

Adjust parameters for your hardware:

```bash
# For smaller GPU (e.g., RTX 3090)
python train_hierarchical_stable.py --num_envs 32 --total_timesteps 5000000

# For massive training run (multi-day on H100)
python train_hierarchical_stable.py --num_envs 128 --total_timesteps 50000000 --warmup_steps 200000

# Quick test run (1M steps, ~1 hour)
python train_hierarchical_stable.py --num_envs 16 --total_timesteps 1000000 --warmup_steps 50000
```

### Advanced Training Options

**Curriculum Learning**

Enable curriculum progression from simple to complex levels:

```bash
python train_hierarchical_stable.py --use_curriculum
```

**Disable Adaptive Learning Rate**

For debugging or controlled experiments:

```bash
python train_hierarchical_stable.py --no_adaptive_lr
```

**Custom Logging Directory**

Organize multiple training runs:

```bash
python train_hierarchical_stable.py --log_dir ./experiments/run_001
```

**Longer Warmup Phase**

For more stable low-level policy before hierarchical coordination:

```bash
python train_hierarchical_stable.py --warmup_steps 500000
```

### Resume Training

To resume from a checkpoint:

```bash
# Training automatically saves checkpoints in:
# training_logs/hierarchical_stable_YYYYMMDD_HHMMSS/checkpoints/

# To resume, you'll need to modify the script or use:
python -c "
from stable_baselines3 import PPO
from npp_rl.agents.hierarchical_ppo import HierarchicalPPO

# Load checkpoint
model = HierarchicalPPO.load('training_logs/.../checkpoints/hierarchical_ppo_500000_steps')

# Continue training
model.learn(total_timesteps=5000000, reset_num_timesteps=False)
"
```

### Training Phases

The training script automatically implements a two-phase approach:

#### Phase 1: Warmup (First 100k steps)
- **Goal**: Stabilize low-level policy
- **Behavior**: High-level policy learns slowly (10% learning rate)
- **Focus**: Basic navigation and action execution
- **Metrics to watch**: 
  - `low_level/policy_loss` should decrease
  - `low_level/entropy` should be high (exploration)
  - Episode length should increase

#### Phase 2: Full Hierarchical Training (Remaining steps)
- **Goal**: Coordinate high-level and low-level policies
- **Behavior**: Both policies train at full learning rates
- **Focus**: Strategic subtask selection and efficient execution
- **Metrics to watch**:
  - `hierarchical/subtask_transitions` should be regular
  - `hierarchical/coordination_efficiency` should increase
  - Success rate should improve

---

## Monitoring and Evaluation

### Real-Time Monitoring with TensorBoard

**Start TensorBoard**

```bash
# Monitor all training runs
tensorboard --logdir training_logs/ --port 6006

# Monitor specific run
tensorboard --logdir training_logs/hierarchical_stable_20251003_143022/tensorboard --port 6006
```

Access at: `http://localhost:6006`

**Key Metrics to Monitor**

1. **Training Progress**
   - `rollout/ep_rew_mean`: Average episode reward (should increase)
   - `rollout/ep_len_mean`: Average episode length
   - `rollout/success_rate`: Level completion rate (target: >50%)

2. **Policy Performance**
   - `high_level/policy_loss`: High-level policy loss (should stabilize low)
   - `low_level/policy_loss`: Low-level policy loss (should stabilize low)
   - `high_level/entropy`: High-level exploration (should decay slowly)
   - `low_level/entropy`: Low-level exploration (should decay slowly)

3. **Stability Indicators**
   - `stability/is_stable`: 1.0 = stable, 0.0 = unstable
   - `stability/gradient_norm_ratio`: Ratio of high/low gradient norms (target: 0.5-2.0)
   - `stability/high_level_gradient_norm_mean`: Should stay < 10.0
   - `stability/low_level_gradient_norm_mean`: Should stay < 10.0

4. **Hierarchical Coordination**
   - `hierarchical/total_transitions`: Number of subtask switches
   - `hierarchical/avg_duration_*`: Average duration per subtask
   - `hierarchical/coordination_efficiency`: How well policies coordinate

5. **Exploration (ICM)**
   - `icm/intrinsic_reward_mean`: Curiosity-driven exploration reward
   - `icm/forward_loss_mean`: ICM prediction error
   - `icm/inverse_loss_mean`: ICM action prediction error

### Console Logging

Training progress is also logged to console:

```
==================================================================================
WARMUP PHASE: Training low-level policy for 100,000 steps
==================================================================================

Episode 50 | Steps: 12800 | Reward: 0.35 | Length: 256 | Success: 0/50
Episode 100 | Steps: 25600 | Reward: 0.48 | Length: 256 | Success: 3/100
...

✓ Warmup phase complete

==================================================================================
MAIN TRAINING: Full hierarchical training for 9,900,000 steps
==================================================================================

Episode 150 | Steps: 38400 | Reward: 0.62 | Length: 256 | Success: 12/150
Stability check: STABLE | Gradient norm ratio: 1.23 | LR: 2.8e-4
...
```

### Log Files

Training generates comprehensive logs:

```
training_logs/hierarchical_stable_YYYYMMDD_HHMMSS/
├── training_config.json          # Training configuration
├── training/
│   ├── progress.csv              # Training metrics (CSV format)
│   └── stdout.log                # Console output
├── tensorboard/                  # TensorBoard event files
│   └── events.out.tfevents.*
├── eval/
│   ├── evaluations.npz           # Evaluation results
│   └── best_model.zip            # Best performing model
├── checkpoints/                  # Periodic checkpoints
│   ├── hierarchical_ppo_50000_steps.zip
│   ├── hierarchical_ppo_100000_steps.zip
│   └── ...
└── final_model.zip               # Final trained model
```

---

## Benchmarking

### Standard Benchmark Suite

Run standardized benchmarks to compare performance:

```bash
# Create benchmark script
cat > benchmark.py << 'EOF'
import numpy as np
from npp_rl.agents.hierarchical_ppo import HierarchicalPPO
from nclone.gym_environment import create_hierarchical_env

# Load trained model
model = HierarchicalPPO.load("training_logs/.../best_model/best_model")

# Benchmark on 100 episodes
env = create_hierarchical_env()
episodes = 100
rewards = []
lengths = []
successes = []

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        ep_length += 1
    
    rewards.append(ep_reward)
    lengths.append(ep_length)
    successes.append(info.get('is_success', False))
    
    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1}/{episodes}: Reward={ep_reward:.2f}, Length={ep_length}, Success={info.get('is_success', False)}")

# Print results
print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)
print(f"Episodes: {episodes}")
print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
print(f"Success Rate: {np.mean(successes):.1%}")
print(f"Max Reward: {np.max(rewards):.2f}")
print(f"Min Reward: {np.min(rewards):.2f}")
print("="*80)
EOF

python benchmark.py
```

### Performance Targets

Based on Phase 2 objectives:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Success Rate | >30% | >50% | >70% |
| Mean Reward | >0.5 | >0.8 | >1.2 |
| Episode Length | <2000 | <1500 | <1000 |
| Training Stability | Stable 90%+ | Stable 95%+ | Stable 99%+ |
| Subtask Transitions | 5-20 per episode | 8-15 per episode | 10-12 per episode |

### Level-Specific Benchmarks

Test performance on different level types:

```python
# benchmark_by_level_type.py
from nclone.gym_environment import NppEnvironment

level_types = {
    'simple': [0, 1, 2],  # Simple single-switch levels
    'medium': [10, 11, 12],  # Two-switch levels
    'complex': [20, 21, 22],  # Multi-switch with mines
}

for level_type, level_ids in level_types.items():
    print(f"\nBenchmarking {level_type} levels...")
    # Run benchmark on each level type
    # ...
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce number of environments
python train_hierarchical_stable.py --num_envs 32

# Use smaller batch size (modify hyperparameters)
# Or enable gradient accumulation in GPU_OPTIMIZATION config
```

#### 2. Training Instability

**Symptoms**: Loss diverges, NaN values, crashes

**Solutions**:
```bash
# Disable adaptive LR to use fixed schedule
python train_hierarchical_stable.py --no_adaptive_lr

# Increase warmup period
python train_hierarchical_stable.py --warmup_steps 200000

# Check stability metrics in TensorBoard
# If gradient norms explode, reduce learning rates in hierarchical_hyperparameters.py
```

#### 3. Slow Training

**Symptoms**: Very slow progress, low GPU utilization

**Solutions**:
```bash
# Check GPU usage
nvidia-smi

# Increase number of environments (if CPU allows)
python train_hierarchical_stable.py --num_envs 128

# Enable TF32 (should be default on H100)
# Verify in GPU_OPTIMIZATION config
```

#### 4. Low Success Rate

**Symptoms**: Agent doesn't complete levels even after long training

**Solutions**:
- Check if warmup phase was sufficient (should see basic navigation working)
- Verify hierarchical coordination (check `hierarchical/subtask_transitions`)
- Increase training time (10M steps may not be enough for complex levels)
- Enable curriculum learning to start with simpler levels

#### 5. Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'nclone'` or similar

**Solutions**:
```bash
# Ensure nclone is installed
cd ../nclone
pip install -e .

# Ensure npp-rl is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/npp-rl"

# Or install npp-rl in development mode
cd /path/to/npp-rl
pip install -e .
```

### Debugging Tools

**Check Environment Creation**

```python
from nclone.gym_environment import create_hierarchical_env
env = create_hierarchical_env()
obs, info = env.reset()
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Observation keys:", obs.keys())
env.close()
```

**Test Single Training Step**

```python
from npp_rl.agents.hierarchical_ppo import HierarchicalPPO
from nclone.gym_environment import create_hierarchical_env

env = create_hierarchical_env()
model = HierarchicalPPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000)  # Quick test
```

**Profile Performance**

```bash
# Profile training script
python -m cProfile -o profile.stats train_hierarchical_stable.py --total_timesteps 100000

# View results
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(20)"
```

---

## Performance Metrics

### Key Performance Indicators (KPIs)

#### 1. Task Completion Metrics

- **Level Completion Rate**: Percentage of levels completed successfully
  - Formula: `successes / total_episodes`
  - Target: >50% after 10M steps

- **Average Episode Reward**: Mean reward across episodes
  - Formula: `mean(episode_rewards)`
  - Target: >0.8 after 10M steps

- **Time to Completion**: Average steps to complete level
  - Formula: `mean(episode_lengths[successful_episodes])`
  - Target: <1500 steps

#### 2. Training Stability Metrics

- **Gradient Norm Stability**: Consistency of gradient norms
  - Measure: `std(gradient_norms) / mean(gradient_norms)`
  - Target: <0.5 (low variance)

- **Value Loss Convergence**: How well value function approximates returns
  - Measure: `value_loss` over time
  - Target: Decreasing trend, stabilizing <1.0

- **Training Crashes**: Number of instability events
  - Measure: Count of `is_stable = 0` flags
  - Target: <1% of training steps

#### 3. Hierarchical Coordination Metrics

- **Subtask Transition Rate**: How often agent switches subtasks
  - Measure: `transitions / total_steps`
  - Target: 0.01-0.05 (1-5% of steps)

- **Subtask Success Rate**: Percentage of subtasks completed successfully
  - Measure: `successful_subtask_completions / total_subtasks`
  - Target: >60%

- **Coordination Efficiency**: How well policies work together
  - Measure: `(successful_episodes * avg_subtask_success) / avg_transitions`
  - Target: >0.5

#### 4. Exploration Metrics

- **State Coverage**: Percentage of state space explored
  - Measure: Unique states visited / estimated state space
  - Target: Increasing over time

- **Intrinsic Reward Magnitude**: Strength of curiosity signal
  - Measure: `mean(intrinsic_rewards)`
  - Target: 0.01-0.1 (balanced with extrinsic)

- **Mine Avoidance Success**: Ability to navigate around mines
  - Measure: `safe_mine_passages / total_mine_encounters`
  - Target: >80%

### Reporting Template

Use this template to report benchmarking results:

```markdown
## Training Run Report

### Configuration
- **Date**: YYYY-MM-DD
- **Hardware**: NVIDIA H100 80GB, 64 CPU cores, 128GB RAM
- **Total Timesteps**: 10,000,000
- **Training Time**: 10.2 hours
- **Environments**: 64 parallel

### Results
- **Success Rate**: 62.3%
- **Mean Reward**: 0.87 ± 0.24
- **Mean Episode Length**: 1423 ± 387 steps
- **Training Stability**: 98.7% stable

### Hierarchical Performance
- **Subtask Transitions**: 14.2 per episode
- **Subtask Success Rate**: 67.8%
- **Coordination Efficiency**: 0.61

### Notable Observations
- Stable training throughout all 10M steps
- Adaptive LR reduced learning rate 3 times during training
- Best performance on medium-complexity levels (78% success)
- Lower performance on complex multi-mine levels (41% success)

### Recommendations
- Increase training to 20M steps for complex levels
- Consider specialized mine avoidance training
- Tune high-level policy learning rate (currently 1e-4)
```

---

## Additional Resources

### Documentation
- [Phase 2 Hierarchical Control Tasks](./tasks/PHASE_2_HIERARCHICAL_CONTROL.md)
- [ICM Integration Guide](./ICM_INTEGRATION_GUIDE.md)
- [Full Implementation Plan](./full_plan.md)

### Research Papers
- PPO: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- ICM: [Curiosity-driven Exploration](https://arxiv.org/abs/1705.05363)
- Hierarchical RL: [Data-Efficient Hierarchical RL](https://arxiv.org/abs/1805.08296)

### Contact and Support
- GitHub Issues: https://github.com/Tetramputechture/npp-rl/issues
- Documentation: https://github.com/Tetramputechture/npp-rl/docs

---

## Changelog

### Task 2.4 (Current)
- Hierarchical training stability optimization
- Adaptive hyperparameter adjustment
- Comprehensive monitoring callbacks
- Warmup phase and curriculum progression
- H100 GPU optimizations

### Task 2.3
- Mine avoidance integration
- Mine-aware curiosity modulation

### Task 2.2
- Subtask-specific reward functions
- PBRS integration

### Task 2.1
- Two-level hierarchical policy architecture
- High-level and low-level policy coordination

---

**Last Updated**: October 3, 2025  
**Task**: 2.4 - Training Stability and Optimization  
**Status**: Implementation Complete, Ready for Testing
