---
agent: 'CodeActAgent'
triggers: ['setup', 'install', 'dependencies', 'environment', 'requirements', 'getting started']
---

# NPP-RL Setup and Installation Guide

## System Requirements

### Operating System Support
- **Linux**: Primary development platform (Ubuntu 20.04+ recommended)
- **Windows**: Supported via WSL2 
- **macOS**: Supported but GPU training requires compatible hardware

### Hardware Requirements
- **CPU**: 8+ cores recommended for parallel training environments
- **RAM**: 16GB minimum, 32GB recommended for large batch training
- **GPU**: CUDA-capable GPU with 8GB+ VRAM recommended (RTX 3070/4070 or better)
- **Storage**: 10GB+ free space for models, logs, and datasets

### Critical System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libcairo2-dev pkg-config python3-dev git python3-pip python3-venv
```

**CentOS/RHEL:**
```bash
sudo yum install cairo-devel pkgconfig python3-devel git python3-pip
```

**macOS:**
```bash
brew install cairo pkg-config python@3.11
```

## Installation Process

### Step 1: Directory Structure Setup

Create a parent directory for both repositories:
```bash
mkdir ~/npp-projects
cd ~/npp-projects
```

### Step 2: Clone Repositories

**CRITICAL: nclone must be installed FIRST as a sibling directory:**

```bash
# Clone the N++ simulator (required dependency)
git clone https://github.com/tetramputechture/nclone.git
cd nclone
pip install -e .
cd ..

# Clone the RL training project
git clone https://github.com/tetramputechture/npp-rl.git
cd npp-rl
```

### Step 3: Python Environment Setup

**Option A: Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

**Option B: Conda Environment**
```bash
conda create -n npp-rl python=3.11
conda activate npp-rl
```

### Step 4: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify nclone installation
python -c "import nclone; print('nclone installed successfully')"

# Verify core ML dependencies
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')"
```

### Step 5: Development Tools Setup

```bash
# Install linting and development tools
make dev-setup

# Verify code quality tools
make lint
```

## Dependency Details

### Core Dependencies Breakdown

**Deep Learning Stack:**
- `torch>=2.0.0` - PyTorch for neural networks
- `torchvision` - Computer vision utilities (installed with torch)

**Reinforcement Learning:**
- `stable-baselines3>=2.1.0` - PPO and other RL algorithms
- `sb3-contrib>=2.0.0` - Additional RL algorithms
- `gymnasium>=0.29.0` - Environment interface (replaces gym)

**Computer Vision:**
- `opencv-python>=4.8.0` - Image processing and video recording
- `pillow>=10.0.0` - Image manipulation
- `albumentations>=2.0.0` - Image augmentations

**Scientific Computing:**
- `numpy>=1.21.0` - Numerical computations
- `imageio>=2.31.0` - Video/image I/O

**Optimization and Logging:**
- `optuna>=3.3.0` - Hyperparameter tuning
- `tensorboard>=2.14.0` - Training visualization and logging

**Testing:**
- `pytest>=8.4.1` - Testing framework

### GPU Setup (CUDA)

**Check CUDA availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

**For CUDA installation issues:**
```bash
# Install PyTorch with specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# Or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

## Verification and Testing

### Quick Environment Test
```bash
# Test basic functionality
python -c "
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
env = BasicLevelNoGold()
obs = env.reset()
print('Environment setup successful!')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
env.close()
"
```

### Run Test Suite
```bash
# Run basic tests to verify setup
pytest tests/test_phase2_basic.py -v

# Run a quick training verification (1000 steps)
python -c "
from npp_rl.agents.training import train_enhanced_agent
model, log_dir = train_enhanced_agent(num_envs=4, total_timesteps=1000, render_mode='rgb_array')
print(f'Quick training test completed. Logs: {log_dir}')
"
```

### Development Environment Verification
```bash
# Test linting setup
make lint

# Test that you can run a minimal training
python -m npp_rl.agents.training --num_envs 4 --total_timesteps 1000
```

## Common Installation Issues

### Issue: "Cairo not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt install libcairo2-dev pkg-config python3-dev

# macOS  
brew install cairo pkg-config

# Then reinstall nclone
cd ../nclone
pip uninstall nclone
pip install -e .
```

### Issue: "nclone module not found"
**Solution:**
```bash
# Verify nclone is in the correct location
ls ../nclone/  # Should show nclone source code

# Reinstall nclone in editable mode
cd ../nclone
pip install -e .
cd ../npp-rl

# Verify installation
python -c "import nclone; print(nclone.__file__)"
```

### Issue: PyTorch CUDA not available
**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Permission denied" during pip install
**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or install user-local (not recommended for development)
pip install --user -r requirements.txt
```

## Development Configuration

### IDE Setup (VSCode/Cursor)
```bash
# The project includes .cursor/rules for development standards
# Key rules are automatically applied:
# - Python coding standards (file size limits, import organization)
# - Physics integration guidelines (use nclone.constants)
# - ML model standards (architecture, testing patterns)
# - Testing guidelines (behavioral vs static testing)
```

### Environment Variables
```bash
# Optional: Set for better performance logging
export CUDA_LAUNCH_BLOCKING=1  # For debugging CUDA issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # If import issues persist
```

## Post-Installation Next Steps

### 1. Quick Training Test
```bash
# Run a short training to verify everything works
python -m npp_rl.agents.training --num_envs 16 --total_timesteps 50000
```

### 2. Monitor Training
```bash
# In another terminal, start TensorBoard
tensorboard --logdir ./training_logs/enhanced_ppo_training/
# Open http://localhost:6006 in browser
```

### 3. Explore Documentation
```bash
# Read technical documentation
ls docs/
# Key files: README.md, docs/PHASE_2_IMPLEMENTATION_COMPLETE.md
```

### 4. Run Full Test Suite
```bash
# Comprehensive testing
pytest tests/ -v
```

## Performance Optimization

### For Training Performance
```bash
# Adjust based on your hardware
python -m npp_rl.agents.training \
    --num_envs $(nproc)  \  # Use all CPU cores
    --total_timesteps 10000000
```

### For Development
```bash
# Quick iteration with fewer environments
python -m npp_rl.agents.training --num_envs 4 --total_timesteps 100000
```

The setup is complete when you can successfully run training and see meaningful logs in TensorBoard without errors.
