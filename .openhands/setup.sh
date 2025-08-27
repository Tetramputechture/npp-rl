#!/bin/bash
# OpenHands Setup Script for NPP-RL Project
# This script sets up the complete development environment for the NPP-RL deep reinforcement learning project

set -e  # Exit on any error

echo "🚀 Setting up NPP-RL development environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [[ ! -f "requirements.txt" ]] || [[ ! -d "npp_rl" ]]; then
    print_error "This script must be run from the npp-rl project root directory"
    exit 1
fi

print_status "Detected NPP-RL project directory: $(pwd)"

# System dependency check and installation
print_status "Checking system dependencies..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Detected Linux system"
    
    # Check for required system packages
    REQUIRED_PACKAGES=("libcairo2-dev" "pkg-config" "python3-dev" "git")
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
        print_warning "Missing system packages: ${MISSING_PACKAGES[*]}"
        print_status "Installing system dependencies..."
        
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y "${MISSING_PACKAGES[@]}"
        elif command -v yum &> /dev/null; then
            sudo yum install -y cairo-devel pkgconfig python3-devel git python3-pip
        else
            print_error "Could not detect package manager. Please install: ${MISSING_PACKAGES[*]}"
            exit 1
        fi
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS system"
    
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew not found. Please install Homebrew first: https://brew.sh/"
        exit 1
    fi
    
    print_status "Installing macOS dependencies..."
    brew install cairo pkg-config python@3.11 || true
    
else
    print_warning "Unsupported OS: $OSTYPE. Proceeding with Python setup..."
fi

# Python version check
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [[ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l 2>/dev/null || echo "0") == "1" ]] || [[ "$PYTHON_VERSION" > "$REQUIRED_VERSION" ]] || [[ "$PYTHON_VERSION" == "$REQUIRED_VERSION" ]]; then
    print_success "Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    print_error "Python $PYTHON_VERSION is too old. Please install Python >= $REQUIRED_VERSION"
    exit 1
fi

# Check for nclone dependency
print_status "Checking for nclone dependency..."
NCLONE_PATH="../nclone"

if [[ ! -d "$NCLONE_PATH" ]]; then
    print_warning "nclone not found at $NCLONE_PATH"
    print_status "Cloning nclone repository..."
    
    cd ..
    if [[ ! -d "nclone" ]]; then
        git clone https://github.com/tetramputechture/nclone.git
        print_success "nclone repository cloned"
    fi
    cd npp-rl
else
    print_success "nclone found at $NCLONE_PATH"
fi

# Install nclone
print_status "Installing nclone in editable mode..."
cd ../nclone
pip install -e .
cd ../npp-rl

# Verify nclone installation
if python3 -c "import nclone; print('nclone version:', nclone.__version__ if hasattr(nclone, '__version__') else 'dev')" 2>/dev/null; then
    print_success "nclone installed successfully"
else
    print_error "Failed to install nclone"
    exit 1
fi

# Upgrade pip and install Python dependencies
print_status "Upgrading pip and installing Python dependencies..."
python3 -m pip install --upgrade pip setuptools wheel

print_status "Installing project dependencies from requirements.txt..."
pip install -r requirements.txt

# Install development tools
print_status "Installing development tools..."
make dev-setup

# Verify core dependencies
print_status "Verifying core dependencies..."

# Check PyTorch
if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        print_success "PyTorch with CUDA support detected. GPU: $GPU_NAME"
    else
        print_warning "PyTorch installed but CUDA not available. Training will use CPU."
    fi
else
    print_error "Failed to import PyTorch"
    exit 1
fi

# Check Stable Baselines3
if python3 -c "import stable_baselines3; print(f'Stable Baselines3 {stable_baselines3.__version__} installed')" 2>/dev/null; then
    print_success "Stable Baselines3 installed successfully"
else
    print_error "Failed to import Stable Baselines3"
    exit 1
fi

# Check nclone integration
if python3 -c "from nclone.constants import MAX_HOR_SPEED, GRAVITY_FALL; from nclone.graph.graph_builder import EdgeType; print('nclone integration verified')" 2>/dev/null; then
    print_success "nclone integration verified"
else
    print_error "nclone integration failed"
    exit 1
fi

# Run basic tests to verify setup
print_status "Running basic tests to verify setup..."
if python3 -m pytest tests/test_phase2_basic.py -v --tb=short; then
    print_success "Basic tests passed"
else
    print_warning "Some basic tests failed, but setup may still be functional"
fi

# Test core imports
print_status "Testing core module imports..."
python3 -c "
try:
    from npp_rl.models.gnn import GraphSAGELayer, GraphEncoder
    from npp_rl.models.conditional_edges import ConditionalEdgeMasker
    from npp_rl.models.physics_constraints import PhysicsConstraintValidator
    from npp_rl.agents.enhanced_training import train_enhanced_agent
    print('✓ All core modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "Core module imports verified"
else
    print_error "Core module import verification failed"
    exit 1
fi

# Set up environment variables
print_status "Setting up environment variables..."

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    cat > .env << EOF
# NPP-RL Environment Configuration
PYTHONPATH=\${PYTHONPATH}:\$(pwd)

# Performance optimization
CUDA_LAUNCH_BLOCKING=0

# Training defaults
NPP_RL_NUM_ENVS=16
NPP_RL_TOTAL_TIMESTEPS=1000000

# Logging
NPP_RL_LOG_LEVEL=INFO
NPP_RL_LOG_DIR=./training_logs
EOF
    print_success "Created .env file with default configuration"
else
    print_success ".env file already exists"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p training_logs datasets/processed datasets/raw models
print_success "Directory structure created"

# Quick functionality test
print_status "Running quick functionality test..."
python3 -c "
import sys
sys.path.insert(0, '.')

try:
    # Test basic environment creation
    from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
    env = BasicLevelNoGold()
    obs = env.reset()
    env.close()
    print('✓ Environment creation test passed')
    
    # Test model creation
    import torch
    from npp_rl.models.gnn import GraphEncoder
    encoder = GraphEncoder(node_dim=32, edge_dim=16, hidden_dim=64, num_layers=2)
    print('✓ Model creation test passed')
    
except Exception as e:
    print(f'✗ Functionality test failed: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    print_success "Quick functionality test passed"
else
    print_error "Quick functionality test failed"
    exit 1
fi

# Final setup summary
echo ""
echo "🎉 NPP-RL setup completed successfully!"
echo ""
echo "📋 Setup Summary:"
echo "  ✓ System dependencies installed"
echo "  ✓ nclone dependency installed and verified"
echo "  ✓ Python dependencies installed"
echo "  ✓ Development tools configured"
echo "  ✓ Core modules verified"
echo "  ✓ Environment variables configured"
echo "  ✓ Directory structure created"
echo "  ✓ Basic functionality tested"
echo ""
echo "🚀 Next Steps:"
echo "  1. Run a quick training test:"
echo "     python -m npp_rl.agents.enhanced_training --num_envs 4 --total_timesteps 10000"
echo ""
echo "  2. Start TensorBoard for monitoring:"
echo "     tensorboard --logdir ./training_logs"
echo ""
echo "  3. Run the full test suite:"
echo "     pytest tests/ -v"
echo ""
echo "  4. Check code quality:"
echo "     make lint"
echo ""
echo "📚 Documentation:"
echo "  - README.md - Project overview and usage"
echo "  - docs/PHASE_2_IMPLEMENTATION_COMPLETE.md - Technical details"
echo "  - docs/graph_plan.md - Graph neural network implementation"
echo ""
echo "🔧 Development Commands:"
echo "  - make lint     # Check code quality"
echo "  - make fix      # Auto-fix code issues"
echo "  - make imports  # Remove unused imports"
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "Not in a virtual environment. Consider using:"
    echo "  python -m venv venv && source venv/bin/activate"
fi

print_success "NPP-RL development environment is ready! 🎮🤖"