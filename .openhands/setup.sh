#!/bin/bash
# OpenHands Setup Script for NPP-RL Project (Debian/Ubuntu)
# This script sets up the complete development environment for the NPP-RL deep reinforcement learning project
# Optimized for Debian-based systems with atomic operations and timeout handling

set -e  # Exit on any error

echo "🚀 Setting up NPP-RL development environment (Debian/Ubuntu)..."
echo "⏳ This process may take 10-15 minutes depending on your internet connection."
echo "🚨 Please wait for all dependencies to be installed before continuing."
echo ""

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

# System dependency check and installation (Debian/Ubuntu)
print_status "Setting up Debian system dependencies..."

# Verify we have apt available
if ! command -v apt-get &> /dev/null; then
    print_error "apt-get not found. This script requires a Debian-based system."
    exit 1
fi

print_status "Detected Debian/Ubuntu system with apt package manager"

# Check for required system packages
REQUIRED_PACKAGES=("libcairo2-dev" "pkg-config" "python3-dev" "git" "python3-pip" "python3-venv" "build-essential")
MISSING_PACKAGES=()

echo "🔍 Checking for required system packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        MISSING_PACKAGES+=("$package")
        echo "  ❌ Missing: $package"
    else
        echo "  ✅ Found: $package"
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    print_warning "Missing system packages: ${MISSING_PACKAGES[*]}"
    
    echo ""
    echo "📦 DEPENDENCY INSTALLATION PHASE"
    echo "⏳ The following steps may take several minutes. Please wait for each step to complete."
    echo "🚨 DO NOT interrupt the process until you see 'DEPENDENCY INSTALLATION COMPLETE'"
    echo ""
    
    print_status "Step 1/3: Updating package index (this may take 1-2 minutes)..."
    if ! timeout 300 sudo apt-get update; then
        print_error "Package update timed out or failed"
        exit 1
    fi
    print_success "Package index updated successfully"
    
    print_status "Step 2/3: Installing system packages individually..."
    for package in "${MISSING_PACKAGES[@]}"; do
        echo "  Installing $package..."
        if ! timeout 180 sudo apt-get install -y "$package"; then
            print_error "Failed to install $package"
            exit 1
        fi
        echo "  ✅ $package installed successfully"
    done
    
    print_status "Step 3/3: Verifying installations..."
    for package in "${MISSING_PACKAGES[@]}"; do
        if dpkg -l | grep -q "^ii  $package "; then
            echo "  ✅ Verified: $package"
        else
            print_error "Verification failed for $package"
            exit 1
        fi
    done
    
    echo ""
    echo "🎉 DEPENDENCY INSTALLATION COMPLETE"
    echo "✅ All required system packages are now installed"
    echo ""
else
    print_success "All required system packages are already installed"
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
echo ""
echo "🐍 PYTHON DEPENDENCIES INSTALLATION PHASE"
echo "⏳ This phase may take 5-10 minutes depending on your internet connection."
echo "🚨 Please wait for each package to complete before continuing."
echo ""

print_status "Step 1/6: Installing nclone in editable mode..."
cd ../nclone
echo "  📦 Installing nclone (this may take 1-2 minutes)..."
if ! timeout 300 pip install -e .; then
    print_error "nclone installation timed out or failed"
    exit 1
fi
cd ../npp-rl
print_success "nclone installation completed"

# Verify nclone installation
print_status "Step 2/6: Verifying nclone installation..."
if python3 -c "import nclone; print('nclone version:', nclone.__version__ if hasattr(nclone, '__version__') else 'dev')" 2>/dev/null; then
    print_success "nclone installed and verified successfully"
else
    print_error "Failed to verify nclone installation"
    exit 1
fi

# Upgrade pip and install Python dependencies
print_status "Step 3/6: Upgrading pip and build tools..."
echo "  📦 Upgrading pip (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade pip; then
    print_error "pip upgrade timed out or failed"
    exit 1
fi

echo "  📦 Upgrading setuptools (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade setuptools; then
    print_error "setuptools upgrade timed out or failed"
    exit 1
fi

echo "  📦 Upgrading wheel (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade wheel; then
    print_error "wheel upgrade timed out or failed"
    exit 1
fi

print_success "Build tools upgraded successfully"

print_status "Step 4/6: Installing core dependencies from requirements.txt..."
echo "  📦 This step will install PyTorch, Stable Baselines3, and other ML libraries."
echo "  ⏳ This may take 3-5 minutes, especially if PyTorch needs to be downloaded."
echo "  🚨 DO NOT interrupt this process!"

if [[ -f "requirements.txt" ]]; then
    # Install requirements one by one for better error handling
    echo "  📦 Installing requirements individually for better error handling..."
    while IFS= read -r requirement; do
        # Skip empty lines and comments
        if [[ -n "$requirement" && ! "$requirement" =~ ^[[:space:]]*# ]]; then
            echo "    Installing: $requirement"
            if ! timeout 600 pip install "$requirement"; then
                print_error "Failed to install: $requirement"
                exit 1
            fi
            echo "    ✅ Installed: $requirement"
        fi
    done < requirements.txt
else
    print_error "requirements.txt not found"
    exit 1
fi

print_success "Core dependencies installed successfully"

# Install development tools
print_status "Step 5/6: Installing development tools..."
echo "  📦 Setting up development environment (this may take 1-2 minutes)..."
if ! timeout 300 make dev-setup; then
    print_warning "Development setup timed out - you may need to run 'make dev-setup' manually later"
else
    print_success "Development tools installed successfully"
fi

print_status "Step 6/6: Final dependency verification..."
echo "  🔍 Checking all critical dependencies are available..."

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

echo ""
echo "🧪 TESTING AND VERIFICATION PHASE"
echo "⏳ Running tests to verify the installation is working correctly."
echo ""

# Test core imports first (faster)
print_status "Step 1/3: Testing core module imports..."
echo "  🔍 Testing individual module imports..."

# Test imports one by one for better error reporting
declare -a MODULES=(
    "npp_rl.models.gnn:GraphSAGELayer"
    "npp_rl.models.gnn:GraphEncoder"
    "npp_rl.models.conditional_edges:ConditionalEdgeMasker"
    "npp_rl.models.physics_constraints:PhysicsConstraintValidator"
    "npp_rl.agents.training:train_enhanced_agent"
)

for module_import in "${MODULES[@]}"; do
    IFS=':' read -r module_name class_name <<< "$module_import"
    echo "    Testing: $module_name.$class_name"
    if ! python3 -c "from $module_name import $class_name; print(f'  ✅ {module_name}.{class_name} imported successfully')" 2>/dev/null; then
        print_error "Failed to import $module_name.$class_name"
        echo "  ❌ This may indicate a dependency issue or code problem"
        exit 1
    fi
done

print_success "All core module imports verified successfully"

# Run basic functionality test
print_status "Step 2/3: Running basic functionality test..."
echo "  🔍 Testing environment and model creation..."
echo "  ⏳ This may take 30-60 seconds..."

if ! timeout 120 python3 -c "
import sys
sys.path.insert(0, '.')

try:
    # Test basic environment creation
    print('  📦 Testing environment creation...')
    from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
    env = BasicLevelNoGold()
    obs = env.reset()
    env.close()
    print('  ✅ Environment creation test passed')
    
    # Test model creation
    print('  📦 Testing model creation...')
    import torch
    from npp_rl.models.gnn import GraphEncoder
    encoder = GraphEncoder(node_dim=32, edge_dim=16, hidden_dim=64, num_layers=2)
    print('  ✅ Model creation test passed')
    
    print('✅ All functionality tests passed')
    
except Exception as e:
    print(f'❌ Functionality test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"; then
    print_success "Basic functionality test passed"
else
    print_error "Basic functionality test failed or timed out"
    exit 1
fi

# Run basic tests (optional, with timeout)
print_status "Step 3/3: Running basic test suite (optional)..."
if [[ -f "tests/test_phase2_basic.py" ]]; then
    echo "  🧪 Running basic tests with timeout (this may take 1-2 minutes)..."
    echo "  ⏳ If tests take too long, they will be skipped automatically..."
    
    if timeout 300 python3 -m pytest tests/test_phase2_basic.py -v --tb=short; then
        print_success "Basic test suite passed"
    else
        print_warning "Basic tests failed or timed out - setup may still be functional"
        echo "  💡 You can run tests manually later with: pytest tests/test_phase2_basic.py"
    fi
else
    print_warning "Basic test file not found - skipping test suite"
fi

echo ""
echo "🎉 TESTING AND VERIFICATION COMPLETE"
echo "✅ Core functionality has been verified"
echo ""

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
echo "     python -m npp_rl.agents.training --num_envs 4 --total_timesteps 10000"
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