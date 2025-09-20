#!/bin/bash
# OpenHands Setup Script for NPP-RL Project (Debian/Ubuntu)
# This script sets up the complete development environment for the NPP-RL deep reinforcement learning project
# Optimized for Debian-based systems with atomic operations and timeout handling

set -e  # Exit on any error

echo "Setting up NPP-RL development environment (Debian/Ubuntu)..."
echo "This process may take 10-15 minutes depending on your internet connection."
echo "Please wait for all dependencies to be installed before continuing."
echo ""

# Check if we're in the correct directory
if [[ ! -f "requirements.txt" ]] || [[ ! -d "npp_rl" ]]; then
    echo "[ERROR] This script must be run from the npp-rl project root directory"
    exit 1
fi

echo "[INFO] Detected NPP-RL project directory: $(pwd)"

# System dependency check and installation (Debian/Ubuntu)
echo "[INFO] Setting up Debian system dependencies..."

# Verify we have apt available
if ! command -v apt-get &> /dev/null; then
    echo "[ERROR] apt-get not found. This script requires a Debian-based system."
    exit 1
fi

echo "[INFO] Detected Debian/Ubuntu system with apt package manager"

# Check for required system packages
REQUIRED_PACKAGES=("libcairo2-dev" "pkg-config" "python3-dev" "git" "python3-pip" "python3-venv" "build-essential")
MISSING_PACKAGES=()

echo "Checking for required system packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        MISSING_PACKAGES+=("$package")
        echo "  Missing: $package"
    else
        echo "  Found: $package"
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    echo "[WARNING] Missing system packages: ${MISSING_PACKAGES[*]}"
    
    echo ""
    echo "DEPENDENCY INSTALLATION PHASE"
    echo "The following steps may take several minutes. Please wait for each step to complete."
    echo "DO NOT interrupt the process until you see 'DEPENDENCY INSTALLATION COMPLETE'"
    echo ""
    
    echo "[INFO] Step 1/3: Updating package index (this may take 1-2 minutes)..."
    if ! timeout 300 sudo apt-get update; then
        echo "[ERROR] Package update timed out or failed"
        exit 1
    fi
    echo "[SUCCESS] Package index updated successfully"
    
    echo "[INFO] Step 2/3: Installing system packages individually..."
    for package in "${MISSING_PACKAGES[@]}"; do
        echo "  Installing $package..."
        if ! timeout 180 sudo apt-get install -y "$package"; then
            echo "[ERROR] Failed to install $package"
            exit 1
        fi
        echo "  $package installed successfully"
    done
    
    echo "[INFO] Step 3/3: Verifying installations..."
    for package in "${MISSING_PACKAGES[@]}"; do
        if dpkg -l | grep -q "^ii  $package "; then
            echo "  Verified: $package"
        else
            echo "[ERROR] Verification failed for $package"
            exit 1
        fi
    done
    
    echo ""
    echo "DEPENDENCY INSTALLATION COMPLETE"
    echo "All required system packages are now installed"
    echo ""
else
    echo "[SUCCESS] All required system packages are already installed"
fi

# Python version check
echo "[INFO] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [[ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l 2>/dev/null || echo "0") == "1" ]] || [[ "$PYTHON_VERSION" > "$REQUIRED_VERSION" ]] || [[ "$PYTHON_VERSION" == "$REQUIRED_VERSION" ]]; then
    echo "[SUCCESS] Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    echo "[ERROR] Python $PYTHON_VERSION is too old. Please install Python >= $REQUIRED_VERSION"
    exit 1
fi

# Check for nclone dependency
echo "[INFO] Checking for nclone dependency..."
NCLONE_PATH="../nclone"

if [[ ! -d "$NCLONE_PATH" ]]; then
    echo "[WARNING] nclone not found at $NCLONE_PATH"
    echo "[INFO] Cloning nclone repository..."
    
    cd ..
    if [[ ! -d "nclone" ]]; then
        git clone https://github.com/tetramputechture/nclone.git
        echo "[SUCCESS] nclone repository cloned"
    fi
    cd npp-rl
else
    echo "[SUCCESS] nclone found at $NCLONE_PATH"
fi

# Install nclone
echo ""
echo "PYTHON DEPENDENCIES INSTALLATION PHASE"
echo "This phase may take 5-10 minutes depending on your internet connection."
echo "Please wait for each package to complete before continuing."
echo ""

echo "[INFO] Step 1/6: Installing nclone in editable mode..."
cd ../nclone
echo "  Installing nclone (this may take 1-2 minutes)..."
if ! timeout 300 pip install -e .; then
    echo "[ERROR] nclone installation timed out or failed"
    exit 1
fi
cd ../npp-rl
echo "[SUCCESS] nclone installation completed"

# Verify nclone installation
echo "[INFO] Step 2/6: Verifying nclone installation..."
if python3 -c "import nclone; print('nclone version:', nclone.__version__ if hasattr(nclone, '__version__') else 'dev')" 2>/dev/null; then
    echo "[SUCCESS] nclone installed and verified successfully"
else
    echo "[ERROR] Failed to verify nclone installation"
    exit 1
fi

# Upgrade pip and install Python dependencies
echo "[INFO] Step 3/6: Upgrading pip and build tools..."
echo "  Upgrading pip (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade pip; then
    echo "[ERROR] pip upgrade timed out or failed"
    exit 1
fi

echo "  Upgrading setuptools (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade setuptools; then
    echo "[ERROR] setuptools upgrade timed out or failed"
    exit 1
fi

echo "  Upgrading wheel (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade wheel; then
    echo "[ERROR] wheel upgrade timed out or failed"
    exit 1
fi

echo "[SUCCESS] Build tools upgraded successfully"

echo "[INFO] Step 4/6: Installing core dependencies from requirements.txt..."
echo "  This step will install PyTorch, Stable Baselines3, and other ML libraries."
echo "  This may take 3-5 minutes, especially if PyTorch needs to be downloaded."
echo "  DO NOT interrupt this process!"

if [[ -f "requirements.txt" ]]; then
    # Install requirements one by one for better error handling
    echo "  Installing requirements individually for better error handling..."
    while IFS= read -r requirement; do
        # Skip empty lines and comments
        if [[ -n "$requirement" && ! "$requirement" =~ ^[[:space:]]*# ]]; then
            echo "    Installing: $requirement"
            if ! timeout 600 pip install "$requirement"; then
                echo "[ERROR] Failed to install: $requirement"
                exit 1
            fi
            echo "    Installed: $requirement"
        fi
    done < requirements.txt
else
    echo "[ERROR] requirements.txt not found"
    exit 1
fi

echo "[SUCCESS] Core dependencies installed successfully"

# Install development tools
echo "[INFO] Step 5/6: Installing development tools..."
echo "  Setting up development environment (this may take 1-2 minutes)..."
if ! timeout 300 make dev-setup; then
    echo "[WARNING] Development setup timed out - you may need to run 'make dev-setup' manually later"
else
    echo "[SUCCESS] Development tools installed successfully"
fi

echo "[INFO] Step 6/6: Final dependency verification..."
echo "  Checking all critical dependencies are available..."

# Verify core dependencies
echo "[INFO] Verifying core dependencies..."

# Check PyTorch
if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        echo "[SUCCESS] PyTorch with CUDA support detected. GPU: $GPU_NAME"
    else
        echo "[WARNING] PyTorch installed but CUDA not available. Training will use CPU."
    fi
else
    echo "[ERROR] Failed to import PyTorch"
    exit 1
fi

# Check Stable Baselines3
if python3 -c "import stable_baselines3; print(f'Stable Baselines3 {stable_baselines3.__version__} installed')" 2>/dev/null; then
    echo "[SUCCESS] Stable Baselines3 installed successfully"
else
    echo "[ERROR] Failed to import Stable Baselines3"
    exit 1
fi

# Check nclone integration
if python3 -c "from nclone.constants import MAX_HOR_SPEED, GRAVITY_FALL; from nclone.graph.graph_builder import EdgeType; print('nclone integration verified')" 2>/dev/null; then
    echo "[SUCCESS] nclone integration verified"
else
    echo "[ERROR] nclone integration failed"
    exit 1
fi

echo ""
echo "DEPENDENCY VERIFICATION COMPLETE"
echo "All core dependencies have been verified"
echo ""

# Set up environment variables
echo "[INFO] Setting up environment variables..."

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
    echo "[SUCCESS] Created .env file with default configuration"
else
    echo "[SUCCESS] .env file already exists"
fi

# Create necessary directories
echo "[INFO] Creating necessary directories..."
mkdir -p training_logs datasets/processed datasets/raw models
echo "[SUCCESS] Directory structure created"


# Final setup summary
echo ""
echo "NPP-RL setup completed successfully!"
echo ""
echo "Setup Summary:"
echo "  - System dependencies installed"
echo "  - nclone dependency installed and verified"
echo "  - Python dependencies installed"
echo "  - Development tools configured"
echo "  - Core dependencies verified"
echo "  - Environment variables configured"
echo "  - Directory structure created"
echo ""
echo "Documentation:"
echo "  - README.md - Project overview and usage"
echo ""
echo "Development Commands:"
echo "  - make lint     # Check code quality"
echo "  - make fix      # Auto-fix code issues"
echo "  - make imports  # Remove unused imports"
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "[WARNING] Not in a virtual environment. Consider using:"
    echo "  python -m venv venv && source venv/bin/activate"
fi

echo "[SUCCESS] NPP-RL development environment is ready!"