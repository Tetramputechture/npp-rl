#!/bin/bash
# OpenHands Setup Script for NPP-RL Project (Debian/Ubuntu)
# This script sets up the complete development environment for the NPP-RL deep reinforcement learning project
# Optimized for Debian-based systems with atomic operations and timeout handling

set -e  # Exit on any error

# Setup logging

log_message() {
    local message="$1"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    # Echo to console
    echo "$message"
    
    # Write to log file with timestamp
    echo "[$timestamp] $message" >> ".openhands/.setup.log"
}

log_message "Setting up NPP-RL development environment (Debian/Ubuntu)..."
log_message "This process may take 10-15 minutes depending on your internet connection."
log_message "Please wait for all dependencies to be installed before continuing."
log_message "Logging to: npp-rl-setup-$(date +'%Y%m%d_%H%M%S').log"
log_message ""

# Check if we're in the correct directory
if [[ ! -f "requirements.txt" ]] || [[ ! -d "npp_rl" ]]; then
    log_message "[ERROR] This script must be run from the npp-rl project root directory"
    exit 1
fi

log_message "[INFO] Detected NPP-RL project directory: $(pwd)"

# System dependency check and installation (Debian/Ubuntu)
log_message "[INFO] Setting up Debian system dependencies..."

# Verify we have apt available
if ! command -v apt-get &> /dev/null; then
    log_message "[ERROR] apt-get not found. This script requires a Debian-based system."
    exit 1
fi

log_message "[INFO] Detected Debian/Ubuntu system with apt package manager"

# Check for required system packages
REQUIRED_PACKAGES=("libcairo2-dev" "pkg-config" "python3-dev" "git" "python3-pip" "python3-venv" "build-essential")
MISSING_PACKAGES=()

log_message "Checking for required system packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        MISSING_PACKAGES+=("$package")
        log_message "  Missing: $package"
    else
        log_message "  Found: $package"
    fi
done

if [[ ${#MISSING_PACKAGES[@]} -gt 0 ]]; then
    log_message "[WARNING] Missing system packages: ${MISSING_PACKAGES[*]}"
    
    log_message ""
    log_message "DEPENDENCY INSTALLATION PHASE"
    log_message "The following steps may take several minutes. Please wait for each step to complete."
    log_message "DO NOT interrupt the process until you see 'DEPENDENCY INSTALLATION COMPLETE'"
    log_message ""
    
    log_message "[INFO] Step 1/3: Updating package index (this may take 1-2 minutes)..."
    if ! timeout 300 sudo apt-get update; then
        log_message "[ERROR] Package update timed out or failed"
        exit 1
    fi
    log_message "[SUCCESS] Package index updated successfully"
    
    log_message "[INFO] Step 2/3: Installing system packages individually..."
    for package in "${MISSING_PACKAGES[@]}"; do
        log_message "  Installing $package..."
        if ! timeout 180 sudo apt-get install -y "$package"; then
            log_message "[ERROR] Failed to install $package"
            exit 1
        fi
        log_message "  $package installed successfully"
    done
    
    log_message "[INFO] Step 3/3: Verifying installations..."
    for package in "${MISSING_PACKAGES[@]}"; do
        if dpkg -l | grep -q "^ii  $package "; then
            log_message "  Verified: $package"
        else
            log_message "[ERROR] Verification failed for $package"
            exit 1
        fi
    done
    
    log_message ""
    log_message "DEPENDENCY INSTALLATION COMPLETE"
    log_message "All required system packages are now installed"
    log_message ""
else
    log_message "[SUCCESS] All required system packages are already installed"
fi

# Python version check
log_message "[INFO] Checking Python version..."
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [[ $(log_message "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l 2>/dev/null || log_message "0") == "1" ]] || [[ "$PYTHON_VERSION" > "$REQUIRED_VERSION" ]] || [[ "$PYTHON_VERSION" == "$REQUIRED_VERSION" ]]; then
    log_message "[SUCCESS] Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    log_message "[ERROR] Python $PYTHON_VERSION is too old. Please install Python >= $REQUIRED_VERSION"
    exit 1
fi

# Check for nclone dependency
log_message "[INFO] Checking for nclone dependency..."
NCLONE_PATH="../nclone"

if [[ ! -d "$NCLONE_PATH" ]]; then
    log_message "[WARNING] nclone not found at $NCLONE_PATH"
    log_message "[INFO] Cloning nclone repository..."
    
    cd ..
    if [[ ! -d "nclone" ]]; then
        git clone https://github.com/tetramputechture/nclone.git
        log_message "[SUCCESS] nclone repository cloned"
    fi
    cd npp-rl
else
    log_message "[SUCCESS] nclone found at $NCLONE_PATH"
fi

# Install nclone
log_message ""
log_message "PYTHON DEPENDENCIES INSTALLATION PHASE"
log_message "This phase may take 5-10 minutes depending on your internet connection."
log_message "Please wait for each package to complete before continuing."
log_message ""

log_message "[INFO] Step 1/6: Installing nclone in editable mode..."
cd ../nclone
log_message "  Installing nclone (this may take 1-2 minutes)..."
if ! timeout 300 pip install -e .; then
    log_message "[ERROR] nclone installation timed out or failed"
    exit 1
fi
cd ../npp-rl
log_message "[SUCCESS] nclone installation completed"

# Verify nclone installation
log_message "[INFO] Step 2/6: Verifying nclone installation..."
if python3 -c "import nclone; print('nclone version:', nclone.__version__ if hasattr(nclone, '__version__') else 'dev')" 2>/dev/null; then
    log_message "[SUCCESS] nclone installed and verified successfully"
else
    log_message "[ERROR] Failed to verify nclone installation"
    exit 1
fi

# Upgrade pip and install Python dependencies
log_message "[INFO] Step 3/6: Upgrading pip and build tools..."
log_message "  Upgrading pip (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade pip; then
    log_message "[ERROR] pip upgrade timed out or failed"
    exit 1
fi

log_message "  Upgrading setuptools (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade setuptools; then
    log_message "[ERROR] setuptools upgrade timed out or failed"
    exit 1
fi

log_message "  Upgrading wheel (this may take 30-60 seconds)..."
if ! timeout 180 python3 -m pip install --upgrade wheel; then
    log_message "[ERROR] wheel upgrade timed out or failed"
    exit 1
fi

log_message "[SUCCESS] Build tools upgraded successfully"

log_message "[INFO] Step 4/6: Installing core dependencies from requirements.txt..."
log_message "  This step will install PyTorch, Stable Baselines3, and other ML libraries."
log_message "  This may take 3-5 minutes, especially if PyTorch needs to be downloaded."
log_message "  DO NOT interrupt this process!"

if [[ -f "requirements.txt" ]]; then
    # Install requirements one by one for better error handling
    log_message "  Installing requirements individually for better error handling..."
    while IFS= read -r requirement; do
        # Skip empty lines and comments
        if [[ -n "$requirement" && ! "$requirement" =~ ^[[:space:]]*# ]]; then
            log_message "    Installing: $requirement"
            if ! timeout 600 pip install "$requirement"; then
                log_message "[ERROR] Failed to install: $requirement"
                exit 1
            fi
            log_message "    Installed: $requirement"
        fi
    done < requirements.txt
else
    log_message "[ERROR] requirements.txt not found"
    exit 1
fi

log_message "[SUCCESS] Core dependencies installed successfully"

# Install development tools
log_message "[INFO] Step 5/6: Installing development tools..."
log_message "  Setting up development environment (this may take 1-2 minutes)..."
if ! timeout 300 make dev-setup; then
    log_message "[WARNING] Development setup timed out - you may need to run 'make dev-setup' manually later"
else
    log_message "[SUCCESS] Development tools installed successfully"
fi

log_message "[INFO] Step 6/6: Final dependency verification..."
log_message "  Checking all critical dependencies are available..."

# Verify core dependencies
log_message "[INFO] Verifying core dependencies..."

# Check PyTorch
if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        log_message "[SUCCESS] PyTorch with CUDA support detected. GPU: $GPU_NAME"
    else
        log_message "[WARNING] PyTorch installed but CUDA not available. Training will use CPU."
    fi
else
    log_message "[ERROR] Failed to import PyTorch"
    exit 1
fi

# Check Stable Baselines3
if python3 -c "import stable_baselines3; print(f'Stable Baselines3 {stable_baselines3.__version__} installed')" 2>/dev/null; then
    log_message "[SUCCESS] Stable Baselines3 installed successfully"
else
    log_message "[ERROR] Failed to import Stable Baselines3"
    exit 1
fi

# Check nclone integration
if python3 -c "from nclone.constants import MAX_HOR_SPEED, GRAVITY_FALL; from nclone.graph.graph_builder import EdgeType; print('nclone integration verified')" 2>/dev/null; then
    log_message "[SUCCESS] nclone integration verified"
else
    log_message "[ERROR] nclone integration failed"
    exit 1
fi

log_message ""
log_message "DEPENDENCY VERIFICATION COMPLETE"
log_message "All core dependencies have been verified"
log_message ""

# Set up environment variables
log_message "[INFO] Setting up environment variables..."

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
    log_message "[SUCCESS] Created .env file with default configuration"
else
    log_message "[SUCCESS] .env file already exists"
fi

# Create necessary directories
log_message "[INFO] Creating necessary directories..."
mkdir -p training_logs datasets/processed datasets/raw models
log_message "[SUCCESS] Directory structure created"


# Final setup summary
log_message ""
log_message "NPP-RL setup completed successfully!"
log_message ""
log_message "Setup Summary:"
log_message "  - System dependencies installed"
log_message "  - nclone dependency installed and verified"
log_message "  - Python dependencies installed"
log_message "  - Development tools configured"
log_message "  - Core dependencies verified"
log_message "  - Environment variables configured"
log_message "  - Directory structure created"
log_message ""
log_message "Documentation:"
log_message "  - README.md - Project overview and usage"
log_message ""
log_message "Development Commands:"
log_message "  - make lint     # Check code quality"
log_message "  - make fix      # Auto-fix code issues"
log_message "  - make imports  # Remove unused imports"
log_message ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    log_message "[WARNING] Not in a virtual environment. Consider using:"
    log_message "  python -m venv venv && source venv/bin/activate"
fi

log_message "[SUCCESS] NPP-RL development environment is ready!"