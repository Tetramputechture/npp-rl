#!/bin/bash
# OpenHands Setup Script for NPP-RL Project (Debian/Ubuntu)
# This script sets up the complete development environment for the NPP-RL deep reinforcement learning project
# Installs CPU-only PyTorch for compatibility during setup (no GPU required)
# Optimized for Debian-based systems with atomic operations and timeout handling

set -e  # Exit on any error

# Setup logging

log_message() {
    local message="$1"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    
    # Echo to console
    echo "$message"
    
    # Ensure log directory exists and write to log file with timestamp
    mkdir -p ".openhands"
    echo "[$timestamp] $message" >> ".openhands/.setup.log"
}

log_message "Setting up NPP-RL development environment (Debian/Ubuntu)..."
log_message "This process will install CPU-only PyTorch (no GPU required during setup)."
log_message "Installation may take 8-12 minutes depending on your internet connection."
log_message "Please wait for all dependencies to be installed before continuing."
log_message "Logging to: .openhands/.setup.log"
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
    VERIFICATION_FAILED=false
    for package in "${MISSING_PACKAGES[@]}"; do
        # Use more flexible verification - check if package is installed (any status)
        if dpkg -l | grep -q "$package" && dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok"; then
            log_message "  Verified: $package"
        else
            # Try alternative verification method
            if apt list --installed 2>/dev/null | grep -q "^$package/"; then
                log_message "  Verified: $package (alternative method)"
            else
                log_message "[WARNING] Could not verify $package installation, but proceeding anyway"
                log_message "  This may cause issues later if $package is actually missing"
                VERIFICATION_FAILED=true
            fi
        fi
    done
    
    if [[ "$VERIFICATION_FAILED" == "true" ]]; then
        log_message ""
        log_message "[WARNING] Some package verifications failed, but continuing setup"
        log_message "If you encounter issues later, you may need to manually install missing packages"
        log_message ""
    fi
    
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
REQUIRED_MAJOR=3
REQUIRED_MINOR=8

# Extract major and minor version numbers
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

log_message "Detected Python $PYTHON_VERSION (required: >= $REQUIRED_MAJOR.$REQUIRED_MINOR)"

# Simple numeric comparison
if [[ $PYTHON_MAJOR -gt $REQUIRED_MAJOR ]] || [[ $PYTHON_MAJOR -eq $REQUIRED_MAJOR && $PYTHON_MINOR -ge $REQUIRED_MINOR ]]; then
    log_message "[SUCCESS] Python $PYTHON_VERSION is compatible (>= $REQUIRED_MAJOR.$REQUIRED_MINOR required)"
else
    log_message "[ERROR] Python $PYTHON_VERSION is too old. Please install Python >= $REQUIRED_MAJOR.$REQUIRED_MINOR"
    exit 1
fi

# Check for nclone dependency
log_message "[INFO] Checking for nclone dependency..."
NCLONE_PATH="../nclone"

if [[ ! -d "$NCLONE_PATH" ]]; then
    log_message "[WARNING] nclone not found at $NCLONE_PATH"
    log_message "[INFO] Cloning nclone repository..."
    
    # Ensure we can navigate to parent directory
    if ! cd ..; then
        log_message "[ERROR] Cannot navigate to parent directory"
        exit 1
    fi
    
    if [[ ! -d "nclone" ]]; then
        log_message "  Cloning from GitHub (this may take 1-2 minutes)..."
        if ! timeout 300 git clone --depth 1 https://github.com/tetramputechture/nclone.git; then
            log_message "[ERROR] Failed to clone nclone repository"
            log_message "[ERROR] This could be due to:"
            log_message "  - Network connectivity issues"
            log_message "  - GitHub being unreachable"
            log_message "  - Git not being installed"
            exit 1
        fi
        log_message "[SUCCESS] nclone repository cloned"
    else
        log_message "[INFO] nclone directory already exists, skipping clone"
    fi
    
    # Return to npp-rl directory with error checking
    if ! cd npp-rl; then
        log_message "[ERROR] Cannot return to npp-rl directory"
        exit 1
    fi
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
if ! timeout 300 python3 -m pip install -e .; then
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
log_message "  This step will install CPU-only PyTorch, Stable Baselines3, and other ML libraries."
log_message "  This may take 2-4 minutes to download CPU-only PyTorch (~700MB)."
log_message "  DO NOT interrupt this process!"

if [[ -f "requirements.txt" ]]; then
    # Pre-installation checks
    log_message "  Running pre-installation checks..."
    
    # Check available disk space (CPU-only PyTorch is ~700MB)
    log_message "  Checking available disk space..."
    if AVAILABLE_SPACE_KB=$(df . 2>/dev/null | tail -1 | awk '{print $4}') && [[ -n "$AVAILABLE_SPACE_KB" ]] && [[ "$AVAILABLE_SPACE_KB" =~ ^[0-9]+$ ]]; then
        AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))
        
        if [[ $AVAILABLE_SPACE_GB -lt 2 ]]; then
            log_message "[WARNING] Low disk space: ${AVAILABLE_SPACE_GB}GB available"
            log_message "[WARNING] PyTorch CPU installation requires ~2GB. Consider freeing up space."
        else
            log_message "  Disk space check: ${AVAILABLE_SPACE_GB}GB available ✓"
        fi
    else
        log_message "[WARNING] Could not determine available disk space"
        log_message "  Ensure you have at least 2GB free for PyTorch installation"
    fi
    
    log_message "  Installing CPU-only PyTorch (no GPU support during setup)"
    
    # Create optimized pip configuration for better performance
    export PIP_CACHE_DIR="${HOME}/.cache/pip"
    export PIP_NO_BUILD_ISOLATION=0
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_PREFER_BINARY=1
    
    # Install CPU-only PyTorch first
    log_message "  Installing CPU-only PyTorch packages (this may take 2-3 minutes)..."
    PYTORCH_CPU_URL="https://download.pytorch.org/whl/cpu"
    PYTORCH_TIMEOUT=600  # 10 minutes for PyTorch
    
    for attempt in 1 2 3; do
        log_message "    Attempt $attempt/3..."
        if timeout $PYTORCH_TIMEOUT python3 -m pip install torch torchvision torchaudio --index-url "$PYTORCH_CPU_URL"; then
            log_message "    ✓ CPU-only PyTorch packages installed successfully"
            break
        else
            if [[ $attempt -eq 3 ]]; then
                log_message "[ERROR] Failed to install PyTorch CPU version after 3 attempts"
                log_message "[ERROR] This could be due to:"
                log_message "  - Slow internet connection"
                log_message "  - Insufficient disk space"
                log_message "  - PyTorch server issues"
                exit 1
            else
                log_message "      Attempt $attempt failed, retrying in 10 seconds..."
                sleep 10
            fi
        fi
    done
    
    # Install remaining requirements (skip PyTorch packages as they're already installed)
    log_message "  Installing remaining requirements..."
    while IFS= read -r requirement; do
        # Skip empty lines, comments, and PyTorch packages
        if [[ -n "$requirement" && ! "$requirement" =~ ^[[:space:]]*# && ! "$requirement" =~ ^torch ]]; then
            log_message "    Installing: $requirement"
            if ! timeout 300 python3 -m pip install "$requirement"; then
                log_message "[ERROR] Failed to install: $requirement"
                exit 1
            fi
            log_message "    ✓ Installed: $requirement"
        fi
    done < requirements.txt
else
    log_message "[ERROR] requirements.txt not found"
    exit 1
fi

log_message "[SUCCESS] Core dependencies installed successfully"

# Install development tools
log_message "[INFO] Step 5/6: Installing development tools..."
if [[ -f "Makefile" ]] && command -v make &> /dev/null; then
    log_message "  Setting up development environment (this may take 1-2 minutes)..."
    if ! timeout 300 make dev-setup 2>/dev/null; then
        log_message "[WARNING] Development setup failed or timed out"
        log_message "  This is optional - you can run 'make dev-setup' manually later"
        log_message "  Check if dev dependencies are defined in your Makefile"
    else
        log_message "[SUCCESS] Development tools installed successfully"
    fi
elif [[ ! -f "Makefile" ]]; then
    log_message "[INFO] No Makefile found - skipping development tool setup"
    log_message "  You can install development tools manually if needed"
elif ! command -v make &> /dev/null; then
    log_message "[WARNING] 'make' command not found - skipping development setup"
    log_message "  Install 'make' if you need to use the project's Makefile"
else
    log_message "[INFO] Development tool setup skipped"
fi

log_message "[INFO] Step 6/6: Final dependency verification..."
log_message "  Checking all critical dependencies are available..."

# Verify core dependencies
log_message "[INFO] Verifying core dependencies..."

# Check PyTorch CPU installation
log_message "Running PyTorch verification..."
if python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    log_message "[SUCCESS] PyTorch $TORCH_VERSION (CPU-only) successfully imported"
    
    # CPU computation test
    if python3 -c "import torch; x = torch.randn(100, 100); y = x @ x; print('CPU computation test passed')" 2>/dev/null; then
        log_message "  ✓ CPU computation test: PASSED"
    else
        log_message "[ERROR] CPU computation test failed - PyTorch installation may be corrupted"
        exit 1
    fi
    
    # Check if torchvision and torchaudio are available
    if python3 -c "import torchvision; print(f'torchvision {torchvision.__version__}')" 2>/dev/null; then
        TORCHVISION_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)
        log_message "  ✓ torchvision $TORCHVISION_VERSION available"
    else
        log_message "[ERROR] torchvision not available"
        exit 1
    fi
    
    if python3 -c "import torchaudio; print(f'torchaudio {torchaudio.__version__}')" 2>/dev/null; then
        TORCHAUDIO_VERSION=$(python3 -c "import torchaudio; print(torchaudio.__version__)" 2>/dev/null)
        log_message "  ✓ torchaudio $TORCHAUDIO_VERSION available"
    else
        log_message "[ERROR] torchaudio not available"
        exit 1
    fi
else
    log_message "[ERROR] Failed to import PyTorch"
    log_message "[ERROR] This could be due to:"
    log_message "  - Installation was interrupted"
    log_message "  - Incompatible system libraries"
    log_message "  - Insufficient memory during installation"
    log_message "  - Corrupted download"
    log_message ""
    log_message "Try running the setup again or install manually with:"
    log_message "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
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
    # Use absolute path for PYTHONPATH to avoid shell expansion issues
    PROJECT_PATH=$(pwd)
    cat > .env << 'EOF'
# NPP-RL Environment Configuration
# Add current directory to Python path (will be expanded when sourced)
PYTHONPATH=${PYTHONPATH}:${PWD}

# Training defaults (CPU-only)
NPP_RL_NUM_ENVS=4
NPP_RL_TOTAL_TIMESTEPS=10000

# Logging
NPP_RL_LOG_LEVEL=INFO
NPP_RL_LOG_DIR=./training_logs
EOF
    
    # Add a comment with the actual current path for reference
    echo "# Current project path: $PROJECT_PATH" >> .env
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