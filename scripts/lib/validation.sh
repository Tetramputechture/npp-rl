#!/bin/bash
###############################################################################
# Validation Functions
# 
# This module contains functions for validating local and remote environments,
# including Python version checks, SSH connection tests, CUDA detection,
# and dependency verification.
###############################################################################

# ============================================================================
# Validation functions
# ============================================================================
validate_local_environment() {
    log INFO "Validating local environment..."
    
    # Check if we're in npp-rl directory
    # if [[ ! -f "setup.py" ]] || [[ ! -d "npp_rl" ]]; then
    #     log ERROR "This script must be run from the npp-rl directory"
    #     log ERROR "Current directory: $(pwd)"
    #     return 1
    # fi
    
    # Check if nclone directory exists
    if [[ ! -d "$NCLONE_DIR" ]]; then
        log ERROR "nclone directory not found at: ${NCLONE_DIR}"
        log ERROR "Expected nclone to be in the same parent directory as npp-rl"
        return 1
    fi
    
    log SUCCESS "Local environment validated"
    log INFO "  npp-rl: ${NPP_RL_DIR}"
    log INFO "  nclone: ${NCLONE_DIR}"
    return 0
}

validate_ssh_connection() {
    log INFO "Validating SSH connection to ${INSTANCE_IP}..."
    
    if ssh_cmd_basic "echo 'SSH connection successful'" > /dev/null 2>&1; then
        log SUCCESS "SSH connection established"
        return 0
    else
        log ERROR "Failed to establish SSH connection"
        return 1
    fi
}

verify_python_version() {
    log INFO "Verifying Python version requirements..."
    
    # Check if Python is available
    if ! ssh_cmd_basic "which python3" > /dev/null 2>&1; then
        log ERROR "Python3 not found on remote instance"
        return 1
    fi
    
    # Check current Python version
    local python_version=$(ssh_cmd_basic "python3 -c 'import sys; print(\".\".join(map(str, sys.version_info[:3])))'" 2>/dev/null)
    log INFO "Python version: ${python_version}"
    
    # Check for Python 3.11+ (required by nclone and npp-rl)
    if ssh_cmd_basic "python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'" > /dev/null 2>&1; then
        log SUCCESS "Python ${python_version} meets requirements (>=3.11)"
        return 0
    else
        log ERROR "Python ${python_version} does not meet requirements (>=3.11)"
        log ERROR "Instance should have Python 3.11+ pre-installed"
        return 1
    fi
}

detect_cuda_home() {
    log INFO "Detecting CUDA installation path..."
    
    # Common CUDA installation paths to check
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12.3"
        "/usr/local/cuda-12.2"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda-11.8"
        "/opt/cuda"
    )
    
    for path in "${cuda_paths[@]}"; do
        if ssh_cmd_basic "test -d ${path}/bin && test -f ${path}/bin/nvcc" > /dev/null 2>&1; then
            log SUCCESS "Found CUDA installation at: ${path}"
            
            # Get CUDA version
            local cuda_version=$(ssh_cmd_basic "${path}/bin/nvcc --version 2>/dev/null | grep 'release' | awk '{print \$5}' | cut -d',' -f1" 2>/dev/null || echo "unknown")
            log INFO "  CUDA version: ${cuda_version}"
            
            # Export to global variable for use in other functions
            DETECTED_CUDA_HOME="$path"
            return 0
        fi
    done
    
    # Try to detect from nvidia-smi
    if ssh_cmd_basic "which nvidia-smi" > /dev/null 2>&1; then
        log WARNING "CUDA installation directory not found in standard locations"
        log WARNING "nvidia-smi is available but CUDA toolkit may not be fully installed"
        log INFO "Will use default path: /usr/local/cuda"
        DETECTED_CUDA_HOME="/usr/local/cuda"
        return 0
    fi
    
    log WARNING "Could not detect CUDA installation path"
    DETECTED_CUDA_HOME="/usr/local/cuda"
    return 0
}

configure_cuda_environment() {
    log INFO "Configuring CUDA environment on remote instance..."
    
    # Create a CUDA environment setup script
    local cuda_env_script="
# CUDA Environment Configuration
export CUDA_HOME=${DETECTED_CUDA_HOME}
export CUDA_PATH=${DETECTED_CUDA_HOME}
export PATH=\${CUDA_HOME}/bin:\${PATH}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}
"
    
    # Add to ~/.bashrc if not already present
    ssh_cmd_basic "grep -q 'CUDA Environment Configuration' ~/.bashrc 2>/dev/null || echo '${cuda_env_script}' >> ~/.bashrc"
    
    # Also create a separate file that can be sourced
    ssh_cmd_basic "cat > ~/cuda_env.sh << 'EOF'
${cuda_env_script}
EOF"
    
    # Verify the configuration was written
    if ssh_cmd_basic "test -f ~/cuda_env.sh"; then
        log SUCCESS "CUDA environment configuration saved to ~/cuda_env.sh"
        log INFO "Configuration will be automatically loaded in new shell sessions"
        
        # Mark CUDA environment as configured
        CUDA_ENV_CONFIGURED=true
        
        # Verify the environment variables are set
        local cuda_home_check=$(ssh_cmd "echo \$CUDA_HOME")
        local ld_library_check=$(ssh_cmd "echo \$LD_LIBRARY_PATH | grep -o 'cuda' | head -1" || echo "")
        
        if [[ -n "$cuda_home_check" ]] && [[ "$cuda_home_check" != "\$CUDA_HOME" ]]; then
            log SUCCESS "✓ CUDA_HOME is set: ${cuda_home_check}"
        else
            log WARNING "CUDA_HOME may not be set correctly"
        fi
        
        if [[ -n "$ld_library_check" ]]; then
            log SUCCESS "✓ LD_LIBRARY_PATH includes CUDA libraries"
        else
            log WARNING "LD_LIBRARY_PATH may not include CUDA libraries"
        fi
        
    else
        log ERROR "Failed to create CUDA environment configuration"
        return 1
    fi
    
    log INFO "CUDA environment variables will be sourced for all subsequent remote commands"
    
    return 0
}

validate_remote_dependencies() {
    log INFO "Validating remote dependencies..."
    
    # Verify Python version
    if ! verify_python_version; then
        log ERROR "Python version requirement not met"
        return 1
    fi
    
    # Detect CUDA installation path
    detect_cuda_home
    
    # Configure CUDA environment system-wide
    if ! configure_cuda_environment; then
        log WARNING "Failed to configure CUDA environment, but continuing..."
        # Don't fail here - we can still try to continue
    fi
    
    # Check for system ML packages (optional - will be installed via pip if missing/incompatible)
    log INFO "Checking for system ML packages (PyTorch, TensorFlow, JAX)..."
    if ssh_cmd "python3 -c 'import torch; import tensorflow; import jax' 2>/dev/null"; then
        log SUCCESS "System ML packages found"
        
        # Log versions
        local torch_version=$(ssh_cmd "python3 -c 'import torch; print(torch.__version__)' 2>/dev/null")
        local tf_version=$(ssh_cmd "python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null")
        local jax_version=$(ssh_cmd "python3 -c 'import jax; print(jax.__version__)' 2>/dev/null")
        log INFO "  PyTorch: ${torch_version}"
        log INFO "  TensorFlow: ${tf_version}"
        log INFO "  JAX: ${jax_version}"
        log INFO "  (Will check compatibility and install via pip if needed)"
    else
        log WARNING "System ML packages not found"
        log INFO "  Will install compatible versions via pip"
    fi
    
    # Check for GPU
    if ssh_cmd "which nvidia-smi" > /dev/null 2>&1; then
        log SUCCESS "GPU detected on remote instance (nvidia-smi)"
        ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" | while read line; do
            log INFO "  GPU: $line"
        done
        
        # Verify PyTorch can access CUDA
        log INFO "Verifying PyTorch CUDA compatibility..."
        local cuda_check=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null" || echo "false 0")
        local cuda_available=$(echo $cuda_check | awk '{print $1}')
        local gpu_count=$(echo $cuda_check | awk '{print $2}')
        
        if [ "$cuda_available" = "True" ]; then
            log SUCCESS "PyTorch CUDA is available (${gpu_count} GPU(s) detected)"
        else
            log WARNING "PyTorch cannot access CUDA! GPUs detected but PyTorch CUDA unavailable"
            log WARNING "Possible issues:"
            log WARNING "  - Verify system packages: dpkg -l | grep python3-torch-cuda"
            log WARNING "  - CUDA driver version mismatch"
            log WARNING "  - Environment variable issues (CUDA_VISIBLE_DEVICES)"
            log WARNING "Training will fall back to CPU (very slow)"
        fi
    else
        log WARNING "nvidia-smi not found. Training will use CPU (slow)"
    fi
    
    log SUCCESS "Remote dependencies validated"
    return 0
}

install_system_dependencies() {
    log INFO "Verifying system dependencies on remote..."
    
    # Check if all required build dependencies are available
    local missing_deps=()
    
    # Check for cairo libraries (required for pycairo)
    if ! ssh_cmd "pkg-config --exists cairo" > /dev/null 2>&1; then
        missing_deps+=("libcairo2-dev")
    fi
    
    # Check for pkg-config
    if ! ssh_cmd "which pkg-config" > /dev/null 2>&1; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for build-essential (gcc, make, etc.)
    if ! ssh_cmd "which gcc" > /dev/null 2>&1; then
        missing_deps+=("build-essential")
    fi
    
    # Check for python3-dev (Python.h header files)
    if ! ssh_cmd "test -f /usr/include/python3.*/Python.h" > /dev/null 2>&1; then
        missing_deps+=("python3-dev")
    fi
    
    # If all dependencies are present, we're done
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log SUCCESS "All system dependencies already installed"
        return 0
    fi
    
    # Some dependencies are missing, try to install if we have sudo
    log WARNING "Missing system dependencies: ${missing_deps[*]}"
    log WARNING "Attempting to install missing dependencies..."
    
    if ssh_cmd "sudo -n true" > /dev/null 2>&1; then
        log INFO "Installing build dependencies: ${missing_deps[*]}"
        # Install all missing dependencies in one command
        ssh_cmd "sudo apt-get update -qq && sudo apt-get install -y -qq ${missing_deps[*]}" > /dev/null 2>&1 || {
            log ERROR "Failed to install system dependencies"
            log ERROR "Please manually install: sudo apt-get install ${missing_deps[*]}"
            return 1
        }
        log SUCCESS "System dependencies installed successfully"
    else
        log ERROR "Missing system dependencies and no sudo access to install them"
        log ERROR "Please manually install: sudo apt-get install ${missing_deps[*]}"
        return 1
    fi
    
    return 0
}

