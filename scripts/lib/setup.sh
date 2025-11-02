#!/bin/bash
###############################################################################
# Setup Functions
# 
# This module contains functions for setting up local and remote environments,
# including TensorBoard forwarding, code copying, and package installation.
###############################################################################

# ============================================================================
# Setup functions
# ============================================================================
setup_local_environment() {
    log HEADER "Setting up local environment"
    
    # Ensure local log directory exists (already created at script start)
    mkdir -p "$LOCAL_LOG_DIR"
    log SUCCESS "Local log directory ready: ${LOCAL_LOG_DIR}"
    
    # Save configuration
    cat > "${LOCAL_LOG_DIR}/config.json" << EOF
{
    "instance_ip": "${INSTANCE_IP}",
    "s3_bucket": "${S3_BUCKET}",
    "experiment_name": "${EXPERIMENT_NAME}",
    "timestamp": "${TIMESTAMP}",
    "architectures": $(printf '%s\n' "${ARCHITECTURES[@]}" | jq -R . | jq -s .),
    "total_timesteps_per_arch": 1000000,
    "num_envs": 64
}
EOF
    
    log SUCCESS "Configuration saved to ${LOCAL_LOG_DIR}/config.json"
}

setup_tensorboard_forwarding() {
    log INFO "Setting up TensorBoard port forwarding..."
    
    # Kill any existing port forward on 6006
    lsof -ti:${TENSORBOARD_PORT} | xargs kill -9 2>/dev/null || true
    
    # Start SSH tunnel for TensorBoard in background
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -N -L ${TENSORBOARD_PORT}:localhost:${TENSORBOARD_PORT} \
        "${SSH_USER}@${INSTANCE_IP}" &
    TENSORBOARD_PID=$!
    
    sleep 2
    
    if ps -p $TENSORBOARD_PID > /dev/null; then
        log SUCCESS "TensorBoard port forwarding active (PID: ${TENSORBOARD_PID})"
        log INFO "TensorBoard available at: ${GREEN}http://localhost:${TENSORBOARD_PORT}${NC}"
        echo ""
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  TensorBoard: http://localhost:${TENSORBOARD_PORT}${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
    else
        log WARNING "TensorBoard port forwarding may have failed"
    fi
}

copy_code_to_remote() {
    log HEADER "Copying code to remote instance"
    
    # Create remote base directory
    log INFO "Creating remote base directory: ${REMOTE_BASE_DIR}"
    ssh_cmd "mkdir -p ${REMOTE_BASE_DIR}"
    
    # Copy npp-rl
    log INFO "Copying npp-rl to remote (this may take a few minutes)..."
    log INFO "  Excluding: .git, __pycache__, compiled modules (*.pyc, *.so), experiments, logs, datasets"
    
    # Create tarball excluding unnecessary files and ALL compiled modules
    tar -czf /tmp/npp-rl-${TIMESTAMP}.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.so' \
        --exclude='*.so.*' \
        --exclude='*.pyd' \
        --exclude='*.dll' \
        --exclude='experiments' \
        --exclude='logs' \
        --exclude='datasets' \
        --exclude='*.egg-info' \
        --exclude='build' \
        --exclude='dist' \
        -C "$(dirname "$NPP_RL_DIR")" \
        "$(basename "$NPP_RL_DIR")" 2>&1 | grep -v "Removing leading" || true
    
    # Copy tarball and extract
    scp_to_remote "/tmp/npp-rl-${TIMESTAMP}.tar.gz" "${REMOTE_BASE_DIR}/"
    ssh_cmd "cd ${REMOTE_BASE_DIR} && tar -xzf npp-rl-${TIMESTAMP}.tar.gz && rm npp-rl-${TIMESTAMP}.tar.gz"
    rm /tmp/npp-rl-${TIMESTAMP}.tar.gz
    
    log SUCCESS "npp-rl copied successfully"
    
    # Copy nclone
    log INFO "Copying nclone to remote..."
    
    tar -czf /tmp/nclone-${TIMESTAMP}.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.so' \
        --exclude='*.so.*' \
        --exclude='*.pyd' \
        --exclude='*.dll' \
        --exclude='*.egg-info' \
        --exclude='build' \
        --exclude='dist' \
        -C "$(dirname "$NCLONE_DIR")" \
        "$(basename "$NCLONE_DIR")" 2>&1 | grep -v "Removing leading" || true
    
    scp_to_remote "/tmp/nclone-${TIMESTAMP}.tar.gz" "${REMOTE_BASE_DIR}/"
    ssh_cmd "cd ${REMOTE_BASE_DIR} && tar -xzf nclone-${TIMESTAMP}.tar.gz && rm nclone-${TIMESTAMP}.tar.gz"
    rm /tmp/nclone-${TIMESTAMP}.tar.gz
    
    log SUCCESS "nclone copied successfully"
    
    # Install system dependencies required for Python packages
    if ! install_system_dependencies; then
        log ERROR "Failed to install system dependencies"
        return 1
    fi
    
    # Install packages in development mode
    # Using system ML packages where compatible (PyTorch, etc.) but enforcing NumPy 1.x
    log INFO "Installing packages on remote (using system ML packages with NumPy 1.x)..."
    
    # Verify pip is available
    if ! ssh_cmd "python3 -m pip --version" > /dev/null 2>&1; then
        log INFO "Installing pip..."
        ssh_cmd "curl -sS https://bootstrap.pypa.io/get-pip.py | python3" > /dev/null 2>&1 || {
            log ERROR "Failed to install pip"
            return 1
        }
    fi
    
    # Upgrade pip, setuptools, and wheel
    log INFO "Upgrading pip, setuptools, and wheel..."
    ssh_cmd "python3 -m pip install --user --upgrade pip setuptools>=64 wheel" > /dev/null 2>&1 || {
        log ERROR "Failed to upgrade pip/setuptools"
        return 1
    }
    
    # Install NumPy 1.x for compatibility (must be done before other packages)
    log INFO "Installing NumPy 1.x for compatibility with system packages..."
    ssh_cmd "python3 -m pip install --user 'numpy>=1.24.0,<2.0.0'" > /dev/null 2>&1 || {
        log ERROR "Failed to install NumPy 1.x"
        return 1
    }
    log SUCCESS "NumPy 1.x installed successfully"
    
    # Install nclone with all dependencies (will use system PyTorch/NumPy where compatible)
    log INFO "Installing nclone..."
    ssh_cmd "cd ${REMOTE_NCLONE_DIR} && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NCLONE=0.1.0 python3 -m pip install --user -e ." || {
        log ERROR "nclone installation failed"
        return 1
    }
    log SUCCESS "nclone installed successfully"
    
    # Install npp-rl with all dependencies (will use system PyTorch/NumPy where compatible)
    log INFO "Installing npp-rl..."
    ssh_cmd "cd ${REMOTE_NPP_RL_DIR} && python3 -m pip install --user -e ." || {
        log ERROR "npp-rl installation failed"
        return 1
    }
    log SUCCESS "npp-rl installed successfully"
    
    log SUCCESS "All packages installed successfully (using system ML packages with NumPy 1.x for compatibility)"
}

setup_remote_environment() {
    log HEADER "Setting up remote environment"
    
    # Set AWS credentials
    log INFO "Setting AWS credentials on remote instance..."
    ssh_cmd "mkdir -p ~/.aws"
    ssh_cmd "cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${AWS_ACCESS_KEY}
aws_secret_access_key = ${AWS_SECRET_KEY}
EOF"
    log SUCCESS "AWS credentials configured"
    
    # Create experiment directory
    log INFO "Creating remote experiment directory..."
    ssh_cmd "mkdir -p ~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"
    log SUCCESS "Created remote directory: ~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"
}

