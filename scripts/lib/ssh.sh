#!/bin/bash
###############################################################################
# SSH Helper Functions
# 
# This module contains SSH and SCP wrapper functions for remote command
# execution and file transfers.
###############################################################################

# ============================================================================
# SSH helper functions
# ============================================================================

# Basic ssh command without CUDA environment (used for initial setup)
ssh_cmd_basic() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR "${SSH_USER}@${INSTANCE_IP}" "export PATH=\$HOME/.local/bin:\$PATH; $@"
}

# SSH command that sources CUDA environment if configured
ssh_cmd() {
    if [ "$CUDA_ENV_CONFIGURED" = true ] && ssh_cmd_basic "test -f ~/cuda_env.sh" > /dev/null 2>&1; then
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -o LogLevel=ERROR "${SSH_USER}@${INSTANCE_IP}" "\
export PATH=\$HOME/.local/bin:\$PATH; \
source ~/cuda_env.sh 2>/dev/null || true; \
$@"
    else
        ssh_cmd_basic "$@"
    fi
}

# SSH command with CUDA environment variables explicitly set (for training commands)
ssh_cmd_cuda() {
    local cuda_home="${DETECTED_CUDA_HOME:-/usr/local/cuda}"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR "${SSH_USER}@${INSTANCE_IP}" "\
export PATH=\$HOME/.local/bin:\$PATH; \
source ~/cuda_env.sh 2>/dev/null || true; \
export CUDA_HOME=${cuda_home}; \
export CUDA_PATH=${cuda_home}; \
export PATH=${cuda_home}/bin:\$PATH; \
export LD_LIBRARY_PATH=${cuda_home}/lib64:\${LD_LIBRARY_PATH}; \
$@"
}

scp_from_remote() {
    local remote_path=$1
    local local_path=$2
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -r "${SSH_USER}@${INSTANCE_IP}:${remote_path}" "$local_path"
}

scp_to_remote() {
    local local_path=$1
    local remote_path=$2
    scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -r "$local_path" "${SSH_USER}@${INSTANCE_IP}:${remote_path}"
}

