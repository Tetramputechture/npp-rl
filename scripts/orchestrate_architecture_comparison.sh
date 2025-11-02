#!/bin/bash
###############################################################################
# Architecture Comparison Orchestration Script
#
# Automates end-to-end architecture comparison on a remote GPU training instance:
# - Validates Python 3.11+ and installs compatible ML packages (PyTorch, NumPy)
# - Validates PyTorch CUDA compatibility and provides troubleshooting
# - Copies npp-rl and nclone code to remote instance via SCP
# - Dataset generation (500 levels: train + test)
# - Sequential training of 10 architectures (1M timesteps each)
# - Real-time TensorBoard monitoring via SSH port forwarding
# - Automatic S3 artifact uploads (checkpoints, videos, logs)
# - Comprehensive logging and error handling
#
# Prerequisites:
#   - Run this script from the npp-rl directory
#   - nclone directory must be in the same parent directory as npp-rl
#   - Remote instance must have Python 3.11+ pre-installed
#   - Remote instance should have build tools (or sudo access to install):
#     * build-essential, python3-dev, libcairo2-dev, pkg-config
#   - GPU with CUDA support for accelerated training
#
# Optional (will be installed via pip if missing or incompatible):
#   - System ML packages: python3-torch-cuda, python3-tensorflow-cuda
#   - CUDA toolkit: nvidia-cuda-toolkit, libnccl2
#
# Note: Script automatically handles TensorBoard compatibility issues by
# uninstalling system packages and reinstalling from pip if needed
#
# Usage:
#   cd /path/to/npp-rl
#   ./scripts/orchestrate_architecture_comparison.sh \
#     --instance-ip 54.123.45.67 \
#     --aws-access-key AKIA... \
#     --aws-secret-key wJalr... \
#     --s3-bucket npp-rl-experiments \
#     --experiment-name arch_comparison_2025
#
# Optional:
#   --ssh-key ~/.ssh/my_key.pem
#   --ssh-user ubuntu
#   --resume  # Skip already completed architectures
#
###############################################################################

set -e
set -o pipefail

# ============================================================================
# Get script directory for sourcing library modules
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"

# ============================================================================
# Source all library modules
# ============================================================================
source "${LIB_DIR}/config.sh"
source "${LIB_DIR}/logging.sh"
source "${LIB_DIR}/arguments.sh"
source "${LIB_DIR}/ssh.sh"
source "${LIB_DIR}/validation.sh"
source "${LIB_DIR}/setup.sh"
source "${LIB_DIR}/environment.sh"
source "${LIB_DIR}/training.sh"
source "${LIB_DIR}/reporting.sh"

# Trap signals for graceful cleanup
trap cleanup EXIT INT TERM

# ============================================================================
# Main execution
# ============================================================================
main() {
    # Create local log directory first (before any log calls that use tee)
    mkdir -p "$LOCAL_LOG_DIR"
    
    log HEADER "NPP-RL Architecture Comparison Orchestration"
    
    # Parse arguments
    parse_args "$@"
    
    log INFO "Configuration:"
    log INFO "  Instance: ${INSTANCE_IP}"
    log INFO "  Experiment: ${EXPERIMENT_NAME}"
    log INFO "  S3 Bucket: ${S3_BUCKET}"
    log INFO "  SSH Key: ${SSH_KEY}"
    log INFO "  SSH User: ${SSH_USER}"
    log INFO "  Resume Mode: ${RESUME}"
    log INFO "  CUDA Home: ${DETECTED_CUDA_HOME}"
    echo ""
    
    # Validate local environment first
    if ! validate_local_environment; then
        log ERROR "Local environment validation failed"
        exit 1
    fi
    
    # Setup local environment
    setup_local_environment
    
    # Validate SSH connection
    if ! validate_ssh_connection; then
        log ERROR "Cannot proceed without SSH connection"
        exit 1
    fi
    
    # Validate remote dependencies
    if ! validate_remote_dependencies; then
        log ERROR "Remote environment validation failed"
        exit 1
    fi
    
    # Copy code to remote instance
    copy_code_to_remote
    
    # Fix NumPy compatibility issues (must be done before verifying other packages)
    if ! fix_numpy_compatibility; then
        log WARNING "NumPy compatibility fix failed, but continuing..."
        log WARNING "You may encounter import errors during training"
        # Don't exit - we can try to continue
    fi
    
    # Verify PyTorch CUDA compatibility
    if ! verify_pytorch_cuda_compatibility; then
        log ERROR "PyTorch CUDA verification failed"
        log ERROR "Training cannot proceed without GPU access on GPU instance"
        exit 1
    fi
    
    # Verify TensorBoard installation
    if ! verify_tensorboard_installation; then
        log ERROR "TensorBoard verification failed"
        log ERROR "Training will proceed but TensorBoard logging may not work"
        # Don't exit - we can continue without TensorBoard
    fi
    
    # Setup TensorBoard forwarding
    setup_tensorboard_forwarding
    
    # Setup remote environment (AWS credentials, experiment dirs)
    setup_remote_environment
    
    # Generate datasets
    if ! generate_datasets; then
        log ERROR "Dataset generation failed"
        exit 1
    fi
    
    # Train all architectures
    log HEADER "Starting Architecture Training Loop"
    log INFO "Training ${#ARCHITECTURES[@]} architectures sequentially (1M timesteps each)"
    log INFO "Estimated total time: 11-22 hours (depending on hardware)"
    echo ""
    
    local arch_num=1
    for arch in "${ARCHITECTURES[@]}"; do
        if ! train_single_architecture "$arch" $arch_num ${#ARCHITECTURES[@]}; then
            log WARNING "Architecture ${arch} failed, continuing with next..."
        fi
        arch_num=$((arch_num + 1))
        
        # Brief pause between architectures
        sleep 5
    done
    
    # Download final comparison results
    log INFO "Downloading final comparison results..."
    mkdir -p "${LOCAL_LOG_DIR}/final_results"
    scp_from_remote "~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/*" "${LOCAL_LOG_DIR}/final_results/" 2>/dev/null || true
    
    # Save and print summary
    save_summary
    print_final_summary
    
    log SUCCESS "Orchestration complete!"
}

# ============================================================================
# Script entry point
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
