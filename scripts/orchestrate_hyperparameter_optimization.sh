#!/bin/bash
###############################################################################
# Hyperparameter Optimization Orchestration Script
#
# Automates end-to-end hyperparameter optimization on a remote GPU training instance:
# - Validates Python 3.11+ and installs compatible ML packages (PyTorch, NumPy)
# - Validates PyTorch CUDA compatibility and provides troubleshooting
# - Copies npp-rl and nclone code to remote instance via SCP
# - Dataset generation (if needed)
# - Optuna-based hyperparameter optimization
# - Real-time TensorBoard and Optuna dashboard monitoring via SSH port forwarding
# - Automatic S3 artifact uploads (checkpoints, study database, logs)
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
# Usage:
#   cd /path/to/npp-rl
#   ./scripts/orchestrate_hyperparameter_optimization.sh \
#     --instance-ip 54.123.45.67 \
#     --aws-access-key AKIA... \
#     --aws-secret-key wJalr... \
#     --s3-bucket npp-rl-experiments \
#     --experiment-name hpo_mlp_v1 \
#     --architecture mlp_cnn
#
# Optional:
#   --ssh-key ~/.ssh/my_key.pem
#   --ssh-user ubuntu
#   --num-trials 20
#   --timesteps-per-trial 500000
#   --resume  # Resume from existing study
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
source "${LIB_DIR}/optuna.sh"  # NEW

# ============================================================================
# HPO-specific argument parsing
# ============================================================================
parse_hpo_args() {
    # Defaults
    ARCHITECTURE=""
    NUM_TRIALS=20
    TIMESTEPS_PER_TRIAL=1000000
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --instance-ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            --aws-access-key)
                AWS_ACCESS_KEY="$2"
                shift 2
                ;;
            --aws-secret-key)
                AWS_SECRET_KEY="$2"
                shift 2
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --experiment-name)
                EXPERIMENT_NAME="$2"
                shift 2
                ;;
            --architecture)
                ARCHITECTURE="$2"
                shift 2
                ;;
            --num-trials)
                NUM_TRIALS="$2"
                shift 2
                ;;
            --timesteps-per-trial)
                TIMESTEPS_PER_TRIAL="$2"
                shift 2
                ;;
            --ssh-key)
                SSH_KEY="$2"
                shift 2
                ;;
            --ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --help)
                cat << EOF
Usage: $0 [OPTIONS]

Required:
  --instance-ip IP          Remote GPU instance IP address
  --aws-access-key KEY      AWS access key ID
  --aws-secret-key SECRET   AWS secret access key
  --s3-bucket BUCKET        S3 bucket for artifacts
  --experiment-name NAME    Unique experiment identifier
  --architecture ARCH       Architecture name to optimize

Optional:
  --num-trials N            Number of Optuna trials (default: 20)
  --timesteps-per-trial N   Training timesteps per trial (default: 500000)
  --ssh-key PATH            Path to SSH private key (default: ~/.ssh/id_rsa)
  --ssh-user USER           SSH username (default: ubuntu)
  --resume                  Resume from existing study
  --help                    Show this help message

Example:
  cd /path/to/npp-rl
  $0 \\
    --instance-ip 54.123.45.67 \\
    --aws-access-key AKIA... \\
    --aws-secret-key wJalr... \\
    --s3-bucket npp-rl-experiments \\
    --experiment-name hpo_mlp_v1 \\
    --architecture mlp_cnn \\
    --num-trials 20 \\
    --timesteps-per-trial 500000

EOF
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$INSTANCE_IP" ]] || [[ -z "$AWS_ACCESS_KEY" ]] || [[ -z "$AWS_SECRET_KEY" ]] || \
       [[ -z "$S3_BUCKET" ]] || [[ -z "$EXPERIMENT_NAME" ]] || [[ -z "$ARCHITECTURE" ]]; then
        echo "Error: Missing required arguments"
        echo "Use --help for usage information"
        exit 1
    fi
    
    # Validate SSH key exists
    if [[ ! -f "$SSH_KEY" ]]; then
        echo "Error: SSH key not found at $SSH_KEY"
        exit 1
    fi
    
    # Update LOCAL_LOG_DIR for HPO and create the directory
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOCAL_LOG_DIR="./logs/hpo_${EXPERIMENT_NAME}_${TIMESTAMP}"
    mkdir -p "$LOCAL_LOG_DIR"
}

# ============================================================================
# Cleanup function
# ============================================================================
cleanup() {
    log INFO "Cleaning up..."
    
    # Kill port forwarding processes
    if [ -n "$TENSORBOARD_PID" ]; then
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
    
    if [ -n "$OPTUNA_DASHBOARD_PID" ]; then
        kill $OPTUNA_DASHBOARD_PID 2>/dev/null || true
    fi
    
    log INFO "Cleanup complete"
}

# Trap signals for graceful cleanup
trap cleanup EXIT INT TERM

# ============================================================================
# Main execution
# ============================================================================
main() {
    # Create local log directory first (before any log calls that use tee)
    mkdir -p "$LOCAL_LOG_DIR"
    
    log HEADER "NPP-RL Hyperparameter Optimization Orchestration"
    
    # Parse arguments
    parse_hpo_args "$@"
    
    log INFO "Configuration:"
    log INFO "  Instance: ${INSTANCE_IP}"
    log INFO "  Experiment: ${EXPERIMENT_NAME}"
    log INFO "  Architecture: ${ARCHITECTURE}"
    log INFO "  S3 Bucket: ${S3_BUCKET}"
    log INFO "  SSH Key: ${SSH_KEY}"
    log INFO "  SSH User: ${SSH_USER}"
    log INFO "  Num Trials: ${NUM_TRIALS}"
    log INFO "  Timesteps per Trial: ${TIMESTEPS_PER_TRIAL}"
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
        log WARNING "TensorBoard verification failed"
        log WARNING "Training will proceed but TensorBoard logging may not work"
        # Don't exit - we can continue without TensorBoard
    fi
    
    # Setup Optuna environment
    if ! setup_optuna_environment; then
        log ERROR "Optuna setup failed"
        exit 1
    fi
    
    if ! verify_optuna_installation; then
        log ERROR "Optuna verification failed"
        exit 1
    fi
    
    # Setup TensorBoard forwarding
    setup_tensorboard_forwarding
    
    # Setup remote environment (AWS credentials, experiment dirs)
    setup_remote_environment
    
    # Generate datasets (if needed)
    if ! generate_datasets; then
        log ERROR "Dataset generation failed"
        exit 1
    fi
    
    # Get absolute path for study storage (expand ~ on remote)
    local remote_study_dir_abs=$(ssh_cmd "echo ~/hpo_results/${EXPERIMENT_NAME}" | tr -d '\r\n')
    local study_storage="sqlite:///${remote_study_dir_abs}/optuna_study.db"
    
    log INFO "Study storage path: ${study_storage}"
    
    # Start Optuna dashboard
    start_optuna_dashboard "${study_storage}"
    
    # Run hyperparameter optimization
    log HEADER "Starting Hyperparameter Optimization"
    log INFO "Running ${NUM_TRIALS} trials with ${TIMESTEPS_PER_TRIAL} timesteps each"
    log INFO "Estimated time: 10-20 hours (depending on hardware and pruning)"
    echo ""
    
    if ! run_hyperparameter_optimization \
        "${ARCHITECTURE}" \
        "${EXPERIMENT_NAME}" \
        "${NUM_TRIALS}" \
        "${TIMESTEPS_PER_TRIAL}"; then
        log ERROR "Hyperparameter optimization failed"
        exit 1
    fi
    
    # Download optimization results
    download_optimization_results "${EXPERIMENT_NAME}" "${ARCHITECTURE}"
    
    # Print summary
    print_hpo_summary "${EXPERIMENT_NAME}" "${ARCHITECTURE}"
    
    log SUCCESS "Hyperparameter optimization orchestration complete!"
}

# ============================================================================
# Script entry point
# ============================================================================
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

