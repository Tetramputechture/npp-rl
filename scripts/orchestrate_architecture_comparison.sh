#!/bin/bash
###############################################################################
# Architecture Comparison Orchestration Script
#
# Automates end-to-end architecture comparison on a remote GPU training instance:
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
#   - Remote instance must have Python 3, pip, and optionally CUDA/GPU
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
# Color codes for output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Global variables
# ============================================================================
INSTANCE_IP=""
AWS_ACCESS_KEY=""
AWS_SECRET_KEY=""
S3_BUCKET=""
EXPERIMENT_NAME=""
SSH_KEY="${HOME}/.ssh/id_rsa"
SSH_USER="ubuntu"
RESUME=false

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_LOG_DIR="./logs/arch_comparison_${TIMESTAMP}"
TENSORBOARD_PORT=6006
TENSORBOARD_PID=""
LOG_TAIL_PID=""

# Local paths (script runs from npp-rl directory)
NPP_RL_DIR="$(pwd)"
NCLONE_DIR="$(dirname "$NPP_RL_DIR")/../nclone"

# Remote paths
REMOTE_BASE_DIR="~/npp-rl-training"
REMOTE_NPP_RL_DIR="${REMOTE_BASE_DIR}/npp-rl"
REMOTE_NCLONE_DIR="${REMOTE_BASE_DIR}/nclone"

# All 11 architectures to compare
ARCHITECTURES=(
    "full_hgt"
    "simplified_hgt"
    "gat"
    "gcn"
    "mlp_baseline"
    "vision_free"
    "vision_free_gat"
    "vision_free_gcn"
    "vision_free_simplified"
    "no_global_view"
    "local_frames_only"
)

COMPLETED_ARCHITECTURES=()
FAILED_ARCHITECTURES=()

# ============================================================================
# Logging functions
# ============================================================================
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)
            echo -e "${BLUE}[${timestamp}]${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] ✓${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        WARNING)
            echo -e "${YELLOW}[${timestamp}] ⚠${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] ✗${NC} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        HEADER)
            echo -e "\n${CYAN}========================================${NC}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            echo -e "${CYAN}${message}${NC}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            echo -e "${CYAN}========================================${NC}\n" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
        *)
            echo -e "${timestamp} ${message}" | tee -a "${LOCAL_LOG_DIR}/orchestration.log"
            ;;
    esac
}

# ============================================================================
# Cleanup function
# ============================================================================
cleanup() {
    log INFO "Cleaning up..."
    
    # Kill TensorBoard port forwarding
    if [[ -n "$TENSORBOARD_PID" ]] && ps -p $TENSORBOARD_PID > /dev/null 2>&1; then
        log INFO "Closing TensorBoard port forwarding (PID: $TENSORBOARD_PID)"
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
    
    # Kill log tail process
    if [[ -n "$LOG_TAIL_PID" ]] && ps -p $LOG_TAIL_PID > /dev/null 2>&1; then
        kill $LOG_TAIL_PID 2>/dev/null || true
    fi
    
    # Save partial results summary
    save_summary
    
    log SUCCESS "Cleanup complete"
}

# Trap signals for graceful cleanup
trap cleanup EXIT INT TERM

# ============================================================================
# Argument parsing
# ============================================================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Prerequisites:
  - Run this script from the npp-rl directory
  - nclone directory must be in ../nclone (same parent as npp-rl)
  - Remote instance must have Python 3, pip, and SSH access

Required:
  --instance-ip IP          Remote GPU instance IP address
  --aws-access-key KEY      AWS access key ID
  --aws-secret-key SECRET   AWS secret access key
  --s3-bucket BUCKET        S3 bucket for artifacts
  --experiment-name NAME    Unique experiment identifier

Optional:
  --ssh-key PATH            Path to SSH private key (default: ~/.ssh/id_rsa)
  --ssh-user USER           SSH username (default: ubuntu)
  --resume                  Skip already completed architectures
  --help                    Show this help message

Example:
  cd /path/to/npp-rl
  $0 \\
    --instance-ip 54.123.45.67 \\
    --aws-access-key AKIA... \\
    --aws-secret-key wJalr... \\
    --s3-bucket npp-rl-experiments \\
    --experiment-name arch_comparison_2025

EOF
    exit 1
}

parse_args() {
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
                usage
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$INSTANCE_IP" ]] || [[ -z "$AWS_ACCESS_KEY" ]] || [[ -z "$AWS_SECRET_KEY" ]] || \
       [[ -z "$S3_BUCKET" ]] || [[ -z "$EXPERIMENT_NAME" ]]; then
        echo "Error: Missing required arguments"
        usage
    fi
    
    # Validate SSH key exists
    if [[ ! -f "$SSH_KEY" ]]; then
        echo "Error: SSH key not found at $SSH_KEY"
        exit 1
    fi
}

# ============================================================================
# SSH helper functions
# ============================================================================
ssh_cmd() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR "${SSH_USER}@${INSTANCE_IP}" "$@"
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

# ============================================================================
# Validation functions
# ============================================================================
validate_local_environment() {
    log INFO "Validating local environment..."
    
    # Check if we're in npp-rl directory
    if [[ ! -f "setup.py" ]] || [[ ! -d "npp_rl" ]]; then
        log ERROR "This script must be run from the npp-rl directory"
        log ERROR "Current directory: $(pwd)"
        return 1
    fi
    
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
    
    if ssh_cmd "echo 'SSH connection successful'" > /dev/null 2>&1; then
        log SUCCESS "SSH connection established"
        return 0
    else
        log ERROR "Failed to establish SSH connection"
        return 1
    fi
}

validate_remote_dependencies() {
    log INFO "Checking remote dependencies..."
    
    # Check if Python is available
    if ! ssh_cmd "which python3" > /dev/null 2>&1; then
        log ERROR "Python3 not found on remote instance"
        return 1
    fi
    
    # Check for GPU (optional warning)
    if ssh_cmd "which nvidia-smi" > /dev/null 2>&1; then
        log SUCCESS "GPU detected on remote instance"
        ssh_cmd "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" | while read line; do
            log INFO "  GPU: $line"
        done
    else
        log WARNING "nvidia-smi not found. Training will use CPU (slow)"
    fi
    
    log SUCCESS "Remote dependencies validated"
    return 0
}

# ============================================================================
# Setup functions
# ============================================================================
setup_local_environment() {
    log HEADER "Setting up local environment"
    
    # Create local log directory
    mkdir -p "$LOCAL_LOG_DIR"
    log SUCCESS "Created local log directory: ${LOCAL_LOG_DIR}"
    
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
    log INFO "  Excluding: .git, __pycache__, *.pyc, experiments, logs, datasets"
    
    # Create tarball excluding unnecessary files
    tar -czf /tmp/npp-rl-${TIMESTAMP}.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
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
        --exclude='*.egg-info' \
        --exclude='build' \
        --exclude='dist' \
        -C "$(dirname "$NCLONE_DIR")" \
        "$(basename "$NCLONE_DIR")" 2>&1 | grep -v "Removing leading" || true
    
    scp_to_remote "/tmp/nclone-${TIMESTAMP}.tar.gz" "${REMOTE_BASE_DIR}/"
    ssh_cmd "cd ${REMOTE_BASE_DIR} && tar -xzf nclone-${TIMESTAMP}.tar.gz && rm nclone-${TIMESTAMP}.tar.gz"
    rm /tmp/nclone-${TIMESTAMP}.tar.gz
    
    log SUCCESS "nclone copied successfully"
    
    # Install packages in development mode
    log INFO "Installing packages on remote..."
    ssh_cmd "cd ${REMOTE_NCLONE_DIR} && pip install -q -e ." || {
        log WARNING "nclone installation had warnings, continuing..."
    }
    ssh_cmd "cd ${REMOTE_NPP_RL_DIR} && pip install -q -e ." || {
        log WARNING "npp-rl installation had warnings, continuing..."
    }
    log SUCCESS "Packages installed"
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

# ============================================================================
# Dataset generation
# ============================================================================
generate_datasets() {
    log HEADER "Generating Datasets"
    
    # Check if datasets already exist
    if ssh_cmd "test -d ~/datasets/train && test -d ~/datasets/test"; then
        log WARNING "Datasets already exist at ~/datasets/"
        read -p "Regenerate datasets? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log INFO "Using existing datasets"
            return 0
        fi
    fi
    
    log INFO "Generating 500 levels (250 train + 250 test)..."
    log INFO "This may take 5-15 minutes..."
    
    # Generate datasets
    ssh_cmd "cd ${REMOTE_NCLONE_DIR} && python -m nclone.map_generation.generate_test_suite_maps --mode both --output_dir ~/datasets" 2>&1 | tee -a "${LOCAL_LOG_DIR}/dataset_generation.log"
    
    if ssh_cmd "test -d ~/datasets/train && test -d ~/datasets/test"; then
        log SUCCESS "Datasets generated successfully"
        
        # Verify dataset contents
        local train_count=$(ssh_cmd "find ~/datasets/train -name '*.pkl' | wc -l")
        local test_count=$(ssh_cmd "find ~/datasets/test -name '*.pkl' | wc -l")
        log INFO "Dataset summary: ${train_count} training levels, ${test_count} test levels"
    else
        log ERROR "Dataset generation failed"
        return 1
    fi
}

# ============================================================================
# Training functions
# ============================================================================
check_architecture_completed() {
    local arch=$1
    local remote_result_file="~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/${arch}/eval_results.json"
    
    if ssh_cmd "test -f ${remote_result_file}"; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

train_single_architecture() {
    local arch=$1
    local arch_num=$2
    local total_archs=$3
    
    log HEADER "Training Architecture ${arch_num}/${total_archs}: ${arch}"
    
    # Check if already completed (resume mode)
    if $RESUME && check_architecture_completed "$arch"; then
        log WARNING "Architecture ${arch} already completed, skipping (--resume mode)"
        COMPLETED_ARCHITECTURES+=("$arch")
        return 0
    fi
    
    local start_time=$(date +%s)
    local arch_log="${LOCAL_LOG_DIR}/${arch}_training.log"
    
    log INFO "Starting training for ${arch}..."
    log INFO "Training logs: ${arch_log}"
    
    # Build training command optimized for 8x A100 (80 GB)
    # Use hardware profile for automatic optimization
    local train_cmd="cd ${REMOTE_NPP_RL_DIR} && \
        export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} && \
        export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_KEY} && \
        python scripts/train_and_compare.py \
            --experiment-name ${EXPERIMENT_NAME}_${TIMESTAMP} \
            --architectures ${arch} \
            --train-dataset ~/datasets/train \
            --test-dataset ~/datasets/test \
            --hardware-profile 8xA100-80GB \
            --total-timesteps 10000000 \
            --distributed-backend nccl \
            --record-eval-videos \
            --max-videos-per-category 5 \
            --video-fps 30 \
            --s3-bucket ${S3_BUCKET} \
            --s3-prefix experiments/ \
            --output-dir ~/experiments \
            --no-pretraining"
    
    # Execute training
    if ssh_cmd "$train_cmd" 2>&1 | tee "$arch_log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        log SUCCESS "Architecture ${arch} completed in ${hours}h ${minutes}m"
        COMPLETED_ARCHITECTURES+=("$arch")
        
        # Download results summary
        download_architecture_results "$arch"
        
        return 0
    else
        log ERROR "Architecture ${arch} training failed"
        FAILED_ARCHITECTURES+=("$arch")
        return 1
    fi
}

download_architecture_results() {
    local arch=$1
    local remote_dir="~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/${arch}"
    local local_dir="${LOCAL_LOG_DIR}/results/${arch}"
    
    log INFO "Downloading results for ${arch}..."
    
    mkdir -p "$local_dir"
    
    # Download evaluation results
    if ssh_cmd "test -f ${remote_dir}/eval_results.json"; then
        scp_from_remote "${remote_dir}/eval_results.json" "${local_dir}/" 2>/dev/null || true
    fi
    
    # Download config
    if ssh_cmd "test -f ${remote_dir}/config.json"; then
        scp_from_remote "${remote_dir}/config.json" "${local_dir}/" 2>/dev/null || true
    fi
    
    # Download all_results.json if it exists
    if ssh_cmd "test -f ~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/all_results.json"; then
        scp_from_remote "~/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/all_results.json" "${LOCAL_LOG_DIR}/" 2>/dev/null || true
    fi
    
    log SUCCESS "Results downloaded to ${local_dir}"
}

# ============================================================================
# Summary and reporting
# ============================================================================
save_summary() {
    local summary_file="${LOCAL_LOG_DIR}/experiment_summary.json"
    
    log INFO "Saving experiment summary..."
    
    # Create summary JSON
    cat > "$summary_file" << EOF
{
    "experiment_name": "${EXPERIMENT_NAME}",
    "timestamp": "${TIMESTAMP}",
    "instance_ip": "${INSTANCE_IP}",
    "s3_bucket": "${S3_BUCKET}",
    "total_architectures": ${#ARCHITECTURES[@]},
    "completed_architectures": $(printf '%s\n' "${COMPLETED_ARCHITECTURES[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
    "failed_architectures": $(printf '%s\n' "${FAILED_ARCHITECTURES[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]'),
    "s3_artifacts_prefix": "s3://${S3_BUCKET}/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/",
    "local_logs": "${LOCAL_LOG_DIR}"
}
EOF
    
    log SUCCESS "Summary saved to ${summary_file}"
}

print_final_summary() {
    log HEADER "Experiment Complete"
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ARCHITECTURE COMPARISON COMPLETE${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Experiment: ${YELLOW}${EXPERIMENT_NAME}_${TIMESTAMP}${NC}"
    echo -e "Total Architectures: ${#ARCHITECTURES[@]}"
    echo -e "${GREEN}Completed:${NC} ${#COMPLETED_ARCHITECTURES[@]}"
    echo -e "${RED}Failed:${NC} ${#FAILED_ARCHITECTURES[@]}"
    echo ""
    
    if [[ ${#COMPLETED_ARCHITECTURES[@]} -gt 0 ]]; then
        echo -e "${GREEN}Completed Architectures:${NC}"
        printf '  - %s\n' "${COMPLETED_ARCHITECTURES[@]}"
        echo ""
    fi
    
    if [[ ${#FAILED_ARCHITECTURES[@]} -gt 0 ]]; then
        echo -e "${RED}Failed Architectures:${NC}"
        printf '  - %s\n' "${FAILED_ARCHITECTURES[@]}"
        echo ""
    fi
    
    echo -e "${CYAN}Artifacts:${NC}"
    echo -e "  S3: s3://${S3_BUCKET}/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}/"
    echo -e "  Local logs: ${LOCAL_LOG_DIR}"
    echo ""
    echo -e "${CYAN}TensorBoard Comparison:${NC}"
    echo -e "  tensorboard --logdir=${LOCAL_LOG_DIR}/results/"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ============================================================================
# Main execution
# ============================================================================
main() {
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

