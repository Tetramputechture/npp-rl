#!/bin/bash
###############################################################################
# Configuration and Constants
# 
# This module contains all global configuration variables, constants, and 
# default settings used throughout the orchestration scripts.
###############################################################################

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
# Global variables (will be set by argument parsing or defaults)
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
NCLONE_DIR="$(pwd)/../nclone"

# Remote paths
REMOTE_BASE_DIR="~/npp-rl-training"
REMOTE_NPP_RL_DIR="${REMOTE_BASE_DIR}/npp-rl"
REMOTE_NCLONE_DIR="${REMOTE_BASE_DIR}/nclone"

# CUDA environment (detected during validation)
DETECTED_CUDA_HOME="/usr/local/cuda"

# Architectures to compare
# Our HGT based archiectures are way too slow for now, so we're only comparing the other architectures.
ARCHITECTURES=(
    # "full_hgt"
    # "simplified_hgt"
    # "gat"
    # "gcn"
    "mlp_baseline"
    # "vision_free"
    # "vision_free_gat"
    # "vision_free_gcn"
    # "vision_free_simplified"
    # "no_global_view"
)

COMPLETED_ARCHITECTURES=()
FAILED_ARCHITECTURES=()

# Flag to track if CUDA environment is configured
CUDA_ENV_CONFIGURED=false

