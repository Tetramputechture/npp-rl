#!/bin/bash
###############################################################################
# Argument Parsing Functions
# 
# This module contains functions for parsing command-line arguments and
# displaying usage information.
###############################################################################

# ============================================================================
# Argument parsing
# ============================================================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Prerequisites:
  - Run this script from the npp-rl directory
  - nclone directory must be in ../nclone (same parent as npp-rl)
  - Remote instance must have Python 3.11+, pip, and SSH access
  - Remote instance must have GPU with CUDA drivers
  - Remote instance should have build tools (or sudo to install):
    build-essential, python3-dev, libcairo2-dev, pkg-config
  
  Optional (will be installed via pip if missing or incompatible):
  - System ML packages (PyTorch, TensorFlow, JAX with CUDA)
  - nvidia-cuda-toolkit

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

