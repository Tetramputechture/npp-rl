#!/bin/bash
###############################################################################
# Example: Architecture Comparison Orchestration
#
# This is a template script showing how to run the orchestration.
# Copy this file and fill in your actual values.
###############################################################################

# === CONFIGURATION - FILL IN YOUR VALUES ===

# Remote GPU instance IP address
INSTANCE_IP="54.123.45.67"  # Replace with your instance IP

# AWS credentials for S3 uploads
AWS_ACCESS_KEY="AKIAIOSFODNN7EXAMPLE"  # Replace with your AWS access key
AWS_SECRET_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # Replace with your AWS secret

# S3 bucket for storing artifacts
S3_BUCKET="npp-rl-experiments"  # Replace with your S3 bucket name

# Experiment name (will be timestamped automatically)
EXPERIMENT_NAME="arch_comparison_$(date +%Y%m%d)"

# SSH configuration (optional, adjust if needed)
SSH_KEY="${HOME}/.ssh/id_rsa"  # Path to your SSH private key
SSH_USER="ubuntu"  # SSH username for the instance

# === END CONFIGURATION ===

# Color output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  NPP-RL Architecture Comparison Orchestration             ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Instance IP:     ${INSTANCE_IP}"
echo "  S3 Bucket:       ${S3_BUCKET}"
echo "  Experiment:      ${EXPERIMENT_NAME}"
echo "  SSH Key:         ${SSH_KEY}"
echo "  SSH User:        ${SSH_USER}"
echo ""
echo -e "${GREEN}Estimated time: 11-22 hours${NC}"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Run the orchestration script
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip "${INSTANCE_IP}" \
  --aws-access-key "${AWS_ACCESS_KEY}" \
  --aws-secret-key "${AWS_SECRET_KEY}" \
  --s3-bucket "${S3_BUCKET}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --ssh-key "${SSH_KEY}" \
  --ssh-user "${SSH_USER}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Orchestration Complete!                                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Results available at:"
    echo "  Local:  ./logs/arch_comparison_*/"
    echo "  S3:     s3://${S3_BUCKET}/experiments/${EXPERIMENT_NAME}_*/"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: cat ./logs/arch_comparison_*/experiment_summary.json | jq"
    echo "  2. View TensorBoard: tensorboard --logdir=./logs/arch_comparison_*/results/"
    echo "  3. Download videos: aws s3 sync s3://${S3_BUCKET}/experiments/${EXPERIMENT_NAME}_*/*/videos/ ./videos/"
else
    echo ""
    echo -e "${RED}Orchestration failed. Check logs in ./logs/arch_comparison_*/${NC}"
    exit 1
fi

