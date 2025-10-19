#!/bin/bash
###############################################################################
# Summary and Reporting Functions
# 
# This module contains functions for saving experiment summaries and printing
# final results.
###############################################################################

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

