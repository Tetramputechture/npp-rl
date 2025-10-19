#!/bin/bash
###############################################################################
# Logging and Cleanup Functions
# 
# This module contains logging functions with color-coded output and cleanup
# functions for graceful shutdown.
###############################################################################

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

