#!/bin/bash
###############################################################################
# Optuna-specific Orchestration Functions
# 
# This module contains functions for setting up and running Optuna
# hyperparameter optimization on remote instances.
###############################################################################

# ============================================================================
# Optuna environment setup
# ============================================================================
setup_optuna_environment() {
    log INFO "Setting up Optuna environment on remote instance..."
    
    # Install Optuna and visualization dependencies
    ssh_cmd "pip install optuna optuna-dashboard plotly kaleido" 2>&1 | tee -a "${LOCAL_LOG_DIR}/optuna_setup.log"
    
    if [ $? -ne 0 ]; then
        log ERROR "Optuna installation failed"
        return 1
    fi
    
    log SUCCESS "Optuna environment setup complete"
    return 0
}

verify_optuna_installation() {
    log INFO "Verifying Optuna installation..."
    
    ssh_cmd "python -c 'import optuna; print(f\"Optuna version: {optuna.__version__}\")'" 2>&1 | tee -a "${LOCAL_LOG_DIR}/optuna_verify.log"
    
    if [ $? -eq 0 ]; then
        log SUCCESS "Optuna is properly installed"
        return 0
    else
        log ERROR "Optuna verification failed"
        return 1
    fi
}

# ============================================================================
# Optuna dashboard
# ============================================================================
start_optuna_dashboard() {
    local study_storage=$1  # e.g., "sqlite:////home/ubuntu/hpo_results/optuna_study.db" (must be absolute path)
    
    log INFO "Starting Optuna dashboard with SSH port forwarding..."
    
    local local_port=8080
    local remote_port=8080
    
    # Start dashboard on remote (in background)
    ssh_cmd "nohup optuna-dashboard ${study_storage} --port ${remote_port} > ~/optuna_dashboard.log 2>&1 &" 2>&1 | tee -a "${LOCAL_LOG_DIR}/optuna_dashboard_start.log"
    
    # Brief pause to let dashboard start
    sleep 2
    
    # Setup SSH port forwarding (store PID in global variable)
    ssh -f -N -L ${local_port}:localhost:${remote_port} \
        -i "${SSH_KEY}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR \
        ${SSH_USER}@${INSTANCE_IP} \
        2>&1 | tee -a "${LOCAL_LOG_DIR}/optuna_dashboard_forward.log" &
    
    export OPTUNA_DASHBOARD_PID=$!
    
    log SUCCESS "Optuna dashboard available at: http://localhost:${local_port}"
    log INFO "Dashboard PID: ${OPTUNA_DASHBOARD_PID}"
}

# ============================================================================
# Hyperparameter optimization execution
# ============================================================================
run_hyperparameter_optimization() {
    local architecture=$1
    local experiment_name=$2
    local num_trials=$3
    local timesteps_per_trial=$4
    
    log HEADER "Running Hyperparameter Optimization"
    log INFO "Architecture: ${architecture}"
    log INFO "Trials: ${num_trials}"
    log INFO "Timesteps per trial: ${timesteps_per_trial}"
    
    local study_name="${experiment_name}_${architecture}"
    local remote_study_dir="~/hpo_results/${experiment_name}"
    
    # Get absolute path for remote study directory (expand ~)
    local remote_study_dir_abs=$(ssh_cmd "echo ${remote_study_dir}" | tr -d '\r\n')
    local remote_storage="sqlite:///${remote_study_dir_abs}/optuna_study.db"
    
    log INFO "Study storage: ${remote_storage}"
    
    # Create remote study directory
    ssh_cmd "mkdir -p ${remote_study_dir}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/hpo_setup.log"
    
    local hpo_cmd="cd ${REMOTE_NPP_RL_DIR} && \
        export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} && \
        export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_KEY} && \
        export CUDA_HOME=${DETECTED_CUDA_HOME} && \
        export CUDA_PATH=${DETECTED_CUDA_HOME} && \
        export LD_LIBRARY_PATH=${DETECTED_CUDA_HOME}/lib64:\${LD_LIBRARY_PATH} && \
        export PATH=${DETECTED_CUDA_HOME}/bin:\${PATH} && \
        python scripts/optuna_hyperparameter_optimization.py \
            --experiment-name ${experiment_name} \
            --architecture ${architecture} \
            --train-dataset ~/datasets/train \
            --test-dataset ~/datasets/test \
            --num-trials ${num_trials} \
            --timesteps-per-trial ${timesteps_per_trial} \
            --study-name ${study_name}_$(date +%Y%m%d_%H%M%S) \
            --storage ${remote_storage} \
            --output-dir ~/hpo_results/${experiment_name} \
            --s3-bucket ${S3_BUCKET} \
            --s3-prefix hpo/${experiment_name}/"
    
    # Add --resume flag if requested
    if [ "$RESUME" = true ]; then
        hpo_cmd="${hpo_cmd} --resume"
    fi
    
    # Just print the command
    echo "${hpo_cmd}"
    # Run optimization (blocking)
    # ssh_cmd_cuda "${hpo_cmd}" 2>&1 | tee -a "${LOCAL_LOG_DIR}/hpo_training.log"
    
    if [ $? -eq 0 ]; then
        log SUCCESS "Hyperparameter optimization completed successfully"
        return 0
    else
        log ERROR "Hyperparameter optimization failed"
        return 1
    fi
}

# ============================================================================
# Result downloading
# ============================================================================
download_optimization_results() {
    local experiment_name=$1
    local architecture=$2
    
    log INFO "Downloading optimization results..."
    
    local remote_results_dir="~/hpo_results/${experiment_name}"
    local local_results_dir="${LOCAL_LOG_DIR}/hpo_results"
    
    mkdir -p "${local_results_dir}"
    mkdir -p "${local_results_dir}/plots"
    mkdir -p "${local_results_dir}/best_checkpoint"
    
    # Download study database
    scp_from_remote "${remote_results_dir}/optuna_study.db" "${local_results_dir}/" 2>/dev/null || log WARNING "Could not download study database"
    
    # Download best hyperparameters JSON
    scp_from_remote "${remote_results_dir}/best_hyperparameters_${architecture}.json" "${local_results_dir}/" 2>/dev/null || log WARNING "Could not download best hyperparameters"
    
    # Download visualization plots
    scp_from_remote "${remote_results_dir}/plots/*" "${local_results_dir}/plots/" 2>/dev/null || log WARNING "Could not download plots"
    
    # Download best model checkpoint (if exists)
    scp_from_remote "${remote_results_dir}/best_trial/checkpoints/*" "${local_results_dir}/best_checkpoint/" 2>/dev/null || log WARNING "Could not download best checkpoint"
    
    log SUCCESS "Results downloaded to: ${local_results_dir}"
}

# ============================================================================
# Summary generation
# ============================================================================
print_hpo_summary() {
    local experiment_name=$1
    local architecture=$2
    
    log HEADER "Hyperparameter Optimization Summary"
    
    local local_results_dir="${LOCAL_LOG_DIR}/hpo_results"
    local best_params_file="${local_results_dir}/best_hyperparameters_${architecture}.json"
    
    if [ -f "${best_params_file}" ]; then
        log INFO "Best hyperparameters found:"
        cat "${best_params_file}" | python3 -m json.tool 2>/dev/null || cat "${best_params_file}"
    else
        log WARNING "Best hyperparameters file not found"
    fi
    
    log INFO "Results directory: ${local_results_dir}"
    log INFO "Study database: ${local_results_dir}/optuna_study.db"
    log INFO "Visualization plots: ${local_results_dir}/plots/"
}

