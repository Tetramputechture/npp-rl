#!/bin/bash
###############################################################################
# Training and Dataset Functions
# 
# This module contains functions for dataset generation, architecture training,
# and downloading training results.
###############################################################################

# ============================================================================
# Dataset generation
# ============================================================================
generate_datasets() {
    log HEADER "Generating tile preconnectivity data"
    ssh_cmd "cd ${REMOTE_NCLONE_DIR} && python -m nclone.graph.reachability.tile_connectivity_precomputer" 2>&1 | tee -a "${LOCAL_LOG_DIR}/tile_connectivity_precomputation.log"
    if [ $? -ne 0 ]; then
        log ERROR "Tile preconnectivity data generation failed"
        return 1
    fi
    
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
    ssh_cmd "cd ${REMOTE_NCLONE_DIR} && python -m nclone.map_generation.generate_test_suite_maps --mode both --output_dir ~/datasets --map_count 6000" 2>&1 | tee -a "${LOCAL_LOG_DIR}/dataset_generation.log"
    
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
    
    local train_cmd="cd ${REMOTE_NPP_RL_DIR} && \
        export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} && \
        export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_KEY} && \
        export CUDA_HOME=${DETECTED_CUDA_HOME} && \
        export CUDA_PATH=${DETECTED_CUDA_HOME} && \
        export LD_LIBRARY_PATH=${DETECTED_CUDA_HOME}/lib64:\${LD_LIBRARY_PATH} && \
        export PATH=${DETECTED_CUDA_HOME}/bin:\${PATH} && \
        python scripts/train_and_compare.py \
            --experiment-name ${EXPERIMENT_NAME}_${TIMESTAMP} \
            --architectures ${arch} \
            --train-dataset ~/datasets/train \
            --test-dataset ~/datasets/test \
            --use-curriculum \
            --enable-auto-curriculum-adjustment \
            --enable-early-stopping \
            --replay-data-dir ../nclone/bc_replays \
            --bc-epochs 10 \
            --bc-batch-size 128 \
            --bc-num-workers 8 \
            --hardware-profile auto \
            --total-timesteps 2000000 \
            --distributed-backend nccl \
            --record-eval-videos \
            --max-videos-per-category 2 \
            --num-eval-episodes 10 \
            --video-fps 60 \
            --s3-bucket ${S3_BUCKET} \
            --s3-prefix experiments/ \
            --output-dir ~/experiments \
            --enable-lr-annealing"

    echo $train_cmd
    return 0
    
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

