# NPP-RL Training and Comparison System

## Overview

This document describes the comprehensive training and architecture comparison system for NPP-RL. The system provides:

- **Multi-Architecture Training**: Train and compare multiple architecture variants
- **Pretraining Support**: Optional behavioral cloning pretraining from human replays
- **Multi-GPU Training**: Distributed training across multiple GPUs
- **Comprehensive Evaluation**: Standardized evaluation on test suite
- **Artifact Management**: Automatic upload to AWS S3
- **TensorBoard Logging**: Detailed training and comparison metrics

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Master Training Script                        │
│                 (train_and_compare.py)                       │
└───────────────┬─────────────────────────────────────────────┘
                │
                ├─► Architecture Configuration Loading
                │   └─► npp_rl/training/architecture_configs.py
                │
                ├─► Pretraining Pipeline (Optional)
                │   └─► npp_rl/training/pretraining_pipeline.py
                │
                ├─► Architecture Trainer
                │   └─► npp_rl/training/architecture_trainer.py
                │
                ├─► Comprehensive Evaluator
                │   └─► npp_rl/evaluation/comprehensive_evaluator.py
                │
                ├─► Logging & Monitoring
                │   ├─► npp_rl/utils/logging_utils.py
                │   └─► TensorBoard integration
                │
                └─► S3 Artifact Upload
                    └─► npp_rl/utils/s3_uploader.py
```

## Components

### 1. Core Infrastructure

#### S3 Uploader (`npp_rl/utils/s3_uploader.py`)
- Handles uploading training artifacts to AWS S3
- Supports incremental sync during training
- Tracks uploaded files with manifest
- Dry-run mode for testing

**Key Features:**
- File and directory uploads
- Metadata attachment
- TensorBoard log syncing
- Manifest generation

#### Logging Utilities (`npp_rl/utils/logging_utils.py`)
- Structured experiment logging
- TensorBoard management
- Configuration save/load
- Experiment summaries

**Key Features:**
- Hierarchical TensorBoard writers
- Multi-architecture logging
- Comprehensive metrics tracking

### 2. Distributed Training

#### Distributed Utils (`npp_rl/training/distributed_utils.py`)
- PyTorch DDP setup and management
- Automatic Mixed Precision (AMP) support
- Environment distribution across GPUs
- CUDA optimization configuration

**Key Features:**
- Multi-GPU synchronization
- Gradient scaling for AMP
- TF32 support for Ampere+ GPUs
- Context manager for clean setup/teardown

### 3. Pretraining Pipeline

#### Pretraining Pipeline (`npp_rl/training/pretraining_pipeline.py`)
- Automates behavioral cloning pretraining
- Replay data validation
- Checkpoint management
- Integration with BC trainer

**Key Features:**
- Automatic data preparation
- Checkpoint validation
- Error handling and fallback

**Note:** Full integration with BC trainer is simplified in current implementation.
For complete BC pretraining, use `bc_pretrain.py` directly.

### 4. Training Management

#### Architecture Trainer (`npp_rl/training/architecture_trainer.py`)
- Manages training for single architecture
- Model setup from architecture config
- Environment creation and management
- Checkpointing and evaluation

**Key Features:**
- Architecture-specific configuration
- Pretrained weight loading
- Vectorized environment setup
- Comprehensive evaluation integration

### 5. Evaluation System

#### Test Suite Loader (`npp_rl/evaluation/test_suite_loader.py`)
- Loads standardized test levels
- Category-based organization
- Metadata management

**Supported Categories:**
- Simple
- Medium
- Complex
- Mine-heavy
- Exploration

#### Comprehensive Evaluator (`npp_rl/evaluation/comprehensive_evaluator.py`)
- Evaluates models on full test suite
- Per-category performance metrics
- Success rate, efficiency, safety scoring
- Result export and reporting

**Key Metrics:**
- Success rate (overall and per-category)
- Average steps to completion
- Efficiency (success / normalized steps)
- Episode-level statistics

### 6. Master Training Script

#### train_and_compare.py (`scripts/train_and_compare.py`)
- Orchestrates full training experiments
- Multi-architecture comparison
- Pretraining condition testing
- Result aggregation and upload

## Usage

### Basic Usage

**Single Architecture Training:**
```bash
python scripts/train_and_compare.py \
    --experiment-name "test_vision_free" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 64 \
    --num-gpus 1 \
    --output-dir experiments/
```

### Multi-Architecture Comparison

**Compare Multiple Architectures:**
```bash
python scripts/train_and_compare.py \
    --experiment-name "arch_comparison" \
    --architectures full_hgt vision_free gat \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000000 \
    --num-envs 64 \
    --num-gpus 4 \
    --mixed-precision \
    --output-dir experiments/
```

### Pretraining Comparison

**Test Impact of Pretraining:**
```bash
python scripts/train_and_compare.py \
    --experiment-name "pretraining_impact" \
    --architectures full_hgt \
    --test-pretraining \
    --replay-data-dir datasets/human_replays \
    --bc-epochs 10 \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000000 \
    --num-envs 64 \
    --output-dir experiments/
```

### With S3 Upload

**Training with Artifact Upload:**
```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

python scripts/train_and_compare.py \
    --experiment-name "production_run" \
    --architectures full_hgt vision_free \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000000 \
    --num-envs 256 \
    --num-gpus 4 \
    --mixed-precision \
    --s3-bucket npp-rl-experiments \
    --s3-prefix experiments/ \
    --s3-sync-freq 500000 \
    --output-dir experiments/
```

## Command-Line Arguments

### Required Arguments

- `--experiment-name`: Unique experiment identifier
- `--architectures`: Space-separated list of architecture names
- `--train-dataset`: Path to training dataset directory
- `--test-dataset`: Path to test dataset directory

### Training Options

- `--total-timesteps`: Total training timesteps (default: 10M)
- `--num-envs`: Number of parallel environments (default: 64)
- `--eval-freq`: Evaluation frequency in timesteps (default: 100K)
- `--save-freq`: Checkpoint save frequency (default: 500K)

### Pretraining Options

- `--test-pretraining`: Compare with and without pretraining
- `--no-pretraining`: Skip pretraining entirely
- `--replay-data-dir`: Directory with replay data for BC
- `--bc-epochs`: Number of BC training epochs (default: 10)
- `--bc-batch-size`: BC batch size (default: 64)

### Multi-GPU Options

- `--num-gpus`: Number of GPUs to use (default: 1)
- `--distributed-backend`: Backend for distributed training (nccl/gloo)
- `--mixed-precision`: Enable mixed precision training

### S3 Options

- `--s3-bucket`: S3 bucket name for uploads
- `--s3-prefix`: S3 prefix for organization (default: experiments/)
- `--s3-sync-freq`: Sync frequency in timesteps (default: 500K)

### Other Options

- `--output-dir`: Base output directory (default: experiments/)
- `--resume-from`: Resume from existing experiment directory
- `--debug`: Enable debug logging

## Available Architectures

The following architectures are available in `ARCHITECTURE_REGISTRY`:

1. **full_hgt**: Full Heterogeneous Graph Transformer (all modalities)
2. **vision_free**: No visual input (graph + state only)
3. **local_only**: Local frames + graph + state (no global view)
4. **gat**: Graph Attention Network variant
5. **gcn**: Graph Convolutional Network variant
6. **simplified_hgt**: Reduced HGT complexity
7. **mlp_baseline**: Simple MLP baseline (state only)

Configurations defined in: `npp_rl/training/architecture_configs.py`

## Output Structure

Each experiment creates the following directory structure:

```
experiments/
└── {experiment_name}_{timestamp}/
    ├── config.json                    # Experiment configuration
    ├── {experiment_name}.log          # Training logs
    ├── {architecture_name}/
    │   ├── no_pretrain/               # Without pretraining
    │   │   ├── checkpoints/           # Model checkpoints
    │   │   ├── tensorboard/           # TensorBoard logs
    │   │   ├── eval_results.json      # Evaluation results
    │   │   └── final_model.zip        # Final model
    │   └── with_pretrain/             # With pretraining (if --test-pretraining)
    │       └── ...
    ├── all_results.json               # Aggregated results
    └── s3_manifest.json               # S3 upload manifest (if used)
```

## TensorBoard Monitoring

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir experiments/{experiment_name}_{timestamp}/
```

**Available Metrics:**
- Training: episode reward, length, success rate, losses, entropy
- Evaluation: per-category success rates, efficiency, safety
- Comparison: cross-architecture performance

## Implementation Status

### ✅ Completed Components

1. **Core Infrastructure**
   - ✅ S3 uploader with manifest tracking
   - ✅ Logging utilities with TensorBoard management
   - ✅ Experiment configuration management

2. **Distributed Training**
   - ✅ PyTorch DDP setup utilities
   - ✅ AMP support for mixed precision
   - ✅ Environment distribution logic
   - ✅ CUDA optimization helpers

3. **Pretraining**
   - ✅ Pretraining pipeline framework
   - ⚠️ Simplified BC integration (use bc_pretrain.py for full BC)

4. **Training**
   - ✅ Architecture trainer with full lifecycle
   - ✅ Model setup from architecture configs
   - ✅ Vectorized environment creation
   - ✅ Checkpointing and evaluation

5. **Evaluation**
   - ✅ Test suite loader
   - ✅ Comprehensive evaluator with per-category metrics
   - ✅ Result export and reporting

6. **Master Script**
   - ✅ Multi-architecture orchestration
   - ✅ Pretraining condition testing
   - ✅ S3 integration
   - ✅ Result aggregation

### 🚧 TODO / Improvements Needed

1. **Analysis and Visualization**
   - ❌ Comparison analysis module (`npp_rl/analysis/experiment_comparison.py`)
   - ❌ Visualization generation (matplotlib/seaborn plots)
   - ❌ Architecture decision matrix
   - ❌ Training curve comparisons

2. **Advanced Features**
   - ⚠️ Full multi-GPU training integration (currently single GPU)
   - ⚠️ Distributed environment management
   - ❌ Inference time benchmarking
   - ❌ Architecture search/optimization

3. **BC Pretraining**
   - ⚠️ Full integration with BC trainer
   - ⚠️ Replay data processing pipeline
   - ❌ BC dataset generation from replays

4. **Testing**
   - ❌ Unit tests for all modules
   - ❌ Integration tests
   - ❌ End-to-end validation

5. **Documentation**
   - ❌ Launch scripts for common scenarios
   - ❌ Troubleshooting guide
   - ❌ S3 bucket setup instructions
   - ❌ Example notebooks

## Dependencies

The system requires the following Python packages:

```txt
torch>=2.0.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
tensorboard>=2.14.0
boto3>=1.28.0  # For S3 upload
tqdm>=4.65.0
numpy>=1.21.0
```

Plus the `nclone` simulator (required):
```bash
cd ../nclone && pip install -e .
```

## AWS S3 Setup

### Creating S3 Bucket

```bash
aws s3 mb s3://npp-rl-experiments --region us-east-1
```

### IAM Policy

Required permissions for S3 upload:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::npp-rl-experiments",
                "arn:aws:s3:::npp-rl-experiments/*"
            ]
        }
    ]
}
```

### Setting Credentials

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Option 2: AWS CLI configuration
aws configure

# Option 3: IAM role (for EC2 instances)
# Attach IAM role with above policy to EC2 instance
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `--num-envs`
- Enable `--mixed-precision`
- Use smaller architecture (e.g., vision_free)

**2. S3 Upload Failures**
- Check AWS credentials
- Verify bucket exists and permissions
- Try `--s3-bucket` with `None` to disable

**3. Dataset Not Found**
- Verify dataset paths exist
- Check for `.pkl` files in category subdirectories
- Generate test suite if needed (see nclone docs)

**4. Architecture Loading Errors**
- Verify architecture name in `ARCHITECTURE_REGISTRY`
- Check `npp_rl/training/architecture_configs.py`
- Use `--debug` for detailed error messages

**5. Multi-GPU Issues**
- Currently simplified to single GPU
- Full distributed training needs additional implementation

## Future Enhancements

### High Priority
1. Complete analysis and visualization module
2. Full multi-GPU distributed training
3. Comprehensive test suite
4. Architecture comparison report generation

### Medium Priority
5. Inference time benchmarking
6. Training curve analysis and comparison
7. Hyperparameter optimization integration
8. Resume from checkpoint functionality

### Low Priority
9. Web-based dashboard for experiment monitoring
10. Automated architecture search
11. Transfer learning between architectures
12. Ensemble model support

## Contributing

When adding new components:

1. Follow existing code structure and patterns
2. Add comprehensive docstrings
3. Update this documentation
4. Add unit tests (TODO: test infrastructure)
5. Keep files under 500 lines
6. Use type hints for function parameters

## References

- Main project README: `../README.md`
- Architecture configs: `../npp_rl/optimization/README.md`
- Training guide: `../docs/TRAINING_AND_TESTING.md`
- Phase 3 tasks: `../docs/tasks/PHASE_3_ROBUSTNESS_OPTIMIZATION.md`
- Phase 4 tasks: `../docs/tasks/PHASE_4_ADVANCED_FEATURES.md`
