# NPP-RL Training Scripts

This directory contains scripts for training and comparing NPP-RL architectures.

## Main Scripts

### train_and_compare.py

**Master training and comparison script**

Orchestrates training of multiple architectures with evaluation and artifact management.

```bash
python scripts/train_and_compare.py \
    --experiment-name "my_experiment" \
    --architectures full_hgt vision_free \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 10000000 \
    --num-envs 64 \
    --output-dir experiments/
```

See `docs/TRAINING_SYSTEM.md` for full documentation.

### list_architectures.py

**List available architectures**

Displays all available architecture configurations with descriptions.

```bash
python scripts/list_architectures.py
```

Output:
```
======================================================================
Available NPP-RL Architectures
======================================================================

1. full_hgt
   Description: Full HGT Architecture
   Features dim: 512
   Modalities: player frame, global view, graph (hgt), game state, reachability

2. vision_free
   Description: Vision-Free Architecture
   Features dim: 512
   Modalities: graph (hgt), game state, reachability

...
```

## Example Scripts

### example_single_arch.sh

**Train single architecture for testing**

Quick test run with reduced timesteps for validation.

```bash
./scripts/example_single_arch.sh
```

Trains `vision_free` architecture for 1M timesteps.

### example_multi_arch.sh

**Compare multiple architectures**

Full training run comparing three architectures.

```bash
./scripts/example_multi_arch.sh
```

Trains `full_hgt`, `vision_free`, and `gat` for 10M timesteps each.

### example_with_s3.sh

**Production training with S3 upload**

Full training with automatic artifact upload to S3.

```bash
# Set AWS credentials first
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

./scripts/example_with_s3.sh
```

## Quick Reference

### Common Commands

**List architectures:**
```bash
python scripts/list_architectures.py
```

**Quick test (5 minutes):**
```bash
python scripts/train_and_compare.py \
    --experiment-name "quick_test" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 16 \
    --output-dir experiments/
```

**Full comparison:**
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

**Monitor training:**
```bash
tensorboard --logdir experiments/
```

## Configuration

### Essential Arguments

- `--experiment-name`: Unique identifier for experiment
- `--architectures`: Space-separated list of architecture names
- `--train-dataset`: Path to training dataset
- `--test-dataset`: Path to test dataset

### Training Control

- `--total-timesteps`: Total training timesteps (default: 10M)
- `--num-envs`: Number of parallel environments (default: 64)
- `--eval-freq`: Evaluation frequency (default: 100K)
- `--save-freq`: Checkpoint save frequency (default: 500K)

### GPU Options

- `--num-gpus`: Number of GPUs to use (default: 1)
- `--mixed-precision`: Enable mixed precision training (faster)
- `--distributed-backend`: nccl or gloo (default: nccl)

### S3 Options

- `--s3-bucket`: S3 bucket for artifact upload
- `--s3-prefix`: S3 key prefix (default: experiments/)
- `--s3-sync-freq`: Upload frequency (default: 500K)

## Output Structure

```
experiments/
└── {experiment_name}_{timestamp}/
    ├── config.json                 # Experiment configuration
    ├── {experiment_name}.log       # Training logs
    ├── {architecture}/
    │   ├── checkpoints/            # Model checkpoints
    │   ├── tensorboard/            # TensorBoard logs
    │   ├── eval_results.json       # Evaluation metrics
    │   └── final_model.zip         # Final trained model
    ├── all_results.json            # Aggregated results
    └── s3_manifest.json            # S3 upload manifest
```

## Documentation

- **Quick Start**: `docs/QUICK_START_TRAINING.md`
- **Full System Docs**: `docs/TRAINING_SYSTEM.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Project README**: `../README.md`

## Troubleshooting

### Dataset not found
Generate test suite or verify path:
```bash
cd ../nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output-dir datasets \
    --train-count 250 \
    --test-count 250
```

### CUDA out of memory
- Reduce `--num-envs` (try 32 or 16)
- Enable `--mixed-precision`
- Use smaller architecture (vision_free)

### S3 upload failed
- Check AWS credentials are set
- Verify bucket exists and permissions
- Or omit `--s3-bucket` to skip upload

## Examples

### Test on subset
```bash
python scripts/train_and_compare.py \
    --experiment-name "subset_test" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 500000 \
    --num-envs 32 \
    --output-dir experiments/
```

### Production run
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

python scripts/train_and_compare.py \
    --experiment-name "production_v1" \
    --architectures full_hgt vision_free gat \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 20000000 \
    --num-envs 256 \
    --num-gpus 4 \
    --mixed-precision \
    --s3-bucket npp-rl-experiments \
    --s3-prefix production/ \
    --eval-freq 100000 \
    --save-freq 500000 \
    --output-dir experiments/
```

## Support

For issues or questions:
1. Check documentation in `docs/`
2. Review log files in experiment directory
3. Enable `--debug` for detailed logging
4. Check TensorBoard for training metrics

---

**For complete documentation, see:** `docs/TRAINING_SYSTEM.md`
