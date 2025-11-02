# Quick Start: NPP-RL Training System

## Prerequisites

1. **Install nclone** (required):
   ```bash
   cd /path/to/parent/directory
   git clone https://github.com/tetramputechture/nclone.git
   cd nclone
   pip install -e .
   ```

2. **Install npp-rl**:
   ```bash
   cd ../npp-rl
   pip install -r requirements.txt
   ```

3. **Verify GPU** (optional but recommended):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Quick Test (5 minutes)

Train a single architecture for quick validation:

```bash
cd /path/to/npp-rl

python scripts/train_and_compare.py \
    --experiment-name "quick_test" \
    --architectures mlp_baseline \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100 \
    --num-envs 2 \
    --num-eval-episodes 1 \
    --num-gpus 1 \
    --output-dir experiments/

# Monitor progress
tensorboard --logdir experiments/quick_test_*/
```

## Common Workflows

### 1. Single Architecture Training

**Example: Train vision-free architecture**

```bash
# Using provided script
./scripts/example_single_arch.sh

# Or manually
python scripts/train_and_compare.py \
    --experiment-name "vision_free_baseline" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 5000000 \
    --num-envs 64 \
    --num-gpus 1 \
    --output-dir experiments/
```

**Output:**
```
experiments/
└── vision_free_baseline_20250113_103000/
    ├── config.json
    ├── vision_free_baseline.log
    ├── vision_free/
    │   ├── checkpoints/
    │   ├── tensorboard/
    │   ├── eval_results.json
    │   └── final_model.zip
    └── all_results.json
```

### 2. Multi-Architecture Comparison

**Example: Compare three architectures**

```bash
# Using provided script
./scripts/example_multi_arch.sh

# Or manually
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
tensorboard --logdir experiments/arch_comparison_*/
```

### 3. Test Pretraining Impact

**Example: Compare with and without BC pretraining**

```bash
python scripts/train_and_compare.py \
    --experiment-name "pretraining_test" \
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

**Note:** BC pretraining integration is currently simplified. For full pretraining,
use `bc_pretrain.py` directly and then provide the checkpoint path.

### 4. Production Training with S3 Upload

**Example: Train and upload to S3**

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Run training
./scripts/example_with_s3.sh
```

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir experiments/your_experiment_*/
```

Navigate to `http://localhost:6006` to view:
- Training metrics (rewards, losses, entropy)
- Evaluation results (success rates per category)
- Architecture comparisons

### Log Files

Check training logs:

```bash
tail -f experiments/your_experiment_*/your_experiment.log
```

### Results

View evaluation results:

```bash
cat experiments/your_experiment_*/*/eval_results.json | python -m json.tool
```

## Available Architectures

List all available architectures:

```python
from npp_rl.training.architecture_configs import ARCHITECTURE_REGISTRY
print(list(ARCHITECTURE_REGISTRY.keys()))
```

**Current architectures:**
- `full_hgt`: Full HGT with all modalities (baseline)
- `vision_free`: No visual input (graph + state only)
- `local_only`: Local frames only (no global view)
- `gat`: Graph Attention Network variant
- `gcn`: Graph Convolutional Network variant
- `simplified_hgt`: Reduced HGT complexity
- `mlp_baseline`: Simple MLP (state only)

## Command-Line Arguments Reference

### Essential Arguments

```bash
--experiment-name NAME          # Unique experiment identifier
--architectures ARCH [ARCH ...]  # Space-separated architecture names
--train-dataset PATH            # Path to training dataset
--test-dataset PATH             # Path to test dataset
```

### Training Control

```bash
--total-timesteps N             # Total training timesteps (default: 10M)
--num-envs N                    # Parallel environments (default: 64)
--eval-freq N                   # Evaluation frequency (default: 100K)
--save-freq N                   # Checkpoint frequency (default: 500K)
```

### GPU Options

```bash
--num-gpus N                    # Number of GPUs (default: 1)
--mixed-precision               # Enable AMP for faster training
--distributed-backend BACKEND   # 'nccl' or 'gloo' (default: nccl)
```

### Pretraining

```bash
--test-pretraining              # Compare with/without pretraining
--no-pretraining                # Skip pretraining entirely
--replay-data-dir PATH          # Replay data directory
--bc-epochs N                   # BC training epochs (default: 10)
```

### S3 Upload

```bash
--s3-bucket BUCKET              # S3 bucket name
--s3-prefix PREFIX              # S3 prefix (default: experiments/)
--s3-sync-freq N                # Sync frequency (default: 500K)
```

### Other

```bash
--output-dir DIR                # Output directory (default: experiments/)
--debug                         # Enable debug logging
```

## Troubleshooting

### Dataset Not Found

**Error:** `Dataset path not found: ../nclone/datasets/train`

**Solution:** Generate test suite or verify path
```bash
cd ../nclone
python -m nclone.map_generation.generate_test_suite_maps \
    --output-dir datasets \
    --train-count 250 \
    --test-count 250
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `--num-envs` (try 32 or 16)
2. Enable `--mixed-precision`
3. Use smaller architecture (`vision_free` instead of `full_hgt`)

### S3 Upload Failed

**Error:** `NoCredentialsError: Unable to locate credentials`

**Solutions:**
1. Set AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```
2. Or configure AWS CLI:
   ```bash
   aws configure
   ```
3. Or omit `--s3-bucket` to skip S3 upload

### Architecture Not Found

**Error:** `Unknown architecture 'my_arch'`

**Solution:** Check available architectures:
```python
from npp_rl.training.architecture_configs import ARCHITECTURE_REGISTRY
print(list(ARCHITECTURE_REGISTRY.keys()))
```

## Performance Tips

### Faster Training

1. **Use mixed precision**: Add `--mixed-precision` (2x speedup on modern GPUs)
2. **Increase parallel environments**: Use `--num-envs 128` or higher
3. **Multi-GPU**: Use `--num-gpus 4` if available
4. **Smaller architectures**: Try `vision_free` for faster iteration

### Better Results

1. **Longer training**: Use `--total-timesteps 20000000` or more
2. **Pretraining**: Use `--test-pretraining` if replay data available
3. **Hyperparameter tuning**: Modify PPO hyperparameters in code
4. **Architecture selection**: Test multiple with `--architectures`

## Next Steps

1. **Read full documentation**: See `docs/TRAINING_SYSTEM.md`
2. **Explore architectures**: Review `npp_rl/training/architecture_configs.py`
3. **Customize training**: Modify `npp_rl/training/architecture_trainer.py`
4. **Add analysis**: Implement comparison and visualization tools
5. **Run experiments**: Use the provided example scripts

## Support

- **Documentation**: `docs/TRAINING_SYSTEM.md`
- **Architecture guide**: `npp_rl/optimization/README.md`
- **Project README**: `README.md`
- **Issue tracker**: GitHub issues

## Example Session

Complete example from start to finish:

```bash
# 1. Navigate to project
cd /path/to/npp-rl

# 2. Verify setup
python -c "import nclone, torch; print('Setup OK')"

# 3. Quick test
python scripts/train_and_compare.py \
    --experiment-name "test_run" \
    --architectures vision_free \
    --no-pretraining \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 100000 \
    --num-envs 16 \
    --output-dir experiments/ \
    --debug

# 4. Monitor
tensorboard --logdir experiments/test_run_*/

# 5. Check results
cat experiments/test_run_*/all_results.json | python -m json.tool

# 6. Run full training
./scripts/example_multi_arch.sh
```

---

**Happy Training!** 🚀
