# Architecture Comparison Orchestration

This directory contains the comprehensive orchestration script for running end-to-end architecture comparison experiments on remote GPU instances.

## Overview

`orchestrate_architecture_comparison.sh` automates the complete architecture comparison pipeline:

1. **Dataset Generation**: Generates 500 deterministic levels (250 train + 250 test) using `nclone`'s map generation
2. **Sequential Training**: Trains all 11 architectures with 1M timesteps each
3. **Real-time Monitoring**: Sets up TensorBoard port forwarding for live training metrics
4. **Artifact Management**: Automatically uploads checkpoints, videos, and logs to S3
5. **Result Collection**: Downloads evaluation results and creates comparison summaries

## Architectures Compared

The script trains and evaluates these 11 architectures:

1. `full_hgt` - Full Heterogeneous Graph Transformer (baseline)
2. `simplified_hgt` - Reduced HGT with lower complexity
3. `gat` - Graph Attention Network
4. `gcn` - Graph Convolutional Network
5. `mlp_baseline` - MLP without graph processing
6. `vision_free` - Graph + state only (no vision)
7. `vision_free_gat` - Vision-free with GAT
8. `vision_free_gcn` - Vision-free with GCN
9. `vision_free_simplified` - Vision-free with simplified HGT
10. `no_global_view` - Full architecture without global view
11. `local_frames_only` - Temporal frames + graph + state

## Requirements

### Local Machine
- SSH access to remote GPU instance
- `jq` for JSON processing: `apt-get install jq` or `brew install jq`
- SSH key for authentication

### Remote GPU Instance
- Ubuntu with GPU support
- `npp-rl` and `nclone` repositories at `~/projects/npp-rl` and `~/projects/nclone`
- Python 3.8+ with dependencies installed
- AWS credentials configured (or provided via script)

## Usage

### Basic Usage

```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key AKIAIOSFODNN7EXAMPLE \
  --aws-secret-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --s3-bucket npp-rl-experiments \
  --experiment-name arch_comparison_2025
```

### With Custom SSH Key

```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --ssh-key ~/.ssh/my_gpu_key.pem \
  --ssh-user ubuntu \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --s3-bucket npp-rl-experiments \
  --experiment-name my_experiment
```

### Resume Incomplete Run

If the script is interrupted, you can resume and skip completed architectures:

```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --s3-bucket npp-rl-experiments \
  --experiment-name arch_comparison_2025 \
  --resume
```

## Arguments

### Required
- `--instance-ip IP` - Remote GPU instance IP address
- `--aws-access-key KEY` - AWS access key ID for S3 uploads
- `--aws-secret-key SECRET` - AWS secret access key
- `--s3-bucket BUCKET` - S3 bucket name for artifact storage
- `--experiment-name NAME` - Unique experiment identifier

### Optional
- `--ssh-key PATH` - Path to SSH private key (default: `~/.ssh/id_rsa`)
- `--ssh-user USER` - SSH username (default: `ubuntu`)
- `--resume` - Skip already completed architectures
- `--help` - Show usage information

## What Happens During Execution

### Phase 1: Setup (2-5 minutes)
1. Validates SSH connection to remote instance
2. Sets up TensorBoard port forwarding on localhost:6006
3. Configures AWS credentials on remote instance
4. Creates experiment directories

### Phase 2: Dataset Generation (5-15 minutes)
1. Runs `generate_test_suite_maps.py` on remote instance
2. Generates 500 deterministic levels across 5 difficulty categories
3. Validates dataset integrity

### Phase 3: Architecture Training (11-22 hours total)
For each architecture (1-2 hours per architecture):
1. Trains model for 1M timesteps with 64 parallel environments
2. Records evaluation videos (5 per category)
3. Uploads checkpoints and artifacts to S3
4. Logs training metrics to TensorBoard

### Phase 4: Result Collection
1. Downloads evaluation results to local machine
2. Creates comparison summary JSON
3. Generates final report

## Real-time Monitoring

### TensorBoard
While the script is running, you can monitor training progress in real-time:

```bash
# Open in browser: http://localhost:6006
```

TensorBoard will show:
- Training loss curves for all architectures
- Episode rewards and success rates
- Policy/value network metrics
- Curriculum progression (if enabled)

### Logs
Local logs are saved to `./logs/arch_comparison_TIMESTAMP/`:
- `orchestration.log` - Main orchestration log
- `{architecture}_training.log` - Per-architecture training logs
- `dataset_generation.log` - Dataset generation output
- `config.json` - Experiment configuration
- `experiment_summary.json` - Final summary

## Output Structure

### Local Files
```
./logs/arch_comparison_20251015_143000/
├── orchestration.log              # Main log file
├── config.json                    # Experiment configuration
├── dataset_generation.log         # Dataset generation output
├── experiment_summary.json        # Final summary
├── full_hgt_training.log         # Per-architecture logs
├── simplified_hgt_training.log
├── ...
└── results/
    ├── full_hgt/
    │   ├── eval_results.json
    │   └── config.json
    ├── simplified_hgt/
    └── ...
```

### S3 Artifacts
```
s3://your-bucket/experiments/arch_comparison_2025_20251015_143000/
├── full_hgt/
│   ├── checkpoints/
│   │   ├── checkpoint_500000.zip
│   │   └── final_model.zip
│   ├── videos/
│   │   ├── simple_001.mp4
│   │   └── ...
│   ├── tensorboard/
│   └── eval_results.json
├── simplified_hgt/
└── ...
```

## Error Handling

The script includes comprehensive error handling:

- **SSH Connection Lost**: Script will exit cleanly and save partial results
- **Training Failure**: Failed architectures are logged, script continues with remaining ones
- **Ctrl+C**: Graceful cleanup of SSH tunnels and background processes
- **Dataset Generation Failure**: Script exits before starting training

## Resuming After Interruption

If the orchestration is interrupted:

1. The script saves partial results to `experiment_summary.json`
2. S3 artifacts remain intact for completed architectures
3. Use `--resume` flag to skip completed architectures:

```bash
./scripts/orchestrate_architecture_comparison.sh \
  --resume \
  --instance-ip ... \
  --experiment-name arch_comparison_2025  # Same name as before
```

The script checks for existing `eval_results.json` files to determine completion status.

## Performance Expectations

### Training Time (approximate)
- **Per Architecture**: 1-2 hours (1M timesteps, 64 envs)
- **Total (11 architectures)**: 11-22 hours
- Varies by GPU (faster on A100, slower on T4)

### Storage Requirements
- **Local logs**: ~500 MB - 2 GB (depending on logging verbosity)
- **S3 artifacts per architecture**: 
  - Checkpoints: ~500 MB - 2 GB
  - Videos: ~50-200 MB
  - TensorBoard logs: ~100-500 MB
- **Total S3 storage**: ~10-30 GB for all 11 architectures

### Network Usage
- **S3 uploads**: ~10-30 GB during training
- **Result downloads**: ~100-500 MB at completion
- **SSH tunneling**: Minimal (TensorBoard metrics only)

## Troubleshooting

### "SSH connection failed"
- Verify instance IP is correct
- Check SSH key permissions: `chmod 400 ~/.ssh/your_key.pem`
- Ensure security group allows SSH (port 22)

### "Remote repositories not found"
- SSH to instance and verify: `ls ~/projects/npp-rl ~/projects/nclone`
- Clone if missing:
  ```bash
  cd ~/projects
  git clone <npp-rl-repo-url>
  git clone <nclone-repo-url>
  ```

### "Dataset generation failed"
- SSH to instance and check Python version: `python3 --version`
- Verify nclone installation: `cd ~/projects/nclone && python -m nclone.map_generation.generate_test_suite_maps --help`

### "TensorBoard not accessible"
- Check if port 6006 is already in use: `lsof -i :6006`
- Kill existing process: `kill $(lsof -ti:6006)`
- Restart script

### "S3 upload failed"
- Verify AWS credentials are correct
- Check S3 bucket exists and you have write permissions
- Verify boto3 is installed on remote: `pip install boto3`

## Post-Experiment Analysis

### Compare Results
```bash
# View all results in TensorBoard
tensorboard --logdir=./logs/arch_comparison_TIMESTAMP/results/

# Analyze evaluation results
cd ./logs/arch_comparison_TIMESTAMP/results/
cat */eval_results.json | jq '.overall.success_rate'
```

### Generate Comparison Report
```python
import json
from pathlib import Path

# Load all results
results_dir = Path("./logs/arch_comparison_TIMESTAMP/results")
comparison = {}

for arch_dir in results_dir.iterdir():
    if arch_dir.is_dir():
        results_file = arch_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                comparison[arch_dir.name] = data['overall']['success_rate']

# Sort by success rate
sorted_archs = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
print("Architecture Rankings:")
for i, (arch, success_rate) in enumerate(sorted_archs, 1):
    print(f"{i}. {arch}: {success_rate:.2%}")
```

## Tips for Best Results

1. **Use a dedicated GPU instance**: Don't run other training jobs simultaneously
2. **Monitor initial progress**: Watch the first architecture complete to ensure everything works
3. **Keep local machine awake**: Use `caffeinate` (macOS) or similar to prevent sleep
4. **Check S3 costs**: Monitor your S3 bucket size during training
5. **Use tmux on remote**: Consider running in tmux on the remote instance for added resilience

## Support

For issues or questions:
1. Check the orchestration log: `./logs/arch_comparison_TIMESTAMP/orchestration.log`
2. Review architecture-specific logs for training failures
3. Verify remote instance logs: `ssh instance "tail -n 100 ~/experiments/*/training.log"`

## Example: Complete Workflow

```bash
# 1. Start orchestration
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --s3-bucket npp-rl-experiments \
  --experiment-name production_comparison

# 2. Monitor in real-time (in another terminal)
# Open http://localhost:6006 in browser

# 3. Wait for completion (11-22 hours)

# 4. Analyze results
cd logs/arch_comparison_20251015_143000/results/
cat */eval_results.json | jq '.overall.success_rate'

# 5. View artifacts in S3
aws s3 ls s3://npp-rl-experiments/experiments/production_comparison_20251015_143000/

# 6. Compare in TensorBoard
tensorboard --logdir=logs/arch_comparison_20251015_143000/results/
```

## Next Steps After Comparison

Based on the results, you can:
1. **Choose the best architecture** for longer training (10M+ timesteps)
2. **Fine-tune hyperparameters** for top-performing architectures
3. **Run ablation studies** on specific components
4. **Deploy the best model** for actual gameplay evaluation

