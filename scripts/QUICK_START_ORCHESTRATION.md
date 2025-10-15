# Quick Start: Architecture Comparison Orchestration

## TL;DR - Run This

```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip YOUR_INSTANCE_IP \
  --aws-access-key YOUR_AWS_KEY \
  --aws-secret-key YOUR_AWS_SECRET \
  --s3-bucket YOUR_S3_BUCKET \
  --experiment-name arch_comparison_$(date +%Y%m%d)
```

Then open http://localhost:6006 in your browser to watch training in real-time.

## Prerequisites Checklist

- [ ] SSH key for instance access
- [ ] Instance IP address
- [ ] AWS credentials (with S3 write access)
- [ ] S3 bucket created
- [ ] `jq` installed locally (`brew install jq` or `apt-get install jq`)
- [ ] Remote instance has `npp-rl` and `nclone` repos at `~/projects/`

## Common Usage Patterns

### 1. Standard Run
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key AKIAIOSFODNN7EXAMPLE \
  --aws-secret-key wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  --s3-bucket npp-rl-experiments \
  --experiment-name production_comparison
```

### 2. With Custom SSH Key
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --ssh-key ~/.ssh/gpu_instance.pem \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --s3-bucket my-bucket \
  --experiment-name my_experiment
```

### 3. Resume Interrupted Run
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key AKIA... \
  --aws-secret-key wJal... \
  --s3-bucket my-bucket \
  --experiment-name my_experiment \
  --resume  # Skip completed architectures
```

## What to Expect

### Timeline
```
[00:00] Script starts, SSH connection established
[00:01] TensorBoard available at http://localhost:6006
[00:05] Dataset generation begins (500 levels)
[00:15] Dataset complete, training starts
[01:00] Architecture 1/11 complete (full_hgt)
[02:30] Architecture 2/11 complete (simplified_hgt)
...
[18:00] All 11 architectures complete
[18:05] Results downloaded and summarized
```

**Total time**: 11-22 hours (depends on GPU)

### Live Monitoring

Open http://localhost:6006 in your browser to see:
- Training loss curves
- Success rates per architecture
- Episode rewards
- Real-time training metrics

### Output Files

```
./logs/arch_comparison_20251015_143000/
├── orchestration.log              # Main log
├── config.json                    # Configuration
├── experiment_summary.json        # Final results
└── results/                       # Per-architecture results
    ├── full_hgt/
    ├── simplified_hgt/
    └── ...
```

### S3 Artifacts

```
s3://your-bucket/experiments/arch_comparison_20251015_143000/
├── full_hgt/
│   ├── checkpoints/
│   ├── videos/
│   └── tensorboard/
└── ...
```

## After Completion

### 1. View Results Summary
```bash
cat logs/arch_comparison_*/experiment_summary.json | jq
```

### 2. Compare Success Rates
```bash
cd logs/arch_comparison_*/results/
for arch in */; do
  echo -n "$arch: "
  cat $arch/eval_results.json | jq -r '.overall.success_rate'
done | sort -t: -k2 -rn
```

### 3. View All Metrics in TensorBoard
```bash
tensorboard --logdir=logs/arch_comparison_*/results/
```

### 4. Download Videos from S3
```bash
aws s3 sync \
  s3://your-bucket/experiments/arch_comparison_*/full_hgt/videos/ \
  ./videos/full_hgt/
```

## Troubleshooting

### Can't connect to instance
```bash
# Test SSH connection
ssh -i ~/.ssh/your_key.pem ubuntu@YOUR_INSTANCE_IP "echo 'Connection successful'"
```

### TensorBoard not loading
```bash
# Check if port is in use
lsof -i :6006

# Kill and restart script
kill $(lsof -ti:6006)
./scripts/orchestrate_architecture_comparison.sh ...
```

### Check remote progress
```bash
# SSH to instance and check logs
ssh -i ~/.ssh/your_key.pem ubuntu@YOUR_INSTANCE_IP
tail -f ~/experiments/*/full_hgt/training.log
```

## Pro Tips

1. **Use tmux locally** to keep script running if connection drops
   ```bash
   tmux new -s arch_comparison
   ./scripts/orchestrate_architecture_comparison.sh ...
   # Detach: Ctrl+B, then D
   # Reattach later: tmux attach -t arch_comparison
   ```

2. **Monitor GPU usage** on remote instance
   ```bash
   ssh ubuntu@YOUR_IP "watch -n 1 nvidia-smi"
   ```

3. **Check estimated completion time**
   - 1 architecture ≈ 1-2 hours
   - 11 architectures ≈ 11-22 hours
   - Monitor first architecture to estimate total time

4. **Keep terminal alive**
   - macOS: `caffeinate -i ./scripts/orchestrate_architecture_comparison.sh ...`
   - Linux: Use `tmux` or `screen`

## Full Documentation

See [ORCHESTRATION_README.md](ORCHESTRATION_README.md) for complete documentation.

## Example Commands

### Minimal (all defaults)
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --aws-access-key $AWS_ACCESS_KEY_ID \
  --aws-secret-key $AWS_SECRET_ACCESS_KEY \
  --s3-bucket my-bucket \
  --experiment-name test_run
```

### Production Run (recommended)
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip 54.123.45.67 \
  --ssh-key ~/.ssh/gpu_key.pem \
  --ssh-user ubuntu \
  --aws-access-key $AWS_ACCESS_KEY_ID \
  --aws-secret-key $AWS_SECRET_ACCESS_KEY \
  --s3-bucket npp-rl-production \
  --experiment-name arch_comparison_$(date +%Y%m%d)
```

### Test Run (verify setup)
First, test SSH connection:
```bash
ssh -i ~/.ssh/your_key.pem ubuntu@YOUR_IP "echo 'Connected!'"
```

Then run orchestration:
```bash
./scripts/orchestrate_architecture_comparison.sh \
  --instance-ip YOUR_IP \
  --ssh-key ~/.ssh/your_key.pem \
  --aws-access-key $AWS_ACCESS_KEY_ID \
  --aws-secret-key $AWS_SECRET_ACCESS_KEY \
  --s3-bucket your-bucket \
  --experiment-name test_$(date +%H%M%S)
```

Monitor the first few minutes to ensure everything works, then let it run.

