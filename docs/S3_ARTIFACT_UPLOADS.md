# S3 Artifact Upload Guide

This guide explains how to configure automatic S3 uploads for all training artifacts, including checkpoints, logs, videos, and visualizations.

## Overview

The training system automatically uploads all artifacts to AWS S3 when credentials are provided. This ensures that:
- Training runs are fully reproducible
- No critical artifacts are lost
- Results can be shared across teams
- Long-running experiments are backed up incrementally

## Quick Start

### 1. Configure AWS Credentials

Set your AWS credentials as environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # Optional, defaults to us-east-1
```

Alternatively, configure using AWS CLI:

```bash
aws configure
```

### 2. Run Training with S3 Upload

```bash
python scripts/train_and_compare.py \
    --experiment-name "my_experiment" \
    --architectures vision_free \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 1000000 \
    --num-envs 16 \
    --s3-bucket "my-npp-rl-bucket" \
    --s3-prefix "experiments/" \
    --record-eval-videos \
    --output-dir experiments/
```

## Uploaded Artifacts

The system automatically uploads the following artifacts for each training run:

### Per-Architecture Artifacts

**Checkpoints** (`{architecture}/checkpoints/`)
- `checkpoint_500000.zip`
- `checkpoint_1000000.zip`
- `checkpoint_1500000.zip`
- ... (every 500K timesteps by default)

**Final Model** (`{architecture}/`)
- `final_model.zip` - Final trained model ready for deployment

**TensorBoard Logs** (`{architecture}/tensorboard/`)
- `events.out.tfevents.*` - Complete training metrics
- Viewable with: `tensorboard --logdir=s3://bucket/prefix/experiment/`

**Evaluation Results** (`{architecture}/`)
- `eval_results.json` - Comprehensive test suite performance
- Includes per-level-type breakdown and success rates

**Videos** (`{architecture}/videos/`)
- `{category}_{idx:03d}_success.mp4` - Successful episode recordings
- `{category}_{idx:03d}_failure.mp4` - Failed episode recordings
- Organized by test category (easy, medium, hard, etc.)

**Route Visualizations** (`{architecture}/route_visualizations/`)
- `route_step{step:09d}_{stage}_{level_id}.png` - Agent path visualizations
- Shows successful routes taken through levels
- Useful for understanding learned behaviors

**Training Config** (`{architecture}/`)
- `training_config.json` - Complete hyperparameter configuration

### Pretraining Artifacts (if enabled)

**BC Pretraining** (`{architecture}/pretrain/`)
- `bc_checkpoint.pt` - Pretrained behavioral cloning weights
- `tensorboard/events.out.tfevents.*` - Pretraining metrics

### Experiment-Level Artifacts

**Aggregated Results** (experiment root)
- `all_results.json` - Combined results from all architectures
- `config.json` - Experiment configuration
- `s3_manifest.json` - Complete list of uploaded files with metadata

## S3 Upload Options

### Command-Line Arguments

```
--s3-bucket BUCKET        S3 bucket name for artifact upload
--s3-prefix PREFIX        S3 prefix for uploads (default: experiments/)
--s3-sync-freq FREQ       Upload frequency in timesteps (default: 500000)
```

### S3 Bucket Structure

After training, your S3 bucket will have the following structure:

```
s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/
├── config.json
├── all_results.json
├── s3_manifest.json
├── vision_free/
│   ├── final_model.zip
│   ├── eval_results.json
│   ├── training_config.json
│   ├── checkpoints/
│   │   ├── checkpoint_500000.zip
│   │   ├── checkpoint_1000000.zip
│   │   └── ...
│   ├── tensorboard/
│   │   └── events.out.tfevents.1234567890.hostname.123.0
│   ├── videos/
│   │   ├── easy/
│   │   │   ├── easy_001_success.mp4
│   │   │   └── easy_002_failure.mp4
│   │   └── medium/
│   │       └── medium_001_success.mp4
│   ├── route_visualizations/
│   │   ├── route_step000500000_stage_3_env_12.png
│   │   └── route_step001000000_stage_4_env_05.png
│   └── pretrain/
│       ├── bc_checkpoint.pt
│       └── tensorboard/
│           └── events.out.tfevents.*
└── full_hgt/
    └── ... (same structure)
```

## Video Recording

### Enable Video Recording

Add the `--record-eval-videos` flag to record evaluation episodes:

```bash
python scripts/train_and_compare.py \
    --architectures vision_free \
    --record-eval-videos \
    --max-videos-per-category 10 \
    --video-fps 30 \
    ...
```

### Video Recording Options

```
--record-eval-videos              Enable video recording during evaluation
--max-videos-per-category N       Maximum videos per category (default: 10)
--video-fps FPS                   Video framerate (default: 30)
```

### Video Details

- **Format**: MP4 (H.264 codec via libx264)
- **Resolution**: Native game resolution (variable based on level)
- **Naming**: `{category}_{idx:03d}_{success|failure}.mp4`
- **Rotation**: Automatically rotated 90° for correct orientation
- **Categories**: Organized by test suite categories (easy, medium, hard, mines, switches, etc.)

### Video Use Cases

1. **Debugging**: Identify failure modes and edge cases
2. **Analysis**: Study learned behaviors and strategies
3. **Demonstrations**: Create training progress showcases
4. **Comparison**: Compare different architectures visually

## S3 Manifest

The `s3_manifest.json` file provides a complete record of all uploads:

```json
{
  "experiment_name": "my_experiment_20251027_120000",
  "bucket": "my-npp-rl-bucket",
  "base_prefix": "experiments/my_experiment_20251027_120000",
  "dry_run": false,
  "uploaded_files": [
    {
      "local_path": "/path/to/final_model.zip",
      "s3_key": "experiments/my_experiment_20251027_120000/vision_free/final_model.zip",
      "timestamp": "2025-10-27T12:30:45.123456",
      "size_bytes": 52428800
    },
    ...
  ],
  "total_files": 142,
  "total_size_bytes": 2147483648,
  "created_at": "2025-10-27T13:00:00.000000"
}
```

## Accessing Uploaded Artifacts

### Download Entire Experiment

```bash
aws s3 sync s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/ ./local_dir/
```

### Download Specific Artifacts

```bash
# Download final models only
aws s3 cp s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/vision_free/final_model.zip ./

# Download all videos
aws s3 sync s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/vision_free/videos/ ./videos/

# Download TensorBoard logs
aws s3 sync s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/vision_free/tensorboard/ ./tb_logs/
```

### View TensorBoard from S3

TensorBoard can directly read from S3 (requires AWS credentials):

```bash
tensorboard --logdir=s3://my-npp-rl-bucket/experiments/my_experiment_20251027_120000/vision_free/tensorboard/
```

## Troubleshooting

### No Artifacts Uploaded

**Issue**: Training completes but no S3 uploads occur.

**Solutions**:
1. Verify AWS credentials are configured:
   ```bash
   aws s3 ls s3://my-npp-rl-bucket/
   ```
2. Check S3 bucket exists and you have write permissions
3. Review logs for S3 upload errors

### Videos Not Generated

**Issue**: `--record-eval-videos` is set but no videos are created.

**Solutions**:
1. Ensure `imageio` and `imageio-ffmpeg` are installed:
   ```bash
   pip install imageio>=2.31.0 imageio-ffmpeg>=0.4.8
   ```
2. Check that evaluation is not skipped (`--skip-final-eval`)
3. Verify sufficient disk space for temporary video files
4. Check logs for video recording errors

### Incomplete Uploads

**Issue**: Some artifacts are missing from S3.

**Solutions**:
1. Check the `s3_manifest.json` to see what was uploaded
2. Verify the training completed successfully (not interrupted)
3. Check S3 bucket permissions and quotas
4. Review training logs for upload failures

## Best Practices

### 1. Organize by Experiment Type

Use descriptive S3 prefixes to organize experiments:

```bash
--s3-prefix "experiments/curriculum_learning/"
--s3-prefix "experiments/ablation_studies/"
--s3-prefix "experiments/architecture_comparison/"
```

### 2. Use Descriptive Experiment Names

Include key parameters in experiment names:

```bash
--experiment-name "vision_free_5M_timesteps_curriculum"
--experiment-name "full_hgt_10M_timesteps_no_pretrain"
```

### 3. Record Videos Selectively

Videos are large; record strategically:
- Record for final evaluations
- Limit to 5-10 videos per category
- Focus on challenging level types

### 4. Clean Up Old Experiments

Periodically archive or delete old S3 experiments:

```bash
# List experiments
aws s3 ls s3://my-npp-rl-bucket/experiments/

# Archive old experiment
aws s3 sync s3://my-npp-rl-bucket/experiments/old_experiment/ ./archive/
aws s3 rm s3://my-npp-rl-bucket/experiments/old_experiment/ --recursive
```

### 5. Monitor S3 Costs

Track S3 storage and data transfer costs:
- Enable S3 storage analytics
- Use lifecycle policies to archive old data to Glacier
- Monitor monthly bills for unexpected usage

## Security Considerations

### IAM Permissions

Your AWS IAM user/role needs the following S3 permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::my-npp-rl-bucket",
        "arn:aws:s3:::my-npp-rl-bucket/*"
      ]
    }
  ]
}
```

### Credential Management

**Never commit AWS credentials to version control**:
- Use environment variables or AWS CLI configuration
- Use IAM roles when running on EC2 instances
- Use temporary credentials (STS) when possible
- Rotate credentials regularly

### Bucket Access Control

Configure bucket policies to restrict access:
- Enable bucket versioning for accident recovery
- Use server-side encryption (SSE-S3 or SSE-KMS)
- Configure bucket policies to restrict public access
- Enable access logging for audit trails

## Integration Examples

### Multi-GPU Training with S3

```bash
python scripts/train_and_compare.py \
    --experiment-name "distributed_training" \
    --architectures full_hgt \
    --num-gpus 4 \
    --num-envs 128 \
    --total-timesteps 50000000 \
    --s3-bucket "my-npp-rl-bucket" \
    --s3-prefix "experiments/multi_gpu/" \
    --record-eval-videos \
    --mixed-precision \
    --output-dir experiments/
```

### Curriculum Learning with S3

```bash
python scripts/train_and_compare.py \
    --experiment-name "curriculum_learning" \
    --architectures vision_free \
    --use-curriculum \
    --curriculum-start-stage 0 \
    --curriculum-threshold 0.7 \
    --total-timesteps 10000000 \
    --s3-bucket "my-npp-rl-bucket" \
    --s3-prefix "experiments/curriculum/" \
    --record-eval-videos \
    --output-dir experiments/
```

### Pretraining + RL with S3

```bash
python scripts/train_and_compare.py \
    --experiment-name "bc_pretraining" \
    --architectures full_hgt \
    --replay-data-dir ./datasets/replays/ \
    --bc-epochs 10 \
    --total-timesteps 5000000 \
    --s3-bucket "my-npp-rl-bucket" \
    --s3-prefix "experiments/pretraining/" \
    --record-eval-videos \
    --output-dir experiments/
```

## Summary

The NPP-RL training system provides comprehensive artifact management with automatic S3 uploads:
- ✅ All checkpoints and models uploaded
- ✅ Complete TensorBoard metrics preserved
- ✅ Evaluation results and videos archived
- ✅ Route visualizations saved for analysis
- ✅ Manifest tracking for reproducibility
- ✅ No manual intervention required

Simply provide S3 credentials and bucket configuration, and all artifacts are automatically backed up throughout training.
