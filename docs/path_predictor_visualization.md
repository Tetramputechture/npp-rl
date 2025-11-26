# Path Predictor Training Visualization

## Overview

The path predictor training script now automatically generates visualizations of model performance after training completes. This helps you quickly assess how well your trained model is learning to predict paths.

## Features

### Automatic Visualization After Training

When training completes, the script can automatically:
- Load validation samples
- Generate path predictions using the final trained model
- Visualize predicted paths vs. expert (ground truth) paths
- Save visualizations as PNG images

### What's Visualized

Each visualization shows:
- **Tiles**: Level geometry rendered as solid blocks
- **Expert Path (Ground Truth)**: Green dashed line with square markers showing the actual path from the replay
- **Predicted Paths**: Up to 4 candidate paths with different colors, each showing:
  - Waypoint sequence as colored dots
  - Lines connecting waypoints
  - Confidence score for each path
  - Start position (green circle)
  - Goal position (red star)

## Usage

### Command Line

Enable visualization with the `--visualize-after-training` flag:

```bash
python scripts/train_path_predictor.py \
    --mode offline \
    --replay-dir datasets/expert_replays/ \
    --val-replay-dir datasets/expert_replays_val/ \
    --output-dir models/path_predictor/ \
    --num-epochs 50 \
    --batch-size 32 \
    --visualize-after-training \
    --num-viz-samples 5
```

### Configuration File

You can also enable visualization in your config YAML:

```yaml
# configs/path_predictor_training.yaml
visualization:
  # Generate visualizations after training completes
  visualize_after_training: true
  
  # Number of validation samples to visualize
  num_viz_samples: 5
  
  # Include expert paths in visualizations
  show_expert_paths: true
  
  # Show confidence scores in visualizations
  show_confidence_scores: true
```

Then run with the config:

```bash
python scripts/train_path_predictor.py \
    --config configs/path_predictor_training.yaml \
    --mode offline \
    --replay-dir datasets/expert_replays/ \
    --val-replay-dir datasets/expert_replays_val/
```

## Output

Visualizations are saved to:
```
<output-dir>/visualizations/validation_sample_001.png
<output-dir>/visualizations/validation_sample_002.png
...
```

## Interpreting Visualizations

### Good Predictions
- Predicted paths closely follow the expert path (green dashed line)
- High confidence scores (close to 1.0)
- Multiple diverse candidate paths covering different strategies
- Paths stay within playable areas (don't go through walls)

### Poor Predictions
- Predicted paths diverge significantly from expert path
- Low confidence scores (close to 0.0)
- Paths that go through walls or obstacles
- All candidates very similar (low diversity)

## Training Progress

The script also saves:
- **best_model.pt**: Model with lowest validation loss
- **final_model.pt**: Model after final epoch
- **checkpoint_epoch_N.pt**: Periodic checkpoints
- **training_stats.json**: Loss curves and metrics
- **visualizations/**: Path prediction visualizations

## Examples

### Example 1: Well-Trained Model

```
Validation Sample 1/5
  - Pred 1 (conf=0.92): Closely matches expert path
  - Pred 2 (conf=0.81): Alternative route avoiding obstacles
  - Pred 3 (conf=0.65): Conservative path with more waypoints
  - Pred 4 (conf=0.43): Exploratory path
```

### Example 2: Undertrained Model

```
Validation Sample 1/5
  - Pred 1 (conf=0.23): Random waypoints, poor path quality
  - Pred 2 (conf=0.19): Goes through walls
  - Pred 3 (conf=0.15): Very short path
  - Pred 4 (conf=0.12): No clear strategy
```

## Tips

1. **Start Small**: Use `--num-viz-samples 3` for quick checks
2. **Validate Diversity**: Look for diverse candidate paths, not just copies
3. **Check Confidence**: Well-calibrated models should have higher confidence for better paths
4. **Compare Epochs**: Visualize at different checkpoints to see learning progress

## Advanced Usage

### Visualize Specific Checkpoint

Use the evaluation script to visualize any checkpoint:

```bash
python scripts/evaluate_path_predictor.py \
    --checkpoint models/path_predictor/checkpoint_epoch_20.pt \
    --num-levels 5 \
    --output-dir visualizations/epoch_20/
```

### Batch Visualization

Visualize multiple checkpoints:

```bash
for epoch in 10 20 30 40 50; do
    python scripts/evaluate_path_predictor.py \
        --checkpoint models/path_predictor/checkpoint_epoch_${epoch}.pt \
        --num-levels 3 \
        --output-dir visualizations/epoch_${epoch}/
done
```

## Troubleshooting

### No Visualizations Generated

- Check that `--val-replay-dir` is provided (required for validation)
- Ensure validation dataset has samples
- Check logs for errors during visualization

### Poor Quality Visualizations

- Increase training epochs
- Adjust loss weights in config
- Verify expert replays are high-quality
- Check that replay dataset is loading correctly

### Memory Issues

- Reduce `--num-viz-samples`
- Use smaller `--batch-size`
- Close visualizations are generated (they're saved automatically)

