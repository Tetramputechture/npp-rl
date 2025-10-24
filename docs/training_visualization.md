# Real-Time Training Visualization

The NPP-RL training system now includes real-time visualization capabilities, allowing you to see what the agent is doing during training.

## Overview

The training visualization system renders one of the training environments in real-time as the agent learns. This helps you:

- Monitor agent behavior during training
- Debug training issues visually
- Observe how the agent improves over time
- Identify problematic behaviors early

## Usage

### Basic Usage

Add the `--visualize-training` flag to enable visualization:

```bash
python scripts/train_and_compare.py \
    --experiment-name my_experiment \
    --architectures gnn \
    --train-dataset data/train/ \
    --test-dataset data/test/ \
    --visualize-training
```

### Configuration Options

#### Render Frequency

Control how often frames are rendered (default: every 100 timesteps):

```bash
--vis-render-freq 50  # Render every 50 timesteps
```

Higher frequency = more responsive but may slow training slightly.
Lower frequency = better performance but less frequent updates.

#### Environment Selection

Choose which environment to visualize (default: environment 0):

```bash
--vis-env-idx 2  # Visualize the 3rd environment
```

Useful if you want to see a specific environment in your vectorized setup.

#### Frame Rate Limiting

Control the visualization frame rate (default: 60 FPS):

```bash
--vis-fps 30  # Limit to 30 FPS
--vis-fps 0   # Unlimited (render as fast as possible)
```

Lower FPS reduces CPU usage. Set to 0 for maximum responsiveness.

### Complete Example

```bash
python scripts/train_and_compare.py \
    --experiment-name vis_test \
    --architectures gnn \
    --train-dataset data/train/ \
    --test-dataset data/test/ \
    --num-envs 4 \
    --total-timesteps 100000 \
    --visualize-training \
    --vis-render-freq 50 \
    --vis-env-idx 0 \
    --vis-fps 60
```

## Interactive Controls

While the visualization window is open, you can use these keyboard controls:

- **SPACE**: Pause/unpause the visualization (training continues)
- **ESC** or **Q**: Close the visualization window (training continues)

## Performance Considerations

### Number of Environments

For best visualization performance, use **4 or fewer environments**:

```bash
--num-envs 4
```

When using more than 4 environments, the system uses `SubprocVecEnv` (multiprocessing), which can cause issues with pygame rendering across processes. The system will warn you if this happens.

### Training Speed Impact

Visualization has minimal impact on training speed:

- **Render frequency**: Higher frequency increases overhead slightly
- **Frame rate limiting**: Lower FPS reduces CPU usage
- **Pausing**: Use SPACE to pause visualization while keeping training running

### Multi-GPU Training

Visualization is only supported on the main GPU (rank 0). When using multi-GPU training:

```bash
--num-gpus 2 --visualize-training  # Only GPU 0 will show visualization
```

## Troubleshooting

### "Cannot access individual environments"

This occurs when:
1. Using SubprocVecEnv with >4 environments
2. Solution: Use `--num-envs 4` or less

### Window freezes or doesn't respond

This can happen with multiprocessing. Solutions:
1. Reduce number of environments: `--num-envs 4`
2. Lower render frequency: `--vis-render-freq 500`
3. Use DummyVecEnv (automatically used with ≤4 environments)

### Visualization is too fast/slow

Adjust the frame rate:
- Too fast: `--vis-fps 30` (reduce FPS)
- Too slow: `--vis-fps 0` (unlimited) or increase `--vis-render-freq`

### Training is slower with visualization

This is normal. To minimize impact:
1. Increase render frequency: `--vis-render-freq 500`
2. Lower FPS: `--vis-fps 30`
3. Pause visualization: Press SPACE during training

## Technical Details

### Implementation

The visualization system consists of:

1. **TrainingVisualizationCallback**: A Stable Baselines3 callback that triggers rendering
2. **Environment Setup**: Configures one environment with `render_mode="human"`
3. **Pygame Integration**: Handles window display and event processing

### Architecture

```
Training Loop (SB3 PPO)
    ↓
TrainingVisualizationCallback._on_step()
    ↓
Environment.render()  (render_mode="human")
    ↓
NSimRenderer.draw()  (pygame display)
```

### Environment Wrappers

The visualization system correctly handles:
- CurriculumEnv wrapper
- FrameStack wrapper
- CurriculumVecEnvWrapper

It automatically unwraps these to find the base environment's render method.

## Examples

### Quick Test (CPU)

```bash
python scripts/train_and_compare.py \
    --experiment-name quick_vis \
    --architectures simple_mlp \
    --train-dataset data/train/ \
    --test-dataset data/test/ \
    --num-envs 2 \
    --total-timesteps 10000 \
    --visualize-training \
    --vis-render-freq 20
```

### Production Training with Occasional Visualization

```bash
python scripts/train_and_compare.py \
    --experiment-name production \
    --architectures gnn \
    --train-dataset data/train/ \
    --test-dataset data/test/ \
    --num-envs 4 \
    --total-timesteps 10000000 \
    --visualize-training \
    --vis-render-freq 1000 \  # Less frequent for minimal overhead
    --vis-fps 30
```

### Debugging a Specific Environment

```bash
python scripts/train_and_compare.py \
    --experiment-name debug \
    --architectures gnn \
    --train-dataset data/train/ \
    --test-dataset data/test/ \
    --num-envs 4 \
    --visualize-training \
    --vis-env-idx 2 \  # Watch environment 2 specifically
    --vis-render-freq 10 \  # Very frequent updates
    --vis-fps 0  # Unlimited FPS
```

## Future Enhancements

Potential improvements for future versions:

- [ ] Multi-environment grid view (show all environments simultaneously)
- [ ] Recording visualization to video during training
- [ ] Overlay training metrics on visualization
- [ ] Support for remote visualization (training on server, viewing on client)
- [ ] Configurable debug overlays showing agent observations/actions

