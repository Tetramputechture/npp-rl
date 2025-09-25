# Visual Data Augmentation System

## Overview

The NPP-RL system includes a comprehensive visual data augmentation pipeline designed specifically for reinforcement learning applications. The augmentation system is implemented in `nclone/gym_environment/frame_augmentation.py` and is based on research from leading visual RL methods including RAD, DrQ, and DrQ-v2.

## Key Features

### Research-Backed Augmentations

The pipeline includes the most effective augmentation techniques for visual RL:

1. **Random Translation/Shift** - Most stable and effective for RL training
2. **Color Jitter** - Improves generalization across visual variations  
3. **Cutout** - Encourages focus on global context rather than local features
4. **Gaussian Noise** - Improves robustness to sensor noise
5. **Advanced Augmentations** - Optional texture invariance techniques

### Temporal Consistency

Critical for RL applications, the system ensures that all frames in a temporal stack receive identical augmentations to preserve temporal relationships between consecutive time steps.

### Configurable Intensity

The system supports three intensity levels:
- **Light**: Conservative augmentation for stable early training
- **Medium**: Standard augmentation for main training phase  
- **Strong**: Aggressive augmentation for final generalization

## Usage

### Basic Usage

```python
from nclone.gym_environment.frame_augmentation import apply_augmentation

# Apply augmentation to a single frame
augmented_frame, replay_params = apply_augmentation(
    frame,
    p=0.5,                    # Probability of applying each augmentation
    intensity="medium",       # Intensity level
    enable_advanced=False     # Whether to include advanced augmentations
)
```

### Consistent Frame Stack Augmentation

```python
from nclone.gym_environment.frame_augmentation import apply_consistent_augmentation

# Apply same augmentation to all frames in a stack
augmented_frames = apply_consistent_augmentation(
    frame_stack,
    p=0.5,
    intensity="medium",
    enable_advanced=False
)
```

### Training Stage Configurations

```python
from nclone.gym_environment.frame_augmentation import get_recommended_config

# Get recommended configuration for training stage
early_config = get_recommended_config("early")   # Conservative
mid_config = get_recommended_config("mid")       # Standard  
late_config = get_recommended_config("late")     # Aggressive
```

### Integration with ObservationProcessor

The augmentation system is automatically integrated with the `ObservationProcessor`:

```python
from nclone.gym_environment.observation_processor import ObservationProcessor

# Initialize with augmentation enabled
processor = ObservationProcessor(
    enable_augmentation=True,
    augmentation_config=get_recommended_config("mid")
)

# Update configuration during training
processor.update_augmentation_config(training_stage="late")
```

## Technical Details

### Augmentation Components

#### 1. Random Translation (Affine Transform)
- **Purpose**: Most effective augmentation for visual RL
- **Implementation**: Small pixel shifts (4 pixels for 84x84 frames)
- **Stability**: No rotation or scaling to avoid training instability
- **Probability**: 80% of base probability (highest priority)

#### 2. Color Jitter
- **Purpose**: Generalization across lighting and color variations
- **Parameters**: Brightness, contrast, saturation, hue adjustments
- **Intensity Scaling**: Adjusts based on intensity level
- **Probability**: 60% of base probability

#### 3. Cutout (CoarseDropout)
- **Purpose**: Encourages global context learning
- **Implementation**: 1-2 rectangular holes per frame
- **Size**: 8-16 pixels (scaled by intensity)
- **Probability**: 50% of base probability

#### 4. Gaussian Noise
- **Purpose**: Robustness to sensor noise and variations
- **Implementation**: Per-channel noise addition
- **Variance**: 5-15 (scaled by intensity)
- **Probability**: 30% of base probability

#### 5. Advanced Augmentations (Optional)
- **Purpose**: Texture invariance (use with caution)
- **Implementation**: Additional brightness/contrast adjustments
- **Stability**: Can cause training instability if overused
- **Probability**: 20% of base probability

### Performance Considerations

- **Caching**: Augmentation pipelines are cached using `functools.lru_cache`
- **Replay Capability**: All augmentations can be replayed with exact parameters
- **Memory Efficiency**: Minimal memory overhead with in-place operations where possible
- **Speed**: Optimized for real-time RL training scenarios

## Research Background

The augmentation techniques are based on extensive research in visual reinforcement learning:

### Key Papers

1. **RAD (Laskin et al., 2020)**: "Reinforcement Learning with Augmented Data"
   - Demonstrated effectiveness of simple augmentations in RL
   - Showed that translation/crop is most stable and effective

2. **DrQ (Kostrikov et al., 2020)**: "Image Augmentation Is All You Need"
   - Proved that augmentation alone can achieve strong performance
   - Introduced regularization techniques for stable training

3. **DrQ-v2 (Yarats et al., 2021)**: "Mastering Visual Continuous Control"
   - Optimized augmentation pipeline for continuous control
   - Demonstrated importance of consistent augmentation across time steps

### Key Findings

- **Translation/Shift**: Most effective single augmentation for RL
- **Temporal Consistency**: Critical for maintaining temporal relationships
- **Intensity Scheduling**: Different intensities work better at different training stages
- **Stability**: Avoid rotation and large crops which can cause training instability

## Configuration Examples

### Early Training (Conservative)
```python
config = {
    "p": 0.3,
    "intensity": "light", 
    "enable_advanced": False
}
```

### Main Training (Standard)
```python
config = {
    "p": 0.5,
    "intensity": "medium",
    "enable_advanced": False  
}
```

### Final Training (Aggressive)
```python
config = {
    "p": 0.7,
    "intensity": "strong",
    "enable_advanced": True
}
```

## Best Practices

1. **Start Conservative**: Begin with light augmentation to ensure stable training
2. **Increase Gradually**: Ramp up intensity as training progresses
3. **Monitor Performance**: Watch for training instability with aggressive settings
4. **Temporal Consistency**: Always use consistent augmentation for frame stacks
5. **Reproducibility**: Use fixed seeds for debugging and analysis
6. **Advanced Features**: Use advanced augmentations sparingly and monitor carefully

## Integration Points

The augmentation system integrates with:

- **ObservationProcessor**: Automatic augmentation of frame stacks
- **HGTMultimodalExtractor**: Augmented visual features for the HGT model
- **Training Pipeline**: Configurable augmentation scheduling during training
- **Evaluation**: Ability to disable augmentation for consistent evaluation

## Future Enhancements

Potential improvements to consider:

1. **Adaptive Augmentation**: Automatically adjust intensity based on training progress
2. **Task-Specific Augmentation**: Different augmentation strategies for different level types
3. **Learned Augmentation**: Use AutoAugment or similar techniques to learn optimal policies
4. **Multi-Scale Augmentation**: Different augmentation for different visual scales
5. **Adversarial Augmentation**: Augmentations that specifically target model weaknesses