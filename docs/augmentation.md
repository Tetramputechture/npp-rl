# Visual Data Augmentation System

## Overview

The NPP-RL system includes a comprehensive visual data augmentation pipeline designed specifically for game environments and reinforcement learning applications. The augmentation system is implemented in `nclone/gym_environment/frame_augmentation.py` and is based on research from leading visual RL methods including RAD, DrQ, DrQ-v2, and game-specific studies.

## Key Features

### Game-Optimized Augmentations

The pipeline includes the most effective augmentation techniques for visual game environments:

1. **Random Translation/Shift** - Most stable and effective for RL training, helps with position invariance
2. **Horizontal Flipping** - Leverages symmetry in platformer games like N++
3. **Cutout** - Encourages focus on global context rather than local features
4. **Brightness/Contrast Variations** - Subtle visual robustness without destroying game aesthetics
5. **Advanced Augmentations** - Optional color variations for RGB games

**Note**: Gaussian noise is intentionally excluded as it's inappropriate for clean game visuals, unlike sensor data applications.

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
    enable_advanced=False,    # Whether to include advanced augmentations
    game_symmetric=True       # Enable horizontal flipping for symmetric games
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
    enable_advanced=False,
    game_symmetric=True       # Enable horizontal flipping for symmetric games
)
```

### Training Stage Configurations

```python
from nclone.gym_environment.frame_augmentation import get_recommended_config

# Get recommended configuration for training stage and game type
early_config = get_recommended_config("early", "platformer")   # Conservative
mid_config = get_recommended_config("mid", "platformer")       # Standard  
late_config = get_recommended_config("late", "platformer")     # Moderate (games need less than sensor data)
```

### Integration with ObservationProcessor

The augmentation system is automatically integrated with the `ObservationProcessor`:

```python
from nclone.gym_environment.observation_processor import ObservationProcessor

# Initialize with augmentation enabled (defaults to platformer game type)
processor = ObservationProcessor(
    enable_augmentation=True,
    augmentation_config=get_recommended_config("mid", "platformer")
)

# Update configuration during training
processor.update_augmentation_config(training_stage="late", game_type="platformer")
```

## Technical Details

### Augmentation Components

#### 1. Random Translation (Affine Transform)
- **Purpose**: Most effective augmentation for visual RL, helps with position invariance
- **Implementation**: Small pixel shifts (4 pixels for 84x84 frames)
- **Stability**: No rotation or scaling to avoid training instability
- **Probability**: 80% of base probability (highest priority)

#### 2. Horizontal Flip
- **Purpose**: Leverages symmetry in platformer games like N++
- **Implementation**: 50% chance horizontal flip when game_symmetric=True
- **Game-Specific**: Only enabled for symmetric games
- **Probability**: 40% of base probability

#### 3. Cutout (CoarseDropout)
- **Purpose**: Encourages global context learning
- **Implementation**: 1-2 rectangular holes per frame
- **Size**: 6-12 pixels (smaller for games than sensor data)
- **Probability**: 50% of base probability

#### 4. Brightness/Contrast Variations
- **Purpose**: Subtle visual robustness without destroying game aesthetics
- **Implementation**: Small brightness and contrast adjustments
- **Game-Optimized**: Much subtler than sensor data applications
- **Probability**: 40% of base probability

#### 5. Advanced Augmentations (Optional)
- **Purpose**: Color variations for RGB games (minimal effect on grayscale)
- **Implementation**: Very subtle color jitter and optional grayscale conversion
- **Caution**: Use sparingly as games have consistent visual style
- **Probability**: 10-20% of base probability

### Game-Specific Considerations

#### Why Games Need Different Augmentations Than Sensor Data

1. **Clean Visuals**: Games have crisp, deterministic graphics unlike noisy sensor data
2. **Consistent Lighting**: Game environments have predictable lighting conditions
3. **Symmetric Mechanics**: Many platformers can be approached from either direction
4. **Aesthetic Preservation**: Aggressive augmentation can destroy important visual cues

#### Game Type Recommendations

- **Platformer Games** (like N++): Enable horizontal flipping, moderate augmentation
- **Puzzle Games**: Disable flipping, reduce overall augmentation intensity
- **Action Games**: Careful augmentation to preserve reaction timing cues

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

4. **Procgen Study (Raileanu et al., 2021)**: "Automatic Data Augmentation for Generalization in Reinforcement Learning"
   - Studied augmentation effectiveness across different game types
   - Showed that different games benefit from different augmentation strategies

### Key Findings

- **Translation/Shift**: Most effective single augmentation for RL
- **Game Symmetry**: Horizontal flipping highly effective for symmetric games
- **Temporal Consistency**: Critical for maintaining temporal relationships
- **Intensity Scheduling**: Different intensities work better at different training stages
- **Game-Specific**: Clean game visuals need different treatment than sensor data
- **Stability**: Avoid rotation and large crops which can cause training instability

## Configuration Examples

### Early Training (Conservative)
```python
config = {
    "p": 0.3,
    "intensity": "light", 
    "enable_advanced": False,
    "game_symmetric": True
}
```

### Main Training (Standard)
```python
config = {
    "p": 0.5,
    "intensity": "medium",
    "enable_advanced": False,
    "game_symmetric": True
}
```

### Final Training (Moderate for Games)
```python
config = {
    "p": 0.6,  # Lower than sensor data applications
    "intensity": "strong",
    "enable_advanced": True,
    "game_symmetric": True
}
```

## Best Practices

1. **Start Conservative**: Begin with light augmentation to ensure stable training
2. **Increase Gradually**: Ramp up intensity as training progresses (but less than sensor data)
3. **Monitor Performance**: Watch for training instability with aggressive settings
4. **Temporal Consistency**: Always use consistent augmentation for frame stacks
5. **Game-Specific Tuning**: Adjust symmetry and intensity based on game type
6. **Preserve Aesthetics**: Avoid destroying important visual game cues
7. **Reproducibility**: Use fixed seeds for debugging and analysis
8. **Advanced Features**: Use advanced augmentations sparingly and monitor carefully

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