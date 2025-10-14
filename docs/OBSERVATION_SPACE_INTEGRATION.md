# Observation Space Integration Guide for NPP-RL

## Overview

This document describes how npp-rl integrates with the nclone observation space for training RL agents. The observation space provides multiple modalities designed to support different architectural approaches (CNNs, MLPs, GNNs).

## Observation Modalities

The nclone observation space provides:

### 1. Visual Observations
- **player_frame**: `(84, 84, 12)` - Local player-centered view with 12-frame temporal stacking
- **global_view**: `(176, 100, 1)` - Downsampled full-level overview

### 2. Game State Vector  
- **game_state**: `(32,)` float32 - Physics and entity state
  - Ninja position, velocity, movement state
  - Jump state, wall contact, ground contact
  - Entity positions and states
  - Goal information

### 3. Graph Representation (for GNN architectures)
- **graph_obs**: Dictionary containing:
  - `node_features`: `(N, 56)` - Comprehensive node features
  - `edge_features`: `(E, 6)` - Edge features
  - `edge_index`: `(2, E)` - Edge connectivity
  - `node_mask`: `(N,)` - Valid node mask
  - `edge_mask`: `(E,)` - Valid edge mask

## Node Features (56 dimensions)

From `nclone.graph.common.NODE_FEATURE_DIM = 56`:

- **Spatial (3)**: x, y position + resolution level
- **Type (6)**: One-hot [EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT]
- **Entity (5)**: Type, subtype, active, state, radius (reduced from 10)
- **Tile (38)**: One-hot encoding of all tile types
- **Reachability (2)**: From flood-fill - reachable from ninja, on critical path
- **Proximity (2)**: Distance to ninja, distance to goal

## Edge Features (6 dimensions)

From `nclone.graph.common.EDGE_FEATURE_DIM = 6`:

- **Edge Type (4)**: One-hot [ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED]
- **Connectivity (2)**: Weight (distance), reachability confidence

## Using Full Graph Features

To use the comprehensive 56-dimensional node features in your models:

### Option 1: Update HGT Config

```python
# npp_rl/models/hgt_config.py
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM

@dataclass(frozen=True)
class HGTConfig:
    node_feat_dim: int = NODE_FEATURE_DIM  # 56
    edge_feat_dim: int = EDGE_FEATURE_DIM  # 6
    # ... other config
```

### Option 2: Create Custom Config

```python
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from npp_rl.models.hgt_config import HGTConfig

# Create config with full features
full_feature_config = HGTConfig(
    node_feat_dim=NODE_FEATURE_DIM,  # 56
    edge_feat_dim=EDGE_FEATURE_DIM,  # 6
    hidden_dim=256,  # Increase for richer features
    num_heads=8,
    num_layers=3,
    dropout=0.1,
    num_node_types=6,
    num_edge_types=4
)
```

## Architecture Comparison Support

The observation space supports multiple architectural configurations:

| Architecture | Local Frames | Global View | Game State | Graph |
|--------------|--------------|-------------|------------|-------|
| vision_and_state | ✓ | ✓ | ✓ | - |
| graph_only | - | - | ✓ | ✓ |
| vision_graph_hybrid | ✓ | ✓ | ✓ | ✓ |
| no_global_view | ✓ | - | ✓ | ✓ |
| local_frames_only | ✓ | - | ✓ | - |

## Feature Extractors

### Current Implementation

npp-rl currently uses simplified features:
- **node_feat_dim**: 8 (simplified)
- **edge_feat_dim**: 4 (simplified)

### Full Feature Integration

To integrate full 56-dimensional features:

1. **Update model configuration**:
   ```python
   from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM
   
   policy_kwargs = {
       'features_extractor_class': MultiModalFeatureExtractor,
       'features_extractor_kwargs': {
           'node_feat_dim': NODE_FEATURE_DIM,  # 56
           'edge_feat_dim': EDGE_FEATURE_DIM,  # 6
       }
   }
   ```

2. **Ensure graph builder uses new features**:
   ```python
   from nclone.graph.feature_builder import NodeFeatureBuilder, EdgeFeatureBuilder
   
   # These automatically use the 56/6 dimensions
   node_builder = NodeFeatureBuilder()
   edge_builder = EdgeFeatureBuilder()
   ```

3. **Re-train models**: Full features require retraining as architecture changes

## Reachability System

The observation space includes reachability information from nclone's fast flood-fill system:

```python
from nclone.graph.reachability.reachability_system import ReachabilitySystem

# Used internally by nclone graph builder
reachability_sys = ReachabilitySystem()
result = reachability_sys.analyze_reachability(
    level_data=level_data,
    ninja_position=(x, y),
    switch_states=switch_states
)
```

**Key Points**:
- Uses OpenCV flood-fill (<1ms computation)
- Provides connectivity information, NOT physics simulation
- Agent learns movement dynamics from temporal frames
- Integrated automatically in node features (indices 52-53)

## Best Practices

The observation space follows RL/ML best practices:

✅ **Normalized observations**: All features in [0, 1] range  
✅ **Minimal redundancy**: Removed 5 unused entity features (61 → 56 dims)  
✅ **Temporal context**: 12-frame stacking with 3D CNNs  
✅ **Spatial invariance**: CNNs for vision, GNNs for graphs  
✅ **Multi-modal fusion**: Specialized encoders + late fusion  
✅ **Markov property**: Sufficient information for optimal decisions  
✅ **Computational efficiency**: <2ms feature extraction per step  

See `/workspace/RL_BEST_PRACTICES_VALIDATION.md` for detailed validation.

## Entity Support

Supported entity types (as per level constraints):
- ✅ Ninja (player)
- ✅ Exit switch and door
- ✅ Locked doors (up to 16) with switches
- ✅ Toggle mines (up to 256 total)
- ✅ All tile types (38 types)

Not supported:
- ❌ Regular doors (not needed for constrained levels)
- ❌ Other entity types (drones, enemies, etc.)

## Example Usage

### Using with PPO

```python
from stable_baselines3 import PPO
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from npp_rl.models.hgt_config import HGTConfig

# Create environment with graph observations
env = create_npp_env(include_graph=True)

# Configure policy with full features
policy_kwargs = {
    'features_extractor_class': MultiModalFeatureExtractor,
    'features_extractor_kwargs': {
        'use_graph': True,
        'node_feat_dim': NODE_FEATURE_DIM,  # 56
        'edge_feat_dim': EDGE_FEATURE_DIM,  # 6
    }
}

# Train
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1_000_000)
```

### Architecture Comparison

```python
# Compare different observation modalities
architectures = [
    {"name": "vision_only", "use_graph": False, "use_global": True},
    {"name": "graph_only", "use_graph": True, "use_global": False},
    {"name": "hybrid", "use_graph": True, "use_global": True},
]

for arch in architectures:
    env = create_npp_env(**arch)
    model = train_agent(env, arch["name"])
    evaluate_agent(model, env)
```

## References

- **nclone observation space**: `/workspace/nclone/OBSERVATION_SPACE_README.md`
- **Feature builder**: `nclone/graph/feature_builder.py`
- **Constants**: `nclone/graph/common.py`
- **Best practices**: `/workspace/RL_BEST_PRACTICES_VALIDATION.md`
- **Implementation summary**: `/workspace/IMPLEMENTATION_SUMMARY.md`

## Troubleshooting

### Feature Dimension Mismatch

If you get dimension mismatch errors:

```python
# Check nclone constants
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM
print(f"Expected node dims: {NODE_FEATURE_DIM}")  # Should be 56
print(f"Expected edge dims: {EDGE_FEATURE_DIM}")  # Should be 6

# Update your model config
config.node_feat_dim = NODE_FEATURE_DIM
config.edge_feat_dim = EDGE_FEATURE_DIM
```

### Graph Observations Not Available

Ensure environment is configured to provide graph observations:

```python
env = create_npp_env(
    include_graph=True,  # Enable graph observations
    graph_resolution=4,  # Sub-grid resolution
)
```

### Performance Issues

If feature extraction is slow:
- Graph construction: Ensure using hierarchical builder (optimized)
- Reachability: Should be <1ms (uses OpenCV, not Python loops)
- Check that masking is used for variable-size graphs

## Next Steps

1. **Baseline Testing**: Test current simplified features (8/4 dims)
2. **Full Feature Integration**: Upgrade to 56/6 dims when ready
3. **Architecture Comparison**: Ablation studies on modalities
4. **Hyperparameter Tuning**: Adjust network sizes for richer features

For questions or issues, see:
- nclone documentation: `OBSERVATION_SPACE_README.md`
- npp-rl repository: `README.md`
- GitHub issues: nclone and npp-rl repositories
