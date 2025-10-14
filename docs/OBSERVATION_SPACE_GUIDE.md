# NPP-RL Observation Space Integration Guide

## Overview

This guide explains how the NPP-RL project integrates with the comprehensive observation space provided by nclone. It covers feature extractors, architecture configurations, and best practices for multi-modal RL training.

## Observation Space Summary

The nclone environment provides 5 observation modalities:

1. **Visual**: Player-centered frames (84×84×12) and global view (176×100×1)
2. **Game State**: Physics vector (26 core + entity states)
3. **Reachability**: Path planning features (8 dimensions)
4. **Graph**: GNN-compatible structure (nodes, edges, masks)
5. **Entity Positions**: Direct position information (6 dimensions)

See [nclone/OBSERVATION_SPACE_README.md](../../nclone/OBSERVATION_SPACE_README.md) for detailed specifications.

## Architecture Support

### Current Architectures

NPP-RL supports the following architectures in `npp_rl/optimization/architecture_configs.py`:

| Architecture | Visual | State | Reachability | Graph | Description |
|---|---|---|---|---|---|
| `full_hgt` | ✓ | ✓ | ✓ | ✓ HGT | All modalities with HGT |
| `simplified_hgt` | ✓ | ✓ | ✓ | ✓ Simple | Lightweight HGT variant |
| `gat` | ✓ | ✓ | ✓ | ✓ GAT | Graph Attention Network |
| `gcn` | ✓ | ✓ | ✓ | ✓ GCN | Graph Convolutional Network |
| `mlp_baseline` | - | ✓ | ✓ | - | MLP only (no vision/graph) |
| `vision_free` | - | ✓ | ✓ | ✓ | Graph + state only |
| `no_global_view` | Local only | ✓ | ✓ | ✓ | No global vision |
| `local_frames_only` | Local only | ✓ | ✓ | - | CNN + state only |

### Feature Extractors

#### HGT Multimodal Extractor

Located in `npp_rl/feature_extractors/hgt_multimodal.py`:

```python
class HGTMultiModalExtractor(BaseFeaturesExtractor):
    """
    Multimodal feature extractor with Heterogeneous Graph Transformer.
    
    Supports:
    - 3D CNN for temporal frames
    - 2D CNN for global view
    - MLP for game state + reachability
    - HGT for graph representation
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Visual encoders
        if 'player_frame' in observation_space.spaces:
            self.cnn_temporal = self._build_3d_cnn()
        
        if 'global_view' in observation_space.spaces:
            self.cnn_global = self._build_2d_cnn()
        
        # State encoder
        state_dim = (observation_space['game_state'].shape[0] +
                    observation_space['reachability_features'].shape[0] +
                    observation_space['entity_positions'].shape[0])
        self.mlp_state = self._build_mlp(state_dim)
        
        # Graph encoder
        if 'graph_node_feats' in observation_space.spaces:
            self.hgt = HGTEncoder(
                node_feat_dim=observation_space['graph_node_feats'].shape[1],
                edge_feat_dim=observation_space['graph_edge_feats'].shape[1],
                hidden_dim=256,
                num_layers=3,
                output_dim=256
            )
        
        # Fusion
        self.fusion = self._build_fusion_layer(features_dim)
```

#### Vision-Free Extractor

Located in `npp_rl/feature_extractors/vision_free_extractor.py`:

```python
class VisionFreeExtractor(BaseFeaturesExtractor):
    """
    Feature extractor without visual modalities.
    
    Uses only:
    - Game state vector
    - Reachability features
    - Graph representation (optional)
    """
    
    def forward(self, observations):
        # Combine non-visual modalities
        state_feat = self.mlp_state(torch.cat([
            observations['game_state'],
            observations['reachability_features'],
            observations['entity_positions']
        ], dim=1))
        
        if self.use_graph:
            graph_feat = self.gnn(
                observations['graph_node_feats'],
                observations['graph_edge_index'],
                observations['graph_node_mask'],
                observations['graph_edge_mask']
            )
            return self.fusion(torch.cat([state_feat, graph_feat], dim=1))
        
        return state_feat
```

## Node and Edge Feature Dimensions

### Current Implementation

**Node Features**: Currently 3 dimensions per node
- `x_position`: Normalized X coordinate
- `y_position`: Normalized Y coordinate  
- `node_type`: Type encoding (0-5)

**Edge Features**: Currently 1 dimension per edge
- `weight`: Traversal cost

### Enhanced Implementation (Future)

**Node Features**: 67 dimensions per node (see nclone/graph/feature_builder.py)
- Spatial (3): position, resolution
- Type (6): one-hot encoding
- Entity (10): type, state, radius, activation
- Tile (38): full tile type encoding
- Reachability (8): path information
- Proximity (2): distances to key points

**Edge Features**: 9 dimensions per edge
- Type (4): one-hot encoding
- Movement (5): requirements and costs

To upgrade feature extractors for enhanced features:

```python
# Update HGT encoder configuration
self.hgt = HGTEncoder(
    node_feat_dim=67,  # Enhanced node features
    edge_feat_dim=9,   # Enhanced edge features
    hidden_dim=256,
    num_layers=3,
    output_dim=256
)
```

## Training with Different Modalities

### Full Multimodal (Recommended)

```bash
python -m npp_rl.agents.training \
    --architecture full_hgt \
    --num_envs 64 \
    --total_timesteps 10000000 \
    --enable_graph \
    --enable_reachability
```

### Vision-Free (State + Graph)

```bash
python -m npp_rl.agents.training \
    --architecture vision_free \
    --num_envs 64 \
    --total_timesteps 10000000 \
    --enable_graph \
    --enable_reachability
```

### MLP Baseline (State Only)

```bash
python -m npp_rl.agents.training \
    --architecture mlp_baseline \
    --num_envs 64 \
    --total_timesteps 10000000 \
    --disable_graph
```

## Architecture Comparison

Use the comparison script to benchmark multiple architectures:

```bash
python scripts/train_and_compare.py \
    --experiment-name "modality_comparison" \
    --architectures full_hgt vision_free mlp_baseline \
    --train-dataset datasets/train \
    --test-dataset datasets/test \
    --total-timesteps 5000000 \
    --num-seeds 3
```

This will train each architecture and generate comparison metrics:
- Sample efficiency curves
- Final performance statistics
- Wall-clock time comparisons
- Architecture-specific insights

## Custom Feature Extractors

To create a custom feature extractor:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Build your custom architecture
        # Access observation spaces:
        #   observation_space['player_frame'].shape
        #   observation_space['game_state'].shape
        #   observation_space['graph_node_feats'].shape
        # etc.
        
        self._features_dim = features_dim
    
    def forward(self, observations):
        # Process observations and return features
        # Must return tensor of shape (batch_size, features_dim)
        pass
```

Register in architecture configs:

```python
# In npp_rl/optimization/architecture_configs.py

def create_custom_config():
    return ArchitectureConfig(
        name="custom",
        description="Custom architecture",
        modalities=ModalityConfig(
            use_temporal_frames=True,
            use_global_view=True,
            use_graph=True,
            use_game_state=True,
            use_reachability=True
        ),
        features_dim=512,
        # ... other config
    )

ARCHITECTURE_REGISTRY["custom"] = create_custom_config()
```

## Information Completeness

All architectures must have access to sufficient information for level completion:

**Required Information**:
1. ✅ Ninja position → Vision, State, Graph
2. ✅ Goal positions → Vision, State, Positions
3. ✅ Obstacle locations → Vision, Graph
4. ✅ Entity states (mines, doors) → State, Graph
5. ✅ Movement physics → State (velocity, contacts, buffers)
6. ✅ Reachability → Reachability features, Graph
7. ✅ Path planning → Vision, Graph, Reachability

**Minimum Viable Modalities**:
- Vision-only: Difficult, lacks explicit state
- State-only: Feasible, lacks spatial context
- Graph-only: Difficult, lacks fine-grained movement
- **State + Reachability**: Baseline viable
- **Vision + State**: Strong baseline
- **All modalities**: Best performance (recommended)

## Best Practices

### 1. Start with Full Multimodal
Begin training with all modalities enabled to establish a performance ceiling.

### 2. Ablate Systematically
Remove one modality at a time to understand its contribution:
```bash
# Full
full_hgt

# Without global vision  
no_global_view

# Without any vision
vision_free

# Without graph
local_frames_only

# Minimal
mlp_baseline
```

### 3. Monitor Modality Usage
Use attention weights or gradient magnitudes to see which modalities the agent relies on most.

### 4. Consider Compute Tradeoffs
- **GNN (HGT)**: Expensive but powerful for structure
- **3D CNN**: Moderate cost, good for motion
- **2D CNN**: Cheap, useful for global strategy
- **MLP**: Very cheap, essential for state

### 5. Normalize Consistently
Ensure all features are normalized appropriately:
- Visual: Already uint8 [0, 255]
- State: Normalized to [-1, 1] or [0, 1]
- Graph nodes: Mixed (positions [0,1], one-hot, etc.)
- Graph edges: [0, 1]

## Troubleshooting

### "Graph features have wrong dimension"
Check that your GNN expects the correct node/edge feature dimensions:
```python
# If using enhanced features
node_feat_dim = 67
edge_feat_dim = 9

# If using basic features (current)
node_feat_dim = 3
edge_feat_dim = 1
```

### "Observation space mismatch"
Verify environment configuration matches feature extractor expectations:
```python
env = NPPEnvironment(
    enable_graph_updates=True,  # Required for graph obs
    enable_reachability=True,   # Required for reachability obs
)
```

### "Out of memory during training"
Reduce graph size or batch size:
```python
# In training script
--num_envs 32  # Instead of 64
--batch_size 128  # Instead of 256
```

## Future Work

### Planned Enhancements
1. **Richer node features** (67 dim) - Already implemented in `nclone/graph/feature_builder.py`
2. **Richer edge features** (9 dim) - Already implemented
3. **Locked door tracking** - Explicit door/switch state tracking
4. **Hierarchical graphs** - Multi-resolution (6px, 24px, 96px)
5. **Attention fusion** - Replace concatenation with cross-modal attention

### Migration Path
When enhanced features are integrated into the main graph builder:

1. Update `N_MAX_NODES`, `NODE_FEATURE_DIM`, `EDGE_FEATURE_DIM` in nclone
2. Update feature extractor configurations in npp-rl
3. Retrain all architectures with new feature dimensions
4. Compare performance before/after enhancement

## References

- nclone observation space: `nclone/OBSERVATION_SPACE_README.md`
- Architecture configs: `npp_rl/optimization/architecture_configs.py`
- Feature extractors: `npp_rl/feature_extractors/`
- Training guide: `docs/TRAINING_SYSTEM.md`
- Architecture comparison: `docs/ARCHITECTURE_COMPARISON_GUIDE.md`
