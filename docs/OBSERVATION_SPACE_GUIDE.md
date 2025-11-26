# NPP-RL Observation Space Integration Guide

## Overview

This guide explains how the NPP-RL project integrates with the comprehensive observation space provided by nclone. It covers feature extractors, architecture configurations, and best practices for multi-modal RL training.

## Observation Space Summary

The nclone environment provides 5 observation modalities:

1. **Visual**: Player-centered frame (84×84×1 grayscale) and global view (176×100×1 grayscale)
2. **Game State**: Physics vector (64 features total: 29 ninja state + 15 path-aware + 8 mine + 3 progress + 3 sequential goal + 6 death probabilities)
3. **Reachability**: Path planning features (8 dimensions)
4. **Graph**: GNN-compatible structure (nodes, edges, masks)
5. **Entity Positions**: Direct position information (6 dimensions)

**Note**: Uses single-frame grayscale system with explicit velocity for 6.66x faster performance.

See [nclone/OBSERVATION_SPACE_README.md](../../nclone/OBSERVATION_SPACE_README.md) for detailed specifications.

## Complete Observation Vector Reference

### `game_state` - Complete Feature Breakdown

The `game_state` vector contains **64 features** total, organized into 6 sections:

**Note on Distance Normalization**: All distance-based features (path distances, relative positions) are normalized using **reachable area scale** (`sqrt(reachable_surface_area) * SUB_NODE_SIZE`) instead of `LEVEL_DIAGONAL`. This provides level-adaptive scaling that accounts for the actual navigable space in each level, resulting in more uniform feature distributions across different level sizes and improving learning stability. The reachable area is computed via flood-fill from the start position and cached per level ID for performance.

#### Indices 0-28: Ninja Physics State (29 features)

All features normalized to range `[-1, 1]` unless otherwise specified.

**Core Movement State (8 features):**

- `[0]` **Velocity magnitude**: Normalized speed `(sqrt(xspeed² + yspeed²) / (MAX_HOR_SPEED * 2)) * 2 - 1`
  - Range: `[-1, 1]` where 1.0 = maximum possible velocity
  - Accounts for both horizontal and vertical velocity components

- `[1]` **Velocity direction X**: Unit vector X component `xspeed / velocity_magnitude`
  - Range: `[-1, 1]` (left = -1, right = 1, zero velocity = 0)

- `[2]` **Velocity direction Y**: Unit vector Y component `yspeed / velocity_magnitude`
  - Range: `[-1, 1]` (down = -1, up = 1, zero velocity = 0)

- `[3]` **Ground movement category**: Binary indicator for ground states
  - Value: `1.0` if ninja.state in [0, 1, 2] (Immobile, Running, Ground sliding), else `-1.0`

- `[4]` **Air movement category**: Binary indicator for air states
  - Value: `1.0` if ninja.state in [3, 4] (Jumping, Falling), else `-1.0`

- `[5]` **Wall interaction category**: Binary indicator for wall sliding
  - Value: `1.0` if ninja.state == 5 (Wall sliding), else `-1.0`

- `[6]` **Special states category**: Binary indicator for terminal/special states
  - Value: `1.0` if ninja.state in [6, 7, 8, 9] (Dead, Awaiting death, Celebrating, Disabled), else `-1.0`

- `[7]` **Airborne status**: Boolean indicator for airborne state
  - Value: `1.0` if `ninja.airborn` is True, else `-1.0`

**Input and Buffer State (5 features):**

- `[8]` **Horizontal input**: Current horizontal movement input
  - Value: `-1.0` (left), `0.0` (none), or `1.0` (right)
  - Directly from `ninja.hor_input`

- `[9]` **Jump input**: Current jump button state
  - Value: `1.0` if jump pressed, else `-1.0`
  - From `ninja.jump_input`

- `[10]` **Jump buffer**: Normalized jump buffer window
  - Value: `(max(jump_buffer, 0) / 5.0) * 2 - 1`
  - Range: `[-1, 1]` where 1.0 = buffer window full (5 frames)
  - Indicates frames remaining for frame-perfect jump execution

- `[11]` **Floor buffer**: Normalized floor buffer window
  - Value: `(max(floor_buffer, 0) / 5.0) * 2 - 1`
  - Range: `[-1, 1]` where 1.0 = buffer window full (5 frames)
  - Indicates frames remaining for frame-perfect floor interaction

- `[12]` **Wall buffer**: Normalized wall buffer window
  - Value: `(max(wall_buffer, 0) / 5.0) * 2 - 1`
  - Range: `[-1, 1]` where 1.0 = buffer window full (5 frames)
  - Indicates frames remaining for frame-perfect wall interaction

**Surface Contact Information (6 features):**

- `[13]` **Floor contact**: Binary indicator for floor contact
  - Value: `(min(floor_count, 1) * 2) - 1` → `1.0` if contact, else `-1.0`
  - From `ninja.floor_count`

- `[14]` **Wall contact**: Binary indicator for wall contact
  - Value: `(min(wall_count, 1) * 2) - 1` → `1.0` if contact, else `-1.0`
  - From `ninja.wall_count`

- `[15]` **Ceiling contact**: Binary indicator for ceiling contact
  - Value: `(min(ceiling_count, 1) * 2) - 1` → `1.0` if contact, else `-1.0`
  - From `ninja.ceiling_count`

- `[16]` **Floor normal strength**: Magnitude of floor normal vector
  - Value: `(sqrt(floor_normalized_x² + floor_normalized_y²) * 2) - 1`
  - Range: `[-1, 1]` where 1.0 = strong floor contact, -1.0 = no floor contact
  - Indicates quality of floor contact

- `[17]` **Wall direction**: Wall contact direction
  - Value: `float(ninja.wall_normal)` if `wall_count > 0`, else `0.0`
  - Range: `[-1, 1]` where -1 = left wall, 1 = right wall, 0 = no wall contact

- `[18]` **Surface slope**: Floor normal Y component (indicates slope angle)
  - Value: `ninja.floor_normalized_y`
  - Range: `[-1, 1]` where -1 = steep downward slope, 1 = steep upward slope, 0 = flat

**Momentum and Physics (2 features):**

- `[19]` **Recent acceleration X**: Change in X velocity normalized by max speed
  - Value: `(xspeed - xspeed_old) / MAX_HOR_SPEED`, clamped to `[-1, 1]`
  - Positive = accelerating right, negative = accelerating left

- `[20]` **Recent acceleration Y**: Change in Y velocity normalized by max speed
  - Value: `(yspeed - yspeed_old) / MAX_HOR_SPEED`, clamped to `[-1, 1]`
  - Positive = accelerating up, negative = accelerating down

**Additional Physics State (8 features):**

- `[21]` **Applied gravity**: Normalized gravity value between GRAVITY_JUMP and GRAVITY_FALL
  - Value: `((applied_gravity - GRAVITY_JUMP) / (GRAVITY_FALL - GRAVITY_JUMP)) * 2 - 1`
  - Range: `[-1, 1]` where -1 = GRAVITY_JUMP (jumping), 1 = GRAVITY_FALL (falling)
  - Gravity is lower during jump, higher during fall

- `[22]` **Jump duration**: Normalized jump duration
  - Value: `min(jump_duration / MAX_JUMP_DURATION, 1.0) * 2 - 1`
  - Range: `[-1, 1]` where 1.0 = maximum jump duration reached

- `[23]` **Walled status**: Boolean indicator for walled state
  - Value: `1.0` if `ninja.walled` is True, else `-1.0`
  - Indicates if ninja is touching a wall

- `[24]` **Floor normal X**: Full X component of floor normal vector
  - Value: `ninja.floor_normalized_x`
  - Range: `[-1, 1]` where -1 = floor slopes left, 1 = floor slopes right, 0 = vertical/horizontal

- `[25]` **Ceiling normal X**: X component of ceiling normal vector
  - Value: `ninja.ceiling_normalized_x`
  - Range: `[-1, 1]` where -1 = ceiling slopes left, 1 = ceiling slopes right

- `[26]` **Ceiling normal Y**: Y component of ceiling normal vector
  - Value: `ninja.ceiling_normalized_y`
  - Range: `[-1, 1]` where -1 = ceiling slopes down, 1 = ceiling slopes up

- `[27]` **Applied drag**: Normalized drag value between DRAG_SLOW and DRAG_REGULAR
  - Value: `((applied_drag - DRAG_SLOW) / (DRAG_REGULAR - DRAG_SLOW)) * 2 - 1`
  - Range: `[-1, 1]` where -1 = DRAG_SLOW, 1 = DRAG_REGULAR
  - Higher drag = more air resistance

- `[28]` **Applied friction**: Normalized friction value between FRICTION_GROUND_SLOW and FRICTION_GROUND
  - Value: `((applied_friction - FRICTION_GROUND_SLOW) / (FRICTION_GROUND - FRICTION_GROUND_SLOW)) * 2 - 1`
  - Range: `[-1, 1]` where -1 = FRICTION_GROUND_SLOW, 1 = FRICTION_GROUND
  - Higher friction = more ground resistance

#### Indices 29-43: Path-Aware Objectives (15 features)

All features normalized to range `[0, 1]` or `[-1, 1]` as specified. Uses graph-based pathfinding for accurate distance calculations.

**Exit Switch Features (4 features):**

- `[29]` **Exit switch collected**: Binary indicator
  - Value: `1.0` if switch activated, else `0.0`
  - From `obs["switch_activated"]`

- `[30]` **Exit switch relative X**: Normalized relative X position
  - Value: `(switch_x - ninja_x) / (LEVEL_WIDTH / 2)`, clamped to `[-1, 1]`
  - Negative = switch left of ninja, positive = right

- `[31]` **Exit switch relative Y**: Normalized relative Y position
  - Value: `(switch_y - ninja_y) / (LEVEL_HEIGHT / 2)`, clamped to `[-1, 1]`
  - Negative = switch below ninja, positive = above

- `[32]` **Exit switch path distance**: Graph-based shortest path distance
  - Value: `min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]` where 0 = at switch, 1 = maximum distance
  - Uses pathfinding with EXIT_SWITCH_RADIUS entity radius
  - Normalized by reachable area scale (computed from flood-fill from start position) for level-adaptive scaling

**Exit Door Features (3 features):**

- `[33]` **Exit door relative X**: Normalized relative X position
  - Value: `(door_x - ninja_x) / (LEVEL_WIDTH / 2)`, clamped to `[-1, 1]`

- `[34]` **Exit door relative Y**: Normalized relative Y position
  - Value: `(door_y - ninja_y) / (LEVEL_HEIGHT / 2)`, clamped to `[-1, 1]`

- `[35]` **Exit door path distance**: Graph-based shortest path distance
  - Value: `min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]`
  - Uses pathfinding with EXIT_DOOR_RADIUS entity radius
  - Normalized by reachable area scale for level-adaptive scaling

**Nearest Locked Door Features (8 features):**

- `[36]` **Nearest locked door present**: Binary indicator
  - Value: `1.0` if any active locked door exists, else `0.0`
  - Finds nearest door by Euclidean distance

- `[37]` **Nearest locked door switch collected**: Binary indicator
  - Value: `0.0` if door active (locked), `1.0` if door inactive (unlocked)
  - From `door.active` attribute (inverted)

- `[38]` **Nearest locked door switch relative X**: Normalized relative X position
  - Value: `(switch_x - ninja_x) / (LEVEL_WIDTH / 2)`, clamped to `[-1, 1]`
  - Uses `door.sw_xpos` or `door.xpos` as fallback

- `[39]` **Nearest locked door switch relative Y**: Normalized relative Y position
  - Value: `(switch_y - ninja_y) / (LEVEL_HEIGHT / 2)`, clamped to `[-1, 1]`

- `[40]` **Nearest locked door switch path distance**: Graph-based shortest path distance
  - Value: `min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]`
  - Uses pathfinding with LOCKED_DOOR_SWITCH_RADIUS entity radius
  - Normalized by reachable area scale for level-adaptive scaling

- `[41]` **Nearest locked door relative X**: Normalized relative X position
  - Value: `(door_x - ninja_x) / (LEVEL_WIDTH / 2)`, clamped to `[-1, 1]`
  - Uses door segment midpoint if available, else `door.xpos`

- `[42]` **Nearest locked door relative Y**: Normalized relative Y position
  - Value: `(door_y - ninja_y) / (LEVEL_HEIGHT / 2)`, clamped to `[-1, 1]`
  - Uses door segment midpoint if available, else `door.ypos`

- `[43]` **Nearest locked door path distance**: Graph-based shortest path distance
  - Value: `min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]`
  - Uses pathfinding with 0.0 entity radius (door is line segment)
  - Normalized by reachable area scale for level-adaptive scaling

#### Indices 44-51: Mine Features (8 features)

All features normalized to range `[0, 1]` or `[-1, 1]` as specified. Based on nearest mine and nearby mine statistics.

**Nearest Mine Features (4 features):**

- `[44]` **Nearest mine relative X**: Normalized relative X position
  - Value: `(mine_x - ninja_x) / area_scale`, clamped to `[-1, 1]` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Negative = mine left of ninja, positive = right
  - Normalized by reachable area scale for level-adaptive scaling

- `[45]` **Nearest mine relative Y**: Normalized relative Y position
  - Value: `(mine_y - ninja_y) / area_scale`, clamped to `[-1, 1]` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Negative = mine below ninja, positive = above
  - Normalized by reachable area scale for level-adaptive scaling

- `[46]` **Nearest mine state**: Mine activation state
  - Value: `0.0` if state == 0 (deadly), `0.5` if state == 2 (toggling), `1.0` if state == 1 (safe), `-1.0` if no mines
  - State 0 = active/deadly, State 1 = inactive/safe, State 2 = transitioning

- `[47]` **Nearest mine path distance**: Graph-based shortest path distance
  - Value: `min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]`
  - Uses pathfinding with mine radius from TOGGLE_MINE_RADII[state]
  - Falls back to Euclidean distance if pathfinding fails
  - Normalized by reachable area scale for level-adaptive scaling

**Nearby Mine Statistics (4 features):**

- `[48]` **Deadly mines nearby count**: Normalized count of deadly mines within 100px
  - Value: `min(deadly_count / 10.0, 1.0)`
  - Range: `[0, 1]` where 1.0 = 10+ deadly mines nearby
  - Counts mines with state == 0 within NEARBY_RADIUS (100px)

- `[49]` **Mine state certainty**: Confidence in mine state knowledge
  - Value: `1.0 - min(nearest_distance / 150.0, 1.0)`
  - Range: `[0, 1]` where 1.0 = mine very close (high certainty), 0.0 = mine far (low certainty)
  - Based on distance to nearest mine (closer = more recently observed)

- `[50]` **Safe mines nearby count**: Normalized count of safe mines within 100px
  - Value: `min(safe_count / 10.0, 1.0)`
  - Range: `[0, 1]` where 1.0 = 10+ safe mines nearby
  - Counts mines with state == 1 within NEARBY_RADIUS (100px)

- `[51]` **Mine avoidance difficulty**: Spatial complexity metric for navigating mines
  - Value: `min(0.7 * danger_ratio + 0.3 * density, 1.0)`
  - Range: `[0, 1]` where 1.0 = very difficult (high danger ratio + high density)
  - Combines danger ratio (deadly/total) and density (total/MAX_NEARBY)
  - Only computed if total_nearby > 0, else 0.0

#### Indices 52-54: Progress Features (3 features)

All features normalized to range `[0, 1]`. Tracks overall level completion progress.

- `[52]` **Current objective type**: Current primary objective
  - Value: `0.0` if switch not collected, `0.33` if switch collected but locked doors remain, `0.67` if all doors unlocked
  - Encodes objective hierarchy: switch → locked doors → exit door

- `[53]` **Objectives completed ratio**: Fraction of objectives completed
  - Value: `completed / max(total, 1)`
  - Range: `[0, 1]` where 1.0 = all objectives complete
  - Counts: exit switch (1) + locked doors (N)
  - Completed = switch collected + unlocked doors

- `[54]` **Total path distance remaining**: Normalized sum of remaining objective path distances
  - Value: `min(total_path_distance / (area_scale * 3), 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]` where 0 = all objectives reached, 1 = maximum distance
  - Sums path distances to: uncollected switch + active locked door switches + exit door
  - Uses graph-based pathfinding for each objective
  - Normalized by reachable area scale for level-adaptive scaling

#### Indices 55-57: Sequential Goal Features (3 features)

All features normalized to range `[-1, 1]` or `[0, 1]` as specified. Encodes hierarchical task structure (switch → door sequence).

- `[55]` **Goal phase**: Current phase in sequential goal progression
  - Value: `-1.0` if switch not collected (pre-switch), `0.0` if switch collected but not at door (post-switch, pre-door), `1.0` if near door (at door)
  - "Near door" threshold: distance < 50.0 pixels

- `[56]` **Switch priority**: Priority weight for switch objective
  - Value: `1.0` if switch not collected, else `0.0`
  - Indicates switch should be prioritized when active

- `[57]` **Door priority**: Priority weight for door objective
  - Value: `1.0` if switch collected, else `0.0`
  - Indicates door should be prioritized after switch collection
  - Note: `switch_priority + door_priority = 1.0` (mutually exclusive priorities)

### `reachability_features` - Complete Feature Breakdown

The `reachability_features` vector contains **8 features**, all normalized to range `[0, 1]`:

- `[0]` **Area ratio**: Fraction of level that is reachable
  - Value: `reachable_area / total_area`
  - Range: `[0, 1]` where 1.0 = entire level reachable
  - Computed via flood-fill from current ninja position

- `[1]` **Distance to next objective**: Normalized, inverted distance to current objective
  - Value: `1.0 - min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]` where 1.0 = at objective, 0 = far from objective
  - Uses objective hierarchy (switch → door → exit)
  - Normalized by reachable area scale for level-adaptive scaling

- `[2]` **Exit distance**: Normalized, inverted distance to exit
  - Value: `1.0 - min(path_distance / area_scale, 1.0)` where `area_scale = sqrt(reachable_surface_area) * SUB_NODE_SIZE`
  - Range: `[0, 1]` where 1.0 = at exit, 0 = far from exit
  - Normalized by reachable area scale for level-adaptive scaling

- `[3]` **Objective path quality**: Ratio of path distance to Euclidean distance
  - Value: `min(euclidean_distance / path_distance, 1.0)` if path_distance > 0, else `1.0`
  - Range: `[0, 1]` where 1.0 = direct path (no obstacles), lower = more obstacles
  - Indicates how direct the path to objective is

- `[4]` **Deadly mines on optimal path**: Normalized count of deadly mines blocking path
  - Value: `min(deadly_mine_count / max_expected, 1.0)`
  - Range: `[0, 1]` where 1.0 = many deadly mines on path
  - Counts mines with state == 0 along shortest path

- `[5]` **Connectivity score**: Edge density of reachable graph
  - Value: `min(edge_count / (node_count * max_edges_per_node), 1.0)`
  - Range: `[0, 1]` where 1.0 = highly connected (many paths), 0 = poorly connected
  - Measures graph connectivity in reachable area

- `[6]` **Next objective reachable**: Binary indicator
  - Value: `1.0` if path exists to next objective, else `0.0`
  - Uses graph pathfinding to verify reachability

- `[7]` **Full completion path exists**: Binary indicator for complete level path
  - Value: `1.0` if path exists: switch → door → exit, else `0.0`
  - Verifies that complete level solution path exists from current position

### `entity_positions` - Complete Feature Breakdown

The `entity_positions` vector contains **6 features**, all normalized to range `[0, 1]`:

- `[0]` **Ninja X position**: Normalized X coordinate
  - Value: `ninja_x / LEVEL_WIDTH`
  - Range: `[0, 1]` where 0 = left edge, 1 = right edge

- `[1]` **Ninja Y position**: Normalized Y coordinate
  - Value: `ninja_y / LEVEL_HEIGHT`
  - Range: `[0, 1]` where 0 = top edge, 1 = bottom edge

- `[2]` **Exit switch X position**: Normalized X coordinate
  - Value: `switch_x / LEVEL_WIDTH`
  - Range: `[0, 1]`
  - Defaults to `0.0` if no switch exists

- `[3]` **Exit switch Y position**: Normalized Y coordinate
  - Value: `switch_y / LEVEL_HEIGHT`
  - Range: `[0, 1]`
  - Defaults to `0.0` if no switch exists

- `[4]` **Exit door X position**: Normalized X coordinate
  - Value: `exit_door_x / LEVEL_WIDTH`
  - Range: `[0, 1]`
  - Defaults to `0.0` if no door exists

- `[5]` **Exit door Y position**: Normalized Y coordinate
  - Value: `exit_door_y / LEVEL_HEIGHT`
  - Range: `[0, 1]`
  - Defaults to `0.0` if no door exists

## Architecture Support

### Current Architectures

NPP-RL supports the following architectures in `npp_rl/training/architecture_configs.py`:

| Architecture | Visual | State | Reachability | Graph | Description |
|---|---|---|---|---|---|
| `full_hgt` | ✓ | ✓ | ✓ | ✓ HGT | All modalities with HGT |
| `simplified_hgt` | ✓ | ✓ | ✓ | ✓ Simple | Lightweight HGT variant |
| `gat` | ✓ | ✓ | ✓ | ✓ GAT | Graph Attention Network |
| `gcn` | ✓ | ✓ | ✓ | ✓ GCN | Graph Convolutional Network |
| `mlp_cnn` | - | ✓ | ✓ | - | MLP only (no vision/graph) |
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
    - 2D CNN for single grayscale frame
    - 2D CNN for global view
    - MLP for game state + reachability
    - HGT for graph representation
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Visual encoders
        if 'player_frame' in observation_space.spaces:
            self.cnn_player_frame = self._build_2d_cnn()
        
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
```

### Vision-Free (State + Graph)

```bash
python -m npp_rl.agents.training \
    --architecture vision_free \
    --num_envs 64 \
    --total_timesteps 10000000 \
```

### MLP Baseline (State Only)

```bash
python -m npp_rl.agents.training \
    --architecture mlp_cnn \
    --num_envs 64 \
    --total_timesteps 10000000 \
    --disable_graph
```

## Architecture Comparison

Use the comparison script to benchmark multiple architectures:

```bash
python scripts/train_and_compare.py \
    --experiment-name "modality_comparison" \
    --architectures full_hgt vision_free mlp_cnn \
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
mlp_cnn
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
    config=EnvironmentConfig(
        graph=GraphConfig(),  # Required for graph obs
        reachability=ReachabilityConfig(),   # Required for reachability obs
    )
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
- Architecture configs: `npp_rl/training/architecture_configs.py`
- Feature extractors: `npp_rl/feature_extractors/`
- Training guide: `docs/TRAINING_SYSTEM.md`
- Architecture comparison: `docs/ARCHITECTURE_COMPARISON_GUIDE.md`
