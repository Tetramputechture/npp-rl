# Observation Space Utilization Analysis & Enhanced Recommendations

**Date**: November 8, 2025  
**Purpose**: Deep dive into nclone's observation space to understand which features are exposed, how they're being used, and how to optimize learning

---

## Executive Summary

After reviewing the comprehensive observation space provided by nclone, we've identified **significant underutilization** of rich features that could dramatically improve agent performance:

### Critical Findings:

1. **Rich Physics State Available** - 29 game_state features capturing velocity, buffers, surface contact, momentum
2. **Fast Reachability Features** - 8 reachability features using OpenCV flood fill (<1ms)
3. **Constant-Time Adjacency Graph** - Available but only used internally for PBRS, not exposed to policy
4. **Visual Observations Ignored** - MLP baseline doesn't use player_frame or global_view
5. **Graph Observations Disabled** - Full GNN-ready graph with 55-dim node features not enabled

### Key Opportunities:

The agent has access to incredibly rich spatial and temporal information, but the current configuration doesn't leverage it effectively. By better utilizing existing observations and enabling currently-disabled features, we can provide much stronger learning signals.

---

## 1. Current Observation Space Breakdown

### 1.1 What's Currently Used (MLP Baseline)

```python
Total Input Dimension: 68 features

game_state:            29 features  ✓ USED
reachability_features:  8 features  ✓ USED  
entity_positions:       6 features  ✓ USED
switch_states:         25 features  ✓ USED
-------------------------------------------------
player_frame:          84×84×1      ✗ NOT USED (visual)
global_view:          176×100×1     ✗ NOT USED (visual)
graph features:        ~18k nodes   ✗ NOT USED (GNN)
```

### 1.2 Detailed Feature Analysis

#### A. `game_state` (29 features) - EXCELLENT COVERAGE

**Core Movement (8 features)**:
```
[0]     Velocity magnitude        → Speed awareness
[1:3]   Velocity direction (x,y)  → Movement direction
[3:7]   Movement state categories → Ground/air/wall/special
[7]     Airborne status           → Jump state
```
✓ **Assessment**: Complete physics state for momentum-based gameplay

**Input & Buffers (5 features)**:
```
[8]     Horizontal input          → Current action
[9]     Jump input                → Jump action
[10:13] Buffer states            → Frame-perfect timing windows
```
✓ **Assessment**: Critical for N++ frame-perfect execution, well-captured

**Surface Contact (6 features)**:
```
[13:16] Contact strength (floor/wall/ceiling)  → Collision detection
[16]    Floor normal strength                  → Surface interaction
[17]    Wall direction                         → Wall-jump orientation
[18]    Surface slope                          → Slope mechanics
```
✓ **Assessment**: Complete surface interaction state

**Momentum & Physics (9 features)**:
```
[19:21] Recent acceleration       → Momentum changes
[21]    Applied gravity           → Physics constants
[22]    Jump duration             → Jump state tracking
[23]    Walled status             → Wall interaction
[24:26] Floor/ceiling normals     → Surface geometry
[27:28] Drag/friction applied     → Physics simulation
```
✓ **Assessment**: Comprehensive physics state

**Overall**: The game_state vector is **EXCELLENT** - captures all relevant physics for platforming.

#### B. `reachability_features` (8 features) - FAST & INFORMATIVE

```
[0] Area ratio (reachable / total)     → Connectivity measure
[1] Switch distance (normalized)       → Primary objective distance
[2] Exit distance (normalized)         → Secondary objective distance
[3] Reachable switches (count)         → Multiple objectives
[4] Reachable hazards (count)          → Danger awareness
[5] Connectivity score                 → Graph connectivity
[6] Exit reachable {0, 1}             → Binary goal reachability
[7] Path to exit exists {0, 1}        → Post-switch path exists
```

**Performance**: <1ms using OpenCV flood fill (constant-time)

✓ **Assessment**: Extremely efficient high-level spatial reasoning features

#### C. `entity_positions` (6 features) - SIMPLE BUT LIMITED

```
[0:2] Ninja position (x, y)    → Self-localization
[2:4] Switch position (x, y)   → Primary objective
[4:6] Exit position (x, y)     → Secondary objective
```

✗ **Issue**: Only 3 entities, no hazard positions
✓ **But**: Combined with reachability_features[4], hazard count is available

#### D. `switch_states` (25 features) - DOOR SYSTEM STATE

```
5 doors × 5 features each:
  [0:2] Switch position (x, y)
  [2:4] Door position (x, y)
  [4]   Collected/open state {0, 1}
```

✓ **Assessment**: Complete for door-based puzzles

### 1.3 What's NOT Being Used (But Available)

#### A. Visual Observations (UNUSED)

**`player_frame`: 84×84×1 grayscale**
- Centered on ninja, covers ~1/6 of level
- Contains local spatial context
- Good for obstacle detection, hazard awareness

**`global_view`: 176×100×1 grayscale**
- Full level at 1/6 resolution
- Global spatial layout
- Navigation and planning

**Why not using**: MLP baseline can't process images

**Opportunity**: CNN could extract powerful spatial features

#### B. Graph Observations (DISABLED)

**Full GNN-ready graph structure:**
```
graph_node_feats:  [~18k nodes, 55 features]
graph_edge_index:  [2, ~144k edges] COO format
graph_edge_feats:  [~144k edges, 6 features]
graph_node_mask:   [~18k] valid node mask
graph_edge_mask:   [~144k] valid edge mask
```

**Node Features (55-dim)**:
- Spatial (2): position
- Type (10): one-hot node type
- Entity (15): entity-specific attributes  
- Tile (20): surrounding context
- Reachability (5): connectivity metrics
- Proximity (3): distances to key entities

**Edge Features (6-dim)**:
- Edge type (ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED)
- Distance/weight
- Connectivity information

**Why not using**: Graph observations disabled in config

**Opportunity**: GNN could learn optimal pathfinding

#### C. Internal Graph for PBRS (NOT EXPOSED TO POLICY)

**Adjacency Graph (_adjacency_graph)**:
- Built using sub-grid resolution (4×4 per tile = 168×92 = 15,456 nodes)
- Constant-time lookups via spatial hashing
- Used internally for PBRS path distance calculations
- **NOT** exposed to the policy network

**Current usage**:
```python
# In main_reward_calculator.py
adjacency = obs.get("_adjacency_graph")  # Internal only
pbrs_reward = calculate_path_based_shaping(adjacency, ...)
```

**Key insight**: The adjacency graph enables PBRS to compute path distances in constant time, but the policy never sees this graph structure directly!

---

## 2. Critical Gaps & Opportunities

### 2.1 Temporal Information Gap ❌ CRITICAL

**Problem**: Single-frame observations in physics-based game

**Evidence**:
- Config: `enable_state_stacking: false`
- Agent sees instantaneous state only
- Cannot infer velocity changes → acceleration
- Cannot predict momentum → trajectory planning

**Impact**: Agent cannot plan multi-step physics-based movements

**Solution**: ✅ **ENABLE STATE STACKING (4 frames)**

```python
"enable_state_stacking": true,
"state_stack_size": 4,
```

**Benefits**:
- Velocity = position[t] - position[t-1]
- Acceleration = velocity[t] - velocity[t-1]  
- Momentum patterns visible
- Jump trajectory prediction
- 68 features × 4 frames = 272 total features

### 2.2 Spatial Reasoning Gap (MLP Limitation)

**Problem**: MLP processes 68 scalar features independently

**What's missing**:
- Spatial relationships (hazard proximity, wall layouts)
- Local obstacle patterns
- Safe/unsafe region identification
- Multi-step path visualization

**Current workaround**: reachability_features provide some spatial info

**Better solution**: Enable visual observations with CNN

### 2.3 Graph Structure Gap (GNN Not Used)

**Problem**: Adjacency graph exists but not exposed to policy

**What adjacency graph provides**:
- Exact walkable paths (sub-cell resolution)
- Dynamic connectivity (updates with doors/switches)
- Spatial hashing for O(1) neighbor lookup
- Multiple path options for exploration

**Current usage**: Only for PBRS (reward shaping)

**Opportunity**: Expose graph to GNN policy

**Trade-off**: GNN adds complexity but could enable learned pathfinding

### 2.4 Reward Signal Gap (Covered in Main Analysis)

**Problem**: Negative reward regime despite rich observations

**Root cause**: Time penalty dominates, PBRS too weak

**Impact**: Agent doesn't learn to use rich observations effectively

---

## 3. Enhanced Recommendations Incorporating Observation Space

### 3.1 IMMEDIATE (Week 1) - Maximum Impact, Minimal Changes

#### ✅ 1. Enable State Stacking (4 frames)

**Change**: `"enable_state_stacking": true, "state_stack_size": 4`

**Impact**:
- Temporal context for physics understanding
- Can infer velocity, acceleration from position changes
- Buffer state timing becomes observable
- **Estimated improvement**: +15-20% success rate

**Rationale**: The game_state vector is already excellent (29 features). Adding temporal context makes it even more powerful.

#### ✅ 2. Rebalance Rewards to Make Observations Useful

**Current problem**: Agent receives negative feedback regardless of how well it uses observations

**Changes**:
```python
# Reduce time penalty 10x
TIME_PENALTY_PER_STEP = -0.00001  # was -0.0001

# Increase PBRS weights 3x
PBRS_OBJECTIVE_WEIGHT = 4.5  # was 1.5 (uses adjacency graph)
PBRS_HAZARD_WEIGHT = 0.15     # was 0.04 (uses reachability_features[4])
```

**Impact**:
- PBRS uses adjacency graph more effectively
- Hazard avoidance rewarded (uses reachability hazard counts)
- **Estimated improvement**: +20-25% success rate

#### ✅ 3. Add Reward for Using Reachability Features

**New reward component**:
```python
# Reward for moving toward high-connectivity regions
CONNECTIVITY_BONUS = 0.01  # per step
# Uses reachability_features[5]: connectivity score

# Reward for reducing distance to objective
DISTANCE_IMPROVEMENT_BONUS = 0.05  # per distance unit
# Uses reachability_features[1] or [2]: switch/exit distance
```

**Impact**: Directly incentivizes using spatial reasoning features

#### ✅ 4. Lower Curriculum Thresholds

**Change**: Stage 1 threshold: 80% → 65%

**Impact**: Agent can progress while still learning spatial reasoning

### 3.2 HIGH PRIORITY (Week 2) - Leverage Visual Information

#### 🎯 5. Upgrade to CNN+MLP Architecture

**Configuration**:
```python
"architectures": ["cnn_mlp_fusion"],
"modality_config": {
    "use_player_frame": true,    # Enable visual (84×84×1)
    "use_global_view": false,    # Keep computational cost down
    "use_graph": false,          # Not yet
    "use_game_state": true,      # Keep physics state
    "use_reachability": true     # Keep fast spatial features
}
```

**Architecture**:
```
player_frame (84×84×1) → CNN → 256-dim
game_state (29×4)      → MLP → 128-dim  (with stacking)
reachability (8×4)     → MLP → 64-dim   (with stacking)
entity_positions (6×4) → MLP → 32-dim   (with stacking)
switch_states (25×4)   → MLP → 64-dim   (with stacking)
                          ↓
                     Concat/Attention Fusion → 544-dim
                          ↓
                     Policy Head (6 actions)
                     Value Head (1 value)
```

**Benefits**:
- CNN extracts local spatial patterns (obstacles, hazards)
- MLP processes temporal physics (velocity, buffers)
- Fusion combines spatial + temporal
- **Estimated improvement**: +10-15% success rate over MLP

#### 🎯 6. Enhanced Reachability-Based Shaping

**Leverage all 8 reachability features**:
```python
# Connectivity-based bonus
connectivity_improvement = obs['reachability_features'][5] - prev_obs['reachability_features'][5]
reward += CONNECTIVITY_WEIGHT * connectivity_improvement

# Area coverage bonus
area_improvement = obs['reachability_features'][0] - prev_obs['reachability_features'][0]
reward += AREA_COVERAGE_WEIGHT * area_improvement

# Hazard awareness (use feature [4])
hazards_visible = obs['reachability_features'][4]
if hazards_visible > 0:
    # Reward cautious movement near hazards
    if speed < CAUTIOUS_SPEED_THRESHOLD:
        reward += CAUTIOUS_NEAR_HAZARD_BONUS
```

**Impact**: Makes full use of fast reachability features

### 3.3 MEDIUM PRIORITY (Week 3) - Graph Neural Networks

#### 🚀 7. Enable Graph Observations for GNN

**Configuration**:
```python
"architectures": ["cnn_gnn_mlp_fusion"],
"graph_config": {
    "enable_graph_for_observations": true,  # Enable graph in obs
    "architecture": "simplified_hgt",        # Start simple
    "hidden_dim": 128,                       # Reduced from 256
    "num_layers": 2,                         # Reduced from 3
    "output_dim": 128
}
```

**Why this works**:
- Adjacency graph already built (for PBRS)
- Sub-cell resolution (168×92 = 15,456 nodes)
- Constant-time updates via spatial hashing
- GNN can learn optimal pathfinding
- **Estimated improvement**: +10-20% on complex stages

**Trade-off**: More computation, but graph is already available

#### 🚀 8. Expose Adjacency Structure to Policy

**Current**: Adjacency graph only used for PBRS internally

**Proposed**: Add adjacency features to observation

**New observation component**:
```python
adjacency_features: (32,) float32
  [0:8]   Connectivity in 8 directions (N/S/E/W/NE/NW/SE/SW)
  [8:16]  Nearest obstacle distance in 8 directions
  [16:24] Path costs to switch in 8 directions
  [24:32] Path costs to exit in 8 directions
```

**Implementation**:
```python
# In npp_environment.py
adjacency = self._get_adjacency_for_rewards()
adjacency_features = self._extract_adjacency_features(
    adjacency, 
    ninja_pos, 
    switch_pos, 
    exit_pos
)
obs["adjacency_features"] = adjacency_features
```

**Benefits**:
- Exposes graph structure without full GNN overhead
- Policy can learn directional preferences
- Works with MLP or CNN+MLP
- Minimal computational cost (already built graph)

### 3.4 ADVANCED (Week 4+) - Full Multi-Modal Learning

#### 🔬 9. Full Multi-Modal Fusion

**Architecture**:
```
Visual:       player_frame (84×84×1) → CNN → 256-dim
Graph:        graph_node/edge_feats   → GNN → 128-dim
Physics:      game_state (29×4)       → MLP → 128-dim
Spatial:      reachability (8×4)      → MLP → 64-dim
Adjacency:    adjacency_features      → MLP → 32-dim
                          ↓
              Multi-Head Cross-Attention Fusion
                          ↓
              Policy/Value Heads
```

**Benefits**:
- Combines all modalities effectively
- Attention learns which features matter when
- Maximum representational power

**Trade-off**: Complex, train simpler versions first

---

## 4. Observation-Aware Training Strategy

### Phase 1: Maximize Current Observations (Week 1)

**Focus**: Make MLP baseline work with existing observations

**Changes**:
1. ✅ Enable state stacking → temporal context
2. ✅ Fix reward structure → positive reinforcement
3. ✅ Add reachability-based shaping → spatial reasoning
4. ✅ Lower curriculum thresholds → progression

**Expected**: Agent learns to use physics state + reachability effectively

### Phase 2: Add Visual Information (Week 2)

**Focus**: Incorporate spatial features from vision

**Changes**:
1. 🎯 CNN+MLP fusion architecture
2. 🎯 Enhanced reachability rewards
3. 🎯 Attention-based fusion

**Expected**: Better spatial understanding, hazard avoidance

### Phase 3: Add Graph Structure (Week 3)

**Focus**: Expose adjacency graph for pathfinding

**Changes**:
1. 🚀 Add adjacency_features (32-dim) to observations
2. 🚀 Or enable full GNN with graph observations
3. 🚀 Graph-aware reward shaping

**Expected**: Learned pathfinding, optimal routing

### Phase 4: Full Multi-Modal (Week 4+)

**Focus**: Combine all modalities with attention

**Changes**:
1. 🔬 Multi-modal attention fusion
2. 🔬 Cross-modal learning objectives
3. 🔬 Hierarchical representations

**Expected**: Human-level performance

---

## 5. Specific Feature Utilization Recommendations

### 5.1 Physics State (game_state) Utilization

**Already well-utilized**, but can improve:

**Buffer State Features [10:13]**: 
- Currently just observed
- **Add reward**: Bonus for successful buffered jumps (already exists: BUFFER_USAGE_BONUS)
- **Increase bonus**: 0.05 → 0.1 to make it more significant

**Momentum Features [0, 19:21]**:
- Velocity magnitude [0] tracked
- Acceleration [19:21] tracked
- **Current reward**: MOMENTUM_BONUS_PER_STEP = 0.0002 (too small)
- **Increase**: 0.0002 → 0.001 (5x) to incentivize speed maintenance

**Surface Contact [13:18]**:
- Floor/wall/ceiling contact tracked
- **Add reward**: Bonus for successful wall-jumps
  ```python
  WALL_JUMP_BONUS = 0.05
  # Reward when:
  #  - Wall contact [14] > 0.5
  #  - Jump action executed
  #  - Successfully left wall
  ```

### 5.2 Reachability Features Utilization

**Currently underutilized**:

**Area Ratio [0]**:
- Tracks reachable / total area
- **Add reward**: Exploration bonus for increasing reachable area
  ```python
  AREA_EXPLORATION_BONUS = 0.02
  area_increase = obs['reachability'][0] - prev_obs['reachability'][0]
  if area_increase > 0:
      reward += AREA_EXPLORATION_BONUS * area_increase
  ```

**Switch/Exit Distance [1, 2]**:
- Currently used in PBRS
- **Also use in policy**: Add to observation explicitly
- **Normalize**: Already normalized, good!

**Connectivity Score [5]**:
- Higher = better connected position
- **Add reward**: Stay in well-connected regions
  ```python
  CONNECTIVITY_PREFERENCE = 0.01
  reward += CONNECTIVITY_PREFERENCE * obs['reachability'][5]
  ```

**Hazard Count [4]**:
- Number of reachable hazards
- **Add penalty**: Negative shaping for high hazard regions
  ```python
  HAZARD_PROXIMITY_PENALTY = -0.005
  reward += HAZARD_PROXIMITY_PENALTY * obs['reachability'][4]
  ```

### 5.3 Entity Positions Utilization

**Current**: Just (x, y) for ninja/switch/exit

**Enhancement**: Add relative positions and distances

```python
# In observation processing
ninja_pos = entity_positions[0:2]
switch_pos = entity_positions[2:4]
exit_pos = entity_positions[4:6]

# Add computed features
relative_switch = switch_pos - ninja_pos
relative_exit = exit_pos - ninja_pos
distance_switch = np.linalg.norm(relative_switch)
distance_exit = np.linalg.norm(relative_exit)

# Augment observation
obs['relative_positions'] = np.concatenate([
    relative_switch,      # [0:2]
    relative_exit,        # [2:4]
    [distance_switch],    # [4]
    [distance_exit]       # [5]
])  # 6 additional features
```

**Benefit**: Agent doesn't need to learn subtraction/distance computation

### 5.4 Visual Observations (When Enabled)

**player_frame (84×84×1)**:
- Local spatial context
- **Use for**: Obstacle detection, immediate hazards
- **CNN architecture**:
  ```python
  Conv2d(1, 32, kernel=8, stride=4)  # → 20×20×32
  Conv2d(32, 64, kernel=4, stride=2)  # → 9×9×64
  Conv2d(64, 64, kernel=3, stride=1)  # → 7×7×64
  Flatten → 3136 → FC(256)
  ```

**global_view (176×100×1)** (if needed):
- Global spatial layout
- **Use for**: Long-term planning, route selection
- **CNN architecture**: Similar but adapted for aspect ratio

### 5.5 Graph Observations (When Enabled)

**Adjacency Graph Structure**:
- 168×92 sub-cells = 15,456 nodes
- ~8 edges/node (directions) = ~123k edges
- Spatial hash for O(1) lookup

**GNN Processing**:
```python
# Simplified HGT (Heterogeneous Graph Transformer)
Node features (55-dim) → Linear(55, 128)
For each layer (2 layers):
    Message passing across edge types
    Attention over neighbors
    Update node embeddings
Global pooling → 128-dim level representation
Concat with ninja position embedding → 256-dim
```

**Benefit**: Learned pathfinding, multi-step planning

---

## 6. Computational Cost Analysis

### 6.1 Current Cost (MLP Baseline)

```
Observation processing:
  - game_state: free (already computed)
  - reachability: <1ms (OpenCV flood fill)
  - entity_positions: free
  - switch_states: free
  Total: <1ms per step

Forward pass:
  - Input: 68 features
  - MLP: [256, 256, 128]
  - Params: ~150K
  - FLOPs: ~150K
  - Time: <1ms on GPU
```

**Throughput**: ~23.5 steps/env/second (observed in training)

### 6.2 With State Stacking (4 frames)

```
Observation processing:
  - 68 features × 4 frames = 272 features
  - Same observation computation (just stacked)
  - Time: <1ms per step

Forward pass:
  - Input: 272 features
  - MLP: [256, 256, 128]
  - Params: ~230K (+50%)
  - FLOPs: ~230K (+50%)
  - Time: <1.5ms on GPU
```

**Estimated throughput**: ~20 steps/env/second (-15%)

**Trade-off**: Worth it for +15-20% success rate

### 6.3 With CNN+MLP Fusion

```
Observation processing:
  - Same as stacking
  - player_frame: free (already rendered)
  - Time: <1ms per step

Forward pass:
  - CNN: 3 conv layers → ~500K FLOPs
  - MLP: 272 features → ~200K FLOPs
  - Fusion: concat → ~50K FLOPs
  - Total: ~750K FLOPs
  - Params: ~500K
  - Time: <2ms on GPU
```

**Estimated throughput**: ~15 steps/env/second (-35%)

**Trade-off**: Worth it for +25-35% total success rate

### 6.4 With GNN (Full Graph)

```
Observation processing:
  - Graph building: amortized <5ms (cached, updated on switch changes)
  - Graph features: <1ms

Forward pass:
  - GNN: 2 layers on ~18k nodes
  - Message passing: ~2M FLOPs
  - CNN+MLP: ~750K FLOPs
  - Total: ~3M FLOPs
  - Params: ~2M
  - Time: <5ms on GPU
```

**Estimated throughput**: ~8 steps/env/second (-65%)

**Trade-off**: Only worth it if complex stages need it (week 3+)

### 6.5 Optimization Strategies

**To maintain throughput**:

1. **Compiled models**: TorchScript reduces overhead
2. **Mixed precision**: FP16 for forward pass (2x speedup)
3. **Batch size**: Increase n_envs to amortize overhead
4. **Async observation**: Compute observations while GPU runs forward pass
5. **Graph caching**: Only rebuild graph on switch state changes (already done)

**Expected final throughput with optimizations**:
- MLP + stacking: 20-22 steps/env/s
- CNN+MLP: 15-18 steps/env/s  
- CNN+GNN+MLP: 10-12 steps/env/s

All acceptable for training at scale with 85 parallel environments.

---

## 7. Recommended Observation Space Configuration

### Week 1 Configuration (Immediate Fixes)

```json
{
  "_observations_comment": "Use existing observations effectively",
  "enable_state_stacking": true,
  "state_stack_size": 4,
  "frame_stack_padding": "repeat",
  
  "modality_config": {
    "use_player_frame": false,
    "use_global_view": false,
    "use_graph": false,
    "use_game_state": true,
    "use_reachability": true
  },
  
  "_total_features": "68 × 4 = 272 (with stacking)",
  "_architecture": "mlp_baseline"
}
```

### Week 2 Configuration (Add Vision)

```json
{
  "_observations_comment": "Add visual spatial features",
  "enable_state_stacking": true,
  "state_stack_size": 4,
  "enable_visual_frame_stacking": false,
  
  "modality_config": {
    "use_player_frame": true,
    "use_global_view": false,
    "use_graph": false,
    "use_game_state": true,
    "use_reachability": true
  },
  
  "_total_features": "272 (stacked) + 256 (CNN) = 528",
  "_architecture": "cnn_mlp_fusion"
}
```

### Week 3 Configuration (Add Graph)

```json
{
  "_observations_comment": "Add graph structure for pathfinding",
  "enable_state_stacking": true,
  "state_stack_size": 4,
  
  "modality_config": {
    "use_player_frame": true,
    "use_global_view": false,
    "use_graph": true,
    "use_game_state": true,
    "use_reachability": true
  },
  
  "graph_config": {
    "enable_graph_for_observations": true,
    "architecture": "simplified_hgt",
    "hidden_dim": 128,
    "num_layers": 2,
    "output_dim": 128
  },
  
  "_total_features": "272 (stacked) + 256 (CNN) + 128 (GNN) = 656",
  "_architecture": "cnn_gnn_mlp_fusion"
}
```

---

## 8. Expected Performance Progression

### Baseline (Current)
- **Observations**: 68 features, no stacking
- **Architecture**: MLP baseline
- **Performance**: Stage 0: 77%, Stage 1: 44% (stuck)

### Week 1: Optimized MLP
- **Observations**: 272 features (68×4 stacking)
- **Architecture**: MLP baseline  
- **Changes**: State stacking + reward fixes
- **Expected**: Stage 0: 85%, Stage 1: 70%, Stage 2: 50%

### Week 2: CNN+MLP
- **Observations**: 272 + CNN (player_frame)
- **Architecture**: CNN+MLP fusion
- **Expected**: Stage 0-2: 85%+, Stage 3: 60%, Stage 4: 45%

### Week 3: CNN+GNN+MLP
- **Observations**: 272 + CNN + GNN
- **Architecture**: Multi-modal fusion
- **Expected**: All stages progressed, final stages 40-50%

### Week 4+: Full Multi-Modal
- **Observations**: All modalities + attention
- **Architecture**: Advanced fusion
- **Expected**: Final stages 60-70%, approaching human-level

---

## 9. Implementation Checklist

### ✅ Week 1 (Immediate)

- [ ] Enable state stacking in config
- [ ] Update reward constants (reduce time penalty, increase PBRS)
- [ ] Add reachability-based reward components
- [ ] Add buffer usage bonus increase
- [ ] Add momentum bonus increase  
- [ ] Lower curriculum thresholds
- [ ] Run 500k step validation experiment

### 🎯 Week 2 (High Priority)

- [ ] Implement CNN feature extractor
- [ ] Create CNN+MLP fusion module
- [ ] Add player_frame to observations
- [ ] Test CNN vs MLP on stage 1
- [ ] Implement enhanced reachability shaping
- [ ] Add relative position features
- [ ] Run architecture comparison experiment

### 🚀 Week 3 (Medium Priority)

- [ ] Create adjacency_features extraction
- [ ] Add adjacency_features to observation space
- [ ] OR: Enable full graph observations
- [ ] Implement simplified HGT if using GNN
- [ ] Create GNN fusion module
- [ ] Test on complex stages
- [ ] Run GNN ablation study

### 🔬 Week 4+ (Advanced)

- [ ] Implement multi-head cross-attention fusion
- [ ] Add global_view if needed
- [ ] Create hierarchical representation learning
- [ ] Implement curriculum metalearning
- [ ] Test on full evaluation suite
- [ ] Compare to human expert performance

---

## 10. Key Takeaways

### What We Have (Excellent)

1. ✅ **Rich physics state** (29 features) - Complete platformer physics
2. ✅ **Fast reachability** (8 features) - <1ms spatial reasoning
3. ✅ **Adjacency graph** - Constant-time pathfinding (for PBRS)
4. ✅ **Visual observations** - Spatial context available
5. ✅ **Graph structure** - GNN-ready format

### What We're Missing (Critical)

1. ❌ **Temporal context** - No frame stacking (EASY FIX)
2. ❌ **Positive rewards** - Negative reward regime (EASY FIX)
3. ❌ **Spatial features for policy** - MLP can't use visual/graph (MEDIUM FIX)
4. ❌ **Adjacency in policy** - Graph only used for PBRS (MEDIUM FIX)

### What Will Work (High Confidence)

1. **State stacking** → +15-20% (physics understanding)
2. **Reward rebalancing** → +20-25% (positive reinforcement)
3. **CNN features** → +10-15% (spatial awareness)
4. **Adjacency features** → +10-15% (pathfinding)

**Combined**: +55-75% improvement possible, progressing through curriculum

### What to Prioritize

**Week 1**: State stacking + reward fixes (easy, high impact)
**Week 2**: CNN+MLP (medium difficulty, high impact)
**Week 3**: Graph features (medium difficulty, medium impact on later stages)
**Week 4+**: Advanced techniques (high difficulty, polish)

---

**Conclusion**: The observation space is **excellent** and comprehensive. The issue is not lack of information but rather: (1) no temporal context, (2) negative reward structure prevents learning, and (3) MLP baseline can't leverage spatial information. All three are fixable with the recommendations above.

The path to success is clear: enable what already exists, fix the reward structure, and progressively add richer architectures as the agent masters simpler stages.

---

**Document Version**: 1.0  
**Last Updated**: November 8, 2025  
**Related**: COMPREHENSIVE_RL_ANALYSIS.md
