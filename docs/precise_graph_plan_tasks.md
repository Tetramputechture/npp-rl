# Precise Graph Plan Implementation Tasks

## Executive Summary

This document breaks down the high-level graph neural network concepts from `graph_plan.md` into concrete, actionable implementation tasks for the N++ reinforcement learning project. Each task includes specific file locations, code modifications, and implementation details based on our current codebase architecture.

**Current State Analysis**:
- **Graph Builder**: `nclone/graph/graph_builder.py` - Sub-grid resolution (4x4 per tile), ~15,856 nodes, physics-aware traversability
- **GNN Implementation**: `npp_rl/models/gnn.py` - Basic GraphSAGE with 3 layers, mean aggregation, mean_max pooling
- **Feature Integration**: `npp_rl/models/feature_extractors.py` - Multimodal CNN+MLP+GNN fusion architecture
- **Physics Constants**: `nclone/constants.py` - All N++ physics parameters (gravity, friction, jump velocities)
- **Simulation**: `nclone/docs/sim_mechanics_doc.md` - Complete physics documentation with 9 movement states

**Goal**: Enhance graph representations with physics-informed features, hierarchical processing, and temporal reasoning while maintaining compatibility with existing PPO training pipeline.

## Architecture Overview

Our system builds on the existing multimodal architecture:

```
Current: CNN (visual) + MLP (symbolic) + GNN (structural) → Fusion → Policy/Value
Planned: CNN + MLP + Physics-GNN + Hierarchical-GNN + Temporal-GNN → Advanced Fusion
```

## Task Breakdown

### Phase 1: Physics-Informed Graph Representations

#### Task 1.1: Trajectory-Based Edge Features
**Priority**: High | **Complexity**: Medium | **Dependencies**: None | **Files**: 4 new, 2 modified

**Objective**: Enhance edge representations with physics-validated trajectory information using N++ simulation constants.

**Current Gap**: `GraphBuilder._determine_sub_cell_traversability()` only checks basic collision. Need physics-based trajectory validation using actual N++ constants from `nclone/constants.py`.

**Detailed Implementation**:

1. **Create Trajectory Calculator** (`npp_rl/models/trajectory_calculator.py`):
```python
class TrajectoryCalculator:
    def __init__(self):
        # Import N++ physics constants
        self.gravity_fall = GRAVITY_FALL  # 0.0667 pixels/frame²
        self.gravity_jump = GRAVITY_JUMP  # 0.0111 pixels/frame²
        self.max_hor_speed = MAX_HOR_SPEED  # 3.333 pixels/frame
        self.ninja_radius = NINJA_RADIUS  # 10 pixels
        
    def calculate_jump_trajectory(self, start_pos, end_pos, ninja_state):
        """Calculate quadratic trajectory: y = ax² + bx + c"""
        # Implementation details:
        # - Use ninja_state.movement_state to determine gravity (jump vs fall)
        # - Calculate initial velocity needed: v0 = (end - start - 0.5*g*t²) / t
        # - Validate trajectory doesn't intersect level geometry
        # - Return TrajectoryResult with feasibility, time, energy_cost
        
    def validate_trajectory_clearance(self, trajectory_points, level_data):
        """Check if trajectory clears all obstacles using ninja radius"""
        # Use existing sweep_circle_vs_tiles from nclone/physics.py
        # Check each trajectory point with NINJA_RADIUS clearance
```

2. **Modify GraphBuilder Edge Features** (`nclone/graph/graph_builder.py`):
   - **Line 91-96**: Expand `edge_feature_dim` from 9 to 16:
     ```python
     self.edge_feature_dim = (
         len(EdgeType) +  # One-hot edge type (6)
         2 +  # Direction (dx, dy normalized) 
         1 +  # Traversability cost
         3 +  # NEW: Trajectory parameters (time_of_flight, energy_cost, success_probability)
         2 +  # NEW: Physics constraints (min_velocity, max_velocity)
         2    # NEW: Movement requirements (requires_jump, requires_wall_contact)
     )
     ```
   
   - **Line 187-192**: Enhance `_determine_sub_cell_traversability()`:
     ```python
     def _determine_sub_cell_traversability(self, level_data, src_row, src_col, tgt_row, tgt_col, 
                                          one_way_index, door_blockers_index, dr, dc):
         # Existing collision checks...
         
         # NEW: Physics-based trajectory validation
         src_pos = (src_col * SUB_CELL_SIZE, src_row * SUB_CELL_SIZE)
         tgt_pos = (tgt_col * SUB_CELL_SIZE, tgt_row * SUB_CELL_SIZE)
         
         # Determine movement type based on height difference and distance
         height_diff = tgt_pos[1] - src_pos[1]
         distance = math.sqrt((tgt_pos[0] - src_pos[0])**2 + height_diff**2)
         
         if abs(dr) <= 1 and abs(dc) <= 1 and distance <= SUB_CELL_SIZE * 1.5:
             edge_type = EdgeType.WALK
             cost = 1.0
         elif height_diff < -SUB_CELL_SIZE:  # Upward movement
             edge_type = EdgeType.JUMP
             # Calculate if jump is physically possible
             trajectory_result = self.trajectory_calc.calculate_jump_trajectory(
                 src_pos, tgt_pos, current_ninja_state
             )
             if not trajectory_result.feasible:
                 return None, None  # Invalid edge
             cost = trajectory_result.energy_cost
         # ... handle other movement types
     ```

3. **Create Movement Classification** (`npp_rl/models/movement_classifier.py`):
```python
class MovementType(IntEnum):
    WALK = 0      # Horizontal ground movement
    JUMP = 1      # Upward trajectory movement  
    FALL = 2      # Downward gravity movement
    WALL_SLIDE = 3  # Wall contact movement
    WALL_JUMP = 4   # Wall-assisted jump
    LAUNCH_PAD = 5  # Launch pad boost

class MovementClassifier:
    def classify_movement(self, src_pos, tgt_pos, ninja_state, level_data):
        """Classify movement type and calculate physics parameters"""
        # Use ninja movement state constants from sim_mechanics_doc.md
        # States: 0=Immobile, 1=Running, 2=Ground Sliding, 3=Jumping, 4=Falling, 5=Wall Sliding
```

4. **Modify Edge Feature Encoding** (`nclone/graph/graph_builder.py` lines 208-217):
```python
# Build edge features with physics information
edge_feat = np.zeros(self.edge_feature_dim, dtype=np.float32)
edge_feat[edge_type] = 1.0  # One-hot edge type
edge_feat[len(EdgeType)] = dx  # Direction x
edge_feat[len(EdgeType) + 1] = dy  # Direction y  
edge_feat[len(EdgeType) + 2] = cost  # Traversability cost

# NEW: Add trajectory parameters
if trajectory_result:
    edge_feat[len(EdgeType) + 3] = trajectory_result.time_of_flight
    edge_feat[len(EdgeType) + 4] = trajectory_result.energy_cost  
    edge_feat[len(EdgeType) + 5] = trajectory_result.success_probability
    edge_feat[len(EdgeType) + 6] = trajectory_result.min_velocity
    edge_feat[len(EdgeType) + 7] = trajectory_result.max_velocity
    edge_feat[len(EdgeType) + 8] = 1.0 if trajectory_result.requires_jump else 0.0
    edge_feat[len(EdgeType) + 9] = 1.0 if trajectory_result.requires_wall_contact else 0.0
```

5. **Update GNN Input Dimensions** (`npp_rl/config/phase2_config.py` line 37):
```python
edge_feature_dim: int = 16   # Updated from 9 to 16
```

6. **Create Physics Validation Tests** (`tests/test_trajectory_physics.py`):
   - Test trajectory calculations against known N++ physics
   - Validate jump distances match simulation results
   - Test edge feasibility with different ninja states

**Expected Outcome**: Edges contain physics-validated trajectory information enabling GNN to reason about movement feasibility, energy costs, and timing requirements.

#### Task 1.2: Momentum-Augmented Node Representations  
**Priority**: High | **Complexity**: Medium | **Dependencies**: Task 1.1 | **Files**: 2 new, 3 modified

**Objective**: Extend node features to include ninja velocity state and momentum information for physics-aware pathfinding.

**Current Gap**: `GraphBuilder._extract_sub_cell_features()` only includes tile type and ninja position flag. Need velocity state `(vx, vy)` and physics state from ninja's 9 movement states (sim_mechanics_doc.md).

**Detailed Implementation**:

1. **Modify Node Feature Dimensions** (`nclone/graph/graph_builder.py` lines 80-89):
```python
# Expand node_feature_dim from 67 to 85
self.node_feature_dim = (
    self.tile_type_dim +  # One-hot tile type (38)
    4 +  # Solidity flags (solid, half, slope, hazard)
    self.entity_type_dim +  # One-hot entity type (20)
    4 +  # Entity state (active, position_x, position_y, custom_state)
    1 +  # Ninja position flag
    # NEW: Physics state features (18 additional)
    2 +  # Ninja velocity (vx, vy) normalized by MAX_HOR_SPEED
    1 +  # Velocity magnitude
    1 +  # Movement state (0-9 from sim_mechanics_doc.md)
    3 +  # Contact flags (ground_contact, wall_contact, airborne)
    2 +  # Momentum direction (normalized)
    1 +  # Kinetic energy (0.5 * m * v²)
    1 +  # Potential energy (relative to level bottom)
    5 +  # Input buffers (jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state)
    2    # Physics capabilities (can_jump, can_wall_jump)
)
```

2. **Create Physics State Extractor** (`npp_rl/models/physics_state_extractor.py`):
```python
class PhysicsStateExtractor:
    def __init__(self):
        self.max_hor_speed = MAX_HOR_SPEED  # 3.333 pixels/frame
        self.level_height = MAP_TILE_HEIGHT * TILE_PIXEL_SIZE  # 552 pixels
        
    def extract_ninja_physics_state(self, ninja_position, ninja_velocity, ninja_state, level_data):
        """Extract comprehensive physics state for node features"""
        vx, vy = ninja_velocity
        x, y = ninja_position
        
        # Normalize velocity components
        vx_norm = vx / self.max_hor_speed
        vy_norm = vy / self.max_hor_speed
        
        # Calculate velocity magnitude
        vel_magnitude = math.sqrt(vx*vx + vy*vy) / self.max_hor_speed
        
        # Movement state from ninja (0-9 states from sim_mechanics_doc.md)
        movement_state = ninja_state.movement_state / 9.0  # Normalize to [0,1]
        
        # Contact state detection
        ground_contact = 1.0 if ninja_state.movement_state in [0, 1, 2] else 0.0  # Immobile, Running, Ground Sliding
        wall_contact = 1.0 if ninja_state.movement_state == 5 else 0.0  # Wall Sliding
        airborne = 1.0 if ninja_state.movement_state in [3, 4] else 0.0  # Jumping, Falling
        
        # Momentum direction (normalized)
        if vel_magnitude > 0.01:
            momentum_x = vx / (vel_magnitude * self.max_hor_speed)
            momentum_y = vy / (vel_magnitude * self.max_hor_speed)
        else:
            momentum_x = momentum_y = 0.0
            
        # Energy calculations
        kinetic_energy = 0.5 * (vx*vx + vy*vy) / (self.max_hor_speed * self.max_hor_speed)
        potential_energy = (self.level_height - y) / self.level_height  # Normalized height
        
        # Input buffers from ninja state (from sim_mechanics_doc.md)
        jump_buffer = ninja_state.jump_buffer / 5.0 if hasattr(ninja_state, 'jump_buffer') else 0.0
        floor_buffer = ninja_state.floor_buffer / 5.0 if hasattr(ninja_state, 'floor_buffer') else 0.0
        wall_buffer = ninja_state.wall_buffer / 5.0 if hasattr(ninja_state, 'wall_buffer') else 0.0
        launch_pad_buffer = ninja_state.launch_pad_buffer / 4.0 if hasattr(ninja_state, 'launch_pad_buffer') else 0.0
        input_state = 1.0 if ninja_state.jump_input else 0.0
        
        # Physics capabilities based on current state
        can_jump = 1.0 if ground_contact or (jump_buffer > 0) or (wall_contact and wall_buffer > 0) else 0.0
        can_wall_jump = 1.0 if wall_contact or (wall_buffer > 0) else 0.0
        
        return np.array([
            vx_norm, vy_norm, vel_magnitude, movement_state,
            ground_contact, wall_contact, airborne,
            momentum_x, momentum_y, kinetic_energy, potential_energy,
            jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state,
            can_jump, can_wall_jump
        ], dtype=np.float32)
```

3. **Modify Sub-Cell Feature Extraction** (`nclone/graph/graph_builder.py` lines 229-285):
```python
def _extract_sub_cell_features(self, level_data, sub_row, sub_col, ninja_position, ninja_velocity=None, ninja_state=None):
    """Extract features for a sub-grid cell node with physics state."""
    features = np.zeros(self.node_feature_dim, dtype=np.float32)
    
    # Existing tile and solidity features (lines 248-268)...
    
    # Entity features (zero for grid cells) (lines 270-273)...
    
    # Ninja position flag (lines 275-283)...
    ninja_offset = entity_offset + self.entity_type_dim + 4
    sub_cell_x = sub_col * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
    sub_cell_y = sub_row * SUB_CELL_SIZE + SUB_CELL_SIZE // 2
    
    ninja_in_cell = (abs(ninja_position[0] - sub_cell_x) < SUB_CELL_SIZE // 2 and
                     abs(ninja_position[1] - sub_cell_y) < SUB_CELL_SIZE // 2)
    features[ninja_offset] = 1.0 if ninja_in_cell else 0.0
    
    # NEW: Physics state features (only for ninja's current cell)
    physics_offset = ninja_offset + 1
    if ninja_in_cell and ninja_velocity is not None and ninja_state is not None:
        physics_features = self.physics_extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, ninja_state, level_data
        )
        features[physics_offset:physics_offset + len(physics_features)] = physics_features
    
    return features
```

4. **Update Graph Builder Interface** (`nclone/graph/graph_builder.py` lines 98-103):
```python
def build_graph(self, level_data, ninja_position, entities, ninja_velocity=None, ninja_state=None):
    """Build graph representation with physics state."""
    # Add ninja_velocity and ninja_state parameters
    # Pass these to _extract_sub_cell_features calls
    
    # Initialize physics extractor
    if not hasattr(self, 'physics_extractor'):
        from npp_rl.models.physics_state_extractor import PhysicsStateExtractor
        self.physics_extractor = PhysicsStateExtractor()
```

5. **Create Momentum History Buffer** (`npp_rl/models/momentum_tracker.py`):
```python
class MomentumTracker:
    """Track ninja momentum history for temporal physics reasoning"""
    def __init__(self, history_length=10):
        self.history_length = history_length
        self.velocity_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length)
        
    def update(self, position, velocity):
        """Update momentum history"""
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        
    def get_acceleration(self):
        """Calculate current acceleration from velocity history"""
        if len(self.velocity_history) < 2:
            return (0.0, 0.0)
        
        recent_vel = self.velocity_history[-1]
        prev_vel = self.velocity_history[-2]
        return (recent_vel[0] - prev_vel[0], recent_vel[1] - prev_vel[1])
        
    def predict_future_position(self, frames_ahead=5):
        """Predict ninja position N frames in the future"""
        if len(self.velocity_history) < 2:
            return self.position_history[-1] if self.position_history else (0, 0)
            
        current_pos = self.position_history[-1]
        current_vel = self.velocity_history[-1]
        acceleration = self.get_acceleration()
        
        # Simple kinematic prediction: pos = pos0 + vel*t + 0.5*acc*t²
        t = frames_ahead
        future_x = current_pos[0] + current_vel[0] * t + 0.5 * acceleration[0] * t * t
        future_y = current_pos[1] + current_vel[1] * t + 0.5 * acceleration[1] * t * t
        
        return (future_x, future_y)
```

6. **Update Configuration** (`npp_rl/config/phase2_config.py` line 36):
```python
node_feature_dim: int = 85  # Updated from 67 to 85
```

7. **Integration with Environment** (modify environment wrapper to pass ninja physics state):
   - Update observation extraction to include `ninja_velocity` and `ninja_state`
   - Modify graph observation building to pass physics parameters

**Expected Outcome**: Nodes contain comprehensive physics state enabling momentum-dependent pathfinding, jump timing prediction, and physics-aware spatial reasoning.

#### Task 1.3: Conditional Edge Activation System
**Priority**: Medium | **Complexity**: High | **Dependencies**: Tasks 1.1, 1.2 | **Files**: 3 new, 2 modified

**Objective**: Implement dynamic edge masking based on ninja's current physics state and movement capabilities.

**Current Gap**: `GraphSAGELayer.forward()` processes all edges equally. Need state-dependent edge availability using ninja physics state.

**Detailed Implementation**:

1. **Create Dynamic Edge Masker** (`npp_rl/models/conditional_edges.py`):
```python
class ConditionalEdgeMasker:
    def __init__(self):
        self.min_wall_jump_speed = 1.0  # Minimum horizontal speed for wall jumps
        self.min_jump_energy = 0.5      # Minimum energy for upward jumps
        
    def compute_dynamic_edge_mask(self, edge_features, ninja_physics_state, base_edge_mask):
        """Compute which edges are available based on current ninja state"""
        # edge_features: [num_edges, 16] with trajectory info from Task 1.1
        # ninja_physics_state: [18] physics features from Task 1.2
        
        dynamic_mask = base_edge_mask.clone()
        
        # Extract ninja state
        vx, vy = ninja_physics_state[0], ninja_physics_state[1]  # Normalized velocity
        vel_magnitude = ninja_physics_state[2]
        movement_state = ninja_physics_state[3] * 9.0  # Denormalize to 0-9
        ground_contact = ninja_physics_state[4]
        wall_contact = ninja_physics_state[5]
        can_jump = ninja_physics_state[16]
        can_wall_jump = ninja_physics_state[17]
        
        for edge_idx in range(edge_features.shape[0]):
            if not base_edge_mask[edge_idx]:
                continue
                
            edge_type = torch.argmax(edge_features[edge_idx, :6])  # First 6 are edge type one-hot
            requires_jump = edge_features[edge_idx, 14]  # From Task 1.1
            requires_wall_contact = edge_features[edge_idx, 15]
            min_velocity = edge_features[edge_idx, 12]
            
            # Disable edges based on physics constraints
            if edge_type == EdgeType.JUMP and can_jump < 0.5:
                dynamic_mask[edge_idx] = 0
            elif edge_type == EdgeType.WALL_JUMP and can_wall_jump < 0.5:
                dynamic_mask[edge_idx] = 0
            elif requires_jump and vel_magnitude < min_velocity:
                dynamic_mask[edge_idx] = 0
            elif requires_wall_contact and wall_contact < 0.5:
                dynamic_mask[edge_idx] = 0
                
        return dynamic_mask
```

2. **Modify GraphSAGE Layer** (`npp_rl/models/gnn.py` lines 62-101):
```python
def forward(self, node_features, edge_index, node_mask, edge_mask, ninja_physics_state=None, edge_features=None):
    """Forward pass with conditional edge masking"""
    
    # Apply dynamic edge masking if physics state available
    if ninja_physics_state is not None and edge_features is not None:
        if not hasattr(self, 'edge_masker'):
            from npp_rl.models.conditional_edges import ConditionalEdgeMasker
            self.edge_masker = ConditionalEdgeMasker()
            
        dynamic_edge_mask = self.edge_masker.compute_dynamic_edge_mask(
            edge_features, ninja_physics_state, edge_mask
        )
    else:
        dynamic_edge_mask = edge_mask
    
    # Use dynamic_edge_mask instead of edge_mask in aggregation
    neighbor_features = self._aggregate_neighbors(
        node_features, edge_index, node_mask, dynamic_edge_mask
    )
    # ... rest of forward pass
```

3. **Create Physics Constraint Validator** (`npp_rl/models/physics_constraints.py`):
```python
class PhysicsConstraintValidator:
    """Validate movement sequences against N++ physics rules"""
    
    def validate_movement_sequence(self, movement_chain, ninja_state):
        """Check if a sequence of movements is physically possible"""
        # movement_chain: List of (edge_type, trajectory_params)
        
        current_velocity = ninja_state.velocity
        current_position = ninja_state.position
        energy_budget = self.calculate_available_energy(ninja_state)
        
        for movement in movement_chain:
            edge_type, trajectory = movement
            
            # Check energy requirements
            if trajectory.energy_cost > energy_budget:
                return False, "Insufficient energy"
                
            # Check velocity constraints
            if edge_type == EdgeType.WALL_JUMP:
                if abs(current_velocity[0]) < self.min_wall_jump_speed:
                    return False, "Insufficient horizontal velocity for wall jump"
                    
            # Update state for next movement
            current_velocity = trajectory.final_velocity
            current_position = trajectory.end_position
            energy_budget -= trajectory.energy_cost
            
        return True, "Valid sequence"
```

**Expected Outcome**: GNN processes only physically feasible edges based on ninja's current state, enabling more realistic pathfinding decisions.

### Phase 2: Hierarchical Graph Architectures

#### Task 2.1: Multi-Resolution Graph Processing
**Priority**: Medium | **Complexity**: High | **Dependencies**: Phase 1 | **Files**: 3 new, 1 modified

**Objective**: Implement hierarchical GNN that processes local movement decisions and global pathfinding simultaneously.

**Current Gap**: Single-resolution GraphSAGE. Need multi-scale processing as described in graph plan's "hierarchical graph neural network architectures."

**Key Implementation Points**:

1. **Create Hierarchical Graph Builder** (`nclone/graph/hierarchical_builder.py`):
   - Build 3 resolution levels: sub-cell (6px), tile (24px), region (96px)
   - Use graph coarsening to create higher-level representations
   - Add inter-scale connectivity for information flow

2. **Implement DiffPool GNN** (`npp_rl/models/diffpool_gnn.py`):
   - Differentiable graph pooling with soft cluster assignments
   - Learnable hierarchical representations
   - End-to-end training capability

3. **Multi-Scale Feature Fusion**:
   - Attention mechanisms for scale-dependent weighting
   - Unified embedding space across resolutions

**Expected Outcome**: GNN processes both precise local movements and strategic global planning.

#### Task 2.2: Heterogeneous Graph Transformer Integration  
**Priority**: Medium | **Complexity**: High | **Dependencies**: Task 2.1 | **Files**: 2 new, 2 modified

**Objective**: Implement HGT for different entity types (platforms, enemies, collectibles, switches).

**Key Implementation Points**:

1. **HGT Architecture** (`npp_rl/models/hgt_gnn.py`):
   - Type-specific attention mechanisms for different node types
   - Edge-type dependent message passing parameters
   - Cross-type attention for functional relationships (switch→door)

2. **Entity Type Specialization**:
   - Separate embedding spaces for different entity types
   - Specialized processing for interactive elements
   - Hazard-aware attention mechanisms

**Expected Outcome**: GNN understands and reasons about different game elements with specialized processing.

### Phase 3: Integration and Optimization

#### Task 3.1: Hybrid CNN-GNN Architecture Enhancement
**Priority**: High | **Complexity**: Medium | **Dependencies**: Phases 1-2 | **Files**: 1 modified

**Current Gap**: Basic concatenation fusion. Need sophisticated cross-modal attention as described in graph plan.

**Key Implementation Points**:

1. **Modify Fusion Network**:
```python
# Replace simple concatenation with cross-modal attention
self.cross_modal_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
self.graph_visual_fusion = nn.TransformerEncoderLayer(d_model=512, nhead=8)

# Add graph-informed CNN feature extraction
self.spatial_attention = SpatialAttentionModule(graph_dim=256, visual_dim=512)
```

2. **Graph-Informed Visual Processing**:
   - Use graph structure to guide CNN attention
   - Spatial attention mechanisms based on graph connectivity
   - Multi-modal gradient flow optimization

**Expected Outcome**: Unified architecture leveraging both visual patterns and structural relationships.

#### Task 3.2: Real-Time Graph Adaptation
**Priority**: Medium | **Complexity**: High | **Dependencies**: Task 3.1 | **Files**: 2 new, 1 modified

**Objective**: Implement efficient real-time graph updates for dynamic environments.

**Key Implementation Points**:

1. **Event-Driven Update System** (`npp_rl/environments/dynamic_graph_wrapper.py`):
   - Efficient graph modification algorithms
   - Event-based edge activation/deactivation
   - Incremental update mechanisms

2. **Dynamic Constraint Propagation**:
   - Priority-based update systems
   - Computational budget management
   - Temporal edge availability windows

**Expected Outcome**: Graph adapts to environmental changes while maintaining real-time performance.

## Implementation Priority and Timeline

### Phase 1: Physics Foundation
- Task 1.1 (Trajectory-Based Edge Features)
- Task 1.2 (Momentum-Augmented Node Representations)

### Phase 2: Advanced Processing  
- Task 1.3 (Conditional Edge Activation)
- Task 3.1 (Hybrid CNN-GNN Enhancement)

### Phase 3: Hierarchical Systems
- Task 2.1 (Multi-Resolution Processing)
- Task 2.2 (Heterogeneous Graph Transformers)

### Phase 4: Optimization
- Task 3.2 (Real-Time Adaptation)
- Performance optimization and integration testing

## Technical Considerations

### Computational Complexity
- **Current Graph Size**: ~15,856 nodes, ~126,848 edges (sub-grid resolution)
- **Enhanced Features**: Node features 67→85 (+27%), Edge features 9→16 (+78%)
- **Memory Impact**: Expect 2-3x increase with physics features and hierarchical processing
- **Training Overhead**: Additional 30-40% computational cost for enhanced graph processing

### Integration Points
- **Environment Wrapper**: Modify to pass ninja physics state to graph builder
- **Feature Extraction**: `npp_rl/models/feature_extractors.py` - Enhanced fusion architecture
- **Training Loop**: `train_phase2.py` - Updated configuration and monitoring
- **Configuration**: `npp_rl/config/phase2_config.py` - New physics-aware parameters

### Validation Strategy
- **Physics Accuracy**: Validate trajectory calculations against N++ simulation
- **Performance Benchmarks**: Compare pathfinding accuracy with traditional A* algorithms  
- **Training Metrics**: Monitor convergence speed and sample efficiency improvements
- **Ablation Studies**: Test individual components (trajectory features, momentum state, conditional edges)

## Success Metrics

### Quantitative Metrics
- **Pathfinding Accuracy**: >95% success rate on complex navigation tasks
- **Training Efficiency**: 25-35% improvement in sample efficiency over baseline
- **Physics Realism**: <3% deviation from ground-truth physics calculations
- **Computational Performance**: <75ms graph processing time per frame (including physics)

### Qualitative Metrics  
- **Movement Quality**: More natural and physics-aware movement patterns
- **Strategic Planning**: Improved long-term pathfinding and goal achievement
- **Adaptability**: Better performance on unseen level configurations
- **Robustness**: Consistent performance across different level types and difficulties

## Research References

### Key Papers
1. **Physics-Informed Neural Networks**: Raissi et al. (2019) - "Physics-informed neural networks: A deep learning framework"
2. **Heterogeneous Graph Transformers**: Wang et al. (2020) - "Heterogeneous Graph Transformer"
3. **Hierarchical Graph Networks**: Ying et al. (2018) - "Hierarchical Graph Representation Learning with Differentiable Pooling"
4. **Graph Neural Networks**: Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs"

### Frameworks and Libraries
- **PyTorch Geometric**: For heterogeneous graph processing and advanced GNN layers
- **DGL (Deep Graph Library)**: For dynamic graph handling and temporal modeling
- **NetworkX**: For graph analysis and algorithm implementation

This implementation plan provides a comprehensive, code-ready roadmap for integrating advanced graph neural network techniques into the N++ reinforcement learning project, with specific file locations, line numbers, and implementation details that enable successful execution by programming agents.
