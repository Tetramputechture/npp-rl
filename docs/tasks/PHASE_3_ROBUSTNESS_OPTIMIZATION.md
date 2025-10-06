# Phase 3: Robustness & Optimization

**Dependencies**: Phase 2 completed  

## Overview

This phase focuses on optimizing the hierarchical system for robustness and performance. The goal is to achieve consistent performance across diverse level types, optimize model architecture for efficiency, and establish comprehensive evaluation frameworks. This phase transforms the system from a working prototype to a production-ready agent.

## Objectives

1. Optimize model architecture for efficiency and generalization
2. Enhance ICM integration for physics-uncertain exploration
3. Establish comprehensive evaluation framework
4. Implement hardware optimization for training efficiency
5. Achieve robust performance across level complexity spectrum

## Task Breakdown

### Task 3.1: Model Architecture Optimization

**What we want to do**: Optimize the neural network architecture to balance complexity with performance, focusing on the simplified 6-node-type entity system and efficient attention mechanisms.

**Current state**:
- HGT implementation with 6 node types from original design
- Complex cross-modal attention mechanisms
- No systematic architecture comparison or optimization
- Potential overengineering for completion-focused task

**Files to modify**:
- `npp_rl/models/hgt_gnn.py`
- `npp_rl/models/hgt_config.py`
- `npp_rl/models/attention_mechanisms.py`
- `npp_rl/feature_extractors/hgt_multimodal.py`
- `npp_rl/models/simplified_architectures.py` (new)

**Architecture optimization requirements**:

#### Graph Neural Network Simplification
**Current HGT complexity**:
- 6 node types: tile, ninja, gold, switch, hazard, collectible
- 3 edge types: adjacent, reachable, interactive
- Multi-head type-specific attention
- Complex message passing with type embeddings

**Target simplified architecture**:
- 6 node types: tile, ninja, mine, exit_switch, exit_door, locked_door
- 2 edge types: adjacent, reachable
- Simplified attention mechanism (GAT or basic HGT)
- Reduced embedding dimensions

**Architecture comparison framework**:
```python
class ArchitectureComparison:
    def __init__(self):
        self.architectures = {
            'full_hgt': FullHGTModel(node_types=6, edge_types=3),
            'simplified_hgt': SimplifiedHGTModel(node_types=4, edge_types=2),
            'gat': GATModel(node_types=4),
            'gcn': GCNModel(node_types=4),
            'mlp_baseline': MLPModel()  # No graph processing
        }
    
    def compare_architectures(self, test_levels):
        results = {}
        for name, model in self.architectures.items():
            results[name] = self.evaluate_architecture(model, test_levels)
        return results
```

#### Attention Mechanism Optimization
**Current cross-modal attention**:
- Multi-head attention between temporal, spatial, graph, and state features
- Complex fusion network with residual connections
- High computational overhead

**Optimization options**:
1. **Simple concatenation**: Baseline approach, minimal computation
2. **Single-head attention**: Reduced complexity while maintaining cross-modal interaction
3. **Hierarchical attention**: Attention within modalities, then across modalities
4. **Adaptive attention**: Learn when to use attention vs simple fusion

**Implementation**:
```python
class OptimizedMultimodalFusion:
    def __init__(self, fusion_type='adaptive'):
        self.fusion_type = fusion_type
        
        if fusion_type == 'simple':
            self.fusion = SimpleConcatenation()
        elif fusion_type == 'single_head':
            self.fusion = SingleHeadAttention()
        elif fusion_type == 'hierarchical':
            self.fusion = HierarchicalAttention()
        elif fusion_type == 'adaptive':
            self.fusion = AdaptiveAttentionFusion()
    
    def forward(self, temporal_features, spatial_features, graph_features, state_features):
        return self.fusion(temporal_features, spatial_features, graph_features, state_features)
```

#### Performance Benchmarking
**Metrics to optimize**:
- Inference time per step (target: <10ms)
- Memory usage during training
- Training convergence speed
- Final performance on test levels
- Generalization across level types

**Benchmarking framework**:
```python
class ArchitectureBenchmark:
    def __init__(self):
        self.metrics = {
            'inference_time': [],
            'memory_usage': [],
            'training_speed': [],
            'final_performance': [],
            'generalization_score': []
        }
    
    def benchmark_architecture(self, model, test_suite):
        # Inference time benchmark
        inference_times = self.measure_inference_time(model)
        
        # Memory usage benchmark
        memory_usage = self.measure_memory_usage(model)
        
        # Training speed benchmark
        training_speed = self.measure_training_convergence(model)
        
        # Performance benchmark
        performance = self.evaluate_performance(model, test_suite)
        
        return {
            'inference_time': np.mean(inference_times),
            'memory_usage': memory_usage,
            'training_speed': training_speed,
            'performance': performance
        }
```

#### Architecture Selection Criteria
**Decision matrix**:
- **Performance weight**: 40% - Final completion rate on test levels
- **Efficiency weight**: 30% - Inference time and memory usage
- **Training speed weight**: 20% - Time to convergence
- **Generalization weight**: 10% - Performance across diverse levels

**Selection process**:
1. Implement all architecture variants
2. Train each on standardized training set
3. Evaluate on comprehensive test suite
4. Select architecture with best weighted score
5. Fine-tune selected architecture

**Acceptance criteria**:
- [ ] Architecture comparison framework implemented

---

#### Vision-Free Architecture Analysis

**Research Question**: Can the NPP-RL agent learn effectively without visual input by relying solely on graph, game state, and reachability features?

**Motivation**: Visual processing (3D CNN for temporal frames + 2D CNN for global view) represents significant computational overhead (~60% of inference time). Understanding whether visual input is necessary or redundant could enable dramatic performance improvements.

##### Available Non-Visual Information

The agent currently receives rich non-visual features that encode comprehensive game state:

**1. Game State Vector (30 features)**:
- **Core Movement State (8)**: Velocity magnitude/direction, movement state categories (ground/air/wall/special), airborne status
- **Input & Buffer State (5)**: Current horizontal/jump inputs, jump/floor/wall buffer states (critical for frame-perfect timing)
- **Surface Contact (6)**: Floor/wall/ceiling contact strength, floor normal strength, wall direction, surface slope
- **Momentum & Physics (4)**: Recent acceleration (x/y), momentum preservation, impact risk assessment
- **Entity Proximity (4)**: Nearest hazard distance, nearest collectible distance, hazard threat level, interaction cooldown
- **Level Progress (3)**: Switch activation status, exit accessibility, completion progress

**2. Graph Representation (up to 18,000 nodes, 144,000 edges)**:
- **Node Features**: Position (x, y), node type (6 types: tile, ninja, hazard, collectible, switch, exit)
- **Edge Features**: Connectivity type (adjacent, reachable), weights
- **Spatial Structure**: Encodes tile types, entity positions, level topology
- **Type-Specific Processing**: HGT processes different node/edge types with specialized attention

**3. Reachability Features (8 features)**:
- Reachable area ratio, current objective distance
- Switch accessibility, exit accessibility
- Hazard proximity, connectivity score
- Analysis confidence, computation time

##### Information Coverage Analysis

**What visual input provides**:
1. **Fine-grained spatial awareness**: Pixel-level detail of nearby tiles and slopes
2. **Temporal motion patterns**: 12-frame stack captures movement dynamics
3. **Global level layout**: 176x100 downsampled view of entire level
4. **Immediate surroundings**: Visual context around ninja position
5. **Slope angles and geometry**: Precise visual representation of terrain

**What non-visual features already encode**:
1. ✅ **Tile types and positions**: Graph nodes encode exact tile positions and types
2. ✅ **Entity locations**: Graph explicitly represents all entity positions
3. ✅ **Movement state**: Game state captures velocity, direction, airborne status, surface contact
4. ✅ **Physics state**: Acceleration, momentum, impact risk, surface normals/slopes
5. ✅ **Connectivity**: Graph edges encode adjacency and reachability
6. ✅ **Spatial relationships**: Graph topology + reachability features
7. ✅ **Objectives**: Switch/exit positions and accessibility explicitly provided
8. ✅ **Timing information**: Input buffers (5-frame windows) encode critical timing state

**Critical capabilities requiring visual input**:
1. ❓ **Precise slope angle perception**: Surface slopes are in game state (feature 19), but visual may help with complex curved geometry
2. ❓ **Immediate hazard patterns**: Visual patterns might help recognize dangerous configurations
3. ❓ **Movement pattern recognition**: Temporal stack captures recent movement, but game state has velocity/acceleration
4. ❓ **Spatial reasoning shortcuts**: Visual CNN may learn spatial shortcuts that graph processing doesn't capture efficiently

##### Comparative Analysis: Graph-Based vs Vision-Based RL

**Research Evidence**:

1. **Structured Representations in RL** (Pathak et al., 2019; Zambaldi et al., 2019):
   - Graph-based representations can match or exceed vision-based performance when structural information is relevant
   - Relational reasoning through GNNs provides strong inductive bias for structured environments
   - Particularly effective in environments with discrete entities and clear spatial relationships (✅ matches N++)

2. **Cross-Modality Transfer** (Lesort et al., 2019):
   - Policies can be trained with one modality and transferred to another
   - Latent representations can be learned that are modality-agnostic
   - Suggests vision may not be strictly necessary if other modalities capture essential information

3. **Tactile/Non-Visual RL** (Van Hoof et al., 2016):
   - Agents can achieve robust performance using non-visual sensory data
   - Key requirement: information richness and task-relevance of available features
   - Success demonstrated in manipulation tasks with tactile feedback only

4. **Attention Mechanisms for Perception** (Mnih et al., 2014):
   - Hard attention can enable effective learning with partial observations
   - Graph attention mechanisms can selectively focus on relevant spatial regions
   - Suggests graph-based attention may substitute for visual attention

**N++ Specific Considerations**:

**Arguments FOR vision-free learning**:
1. **Discrete tile structure**: N++ uses 24x24 pixel tiles with discrete types - perfectly captured by graph nodes
2. **Entity-centric gameplay**: Core challenges involve entity interactions (switches, mines, doors) - all explicitly in graph
3. **Physics state completeness**: Game state vector includes all physics parameters needed for movement decisions
4. **Timing criticality**: Frame-perfect timing depends on buffer states, which are explicitly provided, not visual cues
5. **Reachability information**: Strategic navigation already encoded in reachability features
6. **Deterministic physics**: Unlike real-world robotics, N++ physics is deterministic and fully observable through state features

**Arguments AGAINST vision-free learning**:
1. **Complex slope geometry**: Curved tiles (quarter moons, quarter pipes) have subtle geometry that may be clearer visually
2. **Spatial pattern recognition**: Visual CNNs excel at recognizing spatial patterns that may be harder for GNNs
3. **Immediate danger detection**: Visual processing might enable faster recognition of dangerous configurations
4. **Emergent visual features**: CNNs may learn useful intermediate representations not explicitly in graph/state
5. **Partial observability handling**: Visual frames might help handle any information gaps in explicit features

##### Feasibility Assessment

**HIGH FEASIBILITY scenarios (Recommended for testing)**:

**Scenario 1: Remove Global View Only**
- **Rationale**: Global view (176x100) provides strategic overview, but graph + reachability features encode same information more efficiently
- **Keep**: Local temporal frames (84x84x12) for immediate spatial awareness
- **Expected impact**: 30-40% inference speedup with minimal performance loss
- **Risk**: Low - local frames retain immediate spatial information

**Scenario 2: Keep Local Frames Only, No Global View**
- **Rationale**: Same as Scenario 1, already the architecture we should test
- **Implementation**: Disable global view CNN branch in HGTMultimodalExtractor
- **Expected impact**: Same as Scenario 1

**MEDIUM FEASIBILITY scenarios**:

**Scenario 3: Complete Vision-Free Architecture**
- **Rationale**: All spatial/structural information is in graph, all physics in game state
- **Architecture**: Graph + Game State + Reachability features only
- **Expected impact**: 60-70% inference speedup, uncertain performance impact
- **Risk**: Medium - requires thorough testing to validate information sufficiency

**Scenario 4: Vision-Free with Enhanced Graph Features**
- **Rationale**: Augment graph with additional spatial features to compensate for vision removal
- **Enhancements**: 
  - Add slope angle/curvature as node features
  - Include local tile neighborhood patterns in node embeddings
  - Augment with temporal velocity history in graph
- **Expected impact**: Better than Scenario 3, may approach vision-based performance
- **Risk**: Medium - requires graph feature engineering

**LOW FEASIBILITY scenarios**:

**Scenario 5: Hybrid Attention-Based Vision**
- **Rationale**: Use visual input only when graph-based features are uncertain
- **Implementation**: Learn to attend to vision vs graph based on context
- **Risk**: High complexity, unclear benefit over simpler approaches

##### Recommended Experimental Protocol

**Phase 1: Baseline Validation**
1. Establish baseline performance with full visual input (current architecture)
2. Measure inference time breakdown: Visual (3D CNN + 2D CNN) vs Graph vs State processing
3. Analyze feature importance through ablation studies

**Phase 2: Progressive Vision Reduction**
1. **Experiment 2.1**: Remove global view (Scenario 1)
   - Train for 5M timesteps
   - Evaluate success rate, efficiency, inference time
   - Analyze failure modes vs baseline

2. **Experiment 2.2**: Complete vision removal (Scenario 3)
   - Train for 5M timesteps with graph + state + reachability only
   - Compare convergence speed and final performance
   - Detailed analysis of spatial reasoning capabilities

3. **Experiment 2.3**: Enhanced graph features (Scenario 4)
   - If Experiment 2.2 shows deficits, augment graph features
   - Focus on areas where visual input seemed necessary
   - Retest and compare

**Phase 3: Comparative Analysis**
1. Compare all architectures on standardized test suite (Task 3.3)
2. Analyze per-category performance (simple, medium, complex, mine-heavy, exploration)
3. Identify specific scenarios where vision helps/hurts
4. Generate architecture recommendations

##### Implementation Considerations

**Modified HGTMultimodalExtractor**:
```python
class ConfigurableMultimodalExtractor(nn.Module):
    def __init__(
        self,
        observation_space,
        features_dim=512,
        use_temporal_frames=True,  # 84x84x12 local frames
        use_global_view=True,      # 176x100 global view
        use_graph=True,
        use_state=True,
        use_reachability=True,
        enhanced_graph_features=False
    ):
        self.modality_config = {
            'temporal': use_temporal_frames,
            'global': use_global_view,
            'graph': use_graph,
            'state': use_state,
            'reachability': use_reachability
        }
        
        # Initialize only enabled modalities
        if use_temporal_frames:
            self.temporal_cnn = TemporalCNN3D(...)
        if use_global_view:
            self.spatial_cnn = SpatialCNN2D(...)
        if use_graph:
            if enhanced_graph_features:
                self.graph_processor = EnhancedHGT(...)
            else:
                self.graph_processor = HGTEncoder(...)
        # ... etc
        
        # Adaptive fusion based on enabled modalities
        self.fusion = AdaptiveFusion(enabled_modalities=self.modality_config)
```

**Testing Framework**:
```python
class VisionAblationStudy:
    def __init__(self):
        self.configurations = {
            'full_vision': {'temporal': True, 'global': True, 'graph': True},
            'local_only': {'temporal': True, 'global': False, 'graph': True},
            'vision_free': {'temporal': False, 'global': False, 'graph': True},
            'enhanced_graph': {'temporal': False, 'global': False, 'graph': True, 
                              'enhanced_graph_features': True}
        }
    
    def run_study(self, timesteps=5_000_000):
        results = {}
        for name, config in self.configurations.items():
            model = self.train_model(config, timesteps)
            results[name] = self.evaluate_model(model)
        return self.compare_results(results)
```

##### Expected Outcomes & Recommendations

**Optimistic Scenario** (60% probability):
- Vision-free or local-frames-only achieves >90% of full-vision performance
- Inference time reduces by 40-60%
- Training becomes more sample-efficient due to simpler architecture
- **Recommendation**: Adopt vision-free or local-only architecture

**Moderate Scenario** (30% probability):
- Vision-free achieves 70-85% of full-vision performance
- Specific failure modes in complex geometric scenarios
- Enhanced graph features bridge most of the gap
- **Recommendation**: Use local frames only, or enhanced graph features

**Pessimistic Scenario** (10% probability):
- Vision-free performs <70% of full-vision baseline
- Critical spatial reasoning depends on visual processing
- Graph features insufficient for complex geometries
- **Recommendation**: Keep full visual processing, optimize CNN architectures instead

**Key Decision Factors**:
1. **Performance threshold**: If vision-free achieves >85% of full-vision success rate, the 60% inference speedup is likely worth the trade-off
2. **Generalization**: Test across all level categories - if vision-free generalizes better, it may be preferred even with slightly lower average performance
3. **Training efficiency**: Simpler architectures may train faster and require less data
4. **Real-time constraints**: If deployment requires <5ms inference, vision-free may be necessary regardless of performance impact

##### Integration with Task 3.1

**Add to architecture comparison framework**:
```python
class ArchitectureComparison:
    def __init__(self):
        self.architectures = {
            'full_hgt': FullHGTModel(node_types=6, edge_types=3),
            'simplified_hgt': SimplifiedHGTModel(node_types=4, edge_types=2),
            'gat': GATModel(node_types=4),
            'gcn': GCNModel(node_types=4),
            'mlp_baseline': MLPModel(),  # No graph processing
            
            # Vision ablation variants
            'full_vision': FullMultimodalExtractor(temporal=True, global=True),
            'local_vision_only': MultimodalExtractor(temporal=True, global=False),
            'vision_free': VisionFreeExtractor(temporal=False, global=False),
            'enhanced_graph': EnhancedGraphExtractor(enhanced_features=True)
        }
```

**Acceptance Criteria Updates**:
- [ ] Vision ablation study completed with all scenarios tested
- [ ] Performance comparison across vision configurations documented
- [ ] Inference time improvements measured and validated
- [ ] Optimal vision configuration selected based on performance/efficiency trade-off
- [ ] If vision-free selected, validate across all level categories (Task 3.3 test suite)

---

### Task 3.2: Advanced ICM Integration

**What we want to do**: Optimize ICM for completion-focused exploration in physics-uncertain environments, with reachability-modulated curiosity and subtask-aware exploration.

**Current state**:
- Basic ICM implementation with reachability awareness
- ICM operates at low-level policy level
- No sophisticated curiosity modulation
- Limited integration with hierarchical decision making

**Files to modify**:
- `npp_rl/intrinsic/icm.py`
- `npp_rl/intrinsic/reachability_exploration.py`
- `npp_rl/hrl/subtask_policies.py`
- `npp_rl/intrinsic/advanced_icm.py` (new)

**Advanced ICM features**:

#### Reachability-Modulated Curiosity
**Concept**: Focus curiosity on areas that are reachable but unexplored, reducing wasted exploration in unreachable areas.

**Implementation**:
```python
class ReachabilityModulatedICM:
    def __init__(self, base_icm, reachability_system):
        self.base_icm = base_icm
        self.reachability_system = reachability_system
        self.exploration_history = {}
    
    def calculate_curiosity_reward(self, state, action, next_state):
        # Base ICM curiosity
        base_curiosity = self.base_icm.calculate_curiosity(state, action, next_state)
        
        # Reachability modulation
        reachability_score = self.get_reachability_score(next_state)
        exploration_novelty = self.get_exploration_novelty(next_state)
        
        # Modulate curiosity based on reachability and exploration history
        modulated_curiosity = base_curiosity * reachability_score * exploration_novelty
        
        return modulated_curiosity
    
    def get_reachability_score(self, state):
        # Higher score for reachable areas, lower for unreachable
        ninja_pos = (state['ninja_x'], state['ninja_y'])
        reachability_result = self.reachability_system.quick_check(ninja_pos, ...)
        
        # Score based on connectivity and objective accessibility
        connectivity = reachability_result.reachable_positions / total_positions
        objective_accessibility = reachability_result.objective_reachable
        
        return 0.5 * connectivity + 0.5 * objective_accessibility
```

#### Subtask-Aware Exploration
**Concept**: Modulate curiosity based on current subtask to focus exploration on task-relevant areas.

**Subtask-specific curiosity weights**:
- `navigate_to_exit_switch`: High curiosity near exit switch, moderate elsewhere
- `navigate_to_locked_switch`: High curiosity near locked switches, low elsewhere
- `navigate_to_exit_door`: High curiosity near exit door, low elsewhere
- `explore_for_switches`: Uniform high curiosity, focus on unexplored areas

**Implementation**:
```python
def modulate_curiosity_for_subtask(self, base_curiosity, current_subtask, state):
    ninja_pos = (state['ninja_x'], state['ninja_y'])
    
    if current_subtask == 'navigate_to_exit_switch':
        exit_switch_distance = calculate_distance(ninja_pos, state['exit_switch_pos'])
        proximity_weight = max(0.1, 1.0 - exit_switch_distance / 10.0)
        return base_curiosity * proximity_weight
    
    elif current_subtask == 'navigate_to_locked_switch':
        nearest_switch_distance = min([
            calculate_distance(ninja_pos, switch_pos) 
            for switch_pos in state['locked_switch_positions']
        ])
        proximity_weight = max(0.1, 1.0 - nearest_switch_distance / 10.0)
        return base_curiosity * proximity_weight
    
    elif current_subtask == 'navigate_to_exit_door':
        exit_door_distance = calculate_distance(ninja_pos, state['exit_door_pos'])
        proximity_weight = max(0.1, 1.0 - exit_door_distance / 8.0)
        return base_curiosity * proximity_weight
    
    elif current_subtask == 'explore_for_switches':
        # Encourage broad exploration
        exploration_bonus = self.calculate_exploration_bonus(ninja_pos)
        return base_curiosity * (1.0 + exploration_bonus)
    
    return base_curiosity
```

#### Physics-Uncertainty Compensation
**Concept**: Use ICM to compensate for inability to predict physics accurately, focusing on learning movement patterns emergently.

**Enhanced forward model**:
- Predict next reachability features instead of raw physics state
- Focus on strategic outcomes rather than precise physics
- Learn movement affordances in different contexts

**Implementation**:
```python
class PhysicsUncertaintyICM:
    def __init__(self):
        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel()
        self.reachability_predictor = ReachabilityPredictor()
    
    def forward_prediction(self, state, action):
        # Predict strategic outcomes rather than precise physics
        predicted_reachability = self.reachability_predictor(state, action)
        predicted_position_change = self.forward_model(state, action)
        
        return {
            'reachability_change': predicted_reachability,
            'position_change': predicted_position_change
        }
    
    def calculate_prediction_error(self, prediction, actual_next_state):
        # Weight reachability prediction more heavily than position
        reachability_error = self.reachability_prediction_error(
            prediction['reachability_change'], 
            actual_next_state
        )
        position_error = self.position_prediction_error(
            prediction['position_change'], 
            actual_next_state
        )
        
        # Weighted combination favoring strategic prediction
        return 0.7 * reachability_error + 0.3 * position_error
```

#### Adaptive Exploration Scheduling
**Concept**: Adjust exploration intensity based on training progress and performance.

**Exploration scheduling**:
- High exploration early in training
- Reduce exploration as performance improves
- Increase exploration when performance plateaus
- Subtask-specific exploration schedules

**Implementation**:
```python
class AdaptiveExplorationScheduler:
    def __init__(self):
        self.performance_history = []
        self.exploration_weights = {
            'navigate_to_exit_switch': 1.0,
            'navigate_to_locked_switch': 1.0,
            'navigate_to_exit_door': 0.8,
            'explore_for_switches': 1.5
        }
    
    def update_exploration_weights(self, recent_performance):
        # Increase exploration if performance plateaus
        if self.detect_plateau(recent_performance):
            for subtask in self.exploration_weights:
                self.exploration_weights[subtask] *= 1.2
        
        # Decrease exploration if performance is improving
        elif self.detect_improvement(recent_performance):
            for subtask in self.exploration_weights:
                self.exploration_weights[subtask] *= 0.95
```

**Acceptance criteria**:
- [ ] Reachability-modulated curiosity focuses exploration effectively
- [ ] Subtask-aware exploration improves task-relevant discovery
- [ ] Physics-uncertainty compensation enables robust movement learning
- [ ] Adaptive exploration scheduling optimizes exploration over training
- [ ] ICM integration improves completion rates on complex levels
- [ ] Exploration efficiency metrics show improvement vs baseline

**Testing requirements**:
- Exploration efficiency measurements
- Curiosity reward distribution analysis
- Performance comparison with and without advanced ICM
- Ablation studies on different ICM components

---

### Task 3.3: Comprehensive Evaluation Framework

**What we want to do**: Establish a comprehensive evaluation framework that measures performance across diverse level types and provides detailed analysis of agent capabilities.

**Current state**:
- Basic success rate measurement on training levels
- No systematic evaluation across level complexity spectrum
- Limited performance metrics and analysis
- No standardized test suite for consistent evaluation

**Files to create**:
- `npp_rl/evaluation/evaluation_framework.py` (new)
- `npp_rl/evaluation/test_suite.py` (new)
- `npp_rl/evaluation/metrics.py` (new)
- `npp_rl/evaluation/analysis_tools.py` (new)
- `tests/evaluation/test_evaluation_framework.py` (new)

**Evaluation framework requirements**:

#### Test Suite Design
**Level complexity categories**:
1. **Simple levels** (baseline): Single switch, direct path to exit
2. **Medium levels**: 1-3 switches with simple dependencies. Jumps may be required.
3. **Complex levels**: 4+ switches with complex dependency chains. Jumps may be required.
4. **Mine-heavy levels**: Significant mine obstacles requiring strategic navigation. Jumps may be required.
5. **Exploration levels**: Hidden switches requiring extensive exploration. Jumps and wall jumps may be required.

**Test suite composition**:
- 50 simple levels for baseline performance
- 100 medium levels for standard evaluation
- 50 complex levels for advanced capability testing
- 30 mine-heavy levels for safety evaluation
- 20 exploration levels for discovery capability testing

**Building the Test Suite Maps**:

We can use the tools in `nclone/map_generation/map.py` and `nclone/map_generation/map_generator.py` to generate maps as well as the related files in the map_generation/ folder.
Please place the logic for each map you generate in a new file in `nclone/map_generation/generate_test_suite_maps.py`. This file should contain logic to create
the entire test suite. Our simple levels should have a navigable space of empty tiles of at least 1 tile high, and at least 3 empty tiles wide. The most simple level is a one tile high and 3 tile wide map where the ninja is at one side, the exit switch is in the middle, and the exit door is on the other side. Some of this logic is located in `nclone/map_generation/map_single_chamber.py`, but we want to make deterministic maps for our test suite. Going from most simple to a little less simple, we can introduce vertical deviations in our tiles or exit switches and exit doors. You can also see the logic in our map single chamber for this with our GLOBAL_MAX_UP_DEVIATION and GLOBAL_MAX_DOWN_DEVIATION variables. Less simple, but still 'simple' levels can include levels that require a jump such as our JUMP_REQUIRED type. Medium levels can have mines, or may require exploration and navigation across a small space. Our MAZE type is a good example of a medium level.
We want to make a script that makes a deterministic set of seed maps to satisfy our test suite composition. We want to ensure that we have a wide variety of maps to test our agent on. If you have any questions while building this suite using the tools provided, please ask before continuing.


**Level selection criteria**:
```python
class TestSuiteBuilder:
    def __init__(self):
        self.complexity_metrics = {
            'switch_count': lambda level: len(level.switches),
            'dependency_depth': lambda level: self.calculate_dependency_depth(level),
            'mine_density': lambda level: len(level.mines) / level.area,
            'exploration_requirement': lambda level: self.calculate_exploration_score(level)
        }
    
    def categorize_level(self, level):
        metrics = {name: func(level) for name, func in self.complexity_metrics.items()}
        
        if metrics['switch_count'] == 1:
            return 'simple'
        elif metrics['dependency_depth'] <= 2 and metrics['switch_count'] <= 3:
            return 'medium'
        elif metrics['mine_density'] > 0.1:
            return 'mine_heavy'
        elif metrics['exploration_requirement'] > 0.7:
            return 'exploration'
        else:
            return 'complex'
```

#### Performance Metrics
**Primary metrics**:
- **Success rate**: Percentage of levels completed successfully
- **Efficiency**: Average steps to completion (compared to optimal)
- **Safety**: Percentage of deaths due to avoidable hazards
- **Robustness**: Performance consistency across level types

**Secondary metrics**:
- **Exploration efficiency**: Coverage of reachable areas
- **Subtask coordination**: Effectiveness of hierarchical decisions
- **Mine avoidance**: Success rate in mine-heavy levels
- **Discovery rate**: Speed of finding hidden switches

**Detailed metrics implementation**:
```python
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
    
    def calculate_success_rate(self, results):
        return sum(r['completed'] for r in results) / len(results)
    
    def calculate_efficiency(self, results):
        completed_results = [r for r in results if r['completed']]
        if not completed_results:
            return 0.0
        
        efficiency_scores = []
        for result in completed_results:
            optimal_steps = result['level_metadata']['optimal_steps']
            actual_steps = result['steps_taken']
            efficiency = optimal_steps / actual_steps
            efficiency_scores.append(min(efficiency, 1.0))  # Cap at 1.0
        
        return np.mean(efficiency_scores)
    
    def calculate_safety_score(self, results):
        total_deaths = sum(r['death_count'] for r in results)
        avoidable_deaths = sum(r['avoidable_deaths'] for r in results)
        
        if total_deaths == 0:
            return 1.0
        
        return 1.0 - (avoidable_deaths / total_deaths)
    
    def calculate_robustness(self, results_by_category):
        success_rates = [
            self.calculate_success_rate(results) 
            for results in results_by_category.values()
        ]
        
        # Robustness is inverse of variance in success rates
        return 1.0 / (1.0 + np.var(success_rates))
```

#### Analysis Tools
**Performance analysis**:
- Success rate trends over training
- Performance breakdown by level category
- Failure mode analysis
- Hierarchical decision analysis

**Visualization tools**:
- Performance dashboards
- Training progress plots
- Heatmaps of level difficulty vs performance
- Subtask transition analysis

**Implementation**:
```python
class PerformanceAnalyzer:
    def __init__(self):
        self.analysis_tools = {
            'success_trends': self.analyze_success_trends,
            'failure_modes': self.analyze_failure_modes,
            'hierarchical_decisions': self.analyze_hierarchical_decisions,
            'exploration_patterns': self.analyze_exploration_patterns
        }
    
    def generate_comprehensive_report(self, evaluation_results):
        report = {
            'summary': self.generate_summary(evaluation_results),
            'detailed_analysis': {},
            'visualizations': {},
            'recommendations': []
        }
        
        for analysis_name, analysis_func in self.analysis_tools.items():
            report['detailed_analysis'][analysis_name] = analysis_func(evaluation_results)
        
        report['recommendations'] = self.generate_recommendations(report)
        
        return report
```

#### Benchmarking Framework
**Comparison baselines**:
- Random policy baseline
- Simple heuristic policy (always move toward nearest objective)
- Phase 1 completion-focused agent
- Phase 2 hierarchical agent

**Benchmarking process**:
1. Run all baselines on standardized test suite
2. Compare current agent performance against baselines
3. Track performance improvements over development phases
4. Identify areas where agent underperforms expectations

**Acceptance criteria**:
- [ ] Comprehensive test suite covering all level complexity types
- [ ] Performance metrics capture all relevant aspects of agent capability
- [ ] Analysis tools provide actionable insights
- [ ] Benchmarking framework enables objective performance comparison
- [ ] Evaluation framework is automated and reproducible
- [ ] Performance targets: >70% success rate across all categories

**Testing requirements**:
- Validation of test suite representativeness
- Metric calculation accuracy tests
- Analysis tool correctness verification
- Benchmarking framework reliability tests

---

### Task 3.4: Hardware Optimization

**What we want to do**: Optimize training for H100 GPU hardware with mixed-precision training, efficient memory usage, and distributed training capabilities.

**Current state**:
- Standard PyTorch training without hardware-specific optimizations
- No mixed-precision training
- Basic vectorized environments but no advanced distributed training
- Suboptimal GPU utilization

**Files to modify**:
- `npp_rl/agents/training.py`
- `npp_rl/agents/hierarchical_ppo.py`
- `ppo_train.py`
- `npp_rl/utils/hardware_optimization.py` (new)
- `npp_rl/distributed/distributed_training.py` (new)

**Hardware optimization requirements**:

#### Mixed-Precision Training
**Implementation using PyTorch AMP**:
```python
import torch
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def training_step(self, batch):
        with autocast():
            # Forward pass in mixed precision
            outputs = self.model(batch)
            loss = self.calculate_loss(outputs, batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

**Benefits**:
- ~2x training speed improvement on H100
- Reduced memory usage allowing larger batch sizes
- Maintained numerical stability with gradient scaling

#### Memory Optimization
**Gradient checkpointing**:
```python
class MemoryOptimizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        # Use gradient checkpointing for memory efficiency
        return torch.utils.checkpoint.checkpoint(self.base_model, x)
```

**Efficient data loading**:
- Asynchronous data loading with multiple workers
- Pinned memory for faster GPU transfers
- Optimized batch construction

**Memory profiling and optimization**:
```python
class MemoryProfiler:
    def __init__(self):
        self.memory_stats = []
    
    def profile_training_step(self, training_function):
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        result = training_function()
        
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        self.memory_stats.append({
            'initial': initial_memory,
            'peak': peak_memory,
            'final': final_memory,
            'step_usage': peak_memory - initial_memory
        })
        
        return result
```

#### Distributed Training Enhancement
**Multi-GPU training**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class DistributedHierarchicalTrainer:
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Initialize distributed training
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Wrap model for distributed training
        self.model = DistributedDataParallel(
            model.cuda(rank), 
            device_ids=[rank]
        )
    
    def train_step(self, batch):
        # Distributed training step
        outputs = self.model(batch)
        loss = self.calculate_loss(outputs, batch)
        
        # Gradient synchronization handled automatically
        loss.backward()
        
        return loss.item()
```

**Environment parallelization optimization**:
```python
class OptimizedVectorEnv:
    def __init__(self, env_fns, num_workers=None):
        if num_workers is None:
            num_workers = min(len(env_fns), torch.cuda.device_count() * 4)
        
        self.envs = SubprocVecEnv(env_fns, start_method='spawn')
        self.num_workers = num_workers
    
    def reset(self):
        return self.envs.reset()
    
    def step(self, actions):
        return self.envs.step(actions)
```

#### Performance Monitoring
**GPU utilization tracking**:
```python
class GPUMonitor:
    def __init__(self):
        self.utilization_history = []
        self.memory_history = []
    
    def log_gpu_stats(self):
        if torch.cuda.is_available():
            utilization = torch.cuda.utilization()
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.max_memory_allocated()
            
            self.utilization_history.append(utilization)
            self.memory_history.append(memory_used / memory_total)
    
    def get_average_utilization(self):
        return np.mean(self.utilization_history) if self.utilization_history else 0
```

**Training throughput optimization**:
- Batch size optimization for maximum throughput
- Learning rate scaling for larger batch sizes
- Gradient accumulation for effective large batch training

**Acceptance criteria**:
- [ ] Mixed-precision training implemented and stable
- [ ] GPU utilization >85% during training
- [ ] Memory usage optimized for maximum batch sizes
- [ ] Distributed training scales effectively across multiple GPUs
- [ ] Training throughput improved by >50% vs baseline
- [ ] Numerical stability maintained with optimizations

**Testing requirements**:
- Performance benchmarks with and without optimizations
- Stability testing with mixed-precision training
- Memory usage profiling and optimization validation
- Distributed training correctness verification

## Dependencies and Prerequisites

**Phase 2 completion requirements**:
- Hierarchical architecture working and stable
- Subtask-specific rewards implemented
- Mine avoidance integrated
- Multi-switch level completion >60%

**Hardware requirements**:
- H100 or equivalent GPU for optimization testing
- Multi-GPU setup for distributed training testing
- Sufficient memory for large batch size testing

## Risk Mitigation

**Architecture optimization risks**:
- **Risk**: Performance degradation with simplified architecture
- **Mitigation**: Systematic comparison with fallback to more complex architecture
- **Testing**: Comprehensive benchmarking before architecture changes

**Hardware optimization risks**:
- **Risk**: Training instability with mixed-precision or distributed training
- **Mitigation**: Gradual introduction of optimizations with stability monitoring
- **Fallback**: Disable optimizations if stability issues arise

**Evaluation framework risks**:
- **Risk**: Test suite not representative of real performance
- **Mitigation**: Diverse level selection and validation against human performance
- **Testing**: Cross-validation of evaluation metrics

## Success Criteria for Phase 3

**Primary objectives**:
- [ ] Optimized architecture maintains performance with improved efficiency
- [ ] Advanced ICM integration improves exploration and completion rates
- [ ] Comprehensive evaluation framework provides reliable performance measurement
- [ ] Hardware optimizations achieve >50% training speedup
- [ ] Robust performance: >70% success rate across all level categories

**Performance targets**:
- [ ] Inference time: <10ms per step
- [ ] Training throughput: >50% improvement vs Phase 2
- [ ] GPU utilization: >85% during training
- [ ] Memory efficiency: Support larger batch sizes
- [ ] Generalization: Consistent performance across level types

**Quality gates**:
- [ ] All optimization components tested and validated
- [ ] Performance benchmarks established and met
- [ ] Evaluation framework comprehensive and reliable
- [ ] Code review and documentation completed
- [ ] System ready for production deployment

This phase transforms the hierarchical system into a robust, efficient, and well-evaluated agent ready for advanced features or production deployment.