# TASK 002: Implement Reachability-Aware Curiosity Module

## Overview
Enhance the existing intrinsic motivation system with reachability-aware curiosity that avoids wasting exploration on unreachable areas, improving sample efficiency and training performance.

## Context & Justification

### Current Intrinsic Motivation System
Based on analysis of `/workspace/npp-rl/npp_rl/intrinsic/` and README.md:
- **ICM (Intrinsic Curiosity Module)**: Forward/inverse model prediction errors
- **Novelty Detection**: Count-based exploration with state discretization
- **Adaptive Scaling**: Dynamic adjustment based on training progress
- **Integration**: Combined with extrinsic rewards for exploration guidance

### Problem with Current Approach
- **Wasted Exploration**: Agent explores unreachable areas, reducing sample efficiency
- **Uniform Curiosity**: Equal curiosity for all novel states regardless of reachability
- **No Strategic Guidance**: Curiosity doesn't consider level completion objectives
- **Inefficient Learning**: Time spent on impossible areas delays meaningful progress

### Research Foundation
From `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Reachability-Guided Exploration**: Focus curiosity on reachable but unexplored areas
- **Frontier Exploration**: Prioritize areas that become reachable after switch activation
- **Strategic Curiosity**: Weight exploration by proximity to level objectives
- **Efficiency Gains**: 2-3x improvement in sample efficiency on complex levels

### Theoretical Justification
- **Pathak et al. (2017)**: ICM works best when prediction errors correlate with meaningful exploration
- **Ecoffet et al. (2019)**: Go-Explore shows benefits of avoiding impossible states
- **Burda et al. (2018)**: Random Network Distillation benefits from reachability constraints

## Technical Specification

### Enhanced Curiosity Architecture
```python
class ReachabilityAwareCuriosity(nn.Module):
    """
    Enhanced curiosity module that considers reachability constraints.
    
    Key Components:
    1. Base Curiosity: ICM + Novelty detection (existing)
    2. Reachability Scaling: Modulate curiosity based on reachability
    3. Frontier Detection: Boost curiosity for newly reachable areas
    4. Strategic Weighting: Prioritize exploration near objectives
    """
    
    def __init__(self, base_curiosity_module, reachability_extractor, 
                 observation_space, action_space):
        super().__init__()
        
        # Base curiosity components
        self.base_curiosity = base_curiosity_module
        self.reachability_extractor = reachability_extractor
        
        # Reachability-aware components
        self.reachability_predictor = ReachabilityPredictor(
            observation_space, hidden_dim=128
        )
        self.frontier_detector = FrontierDetector(
            reachability_dim=64, memory_size=1000
        )
        self.strategic_weighter = StrategicWeighter(
            reachability_dim=64, objective_dim=32
        )
        
        # Scaling parameters
        self.reachability_scale_factor = 2.0
        self.frontier_boost_factor = 3.0
        self.strategic_weight_factor = 1.5
        self.unreachable_penalty = 0.1
        
        # Exploration history
        self.exploration_history = ExplorationHistory(max_size=10000)
        self.last_reachable_positions = set()
    
    def compute_intrinsic_reward(self, obs, action, next_obs, info=None):
        """
        Compute reachability-aware intrinsic reward.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            info: Environment info (contains reachability data)
        
        Returns:
            Enhanced intrinsic reward considering reachability
        """
        # Get base curiosity reward
        base_reward = self.base_curiosity.compute_intrinsic_reward(obs, action, next_obs)
        
        # Extract reachability information
        reachability_info = self._extract_reachability_info(obs, next_obs, info)
        
        # Compute reachability scaling
        reachability_scale = self._compute_reachability_scale(reachability_info)
        
        # Detect frontier exploration
        frontier_boost = self._compute_frontier_boost(reachability_info)
        
        # Compute strategic weighting
        strategic_weight = self._compute_strategic_weight(reachability_info)
        
        # Combine all factors
        enhanced_reward = (base_reward * reachability_scale * 
                          frontier_boost * strategic_weight)
        
        # Update exploration history
        self._update_exploration_history(reachability_info, enhanced_reward)
        
        return enhanced_reward
    
    def _compute_reachability_scale(self, reachability_info):
        """
        Scale curiosity based on reachability status.
        
        Scaling Strategy:
        - Reachable areas: 1.0x (full curiosity)
        - Frontier areas: 0.5x (moderate curiosity)
        - Unreachable areas: 0.1x (minimal curiosity)
        """
        current_pos = reachability_info['current_position']
        target_pos = reachability_info['target_position']
        reachable_positions = reachability_info['reachable_positions']
        
        # Check if target position is reachable
        if target_pos in reachable_positions:
            return 1.0  # Full curiosity for reachable areas
        
        # Check if target is on exploration frontier
        if self._is_on_exploration_frontier(target_pos, reachable_positions):
            return 0.5  # Moderate curiosity for frontier
        
        # Check if target has been confirmed unreachable
        if self._is_confirmed_unreachable(target_pos):
            return self.unreachable_penalty  # Minimal curiosity
        
        # Default: unknown area gets moderate curiosity
        return 0.3
    
    def _compute_frontier_boost(self, reachability_info):
        """
        Boost curiosity for newly reachable areas (frontiers).
        """
        current_reachable = set(reachability_info['reachable_positions'])
        
        # Detect newly reachable areas
        newly_reachable = current_reachable - self.last_reachable_positions
        
        if newly_reachable:
            # Update frontier detector
            self.frontier_detector.update_frontiers(newly_reachable)
            
            # Check if current exploration is in frontier area
            target_pos = reachability_info['target_position']
            if self.frontier_detector.is_in_frontier(target_pos):
                return self.frontier_boost_factor
        
        # Update last reachable positions
        self.last_reachable_positions = current_reachable
        
        return 1.0  # No boost
    
    def _compute_strategic_weight(self, reachability_info):
        """
        Weight curiosity based on strategic value (proximity to objectives).
        """
        target_pos = reachability_info['target_position']
        objective_distances = reachability_info.get('objective_distances', [])
        
        if not objective_distances:
            return 1.0  # No strategic information available
        
        # Find closest objective distance
        min_objective_distance = min(objective_distances)
        
        # Strategic weight: higher for areas closer to objectives
        # Use exponential decay: weight = exp(-distance / scale)
        distance_scale = 100.0  # pixels
        strategic_weight = math.exp(-min_objective_distance / distance_scale)
        
        # Scale by strategic weight factor
        return 1.0 + (strategic_weight * (self.strategic_weight_factor - 1.0))
```

### Supporting Components

#### 1. Reachability Predictor
```python
class ReachabilityPredictor(nn.Module):
    """
    Predict reachability of positions based on observations.
    
    This helps the curiosity module make reachability assessments
    even when detailed reachability analysis is not available.
    """
    
    def __init__(self, observation_space, hidden_dim=128):
        super().__init__()
        
        # Extract observation dimensions
        self.obs_dim = self._get_observation_dim(observation_space)
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(self.obs_dim + 2, hidden_dim),  # +2 for target position
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Binary reachability prediction
            nn.Sigmoid()
        )
        
        # Training components
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def predict_reachability(self, obs, target_pos):
        """
        Predict if target position is reachable from current observation.
        
        Returns:
            Probability that target position is reachable (0-1)
        """
        # Flatten observation
        obs_flat = self._flatten_observation(obs)
        
        # Normalize target position
        target_normalized = self._normalize_position(target_pos)
        
        # Combine inputs
        input_tensor = torch.cat([obs_flat, target_normalized], dim=-1)
        
        # Predict reachability
        reachability_prob = self.predictor(input_tensor)
        
        return reachability_prob.squeeze(-1)
    
    def update(self, obs, target_pos, is_reachable):
        """
        Update predictor based on ground truth reachability.
        """
        # Get prediction
        pred_reachability = self.predict_reachability(obs, target_pos)
        
        # Convert ground truth to tensor
        target_tensor = torch.tensor(is_reachable, dtype=torch.float32)
        
        # Compute loss
        loss = self.criterion(pred_reachability, target_tensor)
        
        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

#### 2. Frontier Detector
```python
class FrontierDetector:
    """
    Detect and track exploration frontiers (newly reachable areas).
    """
    
    def __init__(self, reachability_dim=64, memory_size=1000):
        self.reachability_dim = reachability_dim
        self.memory_size = memory_size
        
        # Frontier tracking
        self.current_frontiers = set()
        self.frontier_history = deque(maxlen=memory_size)
        self.frontier_decay_time = 500  # frames
        
        # Frontier analysis
        self.frontier_analyzer = FrontierAnalyzer()
    
    def update_frontiers(self, newly_reachable_positions):
        """
        Update frontier areas based on newly reachable positions.
        """
        # Add new frontiers
        for pos in newly_reachable_positions:
            frontier_info = {
                'position': pos,
                'discovered_time': time.time(),
                'exploration_count': 0
            }
            self.current_frontiers.add(pos)
            self.frontier_history.append(frontier_info)
        
        # Decay old frontiers
        self._decay_old_frontiers()
    
    def is_in_frontier(self, position):
        """
        Check if position is in current frontier areas.
        """
        return position in self.current_frontiers
    
    def get_frontier_value(self, position):
        """
        Get exploration value for frontier position.
        """
        if position not in self.current_frontiers:
            return 0.0
        
        # Find frontier info
        for frontier_info in reversed(self.frontier_history):
            if frontier_info['position'] == position:
                # Value decreases with exploration count and age
                age_factor = self._compute_age_factor(frontier_info['discovered_time'])
                exploration_factor = 1.0 / (1.0 + frontier_info['exploration_count'])
                
                return age_factor * exploration_factor
        
        return 0.5  # Default value
    
    def _decay_old_frontiers(self):
        """
        Remove old frontiers that are no longer novel.
        """
        current_time = time.time()
        expired_frontiers = set()
        
        for pos in self.current_frontiers:
            # Find frontier info
            for frontier_info in reversed(self.frontier_history):
                if frontier_info['position'] == pos:
                    age = current_time - frontier_info['discovered_time']
                    if age > self.frontier_decay_time or frontier_info['exploration_count'] > 10:
                        expired_frontiers.add(pos)
                    break
        
        # Remove expired frontiers
        self.current_frontiers -= expired_frontiers
```

#### 3. Strategic Weighter
```python
class StrategicWeighter:
    """
    Weight exploration based on strategic value for level completion.
    """
    
    def __init__(self, reachability_dim=64, objective_dim=32):
        self.reachability_dim = reachability_dim
        self.objective_dim = objective_dim
        
        # Strategic analysis
        self.objective_analyzer = ObjectiveAnalyzer()
        self.path_analyzer = PathAnalyzer()
    
    def compute_strategic_weight(self, position, reachability_features, level_objectives):
        """
        Compute strategic weight for exploring a position.
        
        Factors:
        1. Distance to key objectives (switches, doors, exit)
        2. Position on critical paths
        3. Potential to unlock new areas
        4. Historical exploration success
        """
        weights = []
        
        # Objective proximity weight
        objective_weight = self._compute_objective_proximity_weight(
            position, level_objectives
        )
        weights.append(objective_weight)
        
        # Critical path weight
        path_weight = self._compute_critical_path_weight(
            position, reachability_features
        )
        weights.append(path_weight)
        
        # Unlock potential weight
        unlock_weight = self._compute_unlock_potential_weight(
            position, reachability_features
        )
        weights.append(unlock_weight)
        
        # Combine weights (geometric mean for balanced influence)
        combined_weight = np.prod(weights) ** (1.0 / len(weights))
        
        return combined_weight
    
    def _compute_objective_proximity_weight(self, position, objectives):
        """
        Weight based on proximity to level objectives.
        """
        if not objectives:
            return 1.0
        
        # Find closest objective
        min_distance = float('inf')
        for objective in objectives:
            distance = self._compute_distance(position, objective['position'])
            min_distance = min(min_distance, distance)
        
        # Convert distance to weight (closer = higher weight)
        max_distance = 500.0  # Maximum meaningful distance
        proximity_weight = 1.0 + (max_distance - min_distance) / max_distance
        
        return max(proximity_weight, 0.1)  # Minimum weight
    
    def _compute_critical_path_weight(self, position, reachability_features):
        """
        Weight based on position being on critical paths.
        """
        # Extract path-related features from reachability features
        # This is a simplified heuristic - could be enhanced with graph analysis
        
        # Use area connectivity features (indices 40-47)
        area_connectivity = reachability_features[40:48]
        
        # Positions in well-connected areas get higher weight
        connectivity_score = np.mean(area_connectivity)
        
        # Convert to weight
        path_weight = 0.5 + connectivity_score
        
        return path_weight
    
    def _compute_unlock_potential_weight(self, position, reachability_features):
        """
        Weight based on potential to unlock new areas.
        """
        # Extract switch-related features (indices 8-23)
        switch_features = reachability_features[8:24]
        
        # Positions near inactive but reachable switches get higher weight
        inactive_reachable_switches = np.sum((switch_features > 0.1) & (switch_features < 0.9))
        
        # Convert to weight
        unlock_weight = 1.0 + 0.5 * inactive_reachable_switches
        
        return unlock_weight
```

## Implementation Plan

### Phase 1: Core Curiosity Enhancement (Week 1)
**Deliverables**:
1. **ReachabilityAwareCuriosity**: Main enhanced curiosity module
2. **Integration Interface**: Clean integration with existing ICM
3. **Basic Testing**: Unit tests for core functionality

**Key Files**:
- `npp_rl/intrinsic/reachability_aware_curiosity.py` (NEW)
- `npp_rl/intrinsic/reachability_predictor.py` (NEW)
- `tests/test_reachability_curiosity.py` (NEW)

### Phase 2: Supporting Components (Week 2)
**Deliverables**:
1. **FrontierDetector**: Track newly reachable areas
2. **StrategicWeighter**: Weight exploration by strategic value
3. **ExplorationHistory**: Track exploration patterns

**Key Files**:
- `npp_rl/intrinsic/frontier_detector.py` (NEW)
- `npp_rl/intrinsic/strategic_weighter.py` (NEW)
- `npp_rl/intrinsic/exploration_history.py` (NEW)

### Phase 3: Training Integration (Week 3)
**Deliverables**:
1. **Enhanced Training Pipeline**: Integrate reachability-aware curiosity
2. **Hyperparameter Tuning**: Optimize curiosity scaling factors
3. **Performance Monitoring**: Track exploration efficiency metrics

**Implementation**:
```python
class ReachabilityAwareTrainingManager:
    """
    Enhanced training manager with reachability-aware curiosity.
    """
    
    def __init__(self, env, model_config, curiosity_config):
        self.env = env
        self.model_config = model_config
        self.curiosity_config = curiosity_config
        
        # Initialize enhanced curiosity
        self.curiosity_module = self._create_curiosity_module()
        
        # Training metrics
        self.exploration_metrics = ExplorationMetrics()
        self.reachability_metrics = ReachabilityMetrics()
    
    def _create_curiosity_module(self):
        """Create reachability-aware curiosity module."""
        # Base curiosity (existing ICM + novelty)
        base_curiosity = self._create_base_curiosity()
        
        # Reachability extractor
        reachability_extractor = self._create_reachability_extractor()
        
        # Enhanced curiosity
        return ReachabilityAwareCuriosity(
            base_curiosity_module=base_curiosity,
            reachability_extractor=reachability_extractor,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **self.curiosity_config
        )
    
    def train_step(self, obs, action, next_obs, reward, done, info):
        """
        Enhanced training step with reachability-aware curiosity.
        """
        # Compute intrinsic reward
        intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
            obs, action, next_obs, info
        )
        
        # Combine with extrinsic reward
        total_reward = reward + intrinsic_reward
        
        # Update metrics
        self.exploration_metrics.update(obs, action, intrinsic_reward)
        self.reachability_metrics.update(info.get('reachability', {}))
        
        return total_reward, intrinsic_reward
    
    def get_training_metrics(self):
        """Get comprehensive training metrics."""
        return {
            'exploration': self.exploration_metrics.get_summary(),
            'reachability': self.reachability_metrics.get_summary(),
            'curiosity': self.curiosity_module.get_metrics()
        }
```

### Phase 4: Evaluation and Optimization (Week 4)
**Deliverables**:
1. **Performance Evaluation**: Compare with baseline curiosity
2. **Ablation Studies**: Analyze individual component contributions
3. **Optimization**: Performance tuning and memory optimization

## Testing Strategy

### Unit Tests
```python
class TestReachabilityAwareCuriosity(unittest.TestCase):
    def setUp(self):
        self.base_curiosity = MockICMModule()
        self.reachability_extractor = MockReachabilityExtractor()
        self.curiosity = ReachabilityAwareCuriosity(
            self.base_curiosity, self.reachability_extractor,
            observation_space=create_mock_obs_space(),
            action_space=create_mock_action_space()
        )
    
    def test_reachability_scaling(self):
        """Test that curiosity is scaled based on reachability."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # Mock reachable target
        info_reachable = {'reachability': {'target_reachable': True}}
        reward_reachable = self.curiosity.compute_intrinsic_reward(
            obs, action, next_obs, info_reachable
        )
        
        # Mock unreachable target
        info_unreachable = {'reachability': {'target_reachable': False}}
        reward_unreachable = self.curiosity.compute_intrinsic_reward(
            obs, action, next_obs, info_unreachable
        )
        
        # Reachable should have higher curiosity
        self.assertGreater(reward_reachable, reward_unreachable)
    
    def test_frontier_boost(self):
        """Test that newly reachable areas get curiosity boost."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # First call - establish baseline
        info1 = {'reachability': {'reachable_positions': {(10, 10), (10, 11)}}}
        reward1 = self.curiosity.compute_intrinsic_reward(obs, action, next_obs, info1)
        
        # Second call - new area becomes reachable
        info2 = {'reachability': {'reachable_positions': {(10, 10), (10, 11), (12, 12)}}}
        reward2 = self.curiosity.compute_intrinsic_reward(obs, action, next_obs, info2)
        
        # Should detect frontier and boost curiosity
        self.assertGreater(reward2, reward1)
    
    def test_strategic_weighting(self):
        """Test that exploration near objectives gets higher weight."""
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        
        # Near objective
        info_near = {
            'reachability': {
                'objective_distances': [10.0, 50.0, 100.0],  # Close to first objective
                'target_position': (10, 10)
            }
        }
        reward_near = self.curiosity.compute_intrinsic_reward(obs, action, next_obs, info_near)
        
        # Far from objectives
        info_far = {
            'reachability': {
                'objective_distances': [200.0, 250.0, 300.0],  # Far from all objectives
                'target_position': (50, 50)
            }
        }
        reward_far = self.curiosity.compute_intrinsic_reward(obs, action, next_obs, info_far)
        
        # Near objective should have higher curiosity
        self.assertGreater(reward_near, reward_far)
```

### Integration Tests
```python
class TestCuriosityTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.env = ReachabilityEnhancedNPPEnv(render_mode='rgb_array')
        self.training_manager = ReachabilityAwareTrainingManager(
            env=self.env,
            model_config={'features_dim': 512},
            curiosity_config={'reachability_scale_factor': 2.0}
        )
    
    def test_training_step_integration(self):
        """Test that training step works with enhanced curiosity."""
        obs = self.env.reset()
        
        for _ in range(100):
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            
            # Enhanced training step
            total_reward, intrinsic_reward = self.training_manager.train_step(
                obs, action, next_obs, reward, done, info
            )
            
            # Verify rewards are reasonable
            self.assertIsInstance(total_reward, (int, float))
            self.assertIsInstance(intrinsic_reward, (int, float))
            self.assertGreaterEqual(intrinsic_reward, 0.0)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
    
    def test_exploration_efficiency(self):
        """Test that reachability-aware curiosity improves exploration efficiency."""
        # This would be a longer test comparing exploration patterns
        # with and without reachability awareness
        pass
```

### Performance Tests
```python
class TestCuriosityPerformance(unittest.TestCase):
    def test_curiosity_computation_speed(self):
        """Test that curiosity computation meets performance requirements."""
        curiosity = create_test_curiosity_module()
        
        obs = create_mock_observation()
        action = 0
        next_obs = create_mock_observation()
        info = create_mock_reachability_info()
        
        # Benchmark curiosity computation
        times = []
        for _ in range(1000):
            start_time = time.perf_counter()
            reward = curiosity.compute_intrinsic_reward(obs, action, next_obs, info)
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)
        
        # Check performance targets
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        self.assertLess(avg_time, 1.0, f"Average curiosity time too high: {avg_time}ms")
        self.assertLess(p95_time, 2.0, f"95th percentile time too high: {p95_time}ms")
    
    def test_memory_usage(self):
        """Test that curiosity module doesn't use excessive memory."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create curiosity module
        curiosity = create_test_curiosity_module()
        
        # Run for extended period
        for _ in range(10000):
            obs = create_mock_observation()
            action = 0
            next_obs = create_mock_observation()
            info = create_mock_reachability_info()
            
            reward = curiosity.compute_intrinsic_reward(obs, action, next_obs, info)
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertLess(memory_increase, 100, f"Memory usage too high: {memory_increase}MB")
```

## Success Criteria

### Performance Requirements
- **Curiosity Computation**: <1ms average per step
- **Memory Usage**: <50MB additional memory for curiosity components
- **Training Speed**: <5% slowdown compared to baseline curiosity

### Quality Requirements
- **Exploration Efficiency**: 20-50% improvement in sample efficiency on complex levels
- **Reachability Awareness**: Demonstrable reduction in exploration of unreachable areas
- **Strategic Focus**: Increased exploration near level objectives

### Training Requirements
- **Convergence**: Faster convergence on complex levels
- **Generalization**: Better performance on unseen levels
- **Stability**: Stable training without curiosity-induced instabilities

## Risk Mitigation

### Technical Risks
1. **Complexity Overhead**: Careful performance monitoring and optimization
2. **Hyperparameter Sensitivity**: Extensive hyperparameter tuning and validation
3. **Integration Issues**: Comprehensive testing with existing systems

### Training Risks
1. **Exploration Bias**: Balance reachability awareness with exploration diversity
2. **Overfitting**: Regularization and validation on diverse levels
3. **Reward Scaling**: Careful tuning of intrinsic vs extrinsic reward balance

## Deliverables

1. **ReachabilityAwareCuriosity**: Enhanced curiosity module with reachability awareness
2. **Supporting Components**: Frontier detection, strategic weighting, exploration history
3. **Training Integration**: Complete integration with PPO training pipeline
4. **Performance Analysis**: Comprehensive evaluation of exploration efficiency improvements
5. **Documentation**: Complete API documentation and usage guide

## Timeline

- **Week 1**: Core curiosity enhancement and basic integration
- **Week 2**: Supporting components and advanced features
- **Week 3**: Training integration and hyperparameter tuning
- **Week 4**: Evaluation, optimization, and documentation

## Dependencies

### Internal Dependencies
- **Reachability Features**: TASK_001 (Compact Reachability Features Integration)
- **Base Curiosity**: Existing ICM and novelty detection modules
- **Training Pipeline**: Enhanced training system with reachability support

### External Dependencies
- **PyTorch**: Neural network implementation
- **NumPy**: Numerical operations
- **Collections**: Data structures for history tracking

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **Current ICM**: `/workspace/npp-rl/npp_rl/intrinsic/` (existing curiosity modules)
3. **Pathak et al. (2017)**: "Curiosity-driven Exploration by Self-supervised Prediction"
4. **Ecoffet et al. (2019)**: "Go-Explore: a New Approach for Hard-Exploration Problems"
5. **Burda et al. (2018)**: "Exploration by Random Network Distillation"