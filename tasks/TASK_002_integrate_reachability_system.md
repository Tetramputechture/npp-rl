# Task 002: Integrate Reachability System with RL Architecture

## Overview
Integrate the enhanced reachability analysis system from nclone with the RL architecture in npp-rl. This includes creating the hierarchical reachability manager, reachability-aware curiosity module, and level completion planner for strategic guidance.

## Context Reference
See [npp-rl comprehensive technical roadmap](../docs/comprehensive_technical_roadmap.md) Section 1.3: "Integration with RL Architecture" and Section 1.2: "Physics-Aware Reachability Analysis Strategy"

## Requirements

### Primary Objectives
1. **Create Hierarchical Reachability Manager** for HRL subgoal selection
2. **Implement Reachability-Aware Curiosity** for efficient exploration
3. **Build Level Completion Planner** for strategic guidance
4. **Integrate with existing HGT architecture** seamlessly
5. **Optimize for real-time RL training** (60 FPS decision making)

### Integration Architecture
The reachability system will integrate with existing components:
- **HGT Multimodal Extractor**: Enhanced with reachability features
- **Intrinsic Motivation (ICM)**: Extended with reachability-aware curiosity
- **Environment Wrappers**: Enhanced with reachability information
- **Training Pipeline**: Modified to use reachability-guided exploration

### Components to Implement

#### 1. Hierarchical Reachability Manager
**New File**: `npp_rl/utils/reachability_manager.py`

**Core Functionality**:
```python
class HierarchicalReachabilityManager:
    def __init__(self, reachability_analyzer):
        self.analyzer = reachability_analyzer  # From nclone
        self.cached_reachability = {}
        self.cache_ttl = {}  # Time-to-live for cache entries
        self.update_threshold = 5  # frames between updates
        self.last_ninja_pos = None
        self.last_switch_states = None
        
    def get_reachable_subgoals(self, ninja_pos: Tuple[float, float], 
                              level_data, 
                              switch_states: Dict[int, bool]) -> List[str]:
        """
        Core integration point for HRL subgoal selection.
        
        Returns list of currently reachable subgoals:
        - 'navigate_to_exit_switch'
        - 'navigate_to_exit_door' 
        - 'activate_door_switch_X'
        - 'avoid_hazard_X_Y'
        """
        # Check cache first
        cache_key = self._generate_cache_key(ninja_pos, switch_states)
        if self._is_cache_valid(cache_key):
            return self.cached_reachability[cache_key]['subgoals']
        
        # Perform reachability analysis
        reachability_state = self.analyzer.analyze_reachability(
            level_data, ninja_pos, switch_states
        )
        
        # Extract reachable subgoals
        subgoals = self._extract_subgoals(reachability_state, level_data, switch_states)
        
        # Cache results
        self._cache_results(cache_key, {
            'subgoals': subgoals,
            'reachability_state': reachability_state,
            'timestamp': time.time()
        })
        
        return subgoals
    
    def _extract_subgoals(self, reachability_state, level_data, switch_states) -> List[str]:
        """Extract actionable subgoals from reachability analysis."""
        subgoals = []
        
        # Check exit switch reachability
        exit_switch_pos = self._find_exit_switch_position(level_data)
        if exit_switch_pos and self._is_position_reachable(exit_switch_pos, reachability_state):
            subgoals.append('navigate_to_exit_switch')
        
        # Check exit door reachability (only if switch activated)
        if switch_states.get('exit_switch', False):
            exit_door_pos = self._find_exit_door_position(level_data)
            if exit_door_pos and self._is_position_reachable(exit_door_pos, reachability_state):
                subgoals.append('navigate_to_exit_door')
        
        # Check door switches
        for door_id, door_switch_pos in self._find_door_switches(level_data):
            if not switch_states.get(f'door_{door_id}', False):
                if self._is_position_reachable(door_switch_pos, reachability_state):
                    subgoals.append(f'activate_door_switch_{door_id}')
        
        return subgoals
```

#### 2. Reachability-Aware Curiosity Module
**New File**: `npp_rl/intrinsic/reachability_curiosity.py`

**Core Functionality**:
```python
class ReachabilityAwareCuriosity(nn.Module):
    def __init__(self, reachability_manager: HierarchicalReachabilityManager, 
                 base_curiosity_module=None):
        super().__init__()
        self.reachability_manager = reachability_manager
        self.base_curiosity = base_curiosity_module or ICMModule()
        self.exploration_frontiers = set()
        self.unreachable_penalties = {}
        
    def compute_exploration_bonus(self, obs: dict, action: int, next_obs: dict) -> float:
        """
        Compute curiosity bonus that considers reachability.
        
        Key insight: Don't waste curiosity on unreachable areas!
        - 1.0x bonus for reachable but unexplored areas
        - 0.5x bonus for frontier areas (might become reachable)
        - 0.0x bonus for confirmed unreachable areas
        """
        # Extract spatial information from observations
        ninja_pos = self._extract_ninja_position(obs)
        target_pos = self._extract_target_position(next_obs, action)
        level_data = self._extract_level_data(obs)
        switch_states = self._extract_switch_states(obs)
        
        # Get base curiosity bonus
        base_bonus = self.base_curiosity.compute_intrinsic_reward(obs, action, next_obs)
        
        # Apply reachability scaling
        reachability_scale = self._compute_reachability_scale(
            ninja_pos, target_pos, level_data, switch_states
        )
        
        return base_bonus * reachability_scale
    
    def _compute_reachability_scale(self, ninja_pos, target_pos, level_data, switch_states) -> float:
        """Compute scaling factor based on reachability."""
        # Get reachable subgoals
        reachable_subgoals = self.reachability_manager.get_reachable_subgoals(
            ninja_pos, level_data, switch_states
        )
        
        # Check if target is in reachable area
        if self._is_target_in_reachable_area(target_pos, reachable_subgoals):
            return 1.0  # Full curiosity for reachable areas
        
        # Check if target is on exploration frontier
        if self._is_target_on_exploration_frontier(target_pos):
            return 0.5  # Reduced curiosity for frontier areas
        
        # Check if target has been confirmed unreachable
        if self._is_target_confirmed_unreachable(target_pos):
            return 0.0  # No curiosity for unreachable areas
        
        # Default to medium curiosity for unknown areas
        return 0.3
    
    def update_exploration_frontiers(self, level_data, switch_states: Dict[int, bool]):
        """
        Update exploration frontiers when switches are activated.
        
        This is crucial: when a switch is activated, previously unreachable
        areas become exploration targets.
        """
        # Get current reachability
        current_reachability = self.reachability_manager.analyzer.analyze_reachability(
            level_data, self.last_ninja_pos, switch_states
        )
        
        # Find newly reachable areas
        if hasattr(self, 'last_reachable_positions'):
            newly_reachable = (current_reachability.reachable_positions - 
                             self.last_reachable_positions)
            
            # Add to exploration frontiers
            self.exploration_frontiers.update(newly_reachable)
            
            # Clear penalties for newly reachable areas
            for pos in newly_reachable:
                if pos in self.unreachable_penalties:
                    del self.unreachable_penalties[pos]
        
        self.last_reachable_positions = current_reachability.reachable_positions
```

#### 3. Level Completion Planner
**New File**: `npp_rl/planning/completion_planner.py`

**Core Functionality**:
```python
class PhysicsAwareLevelCompletionPlanner:
    def __init__(self, reachability_analyzer):
        self.analyzer = reachability_analyzer
        
    def plan_completion_strategy(self, ninja_pos: Tuple[float, float],
                               level_data,
                               switch_states: Dict[int, bool]) -> List[str]:
        """
        Implement the exact level completion heuristic from the roadmap:
        
        1. Is there a possible path from current location to exit switch?
           - If no, find closest locked door switch, trigger it, go to step 1
           - If yes, navigate to exit switch and trigger it
        2. Is there a possible path to exit door (now that switch is triggered)?
           - If no, find closest locked door switch, trigger it, go to step 2  
           - If yes, navigate to exit door and complete level
        """
        # Step 1: Analyze current reachability
        reachability_state = self.analyzer.analyze_reachability(
            level_data, ninja_pos, switch_states
        )
        
        exit_switch_pos = self._find_exit_switch_position(level_data)
        exit_door_pos = self._find_exit_door_position(level_data)
        
        action_plan = []
        
        # Step 2: Check direct path to exit switch
        if self._is_reachable(exit_switch_pos, reachability_state):
            # Direct path available
            if switch_states.get('exit_switch', False):
                # Switch already activated, check door
                if self._is_reachable(exit_door_pos, reachability_state):
                    action_plan.append('navigate_to_exit_door')
                    return action_plan
                else:
                    # Need to unlock path to door
                    door_unlock_sequence = self._find_door_unlock_sequence(
                        exit_door_pos, level_data, switch_states, reachability_state
                    )
                    action_plan.extend(door_unlock_sequence)
                    action_plan.append('navigate_to_exit_door')
                    return action_plan
            else:
                # Activate switch first
                action_plan.extend([
                    'navigate_to_exit_switch',
                    'activate_exit_switch',
                    'navigate_to_exit_door'
                ])
                return action_plan
        
        # Step 3: Find blocking doors and required switches
        blocking_doors = self._find_blocking_doors_to_target(
            ninja_pos, exit_switch_pos, level_data, reachability_state
        )
        
        # Step 4: Plan switch activation sequence
        switch_sequence = self._optimize_switch_sequence(
            ninja_pos, blocking_doors, level_data, reachability_state
        )
        
        # Step 5: Build complete action plan
        for door_id, switch_pos in switch_sequence:
            action_plan.extend([
                f'navigate_to_switch_{door_id}',
                f'activate_switch_{door_id}'
            ])
        
        # Add final exit sequence
        action_plan.extend([
            'navigate_to_exit_switch',
            'activate_exit_switch',
            'navigate_to_exit_door'
        ])
        
        return action_plan
    
    def _find_blocking_doors_to_target(self, start_pos, target_pos, level_data, reachability_state):
        """
        Find doors that block the path to target using graph connectivity analysis.
        
        This uses reachability system's graph structure rather than expensive
        pathfinding to identify bottleneck doors.
        """
        blocking_doors = []
        
        # Find all doors between reachable and unreachable areas
        for door_entity in level_data.get_entities_by_type(EntityType.LOCKED_DOOR):
            door_pos = door_entity.position
            
            # Check if this door is a bottleneck by testing reachability with/without it
            if self._is_door_blocking_path(start_pos, target_pos, door_entity, level_data):
                blocking_doors.append(door_entity)
        
        return blocking_doors
```

#### 4. Enhanced Environment Integration
**Enhanced File**: `npp_rl/environments/npp_env.py`

**Add Reachability Integration**:
```python
class ReachabilityEnhancedNPPEnv(NPPEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize reachability components
        from nclone.graph.reachability_analyzer import ReachabilityAnalyzer
        from nclone.graph.trajectory_calculator import TrajectoryCalculator
        
        self.reachability_manager = HierarchicalReachabilityManager(
            ReachabilityAnalyzer(TrajectoryCalculator())
        )
        self.completion_planner = PhysicsAwareLevelCompletionPlanner(
            self.reachability_manager.analyzer
        )
        
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Add reachability information to observation and info
        ninja_pos = self._get_ninja_position()
        level_data = self._get_level_data()
        switch_states = self._get_switch_states()
        
        # Get reachable subgoals for HRL
        reachable_subgoals = self.reachability_manager.get_reachable_subgoals(
            ninja_pos, level_data, switch_states
        )
        
        # Get strategic plan for guidance
        strategic_plan = self.completion_planner.plan_completion_strategy(
            ninja_pos, level_data, switch_states
        )
        
        # Add to observation
        obs['reachable_subgoals'] = self._encode_subgoals(reachable_subgoals)
        obs['strategic_plan'] = self._encode_plan(strategic_plan)
        
        # Add to info for debugging/logging
        info['reachability'] = {
            'subgoals': reachable_subgoals,
            'strategic_plan': strategic_plan,
            'cache_hit_rate': self.reachability_manager.get_cache_hit_rate()
        }
        
        return obs, reward, done, info
```

## Acceptance Criteria

### Functional Requirements
1. **Seamless Integration**: Reachability system works with existing HGT architecture
2. **Real-time Performance**: Reachability queries complete in <10ms
3. **HRL Support**: Provides filtered subgoals for hierarchical RL
4. **Curiosity Enhancement**: Improves exploration efficiency by avoiding unreachable areas
5. **Strategic Guidance**: Provides actionable level completion plans

### Technical Requirements
1. **Cache Efficiency**: >80% cache hit rate during training
2. **Memory Usage**: <100MB additional memory for reachability data
3. **Thread Safety**: Safe for concurrent access during distributed training
4. **Error Handling**: Graceful degradation when reachability analysis fails

### Quality Requirements
1. **API Consistency**: Follows existing npp-rl coding patterns
2. **Documentation**: Comprehensive docstrings and usage examples
3. **Testing**: Full unit and integration test coverage
4. **Monitoring**: Performance metrics and debugging capabilities

## Test Scenarios

### Unit Tests
**File**: `tests/test_reachability_integration.py`

```python
class TestReachabilityIntegration(unittest.TestCase):
    def test_hierarchical_reachability_manager(self):
        """Test reachability manager integration with RL components."""
        # Mock nclone components
        analyzer = Mock(spec=ReachabilityAnalyzer)
        manager = HierarchicalReachabilityManager(analyzer)
        
        # Test subgoal extraction
        level_data = self._create_test_level_data()
        ninja_pos = (50, 400)
        switch_states = {}
        
        # Mock reachability result
        mock_result = Mock()
        mock_result.reachable_positions = {(10, 20), (15, 25), (20, 30)}
        mock_result.subgoals = [(100, 200, 'exit_switch'), (150, 250, 'door_switch')]
        analyzer.analyze_reachability.return_value = mock_result
        
        # Test subgoal extraction
        subgoals = manager.get_reachable_subgoals(ninja_pos, level_data, switch_states)
        
        self.assertIsInstance(subgoals, list)
        self.assertIn('navigate_to_exit_switch', subgoals)
        
    def test_reachability_aware_curiosity(self):
        """Test curiosity system integration with reachability."""
        manager = Mock(spec=HierarchicalReachabilityManager)
        curiosity = ReachabilityAwareCuriosity(manager)
        
        # Test curiosity computation
        obs = self._create_test_observation()
        action = 1
        next_obs = self._create_test_observation()
        
        # Mock reachable subgoals
        manager.get_reachable_subgoals.return_value = ['navigate_to_exit_switch']
        
        bonus = curiosity.compute_exploration_bonus(obs, action, next_obs)
        
        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)
        self.assertLessEqual(bonus, 1.0)
        
    def test_level_completion_planner(self):
        """Test strategic level completion planning."""
        analyzer = Mock(spec=ReachabilityAnalyzer)
        planner = PhysicsAwareLevelCompletionPlanner(analyzer)
        
        # Mock reachability analysis
        mock_result = Mock()
        mock_result.reachable_positions = {(10, 20), (15, 25)}
        analyzer.analyze_reachability.return_value = mock_result
        
        level_data = self._create_test_level_data()
        ninja_pos = (50, 400)
        switch_states = {}
        
        strategy = planner.plan_completion_strategy(ninja_pos, level_data, switch_states)
        
        self.assertIsInstance(strategy, list)
        self.assertGreater(len(strategy), 0)
        self.assertIn('navigate_to_exit_switch', strategy)
```

### Integration Tests
**File**: `tests/test_reachability_rl_integration.py`

```python
class TestReachabilityRLIntegration(unittest.TestCase):
    def test_enhanced_environment_integration(self):
        """Test reachability integration with RL environment."""
        env = ReachabilityEnhancedNPPEnv(level_name='test_level')
        
        obs = env.reset()
        
        # Should include reachability information
        self.assertIn('reachable_subgoals', obs)
        self.assertIn('strategic_plan', obs)
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # Should include reachability info
        self.assertIn('reachability', info)
        self.assertIn('subgoals', info['reachability'])
        self.assertIn('strategic_plan', info['reachability'])
        
    def test_hrl_subgoal_filtering(self):
        """Test HRL integration with reachability filtering."""
        env = ReachabilityEnhancedNPPEnv(level_name='switch_puzzle_level')
        obs = env.reset()
        
        # Extract reachable subgoals
        reachable_subgoals = obs['reachable_subgoals']
        
        # Should be encoded properly
        self.assertIsInstance(reachable_subgoals, np.ndarray)
        
        # Decode and validate
        decoded_subgoals = env._decode_subgoals(reachable_subgoals)
        self.assertIsInstance(decoded_subgoals, list)
        
        # Should not include unreachable exit door initially
        self.assertNotIn('navigate_to_exit_door', decoded_subgoals)
```

### Performance Tests
**File**: `tests/test_reachability_performance.py`

```python
class TestReachabilityPerformance(unittest.TestCase):
    def test_real_time_performance(self):
        """Test reachability system performance for real-time RL."""
        # Initialize system
        from nclone.graph.reachability_analyzer import ReachabilityAnalyzer
        from nclone.graph.trajectory_calculator import TrajectoryCalculator
        
        manager = HierarchicalReachabilityManager(
            ReachabilityAnalyzer(TrajectoryCalculator())
        )
        
        level_data = self._load_test_level("large_level")
        
        # Simulate 60 FPS updates for 10 seconds
        num_updates = 600
        ninja_positions = self._generate_ninja_trajectory(num_updates)
        switch_states_sequence = self._generate_switch_state_changes(num_updates)
        
        start_time = time.time()
        for i in range(num_updates):
            subgoals = manager.get_reachable_subgoals(
                ninja_positions[i], level_data, switch_states_sequence[i]
            )
        total_time = time.time() - start_time
        
        # Should maintain 60 FPS (16.67ms per frame)
        avg_time_per_update = total_time / num_updates
        self.assertLess(avg_time_per_update, 0.01)  # 10ms budget per update
        
    def test_cache_efficiency(self):
        """Test caching system efficiency."""
        manager = HierarchicalReachabilityManager(
            ReachabilityAnalyzer(TrajectoryCalculator())
        )
        
        level_data = self._load_test_level("switch_puzzle")
        
        # Perform repeated queries with variations
        base_pos = (100, 400)
        positions = [(base_pos[0] + i*5, base_pos[1] + j*5) 
                    for i in range(-3, 4) for j in range(-3, 4)]
        
        # First pass - populate cache
        for pos in positions:
            manager.get_reachable_subgoals(pos, level_data, {})
        
        # Second pass - should hit cache frequently
        start_time = time.time()
        for pos in positions:
            manager.get_reachable_subgoals(pos, level_data, {})
        cached_time = time.time() - start_time
        
        cache_hit_rate = manager.get_cache_hit_rate()
        self.assertGreater(cache_hit_rate, 0.8)  # >80% cache hit rate
```

## Implementation Steps

### Phase 1: Core Integration Components (1 week)
1. **Create Hierarchical Reachability Manager**
   - Implement caching system
   - Add subgoal extraction logic
   - Create performance optimizations

2. **Implement Reachability-Aware Curiosity**
   - Extend existing ICM module
   - Add reachability scaling
   - Implement frontier detection

### Phase 2: Strategic Planning (3-4 days)
1. **Create Level Completion Planner**
   - Implement completion heuristic
   - Add door-switch dependency analysis
   - Create action plan generation

2. **Add Strategic Guidance Integration**
   - Enhance environment with strategic info
   - Add plan encoding/decoding
   - Create debugging visualizations

### Phase 3: Environment Integration (3-4 days)
1. **Enhance NPP Environment**
   - Add reachability information to observations
   - Integrate with existing wrappers
   - Add performance monitoring

2. **Create Reachability Wrappers**
   - Wrapper for HRL subgoal filtering
   - Wrapper for curiosity enhancement
   - Wrapper for strategic guidance

### Phase 4: Testing and Optimization (3-4 days)
1. **Comprehensive Testing**
   - Unit tests for all components
   - Integration tests with RL training
   - Performance benchmarks

2. **Performance Optimization**
   - Profile and optimize bottlenecks
   - Tune caching parameters
   - Add monitoring and debugging tools

## Success Metrics
- **Performance**: <10ms reachability queries, >80% cache hit rate
- **Integration**: Seamless integration with existing HGT architecture
- **HRL Support**: Filtered subgoals improve hierarchical RL sample efficiency
- **Curiosity Enhancement**: Reduced exploration of unreachable areas
- **Strategic Guidance**: Actionable completion plans for complex levels

## Dependencies
- Enhanced reachability system from nclone (Task 003 in nclone)
- Existing HGT multimodal architecture
- ICM curiosity module
- NPP environment implementation

## Estimated Effort
- **Time**: 2-3 weeks
- **Complexity**: High (complex integration)
- **Risk**: Medium (depends on nclone integration)

## Notes
- Coordinate closely with nclone reachability enhancements
- Ensure thread safety for distributed training
- Plan for gradual rollout and A/B testing
- Consider backward compatibility with existing training pipelines