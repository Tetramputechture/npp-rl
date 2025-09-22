# TASK 003: Create Hierarchical Reachability Manager for HRL

## Overview
Implement a hierarchical reachability manager that provides filtered subgoals for hierarchical reinforcement learning, enabling strategic level completion planning and improved sample efficiency.

## Context & Justification

### Hierarchical RL Requirements
Based on analysis of npp-rl architecture and `/workspace/npp-rl/tasks/TASK_002_integrate_reachability_system.md`:
- **Subgoal Selection**: Filter available subgoals based on reachability analysis
- **Strategic Planning**: Provide level completion strategies for high-level policy
- **Dynamic Adaptation**: Update subgoals as switch states change
- **Performance**: Real-time subgoal filtering for 60 FPS gameplay

### Current Limitations
- **No Strategic Guidance**: Agent explores randomly without level completion strategy
- **Inefficient Subgoal Selection**: Attempts impossible subgoals, wasting time
- **No Switch Dependency Understanding**: Doesn't understand switch-door relationships
- **Lack of Hierarchical Structure**: No decomposition of complex navigation tasks

### Research Foundation
From `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **Level Completion Heuristic**: Strategic switch activation sequences
- **Reachability-Guided HRL**: Filter subgoals based on current reachability
- **Dynamic Subgoal Updates**: Adapt to changing game state
- **Performance Benefits**: 2-3x improvement in level completion rates

### Theoretical Background
- **Sutton et al. (1999)**: Options framework for hierarchical RL
- **Bacon et al. (2017)**: Option-Critic architecture for learning hierarchies
- **Nachum et al. (2018)**: Data-Efficient Hierarchical Reinforcement Learning
- **Levy et al. (2019)**: Hierarchical Actor-Critic with reachability constraints

## Technical Specification

### Hierarchical Reachability Manager Architecture
```python
class HierarchicalReachabilityManager:
    """
    Manages hierarchical subgoals based on reachability analysis.
    
    Key Components:
    1. Subgoal Extraction: Convert reachability analysis to actionable subgoals
    2. Strategic Planning: Level completion strategy generation
    3. Dynamic Updates: Adapt subgoals to changing game state
    4. Caching: Efficient subgoal caching and invalidation
    """
    
    def __init__(self, reachability_extractor, level_analyzer=None):
        self.reachability_extractor = reachability_extractor
        self.level_analyzer = level_analyzer or LevelAnalyzer()
        
        # Subgoal management
        self.subgoal_cache = {}
        self.cache_ttl = 200  # milliseconds
        self.last_switch_states = {}
        self.last_ninja_pos = None
        
        # Strategic planning
        self.completion_planner = LevelCompletionPlanner()
        self.subgoal_prioritizer = SubgoalPrioritizer()
        
        # Performance monitoring
        self.cache_hit_rate = 0.0
        self.avg_subgoal_count = 0.0
        self.planning_time_ms = 0.0
    
    def get_available_subgoals(self, ninja_pos, level_data, switch_states, 
                              max_subgoals=8) -> List[Subgoal]:
        """
        Get currently available subgoals based on reachability analysis.
        
        Args:
            ninja_pos: Current ninja position
            level_data: Level data structure
            switch_states: Current switch states
            max_subgoals: Maximum number of subgoals to return
        
        Returns:
            List of available subgoals, prioritized by strategic value
        """
        start_time = time.perf_counter()
        
        # Check cache first
        cache_key = self._generate_cache_key(ninja_pos, switch_states, level_data)
        if self._is_cache_valid(cache_key):
            cached_result = self.subgoal_cache[cache_key]
            self._update_cache_metrics(True)
            return cached_result['subgoals'][:max_subgoals]
        
        # Extract reachability features
        reachability_features = self.reachability_extractor.extract_features(
            ninja_pos, level_data, switch_states, performance_target="balanced"
        )
        
        # Generate subgoals
        subgoals = self._generate_subgoals(
            ninja_pos, level_data, switch_states, reachability_features
        )
        
        # Prioritize subgoals
        prioritized_subgoals = self.subgoal_prioritizer.prioritize(
            subgoals, ninja_pos, level_data, reachability_features
        )
        
        # Cache result
        self._cache_subgoals(cache_key, prioritized_subgoals, reachability_features)
        
        # Update metrics
        self.planning_time_ms = (time.perf_counter() - start_time) * 1000
        self.avg_subgoal_count = len(prioritized_subgoals)
        self._update_cache_metrics(False)
        
        return prioritized_subgoals[:max_subgoals]
    
    def get_completion_strategy(self, ninja_pos, level_data, switch_states) -> CompletionStrategy:
        """
        Get strategic plan for level completion.
        
        Returns:
            High-level strategy with ordered subgoals for level completion
        """
        # Use completion planner to generate strategy
        strategy = self.completion_planner.plan_completion(
            ninja_pos, level_data, switch_states, self.reachability_extractor
        )
        
        return strategy
    
    def update_subgoals_on_switch_change(self, ninja_pos, level_data, 
                                       old_switch_states, new_switch_states):
        """
        Update subgoals when switch states change.
        
        This is called when a switch is activated to immediately update
        available subgoals without waiting for cache expiration.
        """
        # Invalidate relevant cache entries
        self._invalidate_switch_dependent_cache(old_switch_states, new_switch_states)
        
        # Generate new subgoals
        new_subgoals = self.get_available_subgoals(ninja_pos, level_data, new_switch_states)
        
        # Notify about newly available subgoals
        newly_available = self._find_newly_available_subgoals(
            old_switch_states, new_switch_states, level_data
        )
        
        return new_subgoals, newly_available
    
    def _generate_subgoals(self, ninja_pos, level_data, switch_states, 
                          reachability_features) -> List[Subgoal]:
        """
        Generate subgoals from reachability analysis and level data.
        """
        subgoals = []
        
        # Extract objective distances from reachability features
        objective_distances = reachability_features[0:8].numpy()
        switch_features = reachability_features[8:24].numpy()
        
        # Generate navigation subgoals
        navigation_subgoals = self._generate_navigation_subgoals(
            ninja_pos, level_data, objective_distances
        )
        subgoals.extend(navigation_subgoals)
        
        # Generate switch activation subgoals
        switch_subgoals = self._generate_switch_subgoals(
            level_data, switch_states, switch_features
        )
        subgoals.extend(switch_subgoals)
        
        # Generate avoidance subgoals
        avoidance_subgoals = self._generate_avoidance_subgoals(
            ninja_pos, level_data, reachability_features
        )
        subgoals.extend(avoidance_subgoals)
        
        return subgoals
    
    def _generate_navigation_subgoals(self, ninja_pos, level_data, 
                                    objective_distances) -> List[Subgoal]:
        """
        Generate navigation subgoals to key objectives.
        """
        subgoals = []
        
        # Get key objectives from level data
        objectives = self.level_analyzer.get_key_objectives(level_data)
        
        for i, objective in enumerate(objectives[:8]):  # Match feature dimensions
            if i < len(objective_distances) and objective_distances[i] < 1.0:  # Reachable
                subgoal = NavigationSubgoal(
                    target_position=objective.position,
                    target_type=objective.type,
                    distance=objective_distances[i],
                    priority=self._calculate_navigation_priority(objective, objective_distances[i])
                )
                subgoals.append(subgoal)
        
        return subgoals
    
    def _generate_switch_subgoals(self, level_data, switch_states, 
                                switch_features) -> List[Subgoal]:
        """
        Generate switch activation subgoals.
        """
        subgoals = []
        
        switches = self.level_analyzer.get_all_switches(level_data)
        
        for i, switch in enumerate(switches[:16]):  # Match feature dimensions
            if i < len(switch_features):
                switch_feature = switch_features[i]
                
                # Only create subgoal for reachable, inactive switches
                if 0.1 < switch_feature < 0.9:  # Reachable but not activated
                    subgoal = SwitchActivationSubgoal(
                        switch_id=switch.id,
                        switch_position=switch.position,
                        switch_type=switch.type,
                        reachability_score=switch_feature,
                        priority=self._calculate_switch_priority(switch, switch_feature)
                    )
                    subgoals.append(subgoal)
        
        return subgoals
    
    def _generate_avoidance_subgoals(self, ninja_pos, level_data, 
                                   reachability_features) -> List[Subgoal]:
        """
        Generate hazard avoidance subgoals.
        """
        subgoals = []
        
        # Extract hazard proximity features
        hazard_proximities = reachability_features[24:40].numpy()
        
        # Get hazards from level data
        hazards = self.level_analyzer.get_hazards(level_data)
        
        for i, hazard in enumerate(hazards[:16]):  # Match feature dimensions
            if i < len(hazard_proximities) and hazard_proximities[i] > 0.7:  # High threat
                subgoal = AvoidanceSubgoal(
                    hazard_position=hazard.position,
                    hazard_type=hazard.type,
                    threat_level=hazard_proximities[i],
                    safe_distance=hazard.get_safe_distance(),
                    priority=self._calculate_avoidance_priority(hazard, hazard_proximities[i])
                )
                subgoals.append(subgoal)
        
        return subgoals
```

### Subgoal Types and Structures
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any

@dataclass
class Subgoal(ABC):
    """Base class for all subgoals."""
    priority: float
    estimated_time: float
    success_probability: float
    
    @abstractmethod
    def get_target_position(self) -> Tuple[float, float]:
        """Get the target position for this subgoal."""
        pass
    
    @abstractmethod
    def is_completed(self, ninja_pos: Tuple[float, float], 
                    level_data, switch_states: Dict) -> bool:
        """Check if this subgoal has been completed."""
        pass
    
    @abstractmethod
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        """Get reward shaping signal for progress toward this subgoal."""
        pass

@dataclass
class NavigationSubgoal(Subgoal):
    """Navigate to a specific position."""
    target_position: Tuple[float, float]
    target_type: str  # 'exit_door', 'exit_switch', 'door_switch', etc.
    distance: float
    
    def get_target_position(self) -> Tuple[float, float]:
        return self.target_position
    
    def is_completed(self, ninja_pos: Tuple[float, float], 
                    level_data, switch_states: Dict) -> bool:
        # Check if ninja is within interaction range of target
        distance = math.sqrt((ninja_pos[0] - self.target_position[0])**2 + 
                           (ninja_pos[1] - self.target_position[1])**2)
        return distance <= 24.0  # One tile distance
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        # Reward for getting closer to target
        distance = math.sqrt((ninja_pos[0] - self.target_position[0])**2 + 
                           (ninja_pos[1] - self.target_position[1])**2)
        max_distance = 500.0  # Normalize by reasonable max distance
        return (max_distance - distance) / max_distance

@dataclass
class SwitchActivationSubgoal(Subgoal):
    """Activate a specific switch."""
    switch_id: str
    switch_position: Tuple[float, float]
    switch_type: str
    reachability_score: float
    
    def get_target_position(self) -> Tuple[float, float]:
        return self.switch_position
    
    def is_completed(self, ninja_pos: Tuple[float, float], 
                    level_data, switch_states: Dict) -> bool:
        return switch_states.get(self.switch_id, False)
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        # Reward for getting closer to switch
        distance = math.sqrt((ninja_pos[0] - self.switch_position[0])**2 + 
                           (ninja_pos[1] - self.switch_position[1])**2)
        max_distance = 500.0
        proximity_reward = (max_distance - distance) / max_distance
        
        # Bonus for high reachability score
        reachability_bonus = self.reachability_score * 0.5
        
        return proximity_reward + reachability_bonus

@dataclass
class CollectionSubgoal(Subgoal):
    """Collect a specific item."""
    target_position: Tuple[float, float]
    item_type: str
    value: float
    area_connectivity: float
    
    def get_target_position(self) -> Tuple[float, float]:
        return self.target_position
    
    def is_completed(self, ninja_pos: Tuple[float, float], 
                    level_data, switch_states: Dict) -> bool:
        # Check if item still exists at position
        # This would need integration with level data to track collected items
        return False  # Simplified - would need proper item tracking
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        distance = math.sqrt((ninja_pos[0] - self.target_position[0])**2 + 
                           (ninja_pos[1] - self.target_position[1])**2)
        max_distance = 500.0
        proximity_reward = (max_distance - distance) / max_distance
        
        # Scale by item value and area connectivity
        value_bonus = self.value * 0.1
        connectivity_bonus = self.area_connectivity * 0.3
        
        return proximity_reward + value_bonus + connectivity_bonus

@dataclass
class AvoidanceSubgoal(Subgoal):
    """Avoid a specific hazard."""
    hazard_position: Tuple[float, float]
    hazard_type: str
    threat_level: float
    safe_distance: float
    
    def get_target_position(self) -> Tuple[float, float]:
        # Return position that maintains safe distance
        # This is a simplified implementation
        return self.hazard_position
    
    def is_completed(self, ninja_pos: Tuple[float, float], 
                    level_data, switch_states: Dict) -> bool:
        # Check if ninja is at safe distance from hazard
        distance = math.sqrt((ninja_pos[0] - self.hazard_position[0])**2 + 
                           (ninja_pos[1] - self.hazard_position[1])**2)
        return distance >= self.safe_distance
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        distance = math.sqrt((ninja_pos[0] - self.hazard_position[0])**2 + 
                           (ninja_pos[1] - self.hazard_position[1])**2)
        
        if distance >= self.safe_distance:
            return 1.0  # Safe
        else:
            # Penalty increases as ninja gets closer to hazard
            danger_ratio = distance / self.safe_distance
            return danger_ratio * self.threat_level
```

### Level Completion Planner
```python
class LevelCompletionPlanner:
    """
    Strategic planner for level completion using reachability analysis.
    
    Implements the completion heuristic from the strategic analysis:
    1. Check path to exit switch
    2. If blocked, find required door switches
    3. Plan switch activation sequence
    4. Navigate to exit
    """
    
    def __init__(self):
        self.path_analyzer = PathAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def plan_completion(self, ninja_pos, level_data, switch_states, 
                       reachability_extractor) -> 'CompletionStrategy':
        """
        Generate strategic plan for level completion.
        """
        # Extract reachability features
        reachability_features = reachability_extractor.extract_features(
            ninja_pos, level_data, switch_states, performance_target="balanced"
        )
        
        # Identify key objectives
        exit_door = self._find_exit_door(level_data)
        exit_switch = self._find_exit_switch(level_data)
        
        if not exit_door or not exit_switch:
            return CompletionStrategy([], "No exit found", 0.0)
        
        # Plan completion sequence
        completion_steps = []
        
        # Step 1: Check if exit switch is reachable
        exit_switch_reachable = self._is_objective_reachable(
            exit_switch.position, reachability_features
        )
        
        if not exit_switch_reachable:
            # Find blocking doors and required switches
            blocking_analysis = self._analyze_blocking_doors(
                ninja_pos, exit_switch.position, level_data, switch_states
            )
            
            # Plan switch activation sequence
            switch_sequence = self._plan_switch_sequence(
                blocking_analysis, level_data, switch_states, reachability_features
            )
            
            completion_steps.extend(switch_sequence)
        
        # Step 2: Navigate to and activate exit switch
        if not switch_states.get(exit_switch.id, False):
            completion_steps.append(CompletionStep(
                action_type='navigate_and_activate',
                target_position=exit_switch.position,
                target_id=exit_switch.id,
                description=f"Activate exit switch at {exit_switch.position}",
                priority=0.9
            ))
        
        # Step 3: Navigate to exit door
        completion_steps.append(CompletionStep(
            action_type='navigate_to_exit',
            target_position=exit_door.position,
            target_id=exit_door.id,
            description=f"Navigate to exit door at {exit_door.position}",
            priority=1.0
        ))
        
        # Calculate strategy confidence
        confidence = self._calculate_strategy_confidence(
            completion_steps, reachability_features
        )
        
        return CompletionStrategy(
            steps=completion_steps,
            description="Strategic level completion plan",
            confidence=confidence
        )
    
    def _plan_switch_sequence(self, blocking_analysis, level_data, 
                            switch_states, reachability_features):
        """
        Plan optimal sequence for activating required switches.
        """
        required_switches = blocking_analysis['required_switches']
        
        # Sort switches by strategic priority
        switch_priorities = []
        for switch in required_switches:
            priority = self._calculate_switch_strategic_priority(
                switch, level_data, reachability_features
            )
            switch_priorities.append((switch, priority))
        
        # Sort by priority (highest first)
        switch_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Create completion steps
        completion_steps = []
        for switch, priority in switch_priorities:
            if not switch_states.get(switch.id, False):  # Not yet activated
                step = CompletionStep(
                    action_type='navigate_and_activate',
                    target_position=switch.position,
                    target_id=switch.id,
                    description=f"Activate {switch.type} switch at {switch.position}",
                    priority=priority
                )
                completion_steps.append(step)
        
        return completion_steps

@dataclass
class CompletionStrategy:
    """Strategic plan for level completion."""
    steps: List['CompletionStep']
    description: str
    confidence: float
    
    def get_next_subgoal(self) -> Optional[Subgoal]:
        """Get the next subgoal from the completion strategy."""
        if not self.steps:
            return None
        
        next_step = self.steps[0]
        
        if next_step.action_type == 'navigate_and_activate':
            return SwitchActivationSubgoal(
                switch_id=next_step.target_id,
                switch_position=next_step.target_position,
                switch_type='strategic',
                reachability_score=1.0,
                priority=next_step.priority,
                estimated_time=30.0,  # seconds
                success_probability=self.confidence
            )
        elif next_step.action_type == 'navigate_to_exit':
            return NavigationSubgoal(
                target_position=next_step.target_position,
                target_type='exit_door',
                distance=0.0,  # Will be calculated
                priority=next_step.priority,
                estimated_time=20.0,
                success_probability=self.confidence
            )
        
        return None
    
    def update_progress(self, completed_step_id: str):
        """Update strategy progress by removing completed step."""
        self.steps = [step for step in self.steps if step.target_id != completed_step_id]

@dataclass
class CompletionStep:
    """Individual step in completion strategy."""
    action_type: str  # 'navigate_and_activate', 'navigate_to_exit'
    target_position: Tuple[float, float]
    target_id: str
    description: str
    priority: float
```

## Implementation Plan

### Phase 1: Core Manager Implementation (Week 1)
**Deliverables**:
1. **HierarchicalReachabilityManager**: Main manager class
2. **Subgoal Types**: All subgoal classes and structures
3. **Basic Subgoal Generation**: Core subgoal extraction logic

**Key Files**:
- `npp_rl/planning/hierarchical_reachability_manager.py` (NEW)
- `npp_rl/planning/subgoals.py` (NEW)
- `npp_rl/planning/level_analyzer.py` (NEW)

### Phase 2: Strategic Planning (Week 2)
**Deliverables**:
1. **LevelCompletionPlanner**: Strategic level completion planning
2. **SubgoalPrioritizer**: Intelligent subgoal prioritization
3. **DependencyAnalyzer**: Switch-door dependency analysis

**Key Files**:
- `npp_rl/planning/completion_planner.py` (NEW)
- `npp_rl/planning/subgoal_prioritizer.py` (NEW)
- `npp_rl/planning/dependency_analyzer.py` (NEW)

### Phase 3: HRL Integration (Week 3)
**Deliverables**:
1. **HRL Environment Wrapper**: Integration with hierarchical RL
2. **Subgoal Reward Shaping**: Reward shaping for subgoal progress
3. **Dynamic Subgoal Updates**: Real-time subgoal adaptation

**Implementation**:
```python
class HierarchicalRLWrapper(gym.Wrapper):
    """
    Environment wrapper that provides hierarchical RL with reachability-based subgoals.
    """
    
    def __init__(self, env, max_subgoals=8, subgoal_reward_scale=0.1):
        super().__init__(env)
        
        self.max_subgoals = max_subgoals
        self.subgoal_reward_scale = subgoal_reward_scale
        
        # Initialize hierarchical manager
        self.reachability_manager = HierarchicalReachabilityManager(
            reachability_extractor=env.reachability_extractor
        )
        
        # Current subgoals and strategy
        self.current_subgoals = []
        self.completion_strategy = None
        self.active_subgoal = None
        
        # Extend observation space for subgoals
        self._extend_observation_space()
    
    def step(self, action):
        """Enhanced step with hierarchical subgoal management."""
        obs, reward, done, info = self.env.step(action)
        
        # Get current game state
        ninja_pos = self._extract_ninja_position(obs)
        level_data = self._extract_level_data(obs)
        switch_states = self._extract_switch_states(obs)
        
        # Update subgoals if switch states changed
        if self._switch_states_changed(switch_states):
            self._update_subgoals(ninja_pos, level_data, switch_states)
        
        # Check subgoal completion
        subgoal_reward = self._check_subgoal_progress(ninja_pos, level_data, switch_states)
        
        # Add subgoal reward shaping
        total_reward = reward + (subgoal_reward * self.subgoal_reward_scale)
        
        # Add subgoal information to observation
        obs = self._add_subgoal_info(obs)
        
        # Add hierarchical info
        info['hierarchical'] = {
            'current_subgoals': [sg.__class__.__name__ for sg in self.current_subgoals],
            'active_subgoal': self.active_subgoal.__class__.__name__ if self.active_subgoal else None,
            'subgoal_reward': subgoal_reward,
            'completion_strategy_confidence': self.completion_strategy.confidence if self.completion_strategy else 0.0
        }
        
        return obs, total_reward, done, info
    
    def _update_subgoals(self, ninja_pos, level_data, switch_states):
        """Update available subgoals based on current state."""
        # Get available subgoals
        self.current_subgoals = self.reachability_manager.get_available_subgoals(
            ninja_pos, level_data, switch_states, self.max_subgoals
        )
        
        # Update completion strategy
        self.completion_strategy = self.reachability_manager.get_completion_strategy(
            ninja_pos, level_data, switch_states
        )
        
        # Select active subgoal (highest priority)
        if self.current_subgoals:
            self.active_subgoal = max(self.current_subgoals, key=lambda sg: sg.priority)
        else:
            self.active_subgoal = None
    
    def _check_subgoal_progress(self, ninja_pos, level_data, switch_states):
        """Check progress on current subgoals and provide reward shaping."""
        total_subgoal_reward = 0.0
        
        # Check active subgoal
        if self.active_subgoal:
            # Reward shaping for progress
            progress_reward = self.active_subgoal.get_reward_shaping(ninja_pos)
            total_subgoal_reward += progress_reward
            
            # Check completion
            if self.active_subgoal.is_completed(ninja_pos, level_data, switch_states):
                # Completion bonus
                completion_bonus = 1.0 * self.active_subgoal.priority
                total_subgoal_reward += completion_bonus
                
                # Remove completed subgoal
                self.current_subgoals = [sg for sg in self.current_subgoals 
                                       if sg != self.active_subgoal]
                
                # Update completion strategy
                if self.completion_strategy:
                    self.completion_strategy.update_progress(
                        getattr(self.active_subgoal, 'switch_id', 
                               getattr(self.active_subgoal, 'target_type', ''))
                    )
                
                # Select new active subgoal
                if self.current_subgoals:
                    self.active_subgoal = max(self.current_subgoals, key=lambda sg: sg.priority)
                else:
                    self.active_subgoal = None
        
        return total_subgoal_reward
```

### Phase 4: Testing and Optimization (Week 4)
**Deliverables**:
1. **Comprehensive Testing**: Unit and integration tests
2. **Performance Optimization**: Caching and performance tuning
3. **HRL Evaluation**: Performance comparison with baseline

## Testing Strategy

### Unit Tests
```python
class TestHierarchicalReachabilityManager(unittest.TestCase):
    def setUp(self):
        self.reachability_extractor = MockReachabilityExtractor()
        self.manager = HierarchicalReachabilityManager(self.reachability_extractor)
        self.test_level = load_test_level("complex-path-switch-required")
    
    def test_subgoal_generation(self):
        """Test that subgoals are generated correctly."""
        ninja_pos = (100, 400)
        switch_states = {'exit_switch': False, 'door_switch_1': False}
        
        subgoals = self.manager.get_available_subgoals(
            ninja_pos, self.test_level.data, switch_states
        )
        
        # Should generate multiple types of subgoals
        self.assertGreater(len(subgoals), 0)
        
        # Check subgoal types
        subgoal_types = {type(sg).__name__ for sg in subgoals}
        self.assertIn('NavigationSubgoal', subgoal_types)
        self.assertIn('SwitchActivationSubgoal', subgoal_types)
    
    def test_subgoal_prioritization(self):
        """Test that subgoals are properly prioritized."""
        ninja_pos = (100, 400)
        switch_states = {'exit_switch': False}
        
        subgoals = self.manager.get_available_subgoals(
            ninja_pos, self.test_level.data, switch_states
        )
        
        # Subgoals should be sorted by priority
        priorities = [sg.priority for sg in subgoals]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
    
    def test_completion_strategy(self):
        """Test completion strategy generation."""
        ninja_pos = (100, 400)
        switch_states = {'exit_switch': False, 'door_switch_1': False}
        
        strategy = self.manager.get_completion_strategy(
            ninja_pos, self.test_level.data, switch_states
        )
        
        self.assertIsInstance(strategy, CompletionStrategy)
        self.assertGreater(len(strategy.steps), 0)
        self.assertGreater(strategy.confidence, 0.0)
    
    def test_switch_state_updates(self):
        """Test subgoal updates when switch states change."""
        ninja_pos = (100, 400)
        old_states = {'exit_switch': False, 'door_switch_1': False}
        new_states = {'exit_switch': False, 'door_switch_1': True}
        
        new_subgoals, newly_available = self.manager.update_subgoals_on_switch_change(
            ninja_pos, self.test_level.data, old_states, new_states
        )
        
        self.assertIsInstance(new_subgoals, list)
        self.assertIsInstance(newly_available, list)
    
    def test_caching_performance(self):
        """Test that caching improves performance."""
        ninja_pos = (100, 400)
        switch_states = {'exit_switch': False}
        
        # First call (cache miss)
        start_time = time.perf_counter()
        subgoals1 = self.manager.get_available_subgoals(
            ninja_pos, self.test_level.data, switch_states
        )
        first_time = (time.perf_counter() - start_time) * 1000
        
        # Second call (cache hit)
        start_time = time.perf_counter()
        subgoals2 = self.manager.get_available_subgoals(
            ninja_pos, self.test_level.data, switch_states
        )
        second_time = (time.perf_counter() - start_time) * 1000
        
        # Cache hit should be much faster
        self.assertLess(second_time, first_time * 0.1)
        self.assertEqual(len(subgoals1), len(subgoals2))
```

### Integration Tests
```python
class TestHierarchicalRLIntegration(unittest.TestCase):
    def setUp(self):
        base_env = ReachabilityEnhancedNPPEnv(render_mode='rgb_array')
        self.env = HierarchicalRLWrapper(base_env)
    
    def test_hierarchical_wrapper(self):
        """Test hierarchical RL wrapper functionality."""
        obs = self.env.reset()
        
        # Check that observation includes subgoal information
        self.assertIn('subgoals', obs)
        self.assertIn('active_subgoal', obs)
        
        # Take steps and check subgoal updates
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            # Check hierarchical info
            self.assertIn('hierarchical', info)
            hierarchical_info = info['hierarchical']
            
            self.assertIn('current_subgoals', hierarchical_info)
            self.assertIn('active_subgoal', hierarchical_info)
            self.assertIn('subgoal_reward', hierarchical_info)
            
            if done:
                obs = self.env.reset()
    
    def test_subgoal_reward_shaping(self):
        """Test that subgoal reward shaping works correctly."""
        obs = self.env.reset()
        
        total_subgoal_reward = 0.0
        steps = 0
        
        for _ in range(200):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            
            subgoal_reward = info['hierarchical']['subgoal_reward']
            total_subgoal_reward += subgoal_reward
            steps += 1
            
            if done:
                break
        
        # Should receive some subgoal reward over time
        avg_subgoal_reward = total_subgoal_reward / steps
        self.assertGreater(avg_subgoal_reward, 0.0)
    
    def test_performance_requirements(self):
        """Test that hierarchical manager meets performance requirements."""
        obs = self.env.reset()
        
        planning_times = []
        for _ in range(100):
            action = self.env.action_space.sample()
            
            start_time = time.perf_counter()
            obs, reward, done, info = self.env.step(action)
            step_time = (time.perf_counter() - start_time) * 1000
            
            planning_times.append(step_time)
            
            if done:
                obs = self.env.reset()
        
        # Check performance targets
        avg_time = np.mean(planning_times)
        p95_time = np.percentile(planning_times, 95)
        
        self.assertLess(avg_time, 5.0, f"Average planning time too high: {avg_time}ms")
        self.assertLess(p95_time, 10.0, f"95th percentile time too high: {p95_time}ms")
```

## Success Criteria

### Performance Requirements
- **Subgoal Generation**: <3ms average per update
- **Strategy Planning**: <5ms for completion strategy generation
- **Memory Usage**: <50MB additional memory for subgoal management
- **Cache Hit Rate**: >70% during typical training

### Quality Requirements
- **Subgoal Relevance**: Generated subgoals should be achievable and strategic
- **Priority Accuracy**: Higher priority subgoals should lead to faster level completion
- **Strategy Effectiveness**: Completion strategies should improve success rates
- **Dynamic Adaptation**: Subgoals should update appropriately when game state changes

### Training Requirements
- **Sample Efficiency**: 30-50% improvement in sample efficiency on complex levels
- **Success Rate**: Higher level completion rates compared to non-hierarchical approach
- **Convergence Speed**: Faster convergence on levels with complex switch dependencies

## Risk Mitigation

### Technical Risks
1. **Complexity Overhead**: Careful performance monitoring and optimization
2. **Subgoal Quality**: Extensive validation of subgoal generation logic
3. **Integration Issues**: Comprehensive testing with existing RL pipeline

### Training Risks
1. **Reward Shaping Issues**: Careful tuning of subgoal reward scaling
2. **Subgoal Conflicts**: Logic to handle conflicting or impossible subgoals
3. **Overfitting**: Validation on diverse levels and scenarios

## Deliverables

1. **HierarchicalReachabilityManager**: Complete subgoal management system
2. **Subgoal Framework**: All subgoal types and supporting structures
3. **Strategic Planning**: Level completion planning and dependency analysis
4. **HRL Integration**: Complete integration with hierarchical RL training
5. **Performance Analysis**: Comprehensive evaluation of hierarchical benefits

## Timeline

- **Week 1**: Core manager implementation and subgoal types
- **Week 2**: Strategic planning and completion strategy generation
- **Week 3**: HRL integration and environment wrapper
- **Week 4**: Testing, optimization, and performance evaluation

## Dependencies

### Internal Dependencies
- **Compact Reachability Features**: TASK_001 (Feature integration)
- **Reachability-Aware Curiosity**: TASK_002 (Curiosity integration)
- **Base Environment**: ReachabilityEnhancedNPPEnv

### External Dependencies
- **Gym**: Environment wrapper framework
- **NumPy**: Numerical operations
- **PyTorch**: Neural network components (if needed)

## References

1. **Strategic Analysis**: `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`
2. **HRL Integration**: `/workspace/npp-rl/tasks/TASK_002_integrate_reachability_system.md`
3. **Sutton et al. (1999)**: "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning"
4. **Bacon et al. (2017)**: "The Option-Critic Architecture"
5. **Nachum et al. (2018)**: "Data-Efficient Hierarchical Reinforcement Learning"