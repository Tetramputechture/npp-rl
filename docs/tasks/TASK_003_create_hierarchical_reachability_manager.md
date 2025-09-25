# TASK 003: Create Hierarchical Reachability Manager for HRL

## Overview
Implement a hierarchical reachability manager that provides filtered subgoals for hierarchical reinforcement learning, enabling strategic level completion planning and improved sample efficiency.

## Context & Justification

### Hierarchical RL Requirements
Based on analysis of npp-rl architecture and integration with fast neural reachability system:
- **Switch-Focused Subgoals**: Generate subgoals for switch activation and navigation using fast reachability analysis
- **Performance-First Strategy**: Use neural network features (graph transformer + 3D CNN + MLPs) rather than expensive physics calculations
- **NPP Completion Algorithm**: Implement the definitive two-phase switch-based level progression algorithm
- **Real-Time Performance**: <3ms subgoal generation for 60 FPS gameplay using neural feature caching

### Current Limitations
- **No Strategic Guidance**: Agent explores randomly without level completion strategy
- **Inefficient Subgoal Selection**: Attempts impossible subgoals, wasting time
- **No Switch Dependency Understanding**: Doesn't understand switch-door relationships
- **Lack of Hierarchical Structure**: No decomposition of complex navigation tasks

### Research Foundation
From `/workspace/nclone/docs/reachability_analysis_integration_strategy.md`:
- **NPP Level Completion Heuristic**: Two-phase algorithm for systematic switch-based level progression
  1. Phase 1: Ensure exit door switch accessibility through locked door switch activation
  2. Phase 2: Ensure exit door accessibility through additional locked door switch activation
- **Reachability-Guided HRL**: Filter subgoals based on neural network reachability analysis
- **Dynamic Subgoal Updates**: Real-time adaptation to switch state changes
- **Performance Benefits**: 2-3x improvement in level completion rates through systematic progression

### Theoretical Background
- **Sutton et al. (1999)**: Options framework for hierarchical RL
- **Bacon et al. (2017)**: Option-Critic architecture for learning hierarchies
- **Nachum et al. (2018)**: Data-Efficient Hierarchical Reinforcement Learning
- **Levy et al. (2019)**: Hierarchical Actor-Critic with reachability constraints

## Technical Specification

### Code Organization and Documentation Requirements

**Import Requirements**:
- Always prefer top-level imports at the module level
- Import all dependencies at the top of the file before any class or function definitions
- Group imports by: standard library, third-party, nclone, npp_rl modules

**Documentation Requirements**:
- **Top-level module docstrings**: Every modified module must have comprehensive docstrings explaining the hierarchical reachability integration approach and theoretical foundation
- **Inline documentation**: Complex planning algorithms require detailed inline comments explaining the hierarchical reasoning and reachability-guided subgoal generation
- **Paper references**: All hierarchical RL and planning techniques must reference original research papers in docstrings and comments
- **Integration notes**: Document how each component integrates with existing nclone reachability systems and existing agent architecture

**Module Modification Approach**:
- **Update existing agent modules in place** rather than creating separate hierarchical managers
- Extend existing adaptive exploration and agent classes with hierarchical planning functionality
- Add hierarchical reachability management directly to existing modules in `/npp_rl/agents/`

**Defensive Programming Guidelines**:
- **Avoid defensive try/catch blocks** around imports or critical functionality
- **Fail fast and clearly** when dependencies are missing or components are misconfigured
- **No redundant components** - integrate with existing systems rather than duplicating functionality

**Performance and Implementation Guidelines**:
- **Avoid expensive physics calculations** - use the fast reachability analysis system instead of complex physics approximations
- **Leverage existing neural architecture** - remember we have a sophisticated neural network with graph transformer, 3D CNN, and MLPs for reachability and game state analysis
- **No redundant physics validation** - trust the reachability analysis rather than implementing additional physics checks
- **Performance-first approach** - prioritize the <3ms subgoal generation target over physics accuracy

**Production Implementation Requirements**:
- **NO simplified implementations** - avoid placeholder code, stub methods, or "simplified for demonstration" approaches
- **NO "in practice" notes** - implement complete, production-ready functionality from the start
- **Holistic and thorough implementations** - every component must be fully functional and ready for production deployment
- **Complete error handling** - implement robust error handling without defensive programming anti-patterns

### Agent Module Modifications
**Target File**: Extend existing `/npp_rl/agents/adaptive_exploration.py`

**Required Documentation Additions**:
```python
"""
Adaptive Exploration with Hierarchical Reachability-Guided Planning

This module extends the existing adaptive exploration strategies with hierarchical
reinforcement learning capabilities based on reachability analysis from nclone.
The integration provides strategic subgoal generation and level completion planning.

Integration Strategy:
- Hierarchical subgoal extraction converts reachability analysis to actionable objectives
- Strategic planning generates level completion sequences based on switch dependencies  
- Dynamic subgoal updates adapt to changing game state for optimal exploration
- Performance-optimized caching for real-time subgoal management (<3ms target)

Theoretical Foundation:
- Options framework: Sutton et al. (1999) "Between MDPs and semi-MDPs"
- Option-Critic: Bacon et al. (2017) "The Option-Critic Architecture"  
- Data-efficient HRL: Nachum et al. (2018) "Data-Efficient Hierarchical Reinforcement Learning"
- Reachability-guided HRL: Levy et al. (2019) "Hierarchical Actor-Critic with reachability constraints"
- Strategic planning: Custom integration with nclone physics and level completion heuristics

nclone Integration:
- Uses compact reachability features from nclone.graph.reachability for subgoal filtering
- Integrates with ReachabilitySystem for performance-optimized planning queries
- Maintains compatibility with existing NPP physics constants and level objective analysis
"""

# Standard library imports
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# nclone imports (top-level imports preferred)
from nclone.constants import NINJA_RADIUS, GRAVITY_FALL, MAX_HOR_SPEED
from nclone.graph.reachability.compact_features import ReachabilityFeatureExtractor
from nclone.graph.reachability.reachability_system import ReachabilitySystem

# npp_rl imports  
from npp_rl.agents.base import BaseExplorationManager
from npp_rl.utils.level_analysis import LevelAnalyzer


class AdaptiveExplorationManager:
    """
    Adaptive exploration manager with integrated hierarchical reachability-guided planning.
    
    This manager extends the existing curiosity-driven exploration with hierarchical
    reinforcement learning capabilities that provide strategic subgoals based on
    reachability analysis for improved sample efficiency.
    
    Architecture Extensions:
    1. Base Exploration: ICM + Novelty detection (existing functionality preserved)
    2. Subgoal Extraction: Convert reachability analysis to actionable hierarchical objectives
    3. Strategic Planning: Level completion strategy generation using switch dependency analysis
    4. Dynamic Updates: Real-time subgoal adaptation to changing game state
    5. Performance Optimization: Efficient subgoal caching and invalidation system
    
    Hierarchical Integration:
    - Lazy initialization of reachability extractor for dependency management
    - Subgoal prioritization based on strategic value and completion likelihood
    - Dynamic cache management for real-time performance (<3ms subgoal generation)
    
    References:
    - Base exploration: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    - Hierarchical RL: Sutton et al. (1999) "Between MDPs and semi-MDPs" 
    - Strategic planning: Bacon et al. (2017) "The Option-Critic Architecture"
    - Reachability integration: Custom design for NPP level completion
    """
    
    def __init__(self, observation_space, action_space, **kwargs):
        # Initialize base exploration components (existing functionality preserved)
        super().__init__(observation_space, action_space, **kwargs)
        
        # Hierarchical planning components (always enabled, no defensive programming)
        # Direct initialization - fail fast if dependencies are missing
        self.reachability_extractor = ReachabilityFeatureExtractor(ReachabilitySystem())
        self.level_analyzer = LevelAnalyzer()
        
        # Subgoal management system for hierarchical planning
        # Based on Options framework from Sutton et al. (1999)
        self.subgoal_cache = {}                    # Cached subgoals for performance optimization
        self.cache_ttl = 200                       # Cache time-to-live in milliseconds  
        self.last_switch_states = {}               # State tracking for dynamic updates
        self.last_ninja_pos = None                 # Position tracking for cache invalidation
        
        # Strategic planning components based on NPP level completion heuristics
        # Implementation follows completion strategies from nclone strategic analysis
        self.completion_planner = LevelCompletionPlanner()
        self.subgoal_prioritizer = SubgoalPrioritizer()
        
        # Performance monitoring for real-time optimization
        # Target: <3ms subgoal generation, >70% cache hit rate
        self.cache_hit_rate = 0.0
        self.avg_subgoal_count = 0.0
        self.planning_time_ms = 0.0
    
    def get_available_subgoals(self, ninja_pos, level_data, switch_states, 
                              max_subgoals=8) -> List[Subgoal]:
        """
        Get hierarchical subgoals based on reachability analysis and strategic planning.
        
        This method implements the core hierarchical RL subgoal extraction following
        the Options framework from Sutton et al. (1999). Subgoals are generated from
        reachability analysis and prioritized based on strategic value for level completion.
        
        Args:
            ninja_pos: Current ninja position tuple (x, y)
            level_data: Level data structure containing objectives, switches, hazards
            switch_states: Dictionary of current switch states {switch_id: bool}
            max_subgoals: Maximum number of subgoals to return (default 8)
        
        Returns:
            List of hierarchical subgoals, prioritized by strategic completion value
            
        Note:
            Performance target is <3ms for real-time HRL training compatibility.
            Uses cached results when available for efficiency optimization.
        """
        start_time = time.perf_counter()
        
        # Check performance-optimized cache for previously computed subgoals
        # Cache implementation follows real-time system design patterns
        cache_key = self._generate_cache_key(ninja_pos, switch_states, level_data)
        if self._is_cache_valid(cache_key):
            cached_result = self.subgoal_cache[cache_key]
            self._update_cache_metrics(True)
            return cached_result['subgoals'][:max_subgoals]
        
        # Extract compact reachability features using nclone integration
        # Uses ReachabilitySystem for performance-optimized analysis
        reachability_features = self.reachability_extractor.extract_features(
            ninja_pos, level_data, switch_states, performance_target="balanced"
        )
        
        # Generate hierarchical subgoals from reachability analysis
        # Implementation based on strategic level completion heuristics
        subgoals = self._generate_subgoals(
            ninja_pos, level_data, switch_states, reachability_features
        )
        
        # Prioritize subgoals using strategic value computation
        # Prioritization follows completion-oriented planning principles
        prioritized_subgoals = self.subgoal_prioritizer.prioritize(
            subgoals, ninja_pos, level_data, reachability_features
        )
        
        # Cache result for future performance optimization
        self._cache_subgoals(cache_key, prioritized_subgoals, reachability_features)
        
        # Update performance metrics for monitoring and optimization
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
        # Generate switch-focused subgoals from fast reachability analysis
        # Uses neural network output (graph transformer + 3D CNN + MLPs) rather than expensive physics
        subgoals = []
        
        # Extract compact features from neural reachability analysis
        # Trust the sophisticated neural architecture output rather than recalculating physics
        objective_distances = reachability_features[0:8].numpy()
        switch_features = reachability_features[8:24].numpy()
        
        # Generate navigation subgoals to key objectives (exit, switches)
        # Focus on level completion path rather than general exploration
        navigation_subgoals = self._generate_navigation_subgoals(
            ninja_pos, level_data, objective_distances
        )
        subgoals.extend(navigation_subgoals)
        
        # Generate switch activation subgoals (primary focus for NPP level completion)
        # Switch-based subgoals are the core of the strategic completion heuristic
        switch_subgoals = self._generate_switch_subgoals(
            level_data, switch_states, switch_features
        )
        subgoals.extend(switch_subgoals)
        
        # Note: Hazard avoidance subgoals removed - focusing on switch-based completion strategy
        # The neural reachability analysis already incorporates hazard accessibility in its features
        
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
    
```

### Subgoal Types and Structures
**Add to existing agent module** rather than creating separate files

**Target File**: Add to existing `/npp_rl/agents/adaptive_exploration.py`

```python
@dataclass  
class Subgoal(ABC):
    """
    Base class for hierarchical subgoals in the Options framework.
    
    This abstract base class defines the interface for all hierarchical subgoals
    used in the reachability-guided HRL system. Implementation follows the Options
    framework from Sutton et al. (1999) adapted for NPP level completion.
    
    Subgoals represent temporally extended actions with clear termination conditions
    and progress measurement capabilities for reward shaping integration.
    
    References:
    - Options framework: Sutton et al. (1999) "Between MDPs and semi-MDPs"
    - Hierarchical planning: Bacon et al. (2017) "The Option-Critic Architecture"
    - NPP-specific implementation: Custom design for level completion objectives
    
    Note:
        All subgoal classes should be integrated into the existing adaptive_exploration.py
        module to maintain architectural cohesion and avoid module proliferation.
    """
    priority: float              # Strategic priority for subgoal selection (0.0-1.0)
    estimated_time: float        # Estimated completion time in seconds
    success_probability: float   # Likelihood of successful completion (0.0-1.0)
    
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

```

### Level Completion Planner
**Target File**: Add to existing `/npp_rl/agents/adaptive_exploration.py`

```python
class LevelCompletionPlanner:
    """
    Strategic planner for hierarchical level completion using fast reachability analysis.
    
    This planner implements the production-ready NPP level completion heuristic that leverages
    the sophisticated neural architecture (graph transformer + 3D CNN + MLPs) rather than
    expensive physics calculations. The strategy focuses on systematic switch activation
    sequences following the definitive NPP level completion algorithm.
    
    NPP Level Completion Strategy (Production Implementation):
    1. Check if exit door switch is reachable using neural reachability features
       - If reachable: trigger exit door switch, proceed to step 2
       - If not reachable: find nearest reachable locked door switch, trigger it, return to step 1
    2. Check if exit door is reachable using neural reachability analysis
       - If reachable: navigate to exit door and complete level
       - If not reachable: find nearest reachable locked door switch, trigger it, return to step 2
    
    Performance Optimization:
    - Avoids expensive physics calculations in favor of neural reachability features
    - Trusts graph transformer + CNN + MLP output for spatial reasoning
    - Maintains <3ms planning target through fast feature-based decisions
    - Removes complex hazard avoidance in favor of switch-focused strategy
    
    References:
    - Strategic analysis: nclone reachability analysis integration strategy
    - Hierarchical planning: Sutton et al. (1999) "Between MDPs and semi-MDPs"  
    - Strategic RL: Bacon et al. (2017) "The Option-Critic Architecture"
    
    Note:
        Integrated into existing adaptive_exploration.py to avoid creating
        redundant planning modules and maintain architectural coherence.
    """
    
    def __init__(self):
        self.path_analyzer = PathAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def plan_completion(self, ninja_pos, level_data, switch_states, 
                       reachability_extractor) -> 'CompletionStrategy':
        """
        Generate strategic plan for NPP level completion using production-ready algorithm.
        
        Implementation uses fast neural reachability analysis rather than expensive
        physics calculations. Relies on graph transformer + 3D CNN + MLP features
        for spatial reasoning and switch accessibility determination.
        
        NPP Level Completion Algorithm (Production Implementation):
        1. Check if exit door switch is reachable using neural reachability features
           - If reachable: create subgoal to trigger exit door switch, proceed to step 2
           - If not reachable: find nearest reachable locked door switch, create activation subgoal, return to step 1
        2. Check if exit door is reachable using neural reachability analysis
           - If reachable: create navigation subgoal to exit door for level completion
           - If not reachable: find nearest reachable locked door switch, create activation subgoal, return to step 2
        
        This algorithm ensures systematic progression through switch dependencies for level completion.
        """
        # Extract neural reachability features - production-ready feature extraction
        # Trust the sophisticated graph transformer + CNN + MLP architecture
        reachability_features = reachability_extractor.extract_features(
            ninja_pos, level_data, switch_states, performance_target="balanced"
        )
        
        # Identify level objectives using production-ready level analysis
        exit_door = self._find_exit_door(level_data)
        exit_switch = self._find_exit_switch(level_data)
        
        if not exit_door or not exit_switch:
            return CompletionStrategy([], "No exit found", 0.0)
        
        # Implement NPP Level Completion Algorithm (Production Implementation)
        completion_steps = []
        current_state = "check_exit_switch"
        
        while current_state != "complete":
            if current_state == "check_exit_switch":
                # Step 1: Check if exit door switch is reachable
                exit_switch_reachable = self._is_objective_reachable(
                    exit_switch.position, reachability_features
                )
                
                if exit_switch_reachable and not switch_states.get(exit_switch.id, False):
                    # Exit switch is reachable - create activation subgoal
                    completion_steps.append(CompletionStep(
                        action_type='navigate_and_activate',
                        target_position=exit_switch.position,
                        target_id=exit_switch.id,
                        description=f"Activate exit door switch at {exit_switch.position}",
                        priority=1.0
                    ))
                    current_state = "check_exit_door"
                    
                elif not exit_switch_reachable:
                    # Exit switch not reachable - find nearest reachable locked door switch
                    nearest_switch = self._find_nearest_reachable_locked_door_switch(
                        ninja_pos, level_data, switch_states, reachability_features
                    )
                    
                    if nearest_switch:
                        completion_steps.append(CompletionStep(
                            action_type='navigate_and_activate',
                            target_position=nearest_switch.position,
                            target_id=nearest_switch.id,
                            description=f"Activate blocking switch {nearest_switch.id} at {nearest_switch.position}",
                            priority=0.8
                        ))
                        # Return to step 1 after activating blocking switch
                        current_state = "check_exit_switch"
                    else:
                        # No reachable switches found - level may be impossible
                        current_state = "complete"
                else:
                    # Exit switch already activated
                    current_state = "check_exit_door"
            
            elif current_state == "check_exit_door":
                # Step 2: Check if exit door is reachable
                exit_door_reachable = self._is_objective_reachable(
                    exit_door.position, reachability_features
                )
                
                if exit_door_reachable:
                    # Exit door is reachable - create navigation subgoal for level completion
                    completion_steps.append(CompletionStep(
                        action_type='navigate_to_exit',
                        target_position=exit_door.position,
                        target_id=exit_door.id,
                        description=f"Navigate to exit door at {exit_door.position}",
                        priority=1.0
                    ))
                    current_state = "complete"
                    
                else:
                    # Exit door not reachable - find nearest reachable locked door switch
                    nearest_switch = self._find_nearest_reachable_locked_door_switch(
                        ninja_pos, level_data, switch_states, reachability_features
                    )
                    
                    if nearest_switch:
                        completion_steps.append(CompletionStep(
                            action_type='navigate_and_activate',
                            target_position=nearest_switch.position,
                            target_id=nearest_switch.id,
                            description=f"Activate blocking switch {nearest_switch.id} at {nearest_switch.position}",
                            priority=0.8
                        ))
                        # Return to step 2 after activating blocking switch
                        current_state = "check_exit_door"
                    else:
                        # No reachable switches found - level may be impossible
                        current_state = "complete"
        
        # Calculate confidence using production-ready feature analysis
        confidence = self._calculate_strategy_confidence_from_features(
            completion_steps, reachability_features
        )
        
        return CompletionStrategy(
            steps=completion_steps,
            description="NPP Level Completion Strategy (Production Implementation)",
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

### Core Hierarchical Integration
**Objective**: Integrate hierarchical reachability planning into existing agent architecture

**Approach**:
- **Modify existing adaptive exploration module in place** rather than creating separate hierarchical managers
- Extend existing exploration classes with hierarchical subgoal functionality while maintaining backward compatibility
- Follow top-level import patterns and comprehensive documentation standards

**Key Modifications**:
1. **Adaptive Exploration Manager**: Extend existing `AdaptiveExplorationManager` in `/npp_rl/agents/adaptive_exploration.py`
2. **Subgoal Framework**: Add all subgoal classes to same file for architectural cohesion
3. **Strategic Planning**: Integrate `LevelCompletionPlanner` and related components into existing module

**Documentation Requirements**:
- Add comprehensive module-level docstrings with hierarchical RL theoretical foundations
- Include inline documentation explaining reachability-guided subgoal generation and strategic planning
- Reference all relevant research papers (Sutton et al. 1999, Bacon et al. 2017, etc.) in docstrings and comments
- Document nclone integration points and performance optimization strategies

### HRL Environment Integration
**Objective**: Extend existing environment wrappers to support hierarchical RL

**Approach**:
- **Modify existing environment wrappers** rather than creating separate HRL wrappers  
- Extend existing observation and reward systems with hierarchical subgoal information
- Implement efficient subgoal reward shaping integrated with existing reward processing

**Target Environment Wrapper**: Modify existing environment wrappers

**Implementation Example** (extend existing environment wrapper):
```python
# Modify existing NPPEnv or environment wrapper in npp_rl/environments/

class NPPEnv:  # or existing wrapper class
    """
    NPP Environment with integrated hierarchical reachability-guided subgoal support.
    
    This environment extends the existing NPP gameplay with hierarchical RL capabilities
    that provide strategic subgoals based on reachability analysis. Subgoal information
    is provided as part of the observation space and reward system.
    
    Hierarchical Integration Components:
    - Subgoal observation space extension for hierarchical policy training
    - Reward shaping integration with subgoal progress tracking
    - Dynamic subgoal updates based on environment state changes
    - Performance-optimized subgoal caching for real-time gameplay
    
    References:
    - Hierarchical RL: Sutton et al. (1999) "Between MDPs and semi-MDPs"
    - Environment design: Gym environment standards and best practices
    - Performance optimization: Real-time RL training requirements
    """
    
    def __init__(self, max_subgoals=8, subgoal_reward_scale=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.max_subgoals = max_subgoals
        self.subgoal_reward_scale = subgoal_reward_scale
        
        # Initialize hierarchical planning components (always enabled, fail fast if missing)
        # Direct initialization without defensive programming - dependencies must be available
        self.hierarchical_manager = AdaptiveExplorationManager(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
        
        # Hierarchical state tracking for subgoal management
        # Implementation based on Options framework from Sutton et al. (1999)
        self.current_subgoals = []           # Currently available subgoals from reachability analysis
        self.completion_strategy = None      # Strategic level completion plan
        self.active_subgoal = None          # Currently pursued subgoal for reward shaping
        
        # Extend observation space to include hierarchical subgoal information
        # This enables hierarchical policies to condition on current subgoals
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
        """Update available subgoals using integrated hierarchical manager."""
        # Get available subgoals from integrated hierarchical planning system
        # Uses reachability-guided subgoal generation with strategic prioritization  
        self.current_subgoals = self.hierarchical_manager.get_available_subgoals(
            ninja_pos, level_data, switch_states, self.max_subgoals
        )
        
        # Update strategic level completion plan using integrated planner
        # Implementation follows nclone strategic analysis completion heuristics
        self.completion_strategy = self.hierarchical_manager.get_completion_strategy(
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

### Testing and Optimization
**Objective**: Validate hierarchical integration and optimize performance

**Testing Approach**:
- Unit tests for hierarchical subgoal generation and strategic planning components
- Integration tests validating hierarchical RL with existing training pipeline
- Performance tests ensuring <3ms subgoal generation and >70% cache hit rate

**Optimization Focus**:
- Memory efficiency for subgoal caching and completion strategy management
- Computational optimization for real-time hierarchical planning
- Integration efficiency with existing exploration and reward systems

**Key Modifications**:
1. **Test Framework**: Extend existing test suites to include hierarchical components
2. **Performance Monitoring**: Add hierarchical metrics to existing monitoring systems  
3. **Integration Validation**: Ensure seamless integration with existing agent architecture

## Testing Strategy

### Unit Tests
**Test existing modified classes with hierarchical integration**

```python
class TestAdaptiveExplorationManagerWithHierarchical(unittest.TestCase):
    """Test hierarchical integration in the modified AdaptiveExplorationManager."""
    
    def setUp(self):
        self.observation_space = create_mock_obs_space()
        self.action_space = create_mock_action_space()
        
        # Test the modified existing manager with hierarchical features integrated
        self.manager = AdaptiveExplorationManager(
            observation_space=self.observation_space,
            action_space=self.action_space
        )
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
    """Test hierarchical RL integration with modified existing components."""
    
    def setUp(self):
        # Test with modified existing environment with hierarchical features
        self.env = NPPEnv(
            render_mode='rgb_array',
            max_subgoals=8,
            subgoal_reward_scale=0.1
        )
    
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
- **Production-Ready Implementation**: All components fully functional without placeholder code or simplified implementations
- **Algorithm Compliance**: Level completion must follow the exact two-phase NPP heuristic (exit switch → exit door)
- **Neural Feature Integration**: Complete integration with graph transformer + 3D CNN + MLP reachability analysis
- **Strategic Effectiveness**: Systematic switch activation sequences leading to measurable level completion improvements
- **Dynamic Adaptation**: Real-time subgoal updates following switch state changes without performance degradation

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

1. **Modified AdaptiveExplorationManager**: Existing exploration manager extended with hierarchical reachability-guided planning
2. **Integrated Subgoal Framework**: All hierarchical subgoal types and strategic planning integrated into existing modules
3. **Enhanced Environment Support**: Existing environment wrappers updated with hierarchical RL subgoal information
4. **Training Pipeline Integration**: Complete integration with existing PPO training systems
5. **Performance Analysis**: Comprehensive evaluation of hierarchical benefits within existing architecture
6. **Integration Testing**: Test suites validating hierarchical integration in existing components

**Key Principles**: 
- All modifications integrate seamlessly with existing architecture while providing hierarchical capabilities
- Implementation must be production-ready with complete functionality (no placeholders or simplified code)
- Level completion algorithm must follow the exact two-phase NPP heuristic specification

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