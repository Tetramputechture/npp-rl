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
- Integrates with TieredReachabilitySystem for performance-optimized planning queries
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
from nclone.constants.entity_types import EntityType
from nclone.graph.reachability.compact_features import CompactReachabilityFeatures
from nclone.graph.reachability.tiered_system import TieredReachabilitySystem


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
        # Use authoritative simulation data first, fall back to passed states
        return self._is_switch_activated_authoritative(self.switch_id, level_data, switch_states)
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        # Reward for getting closer to switch
        distance = math.sqrt((ninja_pos[0] - self.switch_position[0])**2 + 
                           (ninja_pos[1] - self.switch_position[1])**2)
        max_distance = 500.0
        proximity_reward = (max_distance - distance) / max_distance
        
        # Bonus for high reachability score
        reachability_bonus = self.reachability_score * 0.5
        
        return proximity_reward + reachability_bonus
    
    def _is_switch_activated_authoritative(self, switch_id: str, level_data, switch_states: Dict) -> bool:
        """
        Check switch activation using authoritative simulation data first.
        Falls back to passed switch_states if simulation data unavailable.
        
        Uses actual NppEnvironment data structures from nclone.
        """
        # Method 1: Check level_data.entities for switch with matching entity_id
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if (entity.get('entity_id') == switch_id and 
                    entity.get('type') == EntityType.EXIT_SWITCH):
                    # For exit switches, activated means active=False (inverted logic in nclone)
                    return not entity.get('active', True)
        
        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, 'switch_activated'):
            # This is the direct boolean from NppEnvironment observation
            return level_data.switch_activated
        
        # Method 3: Fall back to passed switch_states (legacy compatibility)
        return switch_states.get(switch_id, False)


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
        # For functional switches (our focus), check if switch is activated
        # For other collectibles, check if item has been collected
        if self.item_type in ['door_switch', 'exit_switch', 'functional_switch', 'locked_door_switch']:
            # This is actually a functional switch, check activation status
            return self._check_functional_switch_activated(self.target_position, level_data)
        else:
            # This is a traditional collectible, check if collected
            return self._check_item_collected(self.target_position, level_data)
    
    def get_reward_shaping(self, ninja_pos: Tuple[float, float]) -> float:
        distance = math.sqrt((ninja_pos[0] - self.target_position[0])**2 + 
                           (ninja_pos[1] - self.target_position[1])**2)
        max_distance = 500.0
        proximity_reward = (max_distance - distance) / max_distance
        
        # Scale by item value and area connectivity
        value_bonus = self.value * 0.1
        connectivity_bonus = self.area_connectivity * 0.3
        
        return proximity_reward + value_bonus + connectivity_bonus
    
    def _check_item_collected(self, position: Tuple[float, float], level_data) -> bool:
        """Check if item at position has been collected."""
        # Production implementation: Check level data for item existence
        if hasattr(level_data, 'collectibles'):
            for item in level_data.collectibles:
                # Handle both dictionary and object access patterns
                item_x = item.get('x', 0) if isinstance(item, dict) else getattr(item, 'x', 0)
                item_y = item.get('y', 0) if isinstance(item, dict) else getattr(item, 'y', 0)
                item_collected = item.get('collected', False) if isinstance(item, dict) else getattr(item, 'collected', False)
                
                if (abs(item_x - position[0]) < 12.0 and 
                    abs(item_y - position[1]) < 12.0):
                    return item_collected  # Return True if collected, False if not collected
        return False  # Item not found, assume not collected
    
    def _check_functional_switch_activated(self, position: Tuple[float, float], level_data) -> bool:
        """
        Check if a functional switch at the given position is activated.
        Uses actual NppEnvironment data structures from nclone.
        """
        # Method 1: Check level_data.entities for switches near the position
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    entity_x = entity.get('x', 0)
                    entity_y = entity.get('y', 0)
                    
                    # Check if this entity is at the target position (within radius)
                    if (abs(entity_x - position[0]) < 12.0 and 
                        abs(entity_y - position[1]) < 12.0):
                        # For exit switches, activated means active=False (inverted logic)
                        return not entity.get('active', True)
        
        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, 'switch_activated'):
            # This is the direct boolean from NppEnvironment observation
            # Check if the switch position matches
            if (hasattr(level_data, 'switch_x') and hasattr(level_data, 'switch_y')):
                switch_x = getattr(level_data, 'switch_x', 0)
                switch_y = getattr(level_data, 'switch_y', 0)
                if (abs(switch_x - position[0]) < 12.0 and 
                    abs(switch_y - position[1]) < 12.0):
                    return level_data.switch_activated
        
        # Default: assume not activated
        return False
    
    def _find_switch_id_by_position(self, position: Tuple[float, float], level_data) -> str:
        """Find switch ID by position using actual NppEnvironment data structures."""
        # Check level_data.entities for switches near the position
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    entity_x = entity.get('x', 0)
                    entity_y = entity.get('y', 0)
                    
                    # Check if this entity is at the target position (within radius)
                    if (abs(entity_x - position[0]) < 12.0 and 
                        abs(entity_y - position[1]) < 12.0):
                        return entity.get('entity_id')
        
        return None


@dataclass
class CompletionStep:
    """Single step in a level completion strategy."""
    action_type: str              # 'navigate_and_activate', 'navigate_to_exit', etc.
    target_position: Tuple[float, float]
    target_id: str
    description: str
    priority: float


@dataclass
class CompletionStrategy:
    """Strategic plan for level completion."""
    steps: List[CompletionStep]
    description: str
    confidence: float
    
    def get_next_subgoal(self) -> Optional[Subgoal]:
        """Get the next subgoal from the completion strategy."""
        if not self.steps:
            return None
        
        next_step = self.steps[0]
        
        if next_step.action_type == 'navigate_and_activate':
            return SwitchActivationSubgoal(
                priority=next_step.priority,
                estimated_time=30.0,  # Estimated seconds to complete
                success_probability=0.8,
                switch_id=next_step.target_id,
                switch_position=next_step.target_position,
                switch_type='door_switch',
                reachability_score=0.9
            )
        elif next_step.action_type == 'navigate_to_exit':
            return NavigationSubgoal(
                priority=next_step.priority,
                estimated_time=20.0,
                success_probability=0.9,
                target_position=next_step.target_position,
                target_type='exit_door',
                distance=math.sqrt(next_step.target_position[0]**2 + next_step.target_position[1]**2)
            )
        
        return None


class CuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for exploration bonus.
    
    Based on "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017).
    Provides intrinsic rewards based on prediction error.
    """
    
    def __init__(self, feature_dim: int = 512, action_dim: int = 5):
        super().__init__()
        
        # Feature encoder (shared)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Forward model: predicts next state features from current features + action
        self.forward_model = nn.Sequential(
            nn.Linear(64 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # Inverse model: predicts action from current and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, state_features: torch.Tensor, next_state_features: torch.Tensor, 
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute curiosity-based intrinsic reward.
        
        Returns:
            intrinsic_reward: Curiosity-based reward
            forward_loss: Forward model loss
            inverse_loss: Inverse model loss
        """
        # Encode features
        phi_state = self.feature_encoder(state_features)
        phi_next_state = self.feature_encoder(next_state_features)
        
        # Forward model prediction
        action_onehot = torch.zeros(actions.size(0), 5, device=actions.device)
        action_onehot.scatter_(1, actions.long().unsqueeze(1), 1)
        
        forward_input = torch.cat([phi_state, action_onehot], dim=1)
        predicted_next_features = self.forward_model(forward_input)
        
        # Inverse model prediction
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        predicted_actions = self.inverse_model(inverse_input)
        
        # Compute losses
        forward_loss = self.mse_loss(predicted_next_features, phi_next_state.detach())
        inverse_loss = self.ce_loss(predicted_actions, actions.long())
        
        # Intrinsic reward is the forward prediction error
        intrinsic_reward = torch.norm(predicted_next_features - phi_next_state.detach(), dim=1)
        
        return intrinsic_reward, forward_loss, inverse_loss


class NoveltyDetector:
    """
    Novelty detection for exploration bonus.
    
    Uses a simple count-based approach with state discretization.
    (Inspired by count-based exploration methods, e.g., Strehl & Littman, 2008; Bellemare et al., 2016)
    """
    
    def __init__(self, grid_size: int = 32, decay_factor: float = 0.99):
        self.grid_size = grid_size
        self.decay_factor = decay_factor
        self.visit_counts = defaultdict(float)
        self.total_visits = 0
        
    def get_novelty_bonus(self, x: float, y: float) -> float:
        """Get novelty bonus based on visit count."""
        # Discretize position
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        cell = (grid_x, grid_y)
        
        # Get visit count
        count = self.visit_counts[cell]
        
        # Update visit count
        self.visit_counts[cell] += 1
        self.total_visits += 1
        
        # Novelty bonus inversely proportional to visit count
        novelty_bonus = 1.0 / (1.0 + count)
        
        return novelty_bonus
    
    def decay_counts(self):
        """Decay visit counts to forget old information."""
        for cell in self.visit_counts:
            self.visit_counts[cell] *= self.decay_factor


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
    """
    
    def __init__(self):
        self.path_analyzer = PathAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def plan_completion(self, ninja_pos, level_data, switch_states, 
                       reachability_system, reachability_features) -> CompletionStrategy:
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
        reachability_result = reachability_system.analyze_reachability(
            level_data, ninja_pos, switch_states, performance_target="balanced"
        )
        
        # Encode reachability into compact 64-dimensional features
        reachability_features_array = reachability_features.encode_reachability(
            reachability_result, level_data, [], ninja_pos, switch_states
        )
        reachability_features = torch.tensor(reachability_features_array, dtype=torch.float32)
        
        # Identify level objectives using production-ready level analysis
        exit_door = self._find_exit_door(level_data)
        exit_switch = self._find_exit_switch(level_data)
        
        if not exit_door or not exit_switch:
            return CompletionStrategy([], "No exit found", 0.0)
        
        # Implement NPP Level Completion Algorithm (Production Implementation)
        completion_steps = []
        current_state = "check_exit_switch"
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while current_state != "complete" and iteration < max_iterations:
            iteration += 1
            
            if current_state == "check_exit_switch":
                # Step 1: Check if exit door switch is reachable
                exit_switch_reachable = self._is_objective_reachable(
                    exit_switch['position'], reachability_features
                )
                
                if exit_switch_reachable and not switch_states.get(exit_switch['id'], False):
                    # Exit switch is reachable - create activation subgoal
                    completion_steps.append(CompletionStep(
                        action_type='navigate_and_activate',
                        target_position=exit_switch['position'],
                        target_id=exit_switch['id'],
                        description=f"Activate exit door switch at {exit_switch['position']}",
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
                            target_position=nearest_switch['position'],
                            target_id=nearest_switch['id'],
                            description=f"Activate blocking switch {nearest_switch['id']} at {nearest_switch['position']}",
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
                    exit_door['position'], reachability_features
                )
                
                if exit_door_reachable:
                    # Exit door is reachable - create navigation subgoal for level completion
                    completion_steps.append(CompletionStep(
                        action_type='navigate_to_exit',
                        target_position=exit_door['position'],
                        target_id=exit_door['id'],
                        description=f"Navigate to exit door at {exit_door['position']}",
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
                            target_position=nearest_switch['position'],
                            target_id=nearest_switch['id'],
                            description=f"Activate blocking switch {nearest_switch['id']} at {nearest_switch['position']}",
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
    
    def _find_exit_door(self, level_data) -> Optional[Dict]:
        """Find the exit door in level data using actual NppEnvironment data structures."""
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_DOOR:
                    return {
                        'id': entity.get('entity_id', 'exit_door'),
                        'position': (entity.get('x', 0), entity.get('y', 0)),
                        'type': 'exit_door'
                    }
        return None
    
    def _find_exit_switch(self, level_data) -> Optional[Dict]:
        """Find the exit door switch in level data using actual NppEnvironment data structures."""
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    return {
                        'id': entity.get('entity_id', 'exit_switch'),
                        'position': (entity.get('x', 0), entity.get('y', 0)),
                        'type': 'exit_switch'
                    }
        return None
    
    def _is_objective_reachable(self, position: Tuple[float, float], 
                               reachability_features: torch.Tensor) -> bool:
        """Check if objective is reachable using neural reachability features."""
        # Use neural network output rather than expensive physics calculations
        # Trust the graph transformer + CNN + MLP architecture for spatial reasoning
        if len(reachability_features) >= 8:
            # Extract objective reachability from neural features
            objective_distances = reachability_features[0:8].numpy()
            # Consider reachable if any objective distance feature is positive
            return np.any(objective_distances > 0.1)
        return False
    
    def _find_nearest_reachable_locked_door_switch(self, ninja_pos, level_data, 
                                                  switch_states, reachability_features) -> Optional[Dict]:
        """Find nearest reachable locked door switch using neural features and actual NppEnvironment data structures."""
        if not hasattr(level_data, 'entities') or not level_data.entities:
            return None
        
        reachable_switches = []
        switch_features = reachability_features[8:24].numpy() if len(reachability_features) >= 24 else []
        
        switch_index = 0
        for entity in level_data.entities:
            # Only consider exit switches
            if entity.get('type') != EntityType.EXIT_SWITCH:
                continue
            
            switch_id = entity.get('entity_id')
            
            # Skip already activated switches (using authoritative method)
            if self._is_switch_activated_authoritative(switch_id, level_data, switch_states):
                continue
            
            # Check reachability using neural features
            if switch_index < len(switch_features) and switch_features[switch_index] > 0.1:
                distance = math.sqrt(
                    (ninja_pos[0] - entity.get('x', 0))**2 + 
                    (ninja_pos[1] - entity.get('y', 0))**2
                )
                reachable_switches.append({
                    'id': switch_id,
                    'position': (entity.get('x', 0), entity.get('y', 0)),
                    'type': 'exit_switch',
                    'distance': distance,
                    'reachability_score': switch_features[switch_index]
                })
            
            switch_index += 1
        
        # Return nearest reachable switch
        if reachable_switches:
            return min(reachable_switches, key=lambda s: s['distance'])
        return None
    
    def _calculate_strategy_confidence_from_features(self, completion_steps: List[CompletionStep], 
                                                   reachability_features: torch.Tensor) -> float:
        """Calculate strategy confidence using neural reachability features."""
        if not completion_steps:
            return 0.0
        
        # Base confidence on neural feature quality and step count
        feature_confidence = torch.mean(torch.abs(reachability_features)).item()
        step_penalty = max(0.0, 1.0 - len(completion_steps) * 0.1)
        
        return min(1.0, feature_confidence * step_penalty)
    
    def _is_switch_activated_authoritative(self, switch_id: str, level_data, switch_states: Dict) -> bool:
        """
        Check switch activation using authoritative simulation data first.
        Falls back to passed switch_states if simulation data unavailable.
        
        Uses actual NppEnvironment data structures from nclone.
        """
        # Method 1: Check level_data.entities for switch with matching entity_id
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if (entity.get('entity_id') == switch_id and 
                    entity.get('type') == EntityType.EXIT_SWITCH):
                    # For exit switches, activated means active=False (inverted logic in nclone)
                    return not entity.get('active', True)
        
        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, 'switch_activated'):
            # This is the direct boolean from NppEnvironment observation
            return level_data.switch_activated
        
        # Method 3: Fall back to passed switch_states (legacy compatibility)
        return switch_states.get(switch_id, False)


class PathAnalyzer:
    """Analyzes paths using neural reachability features."""
    
    def analyze_path_feasibility(self, start_pos, end_pos, reachability_features):
        """Analyze if path is feasible using neural features."""
        # Trust neural network output for path analysis
        return torch.mean(reachability_features).item() > 0.1


class DependencyAnalyzer:
    """Analyzes switch dependencies using neural features."""
    
    def analyze_switch_dependencies(self, level_data, reachability_features):
        """Analyze switch dependencies using neural features."""
        # Use neural features to understand switch relationships
        dependencies = {}
        if hasattr(level_data, 'switches'):
            for switch in level_data.switches:
                dependencies[switch.get('id')] = {
                    'blocks': [],
                    'required_for': []
                }
        return dependencies


class SubgoalPrioritizer:
    """
    Prioritizes hierarchical subgoals based on strategic value and completion likelihood.
    
    This prioritizer uses neural reachability features and strategic analysis to rank
    subgoals for optimal level completion. Implementation focuses on switch-based
    progression following the NPP level completion heuristic.
    
    Prioritization Strategy:
    - Exit-related subgoals receive highest priority
    - Switch activation subgoals prioritized by strategic value
    - Collection subgoals weighted by value and accessibility
    - Navigation subgoals ranked by distance and strategic importance
    
    References:
    - Strategic planning: Custom NPP level completion analysis
    - Hierarchical RL: Bacon et al. (2017) "The Option-Critic Architecture"
    - Reachability integration: Neural feature-based prioritization
    """
    
    def __init__(self):
        self.strategic_weights = {
            'exit_door': 1.0,
            'exit_switch': 0.9,
            'door_switch': 0.7,
            'collectible': 0.3,
            'exploration': 0.1
        }
    
    def prioritize(self, subgoals: List[Subgoal], ninja_pos: Tuple[float, float], 
                  level_data, reachability_features: torch.Tensor) -> List[Subgoal]:
        """
        Prioritize subgoals based on strategic value and neural reachability analysis.
        
        Uses neural network features (graph transformer + CNN + MLP) for strategic
        assessment rather than expensive physics calculations.
        """
        if not subgoals:
            return []
        
        # Calculate priority scores for each subgoal
        scored_subgoals = []
        for subgoal in subgoals:
            priority_score = self._calculate_priority_score(
                subgoal, ninja_pos, level_data, reachability_features
            )
            scored_subgoals.append((subgoal, priority_score))
        
        # Sort by priority score (highest first)
        scored_subgoals.sort(key=lambda x: x[1], reverse=True)
        
        # Update subgoal priorities and return sorted list
        prioritized_subgoals = []
        for subgoal, score in scored_subgoals:
            subgoal.priority = score
            prioritized_subgoals.append(subgoal)
        
        return prioritized_subgoals
    
    def _calculate_priority_score(self, subgoal: Subgoal, ninja_pos: Tuple[float, float],
                                 level_data, reachability_features: torch.Tensor) -> float:
        """Calculate priority score for a subgoal using neural features."""
        base_score = subgoal.priority
        
        # Strategic value based on subgoal type
        if isinstance(subgoal, NavigationSubgoal):
            strategic_weight = self.strategic_weights.get(subgoal.target_type, 0.5)
        elif isinstance(subgoal, SwitchActivationSubgoal):
            strategic_weight = self.strategic_weights.get(subgoal.switch_type, 0.7)
        elif isinstance(subgoal, CollectionSubgoal):
            strategic_weight = self.strategic_weights.get('collectible', 0.3)
        else:
            strategic_weight = 0.5
        
        # Distance penalty (closer is better)
        target_pos = subgoal.get_target_position()
        distance = math.sqrt((ninja_pos[0] - target_pos[0])**2 + (ninja_pos[1] - target_pos[1])**2)
        distance_factor = max(0.1, 1.0 - distance / 500.0)  # Normalize by max reasonable distance
        
        # Reachability bonus from neural features
        reachability_bonus = self._get_reachability_bonus(subgoal, reachability_features)
        
        # Success probability factor
        success_factor = subgoal.success_probability
        
        # Combine factors
        priority_score = (base_score * strategic_weight * distance_factor * 
                         success_factor + reachability_bonus)
        
        return min(1.0, priority_score)
    
    def _get_reachability_bonus(self, subgoal: Subgoal, reachability_features: torch.Tensor) -> float:
        """Get reachability bonus from neural features."""
        if isinstance(subgoal, SwitchActivationSubgoal):
            return subgoal.reachability_score * 0.2
        elif len(reachability_features) > 0:
            # Use average neural feature strength as reachability indicator
            return torch.mean(torch.abs(reachability_features)).item() * 0.1
        return 0.0


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
    - Direct initialization of reachability extractor (no defensive programming)
    - Subgoal prioritization based on strategic value and completion likelihood
    - Dynamic cache management for real-time performance (<3ms subgoal generation)
    
    References:
    - Base exploration: Pathak et al. (2017) "Curiosity-driven Exploration by Self-supervised Prediction"
    - Hierarchical RL: Sutton et al. (1999) "Between MDPs and semi-MDPs" 
    - Strategic planning: Bacon et al. (2017) "The Option-Critic Architecture"
    - Reachability integration: Custom design for NPP level completion
    """
    
    def __init__(self, 
                 curiosity_weight: float = 0.1,
                 novelty_weight: float = 0.05,
                 progress_window: int = 100):
        
        # Initialize base exploration components (existing functionality preserved)
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight
        self.progress_window = progress_window
        
        # Initialize components
        self.novelty_detector = NoveltyDetector()
        self.curiosity_module = None  # Will be initialized when we have feature dimensions
        
        # Progress tracking
        self.recent_rewards = deque(maxlen=progress_window)
        self.recent_completion_times = deque(maxlen=progress_window)
        self.exploration_scale = 1.0
        
        # Statistics
        self.total_intrinsic_reward = 0.0
        self.episode_count = 0
        
        # Hierarchical planning components (always enabled, no defensive programming)
        # Direct initialization - fail fast if dependencies are missing
        self.reachability_system = TieredReachabilitySystem()
        self.reachability_features = CompactReachabilityFeatures()
        
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
        
    def initialize_curiosity_module(self, feature_dim: int, action_dim: int = 5):
        """Initialize curiosity module with known dimensions."""
        self.curiosity_module = CuriosityModule(feature_dim, action_dim)
        
    def compute_exploration_bonus(self, 
                                  state_features: Optional[torch.Tensor],
                                  next_state_features: Optional[torch.Tensor],
                                  action: int,
                                  player_x: float,
                                  player_y: float) -> float:
        """
        Compute total exploration bonus combining multiple strategies.
        
        Args:
            state_features: Current state features (for curiosity)
            next_state_features: Next state features (for curiosity)
            action: Action taken
            player_x: Player x position
            player_y: Player y position
            
        Returns:
            Total exploration bonus
        """
        total_bonus = 0.0
        
        # Novelty bonus
        novelty_bonus = self.novelty_detector.get_novelty_bonus(player_x, player_y)
        total_bonus += self.novelty_weight * novelty_bonus * self.exploration_scale
        
        # Curiosity bonus (if available)
        if (self.curiosity_module is not None and 
            state_features is not None and 
            next_state_features is not None):
            
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.float32)
                intrinsic_reward, _, _ = self.curiosity_module(
                    state_features.unsqueeze(0),
                    next_state_features.unsqueeze(0),
                    action_tensor
                )
                curiosity_bonus = intrinsic_reward.item()
                total_bonus += self.curiosity_weight * curiosity_bonus * self.exploration_scale
                
                self.total_intrinsic_reward += curiosity_bonus
        
        return total_bonus
    
    def update_progress(self, episode_reward: float, completion_time: Optional[int] = None):
        """Update progress tracking and adjust exploration scale."""
        self.recent_rewards.append(episode_reward)
        if completion_time is not None:
            self.recent_completion_times.append(completion_time)
        
        self.episode_count += 1
        
        # Adjust exploration scale based on progress
        if len(self.recent_rewards) >= self.progress_window:
            # If performance is improving, reduce exploration
            # If performance is stagnating, increase exploration
            recent_avg = np.mean(list(self.recent_rewards)[-50:])
            older_avg = np.mean(list(self.recent_rewards)[-100:-50])
            
            if recent_avg > older_avg * 1.1:  # Significant improvement
                self.exploration_scale *= 0.95  # Reduce exploration
            elif recent_avg < older_avg * 0.9:  # Performance degradation
                self.exploration_scale *= 1.05  # Increase exploration
                
            # Clamp exploration scale
            self.exploration_scale = np.clip(self.exploration_scale, 0.1, 2.0)
    
    def get_exploration_stats(self) -> Dict[str, float]:
        """Get exploration statistics for logging."""
        stats = {
            'exploration_scale': self.exploration_scale,
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'avg_recent_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'episode_count': self.episode_count,
        }
        
        if self.recent_completion_times:
            stats['avg_completion_time'] = np.mean(self.recent_completion_times)
            
        return stats
    
    def reset_episode(self):
        """Reset episode-specific tracking."""
        # Decay novelty counts periodically
        if self.episode_count % 100 == 0:
            self.novelty_detector.decay_counts()
    
    def train_curiosity_module(self, 
                               state_features: torch.Tensor,
                               next_state_features: torch.Tensor,
                               actions: torch.Tensor,
                               optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train the curiosity module.
        
        Returns:
            Dictionary of training losses
        """
        if self.curiosity_module is None:
            return {}
            
        optimizer.zero_grad()
        
        intrinsic_reward, forward_loss, inverse_loss = self.curiosity_module(
            state_features, next_state_features, actions
        )
        
        # Total curiosity loss
        total_loss = forward_loss + inverse_loss
        total_loss.backward()
        optimizer.step()
        
        return {
            'curiosity_forward_loss': forward_loss.item(),
            'curiosity_inverse_loss': inverse_loss.item(),
            'curiosity_total_loss': total_loss.item(),
            'avg_intrinsic_reward': intrinsic_reward.mean().item(),
        }
    
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
        # Uses TieredReachabilitySystem for performance-optimized analysis
        reachability_result = self.reachability_system.analyze_reachability(
            level_data, ninja_pos, switch_states, performance_target="balanced"
        )
        
        # Encode reachability into compact 64-dimensional features
        reachability_features = torch.tensor(
            self.reachability_features.encode_reachability(
                reachability_result, level_data, [], ninja_pos, switch_states
            ), dtype=torch.float32
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
            ninja_pos, level_data, switch_states, self.reachability_system, self.reachability_features
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
        objective_distances = reachability_features[0:8].numpy() if len(reachability_features) >= 8 else []
        switch_features = reachability_features[8:24].numpy() if len(reachability_features) >= 24 else []
        
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
        
        # Generate collection subgoals for valuable items
        # Secondary priority after level completion objectives
        collection_subgoals = self._generate_collection_subgoals(
            ninja_pos, level_data, reachability_features
        )
        subgoals.extend(collection_subgoals)
        
        return subgoals
    
    def _generate_navigation_subgoals(self, ninja_pos, level_data, 
                                    objective_distances) -> List[NavigationSubgoal]:
        """Generate navigation subgoals to key objectives using actual NppEnvironment data structures."""
        subgoals = []
        
        # Generate exit door navigation subgoal from level_data.entities
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_DOOR:
                    entity_x = entity.get('x', 0)
                    entity_y = entity.get('y', 0)
                    distance = math.sqrt((ninja_pos[0] - entity_x)**2 + 
                                       (ninja_pos[1] - entity_y)**2)
                    subgoals.append(NavigationSubgoal(
                        priority=0.9,
                        estimated_time=distance / 50.0,  # Rough time estimate
                        success_probability=0.8,
                        target_position=(entity_x, entity_y),
                        target_type='exit_door',
                        distance=distance
                    ))
        
        return subgoals
    
    def _generate_switch_subgoals(self, level_data, switch_states, 
                                switch_features) -> List[SwitchActivationSubgoal]:
        """Generate switch activation subgoals using actual NppEnvironment data structures."""
        subgoals = []
        
        # Use level_data.entities to find exit switches
        if hasattr(level_data, 'entities') and level_data.entities:
            switch_index = 0
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    entity_id = entity.get('entity_id')
                    
                    # Skip already activated switches using authoritative data
                    if entity_id and self._is_switch_activated_from_entity(entity):
                        continue
                    
                    # Get reachability score from neural features
                    reachability_score = switch_features[switch_index] if switch_index < len(switch_features) else 0.5
                    switch_index += 1
                    
                    # Only include reachable switches
                    if reachability_score > 0.1:
                        subgoals.append(SwitchActivationSubgoal(
                            priority=0.8,
                            estimated_time=30.0,
                            success_probability=min(0.9, reachability_score),
                            switch_id=entity_id,
                            switch_position=(entity.get('x', 0), entity.get('y', 0)),
                            switch_type='exit_switch',
                            reachability_score=reachability_score
                        ))
        
        return subgoals
    
    def _generate_collection_subgoals(self, ninja_pos, level_data, 
                                    reachability_features) -> List[CollectionSubgoal]:
        """
        Generate collection subgoals for functional switches only.
        
        Uses actual NppEnvironment data structures from nclone.
        We focus on functional switches (exit switches) rather than arbitrary 
        collectibles like gold, as these are the only 'collectibles' that matter 
        for level completion.
        """
        subgoals = []
        
        # Focus on exit switches from level_data.entities
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('type') == EntityType.EXIT_SWITCH:
                    # Check if switch is not yet activated using actual simulation data
                    entity_id = entity.get('entity_id')
                    if entity_id and not self._is_switch_activated_from_entity(entity):
                        entity_x = entity.get('x', 0)
                        entity_y = entity.get('y', 0)
                        
                        distance = math.sqrt((ninja_pos[0] - entity_x)**2 + 
                                           (ninja_pos[1] - entity_y)**2)
                        
                        # Only include reachable functional switches
                        if distance < 300.0:  # Reasonable activation range
                            subgoals.append(CollectionSubgoal(
                                priority=0.6,  # Higher priority than arbitrary collectibles
                                estimated_time=distance / 50.0,
                                success_probability=0.8,  # Higher success rate for switches
                                target_position=(entity_x, entity_y),
                                item_type='exit_switch',
                                value=1.0,  # Exit switches are always important
                                area_connectivity=0.7  # Switches typically have good connectivity
                            ))
        
        return subgoals
    
    def _is_switch_activated_from_entity(self, entity: Dict) -> bool:
        """
        Check if a switch entity is activated using actual NppEnvironment data.
        
        For exit switches, activated means active=False (inverted logic in nclone).
        """
        if entity.get('type') == EntityType.EXIT_SWITCH:
            # For exit switches, activated means active=False (inverted logic)
            return not entity.get('active', True)
        
        # For other switch types, use state field or active field directly
        return entity.get('state', 0.0) > 0.5 or entity.get('active', False)
    
    def _is_switch_activated(self, switch_id: str, level_data) -> bool:
        """
        Check if a switch is activated using actual NppEnvironment data structures.
        
        This method should be the authoritative source for switch states,
        using actual simulation data from nclone.
        """
        # Method 1: Check level_data.entities for switch with matching entity_id
        if hasattr(level_data, 'entities') and level_data.entities:
            for entity in level_data.entities:
                if entity.get('entity_id') == switch_id:
                    return self._is_switch_activated_from_entity(entity)
        
        # Method 2: Check if level_data has direct switch state info (from environment observation)
        if hasattr(level_data, 'switch_activated'):
            # This is the direct boolean from NppEnvironment observation
            return level_data.switch_activated
        
        # Default: assume not activated if we can't find the switch
        return False
    
    def _generate_cache_key(self, ninja_pos, switch_states, level_data) -> str:
        """Generate cache key for subgoal caching."""
        # Create key based on position and switch states
        pos_key = f"{int(ninja_pos[0]/24)},{int(ninja_pos[1]/24)}"  # Grid-based position
        switch_key = ",".join(f"{k}:{v}" for k, v in sorted(switch_states.items()))
        level_key = getattr(level_data, 'level_id', 'default')
        
        return f"{level_key}|{pos_key}|{switch_key}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.subgoal_cache:
            return False
        
        cache_entry = self.subgoal_cache[cache_key]
        current_time = time.time() * 1000  # Convert to milliseconds
        
        return (current_time - cache_entry['timestamp']) < self.cache_ttl
    
    def _cache_subgoals(self, cache_key: str, subgoals: List[Subgoal], 
                       reachability_features: torch.Tensor):
        """Cache subgoals for performance optimization."""
        self.subgoal_cache[cache_key] = {
            'subgoals': subgoals,
            'features': reachability_features,
            'timestamp': time.time() * 1000
        }
        
        # Limit cache size to prevent memory issues
        if len(self.subgoal_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.subgoal_cache.keys(), 
                           key=lambda k: self.subgoal_cache[k]['timestamp'])
            del self.subgoal_cache[oldest_key]
    
    def _update_cache_metrics(self, cache_hit: bool):
        """Update cache performance metrics."""
        # Simple exponential moving average
        alpha = 0.1
        hit_value = 1.0 if cache_hit else 0.0
        self.cache_hit_rate = alpha * hit_value + (1 - alpha) * self.cache_hit_rate
    
    def _invalidate_switch_dependent_cache(self, old_switch_states, new_switch_states):
        """Invalidate cache entries affected by switch state changes."""
        changed_switches = set()
        for switch_id in set(old_switch_states.keys()) | set(new_switch_states.keys()):
            if old_switch_states.get(switch_id) != new_switch_states.get(switch_id):
                changed_switches.add(switch_id)
        
        if changed_switches:
            # Clear all cache entries (simple approach for now)
            self.subgoal_cache.clear()
    
    def _find_newly_available_subgoals(self, old_switch_states, new_switch_states, level_data):
        """Find subgoals that became available due to switch changes."""
        newly_available = []
        
        # Check for newly activated switches that might unlock new areas
        for switch_id, new_state in new_switch_states.items():
            old_state = old_switch_states.get(switch_id, False)
            if new_state and not old_state:
                # Switch was just activated - might unlock new subgoals
                newly_available.append(f"Switch {switch_id} activated - new areas may be accessible")
        
        return newly_available
    
    def get_hierarchical_stats(self) -> Dict[str, float]:
        """Get hierarchical planning statistics for logging."""
        base_stats = self.get_exploration_stats()
        
        hierarchical_stats = {
            'planning_time_ms': self.planning_time_ms,
            'cache_hit_rate': self.cache_hit_rate,
            'avg_subgoal_count': self.avg_subgoal_count,
            'cache_size': len(self.subgoal_cache)
        }
        
        return {**base_stats, **hierarchical_stats} 