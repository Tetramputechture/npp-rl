"""
Hierarchical controller using completion planner for subgoal generation.

This module implements the CompletionController that integrates the nclone completion
planner with hierarchical RL training, providing strategic subtask selection based
on reachability analysis and level completion heuristics.
"""

import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np

from nclone.planning.completion_planner import LevelCompletionPlanner


class Subtask(Enum):
    """Enumeration of available subtasks for hierarchical control."""
    NAVIGATE_TO_EXIT_SWITCH = 0
    NAVIGATE_TO_LOCKED_DOOR_SWITCH = 1
    NAVIGATE_TO_EXIT_DOOR = 2
    AVOID_MINE = 3


class CompletionController:
    """
    Hierarchical controller using completion planner for subgoal generation.
    
    This controller integrates the nclone completion planner with hierarchical RL
    training, providing strategic subtask selection based on reachability analysis
    and NPP level completion heuristics.
    
    The controller implements the Options framework where:
    - High-level policy selects subtasks based on reachability features
    - Low-level policy executes actions for the current subtask
    - Subtask transitions are managed based on completion planner logic
    
    Architecture:
    - High-level input: 8D reachability features, exit switch state, locked door states
    - High-level output: Subtask selection (4 discrete actions)
    - Low-level input: Full multimodal observations + current subtask
    - Low-level output: Movement actions (6 discrete actions)
    """
    
    def __init__(self, completion_planner: Optional[LevelCompletionPlanner] = None):
        """
        Initialize the completion controller.
        
        Args:
            completion_planner: Optional completion planner instance. If None, creates new one.
        """
        self.completion_planner = completion_planner or LevelCompletionPlanner()
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = 0
        self.subtask_step_count = 0
        self.max_subtask_steps = 1000  # Maximum steps per subtask before forced transition
        
        # Subtask transition history for logging
        self.subtask_history = []
        self.last_switch_states = {}
        self.last_ninja_pos = None
        
    def get_current_subtask(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Subtask:
        """
        Use completion planner to determine current subtask.
        
        Args:
            obs: Environment observation containing multimodal data
            info: Environment info containing game state
            
        Returns:
            Current subtask based on completion planner analysis
        """
        # Extract game state information
        ninja_pos = self._extract_ninja_position(obs, info)
        level_data = self._extract_level_data(obs, info)
        switch_states = self._extract_switch_states(obs, info)
        reachability_features = self._extract_reachability_features(obs)
        
        # Check if we should switch subtasks based on completion planner
        if self.should_switch_subtask(obs, info):
            new_subtask = self._determine_next_subtask(
                ninja_pos, level_data, switch_states, reachability_features
            )
            
            if new_subtask != self.current_subtask:
                self._transition_to_subtask(new_subtask)
        
        return self.current_subtask
    
    def should_switch_subtask(self, obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
        """
        Determine if subtask should change based on completion planner.
        
        Args:
            obs: Environment observation
            info: Environment info
            
        Returns:
            True if subtask should switch, False otherwise
        """
        # Check for forced transition due to step limit
        if self.subtask_step_count >= self.max_subtask_steps:
            return True
            
        # Check for completion of current subtask
        switch_states = self._extract_switch_states(obs, info)
        ninja_pos = self._extract_ninja_position(obs, info)
        
        # Detect switch state changes (subtask completion)
        if self.last_switch_states != switch_states:
            return True
            
        # Check for specific subtask completion conditions
        if self.current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Check if exit switch was activated
            exit_switch_id = self._find_exit_switch_id(obs, info)
            if exit_switch_id and switch_states.get(exit_switch_id, False):
                return True
                
        elif self.current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Check if any locked door switch was activated
            for switch_id, activated in switch_states.items():
                if activated and not self.last_switch_states.get(switch_id, False):
                    return True
                    
        elif self.current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Check if ninja reached exit (level completion)
            if info.get('level_complete', False):
                return True
                
        # Check for significant position change (potential mine avoidance completion)
        if self.current_subtask == Subtask.AVOID_MINE:
            if self.last_ninja_pos and ninja_pos:
                distance_moved = np.linalg.norm(
                    np.array(ninja_pos) - np.array(self.last_ninja_pos)
                )
                if distance_moved > 5.0:  # Moved significant distance
                    return True
        
        return False
    
    def _determine_next_subtask(
        self, 
        ninja_pos: Tuple[int, int], 
        level_data: Dict[str, Any], 
        switch_states: Dict[str, bool],
        reachability_features: np.ndarray
    ) -> Subtask:
        """
        Determine the next subtask using completion planner logic.
        
        Args:
            ninja_pos: Current ninja position
            level_data: Level layout data
            switch_states: Current switch activation states
            reachability_features: 8D reachability features
            
        Returns:
            Next subtask to execute
        """
        # Use completion planner to get strategic plan
        try:
            # Create mock reachability system for planner
            reachability_system = self._create_reachability_system(reachability_features)
            
            completion_strategy = self.completion_planner.plan_completion(
                ninja_pos, level_data, switch_states, reachability_system
            )
            
            if completion_strategy.steps:
                # Map completion step to subtask
                first_step = completion_strategy.steps[0]
                return self._map_completion_step_to_subtask(first_step)
                
        except Exception as e:
            print(f"Warning: Completion planner failed: {e}")
            
        # Fallback logic based on reachability features and game state
        return self._fallback_subtask_selection(switch_states, reachability_features)
    
    def _map_completion_step_to_subtask(self, completion_step) -> Subtask:
        """Map completion planner step to subtask enum."""
        action_type = completion_step.action_type
        
        if action_type == "navigate_and_activate":
            # Determine if it's exit switch or locked door switch
            if "exit" in completion_step.description.lower():
                return Subtask.NAVIGATE_TO_EXIT_SWITCH
            else:
                return Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
        elif action_type == "navigate_to_exit":
            return Subtask.NAVIGATE_TO_EXIT_DOOR
        else:
            # Default to exit switch navigation
            return Subtask.NAVIGATE_TO_EXIT_SWITCH
    
    def _fallback_subtask_selection(
        self, 
        switch_states: Dict[str, bool], 
        reachability_features: np.ndarray
    ) -> Subtask:
        """Fallback subtask selection when completion planner fails."""
        # Simple heuristic: if exit switch not activated, go for it
        # Otherwise, go for exit door
        exit_switch_activated = any(switch_states.values())
        
        if not exit_switch_activated:
            return Subtask.NAVIGATE_TO_EXIT_SWITCH
        else:
            return Subtask.NAVIGATE_TO_EXIT_DOOR
    
    def _transition_to_subtask(self, new_subtask: Subtask):
        """Transition to a new subtask with logging."""
        old_subtask = self.current_subtask
        self.current_subtask = new_subtask
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0
        
        # Log transition
        transition = {
            'timestamp': time.time(),
            'from_subtask': old_subtask.name,
            'to_subtask': new_subtask.name,
            'step_count': self.subtask_step_count
        }
        self.subtask_history.append(transition)
        
        print(f"Subtask transition: {old_subtask.name} -> {new_subtask.name}")
    
    def step(self, obs: Dict[str, Any], info: Dict[str, Any]):
        """Update controller state after environment step."""
        self.subtask_step_count += 1
        self.last_switch_states = self._extract_switch_states(obs, info)
        self.last_ninja_pos = self._extract_ninja_position(obs, info)
    
    def reset(self):
        """Reset controller state for new episode."""
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0
        self.last_switch_states = {}
        self.last_ninja_pos = None
    
    def get_subtask_features(self) -> np.ndarray:
        """
        Get current subtask as one-hot encoded features.
        
        Returns:
            4-dimensional one-hot vector representing current subtask
        """
        features = np.zeros(4, dtype=np.float32)
        features[self.current_subtask.value] = 1.0
        return features
    
    def get_subtask_metrics(self) -> Dict[str, Any]:
        """Get metrics about subtask performance."""
        return {
            'current_subtask': self.current_subtask.name,
            'subtask_step_count': self.subtask_step_count,
            'subtask_duration': time.time() - self.subtask_start_time,
            'total_transitions': len(self.subtask_history),
            'recent_transitions': self.subtask_history[-5:] if self.subtask_history else []
        }
    
    # Helper methods for extracting information from observations
    def _extract_ninja_position(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Tuple[int, int]:
        """Extract ninja position from observation/info."""
        # Try to get from info first
        if 'ninja_pos' in info:
            return tuple(info['ninja_pos'])
        
        # Fallback: try to extract from observation
        # This would need to be implemented based on actual observation structure
        return (0, 0)  # Placeholder
    
    def _extract_level_data(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract level data from observation/info."""
        # This would extract level layout information
        return info.get('level_data', {})
    
    def _extract_switch_states(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, bool]:
        """Extract switch states from observation/info."""
        return info.get('switch_states', {})
    
    def _extract_reachability_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract 8D reachability features from observation."""
        # This would extract reachability features from the observation
        # For now, return zeros as placeholder
        return np.zeros(8, dtype=np.float32)
    
    def _find_exit_switch_id(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Optional[str]:
        """Find the exit switch ID from level data."""
        level_data = self._extract_level_data(obs, info)
        # This would find the exit switch in the level data
        return None  # Placeholder
    
    def _create_reachability_system(self, reachability_features: np.ndarray):
        """Create a mock reachability system for the completion planner."""
        # This would create a reachability system compatible with the planner
        # For now, return a simple mock
        class MockReachabilitySystem:
            def analyze_reachability(self, level_data, ninja_pos, switch_states):
                return {'reachable_positions': set(), 'features': reachability_features}
        
        return MockReachabilitySystem()