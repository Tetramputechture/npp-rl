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
from npp_rl.hrl.high_level_policy import Subtask


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
            if exit_switch_id in switch_states and switch_states[exit_switch_id]:
                return True
                
        elif self.current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Check if any locked door switch was activated
            # Compare with previous switch states to detect new activations
            for switch_id, activated in switch_states.items():
                # Use .get() for last_switch_states since it may not have all current keys
                was_activated = self.last_switch_states.get(switch_id, False)
                if activated and not was_activated:
                    return True
                    
        elif self.current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Check if ninja reached exit (level completion)
            # Use obs['player_won'] which is guaranteed in nclone observations
            if obs['player_won']:
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
        """Extract ninja position from observation.
        
        The nclone environment guarantees player_x and player_y in observations.
        These are pixel coordinates updated every step in base_environment.py.
        
        Args:
            obs: Environment observation dictionary from nclone
            info: Environment info dictionary (unused for position)
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        return (int(obs['player_x']), int(obs['player_y']))
    
    def _extract_level_data(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract level data from observation.
        
        In npp-rl, level_data is not directly available in obs/info. Instead, we reconstruct
        essential level information from observation keys that are always present:
        - Switch and exit door positions from obs['switch_x/y'] and obs['exit_door_x/y']
        - Entity states from obs['entity_states'] if available
        
        This lightweight representation is sufficient for completion planning which relies
        primarily on reachability features rather than full level geometry.
        
        Args:
            obs: Environment observation dictionary from nclone
            info: Environment info dictionary (may contain level_data in some configurations)
            
        Returns:
            Dictionary with essential level information for completion planning
        """
        # Check if full level_data is available in info (e.g., from environment property access)
        if 'level_data' in info and info['level_data']:
            return info['level_data']
        
        # Reconstruct essential level information from observation
        level_data = {
            'entities': [],
            'switches': {},
            'exit_door': {
                'position': (obs['exit_door_x'], obs['exit_door_y']),
                'type': 'exit_door'
            }
        }
        
        # Add main exit switch
        level_data['switches']['exit_switch_0'] = {
            'position': (obs['switch_x'], obs['switch_y']),
            'activated': obs['switch_activated'],
            'type': 'exit',
            'is_exit': True
        }
        
        # Note: entity_states is available in obs but contains flat array format
        # For completion planning, the primary switch/exit from obs keys is sufficient
        # Advanced implementations could parse entity_states for additional switches
        
        return level_data
    
    def _extract_switch_states(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, bool]:
        """
        Extract switch states from observation.
        
        The nclone environment provides switch activation status in obs['switch_activated'].
        For completion planning, we track the main exit switch (which is guaranteed present).
        Additional switches can be inferred from obs['doors_opened'] count.
        
        Args:
            obs: Environment observation dictionary from nclone
            info: Environment info dictionary (may contain switch_states in some configurations)
            
        Returns:
            Dictionary mapping switch IDs to activation status (bool)
        """
        # Use info if available (e.g., from hierarchical mixin)
        if 'switch_states' in info and info['switch_states']:
            return info['switch_states']
        
        # Reconstruct from observation
        switch_states = {
            'exit_switch_0': bool(obs['switch_activated'])
        }
        
        # doors_opened gives us information about locked door switches
        # Each opened door implies an activated locked door switch
        doors_opened = obs['doors_opened']
        for i in range(doors_opened):
            switch_states[f'locked_door_switch_{i}'] = True
        
        return switch_states
    
    def _extract_reachability_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Extract 8D reachability features from observation.
        
        The nclone environment provides reachability features when reachability is enabled.
        These features are computed via flood-fill analysis in reachability_mixin.py:
        
        Features (all normalized 0-1):
        1. Reachable area ratio - fraction of level accessible
        2. Distance to nearest switch - normalized by level size
        3. Distance to exit - normalized by level size
        4. Reachable switches count - normalized
        5. Reachable hazards count - normalized
        6. Connectivity score - measure of path diversity
        7. Exit reachable flag - binary 0/1
        8. Switch-to-exit path exists - binary 0/1
        
        Args:
            obs: Environment observation dictionary from nclone
            
        Returns:
            8D numpy array of float32 reachability features
        """
        return obs['reachability_features'].astype(np.float32)
    
    def _find_exit_switch_id(self, obs: Dict[str, Any], info: Dict[str, Any]) -> str:
        """
        Find the exit switch ID.
        
        In N++ levels, there is always at least one exit switch (the main switch that opens
        the exit door). The nclone environment exposes this via obs['switch_x/y'] and 
        obs['switch_activated'].
        
        For standard levels, we use the canonical ID 'exit_switch_0'. Levels with multiple
        switches would need more complex logic, but the completion planner primarily cares
        about the main exit switch.
        
        Args:
            obs: Environment observation dictionary from nclone
            info: Environment info dictionary
            
        Returns:
            Exit switch ID string (always 'exit_switch_0' for standard levels)
        """
        return "exit_switch_0"
    
    def _create_reachability_system(self, reachability_features: np.ndarray):
        """
        Create a reachability system adapter for the completion planner.
        
        The completion planner expects a reachability_system object with an
        analyze_reachability() method. Since we're working from pre-computed
        reachability features rather than doing analysis on-demand, we create
        an adapter that returns the features in the expected format.
        
        The reachability features (8D vector) contain:
        1. Reachable area ratio
        2. Distance to nearest switch (normalized)
        3. Distance to exit (normalized)
        4. Reachable switches count (normalized)
        5. Reachable hazards count (normalized)
        6. Connectivity score
        7. Exit reachable flag (0/1)
        8. Switch-to-exit path exists (0/1)
        
        Args:
            reachability_features: 8D numpy array from observation
            
        Returns:
            ReachabilitySystemAdapter with analyze_reachability method
        """
        class ReachabilitySystemAdapter:
            """Adapter that wraps pre-computed reachability features."""
            
            def __init__(self, features):
                self.features = features
            
            def analyze_reachability(self, level_data, ninja_pos, switch_states):
                """
                Return reachability analysis results based on pre-computed features.
                
                The features array provides all necessary information:
                - features[6] = exit reachable (0/1)
                - features[3] = reachable switches count
                - features[7] = switch-to-exit path exists
                
                Returns:
                    Dictionary with reachability results compatible with completion planner
                """
                return {
                    'reachable_positions': set(),  # Not used by feature-based planner
                    'features': self.features,
                    'exit_reachable': bool(self.features[6] > 0.5),
                    'switch_reachable': bool(self.features[3] > 0),
                    'path_exists': bool(self.features[7] > 0.5)
                }
        
        return ReachabilitySystemAdapter(reachability_features)