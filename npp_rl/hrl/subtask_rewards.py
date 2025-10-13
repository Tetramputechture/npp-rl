"""
Subtask-specific reward functions for hierarchical RL.

Implements dense rewards for navigate_to_exit_switch, navigate_to_locked_door_switch,
navigate_to_exit_door, and explore_for_switches subtasks. Integrates with PBRS for
theoretically grounded shaping and includes mine avoidance incentives.
"""

from typing import Dict, Any, Optional
import numpy as np

from npp_rl.hrl.high_level_policy import Subtask
from npp_rl.hrl.progress_trackers import ProgressTracker, ExplorationTracker


class SubtaskRewardCalculator:
    """
    Calculate dense rewards specific to each subtask.
    
    This calculator provides subtask-aligned rewards that encourage efficient
    completion of hierarchical objectives. All rewards are scaled to work
    harmoniously with base rewards (+0.1 switch, +1.0 exit, -0.5 death).
    """
    
    # Subtask reward scales (relative to base rewards)
    PROGRESS_REWARD_SCALE = 0.02  # Per unit distance improvement
    PROXIMITY_BONUS_SCALE = 0.05  # When close to target
    ACTIVATION_BONUS_LOCKED = 0.05  # Locked door switch activation
    DOOR_OPENING_BONUS = 0.03  # When locked door opens
    EFFICIENCY_BONUS = 0.2  # Quick completion after switch activation
    EXPLORATION_REWARD = 0.01  # New area discovery
    DISCOVERY_BONUS = 0.05  # Finding unknown objectives
    CONNECTIVITY_BONUS = 0.02  # Improving reachability
    
    # Timeout penalties
    TIMEOUT_PENALTY_MAJOR = -0.1  # For critical subtasks
    
    # Distance thresholds (in game units/tiles)
    PROXIMITY_THRESHOLD_SWITCH = 2.0  # Tiles
    PROXIMITY_THRESHOLD_EXIT = 1.0  # Tiles
    
    # Timeouts (in steps)
    TIMEOUT_NAVIGATE_SWITCH = 300  # Steps
    TIMEOUT_EXPLORE = 200  # Steps
    
    # Efficiency thresholds (in steps)
    EFFICIENT_EXIT_TIME = 150  # Steps from switch to exit
    
    # Mine avoidance parameters
    MINE_PROXIMITY_PENALTY = -0.02  # When too close to toggled mine
    SAFE_NAVIGATION_BONUS = 0.01  # Maintaining safe distance
    MINE_STATE_AWARENESS_BONUS = 0.005  # Correct mine state identification
    MINE_DANGER_DISTANCE = 1.5  # Tiles
    MINE_SAFE_DISTANCE = 3.0  # Tiles
    
    def __init__(
        self,
        enable_mine_avoidance: bool = True,
        enable_pbrs: bool = True,
        pbrs_gamma: float = 0.99,
    ):
        """
        Initialize subtask reward calculator.
        
        Args:
            enable_mine_avoidance: Whether to include mine avoidance rewards
            enable_pbrs: Whether to calculate PBRS potentials
            pbrs_gamma: Discount factor for PBRS
        """
        self.enable_mine_avoidance = enable_mine_avoidance
        self.enable_pbrs = enable_pbrs
        self.pbrs_gamma = pbrs_gamma
        
        # Track progress for each subtask
        self.progress_trackers = {
            Subtask.NAVIGATE_TO_EXIT_SWITCH: ProgressTracker(),
            Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH: ProgressTracker(),
            Subtask.NAVIGATE_TO_EXIT_DOOR: ProgressTracker(),
            Subtask.EXPLORE_FOR_SWITCHES: ExplorationTracker(),
        }
        
        # Track PBRS potentials
        self.prev_potential = None
        
        # Track switch activation time for efficiency bonus
        self.switch_activation_step = None
        self.current_step = 0
    
    def calculate_subtask_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        current_subtask: Subtask,
    ) -> float:
        """
        Calculate total subtask-specific reward.
        
        Args:
            obs: Current observation
            prev_obs: Previous observation
            current_subtask: Currently active subtask
            
        Returns:
            Total subtask reward
        """
        self.current_step += 1
        
        # Route to appropriate subtask calculator
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            reward = self._calculate_navigate_to_exit_switch_reward(obs, prev_obs)
        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            reward = self._calculate_navigate_to_locked_switch_reward(obs, prev_obs)
        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            reward = self._calculate_navigate_to_exit_door_reward(obs, prev_obs)
        elif current_subtask == Subtask.EXPLORE_FOR_SWITCHES:
            reward = self._calculate_explore_for_switches_reward(obs, prev_obs)
        else:
            reward = 0.0
        
        # Add mine avoidance rewards if enabled
        if self.enable_mine_avoidance:
            mine_reward = self._calculate_mine_avoidance_reward(obs, prev_obs)
            reward += mine_reward
        
        # Add PBRS shaping reward if enabled
        if self.enable_pbrs:
            pbrs_reward = self._calculate_pbrs_reward(obs, current_subtask)
            reward += pbrs_reward
        
        # Track switch activation for efficiency bonus
        if obs.get("switch_activated", False) and not prev_obs.get("switch_activated", False):
            self.switch_activation_step = self.current_step
        
        return reward
    
    def _calculate_navigate_to_exit_switch_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> float:
        """Calculate reward for navigating to exit switch (progress, proximity, timeout)."""
        reward = 0.0
        tracker = self.progress_trackers[Subtask.NAVIGATE_TO_EXIT_SWITCH]
        
        # Calculate distance to exit switch
        ninja_pos = np.array([obs["player_x"], obs["player_y"]])
        switch_pos = np.array([obs["switch_x"], obs["switch_y"]])
        current_distance = np.linalg.norm(ninja_pos - switch_pos)
        
        # Progress reward: reward for getting closer
        if tracker.has_previous_distance():
            prev_distance = tracker.get_best_distance()
            if current_distance < prev_distance:
                distance_improvement = prev_distance - current_distance
                reward += distance_improvement * self.PROGRESS_REWARD_SCALE
                tracker.update_distance(current_distance)
        else:
            tracker.update_distance(current_distance)
        
        # Proximity bonus
        if current_distance < self.PROXIMITY_THRESHOLD_SWITCH:
            reward += self.PROXIMITY_BONUS_SCALE
        
        # Timeout penalty
        tracker.increment_steps()
        if tracker.get_steps() > self.TIMEOUT_NAVIGATE_SWITCH:
            reward += self.TIMEOUT_PENALTY_MAJOR
        
        return reward
    
    def _calculate_navigate_to_locked_switch_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> float:
        """Calculate reward for navigating to locked door switches."""
        reward = 0.0
        tracker = self.progress_trackers[Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH]
        
        # Find nearest locked door switch (placeholder - needs actual implementation)
        # In a real implementation, this would query the level state for locked doors
        nearest_locked_switch_pos = self._find_nearest_locked_switch(obs)
        
        if nearest_locked_switch_pos is not None:
            ninja_pos = np.array([obs["player_x"], obs["player_y"]])
            current_distance = np.linalg.norm(ninja_pos - nearest_locked_switch_pos)
            
            # Progress reward
            if tracker.has_previous_distance():
                prev_distance = tracker.get_best_distance()
                if current_distance < prev_distance:
                    distance_improvement = prev_distance - current_distance
                    reward += distance_improvement * self.PROGRESS_REWARD_SCALE
                    tracker.update_distance(current_distance)
            else:
                tracker.update_distance(current_distance)
            
            # Switch selection bonus: reward for moving toward reachable switches
            if current_distance < self.PROXIMITY_THRESHOLD_SWITCH:
                reward += 0.01  # Small bonus for being near a switch
        
        # Locked door switch activation bonus
        # Note: This would require tracking individual switch states
        if self._detect_locked_switch_activation(obs, prev_obs):
            reward += self.ACTIVATION_BONUS_LOCKED
        
        # Door opening bonus: reward when a new path opens
        if self._detect_door_opening(obs, prev_obs):
            reward += self.DOOR_OPENING_BONUS
        
        tracker.increment_steps()
        return reward
    
    def _calculate_navigate_to_exit_door_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> float:
        """
        Calculate reward for navigating to exit door.
        
        Rewards:
        - Progress toward exit door (higher scale than switch navigation)
        - Proximity bonus when very close to exit
        - Efficiency bonus if completed quickly after switch activation
        """
        reward = 0.0
        tracker = self.progress_trackers[Subtask.NAVIGATE_TO_EXIT_DOOR]
        
        # Calculate distance to exit door
        ninja_pos = np.array([obs["player_x"], obs["player_y"]])
        exit_pos = np.array([obs["exit_door_x"], obs["exit_door_y"]])
        current_distance = np.linalg.norm(ninja_pos - exit_pos)
        
        # Progress reward (higher scale for final objective)
        if tracker.has_previous_distance():
            prev_distance = tracker.get_best_distance()
            if current_distance < prev_distance:
                distance_improvement = prev_distance - current_distance
                reward += distance_improvement * (self.PROGRESS_REWARD_SCALE * 1.5)
                tracker.update_distance(current_distance)
        else:
            tracker.update_distance(current_distance)
        
        # Proximity bonus (larger for exit)
        if current_distance < self.PROXIMITY_THRESHOLD_EXIT:
            reward += self.PROXIMITY_BONUS_SCALE * 2.0
        
        # Efficiency bonus: reward quick exit after switch activation
        if obs.get("player_won", False):
            if self.switch_activation_step is not None:
                steps_to_exit = self.current_step - self.switch_activation_step
                if steps_to_exit <= self.EFFICIENT_EXIT_TIME:
                    reward += self.EFFICIENCY_BONUS
        
        tracker.increment_steps()
        return reward
    
    def _calculate_explore_for_switches_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> float:
        """
        Calculate reward for exploration when no clear path exists.
        
        Rewards:
        - Exploration reward for visiting new areas
        - Discovery bonus for finding previously unknown objectives
        - Connectivity bonus for improving reachability
        - Automatic timeout to switch to specific subtask
        """
        reward = 0.0
        tracker = self.progress_trackers[Subtask.EXPLORE_FOR_SWITCHES]
        
        # Exploration reward: reward for visiting new grid cells
        ninja_pos = np.array([obs["player_x"], obs["player_y"]])
        if tracker.visit_new_location(ninja_pos):
            reward += self.EXPLORATION_REWARD
        
        # Discovery bonus: reward for finding new objectives
        # This would track newly visible switches/doors in the observation
        if self._detect_objective_discovery(obs, prev_obs):
            reward += self.DISCOVERY_BONUS
        
        # Connectivity bonus: reward for improving reachability score
        current_connectivity = obs.get("reachability_features", np.zeros(8))[5]
        prev_connectivity = prev_obs.get("reachability_features", np.zeros(8))[5]
        if current_connectivity > prev_connectivity:
            improvement = current_connectivity - prev_connectivity
            reward += improvement * self.CONNECTIVITY_BONUS
        
        tracker.increment_steps()
        
        # Timeout penalty to encourage transition to specific subtask
        if tracker.get_steps() > self.TIMEOUT_EXPLORE:
            reward += self.TIMEOUT_PENALTY_MAJOR * 0.5  # Smaller penalty for exploration
        
        return reward
    
    def _calculate_mine_avoidance_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> float:
        """
        Calculate rewards/penalties for mine avoidance behavior.
        
        Penalties:
        - Being too close to toggled (dangerous) mines
        
        Rewards:
        - Maintaining safe distance from mines
        - Correctly identifying mine states
        """
        reward = 0.0
        
        # Get mine proximity information
        nearest_mine_distance = self._get_nearest_dangerous_mine_distance(obs)
        
        if nearest_mine_distance is not None:
            # Penalty for being too close to dangerous mines
            if nearest_mine_distance < self.MINE_DANGER_DISTANCE:
                reward += self.MINE_PROXIMITY_PENALTY
            
            # Bonus for maintaining safe distance
            elif nearest_mine_distance > self.MINE_SAFE_DISTANCE:
                reward += self.SAFE_NAVIGATION_BONUS
        
        # Bonus for mine state awareness (if implemented in observation)
        if self._check_mine_state_awareness(obs):
            reward += self.MINE_STATE_AWARENESS_BONUS
        
        return reward
    
    def _calculate_pbrs_reward(
        self,
        obs: Dict[str, Any],
        current_subtask: Subtask,
    ) -> float:
        """
        Calculate potential-based reward shaping for current subtask.
        
        PBRS provides theoretically grounded reward shaping that doesn't
        alter the optimal policy.
        """
        current_potential = self._calculate_subtask_potential(obs, current_subtask)
        
        if self.prev_potential is not None:
            # r_shaped = γ * Φ(s') - Φ(s)
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
        else:
            pbrs_reward = 0.0
        
        self.prev_potential = current_potential
        return pbrs_reward
    
    def _calculate_subtask_potential(
        self,
        obs: Dict[str, Any],
        current_subtask: Subtask,
    ) -> float:
        """
        Calculate state potential for current subtask.
        
        Potential is based on negative distance to the subtask's target,
        scaled appropriately for each subtask type.
        """
        ninja_pos = np.array([obs["player_x"], obs["player_y"]])
        
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            target_pos = np.array([obs["switch_x"], obs["switch_y"]])
            distance = np.linalg.norm(ninja_pos - target_pos)
            return -distance * 0.1
        
        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            nearest_locked_switch = self._find_nearest_locked_switch(obs)
            if nearest_locked_switch is not None:
                distance = np.linalg.norm(ninja_pos - nearest_locked_switch)
                return -distance * 0.1
            return 0.0
        
        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            target_pos = np.array([obs["exit_door_x"], obs["exit_door_y"]])
            distance = np.linalg.norm(ninja_pos - target_pos)
            return -distance * 0.15  # Higher weight for final objective
        
        elif current_subtask == Subtask.EXPLORE_FOR_SWITCHES:
            # Exploration potential based on reachability/connectivity
            reachability_features = obs.get("reachability_features", np.zeros(8))
            connectivity_score = reachability_features[5] if len(reachability_features) > 5 else 0.0
            return connectivity_score * 0.05
        
        return 0.0
    
    # Helper methods
    
    def _find_nearest_locked_switch(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Find the nearest locked door switch position.
        
        Attempts to extract locked door switch positions from the observation.
        In N++, locked doors are entity type 4 and have associated switches.
        
        Returns:
            Position array [x, y] of nearest locked door switch, or None if not found
        """
        # Try to get from info dict if provided with observation
        if 'locked_switches' in obs and obs['locked_switches']:
            ninja_pos = np.array([obs['player_x'], obs['player_y']])
            min_distance = float('inf')
            nearest_switch = None
            
            for switch_pos in obs['locked_switches']:
                distance = np.linalg.norm(ninja_pos - np.array(switch_pos))
                if distance < min_distance:
                    min_distance = distance
                    nearest_switch = np.array(switch_pos)
            
            return nearest_switch
        
        # If locked door information not available, return None
        # This is acceptable as not all levels have locked doors
        return None
    
    def _detect_locked_switch_activation(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> bool:
        """
        Detect if a locked door switch was just activated.
        
        Checks if the number of opened doors increased, indicating a switch activation.
        """
        # Check if doors_opened count increased
        current_doors = obs.get('doors_opened', 0)
        prev_doors = prev_obs.get('doors_opened', 0)
        
        # A locked door switch activation would increase doors_opened count
        return current_doors > prev_doors
    
    def _detect_door_opening(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> bool:
        """
        Detect if a locked door just opened.
        
        Uses the same detection as switch activation since door opening
        is triggered by switch activation in N++.
        """
        # Door opening is detected the same way as switch activation
        # since doors open when their associated switch is activated
        return self._detect_locked_switch_activation(obs, prev_obs)
    
    def _detect_objective_discovery(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> bool:
        """
        Detect if new objectives (switches/doors) were discovered.
        
        Uses reachability features to detect improved connectivity,
        which indicates discovering new areas/objectives.
        """
        # Use reachability features to detect new area discovery
        # Feature index 5 typically represents connectivity/reachability score
        current_reachability = obs.get('reachability_features', np.zeros(8))
        prev_reachability = prev_obs.get('reachability_features', np.zeros(8))
        
        if len(current_reachability) >= 6 and len(prev_reachability) >= 6:
            # Check if reachability improved significantly
            improvement = current_reachability[5] - prev_reachability[5]
            return improvement > 0.1  # Threshold for significant discovery
        
        return False
    
    def _get_nearest_dangerous_mine_distance(self, obs: Dict[str, Any]) -> Optional[float]:
        """
        Get distance to nearest dangerous (toggled) mine.
        
        Only TOGGLED mines are dangerous (state 0). UNTOGGLED (state 1) and TOGGLING (state 2) are safe.
        
        Returns:
            Distance to nearest dangerous mine in pixels, or None if no dangerous mines
        """
        # Try to extract mine information from entity_states
        # entity_states contains information about all entities including mines
        if 'entity_states' not in obs or obs['entity_states'] is None:
            return None
        
        ninja_pos = np.array([obs['player_x'], obs['player_y']])
        entity_states = obs['entity_states']
        
        # Mine state information is encoded in entity_states
        # We need to parse this to find toggled mines
        # For now, use a simplified approach checking if mine info is available
        if 'mine_states' in obs:
            mine_states = obs['mine_states']
            min_distance = float('inf')
            found_dangerous_mine = False
            
            for mine_info in mine_states:
                # Only consider TOGGLED mines (state 0) as dangerous
                if mine_info.get('state') == 0:  # TOGGLED state
                    mine_pos = np.array([mine_info['x'], mine_info['y']])
                    distance = np.linalg.norm(ninja_pos - mine_pos)
                    if distance < min_distance:
                        min_distance = distance
                        found_dangerous_mine = True
            
            if found_dangerous_mine:
                return float(min_distance)
        
        return None
    
    def _check_mine_state_awareness(self, obs: Dict[str, Any]) -> bool:
        """
        Check if the agent is correctly tracking mine states.
        
        Returns True if mine state information is available in the observation,
        indicating the agent has access to mine awareness features.
        """
        # Check if mine state information is available
        # This indicates the environment is providing mine awareness data
        return 'mine_states' in obs and obs['mine_states'] is not None and len(obs['mine_states']) > 0
    
    def reset(self):
        """Reset all trackers for new episode."""
        for tracker in self.progress_trackers.values():
            tracker.reset()
        
        self.prev_potential = None
        self.switch_activation_step = None
        self.current_step = 0
    
    def get_subtask_components(self, current_subtask: Subtask) -> Dict[str, float]:
        """
        Get individual reward components for debugging/logging.
        
        Returns:
            Dictionary of reward component values for current subtask
        """
        tracker = self.progress_trackers[current_subtask]
        
        components = {
            "current_distance": tracker.get_best_distance() if tracker.has_previous_distance() else float('inf'),
            "steps_in_subtask": tracker.get_steps(),
            "pbrs_potential": self.prev_potential if self.prev_potential is not None else 0.0,
        }
        
        if current_subtask == Subtask.EXPLORE_FOR_SWITCHES:
            components["locations_visited"] = len(tracker.visited_locations)
        
        return components

