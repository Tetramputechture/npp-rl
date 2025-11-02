"""
Subtask-specific reward functions for hierarchical RL.

Implements dense rewards for navigate_to_exit_switch, navigate_to_locked_door_switch,
navigate_to_exit_door, and explore_for_switches subtasks. Integrates with PBRS for
theoretically grounded shaping and includes mine avoidance incentives.

Reward Scale Design Philosophy:
-------------------------------
All reward constants are designed relative to base environment rewards:
- Exit door: +1.0 (terminal success)
- Switch activation: +0.1 (milestone progress)
- Death: -0.5 (terminal failure)

The hierarchical reward magnitudes follow these principles:
1. Shaped rewards must be small relative to sparse rewards (Ng et al., 1999)
2. Progress rewards scale with difficulty to prevent exploitation
3. PBRS guarantees policy invariance when properly tuned (γ-discounting)

Research Foundations:
---------------------
- PBRS (Potential-Based Reward Shaping): Ng et al. (1999)
  "Policy invariance under reward shaping"
  → Ensures shaped rewards don't change optimal policy

- Hierarchical RL reward decomposition: Dietterich (2000)
  "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
  → Subtask-specific rewards aid credit assignment

- Dense reward design: Popov et al. (2017)
  "Data-efficient Deep Reinforcement Learning for Dexterous Manipulation"
  → Progress-based shaping accelerates learning in sparse environments

Reward Constant Rationale:
--------------------------
PROGRESS_REWARD_SCALE = 0.02
  → 2% of switch bonus per unit distance
  → Typical navigation: 10-50 units → 0.2-1.0 cumulative progress
  → Keeps progress rewards < terminal rewards

PROXIMITY_BONUS_SCALE = 0.05
  → 50% of switch activation bonus when very close
  → Encourages precise positioning for interaction

ACTIVATION_BONUS_LOCKED = 0.05
  → Matches proximity bonus magnitude
  → Reinforces correct switch identification

DOOR_OPENING_BONUS = 0.03
  → Smaller than activation (indirect consequence)
  → Still reinforces correct causal chain

EFFICIENCY_BONUS = 0.2
  → 20% of exit reward for fast completion
  → Encourages optimal paths without overwhelming sparse signal

EXPLORATION_REWARD = 0.0 (DISABLED for efficiency)
  → CHANGED from 0.01 to 0.0
  → Rationale: Conflicts with efficiency goal by encouraging wandering
  → Should ONLY be enabled for exploration curriculum stages

DISCOVERY_BONUS = 0.05
  → 50% of switch bonus for finding new objectives
  → Balances exploration vs exploitation

CONNECTIVITY_BONUS = 0.02
  → Rewards expanding reachable state space
  → Based on reachability graph metrics

TIMEOUT_PENALTY_MAJOR = -0.1
  → Negative but smaller than death penalty
  → Prevents infinite subtask loops

MINE_PROXIMITY_PENALTY = -0.01 (REDUCED for efficiency)
  → CHANGED from -0.02 to -0.01
  → Rationale: Previous penalty too harsh, causing overly conservative detours
  → Still discourages risky paths without dominating route selection

SAFE_NAVIGATION_BONUS = 0.0 (DISABLED for efficiency)
  → CHANGED from 0.01 to 0.0
  → Rationale: Conflicts with efficiency by rewarding MORE steps (longer paths)
  → Mine avoidance already handled by proximity penalty
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
    # CRITICAL FIX: DISABLED EXPLORATION_REWARD (was 0.01)
    # Rationale: Encourages wandering instead of direct/efficient paths
    # Should ONLY be enabled for exploration curriculum stages
    EXPLORATION_REWARD = 0.0  # DISABLED: was 0.01, now 0.0 (efficiency prioritization)
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
    # CRITICAL FIX: Reduced penalty to avoid overly conservative detours
    MINE_PROXIMITY_PENALTY = (
        -0.01
    )  # CHANGED: -0.02 → -0.01 (less harsh, more efficient paths)
    # CRITICAL FIX: REMOVED SAFE_NAVIGATION_BONUS (was 0.01)
    # Rationale: Conflicts with efficiency goal by rewarding MORE steps (longer paths)
    SAFE_NAVIGATION_BONUS = (
        0.0  # DISABLED: was 0.01, now 0.0 (efficiency prioritization)
    )
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
        if obs["switch_activated"] and not prev_obs["switch_activated"]:
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

        # Find nearest locked door switch from observation
        # Uses helper method that checks for 'locked_switches' in obs
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
        if obs["player_won"]:
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
        current_connectivity = obs["reachability_features"][5]
        prev_connectivity = prev_obs["reachability_features"][5]
        if current_connectivity > prev_connectivity:
            improvement = current_connectivity - prev_connectivity
            reward += improvement * self.CONNECTIVITY_BONUS

        tracker.increment_steps()

        # Timeout penalty to encourage transition to specific subtask
        if tracker.get_steps() > self.TIMEOUT_EXPLORE:
            reward += (
                self.TIMEOUT_PENALTY_MAJOR * 0.5
            )  # Smaller penalty for exploration

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
            connectivity_score = obs["reachability_features"][5]
            return connectivity_score * 0.05

        return 0.0

    # Helper methods

    def _find_nearest_locked_switch(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Find the nearest uncollected locked door switch position.

        Locked door entities (types 6 and 7) have their switch position
        stored in xpos, ypos attributes. Only considers switches that
        haven't been collected yet (active=True).

        Args:
            obs: Observation dictionary from environment

        Returns:
            Position array [x, y] of nearest locked door switch, or None if none found
        """
        ninja_pos = np.array([obs["player_x"], obs["player_y"]])
        min_distance = float("inf")
        nearest_switch = None

        # Check locked_doors (type 6 entities)
        locked_doors = obs.get("locked_doors", [])
        for locked_door in locked_doors:
            # Only consider switches that haven't been collected
            if getattr(locked_door, "active", True):
                switch_x = getattr(locked_door, "xpos", None)
                switch_y = getattr(locked_door, "ypos", None)

                if switch_x is not None and switch_y is not None:
                    switch_pos = np.array([switch_x, switch_y])
                    distance = np.linalg.norm(ninja_pos - switch_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_switch = switch_pos

        # Check locked_door_switches (type 7 entities)
        locked_door_switches = obs.get("locked_door_switches", [])
        for switch in locked_door_switches:
            # Only consider switches that haven't been collected
            if getattr(switch, "active", True):
                switch_x = getattr(switch, "xpos", None)
                switch_y = getattr(switch, "ypos", None)

                if switch_x is not None and switch_y is not None:
                    switch_pos = np.array([switch_x, switch_y])
                    distance = np.linalg.norm(ninja_pos - switch_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_switch = switch_pos

        return nearest_switch

    def _detect_locked_switch_activation(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
    ) -> bool:
        """
        Detect if a locked door switch was just activated.

        The nclone environment tracks opened doors via obs['doors_opened'].
        When a locked door switch is activated, this count increases by 1.

        Args:
            obs: Current observation from nclone
            prev_obs: Previous observation from nclone

        Returns:
            True if a switch was just activated (doors_opened increased)
        """
        return obs["doors_opened"] > prev_obs["doors_opened"]

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

        Uses reachability features[5] (connectivity score) to detect improved connectivity,
        which indicates discovering new reachable areas/objectives. This feature is computed
        by the flood-fill reachability analysis in nclone's reachability_mixin.

        A significant jump in connectivity (> 0.2 improvement) indicates the agent discovered
        a new pathway or unlocked a previously blocked area.

        Args:
            obs: Current observation from nclone with reachability_features
            prev_obs: Previous observation from nclone with reachability_features

        Returns:
            True if connectivity improved significantly (new area discovered)
        """
        current_connectivity = obs["reachability_features"][5]
        prev_connectivity = prev_obs["reachability_features"][5]

        improvement = current_connectivity - prev_connectivity
        return improvement > 0.2  # Threshold for significant discovery

    def _get_nearest_dangerous_mine_distance(
        self, obs: Dict[str, Any]
    ) -> Optional[float]:
        """
        Get distance to nearest dangerous (toggled) mine.

        Mine state information is encoded in obs['entity_states'] from nclone.
        Only TOGGLED mines (state 0) are dangerous. UNTOGGLED (state 1) and TOGGLING (state 2) are safe.

        The nclone environment uses MineStateProcessor to track mine states internally,
        but this information is not directly exposed in observations. For hierarchical RL,
        the reachability features provide indirect information about hazards via features[4]
        (reachable hazards count).

        Since detailed mine state information is not available in standard observations,
        this method returns None, indicating that mine avoidance should rely on
        reachability features rather than explicit distance calculations.

        Args:
            obs: Environment observation dictionary from nclone

        Returns:
            None (mine states not directly available in observations)

        Note:
            For mine avoidance, use reachability_features[4] which indicates
            reachable hazards count, computed by the flood-fill reachability system.
        """
        # Mine states are tracked internally by MineStateProcessor but not exposed in obs
        # Hierarchical policies should use reachability_features[4] for hazard awareness
        return None

    def _check_mine_state_awareness(self, obs: Dict[str, Any]) -> bool:
        """
        Check if mine hazard information is available.

        Since detailed mine states are not directly available in nclone observations,
        this checks if reachability features are present, which include hazard
        awareness via features[4] (reachable hazards count).

        Args:
            obs: Environment observation dictionary from nclone

        Returns:
            True if reachability features (including hazard info) are available
        """
        return "reachability_features" in obs and len(obs["reachability_features"]) >= 5

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
            "current_distance": tracker.get_best_distance()
            if tracker.has_previous_distance()
            else float("inf"),
            "steps_in_subtask": tracker.get_steps(),
            "pbrs_potential": self.prev_potential
            if self.prev_potential is not None
            else 0.0,
        }

        if current_subtask == Subtask.EXPLORE_FOR_SWITCHES:
            components["locations_visited"] = len(tracker.visited_locations)

        return components
