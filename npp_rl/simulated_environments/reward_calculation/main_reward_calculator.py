"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any
from npp_rl.simulated_environments.reward_calculation.base_reward_calculator import BaseRewardCalculator
from npp_rl.simulated_environments.reward_calculation.movement_reward_calculator import MovementRewardCalculator
from npp_rl.simulated_environments.reward_calculation.navigation_reward_calculator import NavigationRewardCalculator
from npp_rl.simulated_environments.reward_calculation.exploration_reward_calculator import ExplorationRewardCalculator
from npp_rl.simulated_environments.reward_calculation.progression_tracker import ProgressionTracker
from npp_rl.simulated_environments.reward_calculation.movement_reward_calculator import MovementEvaluator


class RewardCalculator(BaseRewardCalculator):
    """
    A curriculum-based reward calculator for the N++ environment that progressively
    adapts rewards based on the agent's demonstrated capabilities and learning stage.

    The calculator implements three main learning stages:
    1. Movement Mastery: Basic control and precision
    2. Navigation: Efficient path-finding and objective targeting
    3. Optimization: Speed and perfect execution

    Each stage builds upon the skills learned in previous stages, with rewards
    automatically adjusting based on the agent's demonstrated competence.
    """

    def __init__(self, movement_evaluator: MovementEvaluator):
        """Initialize reward calculator with all components.

        Args:
            movement_evaluator: Evaluator for movement success metrics
        """
        super().__init__()
        self.movement_calculator = MovementRewardCalculator(movement_evaluator)
        self.navigation_calculator = NavigationRewardCalculator()
        self.exploration_calculator = ExplorationRewardCalculator()
        self.progression_tracker = ProgressionTracker()

    def calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any], action_taken: int) -> float:
        """Calculate reward.

        Args:
            obs: Current game state

        Returns:
            float: Total reward for the transition
        """
        # Termination penalties
        # Death penalty
        if obs.get('player_dead', False):
            return self.DEATH_PENALTY

        # Win condition
        if obs.get('player_won', False):
            return self.TERMINAL_REWARD

        # Get current reward scales
        scales = self.progression_tracker.get_reward_scales()

        # Initialize total reward
        reward = 0.0

        # Level 2: Movement quality and control
        movement_reward = self.movement_calculator.calculate_movement_reward(
            obs, prev_obs, action_taken, scales['movement']
        )
        reward += movement_reward

        # Level 3: Navigation and objective completion
        navigation_reward, switch_activated = self.navigation_calculator.calculate_navigation_reward(
            obs, prev_obs, scales['navigation']
        )
        reward += navigation_reward

        if switch_activated:
            self.progression_tracker.demonstrated_skills['switch_activation'] = True

        # Level 4: Exploration and discovery
        exploration_reward = self.exploration_calculator.calculate_exploration_reward(
            obs
        )
        reward += exploration_reward

        # Level 5: Time management
        time_reward = self.calculate_time_reward()
        reward += time_reward

        # Update movement-related skills
        if movement_reward > 0:
            self.progression_tracker.demonstrated_skills['precise_movement'] = True
            if not obs['in_air'] and prev_obs['in_air']:
                self.progression_tracker.demonstrated_skills['platform_landing'] = True
            if len(self.movement_calculator.velocity_history) >= 2:
                self.progression_tracker.demonstrated_skills['momentum_control'] = True

        return reward

    def update_progression_metrics(self):
        """Update progression metrics after each episode."""
        self.progression_tracker.update_progression_metrics()
        metrics = self.progression_tracker.get_progression_metrics()

        print("\nProgression Metrics:")
        print(f"Movement Success Rate: {metrics['movement_success_rate']:.3f}")
        print(
            f"Navigation Success Rate: {metrics['navigation_success_rate']:.3f}")
        print(
            f"Completion Success Rate: {metrics['completion_success_rate']:.3f}")
        print(f"Current Scales - Movement: {metrics['movement_scale']:.3f}, "
              f"Navigation: {metrics['navigation_scale']:.3f}, "
              f"Completion: {metrics['completion_scale']:.3f}\n")

    def reset(self):
        """Reset all components for new episode."""
        self.movement_calculator.reset()
        self.navigation_calculator.reset()
        self.exploration_calculator.reset()
        self.progression_tracker.reset()
