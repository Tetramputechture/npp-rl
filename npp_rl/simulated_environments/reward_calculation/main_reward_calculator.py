"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any
from npp_rl.simulated_environments.reward_calculation.base_reward_calculator import BaseRewardCalculator


class RewardCalculator(BaseRewardCalculator):
    """
    A reward calculator for the N++ environment.
    """

    def __init__(self):
        """Initialize reward calculator.
        """
        super().__init__()

    def calculate_reward(self, obs: Dict[str, Any]) -> float:
        """Calculate reward.

        Args:
            obs: Current game state

        Returns:
            float: Total reward for the transition
        """
        if obs.get('player_dead', False):
            return self.DEATH_PENALTY

        # Initialize total reward
        reward = 0.0

        # Subtract BASE_TIME_PENALTY each tick
        reward -= self.BASE_TIME_PENALTY

        # if player has won, add TERMINAL REWARD
        if obs.get('player_won', False):
            reward += self.TERMINAL_REWARD

        return reward
