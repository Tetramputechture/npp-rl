"""Progression tracker for managing skill progression and reward scaling."""
from typing import Dict
import numpy as np


class ProgressionTracker:
    """Tracks agent's skill progression and manages reward scaling."""

    # Curriculum learning constants
    MOVEMENT_MASTERY_THRESHOLD = 0.7
    NAVIGATION_MASTERY_THRESHOLD = 0.8

    # Maximum scale limits
    MAX_MOVEMENT_SCALE = 0.5
    MAX_NAVIGATION_SCALE = 0.2
    MAX_COMPLETION_SCALE = 0.15

    # Minimum scale limits
    MIN_MOVEMENT_SCALE = 0.01
    MIN_NAVIGATION_SCALE = 0.01
    MIN_COMPLETION_SCALE = 0.01

    def __init__(self):
        """Initialize progression tracker."""
        # Success rates
        self.movement_success_rate = 0.0
        self.navigation_success_rate = 0.0
        self.level_completion_rate = 0.0

        # Reward scales
        self.movement_scale = 0.01
        self.navigation_scale = 0.01
        self.completion_scale = 0.01

        # Skill tracking
        self.demonstrated_skills = {
            'precise_movement': False,
            'platform_landing': False,
            'momentum_control': False,
            'switch_activation': False,
            'exit_reaching': False
        }

    def get_reward_scales(self) -> Dict[str, float]:
        """Get current reward scales.

        Returns:
            Dict[str, float]: Current reward scales
        """
        return {
            'movement': self.movement_scale,
            'navigation': self.navigation_scale,
            'completion': self.completion_scale
        }

    def update_progression_metrics(self):
        """Update progression metrics and adjust reward scaling."""
        # Use a small alpha for stable progression
        alpha = 0.05

        # Calculate skill demonstration rates
        movement_success = float(
            self.demonstrated_skills['precise_movement'] and
            self.demonstrated_skills['platform_landing']
        )

        navigation_success = float(
            self.demonstrated_skills['switch_activation']
        )

        completion_success = float(
            self.demonstrated_skills['exit_reaching']
        )

        # Update success rates with bounded EMA
        self.movement_success_rate = np.clip(
            (1 - alpha) * self.movement_success_rate + alpha * movement_success,
            0.0, 1.0
        )

        self.navigation_success_rate = np.clip(
            (1 - alpha) * self.navigation_success_rate +
            alpha * navigation_success,
            0.0, 1.0
        )

        self.level_completion_rate = np.clip(
            (1 - alpha) * self.level_completion_rate + alpha * completion_success,
            0.0, 1.0
        )

        # Adjust scales based on mastery
        if self.movement_success_rate > self.MOVEMENT_MASTERY_THRESHOLD:
            # Gradually reduce movement rewards as mastery increases
            movement_reduction = 0.98
            navigation_increase = 1.02

            self.movement_scale = np.clip(
                self.movement_scale * movement_reduction,
                self.MIN_MOVEMENT_SCALE,
                self.MAX_MOVEMENT_SCALE
            )

            self.navigation_scale = np.clip(
                self.navigation_scale * navigation_increase,
                self.MIN_NAVIGATION_SCALE,
                self.MAX_NAVIGATION_SCALE
            )

        if self.navigation_success_rate > self.NAVIGATION_MASTERY_THRESHOLD:
            # Similarly adjust navigation and completion scales
            navigation_reduction = 0.98
            completion_increase = 1.02

            self.navigation_scale = np.clip(
                self.navigation_scale * navigation_reduction,
                self.MIN_NAVIGATION_SCALE,
                self.MAX_NAVIGATION_SCALE
            )

            self.completion_scale = np.clip(
                self.completion_scale * completion_increase,
                self.MIN_COMPLETION_SCALE,
                self.MAX_COMPLETION_SCALE
            )

    def get_progression_metrics(self) -> Dict[str, float]:
        """Get current progression metrics.

        Returns:
            Dict[str, float]: Current progression metrics
        """
        return {
            'movement_success_rate': self.movement_success_rate,
            'navigation_success_rate': self.navigation_success_rate,
            'completion_success_rate': self.level_completion_rate,
            'movement_scale': self.movement_scale,
            'navigation_scale': self.navigation_scale,
            'completion_scale': self.completion_scale
        }

    def reset(self):
        """Reset progression tracker state."""
        self.movement_success_rate = 0.0
        self.navigation_success_rate = 0.0
        self.level_completion_rate = 0.0
        self.movement_scale = 0.01
        self.navigation_scale = 0.01
        self.completion_scale = 0.01
        self.demonstrated_skills = {
            'precise_movement': False,
            'platform_landing': False,
            'momentum_control': False,
            'switch_activation': False,
            'exit_reaching': False
        }
