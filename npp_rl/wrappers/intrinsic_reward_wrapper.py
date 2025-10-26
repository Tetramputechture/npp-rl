"""
Environment wrapper for adding intrinsic rewards from ICM.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, Union
from collections import deque

from npp_rl.intrinsic.icm import ICMTrainer
from npp_rl.intrinsic.utils import (
    extract_features_from_policy,
    clip_intrinsic_reward,
    RewardCombiner,
)


class IntrinsicRewardWrapper(gym.Wrapper):
    """
    Environment wrapper that adds intrinsic rewards from ICM.

    This wrapper intercepts environment steps and adds intrinsic rewards
    based on the ICM's prediction error. It maintains a buffer of recent
    observations to compute features and train the ICM.
    """

    def __init__(
        self,
        env: gym.Env,
        icm_trainer: ICMTrainer,
        policy: Optional[Any] = None,
        alpha: float = 0.1,
        r_int_clip: float = 1.0,
        update_frequency: int = 1,
        buffer_size: int = 10000,
        enable_logging: bool = True,
    ):
        """
        Initialize intrinsic reward wrapper.

        Args:
            env: Base environment
            icm_trainer: Trained ICM module
            policy: Policy to extract features from (set later if None)
            alpha: Weight for combining intrinsic and extrinsic rewards
            r_int_clip: Maximum intrinsic reward value
            update_frequency: How often to update ICM (every N steps)
            buffer_size: Size of experience buffer for ICM training
            enable_logging: Whether to log intrinsic reward statistics
        """
        super().__init__(env)

        self.icm_trainer = icm_trainer
        self.policy = policy
        self.reward_combiner = RewardCombiner(alpha_start=alpha)
        self.r_int_clip = r_int_clip
        self.update_frequency = update_frequency
        self.enable_logging = enable_logging

        # Experience buffer for ICM training
        self.buffer_size = buffer_size
        self.experience_buffer = deque(maxlen=buffer_size)

        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.current_obs = None
        self.current_features = None

        # Logging
        self.episode_stats = {
            "r_ext_sum": 0.0,
            "r_int_sum": 0.0,
            "r_total_sum": 0.0,
            "step_count": 0,
        }

        self.global_stats = {
            "total_episodes": 0,
            "mean_r_ext": 0.0,
            "mean_r_int": 0.0,
            "icm_updates": 0,
        }

    def set_policy(self, policy: Any):
        """Set the policy for feature extraction."""
        self.policy = policy

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and initialize state tracking."""
        obs, info = self.env.reset(**kwargs)

        # Log previous episode stats
        if self.episode_count > 0 and self.enable_logging:
            self._log_episode_stats(info)

        # Reset episode tracking
        self.current_obs = obs
        self.current_features = None
        self.episode_stats = {
            "r_ext_sum": 0.0,
            "r_int_sum": 0.0,
            "r_total_sum": 0.0,
            "step_count": 0,
        }
        self.episode_count += 1

        return obs, info

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step environment and add intrinsic reward.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Take environment step
        next_obs, ext_reward, terminated, truncated, info = self.env.step(action)

        # Compute intrinsic reward if policy is available
        int_reward = 0.0
        if self.policy is not None and self.current_obs is not None:
            int_reward = self._compute_intrinsic_reward(
                self.current_obs, next_obs, action
            )

        # Combine rewards
        total_reward = ext_reward + self.reward_combiner.get_alpha() * int_reward

        # Update episode statistics
        self.episode_stats["r_ext_sum"] += ext_reward
        self.episode_stats["r_int_sum"] += int_reward
        self.episode_stats["r_total_sum"] += total_reward
        self.episode_stats["step_count"] += 1

        # Update state
        self.current_obs = next_obs
        self.step_count += 1

        # Add intrinsic reward info
        if self.enable_logging:
            info.update(
                {
                    "r_ext": ext_reward,
                    "r_int": int_reward,
                    "r_total": total_reward,
                    "alpha": self.reward_combiner.get_alpha(),
                }
            )

        return next_obs, total_reward, terminated, truncated, info

    def _compute_intrinsic_reward(
        self, current_obs: Any, next_obs: Any, action: Union[int, np.ndarray]
    ) -> float:
        """
        Compute intrinsic reward using ICM.

        Args:
            current_obs: Current observation
            next_obs: Next observation
            action: Action taken

        Returns:
            Intrinsic reward value
        """
        # Extract features from observations
        with torch.no_grad():
            current_features = extract_features_from_policy(
                self.policy, self._prepare_obs_for_policy(current_obs)
            )
            next_features = extract_features_from_policy(
                self.policy, self._prepare_obs_for_policy(next_obs)
            )

        # Ensure action is tensor
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).long()
        else:
            action_tensor = torch.tensor([action], dtype=torch.long)

        # Add batch dimension if needed
        if current_features.dim() == 1:
            current_features = current_features.unsqueeze(0)
        if next_features.dim() == 1:
            next_features = next_features.unsqueeze(0)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)

        # Compute intrinsic reward
        int_rewards = self.icm_trainer.get_intrinsic_reward(
            current_features, next_features, action_tensor
        )

        # Clip and return scalar
        int_reward = clip_intrinsic_reward(int_rewards, clip_max=self.r_int_clip)[0]

        # Store experience for ICM training
        self.experience_buffer.append(
            {
                "current_features": current_features.cpu(),
                "next_features": next_features.cpu(),
                "action": action_tensor.cpu(),
            }
        )

        # Update ICM periodically
        if (
            self.step_count % self.update_frequency == 0
            and len(self.experience_buffer) >= 32
        ):
            self._update_icm()

        return float(int_reward)

    def _prepare_obs_for_policy(self, obs: Any) -> Any:
        """
        Prepare observation for policy feature extraction.

        Args:
            obs: Raw observation

        Returns:
            Observation formatted for policy
        """
        # Handle different observation formats
        if isinstance(obs, dict):
            # Convert numpy arrays to tensors
            prepared_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    prepared_obs[key] = torch.from_numpy(value).float()
                else:
                    prepared_obs[key] = value
            return prepared_obs
        elif isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float()
        else:
            return obs

    def _update_icm(self):
        """Update ICM using buffered experience."""
        if len(self.experience_buffer) < 32:
            return

        # Sample batch from buffer
        batch_size = min(64, len(self.experience_buffer))
        indices = np.random.choice(
            len(self.experience_buffer), batch_size, replace=False
        )

        batch_current = []
        batch_next = []
        batch_actions = []

        for idx in indices:
            exp = self.experience_buffer[idx]
            batch_current.append(exp["current_features"])
            batch_next.append(exp["next_features"])
            batch_actions.append(exp["action"])

        # Stack into tensors
        current_features = torch.cat(batch_current, dim=0)
        next_features = torch.cat(batch_next, dim=0)
        actions = torch.cat(batch_actions, dim=0)

        # Update ICM
        stats = self.icm_trainer.update(current_features, next_features, actions)
        self.global_stats["icm_updates"] += 1

        if self.enable_logging and self.global_stats["icm_updates"] % 100 == 0:
            print(f"ICM Update {self.global_stats['icm_updates']}: {stats}")

    def _log_episode_stats(self, info: Dict[str, Any]):
        """Log episode statistics."""
        # Add episode stats to info (with both old and new key names for compatibility)
        info.update(
            {
                "episode_r_ext_sum": self.episode_stats["r_ext_sum"],
                "episode_r_int_sum": self.episode_stats["r_int_sum"],
                "episode_r_total_sum": self.episode_stats["r_total_sum"],
                "episode_length": self.episode_stats["step_count"],
                # Add keys that EnhancedTensorBoardCallback expects
                "r_ext_episode": self.episode_stats["r_ext_sum"],
                "r_int_episode": self.episode_stats["r_int_sum"],
            }
        )

        # Update global statistics
        self.global_stats["total_episodes"] += 1
        alpha = 0.1  # Exponential moving average factor
        self.global_stats["mean_r_ext"] = (1 - alpha) * self.global_stats[
            "mean_r_ext"
        ] + alpha * self.episode_stats["r_ext_sum"]
        self.global_stats["mean_r_int"] = (1 - alpha) * self.global_stats[
            "mean_r_int"
        ] + alpha * self.episode_stats["r_int_sum"]

    def get_stats(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            "episode_stats": self.episode_stats.copy(),
            "global_stats": self.global_stats.copy(),
            "icm_stats": self.icm_trainer.get_recent_stats(),
        }
