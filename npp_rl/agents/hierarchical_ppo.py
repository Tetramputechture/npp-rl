"""
Hierarchical PPO agent with high-level and low-level policies.

This module implements a hierarchical PPO architecture where:
- High-level policy selects subtasks based on reachability features
- Low-level policy executes actions for the current subtask
- Both policies share a common feature extractor but have separate heads

This is the enhanced Phase 2 Task 2.1 implementation with true two-level
policy architecture and coordinated training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from npp_rl.models.hierarchical_policy import (
    HierarchicalPolicyNetwork,
    HierarchicalExperienceBuffer,
)


class HierarchicalActorCriticPolicy(ActorCriticPolicy):
    """
    Hierarchical Actor-Critic policy with true two-level architecture.

    This policy implements a sophisticated hierarchical architecture where:
    - High-level policy selects subtasks based on reachability features
    - Low-level policy executes actions for the current subtask
    - Both policies share a common feature extractor (HGTMultimodalExtractor)
    - Coordinated training with different update frequencies
    - ICM integration at low-level for enhanced exploration

    Phase 2 Task 2.1 Implementation:
    - True two-level hierarchy (not just subtask conditioning)
    - High-level updates every 50-100 steps (configurable)
    - Low-level updates every step
    - Subtask-specific embeddings and context
    - Proper transition management
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Dict[str, list]] = None,
        activation_fn: nn.Module = nn.ReLU,
        # Standard ActorCriticPolicy parameters (MUST match base class order!)
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: BaseFeaturesExtractor = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Hierarchical-specific parameters
        high_level_update_frequency: int = 50,
        max_steps_per_subtask: int = 500,
        use_icm: bool = True,
    ):
        """
        Initialize hierarchical actor-critic policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture (passed to base class, not used directly)
            activation_fn: Activation function (passed to base class)
            ortho_init: Use orthogonal initialization
            use_sde: Whether to use State Dependent Exploration
            log_std_init: Initial value for log std (if using SDE)
            full_std: Use full std instead of diagonal (if using SDE)
            use_expln: Use expln function for std (if using SDE)
            squash_output: Squash output using tanh
            features_extractor_class: Feature extractor class
            features_extractor_kwargs: Feature extractor kwargs
            share_features_extractor: Share features extractor between policy and value
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            high_level_update_frequency: Steps between high-level updates
            max_steps_per_subtask: Maximum steps per subtask
            use_icm: Whether to use ICM for exploration
        """
        # Store hierarchical parameters before calling super().__init__()
        # These will be used in _build_mlp_extractor()
        self.high_level_update_frequency = high_level_update_frequency
        self.max_steps_per_subtask = max_steps_per_subtask
        self.use_icm = use_icm

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        # Experience buffer for hierarchical training
        self.experience_buffer = HierarchicalExperienceBuffer(
            buffer_size=2048,
            high_level_update_frequency=high_level_update_frequency,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Build hierarchical policy network instead of default MLP extractor.

        This is called by the base ActorCriticPolicy.__init__() and must
        create self.mlp_extractor. We replace it with our HierarchicalPolicyNetwork.
        """
        # Create hierarchical policy network
        self.mlp_extractor = HierarchicalPolicyNetwork(
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            high_level_update_frequency=self.high_level_update_frequency,
            max_steps_per_subtask=self.max_steps_per_subtask,
            use_icm=self.use_icm,
        )

        # Override action and value networks (handled by hierarchical network)
        # We set these to Identity because the HierarchicalPolicyNetwork
        # handles action and value computation internally
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _ensure_obs_dict(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Ensure observation is in dictionary format.

        Args:
            obs: Observation tensor or dictionary

        Returns:
            Observation dictionary with all required keys

        Note:
            When obs is not a dict, this creates a minimal observation dict
            with placeholder values. In practice, the environment should provide
            properly structured dictionary observations.
        """
        if isinstance(obs, dict):
            return obs

        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        return {
            "observation": obs,
            "reachability_features": torch.zeros(batch_size, 8, device=obs.device),
            "switch_states": torch.zeros(batch_size, 5, device=obs.device),
            "ninja_position": torch.zeros(batch_size, 2, device=obs.device),
            "time_remaining": torch.ones(batch_size, 1, device=obs.device),
        }

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical policy.

        Args:
            obs: Observation tensor (should be dict with multiple components)
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (actions, values, log_probs)
        """
        obs_dict = self._ensure_obs_dict(obs)

        # Forward through hierarchical network
        actions, values, log_probs, info = self.mlp_extractor(
            obs_dict,
            deterministic=deterministic,
            update_subtask=True,
        )

        return actions, values, log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            obs: Observations
            actions: Actions to evaluate

        Returns:
            Tuple of (values, log_probs, entropy)
        """
        obs_dict = self._ensure_obs_dict(obs)
        return self.mlp_extractor.evaluate_actions(obs_dict, actions)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for given observations."""
        obs_dict = self._ensure_obs_dict(obs)
        return self.mlp_extractor.get_value(obs_dict)

    def reset_episode(self):
        """Reset policy state for new episode."""
        self.mlp_extractor.reset_episode()

    def get_subtask_metrics(self) -> Dict[str, Any]:
        """Get metrics about current subtask performance."""
        return self.mlp_extractor.get_subtask_metrics()


class HierarchicalPPO:
    """
    Hierarchical PPO wrapper for the two-level policy architecture.

    This class provides a simplified interface for using the hierarchical
    architecture with coordinated training of high-level and low-level policies.

    Features:
    - Two-level policy architecture (strategic + tactical)
    - Coordinated training with different update frequencies
    - ICM integration for enhanced exploration
    - Subtask-aware curiosity modulation
    """

    def __init__(
        self,
        policy_class=HierarchicalActorCriticPolicy,
        high_level_update_frequency: int = 50,
        max_steps_per_subtask: int = 500,
        use_icm: bool = True,
        **ppo_kwargs,
    ):
        """
        Initialize hierarchical PPO.

        Args:
            policy_class: Policy class to use (default: HierarchicalActorCriticPolicy)
            high_level_update_frequency: Steps between high-level updates
            max_steps_per_subtask: Maximum steps per subtask
            use_icm: Whether to use ICM for exploration
            **ppo_kwargs: Additional arguments for PPO
        """
        self.policy_class = policy_class
        self.high_level_update_frequency = high_level_update_frequency
        self.max_steps_per_subtask = max_steps_per_subtask
        self.use_icm = use_icm

        # Add hierarchical-specific parameters to policy kwargs
        if "policy_kwargs" not in ppo_kwargs:
            ppo_kwargs["policy_kwargs"] = {}

        ppo_kwargs["policy_kwargs"]["high_level_update_frequency"] = (
            high_level_update_frequency
        )
        ppo_kwargs["policy_kwargs"]["max_steps_per_subtask"] = max_steps_per_subtask
        ppo_kwargs["policy_kwargs"]["use_icm"] = use_icm

        self.ppo_kwargs = ppo_kwargs
        self.ppo_model = None

    def create_model(self, env, **additional_kwargs):
        """Create the PPO model with hierarchical policy."""
        from stable_baselines3 import PPO

        # Merge additional kwargs
        final_kwargs = {**self.ppo_kwargs, **additional_kwargs}
        final_kwargs["policy"] = self.policy_class

        self.ppo_model = PPO(env=env, **final_kwargs)
        return self.ppo_model

    def learn(self, *args, **kwargs):
        """Train the hierarchical PPO model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.learn(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Make predictions with the hierarchical model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Save the hierarchical model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Load a hierarchical model."""
        from stable_baselines3 import PPO

        self.ppo_model = PPO.load(*args, **kwargs)
        return self.ppo_model
