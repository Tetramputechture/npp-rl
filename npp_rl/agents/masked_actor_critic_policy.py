"""Actor-Critic policy with action masking support for invalid actions.

This policy applies action masks during both action selection and policy evaluation,
ensuring masked actions have zero probability of being selected.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Actor-Critic policy with action masking support for invalid actions.

    This policy applies action masks during both action selection and policy evaluation,
    ensuring masked actions have zero probability of being selected.

    Action masking is particularly useful for:
    - Preventing selection of actions that have no effect (e.g., useless jumps)
    - Reducing exploration waste on invalid actions
    - Improving sample efficiency

    The implementation overrides the distribution creation to apply masks, which is
    the cleanest way to integrate with stable-baselines3's PPO algorithm.
    """

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None
    ) -> CategoricalDistribution:
        """
        Retrieve action distribution given the latent codes.

        This method is called by predict() path only. Since we don't have access
        to observations here, we can't apply masking. The masking is instead
        applied in forward() and evaluate_actions() which are used during training.

        Args:
            latent_pi: Latent code for the actor
            latent_sde: Latent code for state-dependent exploration (not used for discrete)

        Returns:
            Action distribution
        """
        # Get action logits from the actor network
        action_logits = self.action_net(latent_pi)

        # Create and return distribution
        # Note: Masking handled in forward() and evaluate_actions()
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic).

        Args:
            obs: Observation
            deterministic: Whether to sample or use deterministic actions

        Returns:
            action, value and log probability of the action
        """
        # Extract action mask before feature extraction (if present)
        # This prevents feature extractor from trying to process it
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # Create a copy without action_mask for feature extraction
            # to avoid issues if feature extractor expects certain keys
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Preprocess the observation if needed
        features = self.extract_features(obs_for_features)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Apply action mask if present
        if action_mask is not None:
            action_logits = self._apply_action_mask(action_logits, action_mask)

        # Create distribution and sample
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Get values
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(
        self, obs: Union[torch.Tensor, Dict[str, torch.Tensor]], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        Args:
            obs: Observation
            actions: Actions

        Returns:
            estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Extract action mask before feature extraction (if present)
        # This prevents feature extractor from trying to process it
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # Create a copy without action_mask for feature extraction
            # to avoid issues if feature extractor expects certain keys
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Preprocess the observation if needed
        features = self.extract_features(obs_for_features)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get action logits
        action_logits = self.action_net(latent_pi)

        # Apply action mask if present
        if action_mask is not None:
            action_logits = self._apply_action_mask(action_logits, action_mask)

        # Create distribution and evaluate
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Get values
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def _apply_action_mask(
        self, action_logits: torch.Tensor, action_mask: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Apply action mask to logits by setting masked actions to -inf.

        Args:
            action_logits: Raw action logits from actor network
            action_mask: Boolean mask where True = valid, False = invalid
                        (can be int8/uint8, will be converted to bool)

        Returns:
            Masked action logits
        """
        # Convert mask to tensor if needed
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask)

        # Ensure mask is boolean type (it might be int8 from numpy)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask.bool()

        # Ensure mask is on the same device as logits
        action_mask = action_mask.to(action_logits.device)

        # Handle batch dimension - mask might be (batch, actions) or just (actions,)
        if action_mask.dim() == 1:
            # Single mask for all batch elements, expand it
            action_mask = action_mask.unsqueeze(0).expand_as(action_logits)

        # Set masked actions to -inf (will have zero probability after softmax)
        masked_logits = torch.where(
            action_mask,
            action_logits,
            torch.tensor(
                float("-inf"), dtype=action_logits.dtype, device=action_logits.device
            ),
        )

        return masked_logits
