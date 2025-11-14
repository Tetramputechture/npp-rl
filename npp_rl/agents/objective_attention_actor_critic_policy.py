"""Actor-Critic policy with deep ResNet, objective-specific attention, and dueling.

This policy combines three powerful architectural improvements:
1. Deep ResNet MLP with residual connections (5-layer policy, 3-layer value)
2. Objective-specific attention head for variable locked doors (1-16)
3. Dueling value architecture (V(s) + A(s,a))
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union

from npp_rl.agents.deep_resnet_actor_critic_policy import DeepResNetActorCriticPolicy
from npp_rl.models.objective_attention import (
    ObjectiveAttentionPolicy,
    ObjectiveFeatureExtractor,
)
from npp_rl.models.auxiliary_tasks import AuxiliaryTaskHeads


class ObjectiveAttentionActorCriticPolicy(DeepResNetActorCriticPolicy):
    """Deep ResNet policy with objective-specific attention and dueling architecture.

    Combines three architectural improvements (all always enabled):
    1. Deep ResNet MLP extractor (from DeepResNetActorCriticPolicy)
       - 5-layer policy network with residual connections
       - 3-layer value network with residual connections
       - SiLU activation and LayerNorm
    2. Objective-specific attention head (replaces standard action_net)
       - Attention over variable objectives (1-16 locked doors + exit)
       - Permutation invariant over locked doors
       - Multi-head attention (8 heads) for goal prioritization
    3. Dueling value architecture (always enabled)
       - Separate V(s) and A(s,a) streams in value MLP
       - Better value estimation and gradient flow
       - Integrated into DeepResNetMLPExtractor

    Total parameters: ~15-18M
    - Deep ResNet MLPs: ~6-8M (policy + value with residual connections)
    - Objective attention: ~1.5M
    - Dueling: integrated into ResNet value stream
    - Feature extractors: ~7M (when using separate extractors)

    Key features:
    - Handles variable number of locked doors (1-16) with permutation invariance
    - Deep reasoning chains through residual connections
    - Dynamic goal prioritization through attention
    - Better value estimation through dueling decomposition (always on)
    - Gradient isolation through separate feature extractors
    """

    def __init__(
        self,
        *args,
        use_objective_attention: bool = True,
        use_auxiliary_death_head: bool = True,
        **kwargs,
    ):
        """Initialize policy with deep ResNet, objective attention, dueling, and auxiliary death head.

        Note: Dueling is always enabled (dueling=True forced).

        Args:
            use_objective_attention: Whether to use objective attention for policy head
            use_auxiliary_death_head: Whether to enable auxiliary death prediction head
            *args, **kwargs: Arguments passed to DeepResNetActorCriticPolicy
                - Most notably: use_residual, use_layer_norm, dropout
                - Also: net_arch, activation_fn, share_features_extractor, etc.
                - Note: dueling is forced to True
        """
        # Force dueling to be enabled
        kwargs["dueling"] = True

        # Initialize parent (DeepResNetActorCriticPolicy)
        # This sets up:
        # 1. Deep ResNet MLP extractor (5-layer policy, 3-layer value)
        # 2. Residual connections, LayerNorm, SiLU activation
        # 3. Dueling architecture in MLP extractor (forced to True)
        super().__init__(*args, **kwargs)

        self.use_objective_attention = use_objective_attention
        self.use_auxiliary_death_head = use_auxiliary_death_head

        # Replace action_net with objective attention module
        # The parent's action_net is a simple Linear layer, we replace it with attention
        if use_objective_attention:
            policy_latent_dim = self._get_policy_latent_dim()
            self.action_net = ObjectiveAttentionPolicy(
                policy_feature_dim=policy_latent_dim,
                objective_embed_dim=64,
                attention_dim=512,
                num_heads=8,
                max_locked_doors=16,
                num_actions=self.action_space.n,
            )
            self.objective_extractor = ObjectiveFeatureExtractor(max_locked_doors=16)

        # Add auxiliary death prediction head for multi-task learning
        # Uses policy latent features to predict death probability
        if use_auxiliary_death_head:
            policy_latent_dim = self._get_policy_latent_dim()
            self.auxiliary_heads = AuxiliaryTaskHeads(
                feature_dim=policy_latent_dim,
                max_objectives=34,  # 1 exit + 16 locked doors + 16 switches
            )
        else:
            self.auxiliary_heads = None

        # Note: Dueling value head is handled by parent's DeepResNetMLPExtractor (always enabled).
        # The parent's mlp_extractor has built-in dueling support integrated with ResNet layers.

    def _get_policy_latent_dim(self) -> int:
        """Get the dimension of policy latent features.

        For DeepResNetMLPExtractor, this is the last layer size of the policy network.
        """
        if hasattr(self, "mlp_extractor") and hasattr(
            self.mlp_extractor, "latent_dim_pi"
        ):
            return self.mlp_extractor.latent_dim_pi
        # Fallback: the default net_arch last layer for policy
        return 256

    def _get_value_latent_dim(self) -> int:
        """Get the dimension of value latent features.

        For DeepResNetMLPExtractor, this is the last layer size of the value network.
        """
        if hasattr(self, "mlp_extractor") and hasattr(
            self.mlp_extractor, "latent_dim_vf"
        ):
            return self.mlp_extractor.latent_dim_vf
        # Fallback: the default net_arch last layer for value
        return 256

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
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # VALIDATION: action_mask must be present in dictionary observations
            if action_mask is None:
                raise ValueError(
                    "action_mask not found in observation dictionary! "
                    "This is required for masked action selection. "
                    f"Available keys: {list(obs.keys())}"
                )
            # Create a copy without action_mask for feature extraction
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, features_vf = features
        else:
            features_pi = features_vf = features

        # Get policy/value latent
        if hasattr(self.mlp_extractor, "forward_policy") and hasattr(
            self.mlp_extractor, "forward_value"
        ):
            # Separate forward methods (e.g., DeepResNet)
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
            latent_vf = self.mlp_extractor.forward_value(features_vf)
        else:
            # Standard MLP extractor
            latent_pi, latent_vf = self.mlp_extractor(features_pi)

        # Get action logits with attention over objectives
        if self.use_objective_attention and hasattr(self, "objective_extractor"):
            objective_features = self.objective_extractor(obs_for_features)
            action_logits, attn_weights = self.action_net(latent_pi, objective_features)
            # Store attention weights for potential logging/visualization
            self._last_attn_weights = attn_weights
        else:
            action_logits = self.action_net(latent_pi)

        # Compute auxiliary death predictions if enabled
        if self.use_auxiliary_death_head and self.auxiliary_heads is not None:
            auxiliary_predictions = self.auxiliary_heads(latent_pi)
            # Store for loss computation during training
            self._last_auxiliary_predictions = auxiliary_predictions
        else:
            self._last_auxiliary_predictions = None

        # Apply action masking (REQUIRED - validated above)
        # action_mask is guaranteed to be present by validation above
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
        Evaluate actions according to the current policy.

        Args:
            obs: Observation
            actions: Actions

        Returns:
            estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Extract action mask before feature extraction (if present)
        action_mask = None
        if isinstance(obs, dict):
            action_mask = obs.get("action_mask", None)
            # VALIDATION: action_mask must be present in dictionary observations
            if action_mask is None:
                raise ValueError(
                    "action_mask not found in observation dictionary during evaluate_actions! "
                    "This is required for masked action selection. "
                    f"Available keys: {list(obs.keys())}"
                )
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, features_vf = features
        else:
            features_pi = features_vf = features

        # Get policy/value latent
        if hasattr(self.mlp_extractor, "forward_policy") and hasattr(
            self.mlp_extractor, "forward_value"
        ):
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
            latent_vf = self.mlp_extractor.forward_value(features_vf)
        else:
            latent_pi, latent_vf = self.mlp_extractor(features_pi)

        # Get action logits with attention over objectives
        if self.use_objective_attention and hasattr(self, "objective_extractor"):
            objective_features = self.objective_extractor(obs_for_features)
            action_logits, _ = self.action_net(latent_pi, objective_features)
        else:
            action_logits = self.action_net(latent_pi)

        # Compute auxiliary death predictions if enabled
        if self.use_auxiliary_death_head and self.auxiliary_heads is not None:
            auxiliary_predictions = self.auxiliary_heads(latent_pi)
            # Store for loss computation during training
            self._last_auxiliary_predictions = auxiliary_predictions
        else:
            self._last_auxiliary_predictions = None

        # Apply action masking (if present)
        if action_mask is not None:
            action_logits = self._apply_action_mask(action_logits, action_mask)

        # Create distribution and evaluate
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Get values
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Predict action with action mask support.

        Args:
            observation: Observation tensor or dict (may contain action_mask)
            deterministic: Whether to use deterministic action selection

        Returns:
            Predicted action tensor
        """
        # Extract action mask before feature extraction (if present)
        action_mask = None
        if isinstance(observation, dict):
            action_mask = observation.get("action_mask", None)
            obs_for_features = {
                k: v for k, v in observation.items() if k != "action_mask"
            }
        else:
            obs_for_features = observation

        # Get device from model parameters
        device = next(self.parameters()).device

        # Convert numpy arrays to tensors if needed
        if isinstance(obs_for_features, dict):
            obs_for_features = {
                k: torch.as_tensor(v, device=device) if isinstance(v, np.ndarray) else v
                for k, v in obs_for_features.items()
            }
        elif isinstance(obs_for_features, np.ndarray):
            obs_for_features = torch.as_tensor(obs_for_features, device=device)

        # Extract features (handles both shared and separate extractors)
        features = self.extract_features(obs_for_features, self.features_extractor)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, _ = features
        else:
            features_pi = features

        # Get policy latent
        if hasattr(self.mlp_extractor, "forward_policy"):
            latent_pi = self.mlp_extractor.forward_policy(features_pi)
        else:
            latent_pi, _ = self.mlp_extractor(features_pi)

        # Get action logits
        if self.use_objective_attention and hasattr(self, "objective_extractor"):
            objective_features = self.objective_extractor(obs_for_features)
            action_logits, _ = self.action_net(latent_pi, objective_features)
        else:
            action_logits = self.action_net(latent_pi)

        # Apply action mask if present
        if action_mask is not None:
            # Convert mask to tensor if needed
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask, device=device)
            elif not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, device=device)

            action_logits = self._apply_action_mask(action_logits, action_mask)

        # Create distribution and sample action
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)

        return actions

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor, latent_sde: Optional[torch.Tensor] = None
    ):
        """
        Retrieve action distribution given the latent codes.

        This method is called by some internal SB3 paths. Since we don't have access
        to observations here, we cannot provide objective_features, which will cause
        ObjectiveAttentionPolicy to raise an error.

        This is intentional - it forces proper use through forward(), evaluate_actions(),
        or _predict() which have access to observations.

        Args:
            latent_pi: Latent code for the actor
            latent_sde: Latent code for state-dependent exploration (not used)

        Returns:
            Action distribution
        """
        if self.use_objective_attention:
            raise RuntimeError(
                "ObjectiveAttentionActorCriticPolicy._get_action_dist_from_latent() cannot be used "
                "because it doesn't have access to observations needed for objective_features. "
                "This error indicates an unexpected code path in Stable-Baselines3. "
                "Please use forward(), evaluate_actions(), or _predict() instead."
            )

        # Fallback for non-attention mode (shouldn't happen)
        action_logits = self.action_net(latent_pi)
        if isinstance(action_logits, tuple):
            action_logits = action_logits[0]
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for visualization.

        Returns:
            Attention weights [batch, num_heads, 1, num_objectives] or None
        """
        return getattr(self, "_last_attn_weights", None)

    def get_auxiliary_predictions(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get last auxiliary predictions for loss computation.

        Returns:
            Dictionary with auxiliary predictions (death_prob, time_to_goal, next_subgoal_logits)
            or None if auxiliary heads not enabled
        """
        return getattr(self, "_last_auxiliary_predictions", None)
