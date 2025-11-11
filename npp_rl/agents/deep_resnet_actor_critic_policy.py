"""Deep ResNet Actor-Critic Policy with separate feature extractors.

This module implements a sophisticated actor-critic policy using deep residual
networks with separate feature extractors for policy and value to avoid gradient
conflicts and improve learning stability.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from npp_rl.agents.masked_actor_critic_policy import MaskedActorCriticPolicy
from npp_rl.models.deep_resnet_mlp import DeepResNetMLPExtractor


class DeepResNetActorCriticPolicy(MaskedActorCriticPolicy):
    """Deep ResNet-based actor-critic with separate feature extractors.

    This policy extends MaskedActorCriticPolicy to use:
    1. Deep residual networks for policy and value heads
    2. Separate feature extractors for policy and value (gradient isolation)
    3. Dueling architecture for value function (optional)
    4. Modern activations (SiLU) and normalization (LayerNorm)

    Architecture:
        Observations →
            ├─→ Policy Feature Extractor → Policy ResNet → Action Logits
            └─→ Value Feature Extractor → Value ResNet → Value Estimate

    Key advantages over standard ActorCriticPolicy:
    - Deeper networks for complex sequential reasoning
    - Residual connections prevent gradient degradation
    - Separate extractors avoid gradient conflicts
    - Dueling architecture improves value estimation
    - Action masking support (inherited from MaskedActorCriticPolicy)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.SiLU,
        # Standard ActorCriticPolicy parameters
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,  # CRITICAL: Set to False for separate extractors
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Deep ResNet specific parameters
        use_residual: bool = True,
        use_layer_norm: bool = True,
        dueling: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize deep ResNet actor-critic policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture specification
                - If dict: {"pi": [512, 512, 384, 256, 256], "vf": [512, 384, 256]}
                - If list: Same architecture for both (not recommended)
            activation_fn: Activation function class (default: SiLU)
            ortho_init: Use orthogonal initialization
            use_sde: Use State-Dependent Exploration
            log_std_init: Initial log std for SDE
            full_std: Use full covariance for SDE
            use_expln: Use exponential mapping for SDE
            squash_output: Squash output for SDE
            features_extractor_class: Feature extractor class
            features_extractor_kwargs: Feature extractor kwargs
            share_features_extractor: Whether to share feature extractor (should be False)
            normalize_images: Normalize image observations
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            use_residual: Use residual connections in MLP
            use_layer_norm: Use LayerNorm in MLP
            dueling: Use dueling architecture for value
            dropout: Dropout rate for MLP layers
        """
        # Store custom parameters before calling super().__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.dueling = dueling
        self.dropout = dropout

        # Default network architecture if not provided
        if net_arch is None:
            net_arch = {
                "pi": [512, 512, 384, 256, 256],  # 5-layer policy
                "vf": [512, 384, 256],  # 3-layer value
            }

        # Force separate feature extractors for gradient isolation
        if share_features_extractor:
            import warnings

            warnings.warn(
                "DeepResNetActorCriticPolicy is designed to use separate feature extractors. "
                "Setting share_features_extractor=False to avoid gradient conflicts."
            )
            share_features_extractor = False

        # Initialize parent class
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Build deep ResNet MLP extractor.

        This overrides the standard MLP extractor from ActorCriticPolicy
        to use our custom DeepResNetMLPExtractor with residual connections,
        LayerNorm, and dueling architecture.
        """
        # Extract network architecture
        if isinstance(self.net_arch, dict):
            policy_layers = self.net_arch.get("pi", [256, 256, 128])
            value_layers = self.net_arch.get("vf", [256, 256, 128])
        else:
            # If net_arch is a list, use same for both
            policy_layers = self.net_arch
            value_layers = self.net_arch

        # Get number of actions for dueling advantage stream
        if isinstance(self.action_space, spaces.Discrete):
            num_actions = int(self.action_space.n)
        else:
            num_actions = int(self.action_space.shape[0])

        # Create deep ResNet MLP extractor
        self.mlp_extractor = DeepResNetMLPExtractor(
            feature_dim=self.features_dim,
            policy_layers=policy_layers,
            value_layers=value_layers,
            activation_fn=self.activation_fn,
            use_residual=self.use_residual,
            use_layer_norm=self.use_layer_norm,
            dueling=self.dueling,
            num_actions=num_actions,
            dropout=self.dropout,
        )

    def forward(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with action masking support.

        This method is inherited from MaskedActorCriticPolicy and handles
        action masking automatically.

        Args:
            obs: Observation (may contain action_mask key)
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Call parent implementation which handles action masking
        return super().forward(obs, deterministic)

    def evaluate_actions(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions with action masking support.

        This method is inherited from MaskedActorCriticPolicy and handles
        action masking automatically.

        Args:
            obs: Observation (may contain action_mask key)
            actions: Actions to evaluate

        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Call parent implementation which handles action masking
        return super().evaluate_actions(obs, actions)

    def _predict(
        self,
        observation: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Predict action with action masking support.

        This method is inherited from MaskedActorCriticPolicy and handles
        action masking automatically.

        Args:
            observation: Observation (may contain action_mask key)
            deterministic: Whether to use deterministic actions

        Returns:
            Predicted actions
        """
        # Call parent implementation which handles action masking
        return super()._predict(observation, deterministic)

    def predict_values(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Predict values for observations.

        If using dueling architecture, this combines state value and advantages.

        Args:
            obs: Observation

        Returns:
            Value estimates [batch, 1]
        """
        # Extract action mask if present
        if isinstance(obs, dict):
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Extract features using value feature extractor
        features = self.extract_features(obs_for_features, self.vf_features_extractor)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            _, features_vf = features
        else:
            features_vf = features

        # Forward through value network
        if hasattr(self.mlp_extractor, "forward_value"):
            value_latent = self.mlp_extractor.forward_value(features_vf)
        else:
            _, value_latent = self.mlp_extractor(features_vf)

        # Get value estimate
        if self.dueling:
            # Dueling: V(s) + (A(s,a) - mean(A(s,*)))
            state_value, advantages = self.mlp_extractor.get_dueling_values(
                value_latent
            )
            advantage_mean = advantages.mean(dim=1, keepdim=True)
            values = state_value + (advantages - advantage_mean)
            # Return mean value across actions
            return values.mean(dim=1, keepdim=True)
        else:
            # Standard value
            return self.value_net(value_latent)

    def get_policy_latent(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Get policy latent representation for observation.

        Useful for auxiliary tasks that want to use policy features.

        Args:
            obs: Observation

        Returns:
            Policy latent features [batch, latent_dim_pi]
        """
        # Extract action mask if present
        if isinstance(obs, dict):
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Extract features using policy feature extractor
        features = self.extract_features(obs_for_features, self.features_extractor)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            features_pi, _ = features
        else:
            features_pi = features

        # Forward through policy network
        if hasattr(self.mlp_extractor, "forward_policy"):
            policy_latent = self.mlp_extractor.forward_policy(features_pi)
        else:
            policy_latent, _ = self.mlp_extractor(features_pi)

        return policy_latent

    def get_value_latent(
        self,
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Get value latent representation for observation.

        Useful for auxiliary tasks that want to use value features.

        Args:
            obs: Observation

        Returns:
            Value latent features [batch, latent_dim_vf]
        """
        # Extract action mask if present
        if isinstance(obs, dict):
            obs_for_features = {k: v for k, v in obs.items() if k != "action_mask"}
        else:
            obs_for_features = obs

        # Extract features using value feature extractor
        features = self.extract_features(obs_for_features, self.vf_features_extractor)

        # Handle tuple return when using separate feature extractors
        if isinstance(features, tuple):
            _, features_vf = features
        else:
            features_vf = features

        # Forward through value network
        if hasattr(self.mlp_extractor, "forward_value"):
            value_latent = self.mlp_extractor.forward_value(features_vf)
        else:
            _, value_latent = self.mlp_extractor(features_vf)

        return value_latent
