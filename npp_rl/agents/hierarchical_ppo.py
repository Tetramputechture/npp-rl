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
import numpy as np
from typing import Dict, Any, Tuple, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from gymnasium import spaces

from npp_rl.hrl.completion_controller import CompletionController, Subtask
from npp_rl.models.hierarchical_policy import (
    HierarchicalPolicyNetwork as HierarchicalPolicyNetworkV2,
    HierarchicalExperienceBuffer,
)
from npp_rl.hrl.high_level_policy import HighLevelPolicy, SubtaskTransitionManager
from npp_rl.hrl.subtask_policies import LowLevelPolicy, ICMIntegration


class LegacyHierarchicalPolicyNetwork(nn.Module):
    """
    Hierarchical policy network with high-level and low-level components.
    
    Architecture:
    - Shared feature extractor (HGT-based multimodal)
    - High-level policy head: subtask selection (4 actions)
    - Low-level policy head: movement actions (6 actions)
    - Shared value function head
    """
    
    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        high_level_actions: int = 4,  # Number of subtasks
        low_level_actions: int = 6,   # Number of movement actions
        net_arch: Optional[Dict[str, list]] = None,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        
        # Default network architecture
        if net_arch is None:
            net_arch = dict(pi=[256, 256], vf=[256, 256])
        
        # High-level policy network (subtask selection)
        high_level_layers = []
        prev_dim = features_dim + 4  # features + current subtask one-hot
        for layer_size in net_arch["pi"]:
            high_level_layers.extend([
                nn.Linear(prev_dim, layer_size),
                activation_fn(),
            ])
            prev_dim = layer_size
        high_level_layers.append(nn.Linear(prev_dim, high_level_actions))
        self.high_level_policy = nn.Sequential(*high_level_layers)
        
        # Low-level policy network (action execution)
        low_level_layers = []
        prev_dim = features_dim + 4  # features + current subtask one-hot
        for layer_size in net_arch["pi"]:
            low_level_layers.extend([
                nn.Linear(prev_dim, layer_size),
                activation_fn(),
            ])
            prev_dim = layer_size
        low_level_layers.append(nn.Linear(prev_dim, low_level_actions))
        self.low_level_policy = nn.Sequential(*low_level_layers)
        
        # Shared value function
        value_layers = []
        prev_dim = features_dim + 4  # features + current subtask one-hot
        for layer_size in net_arch["vf"]:
            value_layers.extend([
                nn.Linear(prev_dim, layer_size),
                activation_fn(),
            ])
            prev_dim = layer_size
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*value_layers)
        
        # Subtask switching frequency control
        self.subtask_switch_cooldown = 10  # Minimum steps between subtask switches
        self.last_subtask_switch = 0
        
    def forward(
        self, 
        obs: torch.Tensor, 
        current_subtask: torch.Tensor,
        step_count: int = 0,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical network.
        
        Args:
            obs: Observation tensor
            current_subtask: Current subtask as one-hot tensor
            step_count: Current step count for cooldown management
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (high_level_action, low_level_action, high_level_log_prob, low_level_log_prob, value)
        """
        # Extract features
        features = self.features_extractor(obs)
        
        # Combine features with current subtask
        combined_features = torch.cat([features, current_subtask], dim=-1)
        
        # High-level policy (subtask selection)
        high_level_logits = self.high_level_policy(combined_features)
        high_level_dist = CategoricalDistribution(high_level_logits.shape[-1])
        high_level_dist = high_level_dist.proba_distribution(high_level_logits)
        
        # Apply cooldown to high-level policy (reduce switching frequency)
        if step_count - self.last_subtask_switch < self.subtask_switch_cooldown:
            # Bias towards current subtask during cooldown
            current_subtask_idx = torch.argmax(current_subtask, dim=-1)
            high_level_logits = high_level_logits.clone()
            high_level_logits[range(len(current_subtask_idx)), current_subtask_idx] += 2.0
            high_level_dist = high_level_dist.proba_distribution(high_level_logits)
        
        if deterministic:
            high_level_action = high_level_dist.mode()
        else:
            high_level_action = high_level_dist.sample()
        high_level_log_prob = high_level_dist.log_prob(high_level_action)
        
        # Low-level policy (action execution)
        low_level_logits = self.low_level_policy(combined_features)
        low_level_dist = CategoricalDistribution(low_level_logits.shape[-1])
        low_level_dist = low_level_dist.proba_distribution(low_level_logits)
        
        if deterministic:
            low_level_action = low_level_dist.mode()
        else:
            low_level_action = low_level_dist.sample()
        low_level_log_prob = low_level_dist.log_prob(low_level_action)
        
        # Value function
        value = self.value_net(combined_features)
        
        return high_level_action, low_level_action, high_level_log_prob, low_level_log_prob, value
    
    def get_value(self, obs: torch.Tensor, current_subtask: torch.Tensor) -> torch.Tensor:
        """Get value estimate for current state."""
        features = self.features_extractor(obs)
        combined_features = torch.cat([features, current_subtask], dim=-1)
        return self.value_net(combined_features)


class HierarchicalActorCriticPolicy(ActorCriticPolicy):
    """
    Hierarchical Actor-Critic policy for PPO with completion controller integration.
    
    This policy integrates the completion controller for strategic subtask selection
    and implements a hierarchical architecture with high-level and low-level policies.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Dict[str, list]] = None,
        activation_fn: nn.Module = nn.ReLU,
        features_extractor_class: BaseFeaturesExtractor = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        completion_controller: Optional[CompletionController] = None,
    ):
        # Initialize completion controller
        self.completion_controller = completion_controller or CompletionController()
        
        # Modify action space to be hierarchical
        # We'll use the low-level action space for the base class
        # and handle high-level actions internally
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        
        # Replace the default policy network with hierarchical one
        self.mlp_extractor = LegacyHierarchicalPolicyNetwork(
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        
        # Override action and value networks (not used in hierarchical setup)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # Hierarchical state tracking
        self.current_subtask_tensor = None
        self.step_count = 0
        
    def _build_mlp_extractor(self) -> None:
        """Override to prevent building default MLP extractor."""
        pass
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Get current subtask from completion controller
        # Note: This would need to be called with actual obs/info in practice
        if self.current_subtask_tensor is None:
            # Initialize with default subtask
            batch_size = obs.shape[0]
            self.current_subtask_tensor = torch.zeros(batch_size, 4, device=obs.device)
            self.current_subtask_tensor[:, 0] = 1.0  # Default to NAVIGATE_TO_EXIT_SWITCH
        
        # Forward through hierarchical network
        high_level_action, low_level_action, high_level_log_prob, low_level_log_prob, value = \
            self.mlp_extractor(obs, self.current_subtask_tensor, self.step_count, deterministic)
        
        # For now, we use the low-level action as the primary action
        # The high-level action is used internally for subtask management
        actions = low_level_action
        log_probs = low_level_log_prob
        values = value.flatten()
        
        self.step_count += 1
        
        return actions, values, log_probs
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for given observations."""
        if self.current_subtask_tensor is None:
            batch_size = obs.shape[0]
            self.current_subtask_tensor = torch.zeros(batch_size, 4, device=obs.device)
            self.current_subtask_tensor[:, 0] = 1.0
            
        return self.mlp_extractor.get_value(obs, self.current_subtask_tensor).flatten()
    
    def update_subtask(self, obs_dict: Dict[str, Any], info_dict: Dict[str, Any]):
        """
        Update current subtask based on completion controller.
        
        Args:
            obs_dict: Dictionary of observations
            info_dict: Dictionary of environment info
        """
        # Update completion controller
        current_subtask = self.completion_controller.get_current_subtask(obs_dict, info_dict)
        
        # Convert to tensor
        if self.current_subtask_tensor is None:
            batch_size = 1  # Assume single environment for now
            self.current_subtask_tensor = torch.zeros(batch_size, 4)
        
        # Update subtask tensor
        self.current_subtask_tensor.zero_()
        self.current_subtask_tensor[0, current_subtask.value] = 1.0
        
        # Update completion controller state
        self.completion_controller.step(obs_dict, info_dict)
    
    def reset_episode(self):
        """Reset policy state for new episode."""
        self.completion_controller.reset()
        self.current_subtask_tensor = None
        self.step_count = 0
    
    def get_subtask_metrics(self) -> Dict[str, Any]:
        """Get metrics about current subtask performance."""
        return self.completion_controller.get_subtask_metrics()


class HierarchicalPPO:
    """
    Hierarchical PPO wrapper that manages the hierarchical policy and training.
    
    This class provides a simplified interface for using hierarchical PPO with
    the completion controller integration.
    """
    
    def __init__(
        self,
        policy_class=HierarchicalActorCriticPolicy,
        completion_controller: Optional[CompletionController] = None,
        **ppo_kwargs
    ):
        """
        Initialize hierarchical PPO.
        
        Args:
            policy_class: Policy class to use
            completion_controller: Completion controller instance
            **ppo_kwargs: Additional arguments for PPO
        """
        self.completion_controller = completion_controller or CompletionController()
        
        # Add completion controller to policy kwargs
        if 'policy_kwargs' not in ppo_kwargs:
            ppo_kwargs['policy_kwargs'] = {}
        ppo_kwargs['policy_kwargs']['completion_controller'] = self.completion_controller
        
        # Store for later PPO initialization
        self.policy_class = policy_class
        self.ppo_kwargs = ppo_kwargs
        self.ppo_model = None
    
    def create_model(self, env, **additional_kwargs):
        """Create the PPO model with hierarchical policy."""
        from stable_baselines3 import PPO
        
        # Merge additional kwargs
        final_kwargs = {**self.ppo_kwargs, **additional_kwargs}
        final_kwargs['policy'] = self.policy_class
        
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


# ============================================================================
# Phase 2 Task 2.1: Enhanced Two-Level Hierarchical Policy
# ============================================================================


class EnhancedHierarchicalActorCriticPolicy(ActorCriticPolicy):
    """
    Enhanced hierarchical Actor-Critic policy with true two-level architecture.
    
    This is the Phase 2 Task 2.1 implementation that provides:
    - Separate high-level policy for subtask selection
    - Separate low-level policy for action execution
    - Shared feature extractor (HGTMultimodalExtractor)
    - Coordinated training with different update frequencies
    - ICM integration at low-level
    
    Architecture improvements over Phase 1:
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
        features_extractor_class: BaseFeaturesExtractor = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        high_level_update_frequency: int = 50,
        max_steps_per_subtask: int = 500,
        use_icm: bool = True,
    ):
        """
        Initialize enhanced hierarchical actor-critic policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture
            activation_fn: Activation function
            features_extractor_class: Feature extractor class
            features_extractor_kwargs: Feature extractor kwargs
            normalize_images: Whether to normalize images
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer kwargs
            high_level_update_frequency: Steps between high-level updates
            max_steps_per_subtask: Maximum steps per subtask
            use_icm: Whether to use ICM
        """
        self.high_level_update_frequency = high_level_update_frequency
        self.max_steps_per_subtask = max_steps_per_subtask
        self.use_icm = use_icm
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        
        # Replace mlp_extractor with hierarchical policy network
        self.mlp_extractor = HierarchicalPolicyNetworkV2(
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            high_level_update_frequency=high_level_update_frequency,
            max_steps_per_subtask=max_steps_per_subtask,
            use_icm=use_icm,
        )
        
        # Override action and value networks (handled by hierarchical network)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        
        # Experience buffer for hierarchical training
        self.experience_buffer = HierarchicalExperienceBuffer(
            buffer_size=2048,
            high_level_update_frequency=high_level_update_frequency,
        )
    
    def _build_mlp_extractor(self) -> None:
        """Override to prevent building default MLP extractor."""
        pass
    
    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical policy.
        
        Args:
            obs: Observation tensor (should be dict with multiple components)
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Convert obs to dictionary format if needed
        if not isinstance(obs, dict):
            # Create minimal observation dict
            # In practice, this should be properly structured from the environment
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            obs_dict = {
                'observation': obs,
                'reachability_features': torch.zeros(batch_size, 8, device=obs.device),
                'switch_states': torch.zeros(batch_size, 5, device=obs.device),
                'ninja_position': torch.zeros(batch_size, 2, device=obs.device),
                'time_remaining': torch.ones(batch_size, 1, device=obs.device),
            }
        else:
            obs_dict = obs
        
        # Forward through hierarchical network
        actions, values, log_probs, info = self.mlp_extractor(
            obs_dict,
            deterministic=deterministic,
            update_subtask=True,
        )
        
        return actions, values, log_probs
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Convert obs to dict if needed
        if not isinstance(obs, dict):
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            obs_dict = {
                'observation': obs,
                'reachability_features': torch.zeros(batch_size, 8, device=obs.device),
                'switch_states': torch.zeros(batch_size, 5, device=obs.device),
                'ninja_position': torch.zeros(batch_size, 2, device=obs.device),
                'time_remaining': torch.ones(batch_size, 1, device=obs.device),
            }
        else:
            obs_dict = obs
        
        return self.mlp_extractor.evaluate_actions(obs_dict, actions)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for given observations."""
        if not isinstance(obs, dict):
            batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
            obs_dict = {
                'observation': obs,
                'reachability_features': torch.zeros(batch_size, 8, device=obs.device),
                'switch_states': torch.zeros(batch_size, 5, device=obs.device),
                'ninja_position': torch.zeros(batch_size, 2, device=obs.device),
                'time_remaining': torch.ones(batch_size, 1, device=obs.device),
            }
        else:
            obs_dict = obs
        
        return self.mlp_extractor.get_value(obs_dict)
    
    def reset_episode(self):
        """Reset policy state for new episode."""
        self.mlp_extractor.reset_episode()
    
    def get_subtask_metrics(self) -> Dict[str, Any]:
        """Get metrics about current subtask performance."""
        return self.mlp_extractor.get_subtask_metrics()


class EnhancedHierarchicalPPO:
    """
    Enhanced hierarchical PPO wrapper for Phase 2 Task 2.1.
    
    This provides a simplified interface for using the enhanced two-level
    hierarchical architecture with coordinated training.
    """
    
    def __init__(
        self,
        policy_class=EnhancedHierarchicalActorCriticPolicy,
        high_level_update_frequency: int = 50,
        max_steps_per_subtask: int = 500,
        use_icm: bool = True,
        **ppo_kwargs
    ):
        """
        Initialize enhanced hierarchical PPO.
        
        Args:
            policy_class: Policy class to use
            high_level_update_frequency: Steps between high-level updates
            max_steps_per_subtask: Maximum steps per subtask
            use_icm: Whether to use ICM
            **ppo_kwargs: Additional arguments for PPO
        """
        self.policy_class = policy_class
        self.high_level_update_frequency = high_level_update_frequency
        self.max_steps_per_subtask = max_steps_per_subtask
        self.use_icm = use_icm
        
        # Add hierarchical-specific parameters to policy kwargs
        if 'policy_kwargs' not in ppo_kwargs:
            ppo_kwargs['policy_kwargs'] = {}
        
        ppo_kwargs['policy_kwargs']['high_level_update_frequency'] = high_level_update_frequency
        ppo_kwargs['policy_kwargs']['max_steps_per_subtask'] = max_steps_per_subtask
        ppo_kwargs['policy_kwargs']['use_icm'] = use_icm
        
        self.ppo_kwargs = ppo_kwargs
        self.ppo_model = None
    
    def create_model(self, env, **additional_kwargs):
        """Create the PPO model with enhanced hierarchical policy."""
        from stable_baselines3 import PPO
        
        # Merge additional kwargs
        final_kwargs = {**self.ppo_kwargs, **additional_kwargs}
        final_kwargs['policy'] = self.policy_class
        
        self.ppo_model = PPO(env=env, **final_kwargs)
        return self.ppo_model
    
    def learn(self, *args, **kwargs):
        """Train the enhanced hierarchical PPO model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.learn(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """Make predictions with the enhanced hierarchical model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.predict(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        """Save the enhanced hierarchical model."""
        if self.ppo_model is None:
            raise ValueError("Model not created. Call create_model() first.")
        return self.ppo_model.save(*args, **kwargs)
    
    def load(self, *args, **kwargs):
        """Load an enhanced hierarchical model."""
        from stable_baselines3 import PPO
        self.ppo_model = PPO.load(*args, **kwargs)
        return self.ppo_model