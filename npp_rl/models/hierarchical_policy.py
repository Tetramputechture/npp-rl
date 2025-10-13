"""
Hierarchical Policy Network

This module implements the main hierarchical policy that combines high-level
subtask selection and low-level action execution with a shared feature extractor.

Architecture:
- Shared HGTMultimodalExtractor for both policy levels
- High-level policy for strategic subtask selection (updates every 50-100 steps)
- Low-level policy for tactical action execution (updates every step)
- Separate experience buffers for coordinated training
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from npp_rl.hrl.high_level_policy import (
    HighLevelPolicy, 
    Subtask, 
    SubtaskTransitionManager
)
from npp_rl.hrl.subtask_policies import (
    LowLevelPolicy,
    ICMIntegration,
)

# Entity position indices in the entity_positions array
# Array format: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
NINJA_POS_START = 0
NINJA_POS_END = 2
SWITCH_POS_START = 2
SWITCH_POS_END = 4
EXIT_POS_START = 4
EXIT_POS_END = 6


class HierarchicalPolicyNetwork(nn.Module):
    """
    Main hierarchical policy network combining high and low level policies.
    
    This network:
    1. Uses shared feature extractor for both policy levels
    2. High-level policy selects subtasks based on reachability features
    3. Low-level policy executes actions conditioned on current subtask
    4. Manages subtask transitions and coordination
    
    Training:
    - High-level updates every N steps (50-100)
    - Low-level updates every step
    - Coordinated learning rates and separate buffers
    """
    
    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int = 512,
        high_level_update_frequency: int = 50,
        max_steps_per_subtask: int = 500,
        min_steps_between_switches: int = 50,
        use_icm: bool = True,
    ):
        """
        Initialize hierarchical policy network.
        
        Args:
            features_extractor: Shared multimodal feature extractor
            features_dim: Dimension of extracted features
            high_level_update_frequency: Steps between high-level updates
            max_steps_per_subtask: Maximum steps per subtask
            min_steps_between_switches: Minimum cooldown between switches
            use_icm: Whether to use ICM for exploration
        """
        super().__init__()
        
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.high_level_update_frequency = high_level_update_frequency
        self.use_icm = use_icm
        
        # High-level policy for subtask selection
        self.high_level_policy = HighLevelPolicy(
            reachability_dim=8,
            max_switches=5,
            hidden_dim=128,
            num_layers=2,
        )
        
        # Low-level policy for action execution
        self.low_level_policy = LowLevelPolicy(
            observation_dim=features_dim,
            subtask_embedding_dim=64,
            context_dim=32,
            hidden_dim=512,
            num_layers=3,
            num_actions=6,
        )
        
        # Value function (shared between both levels)
        self.value_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),  # features + subtask embedding
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        # ICM integration (if enabled)
        if self.use_icm:
            self.icm_integration = ICMIntegration(
                base_curiosity_weight=0.01,
            )
        
        # Subtask transition manager
        self.transition_manager = SubtaskTransitionManager(
            max_steps_per_subtask=max_steps_per_subtask,
            min_steps_between_switches=min_steps_between_switches,
        )
        
        # Step tracking
        self.step_count = 0
        # Register current_subtask as a buffer so it moves with the model
        self.register_buffer('current_subtask', torch.tensor([0], dtype=torch.long))
        
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
        update_subtask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through hierarchical policy.
        
        Args:
            obs: Dictionary of observations including:
                - Full observation for feature extraction
                - Reachability features
                - Switch states
                - Ninja position
                - Time remaining
            deterministic: Whether to use deterministic actions
            update_subtask: Whether to allow subtask updates this step
            
        Returns:
            Tuple of:
                - actions: [batch_size] selected actions
                - values: [batch_size] value estimates
                - log_probs: [batch_size] log probabilities
                - info: Dictionary with additional information
        """
        batch_size = obs['reachability_features'].shape[0]
        
        # Extract shared features
        shared_features = self.features_extractor(obs['observation'])
        
        # Update high-level policy (subtask selection) if appropriate
        should_update = (
            update_subtask and 
            self.transition_manager.should_update_subtask()
        )
        
        if should_update:
            new_subtask, subtask_log_prob = self.high_level_policy.select_subtask(
                obs['reachability_features'],
                obs['switch_states'],
                obs['ninja_position'],
                obs['time_remaining'],
                deterministic=deterministic,
            )
            
            # Update current subtask if changed
            if new_subtask.item() != self.current_subtask.item():
                self.transition_manager.update_subtask(Subtask(new_subtask.item()))
                self.current_subtask = new_subtask
        else:
            # Keep current subtask
            subtask_log_prob = torch.zeros(batch_size, device=shared_features.device)
        
        # Ensure current_subtask has correct batch size
        if self.current_subtask.shape[0] != batch_size:
            self.current_subtask = self.current_subtask.expand(batch_size)
        
        # Get subtask-specific context
        context = self._extract_subtask_context(obs)
        
        # Low-level policy (action selection)
        actions, action_log_probs = self.low_level_policy.select_action(
            shared_features,
            self.current_subtask,
            context['target_position'],
            context['distance_to_target'],
            context['mine_proximity'],
            context['time_in_subtask'],
            deterministic=deterministic,
        )
        
        # Compute values
        subtask_embed = self.low_level_policy.subtask_embedding(self.current_subtask)
        value_input = torch.cat([shared_features, subtask_embed], dim=-1)
        values = self.value_net(value_input).squeeze(-1)
        
        # Prepare info dictionary
        info = {
            'current_subtask': self.current_subtask,
            'subtask_log_prob': subtask_log_prob,
            'high_level_updated': should_update,
            'subtask_step_count': self.transition_manager.subtask_step_count,
        }
        
        # Update step tracking
        self.step_count += 1
        self.transition_manager.step()
        
        return actions, values, action_log_probs, info
    
    def _extract_subtask_context(
        self, 
        obs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract subtask-specific context from observations.
        
        Args:
            obs: Observation dictionary containing:
                - reachability_features: [batch, 8] - distances to switches/exits, reachability info
                - game_state: [batch, 26] - ninja physics state including position
                - entity_states: [batch, ...] - entity information including mines
            
        Returns:
            Context dictionary with target info, mine proximity, etc.
        """
        batch_size = obs['reachability_features'].shape[0]
        device = obs['reachability_features'].device
        
        # Extract reachability features
        reachability = obs['reachability_features']  # [batch, 8]
        # Feature indices: 0=area_ratio, 1=dist_to_switch, 2=dist_to_exit, 3=reachable_switches,
        #                  4=reachable_hazards, 5=connectivity, 6=exit_reachable, 7=path_exists
        
        # Get actual entity positions from observation (added in observation processor)
        # entity_positions: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
        entity_positions = obs.get('entity_positions', torch.zeros(batch_size, 6, device=device))
        ninja_pos = entity_positions[:, NINJA_POS_START:NINJA_POS_END]
        switch_pos = entity_positions[:, SWITCH_POS_START:SWITCH_POS_END]
        exit_pos = entity_positions[:, EXIT_POS_START:EXIT_POS_END]
        
        # Determine target based on current subtask using ACTUAL positions
        current_subtask = self.transition_manager.current_subtask
        
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Target is the exit switch - use real position
            target_position = switch_pos
            distance_to_target = torch.norm(ninja_pos - switch_pos, dim=1, keepdim=True)
        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Target is locked door switch - for now use same as exit switch
            # (locked doors not fully implemented yet)
            target_position = switch_pos
            distance_to_target = torch.norm(ninja_pos - switch_pos, dim=1, keepdim=True)
        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Target is the exit door - use real position
            target_position = exit_pos
            distance_to_target = torch.norm(ninja_pos - exit_pos, dim=1, keepdim=True)
        else:  # EXPLORE_FOR_SWITCHES or other
            # Use centroid of switch and exit as exploration target
            target_position = (switch_pos + exit_pos) / 2.0
            distance_to_target = torch.norm(ninja_pos - target_position, dim=1, keepdim=True)
        
        # Extract mine proximity from reachability features
        # Feature 4 is reachable_hazards count (normalized, higher = more hazards nearby)
        mine_proximity = reachability[:, 4:5]  # [batch, 1] - already normalized [0, 1]
        
        # Time in subtask (normalized by expected duration)
        time_in_subtask = torch.full(
            (batch_size, 1),
            min(self.transition_manager.subtask_step_count / 500.0, 1.0),
            device=device
        )
        
        context = {
            'target_position': target_position,
            'distance_to_target': distance_to_target,
            'mine_proximity': mine_proximity,
            'time_in_subtask': time_in_subtask,
        }
        
        return context
    
    def get_value(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute value estimate for current state.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            Value estimates [batch_size]
        """
        shared_features = self.features_extractor(obs['observation'])
        
        batch_size = shared_features.shape[0]
        if self.current_subtask.shape[0] != batch_size:
            current_subtask = self.current_subtask.expand(batch_size)
        else:
            current_subtask = self.current_subtask
        
        subtask_embed = self.low_level_policy.subtask_embedding(current_subtask)
        value_input = torch.cat([shared_features, subtask_embed], dim=-1)
        return self.value_net(value_input).squeeze(-1)
    
    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        subtasks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (compute log probs and entropy).
        
        Args:
            obs: Observation dictionary
            actions: Actions to evaluate
            subtasks: Subtasks to use (if None, use current)
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        shared_features = self.features_extractor(obs['observation'])
        
        if subtasks is None:
            subtasks = self.current_subtask
        
        batch_size = shared_features.shape[0]
        if subtasks.shape[0] != batch_size:
            subtasks = subtasks.expand(batch_size)
        
        # Get context
        context = self._extract_subtask_context(obs)
        
        # Get action distribution
        action_dist = self.low_level_policy.get_action_distribution(
            shared_features,
            subtasks,
            context['target_position'],
            context['distance_to_target'],
            context['mine_proximity'],
            context['time_in_subtask'],
        )
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        # Compute values
        subtask_embed = self.low_level_policy.subtask_embedding(subtasks)
        value_input = torch.cat([shared_features, subtask_embed], dim=-1)
        values = self.value_net(value_input).squeeze(-1)
        
        return values, log_probs, entropy
    
    def reset_episode(self):
        """Reset policy state for new episode."""
        self.step_count = 0
        # Reset to NAVIGATE_TO_EXIT_SWITCH, maintaining device
        self.current_subtask = torch.tensor([0], dtype=torch.long, device=self.current_subtask.device)
        self.transition_manager.reset()
    
    def get_subtask_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about subtask performance.
        
        Note: This method assumes single-environment (non-batched) execution.
        For batched environments, metrics would need to be computed per environment.
        """
        return {
            'current_subtask': Subtask(self.current_subtask[0].item()).name,
            'subtask_step_count': self.transition_manager.subtask_step_count,
            'total_steps': self.step_count,
            'high_level_updates': self.step_count // self.high_level_update_frequency,
        }


class HierarchicalExperienceBuffer:
    """
    Experience buffer for hierarchical RL with separate high and low level storage.
    
    This buffer maintains:
    - Low-level experiences (every step)
    - High-level experiences (every N steps)
    - Subtask context for proper credit assignment
    
    Note:
        This implementation uses Python lists for simplicity. For production use
        with large buffers, consider using numpy arrays or circular buffers for
        better memory efficiency and performance.
    """
    
    def __init__(
        self,
        buffer_size: int = 2048,
        high_level_update_frequency: int = 50,
    ):
        """
        Initialize hierarchical experience buffer.
        
        Args:
            buffer_size: Maximum buffer size for low-level experiences
            high_level_update_frequency: Frequency of high-level updates
        """
        self.buffer_size = buffer_size
        self.high_level_update_frequency = high_level_update_frequency
        
        # Low-level experience buffer (stores every step)
        self.low_level_buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'subtasks': [],
            'dones': [],
        }
        
        # High-level experience buffer (stores subtask transitions)
        self.high_level_buffer = {
            'observations': [],
            'subtasks': [],
            'subtask_rewards': [],  # Cumulative reward during subtask
            'values': [],
            'log_probs': [],
            'dones': [],
        }
        
        self.step_count = 0
        self.current_subtask_reward = 0.0
        
    def add_low_level(
        self,
        obs: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        subtask: int,
        done: bool,
    ):
        """Add low-level experience to buffer."""
        self.low_level_buffer['observations'].append(obs)
        self.low_level_buffer['actions'].append(action)
        self.low_level_buffer['rewards'].append(reward)
        self.low_level_buffer['values'].append(value)
        self.low_level_buffer['log_probs'].append(log_prob)
        self.low_level_buffer['subtasks'].append(subtask)
        self.low_level_buffer['dones'].append(done)
        
        self.current_subtask_reward += reward
        self.step_count += 1
        
        # Trim buffer if too large
        if len(self.low_level_buffer['observations']) > self.buffer_size:
            for key in self.low_level_buffer:
                self.low_level_buffer[key].pop(0)
    
    def add_high_level(
        self,
        obs: Dict[str, torch.Tensor],
        subtask: int,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add high-level experience (subtask transition) to buffer."""
        self.high_level_buffer['observations'].append(obs)
        self.high_level_buffer['subtasks'].append(subtask)
        self.high_level_buffer['subtask_rewards'].append(self.current_subtask_reward)
        self.high_level_buffer['values'].append(value)
        self.high_level_buffer['log_probs'].append(log_prob)
        self.high_level_buffer['dones'].append(done)
        
        # Reset subtask reward accumulation
        self.current_subtask_reward = 0.0
    
    def get_low_level_batch(self) -> Dict[str, List]:
        """Get batch of low-level experiences."""
        return self.low_level_buffer.copy()
    
    def get_high_level_batch(self) -> Dict[str, List]:
        """Get batch of high-level experiences."""
        return self.high_level_buffer.copy()
    
    def clear(self):
        """Clear all buffers."""
        for key in self.low_level_buffer:
            self.low_level_buffer[key].clear()
        for key in self.high_level_buffer:
            self.high_level_buffer[key].clear()
        
        self.step_count = 0
        self.current_subtask_reward = 0.0
