"""
Low-Level Subtask Policies for Hierarchical RL

This module implements the low-level policy that executes movement actions
conditioned on the current subtask. The low-level policy handles tactical
decision-making with ICM-enhanced exploration.

Architecture:
- Input: Full multimodal observations (512D) + subtask embedding (64D) + context
- Output: 6 discrete movement actions (NOOP, Left, Right, Jump, Jump+Left, Jump+Right)
- Updates: Every step
- ICM Integration: Curiosity rewards modulated by current subtask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

from npp_rl.hrl.high_level_policy import Subtask


class SubtaskEmbedding(nn.Module):
    """
    Learned embeddings for subtasks that capture subtask-specific context.
    
    This module provides rich representations for each subtask that can be
    concatenated with observation features to condition the low-level policy.
    """
    
    def __init__(
        self,
        num_subtasks: int = 4,
        embedding_dim: int = 64,
    ):
        """
        Initialize subtask embeddings.
        
        Args:
            num_subtasks: Number of different subtasks
            embedding_dim: Dimension of subtask embeddings
        """
        super().__init__()
        
        self.num_subtasks = num_subtasks
        self.embedding_dim = embedding_dim
        
        # Learnable embeddings for each subtask
        self.embeddings = nn.Embedding(num_subtasks, embedding_dim)
        
        # Initialize with Xavier initialization
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, subtask_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for given subtask indices.
        
        Args:
            subtask_indices: [batch_size] or [batch_size, 1] subtask indices
            
        Returns:
            Subtask embeddings [batch_size, embedding_dim]
        """
        if len(subtask_indices.shape) > 1:
            subtask_indices = subtask_indices.squeeze(-1)
        
        return self.embeddings(subtask_indices)


class SubtaskContextEncoder(nn.Module):
    """
    Encodes subtask-specific context information.
    
    Context includes:
    - Target position for current subtask
    - Distance to target
    - Mine proximity warnings
    - Time since subtask started
    """
    
    def __init__(
        self,
        context_dim: int = 32,
    ):
        """
        Initialize context encoder.
        
        Args:
            context_dim: Output dimension for encoded context
        """
        super().__init__()
        
        # Input: target_pos (2) + distance (1) + mine_proximity (1) + time_in_subtask (1)
        self.input_dim = 5
        self.context_dim = context_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        target_position: torch.Tensor,
        distance_to_target: torch.Tensor,
        mine_proximity: torch.Tensor,
        time_in_subtask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode subtask context.
        
        Args:
            target_position: [batch_size, 2] normalized target position
            distance_to_target: [batch_size, 1] normalized distance
            mine_proximity: [batch_size, 1] nearest mine distance
            time_in_subtask: [batch_size, 1] normalized time in subtask
            
        Returns:
            Encoded context [batch_size, context_dim]
        """
        context = torch.cat([
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        ], dim=-1)
        
        return self.encoder(context)


class LowLevelPolicy(nn.Module):
    """
    Low-level policy for movement action execution.
    
    This policy takes full multimodal observations, current subtask embedding,
    and subtask context to produce movement actions. It's designed to be
    trained with ICM for enhanced exploration.
    
    Input Features:
    - Multimodal observations: 512D from HGTMultimodalExtractor
    - Subtask embedding: 64D learned embedding
    - Subtask context: 32D encoded context
    Total input: 608D
    
    Output:
    - Logits over 6 movement actions
    """
    
    def __init__(
        self,
        observation_dim: int = 512,
        subtask_embedding_dim: int = 64,
        context_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_actions: int = 6,
    ):
        """
        Initialize low-level policy network.
        
        Args:
            observation_dim: Dimension of multimodal observations
            subtask_embedding_dim: Dimension of subtask embeddings
            context_dim: Dimension of encoded context
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
            num_actions: Number of movement actions
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.subtask_embedding_dim = subtask_embedding_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Total input dimension
        self.input_dim = observation_dim + subtask_embedding_dim + context_dim
        
        # Subtask embedding module
        self.subtask_embedding = SubtaskEmbedding(
            num_subtasks=4,
            embedding_dim=subtask_embedding_dim,
        )
        
        # Context encoder
        self.context_encoder = SubtaskContextEncoder(context_dim=context_dim)
        
        # Main policy network
        layers = []
        prev_dim = self.input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.policy_net = nn.Sequential(*layers)
        
        # Action head
        self.action_head = nn.Linear(hidden_dim, num_actions)
        
        # Optional: Add residual connection from observations
        self.use_residual = observation_dim == hidden_dim
        if self.use_residual:
            self.residual_proj = nn.Linear(observation_dim, hidden_dim)
    
    def forward(
        self,
        observations: torch.Tensor,
        subtask_indices: torch.Tensor,
        target_position: torch.Tensor,
        distance_to_target: torch.Tensor,
        mine_proximity: torch.Tensor,
        time_in_subtask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute action logits.
        
        Args:
            observations: [batch_size, 512] multimodal observations
            subtask_indices: [batch_size] current subtask indices
            target_position: [batch_size, 2] target position
            distance_to_target: [batch_size, 1] distance to target
            mine_proximity: [batch_size, 1] mine proximity
            time_in_subtask: [batch_size, 1] time in subtask
            
        Returns:
            Action logits [batch_size, num_actions]
        """
        # Get subtask embedding
        subtask_embed = self.subtask_embedding(subtask_indices)
        
        # Encode context
        context = self.context_encoder(
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        )
        
        # Combine all features
        combined = torch.cat([
            observations,
            subtask_embed,
            context,
        ], dim=-1)
        
        # Forward through policy network
        hidden = self.policy_net(combined)
        
        # Optional residual connection
        if self.use_residual:
            hidden = hidden + self.residual_proj(observations)
        
        # Compute action logits
        action_logits = self.action_head(hidden)
        
        return action_logits
    
    def select_action(
        self,
        observations: torch.Tensor,
        subtask_indices: torch.Tensor,
        target_position: torch.Tensor,
        distance_to_target: torch.Tensor,
        mine_proximity: torch.Tensor,
        time_in_subtask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action given current state and subtask.
        
        Args:
            observations: Multimodal observations
            subtask_indices: Current subtask
            target_position: Target position
            distance_to_target: Distance to target
            mine_proximity: Mine proximity
            time_in_subtask: Time in subtask
            deterministic: If True, select argmax; otherwise sample
            
        Returns:
            Tuple of (selected_action, log_probability)
        """
        logits = self.forward(
            observations,
            subtask_indices,
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        )
        
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        log_prob = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, selected_log_prob
    
    def get_action_distribution(
        self,
        observations: torch.Tensor,
        subtask_indices: torch.Tensor,
        target_position: torch.Tensor,
        distance_to_target: torch.Tensor,
        mine_proximity: torch.Tensor,
        time_in_subtask: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """
        Get action distribution for computing various statistics.
        
        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(
            observations,
            subtask_indices,
            target_position,
            distance_to_target,
            mine_proximity,
            time_in_subtask,
        )
        
        return torch.distributions.Categorical(logits=logits)


class ICMIntegration(nn.Module):
    """
    Integration layer for ICM with subtask-aware curiosity modulation.
    
    This module adjusts ICM curiosity rewards based on the current subtask
    and reachability information to focus exploration appropriately.
    """
    
    def __init__(
        self,
        base_curiosity_weight: float = 0.01,
        subtask_modulation_weights: Optional[Dict[Subtask, float]] = None,
    ):
        """
        Initialize ICM integration.
        
        Args:
            base_curiosity_weight: Base weight for curiosity rewards
            subtask_modulation_weights: Per-subtask modulation factors
        """
        super().__init__()
        
        self.base_curiosity_weight = base_curiosity_weight
        
        # Default modulation weights for each subtask
        if subtask_modulation_weights is None:
            subtask_modulation_weights = {
                Subtask.NAVIGATE_TO_EXIT_SWITCH: 0.5,  # Lower exploration, goal-directed
                Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH: 0.7,  # Moderate exploration
                Subtask.NAVIGATE_TO_EXIT_DOOR: 0.3,  # Very goal-directed
                Subtask.EXPLORE_FOR_SWITCHES: 1.5,  # High exploration
            }
        
        self.subtask_modulation = subtask_modulation_weights
    
    def modulate_curiosity(
        self,
        base_curiosity: torch.Tensor,
        subtask_indices: torch.Tensor,
        mine_proximity: torch.Tensor,
        reachability_score: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modulate ICM curiosity rewards based on subtask and context.
        
        Args:
            base_curiosity: [batch_size] base ICM curiosity rewards
            subtask_indices: [batch_size] current subtask indices
            mine_proximity: [batch_size] distance to nearest dangerous mine
            reachability_score: [batch_size] reachability to target
            
        Returns:
            Modulated curiosity rewards [batch_size]
        """
        modulated = base_curiosity.clone()
        
        # Apply subtask-specific modulation (vectorized)
        modulation_factors = torch.ones_like(base_curiosity)
        for subtask, factor in self.subtask_modulation.items():
            mask = subtask_indices == subtask.value
            modulation_factors[mask] = factor
        modulated = modulated * modulation_factors
        
        # Reduce curiosity near dangerous mines (safety)
        mine_danger_threshold = 2.0
        mine_factor = torch.clamp(mine_proximity / mine_danger_threshold, 0.1, 1.0)
        modulated = modulated * mine_factor
        
        # Boost curiosity when reachability is low (stuck)
        reachability_threshold = 0.3
        low_reachability_mask = reachability_score < reachability_threshold
        modulated[low_reachability_mask] *= 1.5
        
        return modulated * self.base_curiosity_weight
    
    def forward(
        self,
        base_curiosity: torch.Tensor,
        subtask_indices: torch.Tensor,
        mine_proximity: torch.Tensor,
        reachability_score: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (alias for modulate_curiosity)."""
        return self.modulate_curiosity(
            base_curiosity,
            subtask_indices,
            mine_proximity,
            reachability_score,
        )


class SubtaskSpecificFeatures:
    """
    Helper class to extract subtask-specific features from observations.
    
    This provides utility methods for computing target positions, distances,
    and other context features needed by the low-level policy.
    
    Note: The methods in this class provide placeholder implementations.
    Users should extend this class and override methods to extract actual
    features from their environment's observation space.
    """
    
    @staticmethod
    def extract_target_position(
        obs: Dict[str, Any],
        subtask: Subtask,
    ) -> np.ndarray:
        """
        Extract target position for current subtask.
        
        Args:
            obs: Environment observation
            subtask: Current subtask
            
        Returns:
            Target position [2] (normalized)
            
        Note:
            This is a placeholder implementation returning dummy values.
            Override this method to extract actual positions from your
            environment's observation structure.
        """
        # Placeholder implementation - override in subclass
        if subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            return np.array([0.5, 0.5], dtype=np.float32)
        elif subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            return np.array([0.3, 0.7], dtype=np.float32)
        elif subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            return np.array([0.8, 0.2], dtype=np.float32)
        else:  # EXPLORE_FOR_SWITCHES
            return np.array([0.5, 0.5], dtype=np.float32)
    
    @staticmethod
    def compute_distance_to_target(
        ninja_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> float:
        """Compute normalized distance to target."""
        return np.linalg.norm(ninja_pos - target_pos)
    
    @staticmethod
    def compute_mine_proximity(obs: Dict[str, Any]) -> float:
        """
        Compute proximity to nearest dangerous mine.
        
        Args:
            obs: Environment observation
            
        Returns:
            Distance to nearest dangerous mine (normalized)
            
        Note:
            This is a placeholder implementation returning a safe distance.
            Override this method to compute actual mine proximity from your
            environment's observation structure.
        """
        # Placeholder implementation - override in subclass
        return 5.0  # Safe distance
