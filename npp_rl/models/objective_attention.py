"""Attention-based policy head for variable objectives.

This module implements multi-head attention over objectives (exit switch, exit door,
locked doors, and their switches) to handle variable numbers of objectives and
learn which objectives to prioritize dynamically.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ObjectiveAttentionPolicy(nn.Module):
    """Attention-based policy that conditions on variable objectives.

    This policy uses multi-head attention to dynamically focus on relevant
    objectives (exit switch, exit door, locked doors, locked door switches)
    based on the current state. This allows the policy to handle variable
    numbers of locked doors (1-16) and learn sequential goal prioritization.

    Architecture:
        1. Encode each objective type separately
        2. Apply multi-head attention: query=current_state, keys/values=objectives
        3. Combine attended objectives with policy features
        4. Output action logits

    Key benefits:
        - Handles variable number of locked doors (1-16)
        - Learns to focus on relevant objectives
        - Permutation invariant over locked doors
        - Explicit sequential reasoning (which door to prioritize)
    """

    def __init__(
        self,
        policy_feature_dim: int = 256,
        objective_embed_dim: int = 64,
        attention_dim: int = 512,
        num_heads: int = 8,
        max_locked_doors: int = 16,
        num_actions: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize objective attention policy.

        Args:
            policy_feature_dim: Dimension of policy features from MLP
            objective_embed_dim: Dimension of objective embeddings
            attention_dim: Dimension for attention mechanism
            num_heads: Number of attention heads
            max_locked_doors: Maximum number of locked doors to handle
            num_actions: Number of actions
            dropout: Dropout rate
        """
        super().__init__()

        self.policy_feature_dim = policy_feature_dim
        self.objective_embed_dim = objective_embed_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.max_locked_doors = max_locked_doors
        self.num_actions = num_actions

        # Objective encoders
        # Exit objectives: [switch_x, switch_y, activated, switch_path_dist, door_x, door_y, door_path_dist]
        self.exit_encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, objective_embed_dim),
            nn.LayerNorm(objective_embed_dim),
        )

        # Locked door encoder: [pos_x, pos_y, open, distance]
        self.locked_door_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, objective_embed_dim),
            nn.LayerNorm(objective_embed_dim),
        )

        # Locked door switch encoder: [pos_x, pos_y, collected, distance]
        self.locked_switch_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, objective_embed_dim),
            nn.LayerNorm(objective_embed_dim),
        )

        # Project policy features to attention dimension
        self.query_proj = nn.Linear(policy_feature_dim, attention_dim)

        # Project objective embeddings to attention dimension
        self.key_value_proj = nn.Linear(objective_embed_dim, attention_dim)

        # Multi-head attention over objectives
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion: combine policy features with attended objectives
        self.fusion = nn.Sequential(
            nn.Linear(policy_feature_dim + attention_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Action head
        self.action_head = nn.Linear(256, num_actions)

        # Initialize attention weights for stability
        self._initialize_attention_weights()

    def _initialize_attention_weights(self):
        """Initialize attention layers with small weights for training stability.

        Uses Xavier initialization with reduced gains to prevent early gradient explosion.
        Orthogonal initialization for action head improves conditioning.
        """
        # Initialize attention projection layers with Xavier uniform
        for module in [
            self.exit_encoder,
            self.locked_door_encoder,
            self.locked_switch_encoder,
            self.fusion,
        ]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize query/key/value projections with smaller scale
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.key_value_proj.weight, gain=0.5)
        nn.init.zeros_(self.key_value_proj.bias)

        # Initialize action head with very small weights
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(
        self,
        policy_features: torch.Tensor,
        objective_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with attention over objectives.

        Args:
            policy_features: Policy latent features [batch, policy_feature_dim]
            objective_features: Optional dictionary containing:
                - exit_switch_pos: [batch, 2] (x, y)
                - exit_switch_activated: [batch, 1]
                - exit_door_pos: [batch, 2] (x, y)
                - exit_door_accessible: [batch, 1]
                - locked_door_positions: [batch, max_doors, 2]
                - locked_door_states: [batch, max_doors, 2] (open, distance)
                - locked_switch_positions: [batch, max_doors, 2]
                - locked_switch_states: [batch, max_doors, 2] (collected, distance)
                - num_locked_doors: [batch] (actual count)
                If None, uses simple MLP without attention (fallback mode).

        Returns:
            Tuple of (action_logits [batch, num_actions], attention_weights or None)
        """
        batch_size = policy_features.shape[0]

        # Objective features are required for this policy to function properly
        if objective_features is None:
            raise ValueError(
                "ObjectiveAttentionPolicy requires objective_features to be provided. "
                "This error indicates that ObjectiveFeatureExtractor was not called before "
                "passing features to the action_net. Check that the policy's forward(), "
                "evaluate_actions(), and _predict() methods properly extract and pass "
                "objective features to action_net(latent_pi, objective_features)."
            )

        device = policy_features.device

        # 1. Encode exit objectives
        exit_features = torch.cat(
            [
                objective_features["exit_switch_pos"],  # 2 dims
                objective_features["exit_switch_activated"],  # 1 dim
                objective_features.get(
                    "exit_switch_path_dist",
                    torch.zeros_like(objective_features["exit_switch_activated"]),
                ),  # 1 dim
                objective_features["exit_door_pos"],  # 2 dims
                objective_features.get(
                    "exit_door_path_dist",
                    torch.zeros_like(objective_features["exit_switch_activated"]),
                ),  # 1 dim
            ],
            dim=-1,
        )  # [batch, 7]

        exit_embed = self.exit_encoder(exit_features)  # [batch, objective_embed_dim]
        exit_embed = exit_embed.unsqueeze(1)  # [batch, 1, objective_embed_dim]

        # 2. Encode locked doors (with masking for variable numbers)
        locked_door_features = torch.cat(
            [
                objective_features["locked_door_positions"],
                objective_features["locked_door_states"],
            ],
            dim=-1,
        )  # [batch, max_doors, 4]

        # Reshape for batch processing
        locked_door_flat = locked_door_features.reshape(
            batch_size * self.max_locked_doors, 4
        )
        locked_door_embed = self.locked_door_encoder(locked_door_flat)
        locked_door_embed = locked_door_embed.reshape(
            batch_size, self.max_locked_doors, self.objective_embed_dim
        )  # [batch, max_doors, objective_embed_dim]

        # 3. Encode locked door switches
        locked_switch_features = torch.cat(
            [
                objective_features["locked_switch_positions"],
                objective_features["locked_switch_states"],
            ],
            dim=-1,
        )  # [batch, max_doors, 4]

        locked_switch_flat = locked_switch_features.reshape(
            batch_size * self.max_locked_doors, 4
        )
        locked_switch_embed = self.locked_switch_encoder(locked_switch_flat)
        locked_switch_embed = locked_switch_embed.reshape(
            batch_size, self.max_locked_doors, self.objective_embed_dim
        )  # [batch, max_doors, objective_embed_dim]

        # 4. Combine all objectives
        # Shape: [batch, 1 + max_doors + max_doors, objective_embed_dim]
        all_objectives = torch.cat(
            [
                exit_embed,
                locked_door_embed,
                locked_switch_embed,
            ],
            dim=1,
        )

        # 5. Create attention mask for variable number of doors (VECTORIZED)
        # Mask shape: [batch, 1 + 2*max_doors]
        num_locked_doors = objective_features["num_locked_doors"]

        # Exit is always valid (first position)
        mask = torch.zeros(
            batch_size, 1 + 2 * self.max_locked_doors, dtype=torch.bool, device=device
        )

        # Vectorized mask creation (no Python loop, no .item() CPU-GPU syncs)
        # Create position indices for broadcasting
        door_range = torch.arange(self.max_locked_doors, device=device).unsqueeze(
            0
        )  # [1, max_doors]
        num_doors_expanded = num_locked_doors.unsqueeze(1)  # [batch, 1]

        # Vectorized comparison: mask positions beyond actual door count
        door_mask = door_range >= num_doors_expanded  # [batch, max_doors]
        switch_mask = (
            door_range >= num_doors_expanded
        )  # [batch, max_doors] (same logic)

        # Assign masks to appropriate positions
        # Doors: positions 1 to 1+max_doors
        mask[:, 1 : 1 + self.max_locked_doors] = door_mask
        # Switches: positions 1+max_doors to 1+2*max_doors
        mask[:, 1 + self.max_locked_doors :] = switch_mask

        # 6. Project objectives to attention dimension
        all_objectives_proj = self.key_value_proj(all_objectives)

        # 7. Project policy features to query
        query = self.query_proj(policy_features).unsqueeze(
            1
        )  # [batch, 1, attention_dim]

        # 8. Apply multi-head attention
        # Query: current state features
        # Keys/Values: all objective embeddings
        attended, attention_weights = self.attention(
            query=query,
            key=all_objectives_proj,
            value=all_objectives_proj,
            key_padding_mask=mask,
            need_weights=True,
        )
        attended = attended.squeeze(1)  # [batch, attention_dim]

        # 9. Combine policy features with attended objectives
        combined = torch.cat([policy_features, attended], dim=-1)
        fused = self.fusion(combined)  # [batch, 256]

        # 10. Output action logits
        action_logits = self.action_head(fused)  # [batch, num_actions]

        # Store mask for entropy computation
        self._last_attention_mask = mask

        return action_logits, attention_weights

    def get_attention_weights(
        self,
        policy_features: torch.Tensor,
        objective_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention weights for visualization.

        Args:
            policy_features: Policy latent features
            objective_features: Objective features dictionary

        Returns:
            Tuple of (action_logits, attention_weights)
            attention_weights shape: [batch, num_heads, 1, num_objectives]
        """
        return self.forward(policy_features, objective_features)

    def compute_attention_entropy(
        self, attention_weights: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy of attention distribution over objectives.

        Higher entropy indicates more diverse attention across objectives,
        which is desirable for robust policy learning. This regularization
        prevents the policy from degenerately focusing on a single objective.

        Args:
            attention_weights: [batch, num_heads, 1, num_objectives]
                Attention probabilities from multi-head attention
            mask: [batch, num_objectives]
                Boolean mask where True indicates invalid/padded positions

        Returns:
            Scalar tensor with mean entropy across batch and heads

        Example:
            - Entropy ≈ 0: Attention focused on single objective (degenerate)
            - Entropy ≈ log(num_valid_objectives): Uniform attention (healthy)
        """
        # Squeeze query dimension: [batch, num_heads, num_objectives]
        attn_probs = attention_weights.squeeze(2)

        # Apply mask: set masked (invalid) positions to 0 probability
        mask_expanded = mask.unsqueeze(1)  # [batch, 1, num_objectives]
        attn_probs = attn_probs.masked_fill(mask_expanded, 0.0)

        # Renormalize after masking to ensure probabilities sum to 1
        attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to prevent log(0)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(
            dim=-1
        )  # [batch, num_heads]

        # Average over heads and batch for scalar loss term
        return entropy.mean()


class ObjectiveFeatureExtractor(nn.Module):
    """Extract structured objective features from raw observations.

    This module processes raw observations to extract structured features
    for exit switch, exit door, locked doors, and locked door switches.
    """

    def __init__(self, max_locked_doors: int = 16):
        """Initialize objective feature extractor.

        Args:
            max_locked_doors: Maximum number of locked doors to handle
        """
        super().__init__()
        self.max_locked_doors = max_locked_doors

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Extract objective features from observations.

        Args:
            obs: Observation dictionary from environment

        Returns:
            Dictionary of objective features suitable for ObjectiveAttentionPolicy
        """
        # Extract exit features from dedicated observation key
        # exit_features: [batch, 7] = [switch_x, switch_y, switch_activated, switch_path_dist, door_x, door_y, door_path_dist]
        exit_features = obs["exit_features"]  # [batch, 7]

        # Split exit features
        exit_switch_pos = exit_features[:, 0:2]  # [batch, 2] - relative x, y
        exit_switch_activated = exit_features[:, 2:3]  # [batch, 1] - 0.0 or 1.0
        exit_switch_path_dist = exit_features[:, 3:4]  # [batch, 1]
        exit_door_pos = exit_features[:, 4:6]  # [batch, 2] - relative x, y
        exit_door_path_dist = exit_features[:, 6:7]  # [batch, 1]

        # Locked doors: from locked_door_features observation
        locked_door_array = obs["locked_door_features"]  # [batch, 16, 8]
        num_locked_doors = obs["num_locked_doors"]  # [batch, 1]

        # Split locked_door_array into components
        locked_switch_pos = locked_door_array[:, :, 0:2]
        locked_switch_collected = locked_door_array[:, :, 2:3]
        locked_switch_path_dist = locked_door_array[:, :, 3:4]
        locked_door_pos = locked_door_array[:, :, 4:6]
        locked_door_open = locked_door_array[:, :, 6:7]  # Same as switch collected
        locked_door_path_dist = locked_door_array[:, :, 7:8]

        # Combine states for encoders
        locked_switch_states = torch.cat(
            [locked_switch_collected, locked_switch_path_dist], dim=-1
        )
        locked_door_states = torch.cat(
            [locked_door_open, locked_door_path_dist], dim=-1
        )

        return {
            "exit_switch_pos": exit_switch_pos,
            "exit_switch_activated": exit_switch_activated,
            "exit_switch_path_dist": exit_switch_path_dist,
            "exit_door_pos": exit_door_pos,
            "exit_door_path_dist": exit_door_path_dist,
            "exit_door_accessible": exit_switch_activated,
            "locked_door_positions": locked_door_pos,
            "locked_door_states": locked_door_states,
            "locked_switch_positions": locked_switch_pos,
            "locked_switch_states": locked_switch_states,
            "num_locked_doors": num_locked_doors.squeeze(-1).long(),
        }
