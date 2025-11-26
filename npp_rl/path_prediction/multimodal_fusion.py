"""Multimodal Fusion Module for Path Prediction.

This module implements a multimodal fusion architecture that combines:
1. Graph structure (global topology understanding)
2. Start context (where the path begins)
3. Goal context (where the path needs to go)
4. Physics state (how the ninja is currently moving)

The architecture mirrors the proven RL agent's ConfigurableMultimodalExtractor
design, using multi-head attention for cross-modal reasoning.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from nclone.gym_environment.constants import GAME_STATE_CHANNELS

logger = logging.getLogger(__name__)


class MultiHeadFusion(nn.Module):
    """Enhanced multi-head attention fusion with true cross-modal reasoning.

    This is a local copy to avoid circular import issues with the training module.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        modality_dims: Optional[list] = None,
        dropout: float = 0.1,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.debug_mode = debug_mode
        # Modality dimensions (default assumes 5 equal-sized modalities)
        if modality_dims is None:
            # Fallback: assume equal split (for backward compatibility)
            assert input_dim % 5 == 0, (
                f"input_dim {input_dim} must be divisible by 5 modalities"
            )
            modality_dims = [input_dim // 5] * 5

        self.modality_dims = modality_dims
        self.num_modalities = len(modality_dims)

        # Uniform dimension for attention (must be divisible by num_heads)
        # Use power of 2 for efficiency
        self.uniform_dim = 256  # 256 / 8 heads = 32 per head
        assert self.uniform_dim % num_heads == 0

        # Project each modality to uniform dimension
        self.modality_projections = nn.ModuleList(
            [nn.Linear(dim, self.uniform_dim) for dim in modality_dims]
        )

        # Learnable modality embeddings (position encoding style)
        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, self.uniform_dim) * 0.02
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            self.uniform_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network (per-token)
        self.ffn = nn.Sequential(
            nn.Linear(self.uniform_dim, self.uniform_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.uniform_dim * 4, self.uniform_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(self.uniform_dim)
        self.norm2 = nn.LayerNorm(self.uniform_dim)

        # Output projection
        self.output_proj = nn.Linear(self.uniform_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with true cross-modal attention.

        Args:
            x: Concatenated modality features [batch, input_dim]

        Returns:
            Fused features [batch, output_dim]
        """
        # Split concatenated features back into separate modalities
        modality_features = []
        start_idx = 0
        for i, modality_dim in enumerate(self.modality_dims):
            end_idx = start_idx + modality_dim
            modality_feat = x[:, start_idx:end_idx]  # [batch, modality_dim]

            modality_features.append(modality_feat)
            start_idx = end_idx

        # Sanity check: should consume entire input
        assert start_idx == x.shape[1], (
            f"Modality dims {self.modality_dims} don't sum to input_dim {x.shape[1]}"
        )

        # Project each modality to uniform dimension
        uniform_features = []
        for i, (feat, proj) in enumerate(
            zip(modality_features, self.modality_projections)
        ):
            uniform_feat = proj(feat)  # [batch, uniform_dim]
            uniform_features.append(uniform_feat)

        # Stack into modality sequence: [batch, num_modalities, uniform_dim]
        modality_tokens = torch.stack(uniform_features, dim=1)

        # Add learned modality embeddings (like positional encoding)
        modality_tokens = modality_tokens + self.modality_embeddings.unsqueeze(0)

        # Multi-head cross-modal attention
        attn_out, _ = self.attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            need_weights=False,
        )

        # Residual + norm
        modality_tokens = self.norm1(modality_tokens + attn_out)

        if self.debug_mode and torch.isnan(modality_tokens).any():
            raise ValueError("[FUSION] NaN detected after norm1")

        # Feed-forward network (applied to each token)
        ffn_out = self.ffn(modality_tokens)

        if self.debug_mode and torch.isnan(ffn_out).any():
            raise ValueError("[FUSION] NaN detected after ffn")

        modality_tokens = self.norm2(modality_tokens + ffn_out)

        if self.debug_mode and torch.isnan(modality_tokens).any():
            raise ValueError("[FUSION] NaN detected after norm2")

        # Pool across modalities (mean pooling)
        fused = modality_tokens.mean(dim=1)  # [batch, uniform_dim]

        if self.debug_mode and torch.isnan(fused).any():
            raise ValueError("[FUSION] NaN detected after pooling")

        # Project to output dimension
        output = self.output_proj(fused)  # [batch, output_dim]

        if self.debug_mode and torch.isnan(output).any():
            raise ValueError("[FUSION] NaN detected after output projection")

        return output


class PathPredictionFusion(nn.Module):
    """Multimodal fusion for path prediction with 4 modalities.

    Combines graph, start, goal, and physics contexts using attention-based
    fusion to generate a unified representation for the pointer decoder.

    Architecture:
    - 4 modality encoders (graph, start, goal, physics)
    - Multi-head attention fusion (reuses proven RL architecture)
    - Output: Context embedding for pointer network queries
    """

    def __init__(
        self,
        graph_output_dim: int = 256,
        node_feature_dim: int = 16,
        hidden_dim: int = 128,
        uniform_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        debug_mode: bool = False,
    ):
        """Initialize multimodal fusion module.

        Args:
            graph_output_dim: Dimension of graph embedding from GNN
            node_feature_dim: Dimension of individual node features
            hidden_dim: Hidden dimension for physics encoder
            uniform_dim: Uniform dimension for fusion (256 for 8 heads)
            context_dim: Output dimension for fused context
            num_heads: Number of attention heads
            dropout: Dropout rate
            debug_mode: Enable NaN checking for debugging
        """
        super().__init__()

        self.graph_output_dim = graph_output_dim
        self.node_feature_dim = node_feature_dim
        self.uniform_dim = uniform_dim
        self.context_dim = context_dim
        self.debug_mode = debug_mode

        # Modality 1: Graph encoder (global context)
        # Takes graph embedding from GNN and projects to uniform dimension
        self.graph_encoder = nn.Sequential(
            nn.Linear(graph_output_dim, uniform_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Modality 2: Start encoder (node embedding + local subgraph context)
        # Takes start node embedding from GNN and projects to uniform dimension
        # NOTE: Uses graph_output_dim (256) not node_feature_dim (16) because
        # we receive processed node embeddings from GNN, not raw node features
        self.start_encoder = nn.Sequential(
            nn.Linear(graph_output_dim, uniform_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Modality 3: Goal encoder (pooled goals + reachability)
        # Takes pooled goal node embeddings from GNN and projects to uniform dimension
        # NOTE: Uses graph_output_dim (256) not node_feature_dim (16) because
        # we receive processed node embeddings from GNN, not raw node features
        self.goal_encoder = nn.Sequential(
            nn.Linear(graph_output_dim, uniform_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Modality 4: Physics encoder (full ninja state)
        # Takes full 40-dim ninja state and projects through MLP
        self.physics_encoder = nn.Sequential(
            nn.Linear(GAME_STATE_CHANNELS, hidden_dim),  # 40 -> hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, uniform_dim),  # hidden_dim -> uniform_dim
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion: Reuse MultiHeadFusion from RL feature extractor
        # All 4 modalities have uniform_dim, so modality_dims are equal
        self.fusion = MultiHeadFusion(
            input_dim=4 * uniform_dim,
            output_dim=context_dim,
            num_heads=num_heads,
            modality_dims=[uniform_dim, uniform_dim, uniform_dim, uniform_dim],
            dropout=dropout,
            debug_mode=debug_mode,
        )

        logger.info(
            f"Initialized PathPredictionFusion: "
            f"4 modalities × {uniform_dim}D → {num_heads} heads → {context_dim}D context"
        )

    def forward(
        self,
        graph_embedding: torch.Tensor,
        start_node_embedding: torch.Tensor,
        goal_node_embeddings: torch.Tensor,
        ninja_state: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse multiple modalities into unified context.

        Args:
            graph_embedding: Global graph features [batch, graph_output_dim]
            start_node_embedding: Start node features [batch, node_feature_dim]
            goal_node_embeddings: Goal node features [batch, num_goals, node_feature_dim]
                                 or [batch, node_feature_dim] if already pooled
            ninja_state: Full ninja physics state [batch, GAME_STATE_CHANNELS (40)]

        Returns:
            Fused context embedding [batch, context_dim]
        """
        batch_size = graph_embedding.size(0)

        # Encode modality 1: Graph structure
        graph_feat = self.graph_encoder(graph_embedding)  # [batch, uniform_dim]

        # Encode modality 2: Start context
        start_feat = self.start_encoder(start_node_embedding)  # [batch, uniform_dim]

        # Encode modality 3: Goal context
        # If multiple goals, pool them first
        if goal_node_embeddings.dim() == 3:
            # [batch, num_goals, node_feature_dim] -> [batch, node_feature_dim]
            goal_pooled = goal_node_embeddings.mean(dim=1)
        else:
            # Already pooled: [batch, node_feature_dim]
            goal_pooled = goal_node_embeddings
        goal_feat = self.goal_encoder(goal_pooled)  # [batch, uniform_dim]

        # Encode modality 4: Physics state
        physics_feat = self.physics_encoder(ninja_state)  # [batch, uniform_dim]

        # Concatenate all modalities
        # [batch, 4 * uniform_dim]
        combined = torch.cat([graph_feat, start_feat, goal_feat, physics_feat], dim=1)

        # Fuse with multi-head attention
        fused_context = self.fusion(combined)  # [batch, context_dim]

        return fused_context

    def extract_modality_features(
        self,
        graph_embedding: torch.Tensor,
        start_node_embedding: torch.Tensor,
        goal_node_embeddings: torch.Tensor,
        ninja_state: torch.Tensor,
    ) -> dict:
        """Extract individual modality features (for debugging/visualization).

        Returns:
            Dictionary with encoded features for each modality
        """
        graph_feat = self.graph_encoder(graph_embedding)
        start_feat = self.start_encoder(start_node_embedding)

        if goal_node_embeddings.dim() == 3:
            goal_pooled = goal_node_embeddings.mean(dim=1)
        else:
            goal_pooled = goal_node_embeddings
        goal_feat = self.goal_encoder(goal_pooled)

        physics_feat = self.physics_encoder(ninja_state)

        return {
            "graph": graph_feat,
            "start": start_feat,
            "goal": goal_feat,
            "physics": physics_feat,
        }
