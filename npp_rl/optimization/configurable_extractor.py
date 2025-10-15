"""
Configurable multimodal feature extractor for architecture comparison (Task 3.1).

This extractor allows enabling/disabling different modalities (temporal frames,
global view, graph, state, reachability) to test which features are necessary
for effective N++ learning.

Based on Task 3.1 vision-free architecture analysis requirements.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from .architecture_configs import ArchitectureConfig, GraphArchitectureType, FusionType
from ..models.gcn import GCNEncoder
from ..models.gat import GATEncoder
from ..models.simplified_hgt import SimplifiedHGTEncoder
from ..models.hgt_factory import create_hgt_encoder


class ConfigurableMultimodalExtractor(BaseFeaturesExtractor):
    """
    Configurable multimodal feature extractor for Task 3.1 experiments.

    Allows selective enabling/disabling of modalities:
    - Temporal frames (3D CNN on 84x84x12)
    - Global view (2D CNN on 176x100)
    - Graph (HGT/GAT/GCN/None)
    - Game state (30-dim vector)
    - Reachability (8-dim vector)

    This enables systematic comparison of architecture variants.
    """

    def __init__(self, observation_space: gym.spaces.Dict, config: ArchitectureConfig):
        """
        Args:
            observation_space: Gymnasium Dict space with observation components
            config: ArchitectureConfig specifying which modalities to use
        """
        # Initialize with final features dimension
        super().__init__(observation_space, config.features_dim)

        self.config = config
        self.modalities = config.modalities

        # Track which modalities are actually available
        self.has_temporal = "player_frame" in observation_space.spaces
        self.has_global = "global_view" in observation_space.spaces
        self.has_graph = "graph_obs" in observation_space.spaces
        self.has_state = "game_state" in observation_space.spaces
        self.has_reachability = "reachability_features" in observation_space.spaces

        # Initialize enabled modality processors
        self.temporal_cnn = None
        self.global_cnn = None
        self.graph_encoder = None
        self.state_mlp = None
        self.reachability_mlp = None

        feature_dims = []

        # 1. Temporal frames processing (3D CNN)
        if self.modalities.use_temporal_frames and self.has_temporal:
            self.temporal_cnn = self._create_temporal_cnn(config.visual)
            feature_dims.append(config.visual.temporal_output_dim)

        # 2. Global view processing (2D CNN)
        if self.modalities.use_global_view and self.has_global:
            self.global_cnn = self._create_global_cnn(config.visual)
            feature_dims.append(config.visual.global_output_dim)

        # 3. Graph processing
        if self.modalities.use_graph and self.has_graph:
            self.graph_encoder = self._create_graph_encoder(config.graph)
            feature_dims.append(config.graph.output_dim)

        # 4. Game state processing
        if self.modalities.use_game_state and self.has_state:
            self.state_mlp = self._create_state_mlp(
                config.state.game_state_dim,
                config.state.hidden_dim,
                config.state.output_dim,
            )
            feature_dims.append(config.state.output_dim)

        # 5. Reachability features processing
        if self.modalities.use_reachability and self.has_reachability:
            self.reachability_mlp = self._create_reachability_mlp(
                config.state.reachability_dim,
                config.state.hidden_dim // 2,
                config.state.output_dim // 2,
            )
            feature_dims.append(config.state.output_dim // 2)

        # Create fusion mechanism
        total_input_dim = sum(feature_dims)
        self.fusion = self._create_fusion(
            total_input_dim, config.features_dim, config.fusion
        )

    def _create_temporal_cnn(self, visual_config) -> nn.Module:
        """Create 3D CNN for temporal frame processing."""
        return nn.Sequential(
            # Input: [batch, 1, 12, 84, 84] (channels, time, height, width)
            nn.Conv3d(
                1,
                visual_config.temporal_channels[0],
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(visual_config.temporal_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(visual_config.cnn_dropout),
            nn.Conv3d(
                visual_config.temporal_channels[0],
                visual_config.temporal_channels[1],
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(visual_config.temporal_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(visual_config.cnn_dropout),
            nn.Conv3d(
                visual_config.temporal_channels[1],
                visual_config.temporal_channels[2],
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.BatchNorm3d(visual_config.temporal_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 4, 4)),  # Pool to fixed size
            nn.Flatten(),
            nn.Linear(
                visual_config.temporal_channels[2] * 1 * 4 * 4,
                visual_config.temporal_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def _create_global_cnn(self, visual_config) -> nn.Module:
        """Create 2D CNN for global view processing."""
        return nn.Sequential(
            # Input: [batch, 1, 176, 100]
            nn.Conv2d(
                1, visual_config.global_channels[0], kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(visual_config.global_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(visual_config.cnn_dropout),
            nn.Conv2d(
                visual_config.global_channels[0],
                visual_config.global_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(visual_config.global_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(visual_config.cnn_dropout),
            nn.Conv2d(
                visual_config.global_channels[1],
                visual_config.global_channels[2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(visual_config.global_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(
                visual_config.global_channels[2] * 4 * 4,
                visual_config.global_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def _create_graph_encoder(self, graph_config) -> nn.Module:
        """Create graph encoder based on architecture type."""
        # N++ graph observations from nclone have 67-dimensional node features
        # This is the standard dimension from HierarchicalGraphBuilder in nclone
        node_feature_dim = 67  # N++ graph node features from nclone

        arch_type = graph_config.architecture

        if arch_type == GraphArchitectureType.FULL_HGT:
            # Use existing production HGT
            return create_hgt_encoder(
                node_feature_dim=node_feature_dim,
                edge_feature_dim=9,
                hidden_dim=graph_config.hidden_dim,
                num_layers=graph_config.num_layers,
                output_dim=graph_config.output_dim,
                num_heads=graph_config.num_heads,
            )

        elif arch_type == GraphArchitectureType.SIMPLIFIED_HGT:
            return SimplifiedHGTEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=graph_config.hidden_dim,
                output_dim=graph_config.output_dim,
                num_layers=graph_config.num_layers,
                num_heads=graph_config.num_heads,
                num_node_types=graph_config.num_node_types,
                num_edge_types=graph_config.num_edge_types,
                dropout=graph_config.dropout,
            )

        elif arch_type == GraphArchitectureType.GAT:
            return GATEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=graph_config.hidden_dim,
                output_dim=graph_config.output_dim,
                num_layers=graph_config.num_layers,
                num_heads=graph_config.num_heads,
                dropout=graph_config.dropout,
            )

        elif arch_type == GraphArchitectureType.GCN:
            return GCNEncoder(
                node_feature_dim=node_feature_dim,
                hidden_dim=graph_config.hidden_dim,
                output_dim=graph_config.output_dim,
                num_layers=graph_config.num_layers,
                dropout=graph_config.dropout,
            )

        else:
            raise ValueError(f"Unknown graph architecture: {arch_type}")

    def _create_state_mlp(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> nn.Module:
        """Create MLP for game state processing."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def _create_reachability_mlp(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> nn.Module:
        """Create MLP for reachability feature processing."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def _create_fusion(
        self, input_dim: int, output_dim: int, fusion_config
    ) -> nn.Module:
        """Create multimodal fusion mechanism."""
        fusion_type = fusion_config.fusion_type

        if fusion_type == FusionType.CONCAT:
            # Simple concatenation + MLP
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(fusion_config.dropout),
                nn.Linear(output_dim, output_dim),
            )

        elif fusion_type == FusionType.SINGLE_HEAD_ATTENTION:
            # Single-head cross-modal attention
            return SingleHeadFusion(input_dim, output_dim, fusion_config.dropout)

        elif fusion_type == FusionType.MULTI_HEAD_ATTENTION:
            # Multi-head cross-modal attention
            return MultiHeadFusion(
                input_dim,
                output_dim,
                fusion_config.num_attention_heads,
                fusion_config.dropout,
            )

        else:
            # Default to concatenation
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Feature tensor of shape [batch_size, features_dim]
        """
        features = []

        # Process temporal frames
        if self.temporal_cnn is not None and "player_frame" in observations:
            temporal_obs = observations["player_frame"]
            # Add channel dimension if needed: [batch, 12, 84, 84] -> [batch, 1, 12, 84, 84]
            if temporal_obs.dim() == 4:
                temporal_obs = temporal_obs.unsqueeze(1)
            temporal_features = self.temporal_cnn(temporal_obs.float() / 255.0)
            features.append(temporal_features)

        # Process global view
        if self.global_cnn is not None and "global_view" in observations:
            global_obs = observations["global_view"]
            # Add channel dimension if needed
            if global_obs.dim() == 3:
                global_obs = global_obs.unsqueeze(1)
            global_features = self.global_cnn(global_obs.float() / 255.0)
            features.append(global_features)

        # Process graph
        if self.graph_encoder is not None and "graph_obs" in observations:
            graph_obs = observations["graph_obs"]
            # Extract graph components
            node_features = graph_obs["node_features"]
            edge_index = graph_obs["edge_index"]
            node_mask = graph_obs.get("node_mask", None)

            # Handle different encoder types
            if isinstance(self.graph_encoder, SimplifiedHGTEncoder):
                node_types = graph_obs.get(
                    "node_types", torch.zeros(node_features.shape[:2], dtype=torch.long)
                )
                _, graph_features = self.graph_encoder(
                    node_features, edge_index, node_types, node_mask
                )
            else:
                _, graph_features = self.graph_encoder(
                    node_features, edge_index, node_mask
                )

            features.append(graph_features)

        # Process game state
        if self.state_mlp is not None and "game_state" in observations:
            state_features = self.state_mlp(observations["game_state"])
            features.append(state_features)

        # Process reachability
        if (
            self.reachability_mlp is not None
            and "reachability_features" in observations
        ):
            reach_features = self.reachability_mlp(
                observations["reachability_features"]
            )
            features.append(reach_features)

        # Concatenate all features
        if len(features) == 0:
            raise ValueError(
                "No features were extracted! Check configuration and observation space."
            )

        combined_features = torch.cat(features, dim=1)

        # Apply fusion
        output = self.fusion(combined_features)

        return output


class SingleHeadFusion(nn.Module):
    """Single-head attention fusion for multimodal features."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for attention: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.squeeze(1)
        return self.mlp(x)


class MultiHeadFusion(nn.Module):
    """Multi-head attention fusion for multimodal features."""

    def __init__(
        self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = x.squeeze(1)
        return self.mlp(x)
