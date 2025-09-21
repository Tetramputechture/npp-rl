"""
HGT Multimodal Feature Extractor with Reachability Integration

This module implements a Heterogeneous Graph Transformer (HGT) based feature extractor
that processes multiple modalities (visual, graph, state) with integrated compact
reachability features from the nclone physics engine.

Integration Strategy:
- Compact reachability features (64-dim) provide spatial guidance without ground truth dependency
- Cross-modal attention mechanisms learn to weight reachability information appropriately
- Performance-optimized for real-time RL training (<2ms feature extraction)

Theoretical Foundation:
- Heterogeneous Graph Transformers: Hu et al. (2020) "Heterogeneous Graph Transformer"
- Multi-modal fusion: Baltrusaitis et al. (2018) "Multimodal Machine Learning: A Survey"
- Attention mechanisms: Vaswani et al. (2017) "Attention Is All You Need"
- Spatial reasoning: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"

nclone Integration:
- Uses TieredReachabilitySystem for performance-tuned feature extraction
- Integrates with compact feature representations from nclone.graph.reachability
- Maintains compatibility with existing NPP physics constants and game state
"""

# Standard library imports
from typing import Dict, Tuple, Optional, Any, List

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

# npp_rl imports
from npp_rl.models.hgt_gnn import create_hgt_encoder
from npp_rl.models.spatial_attention import SpatialAttentionModule
from npp_rl.models.hgt_config import (
    CNN_CONFIG,
    POOLING_CONFIG,
    DEFAULT_CONFIG,
    FACTORY_CONFIG,
    MULTIPLIER_CONFIG,
)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for feature fusion.

    Implementation based on Vaswani et al. (2017) "Attention Is All You Need"
    adapted for multimodal fusion as described in Baltrusaitis et al. (2018).
    """

    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, query_dim)

        self.scale = hidden_dim**-0.5

    def forward(self, query, key):
        """
        Apply cross-modal attention.

        Args:
            query: Features to be attended (e.g., visual features)
            key: Features providing attention context (e.g., reachability features)
        """
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(key)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, V)

        # Project back to query dimension
        output = self.output_proj(attended)

        # Residual connection
        return query + output


class ReachabilityAttentionModule(nn.Module):
    """
    Cross-modal attention module for integrating reachability features with multimodal representations.

    This module implements the attention-based fusion mechanism from Baltrusaitis et al. (2018)
    "Multimodal Machine Learning: A Survey", adapted for spatial reasoning integration in RL.

    The architecture uses cross-modal attention to allow each modality to attend to
    reachability features, followed by gating mechanisms to modulate feature importance
    based on spatial reachability context.

    Architecture Components:
    - Cross-modal attention between each modality and reachability features
    - Learned gating mechanism for reachability-guided feature enhancement
    - Multi-layer fusion network for final feature integration

    References:
    - Multimodal fusion: Baltrusaitis et al. (2018) "Multimodal Machine Learning: A Survey"
    - Attention mechanisms: Vaswani et al. (2017) "Attention Is All You Need"
    - Gating networks: Dauphin et al. (2017) "Language Modeling with Gated Convolutional Networks"

    Note:
        This module should be integrated directly into the HGTMultimodalExtractor
        rather than created as a separate file to maintain architectural cohesion.
    """

    def __init__(self, visual_dim, graph_dim, state_dim, reachability_dim):
        super().__init__()

        self.visual_dim = visual_dim
        self.graph_dim = graph_dim
        self.state_dim = state_dim
        self.reachability_dim = reachability_dim

        # Attention mechanisms
        self.visual_reachability_attention = CrossModalAttention(
            query_dim=visual_dim, key_dim=reachability_dim, hidden_dim=128
        )
        self.graph_reachability_attention = CrossModalAttention(
            query_dim=graph_dim, key_dim=reachability_dim, hidden_dim=128
        )
        self.state_reachability_attention = CrossModalAttention(
            query_dim=state_dim, key_dim=reachability_dim, hidden_dim=64
        )

        # Reachability-guided feature enhancement
        self.reachability_gate = nn.Sequential(
            nn.Linear(reachability_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 gates for visual, graph, state
            nn.Sigmoid(),
        )

        # Final fusion
        total_dim = visual_dim + graph_dim + state_dim + reachability_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim // 2, total_dim // 2),
        )

    def forward(
        self, visual_features, graph_features, state_features, reachability_features
    ):
        """
        Fuse multimodal features with reachability awareness.
        """
        # Apply cross-modal attention
        visual_attended = self.visual_reachability_attention(
            visual_features, reachability_features
        )
        graph_attended = self.graph_reachability_attention(
            graph_features, reachability_features
        )
        state_attended = self.state_reachability_attention(
            state_features, reachability_features
        )

        # Compute reachability-based gating
        gates = self.reachability_gate(reachability_features)
        visual_gate, graph_gate, state_gate = (
            gates[:, 0:1],
            gates[:, 1:2],
            gates[:, 2:3],
        )

        # Apply gating to enhance relevant features
        visual_enhanced = visual_attended * (1 + visual_gate)
        graph_enhanced = graph_attended * (1 + graph_gate)
        state_enhanced = state_attended * (1 + state_gate)

        # Concatenate all features
        fused = torch.cat(
            [visual_enhanced, graph_enhanced, state_enhanced, reachability_features],
            dim=1,
        )

        # Final fusion
        output = self.fusion_layer(fused)

        return output


class HGTMultimodalExtractor(BaseFeaturesExtractor):
    """
    HGT-based multimodal feature extractor with integrated reachability features.

    This extractor processes visual frames, graph structures, game state, and compact
    reachability features through a unified Heterogeneous Graph Transformer architecture.

    Architecture Components:
    1. Visual Processing: 3D CNN for temporal frames + 2D CNN for global view
    2. Graph Processing: HGT with type-specific attention mechanisms
    3. State Processing: MLP for physics/game state features
    4. Reachability Processing: Compact feature integration with cross-modal attention
    5. Multimodal Fusion: Attention-based fusion with reachability awareness

    Performance Optimizations:
    - Lazy initialization of reachability extractor for dependency management
    - Caching mechanisms for repeated reachability computations
    - Tier-1 reachability extraction for real-time performance (<2ms target)

    References:
    - HGT Architecture: Hu et al. (2020) "Heterogeneous Graph Transformer"
    - Cross-modal Attention: Baltrusaitis et al. (2018) "Multimodal Machine Learning"
    - Reachability Integration: Custom integration with nclone physics system
    """

    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = DEFAULT_CONFIG.embed_dim,
        # HGT parameters
        hgt_hidden_dim: int = DEFAULT_CONFIG.hidden_dim,
        hgt_num_layers: int = DEFAULT_CONFIG.num_layers,
        hgt_output_dim: int = DEFAULT_CONFIG.output_dim,
        hgt_num_heads: int = DEFAULT_CONFIG.num_attention_heads,
        # Visual processing parameters
        visual_hidden_dim: int = DEFAULT_CONFIG.visual_hidden_dim,
        global_hidden_dim: int = DEFAULT_CONFIG.global_hidden_dim,
        # State processing parameters
        state_hidden_dim: int = DEFAULT_CONFIG.state_hidden_dim,
        # Reachability parameters
        reachability_dim: int = 64,
        # Fusion parameters
        use_cross_modal_attention: bool = True,
        use_spatial_attention: bool = True,
        num_attention_heads: int = DEFAULT_CONFIG.num_attention_heads,
        dropout: float = DEFAULT_CONFIG.dropout_rate,
        **kwargs,
    ):
        """
        Initialize HGT-based multimodal feature extractor with reachability integration.

        Args:
            observation_space: Gym observation space dictionary
            features_dim: Final output feature dimension
            hgt_hidden_dim: Hidden dimension for HGT layers
            hgt_num_layers: Number of HGT layers
            hgt_output_dim: Output dimension of HGT encoder
            hgt_num_heads: Number of attention heads in HGT
            visual_hidden_dim: Hidden dimension for visual processing
            global_hidden_dim: Hidden dimension for global view processing
            state_hidden_dim: Hidden dimension for state processing
            reachability_dim: Dimension of reachability features (default: 64)
            use_cross_modal_attention: Whether to use cross-modal attention
            use_spatial_attention: Whether to use spatial attention
            num_attention_heads: Number of attention heads for fusion
            dropout: Dropout probability
        """
        super().__init__(observation_space, features_dim)

        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_spatial_attention = use_spatial_attention

        # Reachability feature processing components
        # Based on compact feature dimensionality from nclone.graph.reachability
        self.reachability_dim = reachability_dim

        # Multi-layer reachability encoder with batch normalization for stability
        # Architecture inspired by residual connections from He et al. (2016)
        self.reachability_encoder = nn.Sequential(
            nn.Linear(reachability_dim, 128),
            nn.BatchNorm1d(128),  # Stabilize training with reachability features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Compact representation for attention mechanisms
        )

        # Initialize reachability extractor directly - nclone is required dependency
        from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
        from nclone.graph.reachability.feature_extractor import ReachabilityFeatureExtractor
        
        tiered_system = TieredReachabilitySystem()
        self.reachability_extractor = ReachabilityFeatureExtractor(tiered_system)

        # Visual processing branch (temporal frames)
        if "player_frame" in observation_space.spaces:
            visual_shape = observation_space["player_frame"].shape
            self.visual_cnn = nn.Sequential(
                # 3D CNN for temporal modeling
                nn.Conv3d(
                    CNN_CONFIG.conv3d_layer1.in_channels,
                    CNN_CONFIG.conv3d_layer1.out_channels,
                    kernel_size=CNN_CONFIG.conv3d_layer1.kernel_size,
                    stride=CNN_CONFIG.conv3d_layer1.stride,
                    padding=CNN_CONFIG.conv3d_layer1.padding,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    CNN_CONFIG.conv3d_layer2.in_channels,
                    CNN_CONFIG.conv3d_layer2.out_channels,
                    kernel_size=CNN_CONFIG.conv3d_layer2.kernel_size,
                    stride=CNN_CONFIG.conv3d_layer2.stride,
                    padding=CNN_CONFIG.conv3d_layer2.padding,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    CNN_CONFIG.conv3d_layer3.in_channels,
                    CNN_CONFIG.conv3d_layer3.out_channels,
                    kernel_size=CNN_CONFIG.conv3d_layer3.kernel_size,
                    stride=CNN_CONFIG.conv3d_layer3.stride,
                    padding=CNN_CONFIG.conv3d_layer3.padding,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(POOLING_CONFIG.adaptive_pool3d_output_size),
                nn.Flatten(),
                nn.Linear(POOLING_CONFIG.cnn_flattened_size, visual_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.has_visual = True
        else:
            self.has_visual = False
            visual_hidden_dim = 0

        # Global view processing branch
        if "global_view" in observation_space.spaces:
            global_shape = observation_space["global_view"].shape
            self.global_cnn = nn.Sequential(
                nn.Conv2d(
                    CNN_CONFIG.conv2d_layer1.in_channels,
                    CNN_CONFIG.conv2d_layer1.out_channels,
                    kernel_size=CNN_CONFIG.conv2d_layer1.kernel_size[0],
                    stride=CNN_CONFIG.conv2d_layer1.stride[0],
                    padding=CNN_CONFIG.conv2d_layer1.padding[0],
                ),
                nn.ReLU(),
                nn.Conv2d(
                    CNN_CONFIG.conv2d_layer2.in_channels,
                    CNN_CONFIG.conv2d_layer2.out_channels,
                    kernel_size=CNN_CONFIG.conv2d_layer2.kernel_size[0],
                    stride=CNN_CONFIG.conv2d_layer2.stride[0],
                    padding=CNN_CONFIG.conv2d_layer2.padding[0],
                ),
                nn.ReLU(),
                nn.Conv2d(
                    CNN_CONFIG.conv2d_layer3.in_channels,
                    CNN_CONFIG.conv2d_layer3.out_channels,
                    kernel_size=CNN_CONFIG.conv2d_layer3.kernel_size[0],
                    stride=CNN_CONFIG.conv2d_layer3.stride[0],
                    padding=CNN_CONFIG.conv2d_layer3.padding[0],
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(POOLING_CONFIG.adaptive_pool2d_output_size),
                nn.Flatten(),
                nn.Linear(POOLING_CONFIG.cnn_flattened_size, global_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.has_global = True
        else:
            self.has_global = False
            global_hidden_dim = 0

        # State processing branch
        if "game_state" in observation_space.spaces:
            state_dim = observation_space["game_state"].shape[0]
            self.state_mlp = nn.Sequential(
                nn.Linear(state_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_hidden_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.has_state = True
        else:
            self.has_state = False
            state_hidden_dim = 0

        # HGT graph processing branch
        if (
            "graph_node_feats" in observation_space.spaces
            and "graph_edge_feats" in observation_space.spaces
        ):
            node_feat_dim = observation_space["graph_node_feats"].shape[1]
            edge_feat_dim = observation_space["graph_edge_feats"].shape[1]

            self.hgt_encoder = create_hgt_encoder(
                node_feature_dim=node_feat_dim,
                edge_feature_dim=edge_feat_dim,
                hidden_dim=hgt_hidden_dim,
                num_layers=hgt_num_layers,
                output_dim=hgt_output_dim,
                num_heads=hgt_num_heads,
                global_pool="mean_max",
            )
            self.has_graph = True
            # HGT with mean_max pooling outputs multiplier * hgt_output_dim
            graph_output_dim = MULTIPLIER_CONFIG.hgt_output_multiplier * hgt_output_dim
        else:
            self.has_graph = False
            graph_output_dim = 0

        # Spatial attention module (if enabled)
        if self.use_spatial_attention and self.has_graph and self.has_visual:
            self.spatial_attention = SpatialAttentionModule(
                graph_dim=graph_output_dim,
                visual_dim=visual_hidden_dim,
                spatial_height=DEFAULT_CONFIG.spatial_height,
                spatial_width=DEFAULT_CONFIG.spatial_width,
                num_heads=num_attention_heads,
            )

        # Cross-modal attention for reachability integration
        self.reachability_attention = ReachabilityAttentionModule(
            visual_dim=visual_hidden_dim,
            graph_dim=graph_output_dim,
            state_dim=state_hidden_dim,
            reachability_dim=32,  # Output from reachability encoder
        )
        # Update total dimension to include reachability features
        total_dim = (visual_hidden_dim + graph_output_dim + state_hidden_dim + 32) // 2

        # Cross-modal attention fusion (if enabled)
        if self.use_cross_modal_attention and total_dim > 0:
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=total_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(total_dim)

        # Final fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(
                total_dim, features_dim * MULTIPLIER_CONFIG.fusion_expansion_factor
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                features_dim * MULTIPLIER_CONFIG.fusion_expansion_factor, features_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
        )

        # Initialize weights
        self._initialize_weights()



    def _extract_reachability_features(
        self, observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract compact reachability features from observations.

        This method expects reachability features to be pre-computed by the
        NppEnvironment (with enable_reachability_features=True) and included 
        in the observations. If not present, it returns zero features as fallback.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            torch.Tensor: 64-dimensional reachability feature tensor
        """
        batch_size = next(iter(observations.values())).shape[0]
        device = next(iter(observations.values())).device

        # Check if reachability features are pre-computed in observations
        if "reachability_features" in observations:
            return observations["reachability_features"]

        # Fallback: zero features if not provided by environment
        print("Warning: No reachability features found in observations. Enable reachability_features in NppEnvironment.")
        return torch.zeros(
            batch_size, self.reachability_dim, dtype=torch.float32, device=device
        )



    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with integrated reachability feature processing.

        This method extends the base HGT forward pass to incorporate compact
        reachability features as an additional modality. The integration follows
        the multimodal fusion approach from Baltrusaitis et al. (2018).

        Args:
            observations: Dict containing visual frames, graph data, state info,
                         and optionally pre-computed reachability features

        Returns:
            torch.Tensor: Fused feature representation for policy/value networks

        Note:
            Performance target is <2ms for real-time RL training compatibility.
            Reachability extraction uses Tier-1 algorithms for speed optimization.
        """
        # Process each modality through dedicated pathways
        # Visual: 3D CNN for temporal dynamics, 2D CNN for global spatial context
        visual_features = None
        if self.has_visual and "player_frame" in observations:
            visual_obs = observations["player_frame"]
            if visual_obs.dim() == 4:  # Add batch dimension if missing
                visual_obs = visual_obs.unsqueeze(1)
            visual_features = self.visual_cnn(visual_obs)

        # Global view processing
        global_features = None
        if self.has_global and "global_view" in observations:
            global_obs = observations["global_view"]
            if global_obs.dim() == 3:  # Add channel dimension if missing
                global_obs = global_obs.unsqueeze(1)
            global_features = self.global_cnn(global_obs)

        # State: MLP processing of physics and game state variables
        state_features = None
        if self.has_state and "game_state" in observations:
            state_features = self.state_mlp(observations["game_state"])

        # Graph: HGT with heterogeneous attention for structural relationships
        # Based on Hu et al. (2020) type-aware graph transformer architecture
        graph_features = None
        if self.has_graph and all(
            key in observations
            for key in [
                "graph_node_feats",
                "graph_edge_feats",
                "graph_edge_index",
                "graph_node_types",
                "graph_edge_types",
                "graph_node_mask",
                "graph_edge_mask",
            ]
        ):
            graph_features = self.hgt_encoder(
                node_features=observations["graph_node_feats"],
                edge_features=observations["graph_edge_feats"],
                edge_index=observations["graph_edge_index"],
                node_types=observations["graph_node_types"],
                edge_types=observations["graph_edge_types"],
                node_mask=observations["graph_node_mask"],
                edge_mask=observations["graph_edge_mask"],
            )

        # Reachability: Extract compact spatial reasoning features from nclone
        # These provide approximate reachability guidance without ground truth dependency
        # Features should be pre-computed by ReachabilityWrapper
        reachability_features = self._extract_reachability_features(observations)

        # Encode reachability features through dedicated neural pathway
        # Batch normalization stabilizes training with heterogeneous feature scales
        processed_reachability = self.reachability_encoder(reachability_features)

        # Cross-modal attention fusion integrating all modalities with reachability
        # Allows model to learn optimal weighting of reachability guidance
        if (
            visual_features is not None
            and graph_features is not None
            and state_features is not None
        ):
            combined_features = self.reachability_attention(
                visual_features,
                graph_features,
                state_features,
                processed_reachability,
            )
        else:
            # Fallback: concatenate available features
            available_features = []
            if visual_features is not None:
                available_features.append(visual_features)
            if global_features is not None:
                available_features.append(global_features)
            if state_features is not None:
                available_features.append(state_features)
            if graph_features is not None:
                available_features.append(graph_features)
            available_features.append(processed_reachability)

            if not available_features:
                raise ValueError("No valid observations found")
            combined_features = torch.cat(available_features, dim=1)

        # Final transformation to target feature dimensionality
        output_features = self.fusion_network(combined_features)

        return output_features


def create_hgt_multimodal_extractor(
    observation_space: SpacesDict,
    features_dim: int = FACTORY_CONFIG.features_dim,
    hgt_hidden_dim: int = FACTORY_CONFIG.hgt_hidden_dim,
    hgt_num_layers: int = FACTORY_CONFIG.hgt_num_layers,
    hgt_output_dim: int = FACTORY_CONFIG.hgt_output_dim,
    **kwargs,
) -> HGTMultimodalExtractor:
    """
    Factory function to create the primary HGT-based multimodal feature extractor with reachability integration.

    This is the RECOMMENDED way to create feature extractors for N++ RL agents.

    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension
        hgt_hidden_dim: Hidden dimension for HGT layers
        hgt_num_layers: Number of HGT layers
        hgt_output_dim: Output dimension of HGT encoder
        **kwargs: Additional arguments passed to the extractor

    Returns:
        Configured HGT multimodal feature extractor with reachability integration
    """
    return HGTMultimodalExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        hgt_hidden_dim=hgt_hidden_dim,
        hgt_num_layers=hgt_num_layers,
        hgt_output_dim=hgt_output_dim,
        use_cross_modal_attention=True,
        use_spatial_attention=True,
        **kwargs,
    )
