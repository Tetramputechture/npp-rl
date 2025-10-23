"""
Configurable multimodal feature extractor for architecture comparison.

This extractor allows enabling/disabling different modalities (player frames,
global view, graph, state, reachability) to test which features are necessary
for effective N++ learning.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

# Import nclone constants for graph dimensions
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM

from npp_rl.training.architecture_configs import (
    ArchitectureConfig,
    GraphArchitectureType,
    FusionType,
)
from npp_rl.models.gcn import GCNEncoder
from npp_rl.models.gat import GATEncoder
from npp_rl.models.simplified_hgt import SimplifiedHGTEncoder
from npp_rl.models.hgt_factory import create_hgt_encoder


class PlayerFrameCNN(nn.Module):
    """CNN for player frame processing with dynamic frame stacking support."""
    
    def __init__(self, visual_config):
        super().__init__()
        self.visual_config = visual_config
        self.conv_layers = None
        self.fc = None
    
    def _build_layers(self, in_channels: int):
        """Build convolutional layers based on input channel count."""
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.visual_config.player_frame_channels[0],
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.BatchNorm2d(self.visual_config.player_frame_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.visual_config.cnn_dropout),
            nn.Conv2d(
                self.visual_config.player_frame_channels[0],
                self.visual_config.player_frame_channels[1],
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(self.visual_config.player_frame_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.visual_config.cnn_dropout),
            nn.Conv2d(
                self.visual_config.player_frame_channels[1],
                self.visual_config.player_frame_channels[2],
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.visual_config.player_frame_channels[2]),
            nn.ReLU(inplace=True),
        )
        # Fully connected layers
        # After conv layers (kernel 8 stride 4, kernel 4 stride 2, kernel 3 stride 1),
        # 84x84 input is reduced to 7x7 spatial dimensions
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_config.player_frame_channels[2] * 7 * 7,
                self.visual_config.player_frame_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        """Forward pass with dynamic input channel handling."""
        # Handle different input shapes:
        # Single frame: [batch, H, W, C] or [batch, C, H, W]
        # Stacked frames: [batch, stack_size, H, W, C]
        
        if x.dim() == 4 and x.shape[-1] == 1:
            # [batch, H, W, 1] -> [batch, 1, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 5:
            # [batch, stack_size, H, W, C] -> [batch, stack_size*C, H, W]
            batch_size, stack_size, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = x.view(batch_size, stack_size * C, H, W)
        
        # Build layers on first forward pass
        if self.conv_layers is None:
            in_channels = x.shape[1]
            self._build_layers(in_channels)
            self.conv_layers = self.conv_layers.to(x.device)
            self.fc = self.fc.to(x.device)
        
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class GlobalViewCNN(nn.Module):
    """CNN for global view processing with dynamic frame stacking support."""
    
    def __init__(self, visual_config):
        super().__init__()
        self.visual_config = visual_config
        self.conv_layers = None
        self.fc = None
    
    def _build_layers(self, in_channels: int):
        """Build convolutional layers based on input channel count."""
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels, self.visual_config.global_channels[0], 
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.visual_config.cnn_dropout),
            nn.Conv2d(
                self.visual_config.global_channels[0],
                self.visual_config.global_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.visual_config.cnn_dropout),
            nn.Conv2d(
                self.visual_config.global_channels[1],
                self.visual_config.global_channels[2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Pools to fixed 4x4 spatial size
        )
        # Fully connected layers
        # AdaptiveAvgPool2d ensures output is always 4x4 regardless of input size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_config.global_channels[2] * 4 * 4,
                self.visual_config.global_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        """Forward pass with dynamic input channel handling."""
        # Handle different input shapes similar to PlayerFrameCNN
        if x.dim() == 4 and x.shape[-1] == 1:
            # [batch, H, W, 1] -> [batch, 1, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 5:
            # [batch, stack_size, H, W, C] -> [batch, stack_size*C, H, W]
            batch_size, stack_size, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = x.view(batch_size, stack_size * C, H, W)
        
        # Build layers on first forward pass
        if self.conv_layers is None:
            in_channels = x.shape[1]
            self._build_layers(in_channels)
            self.conv_layers = self.conv_layers.to(x.device)
            self.fc = self.fc.to(x.device)
        
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


class StateMLP(nn.Module):
    """MLP for game state processing with dynamic state stacking support."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.base_input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp = None
    
    def _build_layers(self, actual_input_dim: int):
        """Build MLP layers based on actual input dimension (may be stacked)."""
        self.mlp = nn.Sequential(
            nn.Linear(actual_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """Forward pass with dynamic input dimension handling.
        
        Handles:
        - Single state: [batch, state_dim]
        - Stacked states: [batch, stack_size, state_dim] -> flatten to [batch, stack_size * state_dim]
        """
        if x.dim() == 3:
            # Stacked states: [batch, stack_size, state_dim] -> [batch, stack_size * state_dim]
            batch_size, stack_size, state_dim = x.shape
            x = x.view(batch_size, stack_size * state_dim)
        
        # Build layers on first forward pass
        if self.mlp is None:
            actual_input_dim = x.shape[-1]
            self._build_layers(actual_input_dim)
            self.mlp = self.mlp.to(x.device)
        
        return self.mlp(x)


class ConfigurableMultimodalExtractor(BaseFeaturesExtractor):
    """
    Configurable multimodal feature extractor for Task 3.1 experiments.

    Allows selective enabling/disabling of modalities:
    - Player frame (2D CNN on 84x84x1 grayscale)
    - Global view (2D CNN on 176x100x1 grayscale)
    - Graph (HGT/GAT/GCN/None)
    - Game state (26-dim vector, ninja_state only)
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
        self.has_player_frame = "player_frame" in observation_space.spaces
        self.has_global = "global_view" in observation_space.spaces
        # Graph observations come as separate keys (graph_node_feats, graph_edge_index, etc.)
        self.has_graph = "graph_node_feats" in observation_space.spaces
        self.has_state = "game_state" in observation_space.spaces
        self.has_reachability = "reachability_features" in observation_space.spaces

        # Initialize enabled modality processors
        self.player_frame_cnn = None
        self.global_cnn = None
        self.graph_encoder = None
        self.state_mlp = None
        self.reachability_mlp = None

        feature_dims = []

        # 1. Player frame processing (2D CNN for single grayscale frame)
        if self.has_player_frame:
            self.player_frame_cnn = self._create_player_frame_cnn(config.visual)
            feature_dims.append(config.visual.player_frame_output_dim)

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

    def _create_player_frame_cnn(self, visual_config) -> nn.Module:
        """
        Create 2D CNN for grayscale frame processing with optional frame stacking support.

        Handles both single frames and stacked frames:
        - Single frame: [batch, H, W, 1] -> [batch, 1, H, W] after channel reordering
        - Stacked frames: [batch, stack_size, H, W, 1] -> [batch, stack_size, H, W] after channel reordering
        
        For stacked frames, we treat the stack dimension as the input channel dimension,
        allowing the network to learn temporal patterns through the convolutions.

        Input shapes:
        - Single frame: [batch, 1, 84, 84] (grayscale)
        - Stacked frames: [batch, stack_size, 84, 84] (e.g., [batch, 4, 84, 84] for 4-frame stack)
        
        Output: [batch, player_frame_output_dim] features
        
        References:
        - Mnih et al. (2015): DQN uses 4-frame stacking with similar CNN architecture
        """
        # Determine input channels dynamically based on observation space
        # Will be set during forward pass based on actual input dimensions
        # Default to 1 for single frame, but handle stacked frames dynamically
        return PlayerFrameCNN(visual_config)

    def _create_global_cnn(self, visual_config) -> nn.Module:
        """Create 2D CNN for global view processing with frame stacking support."""
        return GlobalViewCNN(visual_config)

    def _create_graph_encoder(self, graph_config) -> nn.Module:
        """Create graph encoder based on architecture type."""
        arch_type = graph_config.architecture

        if arch_type == GraphArchitectureType.FULL_HGT:
            # Use existing production HGT
            return create_hgt_encoder(
                node_feature_dim=NODE_FEATURE_DIM,
                edge_feature_dim=EDGE_FEATURE_DIM,
                hidden_dim=graph_config.hidden_dim,
                num_layers=graph_config.num_layers,
                output_dim=graph_config.output_dim,
                num_heads=graph_config.num_heads,
            )

        elif arch_type == GraphArchitectureType.SIMPLIFIED_HGT:
            return SimplifiedHGTEncoder(
                node_feature_dim=NODE_FEATURE_DIM,
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
                node_feature_dim=NODE_FEATURE_DIM,
                hidden_dim=graph_config.hidden_dim,
                output_dim=graph_config.output_dim,
                num_layers=graph_config.num_layers,
                num_heads=graph_config.num_heads,
                dropout=graph_config.dropout,
            )

        elif arch_type == GraphArchitectureType.GCN:
            return GCNEncoder(
                node_feature_dim=NODE_FEATURE_DIM,
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
        """Create MLP for game state processing with stacking support."""
        return StateMLP(input_dim, hidden_dim, output_dim)

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

        # Process player frame (single grayscale frame)
        if self.player_frame_cnn is not None and "player_frame" in observations:
            player_frame_obs = observations["player_frame"]
            # Handle different input formats
            if player_frame_obs.dim() == 4:
                # [batch, H, W, 1] -> [batch, 1, H, W]
                if player_frame_obs.shape[-1] == 1:
                    player_frame_obs = player_frame_obs.permute(0, 3, 1, 2)
                # [batch, 1, H, W] -> already correct
            elif player_frame_obs.dim() == 3:
                # [batch, H, W] -> [batch, 1, H, W]
                player_frame_obs = player_frame_obs.unsqueeze(1)
            player_frame_features = self.player_frame_cnn(
                player_frame_obs.float() / 255.0
            )
            features.append(player_frame_features)

        # Process global view
        if self.global_cnn is not None and "global_view" in observations:
            global_obs = observations["global_view"]
            # Ensure correct format: [batch, channels, height, width]
            if global_obs.dim() == 4:
                # If [batch, H, W, C], permute to [batch, C, H, W]
                if global_obs.shape[-1] <= 3:  # channels is last dimension
                    global_obs = global_obs.permute(0, 3, 1, 2)
            elif global_obs.dim() == 3:
                # [batch, H, W] -> [batch, 1, H, W]
                global_obs = global_obs.unsqueeze(1)
            global_features = self.global_cnn(global_obs.float() / 255.0)
            features.append(global_features)

        # Process graph
        # Graph observations come as direct keys in observations (from nclone environment)
        if self.graph_encoder is not None and "graph_node_feats" in observations:
            # Handle different encoder types - some take dict, some take separate args
            from npp_rl.models.hgt_encoder import HGTEncoder

            if isinstance(self.graph_encoder, HGTEncoder):
                # HGTEncoder expects a dict with graph_* prefix keys
                # nclone environment already provides these keys directly
                hgt_graph_obs = {
                    "graph_node_feats": observations["graph_node_feats"].float(),
                    "graph_edge_index": observations[
                        "graph_edge_index"
                    ].long(),  # Convert for indexing
                    "graph_edge_feats": observations["graph_edge_feats"].float(),
                    "graph_node_mask": observations["graph_node_mask"],
                    "graph_edge_mask": observations["graph_edge_mask"],
                }
                # Add type info if available
                if "graph_node_types" in observations:
                    hgt_graph_obs["graph_node_types"] = observations["graph_node_types"]
                if "graph_edge_types" in observations:
                    hgt_graph_obs["graph_edge_types"] = observations["graph_edge_types"]

                graph_features = self.graph_encoder(hgt_graph_obs)
            elif isinstance(self.graph_encoder, SimplifiedHGTEncoder):
                # SimplifiedHGTEncoder takes separate arguments
                node_features = observations["graph_node_feats"].float()
                edge_index = observations[
                    "graph_edge_index"
                ].long()  # Convert for indexing
                node_mask = observations["graph_node_mask"]
                node_types = observations.get(
                    "graph_node_types",
                    torch.zeros(
                        node_features.shape[:2],
                        dtype=torch.long,
                        device=node_features.device,
                    ),
                )
                _, graph_features = self.graph_encoder(
                    node_features, edge_index, node_types, node_mask
                )
            else:
                # GAT and GCN take separate arguments (no type information)
                node_features = observations["graph_node_feats"].float()
                edge_index = observations[
                    "graph_edge_index"
                ].long()  # Convert for indexing
                node_mask = observations["graph_node_mask"]
                _, graph_features = self.graph_encoder(
                    node_features, edge_index, node_mask
                )

            features.append(graph_features)

        # Process game state
        if self.state_mlp is not None and "game_state" in observations:
            state_features = self.state_mlp(observations["game_state"].float())
            features.append(state_features)

        # Process reachability
        if (
            self.reachability_mlp is not None
            and "reachability_features" in observations
        ):
            reach_features = self.reachability_mlp(
                observations["reachability_features"].float()
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
