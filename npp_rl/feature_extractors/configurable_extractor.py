"""
Configurable multimodal feature extractor for architecture comparison.

This extractor allows enabling/disabling different modalities (player frames,
global view, graph, state, reachability) to test which features are necessary
for effective N++ learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import logging

# Import nclone constants for observation space dimensions
from nclone.graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from nclone.gym_environment.constants import (
    REACHABILITY_FEATURES_DIM,
    GAME_STATE_CHANNELS,
)

from npp_rl.training.architecture_configs import (
    ArchitectureConfig,
    GraphArchitectureType,
    FusionType,
)
from npp_rl.models.gcn import GCNEncoder
from npp_rl.models.gat import GATEncoder
from npp_rl.models.simplified_hgt import SimplifiedHGTEncoder
from npp_rl.models.hgt_factory import create_hgt_encoder


logger = logging.getLogger(__name__)


# Constants for CNN architecture
PLAYER_FRAME_SPATIAL_SIZE = 7  # Spatial dimensions after conv layers (84x84 -> 7x7)
GLOBAL_VIEW_SPATIAL_SIZE = 4  # Spatial dimensions after adaptive pooling
CNN_DROPOUT = 0.1  # Dropout rate for convolutional layers
MLP_DROPOUT = 0.2  # Dropout rate for fully connected layers


class PlayerFrameCNN(nn.Module):
    """CNN for player frame processing with dynamic frame stacking support.

    Uses eager initialization to ensure layers exist when loading pretrained weights.
    Supports dynamic rebuilding if input channels change (e.g., frame stacking configuration).
    """

    def __init__(
        self, visual_config, default_in_channels: int = 1, debug_mode: bool = False
    ):
        super().__init__()
        self.visual_config = visual_config
        self.current_in_channels = default_in_channels
        self.debug_mode = debug_mode
        # Build layers immediately during initialization
        self._build_layers(default_in_channels)

        # Log initialization for debugging frame stacking
        if default_in_channels > 1:
            logger.info(
                f"PlayerFrameCNN initialized with {default_in_channels} input channels (frame stacking enabled)"
            )

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
            nn.Dropout2d(CNN_DROPOUT),
            nn.Conv2d(
                self.visual_config.player_frame_channels[0],
                self.visual_config.player_frame_channels[1],
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(self.visual_config.player_frame_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT),
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
        # 84x84 input is reduced to PLAYER_FRAME_SPATIAL_SIZE x PLAYER_FRAME_SPATIAL_SIZE
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_config.player_frame_channels[2]
                * PLAYER_FRAME_SPATIAL_SIZE
                * PLAYER_FRAME_SPATIAL_SIZE,
                self.visual_config.player_frame_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
        )
        self.current_in_channels = in_channels

    def forward(self, x):
        """Forward pass with dynamic input channel handling.

        Handles both single frames and stacked frames with automatic channel reordering.
        Input is expected to be in [0, 1] range (normalized by caller).

        Supported input shapes:
        - Single frame: [batch, H, W, 1] (NHWC format from environment)
        - Stacked frames: [batch, stack_size, H, W, 1] (environment frame stack format)

        Internal conversion to PyTorch NCHW format:
        - Single: [batch, H, W, 1] -> [batch, 1, H, W]
        - Stacked: [batch, stack_size, H, W, 1] -> [batch, stack_size, H, W]

        Args:
            x: Input tensor in [0, 1] range

        Returns:
            Extracted features [batch, player_frame_output_dim]
        """
        # Handle different input shapes from environment
        if x.dim() == 4:
            # Single frame: [batch, H, W, 1] -> [batch, 1, H, W]
            if x.shape[-1] == 1:
                x = x.permute(0, 3, 1, 2).contiguous()
            # else: Already in NCHW format [batch, C, H, W]
        elif x.dim() == 5:
            # Stacked frames: [batch, stack_size, H, W, C] -> [batch, stack_size*C, H, W]
            # Treat stack dimension as additional input channels for temporal learning
            batch_size, stack_size, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            x = x.view(batch_size, stack_size * C, H, W)

        # Validate input channels match expected (strict mode for pretrained weights)
        in_channels = x.shape[1]
        if in_channels != self.current_in_channels:
            # Check if this layer has been trained (non-default weights)
            has_trained_weights = any(
                p.requires_grad and not torch.all(p == 0).item()
                for p in self.parameters()
                if p.numel() > 0
            )

            if has_trained_weights:
                raise ValueError(
                    f"Input channel mismatch detected! "
                    f"Expected {self.current_in_channels} channels but got {in_channels}. "
                    f"This indicates a frame stacking configuration mismatch between "
                    f"BC pretraining and RL training. All pretrained weights would be lost. "
                    f"Fix: Ensure frame_stack_config matches between BC and RL training."
                )

            # Only rebuild if no trained weights exist (random initialization)
            logger.debug(
                f"PlayerFrameCNN: Rebuilding for {in_channels} channels (was {self.current_in_channels})"
            )
            self._build_layers(in_channels)
            self.conv_layers = self.conv_layers.to(x.device)
            self.fc = self.fc.to(x.device)

        x = self.conv_layers(x)
        x = self.fc(x)

        return x


class GlobalViewCNN(nn.Module):
    """CNN for global view processing (single frame only, not stacked).

    Uses eager initialization to ensure layers exist when loading pretrained weights.
    """

    def __init__(
        self, visual_config, default_in_channels: int = 1, debug_mode: bool = False
    ):
        super().__init__()
        self.visual_config = visual_config
        self.current_in_channels = default_in_channels
        self.debug_mode = debug_mode

        # Build layers immediately during initialization
        self._build_layers(default_in_channels)

    def _build_layers(self, in_channels: int):
        """Build convolutional layers based on input channel count."""
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.visual_config.global_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT),
            nn.Conv2d(
                self.visual_config.global_channels[0],
                self.visual_config.global_channels[1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout2d(CNN_DROPOUT),
            nn.Conv2d(
                self.visual_config.global_channels[1],
                self.visual_config.global_channels[2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.visual_config.global_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((GLOBAL_VIEW_SPATIAL_SIZE, GLOBAL_VIEW_SPATIAL_SIZE)),
        )
        # Fully connected layers
        # AdaptiveAvgPool2d ensures output is always GLOBAL_VIEW_SPATIAL_SIZE regardless of input size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_config.global_channels[2]
                * GLOBAL_VIEW_SPATIAL_SIZE
                * GLOBAL_VIEW_SPATIAL_SIZE,
                self.visual_config.global_output_dim,
            ),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
        )
        self.current_in_channels = in_channels

    def forward(self, x):
        """Forward pass for single-frame global view (NOT stacked).

        Global view is never stacked - it provides spatial context at each timestep.
        Input is expected to be in [0, 1] range (normalized by caller).

        Supported input shapes:
        - [batch, H, W, 1] (NHWC format from environment)
        - [batch, H, W] (HW format)

        Internal conversion to PyTorch NCHW format:
        - [batch, H, W, 1] -> [batch, 1, H, W]
        - [batch, H, W] -> [batch, 1, H, W]

        Args:
            x: Input tensor in [0, 1] range

        Returns:
            Extracted features [batch, global_output_dim]
        """
        # Handle single frame input (global view is never stacked)
        if x.dim() == 4 and x.shape[-1] == 1:
            # [batch, H, W, 1] -> [batch, 1, H, W]
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3:
            # [batch, H, W] -> [batch, 1, H, W]
            x = x.unsqueeze(1)

        # Validate input channels (global view should always be 1 channel)
        in_channels = x.shape[1]
        if in_channels != self.current_in_channels:
            logger.debug(
                f"GlobalViewCNN: Rebuilding for {in_channels} channels (was {self.current_in_channels})"
            )
            self._build_layers(in_channels)
            self.conv_layers = self.conv_layers.to(x.device)
            self.fc = self.fc.to(x.device)

        x = self.conv_layers(x)

        x = self.fc(x)
        if self.debug_mode and torch.isnan(x).any():
            nan_mask = torch.isnan(x)
            batch_indices = (
                torch.where(nan_mask.any(dim=1))[0]
                if x.dim() > 1
                else torch.tensor([0])
            )
            raise ValueError(
                f"[GlobalViewCNN] NaN after fc in batch indices: {batch_indices.tolist()}"
            )

        return x


class StateMLP(nn.Module):
    """MLP for game state processing with dynamic state stacking support.

    Uses eager initialization to ensure layers exist when loading pretrained weights.
    Supports dynamic rebuilding if input dimensions change (e.g., state stacking configuration).
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        default_input_dim: Optional[int] = None,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.base_input_dim = GAME_STATE_CHANNELS
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.debug_mode = debug_mode

        self.current_input_dim = GAME_STATE_CHANNELS

        # Build layers immediately during initialization
        self._build_layers()

    def _build_layers(self):
        """Build MLP layers based on actual input dimension (may be stacked)."""
        self.mlp = nn.Sequential(
            nn.Linear(GAME_STATE_CHANNELS, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(CNN_DROPOUT),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )
        self.current_input_dim = GAME_STATE_CHANNELS

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

        # Validate input dimensions match expected
        if x.shape[-1] != GAME_STATE_CHANNELS:
            raise ValueError(
                f"State input dimension mismatch detected! "
                f"Expected {GAME_STATE_CHANNELS} but got {x.shape[-1]}. "
                f"This indicates a state stacking configuration mismatch between "
                f"BC pretraining and RL training. All pretrained weights would be lost. "
                f"Fix: Ensure frame_stack_config matches between BC and RL training."
            )

        return self.mlp(x)


class ConfigurableMultimodalExtractor(BaseFeaturesExtractor):
    """
    Configurable multimodal feature extractor for Task 3.1 experiments.

    Allows selective enabling/disabling of modalities:
    - Player frame (2D CNN on 84x84x1 grayscale)
    - Global view (2D CNN on 176x100x1 grayscale)
    - Graph (HGT/GAT/GCN/None) with NODE_FEATURE_DIM=21, EDGE_FEATURE_DIM=14
    - Game state (NINJA_STATE_DIM=41, enhanced ninja physics + time_remaining)
    - Reachability (REACHABILITY_FEATURES_DIM=7, base features + mine context + phase indicator)

    This enables systematic comparison of architecture variants.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        config: ArchitectureConfig,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        debug_mode: bool = False,
    ):
        """
        Args:
            observation_space: Gymnasium Dict space with observation components
            config: ArchitectureConfig specifying which modalities to use
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str ('zero' or 'repeat')
            debug_mode: Enable expensive NaN validation checks (default: False for performance)
        """
        # Initialize with final features dimension
        super().__init__(observation_space, config.features_dim)

        self.config = config
        self.modalities = config.modalities
        self.frame_stack_config = frame_stack_config or {}
        self.debug_mode = debug_mode

        # Log frame stacking configuration for debugging
        if self.frame_stack_config:
            visual_enabled = self.frame_stack_config.get(
                "enable_visual_frame_stacking", False
            )
            state_enabled = self.frame_stack_config.get("enable_state_stacking", False)
            if visual_enabled or state_enabled:
                logger.info(
                    "ConfigurableMultimodalExtractor initialized with frame stacking:"
                )
                if visual_enabled:
                    logger.info(
                        f"  Visual: {self.frame_stack_config.get('visual_stack_size', 4)} frames"
                    )
                if state_enabled:
                    logger.info(
                        f"  State: {self.frame_stack_config.get('state_stack_size', 4)} states"
                    )

        # Track which modalities are actually available
        self.has_player_frame = "player_frame" in observation_space.spaces
        self.has_global = "global_view" in observation_space.spaces
        # Graph observations use dense format (simplified - no sparse logic)
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
        if self.modalities.use_player_frame and self.has_player_frame:
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
                config.state.hidden_dim,
                config.state.output_dim,
            )
            feature_dims.append(config.state.output_dim)

        # 5. Reachability features processing
        if self.modalities.use_reachability and self.has_reachability:
            reachability_output_dim = config.state.output_dim // 2
            self.reachability_mlp = self._create_reachability_mlp(
                config.state.hidden_dim // 2,
                reachability_output_dim,
            )
            feature_dims.append(reachability_output_dim)
            # Store for fusion
            self._reachability_output_dim = reachability_output_dim
        else:
            self._reachability_output_dim = None

        # Create fusion mechanism
        total_input_dim = sum(feature_dims)
        # Store actual feature dimensions for fusion (in case they differ from config)
        self._actual_feature_dims = feature_dims.copy()
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
        # Calculate initial input channels from frame_stack_config
        default_in_channels = 1
        if self.frame_stack_config.get("enable_visual_frame_stacking", False):
            default_in_channels = self.frame_stack_config.get("visual_stack_size", 4)

        return PlayerFrameCNN(
            visual_config,
            default_in_channels=default_in_channels,
            debug_mode=self.debug_mode,
        )

    def _create_global_cnn(self, visual_config) -> nn.Module:
        """Create 2D CNN for global view processing (single frame only, not stacked)."""
        default_in_channels = 1
        return GlobalViewCNN(
            visual_config,
            default_in_channels=default_in_channels,
            debug_mode=self.debug_mode,
        )

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

    def _create_state_mlp(self, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create simple MLP for game state processing."""
        # Determine input dimension (handle frame stacking)
        default_input_dim = GAME_STATE_CHANNELS
        if self.frame_stack_config.get("enable_state_stacking", False):
            stack_size = self.frame_stack_config.get("state_stack_size", 4)
            default_input_dim = GAME_STATE_CHANNELS * stack_size

        return StateMLP(
            hidden_dim,
            output_dim,
            default_input_dim=default_input_dim,
            debug_mode=self.debug_mode,
        )

    def _create_reachability_mlp(self, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create MLP for reachability feature processing."""
        return nn.Sequential(
            nn.Linear(REACHABILITY_FEATURES_DIM, hidden_dim),
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
            # Use actual feature dimensions from initialization (more reliable than recalculating)
            if hasattr(self, "_actual_feature_dims") and self._actual_feature_dims:
                modality_dims = self._actual_feature_dims
            else:
                # Fallback: Calculate modality dimensions based on enabled modalities
                modality_dims = []

                if self.modalities.use_player_frame:
                    modality_dims.append(
                        self.config.visual.player_frame_output_dim
                    )  # 256
                if self.modalities.use_global_view:
                    modality_dims.append(self.config.visual.global_output_dim)  # 128
                if self.modalities.use_graph:
                    modality_dims.append(self.config.graph.output_dim)  # 256
                if self.modalities.use_game_state:
                    modality_dims.append(self.config.state.output_dim)  # 128
                if self.modalities.use_reachability:
                    # Use stored reachability output dimension
                    if (
                        hasattr(self, "_reachability_output_dim")
                        and self._reachability_output_dim is not None
                    ):
                        modality_dims.append(self._reachability_output_dim)
                    else:
                        # Fallback: use state.output_dim // 2
                        modality_dims.append(self.config.state.output_dim // 2)

            # Verify sum matches input_dim
            total = sum(modality_dims)
            assert total == input_dim, (
                f"Modality dims {modality_dims} sum to {total}, expected {input_dim}. "
                f"Actual feature dims: {getattr(self, '_actual_feature_dims', 'N/A')}"
            )

            return MultiHeadFusion(
                input_dim,
                output_dim,
                num_heads=fusion_config.num_attention_heads,
                modality_dims=modality_dims,  # Pass modality dimensions
                dropout=fusion_config.dropout,
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

        # Process player frame (can be single frame or stacked frames)
        if self.player_frame_cnn is not None and "player_frame" in observations:
            player_frame_obs = observations["player_frame"]

            # Normalize to [0, 1] range before passing to CNN
            # Note: PlayerFrameCNN.forward() handles channel reordering internally
            player_frame_features = self.player_frame_cnn(
                player_frame_obs.float() / 255.0
            )
            if self.debug_mode and torch.isnan(player_frame_features).any():
                nan_mask = torch.isnan(player_frame_features)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[EXTRACTOR] NaN in player_frame features in batch indices: {batch_indices.tolist()}"
                )
            features.append(player_frame_features)

        # Process global view (single frame only, not stacked)
        if self.global_cnn is not None and "global_view" in observations:
            global_obs = observations["global_view"]
            # Global view should always be a single frame: [batch, H, W, C] or [batch, H, W]
            # The GlobalViewCNN will handle the channel dimension conversion
            global_features = self.global_cnn(global_obs.float() / 255.0)
            if self.debug_mode and torch.isnan(global_features).any():
                nan_mask = torch.isnan(global_features)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[EXTRACTOR] NaN in global_view features in batch indices: {batch_indices.tolist()}"
                )
            features.append(global_features)

        # Process graph (dense format only - simplified)
        # Graph observations: node_features (7D), edge_index, masks
        # No edge features or types - all edges are simple adjacency
        has_dense_graph = "graph_node_feats" in observations

        if self.graph_encoder is not None and has_dense_graph:
            logger.debug("[EXTRACTOR] Processing graph observations (dense format)...")

            # Handle different encoder types
            # HGT encoders require edge features and node/edge types which are not provided
            # in GCN-optimized observation space
            from npp_rl.models.hgt_encoder import HGTEncoder

            if isinstance(self.graph_encoder, HGTEncoder):
                raise ValueError(
                    "[EXTRACTOR] HGTEncoder not supported with GCN-optimized observations. "
                    "Observation space no longer includes graph_edge_feats, graph_node_types, "
                    "or graph_edge_types. Use GraphArchitectureType.GCN instead."
                )
            elif isinstance(self.graph_encoder, SimplifiedHGTEncoder):
                raise ValueError(
                    "[EXTRACTOR] SimplifiedHGTEncoder not supported with GCN-optimized observations. "
                    "Observation space no longer includes graph_node_types. "
                    "Use GraphArchitectureType.GCN instead."
                )
            else:
                # GAT and GCN take separate arguments (GCN-optimized)
                logger.debug("[EXTRACTOR] Using GAT/GCN encoder...")
                node_features = observations[
                    "graph_node_feats"
                ].float()  # 7 dims (was 17)
                edge_index = observations[
                    "graph_edge_index"
                ].long()  # Convert for indexing
                # Masks are now computed in sparse-to-dense conversion (not stored in obs)
                node_mask = observations["graph_node_mask"]
                edge_mask = observations.get("graph_edge_mask")

                # Log graph observation shapes for debugging
                if edge_mask is not None:
                    logger.debug(f"[EXTRACTOR] Edge mask shape: {edge_mask.shape}")

                # Validate edge indices before passing to encoder
                batch_size, max_nodes, _ = node_features.shape
                if edge_index.shape[0] != batch_size:
                    raise ValueError(
                        f"[EXTRACTOR] Edge index batch size mismatch: "
                        f"expected {batch_size}, got {edge_index.shape[0]}"
                    )

                # Check for invalid edge indices
                for b in range(batch_size):
                    edges = edge_index[b]  # [2, num_edges]
                    if edges.shape[1] > 0:
                        src_nodes = edges[0]
                        tgt_nodes = edges[1]
                        if (src_nodes >= max_nodes).any() or (src_nodes < 0).any():
                            invalid = torch.where(
                                (src_nodes >= max_nodes) | (src_nodes < 0)
                            )[0]
                            raise ValueError(
                                f"[EXTRACTOR] Invalid source node indices in batch {b}: "
                                f"max_nodes={max_nodes}, invalid_count={len(invalid)}, "
                                f"invalid_indices={src_nodes[invalid[:5]].tolist()}"
                            )
                        if (tgt_nodes >= max_nodes).any() or (tgt_nodes < 0).any():
                            invalid = torch.where(
                                (tgt_nodes >= max_nodes) | (tgt_nodes < 0)
                            )[0]
                            raise ValueError(
                                f"[EXTRACTOR] Invalid target node indices in batch {b}: "
                                f"max_nodes={max_nodes}, invalid_count={len(invalid)}, "
                                f"invalid_indices={tgt_nodes[invalid[:5]].tolist()}"
                            )

                logger.debug("[EXTRACTOR] Calling graph encoder forward pass...")
                _, graph_features = self.graph_encoder(
                    node_features, edge_index, node_mask, edge_mask
                )
                logger.debug(
                    f"[EXTRACTOR] Graph encoder completed, output shape: {graph_features.shape}"
                )

            if self.debug_mode and torch.isnan(graph_features).any():
                nan_mask = torch.isnan(graph_features)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[EXTRACTOR] NaN in graph features in batch indices: {batch_indices.tolist()}"
                )

            # Diagnostic logging for graph feature quality (Stage 2 debugging)
            if self.debug_mode:
                # Log graph size statistics
                num_valid_nodes = node_mask.sum(dim=1)
                num_valid_edges = (
                    edge_mask.sum(dim=1) if edge_mask is not None else torch.tensor([0])
                )

                logger.debug(
                    f"[GRAPH_DIAG] Graph nodes per sample: min={num_valid_nodes.min().item():.0f}, "
                    f"max={num_valid_nodes.max().item():.0f}, mean={num_valid_nodes.float().mean().item():.1f}"
                )
                logger.debug(
                    f"[GRAPH_DIAG] Graph edges per sample: min={num_valid_edges.min().item():.0f}, "
                    f"max={num_valid_edges.max().item():.0f}, mean={num_valid_edges.float().mean().item():.1f}"
                )

                # Log raw node feature statistics (before GNN processing)
                valid_node_feats = node_features[node_mask.bool()]
                if valid_node_feats.numel() > 0:
                    logger.debug(
                        f"[GRAPH_DIAG] Raw node features: mean={valid_node_feats.mean().item():.3f}, "
                        f"std={valid_node_feats.std().item():.3f}, "
                        f"min={valid_node_feats.min().item():.3f}, max={valid_node_feats.max().item():.3f}"
                    )

                    # Check for degenerate features (all zeros or constants)
                    feature_variance = valid_node_feats.var(dim=0)
                    num_zero_variance = (feature_variance < 1e-6).sum().item()
                    if num_zero_variance > 0:
                        logger.warning(
                            f"[GRAPH_DIAG] {num_zero_variance}/{valid_node_feats.shape[1]} node features have near-zero variance!"
                        )

                # Log processed graph features (after GNN)
                logger.debug(
                    f"[GRAPH_DIAG] Processed graph features: mean={graph_features.mean().item():.3f}, "
                    f"std={graph_features.std().item():.3f}, "
                    f"min={graph_features.min().item():.3f}, max={graph_features.max().item():.3f}"
                )

            features.append(graph_features)

        # Process game state (already includes time_remaining as 41st feature)
        if self.state_mlp is not None and "game_state" in observations:
            game_state = observations["game_state"].float()

            state_features = self.state_mlp(game_state)
            if self.debug_mode and torch.isnan(state_features).any():
                nan_mask = torch.isnan(state_features)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[EXTRACTOR] NaN in state features in batch indices: {batch_indices.tolist()}"
                )
            features.append(state_features)

        # Process reachability
        if (
            self.reachability_mlp is not None
            and "reachability_features" in observations
        ):
            reach_features = self.reachability_mlp(
                observations["reachability_features"].float()
            )
            if self.debug_mode and torch.isnan(reach_features).any():
                nan_mask = torch.isnan(reach_features)
                batch_indices = torch.where(nan_mask.any(dim=1))[0]
                raise ValueError(
                    f"[EXTRACTOR] NaN in reachability features in batch indices: {batch_indices.tolist()}"
                )
            features.append(reach_features)

        # Concatenate all features
        if len(features) == 0:
            raise ValueError(
                "No features were extracted! Check configuration and observation space."
            )

        combined_features = torch.cat(features, dim=1)
        if self.debug_mode and torch.isnan(combined_features).any():
            nan_mask = torch.isnan(combined_features)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[EXTRACTOR] NaN after concatenation in batch indices: {batch_indices.tolist()}"
            )

        # Apply fusion
        output = self.fusion(combined_features)
        if self.debug_mode and torch.isnan(output).any():
            nan_mask = torch.isnan(output)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[EXTRACTOR] NaN after fusion in batch indices: {batch_indices.tolist()}"
            )

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

        # Attention
        attn_out, _ = self.attention(x, x, x)

        # Norm with residual
        x = self.norm(x + attn_out)

        x = x.squeeze(1)

        # MLP
        output = self.mlp(x)

        return output


class MultiHeadFusion(nn.Module):
    """Enhanced multi-head attention fusion with true cross-modal reasoning."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 8,
        modality_dims: Optional[list] = None,  # NEW: track modality dimensions
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
               Expected to be concatenation of:
               - player_frame_features
               - global_view_features
               - graph_features
               - state_features
               - reachability_features

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
        # self.modality_embeddings: [num_modalities, uniform_dim]
        # Broadcast to batch: [1, num_modalities, uniform_dim]
        modality_tokens = modality_tokens + self.modality_embeddings.unsqueeze(0)

        # Multi-head cross-modal attention
        # Each modality can attend to all others
        attn_out, attn_weights = self.attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            need_weights=False,  # Set True for visualization
        )
        # attn_out: [batch, num_modalities, uniform_dim]

        # Residual + norm
        modality_tokens = self.norm1(modality_tokens + attn_out)

        # CHECK 8: After first residual + norm
        if self.debug_mode and torch.isnan(modality_tokens).any():
            nan_mask = torch.isnan(modality_tokens)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ENHANCED_FUSION] NaN after norm1 in batch indices: {batch_indices.tolist()}"
            )

        # Feed-forward network (applied to each token)
        ffn_out = self.ffn(modality_tokens)

        # CHECK 9: After FFN
        if self.debug_mode and torch.isnan(ffn_out).any():
            nan_mask = torch.isnan(ffn_out)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ENHANCED_FUSION] NaN after ffn in batch indices: {batch_indices.tolist()}"
            )

        modality_tokens = self.norm2(modality_tokens + ffn_out)
        # Shape: [batch, num_modalities, uniform_dim]

        # CHECK 10: After second residual + norm
        if self.debug_mode and torch.isnan(modality_tokens).any():
            nan_mask = torch.isnan(modality_tokens)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ENHANCED_FUSION] NaN after norm2 in batch indices: {batch_indices.tolist()}"
            )

        # Pool across modalities (mean pooling)
        # Could also use: weighted pooling, CLS token, or attention pooling
        fused = modality_tokens.mean(dim=1)  # [batch, uniform_dim]

        # CHECK 11: After pooling
        if self.debug_mode and torch.isnan(fused).any():
            nan_mask = torch.isnan(fused)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ENHANCED_FUSION] NaN after pooling in batch indices: {batch_indices.tolist()}"
            )

        # Project to output dimension
        output = self.output_proj(fused)  # [batch, output_dim]

        # CHECK 12: After output projection
        if self.debug_mode and torch.isnan(output).any():
            nan_mask = torch.isnan(output)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ENHANCED_FUSION] NaN after output_proj in batch indices: {batch_indices.tolist()}"
            )

        return output
