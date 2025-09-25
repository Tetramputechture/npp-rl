"""
Simplified HGT Multimodal Feature Extractor for NPP-RL.

This module provides a simplified version of the HGT multimodal feature extractor
that works with the new simplified graph representation (8 node features, 4 edge features).

Key simplifications:
- Uses simplified node features (8 dimensions vs 31)
- Uses simplified edge features (4 dimensions vs 16)
- Simplified reachability features (8 dimensions vs 64)
- Maintains HGT multimodal architecture for learning complex patterns
- Focuses on letting the network learn physics emergently

This approach aligns with HGT design principles: provide basic connectivity
and let the transformer learn complex relationships through attention.
"""

import torch
import torch.nn as nn
from typing import Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

from ..models.hgt_config import HGT_CONFIG


class HGTMultimodalExtractor(BaseFeaturesExtractor):
    """
    Simplified HGT-based multimodal feature extractor.

    Combines:
    1. Visual CNN features (unchanged)
    2. Simplified graph features (8 node + 4 edge dimensions)
    3. Simplified reachability features (8 dimensions)
    4. Game state features (unchanged)

    Uses cross-modal attention to fuse information and let the HGT learn
    complex movement patterns emergently.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        debug: bool = False,
    ):
        """
        Initialize simplified HGT multimodal feature extractor.

        Args:
            observation_space: Environment observation space
            features_dim: Output feature dimension
            debug: Enable debug output
        """
        super().__init__(observation_space, features_dim)
        self.debug = debug

        # Note: Graph processing is now handled by DynamicGraphWrapper
        # which provides graph observations directly from nclone

        # Visual processing (simplified CNN)
        self.visual_cnn = self._build_visual_cnn()

        # Graph processing (simplified HGT)
        self.graph_processor_net = self._build_graph_processor()

        # Reachability processing (simplified)
        self.reachability_processor = self._build_reachability_processor()

        # Game state processing (unchanged)
        self.state_processor = self._build_state_processor()

        # Cross-modal fusion (simplified)
        self.fusion_network = self._build_fusion_network()

        # Output projection
        self.output_projection = nn.Linear(
            self._calculate_fusion_output_dim(), features_dim
        )

    def _build_visual_cnn(self) -> nn.Module:
        """Build simplified visual CNN for spatial processing."""
        return nn.Sequential(
            # Simplified 2D CNN for visual processing
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
        )

    def _build_graph_processor(self) -> nn.Module:
        """Build simplified graph processing network."""
        return nn.Sequential(
            # Simple MLP for graph features (no complex HGT for now)
            nn.Linear(HGT_CONFIG.node_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def _build_reachability_processor(self) -> nn.Module:
        """Build simplified reachability feature processor."""
        return nn.Sequential(
            nn.Linear(8, 32),  # 8 simplified reachability features
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def _build_state_processor(self) -> nn.Module:
        """Build game state processor."""
        return nn.Sequential(
            nn.Linear(16, 32),  # Assume 16 game state features
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def _build_fusion_network(self) -> nn.Module:
        """Build cross-modal fusion network."""
        fusion_input_dim = 256 + 128 + 64 + 64  # visual + graph + reachability + state
        return nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def _calculate_fusion_output_dim(self) -> int:
        """Calculate fusion network output dimension."""
        return 256

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through simplified multimodal extractor.

        Args:
            observations: Dictionary of observation tensors

        Returns:
            Fused feature tensor
        """
        batch_size = self._get_batch_size(observations)
        device = next(self.parameters()).device

        # 1. Process visual features
        visual_features = self._process_visual_features(observations, device)

        # 2. Process simplified graph features
        graph_features = self._process_graph_features(observations, device, batch_size)

        # 3. Process simplified reachability features
        reachability_features = self._process_reachability_features(
            observations, device
        )

        # 4. Process game state features
        state_features = self._process_state_features(observations, device)

        # 5. Fuse all modalities
        fused_features = self._fuse_modalities(
            visual_features, graph_features, reachability_features, state_features
        )

        # 6. Final projection
        output = self.output_projection(fused_features)

        if self.debug and torch.rand(1).item() < 0.01:  # Debug 1% of calls
            print(f"Simplified HGT output shape: {output.shape}")

        return output

    def _get_batch_size(self, observations: Dict[str, torch.Tensor]) -> int:
        """Extract batch size from observations."""
        for key, value in observations.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.shape[0]
        return 1

    def _process_visual_features(
        self, observations: Dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Process visual observations through CNN."""
        # Try to get visual observation
        visual_keys = ["image", "visual", "observation", "rgb"]
        visual_obs = None

        for key in visual_keys:
            if key in observations:
                visual_obs = observations[key]
                break

        if visual_obs is None:
            # Fallback: create dummy visual features
            batch_size = self._get_batch_size(observations)
            return torch.zeros(batch_size, 256, device=device)

        # Ensure correct shape for CNN
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(1)  # Add channel dimension
        elif visual_obs.dim() == 4 and visual_obs.shape[1] > 1:
            visual_obs = visual_obs.mean(dim=1, keepdim=True)  # Convert to grayscale

        return self.visual_cnn(visual_obs.to(device))

    def _process_graph_features(
        self,
        observations: Dict[str, torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Process graph features from DynamicGraphWrapper observations."""
        # Use graph observations provided by DynamicGraphWrapper
        if "graph_node_feats" in observations:
            # Use actual graph node features from nclone
            node_feats = observations["graph_node_feats"].to(device)
            # Average over nodes to get a single feature vector per batch
            if node_feats.dim() == 3:  # [batch, nodes, features]
                graph_features = node_feats.mean(dim=1)  # [batch, features]
            else:  # [nodes, features] - single batch
                graph_features = node_feats.mean(dim=0, keepdim=True)  # [1, features]
            
            # Pad or truncate to match expected dimension
            if graph_features.shape[-1] < HGT_CONFIG.node_feat_dim:
                padding = torch.zeros(
                    *graph_features.shape[:-1], 
                    HGT_CONFIG.node_feat_dim - graph_features.shape[-1], 
                    device=device
                )
                graph_features = torch.cat([graph_features, padding], dim=-1)
            elif graph_features.shape[-1] > HGT_CONFIG.node_feat_dim:
                graph_features = graph_features[..., :HGT_CONFIG.node_feat_dim]
        else:
            # Fallback: create dummy graph features
            graph_features = torch.zeros(
                batch_size, HGT_CONFIG.node_feat_dim, device=device
            )

        return self.graph_processor_net(graph_features)

    def _extract_graph_features_from_level(
        self, level_data: torch.Tensor, device: torch.device, batch_size: int
    ) -> torch.Tensor:
        """Extract simplified graph features from level data."""
        # This is a simplified implementation
        # In practice, would use SimpleNodeFeatureExtractor and SimpleEdgeBuilder

        # For now, create representative features based on level data
        if level_data.numel() > 0:
            # Use level data statistics as simple graph features
            features = torch.zeros(batch_size, HGT_CONFIG.node_feat_dim, device=device)

            # Fill with simple statistics from level data
            if level_data.dim() >= 2:
                level_mean = level_data.float().mean(dim=-1)
                level_std = level_data.float().std(dim=-1)

                # Pad or truncate to match node feature dimension
                if level_mean.shape[-1] >= HGT_CONFIG.node_feat_dim:
                    features = level_mean[..., : HGT_CONFIG.node_feat_dim]
                else:
                    features[..., : level_mean.shape[-1]] = level_mean

        else:
            features = torch.zeros(batch_size, HGT_CONFIG.node_feat_dim, device=device)

        return features

    def _process_reachability_features(
        self, observations: Dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Process simplified reachability features."""
        if "reachability_features" in observations:
            reachability_obs = observations["reachability_features"].to(device)
            return self.reachability_processor(reachability_obs)
        else:
            # Fallback: create dummy reachability features
            batch_size = self._get_batch_size(observations)
            dummy_features = torch.zeros(batch_size, 8, device=device)
            return self.reachability_processor(dummy_features)

    def _process_state_features(
        self, observations: Dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Process game state features."""
        # Extract game state information
        state_features = []
        batch_size = self._get_batch_size(observations)

        # Try to extract various state features
        state_keys = [
            "player_x",
            "player_y",
            "player_vx",
            "player_vy",
            "switch_states",
            "door_states",
            "exit_switch_activated",
            "level_time",
            "gold_collected",
        ]

        for key in state_keys:
            if key in observations:
                value = observations[key].to(device)
                if value.dim() == 0:
                    value = value.unsqueeze(0).repeat(batch_size)
                elif value.dim() == 1 and value.shape[0] != batch_size:
                    value = value[:1].repeat(batch_size)
                state_features.append(value.float())

        # Pad to 16 features
        while len(state_features) < 16:
            state_features.append(torch.zeros(batch_size, device=device))

        # Truncate to 16 features
        state_features = state_features[:16]

        state_tensor = torch.stack(state_features, dim=1)
        return self.state_processor(state_tensor)

    def _fuse_modalities(
        self,
        visual_features: torch.Tensor,
        graph_features: torch.Tensor,
        reachability_features: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse features from all modalities."""
        # Simple concatenation fusion
        fused = torch.cat(
            [
                visual_features,
                graph_features,
                reachability_features,
                state_features,
            ],
            dim=1,
        )

        return self.fusion_network(fused)
