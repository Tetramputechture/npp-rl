from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
import torchvision.models as models


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return nn.functional.relu(x)


class NPPFeatureExtractor(BaseFeaturesExtractor):
    """CNN-based feature extractor with residual connections and batch normalization,
    with separate branches for:
    - Recent frames (4 frames)
    - Historical frames (3 frames)
    - Spatial memory (4 channels: recent_visits, visit_frequency, area_exploration, transitions)
    - Player state (6 channels: pos_x, pos_y, vel_x, vel_y, in_air, walled)
    - Goals (2 channels: switch and exit heatmaps)
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Input channels configuration
        self.recent_frames = 4  # Current + 3 recent frames
        self.historical_frames = 3  # Historical frames at fixed intervals
        self.memory_channels = 4  # Spatial memory features
        # Player position, velocity, in_air, and walled status
        self.player_state_channels = 6
        self.goal_channels = 2  # Switch and exit door heatmaps

        # Recent frames processing branch with attention
        self.recent_frame_features = nn.Sequential(
            nn.Conv2d(self.recent_frames, 64,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),

            ResBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(),

            ResBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.Mish(),

            # Added spatial attention
            nn.Conv2d(256, 256, kernel_size=1),  # Changed to preserve channels
            nn.Sigmoid(),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        # Historical frames processing branch
        self.historical_frame_features = nn.Sequential(
            nn.Conv2d(self.historical_frames, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),

            ResBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        # Calculate correct flattened sizes
        recent_flat_size = 256 * 6 * 6  # 9216
        historical_flat_size = 64 * 6 * 6  # 2304

        # Frame processing networks with residual connections
        self.recent_frame_processor = nn.Sequential(
            nn.Linear(recent_flat_size, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish()
        )

        self.historical_frame_processor = nn.Sequential(
            nn.Linear(historical_flat_size, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish()
        )

        # Spatial memory processing branch with attention
        self.memory_processor = nn.Sequential(
            nn.Conv2d(self.memory_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),

            # Added attention mechanism
            nn.Conv2d(64, 64, kernel_size=1),  # Changed to preserve channels
            nn.Sigmoid(),

            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),

            nn.Linear(64 * 6 * 6, 128),  # Corrected input size
            nn.LayerNorm(128),
            nn.Mish()
        )

        # Player state processing branch
        self.player_state_processor = nn.Sequential(
            nn.Conv2d(self.player_state_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 64),  # Corrected input size
            nn.LayerNorm(64),
            nn.Mish()
        )

        # Goal processing branch with attention
        self.goal_processor = nn.Sequential(
            nn.Conv2d(self.goal_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Mish(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(),

            # Added attention
            nn.Conv2d(64, 64, kernel_size=1),  # Changed to preserve channels
            nn.Sigmoid(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 64),  # Corrected input size
            nn.LayerNorm(64),
            nn.Mish()
        )

        # Final integration with skip connections
        total_features = 256 + 128 + 128 + 64 + 64  # 640 total
        self.integration = nn.Sequential(
            nn.Linear(total_features, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(self.features_dim, self.features_dim),
            nn.LayerNorm(self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass for the feature extractor."""
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)

        # Split channels in the same order as observation_processor concatenates them
        start_idx = 0

        # Recent frames (4 channels)
        recent_frames = observations[...,
                                     start_idx:start_idx + self.recent_frames]
        start_idx += self.recent_frames

        # Historical frames (3 channels)
        historical_frames = observations[...,
                                         start_idx:start_idx + self.historical_frames]
        start_idx += self.historical_frames

        # Memory features (4 channels)
        memory_features = observations[...,
                                       start_idx:start_idx + self.memory_channels]
        start_idx += self.memory_channels

        # Player state features (6 channels)
        player_state_features = observations[...,
                                             start_idx:start_idx + self.player_state_channels]
        start_idx += self.player_state_channels

        # Goal features (2 channels)
        goal_features = observations[...,
                                     start_idx:start_idx + self.goal_channels]

        # Process each branch
        recent_frames = recent_frames.permute(0, 3, 1, 2)
        recent_frame_features = self.recent_frame_features(recent_frames)
        recent_frame_features = self.recent_frame_processor(
            recent_frame_features)

        historical_frames = historical_frames.permute(0, 3, 1, 2)
        historical_frame_features = self.historical_frame_features(
            historical_frames)
        historical_frame_features = self.historical_frame_processor(
            historical_frame_features)

        memory_features = memory_features.permute(0, 3, 1, 2)
        memory_features = self.memory_processor(memory_features)

        player_state_features = player_state_features.permute(0, 3, 1, 2)
        player_state_features = self.player_state_processor(
            player_state_features)

        goal_features = goal_features.permute(0, 3, 1, 2)
        goal_features = self.goal_processor(goal_features)

        # Combine all features with skip connections
        combined = torch.cat([
            recent_frame_features,
            historical_frame_features,
            memory_features,
            player_state_features,
            goal_features
        ], dim=1)

        return self.integration(combined)
