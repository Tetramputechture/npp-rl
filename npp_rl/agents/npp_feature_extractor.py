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

        # Input channels configuration matching observation processor
        self.recent_frames = 4  # Current + 3 recent frames
        self.historical_frames = 3  # Historical frames at fixed intervals
        self.memory_channels = 4  # Spatial memory features
        self.player_state_channels = 6  # Player state features
        self.goal_channels = 2  # Switch and exit door heatmaps

        # Enhanced CNN architecture for frame processing
        self.frame_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        # Calculate the flattened size: 128 * 6 * 6 = 4608
        flat_size = 128 * 6 * 6

        # Frame processing for both recent and historical frames
        self.frame_processor = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Memory processing
        self.memory_processor = nn.Sequential(
            nn.Conv2d(self.memory_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Player state processing
        self.player_state_processor = nn.Sequential(
            nn.Conv2d(self.player_state_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Goal processing
        self.goal_processor = nn.Sequential(
            nn.Conv2d(self.goal_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # Final integration
        # Recent frames + historical frames + memory + player state + goal features
        total_features = 128 + 128 + 64 + 64 + 64
        self.integration = nn.Sequential(
            nn.Linear(total_features, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass for the feature extractor."""
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)

        # Split channels according to observation processor's concatenation order
        start_idx = 0

        # Recent frames: first 4 channels
        recent_frames = observations[:,
                                     start_idx:start_idx + self.recent_frames]
        start_idx += self.recent_frames

        # Historical frames: next 3 channels
        historical_frames = observations[:,
                                         start_idx:start_idx + self.historical_frames]
        start_idx += self.historical_frames

        # Memory features: next 4 channels
        memory_features = observations[:,
                                       start_idx:start_idx + self.memory_channels]
        start_idx += self.memory_channels

        # Player state: next 6 channels
        player_state_features = observations[:,
                                             start_idx:start_idx + self.player_state_channels]
        start_idx += self.player_state_channels

        # Goal features: final 2 channels (switch and exit)
        goal_features = observations[:,
                                     start_idx:start_idx + self.goal_channels]

        # Process recent frames (process each frame independently)
        recent_frame_features = []
        for i in range(self.recent_frames):
            frame = recent_frames[:, i:i+1]
            features = self.frame_features(frame)
            recent_frame_features.append(self.frame_processor(features))
        recent_frame_features = torch.mean(
            torch.stack(recent_frame_features, dim=1), dim=1)

        # Process historical frames (process each frame independently)
        historical_frame_features = []
        for i in range(self.historical_frames):
            frame = historical_frames[:, i:i+1]
            features = self.frame_features(frame)
            historical_frame_features.append(self.frame_processor(features))
        historical_frame_features = torch.mean(
            torch.stack(historical_frame_features, dim=1), dim=1)

        # Process other features
        memory_features = self.memory_processor(memory_features)
        player_state_features = self.player_state_processor(
            player_state_features)
        goal_features = self.goal_processor(goal_features)

        # Combine all features
        combined = torch.cat([
            recent_frame_features,
            historical_frame_features,
            memory_features,
            player_state_features,
            goal_features
        ], dim=1)

        return self.integration(combined)
