from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from npp_rl.environments.constants import (
    TEMPORAL_FRAMES,
    MEMORY_CHANNELS,
    PATHFINDING_CHANNELS,
    COLLISION_CHANNELS
)
torch.autograd.set_detect_anomaly(True)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        residual = x
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x * self.scale + residual
        return nn.functional.relu(x)


class NPPFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor with separate pathways for visual and numerical data.
    Visual data is processed with CNNs while numerical data uses MLPs.
    """

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Input channels configuration from constants
        self.temporal_frames = TEMPORAL_FRAMES
        self.memory_channels = MEMORY_CHANNELS
        self.pathfinding_channels = PATHFINDING_CHANNELS
        self.collision_channels = COLLISION_CHANNELS

        # Temporal frame processing branch - 3 layer CNN
        self.temporal_processor = nn.Sequential(
            nn.Conv2d(self.temporal_frames, 32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Memory features processing branch - 3 layer CNN
        self.memory_processor = nn.Sequential(
            nn.Conv2d(self.memory_channels, 16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Pathfinding features processing branch - 3 layer CNN
        self.pathfinding_processor = nn.Sequential(
            nn.Conv2d(self.pathfinding_channels, 32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Collision features processing branch - 3 layer CNN
        self.collision_processor = nn.Sequential(
            nn.Conv2d(self.collision_channels, 16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Update visual merger for the new channel dimensions
        self.visual_merger = nn.Sequential(
            nn.Conv2d(64 + 32 + 64 + 32, 128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # Recalculate visual features size
        self.visual_features_size = 128 * 4 * 4  # 2048

        # Player state MLP branch with smaller layers
        self.player_state_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Goal features MLP branch
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Visual features processor with smaller layers
        self.visual_features_mlp = nn.Sequential(
            nn.Linear(self.visual_features_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.ReLU()
        )

        # Final integration layer with gradient clipping
        combined_features = 192 + 64 + 32
        self.integration = nn.Sequential(
            nn.Linear(combined_features, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.features_dim, self.features_dim),
            nn.LayerNorm(self.features_dim)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Convert uint8 [0, 255] to float32 [0, 1] and add small epsilon to prevent division by zero
        visual_input = observations['visual'].float() / 255.0
        visual_input = visual_input.permute(0, 3, 1, 2).clamp(min=1e-6)

        temporal_frames = visual_input[:, :self.temporal_frames]
        memory_features = visual_input[:,
                                       self.temporal_frames:self.temporal_frames + self.memory_channels]
        pathfinding_features = visual_input[:, self.temporal_frames +
                                            self.memory_channels:self.temporal_frames + self.memory_channels + self.pathfinding_channels]
        collision_features = visual_input[:, -self.collision_channels:]

        # Process each visual component
        temporal_out = self.temporal_processor(temporal_frames)
        memory_out = self.memory_processor(memory_features)
        pathfinding_out = self.pathfinding_processor(pathfinding_features)
        collision_out = self.collision_processor(collision_features)

        # Merge visual features
        merged_visual = torch.cat([
            temporal_out,
            memory_out,
            pathfinding_out,
            collision_out
        ], dim=1)

        visual_features = self.visual_merger(merged_visual)
        visual_features = self.visual_features_mlp(visual_features)

        # Process player state and goal features
        player_features = self.player_state_mlp(
            observations['player_state'])
        goal_features = self.goal_mlp(observations['goal_features'])

        # Combine all features
        combined = torch.cat([
            visual_features,
            player_features,
            goal_features
        ], dim=1)

        # Final integration
        output = self.integration(combined)
        return output
