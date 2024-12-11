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
    """Enhanced feature extractor with residual connections and batch normalization"""

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Input channels configuration
        self.frame_channels = 4  # 1 current + 3 recent frames
        self.memory_channels = 4
        self.hazard_channel = 1  # Channel for mine hazards
        self.goal_channels = 2   # Switch and exit door channels

        # Enhanced CNN architecture
        self.frame_features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(self.frame_channels, 32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # First residual block
            ResBlock(32),

            # Downsample
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second residual block
            ResBlock(64),

            # Final convolution
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Adaptive pooling for better feature extraction
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        # Calculate the flattened size: 128 * 6 * 6 = 4608
        flat_size = 128 * 6 * 6

        # Enhanced frame processing
        self.frame_processor = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # Enhanced memory processing
        self.memory_processor = nn.Sequential(
            nn.Linear(self.memory_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # Hazard processing branch
        self.hazard_processor = nn.Sequential(
            nn.Conv2d(self.hazard_channel, 16,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # Goal processing branch - similar to hazard but with more features
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

        # Final integration with skip connection
        total_features = 128 + 32 + 32 + 64  # Frame + memory + hazard + goal features
        self.integration = nn.Sequential(
            nn.Linear(total_features, self.features_dim),
            nn.LayerNorm(self.features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 3:
            observations = observations.unsqueeze(0)

        # Split channels
        frames = observations[..., :self.frame_channels]
        memory = observations[..., -self.memory_channels -
                              self.hazard_channel - self.goal_channels:-self.hazard_channel - self.goal_channels]
        hazard = observations[..., -self.hazard_channel -
                              self.goal_channels:-self.goal_channels]
        goals = observations[..., -self.goal_channels:]

        # Process frames [B, H, W, C] -> [B, C, H, W]
        frames = frames.permute(0, 3, 1, 2)
        frame_features = self.frame_features(frames)
        frame_features = self.frame_processor(frame_features)

        # Process memory - average over spatial dimensions
        memory = memory.mean(dim=[1, 2])
        memory_features = self.memory_processor(memory)

        # Process hazard channel
        hazard = hazard.permute(0, 3, 1, 2)
        hazard_features = self.hazard_processor(hazard)

        # Process goal channels
        goals = goals.permute(0, 3, 1, 2)
        goal_features = self.goal_processor(goals)

        # Combine features with skip connection
        combined = torch.cat([
            frame_features,
            memory_features,
            hazard_features,
            goal_features
        ], dim=1)

        return self.integration(combined)
