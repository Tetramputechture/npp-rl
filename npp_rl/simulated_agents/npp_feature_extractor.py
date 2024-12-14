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
    with a branch for the stacked frames.
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Input channels configuration
        self.frame_channels = 7  # 1 current + 3 past + 3 historical

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

        # Final integration with skip connection
        total_features = 128 + 128  # Frame + frame features
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

        # Split channels
        frames = observations[..., :self.frame_channels]

        # Process frames [B, H, W, C] -> [B, C, H, W]
        frames = frames.permute(0, 3, 1, 2)
        frame_features = self.frame_features(frames)
        frame_features = self.frame_processor(frame_features)

        return self.integration(frame_features)
