from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from npp_rl.environments.constants import TEMPORAL_FRAMES

torch.autograd.set_detect_anomaly(True)


class ResidualBlock(nn.Module):
    """Residual block used in IMPALA CNN architecture."""

    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + x


class ConvSequence(nn.Module):
    """Convolutional sequence with max pooling and residual blocks."""

    def __init__(self, input_depth, output_depth):
        super().__init__()
        self.conv = nn.Conv2d(input_depth, output_depth, 3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(output_depth)
        self.res2 = ResidualBlock(output_depth)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """IMPALA CNN architecture."""

    def __init__(self, input_channels, depths=[16, 32, 32]):
        super().__init__()
        self.depths = depths

        # Build conv sequences
        self.conv_sequences = nn.ModuleList([
            ConvSequence(
                input_channels if i == 0 else depths[i-1],
                depth
            ) for i, depth in enumerate(depths)
        ])

        # Final dense layer
        # 32 is the last depth, 11x11 is the spatial dimension after conv sequences
        self.final_dense = nn.Linear(32 * 11 * 11, 256)

    def forward(self, x):
        # Initial scaling
        x = x / 255.0

        # Process through conv sequences
        for conv_sequence in self.conv_sequences:
            x = conv_sequence(x)

        # Flatten and process through dense layer
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.final_dense(x)
        x = nn.functional.relu(x)

        return x


class NPPFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor using IMPALA CNN for visual processing."""

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Input channels configuration
        self.temporal_frames = TEMPORAL_FRAMES

        # IMPALA CNN for visual processing
        self.impala_cnn = ImpalaCNN(
            input_channels=self.temporal_frames,
            depths=[16, 32, 32]  # As per IMPALA paper
        )

        # Final layer to match features_dim
        self.final_layer = nn.Sequential(
            nn.Linear(256, self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process frame stack - observations is already (batch_size, height, width, channels)
        visual_input = observations.float()

        # Permute from (batch_size, height, width, channels) to (batch_size, channels, height, width)
        visual_input = visual_input.permute(0, 3, 1, 2)

        # Process through IMPALA CNN
        cnn_features = self.impala_cnn(visual_input)

        # Process through final layer
        output = self.final_layer(cnn_features)

        return output
