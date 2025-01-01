from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from nclone_environments.basic_level_no_gold.constants import (
    TEMPORAL_FRAMES,
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,
)

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

        # Calculate the output size based on input dimensions
        # For 84x84 input:
        # After 3 ConvSequences with stride 2: 84x84 -> 42x42 -> 21x21 -> 11x11
        # Final feature map will be 32 x 11 x 11 = 3,872
        self.final_dense = nn.Linear(3872, 256)

    def forward(self, x):
        # Initial scaling
        x = x / 255.0

        # Process through conv sequences
        for conv_sequence in self.conv_sequences:
            x = conv_sequence(x)

        # Flatten and process through dense layer
        # Flatten all dimensions except batch
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.final_dense(x)
        x = nn.functional.relu(x)

        return x


class NPPFeatureExtractorImpala(BaseFeaturesExtractor):
    """Feature extractor using IMPALA CNN for visual processing and MLP for game state."""

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # IMPALA CNN for player frame processing
        self.player_frame_cnn = ImpalaCNN(
            input_channels=TEMPORAL_FRAMES,  # Three stacked frames
            depths=[16, 32, 32]  # As per IMPALA paper
        )

        # MLP for game state processing
        self.game_state_mlp = nn.Sequential(
            nn.Linear(GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            # 256 from CNN + 64 from MLP
            nn.Linear(256 + 64, self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process player frame stack
        player_frames = observations['player_frame'].float()
        player_frames = player_frames.permute(
            0, 3, 1, 2)  # (batch, channels, height, width)
        player_features = self.player_frame_cnn(player_frames)

        # Process game state
        game_state = observations['game_state'].float()
        game_state_features = self.game_state_mlp(game_state)

        # Concatenate all features
        combined_features = torch.cat([
            player_features,
            game_state_features
        ], dim=1)

        # Final fusion
        output = self.fusion_layer(combined_features)

        return output
