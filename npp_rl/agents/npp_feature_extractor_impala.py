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

        # Calculate the output size based on input dimensions
        # For player frame (50x60):
        # After 3 ConvSequences with stride 2: 50x60 -> 25x30 -> 13x15 -> 7x8
        # Final feature map will be 32 x 7 x 8 = 1792
        # Changed from 32 * 7 * 8 to match actual flattened size
        self.final_dense = nn.Linear(1792, 256)

    def forward(self, x):
        # Initial scaling
        x = x / 255.0

        # Process through conv sequences
        for conv_sequence in self.conv_sequences:
            x = conv_sequence(x)

        # Print the shape of the tensor after conv sequences for debugging
        print('Shape after conv sequences:', x.shape)

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
            input_channels=4,  # Four stacked frames when frame stacking is enabled
            depths=[16, 32, 32]  # As per IMPALA paper
        )

        # IMPALA CNN for base frame processing
        self.base_frame_cnn = ImpalaCNN(
            input_channels=1,  # Single grayscale frame
            depths=[16, 32, 32]
        )

        # MLP for game state processing
        self.game_state_mlp = nn.Sequential(
            nn.Linear(29, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            # 256 from each CNN + 128 from MLP
            nn.Linear(256 + 256 + 128, self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process player frame stack
        player_frames = observations['player_frame'].float()
        player_frames = player_frames.permute(
            0, 3, 1, 2)  # (batch, channels, height, width)
        player_features = self.player_frame_cnn(player_frames)

        # Process base frame
        base_frames = observations['base_frame'].float()
        base_frames = base_frames.permute(0, 3, 1, 2)
        base_features = self.base_frame_cnn(base_frames)

        # Process game state
        game_state = observations['game_state'].float()
        game_state_features = self.game_state_mlp(game_state)

        # Concatenate all features
        combined_features = torch.cat([
            player_features,
            base_features,
            game_state_features
        ], dim=1)

        # Final fusion
        output = self.fusion_layer(combined_features)

        return output
