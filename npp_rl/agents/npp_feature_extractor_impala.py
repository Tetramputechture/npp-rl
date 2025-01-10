from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from nclone_environments.basic_level_no_gold.constants import (
    TEMPORAL_FRAMES,
    GAME_STATE_FEATURES_MAX_ENTITY_COUNT_128,
    PLAYER_FRAME_HEIGHT,
    PLAYER_FRAME_WIDTH,
    RENDERED_VIEW_CHANNELS,
    RENDERED_VIEW_HEIGHT,
    RENDERED_VIEW_WIDTH,
)

torch.autograd.set_detect_anomaly(True)


class ResidualBlock(nn.Module):
    """Residual block used in IMPALA CNN architecture."""

    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(depth)

    def forward(self, x):
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x


class ConvSequence(nn.Module):
    """Convolutional sequence with max pooling and residual blocks."""

    def __init__(self, input_depth, output_depth):
        super().__init__()
        self.conv = nn.Conv2d(input_depth, output_depth, 3, padding=1)
        self.bn = nn.BatchNorm2d(output_depth)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(output_depth)
        self.res2 = ResidualBlock(output_depth)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """IMPALA CNN architecture."""

    def __init__(self, input_channels, input_height, input_width, depths=[16, 32, 32], output_dim=256):
        super().__init__()
        self.depths = depths

        # Build conv sequences
        self.conv_sequences = nn.ModuleList([
            ConvSequence(
                input_channels if i == 0 else depths[i-1],
                depth
            ) for i, depth in enumerate(depths)
        ])

        # Calculate output dimensions after conv sequences
        h, w = input_height, input_width
        for _ in depths:
            # Each ConvSequence halves dimensions due to stride 2 max pooling
            h = (h + 1) // 2  # +1 for odd dimensions
            w = (w + 1) // 2

        final_channels = depths[-1]
        flattened_dim = final_channels * h * w

        self.final_dense = nn.Linear(flattened_dim, output_dim)
        self.final_bn = nn.BatchNorm1d(output_dim)

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
        x = self.final_bn(x)
        x = nn.functional.relu(x)

        return x


class NPPFeatureExtractorImpala(BaseFeaturesExtractor):
    """Feature extractor using IMPALA CNN for visual processing and MLP for game state."""

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512, frame_stack: bool = False):
        super().__init__(observation_space, features_dim)

        player_frame_channels = TEMPORAL_FRAMES if frame_stack else 1
        # IMPALA CNN for player frame processing
        self.player_frame_cnn = ImpalaCNN(
            input_channels=player_frame_channels,  # Frame stacking channels
            input_height=PLAYER_FRAME_HEIGHT,
            input_width=PLAYER_FRAME_WIDTH,
            depths=[16, 32, 32],  # As per IMPALA paper
            output_dim=256
        )

        # IMPALA CNN for global view processing
        self.global_view_cnn = ImpalaCNN(
            input_channels=RENDERED_VIEW_CHANNELS,  # Single channel grayscale
            input_height=RENDERED_VIEW_HEIGHT,
            input_width=RENDERED_VIEW_WIDTH,
            depths=[16, 32, 32],
            output_dim=256
        )

        # MLP for game state processing with batch norm
        self.game_state_mlp = nn.Sequential(
            # First reduce by ~factor of 4
            nn.Linear(GAME_STATE_FEATURES_MAX_ENTITY_COUNT_128, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Further reduce by ~factor of 4
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Further reduce by ~factor of 4
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Final reduction to match other features
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Final fusion layer with batch norm
        self.fusion_layer = nn.Sequential(
            # 256 from player CNN + 256 from global map CNN + 128 from MLP
            nn.Linear(256 + 256 + 128, self.features_dim),
            nn.BatchNorm1d(self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim),
            nn.BatchNorm1d(self.features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process player frame stack
        player_frames = observations['player_frame'].float()
        if len(player_frames.shape) == 3:  # If no batch dimension
            player_frames = player_frames.unsqueeze(0)
        if len(player_frames.shape) == 4:
            player_frames = player_frames.permute(
                0, 3, 1, 2)  # (batch, channels, height, width)
        player_features = self.player_frame_cnn(player_frames)

        # Process global view
        global_view = observations['global_view'].float()
        if len(global_view.shape) == 3:  # If no batch dimension
            global_view = global_view.unsqueeze(0)
        if len(global_view.shape) == 4:
            # (batch, channels, height, width)
            global_view = global_view.permute(0, 3, 1, 2)
        global_features = self.global_view_cnn(global_view)

        # Process game state
        game_state = observations['game_state'].float()
        if len(game_state.shape) == 1:  # If no batch dimension
            game_state = game_state.unsqueeze(0)
        game_state_features = self.game_state_mlp(game_state)

        # Concatenate all features
        combined_features = torch.cat([
            player_features,
            global_features,
            game_state_features
        ], dim=1)

        # Final fusion
        output = self.fusion_layer(combined_features)

        return output
