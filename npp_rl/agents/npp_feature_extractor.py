from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from npp_rl.environments.constants import TEMPORAL_FRAMES

torch.autograd.set_detect_anomaly(True)


class NatureCNN(nn.Module):
    """Nature CNN architecture from the DQN paper."""

    def __init__(self, input_channels, features_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of flattened features
        # For player frame: 50x60 -> 11x13 -> 4x5 -> 2x3
        # For base frame: 120x160 -> 29x39 -> 13x18 -> 11x16
        with torch.no_grad():
            # Use player frame size for calculation
            dummy_input = torch.zeros(1, input_channels, 50, 60)
            conv_out = self.conv_layers(dummy_input)
            n_flatten = conv_out.shape[1]

        # Create linear layers with proper input dimensions
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),  # First reduce to 512
            nn.ReLU(),
            nn.Linear(512, features_dim)  # Then to desired output dimension
        )

    def forward(self, x):
        x = x / 255.0  # Normalize
        return self.linear(self.conv_layers(x))


class NPPFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor using Nature CNN for visual processing and MLP for game state."""

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Nature CNN for player frame processing
        self.player_frame_cnn = NatureCNN(
            input_channels=4,  # Four stacked frames when frame stacking is enabled
            features_dim=256
        )

        # CNN for global map processing - now handling full 42x23 resolution
        self.global_map_cnn = nn.Sequential(
            # Input: 23x42x4 (HxWxC)
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 23x42x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 12x21x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 6x11x64
            nn.Flatten(),
            nn.Linear(6 * 11 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
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
            # 256 from player CNN + 128 from global map CNN + 128 from MLP
            nn.Linear(256 + 128 + 128, self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        # Process player frame stack
        player_frames = observations['player_frame'].float()
        if len(player_frames.shape) == 3:  # If no batch dimension
            player_frames = player_frames.unsqueeze(0)  # Add batch dimension
        if len(player_frames.shape) == 4:
            player_frames = player_frames.permute(0, 3, 1, 2)
        player_features = self.player_frame_cnn(player_frames)

        # Process global map
        global_map = observations['global_map'].float()
        if len(global_map.shape) == 3:  # If no batch dimension
            global_map = global_map.unsqueeze(0)
        if len(global_map.shape) == 4:
            # (batch, channels, height, width)
            global_map = global_map.permute(0, 3, 1, 2)
        global_features = self.global_map_cnn(global_map)

        # Process game state
        game_state = observations['game_state'].float()
        if len(game_state.shape) == 1:  # If no batch dimension
            game_state = game_state.unsqueeze(0)  # Add batch dimension
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
