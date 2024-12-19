from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium
from npp_rl.environments.constants import TEMPORAL_FRAMES

torch.autograd.set_detect_anomaly(True)


class NPPFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor using Nature CNN for visual processing and MLP for player state."""

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Input channels configuration
        self.temporal_frames = TEMPORAL_FRAMES

        # Nature CNN for visual processing (as per OpenAI's DQN paper)
        self.nature_cnn = nn.Sequential(
            # Layer 1: 32 8x8 filters with stride 4
            nn.Conv2d(self.temporal_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            # Layer 2: 64 4x4 filters with stride 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Layer 3: 64 3x3 filters with stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Calculate the size of flattened features from Nature CNN
        # We'll do a forward pass with a dummy tensor to get the size
        with torch.no_grad():
            # Nature CNN expects 84x84 input
            dummy_input = torch.zeros(1, self.temporal_frames, 84, 84)
            n_flatten = self.nature_cnn(dummy_input).shape[1]

        # Dense layer after CNN
        self.visual_net = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 192),
            nn.ReLU()
        )

        # Player state MLP branch
        self.player_state_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Final integration layer
        combined_features = 192 + 64
        self.integration = nn.Sequential(
            nn.Linear(combined_features, self.features_dim),
            nn.ReLU(),
            nn.Linear(self.features_dim, self.features_dim)
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # Process frame stack
        visual_input = observations['visual'].float() / 255.0
        visual_input = visual_input.permute(0, 3, 1, 2)

        # Ensure input is the right size (84x84) as per Nature DQN
        if visual_input.shape[-1] != 84:
            visual_input = nn.functional.interpolate(
                visual_input, size=(84, 84), mode='bilinear', align_corners=False)

        # Process through Nature CNN
        cnn_features = self.nature_cnn(visual_input)
        visual_features = self.visual_net(cnn_features)

        # Process player state features
        player_features = self.player_state_mlp(observations['player_state'])

        # Combine all features
        combined = torch.cat([
            visual_features,
            player_features
        ], dim=1)

        # Final integration
        output = self.integration(combined)
        return output
