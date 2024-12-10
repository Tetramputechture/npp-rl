from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn
import gymnasium


class NppFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for N++ environment.

    Architecture designed to process observations from our N++ environment where:
    Input: Single tensor of shape (84, 84, total_frames + num_features) containing:
        - First frame_stack channels: Recent stacked grayscale frames
        - Next 8 channels: Historical frames at 8, 16, 32, 64, 128, 256, 512, 1024 timesteps
        - Last channels: Numerical features broadcast to 84x84 spatial dimensions
    """

    def __init__(self, observation_space: gymnasium.spaces.Box):
        # Calculate output features dimension
        features_dim = 512  # Output features dimension

        super().__init__(observation_space, features_dim)

        # CNN for processing visual features
        self.cnn = nn.Sequential(
            # First conv layer processes recent frames (4 recent frames)
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),

            # Second conv layer processes historical frames
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),

            # Third conv layer combines all features
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Flatten(),
        )

        # Separate CNN for processing historical frames (8 historical frames)
        self.historical_cnn = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Separate networks for different types of numerical features
        self.state_net = nn.Sequential(
            # time remaining
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        total_numerical_features = 64

        # Combine numerical features with attention
        self.numerical_attention = nn.Sequential(
            nn.Linear(total_numerical_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Gated fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(512 + 512, 512),
            nn.Sigmoid()
        )

        # Final processing
        self.final_net = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process the environment's observation.

        Args:
            observations: torch.Tensor - The input tensor containing stacked frames and features
                        Shape: (batch_size, 84, 84, total_channels)

        Returns:
            torch.Tensor - The extracted features
                        Shape: (batch_size, features_dim)
        """
        # Process recent frames (first 4 channels)
        recent_visual = observations[..., :4]
        recent_visual = recent_visual.permute(0, 3, 1, 2)
        recent_features = self.cnn(recent_visual)

        # Process historical frames (next 8 channels)
        historical_visual = observations[..., 4:12]
        historical_visual = historical_visual.permute(0, 3, 1, 2)
        historical_features = self.historical_cnn(historical_visual)

        # Get numerical features (they're already in the correct order from preprocessing)
        numerical = observations[..., 12:]
        # Take first spatial location since features are broadcast
        numerical = numerical[:, 0, 0, :]

        # Process numerical features
        processed_state = self.state_net(numerical)
        numerical_features = self.numerical_attention(processed_state)

        # Apply gating fusion
        gate_recent = self.fusion_gate(
            torch.cat([recent_features, numerical_features], dim=1))
        gate_historical = self.fusion_gate(
            torch.cat([historical_features, numerical_features], dim=1))

        # Combine features with gating
        fused_features = (
            gate_recent * recent_features +
            gate_historical * historical_features +
            (1 - gate_recent - gate_historical) * numerical_features
        )

        return self.final_net(fused_features)
