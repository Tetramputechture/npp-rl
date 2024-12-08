from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

NUM_NUMERICAL_FEATURES = 11


class NppFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for N++ environment.

    Architecture designed to process observations from our N++ environment where:
    Input: Single tensor of shape (84, 84, frame_stack + 9) containing:
        - First frame_stack channels: Stacked grayscale frames
        - Last NUM_NUMERICAL_FEATURES channels: Numerical features broadcast to 84x84 spatial dimensions

    The network separates and processes visual and numerical data through appropriate 
    pathways before combining them into a final feature representation.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]
        self.frame_stack = n_input_channels - NUM_NUMERICAL_FEATURES

        # Visual processing network
        self.cnn = nn.Sequential(
            # Initial layer to detect basic movement patterns
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # Attention mechanism for movement tracking
            nn.Conv2d(32, 32, kernel_size=1),  # 1x1 conv for channel attention
            nn.Sigmoid(),

            # Feature extraction
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Spatial attention for obstacle detection
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid(),

            # High-level feature extraction
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Multiple residual connections for better gradient flow
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Flatten()
        )

        # Calculate CNN output dimension: 128 channels * 7 * 7 = 6272
        self.cnn_output_dim = 128 * 7 * 7

        # Separate networks for different types of numerical features
        self.position_net = nn.Sequential(
            nn.Linear(4, 128),  # player_x, player_y, vx, vy
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.objective_net = nn.Sequential(
            nn.Linear(4, 128),  # exit_x, exit_y, switch_x, switch_y
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.state_net = nn.Sequential(
            nn.Linear(3, 64),  # time, switch_activated, in_air
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        # Combine numerical features with attention
        self.numerical_attention = nn.Sequential(
            nn.Linear(128 + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Gated fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.cnn_output_dim + 512, 512),
            nn.Sigmoid()
        )

        # Final processing with skip connections
        self.final_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        """
        Forward pass with organized numerical feature processing.

        Args:
            observations: Tensor containing stacked frames and numerical features
                Shape: (batch_size, 84, 84, frame_stack + NUM_NUMERICAL_FEATURES)
        """
        # Process visual features
        visual = observations[..., :self.frame_stack]
        visual = visual.permute(0, 3, 1, 2)
        visual_features = self.cnn(visual)

        # Get numerical features (they're already in the correct order from preprocessing)
        numerical = observations[..., self.frame_stack:]
        # Take first spatial location since features are broadcast
        numerical = numerical[:, 0, 0, :]

        # Split numerical features into their groups
        # First 4 features
        position_features = numerical[:, :4]
        objective_features = numerical[:, 4:8]                # Next 4 features
        # Remaining features
        state_features = numerical[:, 8:NUM_NUMERICAL_FEATURES]

        # Process each group through its specialized network
        processed_position = self.position_net(position_features)
        processed_objectives = self.objective_net(objective_features)
        processed_state = self.state_net(state_features)

        # Combine numerical features with attention
        numerical_combined = torch.cat([
            processed_position,
            processed_objectives,
            processed_state
        ], dim=1)
        numerical_features = self.numerical_attention(numerical_combined)

        # Apply gated fusion
        combined = torch.cat([visual_features, numerical_features], dim=1)
        gate = self.fusion_gate(combined)

        # Final processing
        fused_features = gate * numerical_features + \
            (1 - gate) * visual_features
        return self.final_net(fused_features)
