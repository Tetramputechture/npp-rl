"""Enhanced attention-based state encoder for ninja physics feature learning."""

import torch
import torch.nn as nn
from typing import Dict
from nclone.gym_environment.constants import NINJA_STATE_DIM


class AttentiveStateMLP(nn.Module):
    """Enhanced state encoder with adaptive component dimensions and hierarchical attention.

    Separates ninja physics state (41 dims) into semantic components and applies
    hierarchical cross-attention to learn inter-component relationships for low-level control.

    Physics component breakdown (41 dims total):
    - Velocity (3): magnitude, direction_x, direction_y
    - Movement states (5): ground/air/wall/special, airborne
    - Input (2): horizontal, jump
    - Timing Buffers (3): jump/floor/wall timing buffers
    - Contact (6): floor/wall/ceiling contact, normals, slope, wall direction
    - Forces (7): gravity, walled, floor_normal_x, ceiling_normals, drag, friction
    - Temporal (6): velocity_change, momentum, frames_airborne, jump_duration, state_transition
    - Energy (4): kinetic, potential, force_magnitude, energy_change [NEW]
    - Surface (4): floor_strength, wall_strength, slope, wall_interaction [NEW]
    - Episode (1): time_remaining until truncation [NEW]

    Enhanced Architecture:
    1. Adaptive component-specific encoders with complexity-based dimensions
    2. Hierarchical attention: component encoding → cross-component interaction
    3. Physics context gating based on movement state
    4. Output projection to output_dim
    """

    # Component dimensions (input features per component)
    VELOCITY_DIM = 3
    MOVEMENT_DIM = 5
    INPUT_DIM = 2
    TIMING_BUFFERS_DIM = 3
    CONTACT_DIM = 6
    FORCES_DIM = 7
    TEMPORAL_DIM = 6
    ENERGY_DIM = 4
    SURFACE_DIM = 4
    EPISODE_DIM = 1

    # Component start indices
    VELOCITY_START = 0
    MOVEMENT_START = VELOCITY_START + VELOCITY_DIM  # 3
    INPUT_START = MOVEMENT_START + MOVEMENT_DIM  # 8
    TIMING_BUFFERS_START = INPUT_START + INPUT_DIM  # 10
    CONTACT_START = TIMING_BUFFERS_START + TIMING_BUFFERS_DIM  # 13
    FORCES_START = CONTACT_START + CONTACT_DIM  # 19
    TEMPORAL_START = FORCES_START + FORCES_DIM  # 26
    ENERGY_START = TEMPORAL_START + TEMPORAL_DIM  # 32
    SURFACE_START = ENERGY_START + ENERGY_DIM  # 36
    EPISODE_START = SURFACE_START + SURFACE_DIM  # 40

    # Total expected dimension
    TOTAL_DIM = (
        VELOCITY_DIM
        + MOVEMENT_DIM
        + INPUT_DIM
        + TIMING_BUFFERS_DIM
        + CONTACT_DIM
        + FORCES_DIM
        + TEMPORAL_DIM
        + ENERGY_DIM
        + SURFACE_DIM
        + EPISODE_DIM
    )  # 41

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        debug_mode: bool = False,
    ):
        super().__init__()

        # Validate that our component dimensions add up to NINJA_STATE_DIM
        assert self.TOTAL_DIM == NINJA_STATE_DIM, (
            f"Component dimensions ({self.TOTAL_DIM}) don't match NINJA_STATE_DIM ({NINJA_STATE_DIM})"
        )

        self.debug_mode = debug_mode
        self.num_heads = num_heads

        # Adaptive component dimensions based on physics complexity
        self.component_dims = {
            "velocity": 32,  # 3 → 32 (medium complexity, critical for control)
            "movement": 32,  # 5 → 32 (medium complexity, state machine)
            "input": 16,  # 2 → 16 (low complexity, direct inputs)
            "timing_buffers": 24,  # 3 → 24 (medium complexity, timing-critical)
            "contact": 64,  # 6 → 64 (high complexity, collision physics)
            "forces": 64,  # 7 → 64 (high complexity, applied physics)
            "temporal": 32,  # 6 → 32 (medium complexity, history-dependent)
            "energy": 48,  # 4 → 48 (NEW: high complexity, derived physics)
            "surface": 32,  # 4 → 32 (NEW: medium complexity, contact physics)
            "episode": 16,  # 1 → 16 (NEW: low complexity, episode constraint)
        }

        # Component-specific encoders with adaptive dimensions
        self.component_encoders = nn.ModuleDict(
            {
                "velocity": nn.Sequential(
                    nn.Linear(self.VELOCITY_DIM, self.component_dims["velocity"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "movement": nn.Sequential(
                    nn.Linear(self.MOVEMENT_DIM, self.component_dims["movement"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "input": nn.Sequential(
                    nn.Linear(self.INPUT_DIM, self.component_dims["input"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "timing_buffers": nn.Sequential(
                    nn.Linear(
                        self.TIMING_BUFFERS_DIM, self.component_dims["timing_buffers"]
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "contact": nn.Sequential(
                    nn.Linear(self.CONTACT_DIM, self.component_dims["contact"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "forces": nn.Sequential(
                    nn.Linear(self.FORCES_DIM, self.component_dims["forces"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "temporal": nn.Sequential(
                    nn.Linear(self.TEMPORAL_DIM, self.component_dims["temporal"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "energy": nn.Sequential(
                    nn.Linear(self.ENERGY_DIM, self.component_dims["energy"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "surface": nn.Sequential(
                    nn.Linear(self.SURFACE_DIM, self.component_dims["surface"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "episode": nn.Sequential(
                    nn.Linear(self.EPISODE_DIM, self.component_dims["episode"]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
            }
        )

        # Uniform dimension for attention (must be divisible by num_heads)
        self.uniform_dim = 64  # Fixed uniform dimension for attention
        assert self.uniform_dim % num_heads == 0, (
            f"uniform_dim {self.uniform_dim} must be divisible by num_heads {num_heads}"
        )

        # Project each component to uniform dimension for cross-component attention
        self.uniform_projections = nn.ModuleDict(
            {
                name: nn.Linear(dim, self.uniform_dim)
                for name, dim in self.component_dims.items()
            }
        )

        # Multi-head cross-component attention
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=self.uniform_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for attention
        self.norm = nn.LayerNorm(self.uniform_dim)

        # Physics context gating network (based on movement state)
        self.physics_gate_network = nn.Sequential(
            nn.Linear(5, 32),  # movement state (5) → hidden
            nn.ReLU(),
            nn.Linear(32, 10),  # → gates for 10 components
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.uniform_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _split_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split 41D enhanced physics state into semantic components."""
        return {
            "velocity": x[
                :, self.VELOCITY_START : self.VELOCITY_START + self.VELOCITY_DIM
            ],
            "movement": x[
                :, self.MOVEMENT_START : self.MOVEMENT_START + self.MOVEMENT_DIM
            ],
            "input": x[:, self.INPUT_START : self.INPUT_START + self.INPUT_DIM],
            "timing_buffers": x[
                :,
                self.TIMING_BUFFERS_START : self.TIMING_BUFFERS_START
                + self.TIMING_BUFFERS_DIM,
            ],
            "contact": x[:, self.CONTACT_START : self.CONTACT_START + self.CONTACT_DIM],
            "forces": x[:, self.FORCES_START : self.FORCES_START + self.FORCES_DIM],
            "temporal": x[
                :, self.TEMPORAL_START : self.TEMPORAL_START + self.TEMPORAL_DIM
            ],  # 6 physics temporal features
            "energy": x[:, self.ENERGY_START : self.ENERGY_START + self.ENERGY_DIM],
            "surface": x[:, self.SURFACE_START : self.SURFACE_START + self.SURFACE_DIM],
            "episode": x[
                :, self.EPISODE_START : self.EPISODE_START + self.EPISODE_DIM
            ],  # time_remaining
        }

    def _compute_physics_gates(self, movement_state: torch.Tensor) -> torch.Tensor:
        """Compute physics-aware attention gates based on movement state."""
        # movement_state: [batch, 5] (ground/air/wall/special + airborne)
        gates = self.physics_gate_network(movement_state)  # → [batch, 10]
        return torch.sigmoid(gates)  # Soft gating [0,1] per component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward with hierarchical attention and physics gating.

        Args:
            x: State tensor [batch, 41] (40 physics features + 1 time_remaining)

        Returns:
            Encoded features [batch, output_dim]
        """
        # Validate input tensor for NaN/Inf (only if debug mode enabled)
        if self.debug_mode:
            if torch.isnan(x).any():
                nan_count = torch.isnan(x).sum().item()
                raise ValueError(
                    f"[ENHANCED_ATTENTIVE_STATE_MLP] NaN in input tensor. "
                    f"Shape: {x.shape}, NaN count: {nan_count}"
                )
            if torch.isinf(x).any():
                inf_count = torch.isinf(x).sum().item()
                raise ValueError(
                    f"[ENHANCED_ATTENTIVE_STATE_MLP] Inf in input tensor. "
                    f"Shape: {x.shape}, Inf count: {inf_count}"
                )

        # Handle frame stacking: extract most recent frame
        if x.dim() == 2:
            if x.shape[1] == NINJA_STATE_DIM:
                # Single frame with correct dimension
                pass
            elif x.shape[1] % NINJA_STATE_DIM == 0 and x.shape[1] > NINJA_STATE_DIM:
                # Stacked states: take the most recent frame
                x = x[:, -NINJA_STATE_DIM:]  # Most recent frame
            else:
                raise ValueError(
                    f"Expected {NINJA_STATE_DIM}D state or multiple thereof, got {x.shape[1]}D"
                )
        elif x.dim() == 3:
            # [batch, stack_size, dim] → take last frame
            x = x[:, -1, :]

        # Input is always 41D (40 physics + 1 time_remaining)
        if x.shape[1] != NINJA_STATE_DIM:
            raise ValueError(f"Expected {NINJA_STATE_DIM}D state, got {x.shape[1]}D")

        # Split into 10 semantic components
        components = self._split_components(x)  # 41D → 10 components

        # Level 1: Component-level encoding with adaptive dimensions
        encoded_components = {}
        for name, component in components.items():
            encoded = self.component_encoders[name](component)  # → component_dims[name]
            encoded_components[name] = encoded

            if self.debug_mode and torch.isnan(encoded).any():
                raise ValueError(
                    f"[ENHANCED_ATTENTIVE_STATE_MLP] NaN after {name}_encoder. "
                    f"Input range: [{component.min():.4f}, {component.max():.4f}]"
                )

        # Level 2: Cross-component interaction attention
        # Project all to uniform dimension for attention
        uniform_components = []
        component_names = list(components.keys())
        for name in component_names:
            encoded = encoded_components[name]
            uniform = self.uniform_projections[name](encoded)  # → 64D uniform
            uniform_components.append(uniform)

            if self.debug_mode and torch.isnan(uniform).any():
                raise ValueError(
                    f"[ENHANCED_ATTENTIVE_STATE_MLP] NaN after {name}_projection"
                )

        # Stack for attention: [batch, 10_components, 64]
        component_tokens = torch.stack(uniform_components, dim=1)

        if self.debug_mode and torch.isnan(component_tokens).any():
            raise ValueError(
                "[ENHANCED_ATTENTIVE_STATE_MLP] NaN after stacking physics tokens"
            )

        # Multi-head cross-component attention
        attended_tokens, _ = self.interaction_attention(
            component_tokens, component_tokens, component_tokens
        )

        if self.debug_mode and torch.isnan(attended_tokens).any():
            raise ValueError(
                "[ENHANCED_ATTENTIVE_STATE_MLP] NaN after interaction attention"
            )

        # Residual connection + layer norm
        component_tokens = self.norm(component_tokens + attended_tokens)

        if self.debug_mode and torch.isnan(component_tokens).any():
            raise ValueError("[ENHANCED_ATTENTIVE_STATE_MLP] NaN after layer norm")

        # Physics context gating based on movement state
        movement_state = components["movement"]  # [batch, 5]
        physics_gates = self._compute_physics_gates(movement_state)  # [batch, 9]
        gated_tokens = component_tokens * physics_gates.unsqueeze(-1)  # Apply gates

        if self.debug_mode and torch.isnan(gated_tokens).any():
            raise ValueError("[ENHANCED_ATTENTIVE_STATE_MLP] NaN after physics gating")

        # Global pooling and output projection
        pooled_features = gated_tokens.mean(dim=1)  # [batch, 64]

        if self.debug_mode and torch.isnan(pooled_features).any():
            raise ValueError("[ENHANCED_ATTENTIVE_STATE_MLP] NaN after pooling")

        output = self.output_projection(pooled_features)  # [batch, output_dim]

        if self.debug_mode and torch.isnan(output).any():
            raise ValueError(
                "[ENHANCED_ATTENTIVE_STATE_MLP] NaN after output projection"
            )

        return output
