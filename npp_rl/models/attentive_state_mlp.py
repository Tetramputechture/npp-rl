"""Attention-based state encoder for ninja physics feature learning."""

import torch
import torch.nn as nn
from nclone.gym_environment.constants import NINJA_STATE_DIM


class AttentiveStateMLP(nn.Module):
    """State encoder with multi-head attention over ninja physics components.

    Separates 29-dim ninja physics state into semantic components and applies
    cross-attention to learn inter-component relationships for low-level control.

    Physics component breakdown (29 dims total):
    - Velocity (3): magnitude, direction_x, direction_y
    - Movement states (5): ground/air/wall/special, airborne
    - Input (2): horizontal, jump
    - Buffers (3): jump/floor/wall timing buffers
    - Contact (6): floor/wall/ceiling contact, normals, slope, wall direction
    - Forces (10): acceleration, gravity, jump duration, walled, drag, friction

    Architecture:
    1. Component-specific encoders project to uniform dimension
    2. Multi-head attention fuses physics components
    3. Output projection to output_dim
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.debug_mode = debug_mode

        # Component-specific encoders for each physics group
        # Use different hidden dims based on component importance/size
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # 3 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.movement_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 4),  # 5 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.input_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 8),  # 2 → 16
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.buffer_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 8),  # 3 → 16
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.contact_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 4),  # 6 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.forces_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 2),  # 10 → 64
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Uniform dimension for attention (must be divisible by num_heads)
        # Use 64 for hidden_dim=128 (64 / 4 heads = 16 per head)
        self.uniform_dim = hidden_dim // 2  # 64
        assert self.uniform_dim % num_heads == 0, (
            f"uniform_dim {self.uniform_dim} must be divisible by num_heads {num_heads}"
        )

        # Project each component to uniform dimension for attention
        self.velocity_proj = nn.Linear(hidden_dim // 4, self.uniform_dim)  # 32 → 64
        self.movement_proj = nn.Linear(hidden_dim // 4, self.uniform_dim)  # 32 → 64
        self.input_proj = nn.Linear(hidden_dim // 8, self.uniform_dim)  # 16 → 64
        self.buffer_proj = nn.Linear(hidden_dim // 8, self.uniform_dim)  # 16 → 64
        self.contact_proj = nn.Linear(hidden_dim // 4, self.uniform_dim)  # 32 → 64
        self.forces_proj = nn.Linear(hidden_dim // 2, self.uniform_dim)  # 64 → 64

        # Multi-head attention across physics components
        self.attention = nn.MultiheadAttention(
            embed_dim=self.uniform_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for uniform dim
        self.norm = nn.LayerNorm(self.uniform_dim)

        # Output: pool 6 physics components and project
        self.pool_proj = nn.Linear(self.uniform_dim, output_dim)
        self.output_activation = nn.ReLU()
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attentive physics state encoder.

        Args:
            x: State tensor [batch, 29] or [batch, stack_size * 29] if stacked

        Returns:
            Encoded features [batch, output_dim]
        """
        # Validate input tensor for NaN/Inf (only if debug mode enabled)
        if self.debug_mode:
            if torch.isnan(x).any():
                nan_count = torch.isnan(x).sum().item()
                nan_indices = torch.where(torch.isnan(x.view(-1)))[0][:10].tolist()
                raise ValueError(
                    f"[ATTENTIVE_STATE_MLP] NaN in input tensor. "
                    f"Shape: {x.shape}, NaN count: {nan_count}, "
                    f"First NaN indices (flattened): {nan_indices}"
                )
            if torch.isinf(x).any():
                inf_count = torch.isinf(x).sum().item()
                raise ValueError(
                    f"[ATTENTIVE_STATE_MLP] Inf in input tensor. "
                    f"Shape: {x.shape}, Inf count: {inf_count}"
                )

        # Handle frame stacking: extract most recent frame
        if x.dim() == 2:
            if x.shape[1] % NINJA_STATE_DIM == 0 and x.shape[1] > NINJA_STATE_DIM:
                # Stacked states: [batch, stack_size * 29] → take last 29
                x = x[:, -NINJA_STATE_DIM:]  # Most recent frame
            elif x.shape[1] != NINJA_STATE_DIM:
                raise ValueError(
                    f"Expected state dim {NINJA_STATE_DIM} or multiple of {NINJA_STATE_DIM}, got {x.shape[1]}"
                )
        elif x.dim() == 3:
            # [batch, stack_size, 29] → take last frame
            x = x[:, -1, :]

        # Split into physics components (indices from OBSERVATION_SPACE_README.md)
        velocity = x[:, 0:3]  # [0-2]: vel_mag, vel_dir_x, vel_dir_y
        movement = x[:, 3:8]  # [3-7]: movement categories, airborne
        input_state = x[:, 8:10]  # [8-9]: horizontal input, jump input
        buffers = x[:, 10:13]  # [10-12]: jump/floor/wall buffers
        contact = x[:, 13:19]  # [13-18]: contact strength, normals, slope, wall dir
        forces = x[
            :, 19:29
        ]  # [19-28]: acceleration, gravity, jump duration, walled, drag, friction

        # Validate physics components (only if debug mode)
        if self.debug_mode:
            # Check for all-zero physics (suspicious)
            if x.abs().max() < 1e-8:
                raise ValueError("[ATTENTIVE_STATE_MLP] All-zero physics input")

            # Check for extreme values (potential overflow/underflow issues)
            if x.abs().max() > 1e6:
                raise ValueError(
                    f"[ATTENTIVE_STATE_MLP] Extreme physics values: {x.abs().max()}"
                )

        # Encode each component
        vel_feat = self.velocity_encoder(velocity)  # [batch, 32]
        if self.debug_mode and torch.isnan(vel_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after velocity_encoder. "
                f"Input range: [{velocity.min():.4f}, {velocity.max():.4f}]"
            )

        move_feat = self.movement_encoder(movement)  # [batch, 32]
        if self.debug_mode and torch.isnan(move_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after movement_encoder. "
                f"Input range: [{movement.min():.4f}, {movement.max():.4f}]"
            )

        input_feat = self.input_encoder(input_state)  # [batch, 16]
        if self.debug_mode and torch.isnan(input_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after input_encoder. "
                f"Input range: [{input_state.min():.4f}, {input_state.max():.4f}]"
            )

        buffer_feat = self.buffer_encoder(buffers)  # [batch, 16]
        if self.debug_mode and torch.isnan(buffer_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after buffer_encoder. "
                f"Input range: [{buffers.min():.4f}, {buffers.max():.4f}]"
            )

        contact_feat = self.contact_encoder(contact)  # [batch, 32]
        if self.debug_mode and torch.isnan(contact_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after contact_encoder. "
                f"Input range: [{contact.min():.4f}, {contact.max():.4f}]"
            )

        forces_feat = self.forces_encoder(forces)  # [batch, 64]
        if self.debug_mode and torch.isnan(forces_feat).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after forces_encoder. "
                f"Input range: [{forces.min():.4f}, {forces.max():.4f}]"
            )

        # Project to uniform dimension
        vel_token = self.velocity_proj(vel_feat)  # [batch, 64]
        move_token = self.movement_proj(move_feat)  # [batch, 64]
        input_token = self.input_proj(input_feat)  # [batch, 64]
        buffer_token = self.buffer_proj(buffer_feat)  # [batch, 64]
        contact_token = self.contact_proj(contact_feat)  # [batch, 64]
        forces_token = self.forces_proj(forces_feat)  # [batch, 64]

        if self.debug_mode:
            for token_name, token in [
                ("velocity", vel_token),
                ("movement", move_token),
                ("input", input_token),
                ("buffer", buffer_token),
                ("contact", contact_token),
                ("forces", forces_token),
            ]:
                if torch.isnan(token).any():
                    raise ValueError(
                        f"[ATTENTIVE_STATE_MLP] NaN after {token_name}_proj"
                    )

        # Stack into component sequence for attention
        # Shape: [batch, 6, 64] - 6 physics component tokens
        component_tokens = torch.stack(
            [
                vel_token,
                move_token,
                input_token,
                buffer_token,
                contact_token,
                forces_token,
            ],
            dim=1,
        )
        if self.debug_mode and torch.isnan(component_tokens).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after stacking physics tokens")

        # Check for degenerate tokens before attention (only if debug mode)
        if self.debug_mode:
            token_std = component_tokens.std(dim=-1)
            if (token_std < 1e-8).any():
                raise ValueError(
                    f"[ATTENTIVE_STATE_MLP] Degenerate tokens (near-zero std). "
                    f"Token std range: [{token_std.min():.4e}, {token_std.max():.4e}]"
                )

        # Multi-head cross-component attention
        # Each physics component can attend to all others
        # (e.g., velocity can attend to contact for ground/wall interaction)
        attn_out, _ = self.attention(
            component_tokens,
            component_tokens,
            component_tokens,
            need_weights=False,
        )
        # attn_out shape: [batch, 6, 64]
        if self.debug_mode and torch.isnan(attn_out).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after attention. "
                f"Input tokens range: [{component_tokens.min():.4f}, {component_tokens.max():.4f}]"
            )

        # Residual connection + layer norm
        component_tokens = self.norm(component_tokens + attn_out)
        if self.debug_mode and torch.isnan(component_tokens).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after layer norm")

        # Pool across components (mean pooling)
        # Shape: [batch, 6, 64] → [batch, 64]
        pooled = component_tokens.mean(dim=1)
        if self.debug_mode and torch.isnan(pooled).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after pooling")

        # Project to output dimension with activation
        output = self.pool_proj(pooled)  # [batch, output_dim]
        if self.debug_mode and torch.isnan(output).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after pool_proj")

        output = self.output_activation(output)
        if self.debug_mode and torch.isnan(output).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after output_activation")

        output = self.output_dropout(output)
        if self.debug_mode and torch.isnan(output).any():
            raise ValueError("[ATTENTIVE_STATE_MLP] NaN after output_dropout")

        return output
