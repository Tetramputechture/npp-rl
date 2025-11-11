"""Attention-based state encoder for enhanced feature learning."""

import torch
import torch.nn as nn
from nclone.gym_environment.constants import (
    NINJA_STATE_DIM,
    PATH_AWARE_OBJECTIVES_DIM,
    MINE_FEATURES_DIM,
    PROGRESS_FEATURES_DIM,
    SEQUENTIAL_GOAL_DIM,
)


class AttentiveStateMLP(nn.Module):
    """State encoder with multi-head attention over state components.

    Separates game state into semantic components (physics, objectives, mines)
    and applies cross-attention to learn inter-component relationships.

    State breakdown (58 dims):
    - Physics (ninja): 29 dims
    - Objectives: 15 dims
    - Mines: 8 dims
    - Progress: 3 dims
    - Sequential: 3 dims

    Architecture:
    1. Component-specific encoders project to hidden_dim
    2. Multi-head attention fuses components
    3. Output projection to output_dim
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.physics_encoder = nn.Sequential(
            nn.Linear(NINJA_STATE_DIM, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.objectives_encoder = nn.Sequential(
            nn.Linear(PATH_AWARE_OBJECTIVES_DIM, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mine_encoder = nn.Sequential(
            nn.Linear(MINE_FEATURES_DIM, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.progress_encoder = nn.Sequential(
            nn.Linear(PROGRESS_FEATURES_DIM, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.sequential_encoder = nn.Sequential(
            nn.Linear(SEQUENTIAL_GOAL_DIM, hidden_dim // 8),
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
        self.physics_proj = nn.Linear(hidden_dim // 2, self.uniform_dim)  # 64 → 64
        self.objectives_proj = nn.Linear(hidden_dim // 4, self.uniform_dim)  # 32 → 64
        self.mine_proj = nn.Linear(hidden_dim // 8, self.uniform_dim)  # 16 → 64
        self.progress_proj = nn.Linear(hidden_dim // 8, self.uniform_dim)  # 16 → 64
        self.sequential_proj = nn.Linear(hidden_dim // 8, self.uniform_dim)  # 16 → 64

        # Multi-head attention across components
        self.attention = nn.MultiheadAttention(
            embed_dim=self.uniform_dim,  # Changed from encoded_dim
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm for uniform dim
        self.norm = nn.LayerNorm(self.uniform_dim)

        # Output: pool 5 components and project
        self.pool_proj = nn.Linear(self.uniform_dim, output_dim)
        self.output_activation = nn.ReLU()
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attentive state encoder.

        Args:
            x: State tensor [batch, 58] or [batch, stack_size * 58] if stacked

        Returns:
            Encoded features [batch, output_dim]
        """
        # Validate input tensor for NaN/Inf
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
            if x.shape[1] % 58 == 0 and x.shape[1] > 58:
                # Stacked states: [batch, stack_size * 58] → take last 58
                x = x[:, -58:]  # Most recent frame
            elif x.shape[1] != 58:
                raise ValueError(
                    f"Expected state dim 58 or multiple of 58, got {x.shape[1]}"
                )
        elif x.dim() == 3:
            # [batch, stack_size, 58] → take last frame
            x = x[:, -1, :]

        # Split into components (must match state dimensions exactly)
        physics = x[:, :29]  # Ninja physics state
        objectives = x[:, 29:44]  # Path-aware objective features (15)
        mines = x[
            :, 44:52
        ]  # Enhanced mine features (8) - MINE FEATURES - critical for NaN diagnosis
        progress = x[:, 52:55]  # Progress tracking (3)
        sequential = x[:, 55:58]  # Sequential goal features (3)

        # Validate mine features specifically (most likely source of NaN)
        if torch.isnan(mines).any():
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN in mine features (indices 44-52). "
                f"Mine values: {mines[torch.isnan(mines).any(dim=1)].cpu().numpy()}"
            )

        # Check for degenerate inputs (all-zero physics or extreme values)
        # Note: Mines, objectives, progress, and sequential can legitimately be all-zero
        # (e.g., levels without mines, no active objectives, start of level, no sequential goals)
        if physics.abs().max() < 1e-8:
            raise ValueError("[ATTENTIVE_STATE_MLP] All-zero physics input")

        # Check for extreme values (potential overflow/underflow issues)
        if physics.abs().max() > 1e6:
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] Extreme physics values: {physics.abs().max()}"
            )
        if mines.abs().max() > 1e6:
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] Extreme mines values: {mines.abs().max()}"
            )

        # Encode each component
        phys_feat = self.physics_encoder(physics)  # [batch, 64]
        if torch.isnan(phys_feat).any():
            nan_mask = torch.isnan(phys_feat)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after physics_encoder in batch indices: {batch_indices.tolist()}. "
                f"Input physics range: [{physics.min():.4f}, {physics.max():.4f}]"
            )

        obj_feat = self.objectives_encoder(objectives)  # [batch, 32]
        if torch.isnan(obj_feat).any():
            nan_mask = torch.isnan(obj_feat)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after objectives_encoder in batch indices: {batch_indices.tolist()}. "
                f"Input objectives range: [{objectives.min():.4f}, {objectives.max():.4f}]"
            )

        mine_feat = self.mine_encoder(mines)  # [batch, 16]
        if torch.isnan(mine_feat).any():
            nan_mask = torch.isnan(mine_feat)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after mine_encoder in batch indices: {batch_indices.tolist()}. "
                f"Input mines range: [{mines.min():.4f}, {mines.max():.4f}]"
            )

        prog_feat = self.progress_encoder(progress)  # [batch, 16]
        if torch.isnan(prog_feat).any():
            nan_mask = torch.isnan(prog_feat)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after progress_encoder in batch indices: {batch_indices.tolist()}. "
                f"Input progress range: [{progress.min():.4f}, {progress.max():.4f}]"
            )

        seq_feat = self.sequential_encoder(sequential)  # [batch, 16]
        if torch.isnan(seq_feat).any():
            nan_mask = torch.isnan(seq_feat)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after sequential_encoder in batch indices: {batch_indices.tolist()}. "
                f"Input sequential range: [{sequential.min():.4f}, {sequential.max():.4f}]"
            )

        # Project to uniform dimension
        phys_token = self.physics_proj(phys_feat)  # [batch, 64]
        if torch.isnan(phys_token).any():
            nan_mask = torch.isnan(phys_token)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after physics_proj in batch indices: {batch_indices.tolist()}"
            )

        obj_token = self.objectives_proj(obj_feat)  # [batch, 64]
        if torch.isnan(obj_token).any():
            nan_mask = torch.isnan(obj_token)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after objectives_proj in batch indices: {batch_indices.tolist()}"
            )

        mine_token = self.mine_proj(mine_feat)  # [batch, 64]
        if torch.isnan(mine_token).any():
            nan_mask = torch.isnan(mine_token)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after mine_proj in batch indices: {batch_indices.tolist()}"
            )

        prog_token = self.progress_proj(prog_feat)  # [batch, 64]
        if torch.isnan(prog_token).any():
            nan_mask = torch.isnan(prog_token)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after progress_proj in batch indices: {batch_indices.tolist()}"
            )

        seq_token = self.sequential_proj(seq_feat)  # [batch, 64]
        if torch.isnan(seq_token).any():
            nan_mask = torch.isnan(seq_token)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after sequential_proj in batch indices: {batch_indices.tolist()}"
            )

        # Stack into component sequence for attention
        # Shape: [batch, 5, 64] - 5 component tokens
        component_tokens = torch.stack(
            [phys_token, obj_token, mine_token, prog_token, seq_token], dim=1
        )
        if torch.isnan(component_tokens).any():
            nan_mask = torch.isnan(component_tokens)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after stacking tokens in batch indices: {batch_indices.tolist()}"
            )

        # Check for degenerate tokens before attention
        token_std = component_tokens.std(dim=-1)
        if (token_std < 1e-8).any():
            degenerate_batches = torch.where((token_std < 1e-8).any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] Degenerate tokens (near-zero std) in batch indices: {degenerate_batches.tolist()}. "
                f"Token std range: [{token_std.min():.4e}, {token_std.max():.4e}]"
            )

        # Multi-head cross-component attention
        # Each component can attend to all others
        # Q, K, V all from component_tokens (self-attention across components)
        attn_out, attn_weights = self.attention(
            component_tokens,
            component_tokens,
            component_tokens,
            need_weights=False,  # Set True for debugging/visualization
        )
        # attn_out shape: [batch, 5, 64]
        if torch.isnan(attn_out).any():
            nan_mask = torch.isnan(attn_out)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after attention in batch indices: {batch_indices.tolist()}. "
                f"Input tokens range: [{component_tokens.min():.4f}, {component_tokens.max():.4f}]"
            )

        # Residual connection + layer norm
        component_tokens = self.norm(component_tokens + attn_out)
        if torch.isnan(component_tokens).any():
            nan_mask = torch.isnan(component_tokens)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after layer norm in batch indices: {batch_indices.tolist()}"
            )

        # Pool across components (mean pooling)
        # Shape: [batch, 5, 64] → [batch, 64]
        pooled = component_tokens.mean(dim=1)
        if torch.isnan(pooled).any():
            nan_mask = torch.isnan(pooled)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after pooling in batch indices: {batch_indices.tolist()}"
            )

        # Project to output dimension with activation
        output = self.pool_proj(pooled)  # [batch, output_dim]
        if torch.isnan(output).any():
            nan_mask = torch.isnan(output)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after pool_proj in batch indices: {batch_indices.tolist()}"
            )

        output = self.output_activation(output)
        if torch.isnan(output).any():
            nan_mask = torch.isnan(output)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after output_activation in batch indices: {batch_indices.tolist()}"
            )

        output = self.output_dropout(output)
        if torch.isnan(output).any():
            nan_mask = torch.isnan(output)
            batch_indices = torch.where(nan_mask.any(dim=1))[0]
            raise ValueError(
                f"[ATTENTIVE_STATE_MLP] NaN after output_dropout in batch indices: {batch_indices.tolist()}"
            )

        return output
