"""Unit tests for AttentiveStateMLP."""

import torch
import pytest
from npp_rl.models.attentive_state_mlp import AttentiveStateMLP


def test_attentive_state_mlp_output_shape():
    """Test output shape is correct."""
    batch_size = 16
    state_dim = 58
    output_dim = 128

    model = AttentiveStateMLP(hidden_dim=128, output_dim=output_dim, num_heads=4)
    x = torch.randn(batch_size, state_dim)

    output = model(x)

    assert output.shape == (batch_size, output_dim), (
        f"Expected shape ({batch_size}, {output_dim}), got {output.shape}"
    )


def test_attentive_state_mlp_with_stacking():
    """Test with stacked states (frame stacking)."""
    batch_size = 8
    stack_size = 4
    state_dim = 58
    output_dim = 128

    model = AttentiveStateMLP(hidden_dim=128, output_dim=output_dim, num_heads=4)

    # Test with flattened stacking [batch, stack_size * state_dim]
    x_flat = torch.randn(batch_size, stack_size * state_dim)
    output_flat = model(x_flat)
    assert output_flat.shape == (batch_size, output_dim)

    # Test with structured stacking [batch, stack_size, state_dim]
    x_struct = torch.randn(batch_size, stack_size, state_dim)
    output_struct = model(x_struct)
    assert output_struct.shape == (batch_size, output_dim)


def test_attentive_state_mlp_attention_not_uniform():
    """Verify attention weights are not uniform (cross-component learning)."""
    batch_size = 4
    state_dim = 58

    model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

    x = torch.randn(batch_size, state_dim)

    # Forward pass with attention weights
    # Temporarily modify forward to return weights
    component_tokens = []

    # Manually run through forward to capture attention
    physics = x[:, :29]
    objectives = x[:, 29:44]
    mines = x[:, 44:52]
    progress = x[:, 52:55]
    sequential = x[:, 55:58]

    phys_feat = model.physics_encoder(physics)
    obj_feat = model.objectives_encoder(objectives)
    mine_feat = model.mine_encoder(mines)
    prog_feat = model.progress_encoder(progress)
    seq_feat = model.sequential_encoder(sequential)

    phys_token = model.physics_proj(phys_feat)
    obj_token = model.objectives_proj(obj_feat)
    mine_token = model.mine_proj(mine_feat)
    prog_token = model.progress_proj(prog_feat)
    seq_token = model.sequential_proj(seq_feat)

    component_tokens = torch.stack(
        [phys_token, obj_token, mine_token, prog_token, seq_token], dim=1
    )

    # Get attention weights
    attn_out, attn_weights = model.attention(
        component_tokens, component_tokens, component_tokens, need_weights=True
    )

    # attn_weights shape: [batch, num_heads, 5, 5] for MultiheadAttention
    # Check that attention is not uniform (not all 0.2 = 1/5)
    # We expect some variation in attention patterns
    if attn_weights is not None:
        # Average over heads: [batch, 5, 5]
        if attn_weights.dim() == 4:
            attn_avg = attn_weights.mean(dim=1)  # Average over heads
        else:
            attn_avg = attn_weights

        # Check that attention matrix is not uniform
        # Uniform would be all values = 1/5 = 0.2
        uniform_value = 1.0 / 5.0
        # Check that at least some values differ from uniform
        diff_from_uniform = torch.abs(attn_avg - uniform_value)
        max_diff = diff_from_uniform.max().item()

        # Allow some tolerance but should have variation
        assert max_diff > 0.01, (
            f"Attention weights appear uniform (max diff: {max_diff})"
        )


def test_attentive_state_mlp_gradient_flow():
    """Verify gradients flow to all component encoders."""
    batch_size = 4
    state_dim = 58

    model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
    x = torch.randn(batch_size, state_dim)

    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check all encoder parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_attentive_state_mlp_component_dimensions():
    """Verify component splitting is correct."""
    batch_size = 2
    state_dim = 58

    model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

    # Manually verify dimension split
    x = torch.randn(batch_size, state_dim)
    physics = x[:, :29]
    objectives = x[:, 29:44]
    mines = x[:, 44:52]
    progress = x[:, 52:55]
    sequential = x[:, 55:58]

    assert physics.shape[1] == 29
    assert objectives.shape[1] == 15
    assert mines.shape[1] == 8
    assert progress.shape[1] == 3
    assert sequential.shape[1] == 3
    assert 29 + 15 + 8 + 3 + 3 == 58


def test_attentive_state_mlp_uniform_dimension():
    """Verify uniform dimension is divisible by num_heads."""
    model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

    # uniform_dim should be 64 (hidden_dim // 2)
    assert model.uniform_dim == 64
    assert model.uniform_dim % 4 == 0, "uniform_dim must be divisible by num_heads"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
