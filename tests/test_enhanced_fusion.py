"""Unit tests for MultiHeadFusion."""

import torch
import pytest
import importlib.util
from pathlib import Path

# Import directly from file to avoid circular import issues
spec = importlib.util.spec_from_file_location(
    "configurable_extractor",
    Path(__file__).parent.parent
    / "npp_rl"
    / "feature_extractors"
    / "configurable_extractor.py",
)
configurable_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configurable_extractor_module)

MultiHeadFusion = configurable_extractor_module.MultiHeadFusion


def test_enhanced_fusion_output_shape():
    """Test output shape."""
    batch_size = 8
    modality_dims = [
        256,
        128,
        256,
        128,
        64,
    ]  # 5 modalities (note: reachability is 64, not 128)
    input_dim = sum(modality_dims)  # 832
    output_dim = 512

    fusion = MultiHeadFusion(
        input_dim,
        output_dim,
        num_heads=8,
        modality_dims=modality_dims,
    )

    x = torch.randn(batch_size, input_dim)
    output = fusion(x)

    assert output.shape == (batch_size, output_dim)


def test_enhanced_fusion_all_embeddings_used():
    """Verify all modality embeddings receive gradients."""
    batch_size = 4
    modality_dims = [256, 128, 256, 128, 64]
    input_dim = sum(modality_dims)
    output_dim = 512

    fusion = MultiHeadFusion(
        input_dim,
        output_dim,
        num_heads=8,
        modality_dims=modality_dims,
    )

    x = torch.randn(batch_size, input_dim)
    output = fusion(x)
    loss = output.sum()
    loss.backward()

    # Check all modality embeddings have gradients
    assert fusion.modality_embeddings.grad is not None

    # Check each embedding (not just first) has non-zero gradient
    for i in range(len(modality_dims)):
        emb_grad = fusion.modality_embeddings.grad[i]
        assert not torch.all(emb_grad == 0), f"Modality embedding {i} has zero gradient"


def test_enhanced_fusion_variable_modalities():
    """Test with different numbers of modalities (ablation)."""
    batch_size = 4
    output_dim = 512

    # Test with 3 modalities (e.g., no vision)
    modality_dims_3 = [256, 128, 64]  # graph, state, reachability
    input_dim_3 = sum(modality_dims_3)

    fusion_3 = MultiHeadFusion(
        input_dim_3,
        output_dim,
        num_heads=8,
        modality_dims=modality_dims_3,
    )

    x_3 = torch.randn(batch_size, input_dim_3)
    output_3 = fusion_3(x_3)
    assert output_3.shape == (batch_size, output_dim)

    # Test with 5 modalities (all)
    modality_dims_5 = [256, 128, 256, 128, 64]
    input_dim_5 = sum(modality_dims_5)

    fusion_5 = MultiHeadFusion(
        input_dim_5,
        output_dim,
        num_heads=8,
        modality_dims=modality_dims_5,
    )

    x_5 = torch.randn(batch_size, input_dim_5)
    output_5 = fusion_5(x_5)
    assert output_5.shape == (batch_size, output_dim)


def test_enhanced_fusion_cross_modal_attention():
    """Verify cross-modal attention is working (not uniform)."""
    batch_size = 2
    modality_dims = [256, 128, 256, 128, 64]
    input_dim = sum(modality_dims)
    output_dim = 512

    fusion = MultiHeadFusion(
        input_dim,
        output_dim,
        num_heads=8,
        modality_dims=modality_dims,
    )

    x = torch.randn(batch_size, input_dim)

    # Forward pass with attention weights
    modality_features = []
    start_idx = 0
    for modality_dim in modality_dims:
        end_idx = start_idx + modality_dim
        modality_feat = x[:, start_idx:end_idx]
        modality_features.append(modality_feat)
        start_idx = end_idx

    uniform_features = []
    for feat, proj in zip(modality_features, fusion.modality_projections):
        uniform_feat = proj(feat)
        uniform_features.append(uniform_feat)

    modality_tokens = torch.stack(uniform_features, dim=1)
    modality_tokens = modality_tokens + fusion.modality_embeddings.unsqueeze(0)

    # Get attention weights
    attn_out, attn_weights = fusion.attention(
        modality_tokens, modality_tokens, modality_tokens, need_weights=True
    )

    # Check attention weights shape and non-uniformity
    if attn_weights is not None:
        # attn_weights shape: [batch, num_heads, num_modalities, num_modalities]
        if attn_weights.dim() == 4:
            # Average over heads: [batch, num_modalities, num_modalities]
            attn_avg = attn_weights.mean(dim=1)
        else:
            attn_avg = attn_weights

        # Check that attention is not uniform (not all 1/num_modalities)
        uniform_value = 1.0 / len(modality_dims)
        diff_from_uniform = torch.abs(attn_avg - uniform_value)
        max_diff = diff_from_uniform.max().item()

        # Allow some tolerance but should have variation
        assert max_diff > 0.01, (
            f"Attention weights appear uniform (max diff: {max_diff})"
        )


def test_enhanced_fusion_backward_compatibility():
    """Test backward compatibility with equal-sized modalities."""
    batch_size = 4
    input_dim = 500  # Divisible by 5
    output_dim = 512

    # Should work without modality_dims (assumes equal split)
    fusion = MultiHeadFusion(
        input_dim,
        output_dim,
        num_heads=8,
        modality_dims=None,  # Will use equal split
    )

    x = torch.randn(batch_size, input_dim)
    output = fusion(x)
    assert output.shape == (batch_size, output_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
