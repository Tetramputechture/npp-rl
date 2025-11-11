"""Unit tests for attention-based components."""
import torch
import pytest
from npp_rl.models.attentive_state_mlp import AttentiveStateMLP


def test_attentive_state_mlp_forward():
    """Test forward pass with 58-dim input."""
    model = AttentiveStateMLP(128, 128, 4)
    x = torch.randn(8, 58)
    out = model(x)
    assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"


def test_attentive_state_mlp_stacked():
    """Test with stacked states (4 frames)."""
    model = AttentiveStateMLP(128, 128, 4)
    x = torch.randn(8, 232)  # 4 * 58
    out = model(x)
    assert out.shape == (8, 128)


def test_gradient_flow():
    """Test gradients flow through attention."""
    model = AttentiveStateMLP(128, 128, 4)
    x = torch.randn(8, 58, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    test_attentive_state_mlp_forward()
    test_attentive_state_mlp_stacked()
    test_gradient_flow()
    print("All tests passed!")

