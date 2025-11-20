"""Test suite for enhanced AttentiveStateMLP with adaptive components and hierarchical attention."""

import pytest
import torch
import torch.nn as nn
from npp_rl.models.attentive_state_mlp import AttentiveStateMLP


class TestEnhancedAttentiveStateMLP:
    """Test enhanced AttentiveStateMLP functionality."""

    def test_adaptive_component_dimensions(self):
        """Test that component encoders have correct adaptive dimensions."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Check adaptive component dimensions
        expected_dims = {
            'velocity': 32,      # 3 → 32 (medium complexity)
            'movement': 32,      # 5 → 32 (medium complexity)
            'input': 16,         # 2 → 16 (low complexity)
            'buffers': 24,       # 3 → 24 (medium complexity, timing-critical)
            'contact': 64,       # 6 → 64 (high complexity)
            'forces': 64,        # 7 → 64 (high complexity)
            'temporal': 32,      # 6 → 32 (medium complexity)
            'energy': 48,        # 4 → 48 (NEW: high complexity)
            'surface': 32,       # 4 → 32 (NEW: medium complexity)
        }
        
        for component, expected_dim in expected_dims.items():
            encoder = model.component_encoders[component]
            # Get the output dimension from the first linear layer
            actual_dim = encoder[0].out_features
            assert actual_dim == expected_dim, (
                f"Component {component} should have {expected_dim} output dims, got {actual_dim}"
            )

    def test_40d_state_processing(self):
        """Test that model processes 40D enhanced physics state correctly."""
        batch_size = 8
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Test 40D input (enhanced physics state)
        x = torch.randn(batch_size, 40)
        output = model(x)
        
        assert output.shape == (batch_size, 128), f"Expected output shape ({batch_size}, 128), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_41d_state_with_time_remaining(self):
        """Test that model handles 41D state with time_remaining correctly."""
        batch_size = 4
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Test 41D input (40D enhanced physics + time_remaining)
        x = torch.randn(batch_size, 41)
        output = model(x)
        
        assert output.shape == (batch_size, 128), f"Expected output shape ({batch_size}, 128), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_component_splitting(self):
        """Test that component splitting works correctly for 40D state."""
        batch_size = 2
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Create test state with known values
        x = torch.randn(batch_size, 40)
        
        # Test component splitting
        components = model._split_components(x)
        
        # Check all expected components are present
        expected_components = ['velocity', 'movement', 'input', 'buffers', 'contact', 
                              'forces', 'temporal', 'energy', 'surface']
        assert set(components.keys()) == set(expected_components), (
            f"Missing components: {set(expected_components) - set(components.keys())}"
        )
        
        # Check component dimensions
        expected_shapes = {
            'velocity': (batch_size, 3),    # velocity_mag, velocity_dir_x, velocity_dir_y
            'movement': (batch_size, 5),    # ground/air/wall/special movement states + airborne
            'input': (batch_size, 2),       # horizontal input, jump input
            'buffers': (batch_size, 3),     # jump/floor/wall buffers
            'contact': (batch_size, 6),     # floor/wall/ceiling contact + normals + slope + wall dir
            'forces': (batch_size, 7),      # gravity, walled, normals, drag, friction
            'temporal': (batch_size, 6),    # existing temporal features
            'energy': (batch_size, 4),      # NEW: kinetic, potential, force_mag, energy_change
            'surface': (batch_size, 4),     # NEW: floor_strength, wall_strength, slope, wall_interaction
        }
        
        for component, expected_shape in expected_shapes.items():
            actual_shape = components[component].shape
            assert actual_shape == expected_shape, (
                f"Component {component} should have shape {expected_shape}, got {actual_shape}"
            )

    def test_physics_context_gating(self):
        """Test physics context gating mechanism."""
        batch_size = 4
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Test movement state input for physics gates
        movement_state = torch.randn(batch_size, 5)  # ground/air/wall/special + airborne
        gates = model._compute_physics_gates(movement_state)
        
        # Check gate output shape and range
        assert gates.shape == (batch_size, 9), f"Expected gates shape ({batch_size}, 9), got {gates.shape}"
        assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0), "Gates should be in [0, 1] range (sigmoid output)"

    def test_hierarchical_attention_flow(self):
        """Test that hierarchical attention processes components correctly."""
        batch_size = 3
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4, debug_mode=True)
        
        # Normal input with reasonable values
        x = torch.randn(batch_size, 40) * 0.5  # Scaled to avoid extreme values
        
        # This should complete without NaN validation errors in debug mode
        output = model(x)
        
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()

    def test_uniform_projections(self):
        """Test that all components are projected to uniform dimension correctly."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Check uniform projection dimensions
        for component_name, projection in model.uniform_projections.items():
            input_dim = model.component_dims[component_name]
            assert projection.in_features == input_dim, (
                f"Projection for {component_name} should have input {input_dim}, got {projection.in_features}"
            )
            assert projection.out_features == 64, (  # uniform_dim = 64
                f"Projection for {component_name} should have output 64, got {projection.out_features}"
            )

    def test_attention_head_compatibility(self):
        """Test that uniform dimension is compatible with attention heads."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # uniform_dim should be divisible by num_heads
        assert model.uniform_dim % model.num_heads == 0, (
            f"uniform_dim {model.uniform_dim} should be divisible by num_heads {model.num_heads}"
        )

    def test_stacked_state_handling(self):
        """Test handling of stacked states (frame stacking)."""
        batch_size = 2
        stack_size = 4
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Test stacked input: [batch, stack_size, state_dim]
        x_stacked_3d = torch.randn(batch_size, stack_size, 40)
        output_3d = model(x_stacked_3d)
        
        # Test flattened stacked input: [batch, stack_size * state_dim]
        x_stacked_2d = torch.randn(batch_size, stack_size * 40)
        output_2d = model(x_stacked_2d)
        
        # Both should work and produce same output shape
        assert output_3d.shape == (batch_size, 128)
        assert output_2d.shape == (batch_size, 128)

    def test_gradient_flow(self):
        """Test that gradients flow through all components correctly."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Forward pass
        x = torch.randn(4, 40, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that all component encoders have gradients
        for component_name, encoder in model.component_encoders.items():
            for param in encoder.parameters():
                assert param.grad is not None, f"No gradient for {component_name} encoder"
                assert not torch.all(param.grad == 0), f"Zero gradient for {component_name} encoder"

    def test_physics_gate_network_gradients(self):
        """Test that physics gate network receives gradients."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Forward pass
        x = torch.randn(2, 40, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check physics gate network gradients
        for param in model.physics_gate_network.parameters():
            assert param.grad is not None, "No gradient for physics gate network"
            assert not torch.all(param.grad == 0), "Zero gradient for physics gate network"

    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        batch_sizes = [1, 3, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 40)
            output = model(x)
            assert output.shape == (batch_size, 128), (
                f"Failed for batch_size {batch_size}: expected ({batch_size}, 128), got {output.shape}"
            )

    def test_invalid_input_dimensions(self):
        """Test model behavior with invalid input dimensions."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        # Test with wrong input dimension
        with pytest.raises(ValueError, match="Expected state dim"):
            x = torch.randn(2, 35)  # Wrong dimension
            model(x)

    def test_debug_mode_nan_detection(self):
        """Test that debug mode correctly detects NaN inputs."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4, debug_mode=True)
        
        # Create input with NaN
        x = torch.randn(2, 40)
        x[0, 0] = float('nan')
        
        with pytest.raises(ValueError, match="NaN in input tensor"):
            model(x)

    def test_model_parameters_count(self):
        """Test that model has reasonable number of parameters."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (ballpark: 50k-200k)
        assert 10000 < total_params < 500000, (
            f"Model has {total_params} parameters, which seems unreasonable"
        )

    def test_model_output_deterministic(self):
        """Test that model output is deterministic for same input."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        model.eval()  # Set to eval mode to disable dropout
        
        x = torch.randn(2, 40)
        
        # Multiple forward passes should give same result
        output1 = model(x)
        output2 = model(x)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)
