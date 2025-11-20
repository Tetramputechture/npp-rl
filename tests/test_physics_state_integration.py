"""Integration tests for enhanced physics state system."""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from npp_rl.models.attentive_state_mlp import AttentiveStateMLP
from npp_rl.training.architecture_configs import StateConfig
from npp_rl.feature_extractors.configurable_extractor import StateMLP


class TestPhysicsStateIntegration:
    """Integration tests for enhanced 41D physics state system."""

    def test_state_config_updated_dimensions(self):
        """Test that StateConfig has correct updated dimensions."""
        config = StateConfig()

        assert config.game_state_dim == 41, (
            f"Expected game_state_dim=41, got {config.game_state_dim}"
        )
        assert config.use_attentive_state_mlp == True, (
            "AttentiveStateMLP should be enabled"
        )

    def test_attentive_state_mlp_with_config_dimensions(self):
        """Test AttentiveStateMLP works with StateConfig dimensions."""
        config = StateConfig()

        model = AttentiveStateMLP(
            hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_heads=4
        )

        # Test with config dimensions
        batch_size = 4
        x = torch.randn(batch_size, config.game_state_dim)  # 40D
        output = model(x)

        assert output.shape == (batch_size, config.output_dim)
        assert not torch.isnan(output).any()

    def test_attentive_state_mlp_with_time_remaining(self):
        """Test AttentiveStateMLP with time_remaining feature."""
        config = StateConfig()

        model = AttentiveStateMLP(
            hidden_dim=config.hidden_dim, output_dim=config.output_dim, num_heads=4
        )

        # Test with time_remaining (41D)
        batch_size = 4
        x = torch.randn(batch_size, config.game_state_dim + 1)  # 41D
        output = model(x)

        assert output.shape == (batch_size, config.output_dim)
        assert not torch.isnan(output).any()

    def test_fallback_state_mlp_compatibility(self):
        """Test that fallback StateMLP still works with new dimensions."""
        config = StateConfig()

        # Test fallback StateMLP (for comparison)
        fallback_mlp = StateMLP(
            input_dim=config.game_state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )

        batch_size = 3
        x = torch.randn(batch_size, config.game_state_dim)
        output = fallback_mlp(x)

        assert output.shape == (batch_size, config.output_dim)

    def test_enhanced_physics_state_structure(self):
        """Test that enhanced physics state has correct structure."""
        # Mock a 40D enhanced physics state matching our augmentation
        enhanced_state = np.zeros(40)

        # Verify structure matches our implementation
        # Core movement state (8 features): indices 0-7
        enhanced_state[0] = 0.5  # velocity_mag
        enhanced_state[1] = 0.3  # velocity_dir_x
        enhanced_state[2] = -0.2  # velocity_dir_y
        enhanced_state[3] = 1.0  # ground_movement
        enhanced_state[7] = -1.0  # airborne

        # Input and buffer state (5 features): indices 8-12
        enhanced_state[8] = 0.0  # horizontal_input
        enhanced_state[9] = -1.0  # jump_input

        # Surface contact (6 features): indices 13-18
        enhanced_state[13] = 1.0  # floor_contact

        # Additional physics (7 features): indices 19-25
        enhanced_state[19] = 0.2  # applied_gravity

        # Temporal features (6 features): indices 26-31
        enhanced_state[26] = 0.1  # velocity_change_x

        # NEW: Enhanced physics features (8 features): indices 32-39
        enhanced_state[32] = 0.4  # kinetic_energy
        enhanced_state[33] = 0.6  # potential_energy
        enhanced_state[34] = 0.3  # force_magnitude
        enhanced_state[35] = 0.1  # energy_change_rate
        enhanced_state[36] = 1.0  # floor_contact_strength
        enhanced_state[37] = -1.0  # wall_contact_strength
        enhanced_state[38] = 0.2  # surface_slope
        enhanced_state[39] = 0.0  # wall_interaction

        # Test that AttentiveStateMLP can process this structure
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        x = torch.from_numpy(enhanced_state).unsqueeze(0).float()

        components = model._split_components(x)

        # Verify component extraction
        assert components["velocity"].shape == (1, 3)
        assert components["energy"].shape == (1, 4)  # NEW energy features
        assert components["surface"].shape == (1, 4)  # NEW surface features

        # Verify values are extracted correctly
        assert torch.allclose(components["velocity"][0], torch.tensor([0.5, 0.3, -0.2]))
        assert torch.allclose(
            components["energy"][0], torch.tensor([0.4, 0.6, 0.3, 0.1])
        )
        assert torch.allclose(
            components["surface"][0], torch.tensor([1.0, -1.0, 0.2, 0.0])
        )

    def test_reward_config_physics_discovery_integration(self):
        """Test that reward config properly integrates physics discovery settings."""
        from nclone.gym_environment.reward_calculation.reward_config import RewardConfig

        config = RewardConfig()

        # Test new physics discovery properties
        assert config.enable_physics_discovery == True
        assert config.physics_discovery_weight == 0.6

        # Test reduced PBRS weights
        assert config.pbrs_objective_weight < 2.0  # Should be reduced from original

        # Test that active components include physics discovery
        active_components = config.get_active_components()
        assert "enable_physics_discovery" in active_components
        assert "physics_discovery_weight" in active_components
        assert active_components["enable_physics_discovery"] == True
        assert active_components["physics_discovery_weight"] == 0.6

    def test_main_reward_calculator_integration(self):
        """Test that main reward calculator integrates physics discovery."""
        from nclone.gym_environment.reward_calculation.main_reward_calculator import (
            RewardCalculator,
        )
        from nclone.gym_environment.reward_calculation.reward_config import RewardConfig

        config = RewardConfig()
        calculator = RewardCalculator(reward_config=config)

        # Should have physics discovery system initialized
        assert calculator.physics_discovery is not None
        assert hasattr(calculator.physics_discovery, "calculate_physics_rewards")

    def test_complete_physics_flow_simulation(self):
        """Test complete flow from enhanced state through model and rewards."""
        from nclone.gym_environment.reward_calculation.physics_discovery_rewards import (
            PhysicsDiscoveryRewards,
        )

        # 1. Create enhanced 40D physics state
        batch_size = 2
        enhanced_state = torch.randn(batch_size, 40)

        # 2. Process through AttentiveStateMLP
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)
        model_output = model(enhanced_state)

        assert model_output.shape == (batch_size, 128)
        assert not torch.isnan(model_output).any()

        # 3. Simulate physics discovery rewards calculation
        rewards_system = PhysicsDiscoveryRewards()

        # Create mock observations from enhanced state
        prev_obs = {
            "player_x": 100.0,
            "player_y": 100.0,
            "game_state": enhanced_state[0].numpy().tolist(),
        }

        current_obs = {
            "player_x": 110.0,
            "player_y": 105.0,
            "game_state": enhanced_state[1].numpy().tolist(),
        }

        physics_rewards = rewards_system.calculate_physics_rewards(
            current_obs, prev_obs, action=3
        )

        # Should have all reward components
        assert "efficiency" in physics_rewards
        assert "diversity" in physics_rewards
        assert "utilization" in physics_rewards

        # All should be valid numbers
        for component, value in physics_rewards.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_architecture_consistency(self):
        """Test that all architecture components are consistent with 40D state."""
        from npp_rl.training.architecture_configs import create_attention_config

        # Get attention architecture config
        arch_config = create_attention_config()

        # Should use 40D state
        assert arch_config.state.game_state_dim == 40
        assert arch_config.state.use_attentive_state_mlp == True

        # Test that feature extractor would work with this config
        # (We don't actually create the full feature extractor here to avoid heavy dependencies)
        expected_input_dims = [40, 41]  # With and without time_remaining

        # Mock validation that feature extractor would perform
        from nclone.gym_environment.constants import NINJA_STATE_DIM

        assert NINJA_STATE_DIM == 40
        assert NINJA_STATE_DIM in expected_input_dims
        assert (NINJA_STATE_DIM + 1) in expected_input_dims

    def test_backward_compatibility_removed(self):
        """Test that old 32D state is no longer supported."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

        # Old 32D state should raise error
        with pytest.raises(ValueError, match="Expected state dim"):
            x = torch.randn(2, 32)  # Old dimension
            model(x)

    def test_physics_component_adaptive_sizing(self):
        """Test that different physics components have appropriate sizes."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

        # Check that high-complexity components get more parameters
        contact_dim = model.component_dims["contact"]  # High complexity
        forces_dim = model.component_dims["forces"]  # High complexity
        energy_dim = model.component_dims["energy"]  # High complexity (NEW)

        input_dim = model.component_dims["input"]  # Low complexity
        buffers_dim = model.component_dims["buffers"]  # Medium complexity

        # High complexity should have more dimensions than low complexity
        assert contact_dim > input_dim, (
            f"Contact ({contact_dim}) should be larger than input ({input_dim})"
        )
        assert forces_dim > input_dim, (
            f"Forces ({forces_dim}) should be larger than input ({input_dim})"
        )
        assert energy_dim > input_dim, (
            f"Energy ({energy_dim}) should be larger than input ({input_dim})"
        )

        # Energy should have high dimensions (it's complex derived physics)
        assert energy_dim >= 48, (
            f"Energy component should have at least 48 dims, got {energy_dim}"
        )

    def test_physics_gating_affects_output(self):
        """Test that physics context gating actually affects model output."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

        # Create two states with same physics but different movement states
        base_state = torch.randn(1, 40)

        state_ground = base_state.clone()
        state_ground[0, 3] = 1.0  # ground_movement = 1
        state_ground[0, 4] = -1.0  # air_movement = -1
        state_ground[0, 7] = -1.0  # airborne = False

        state_air = base_state.clone()
        state_air[0, 3] = -1.0  # ground_movement = -1
        state_air[0, 4] = 1.0  # air_movement = 1
        state_air[0, 7] = 1.0  # airborne = True

        # Different movement states should produce different outputs due to gating
        with torch.no_grad():
            output_ground = model(state_ground)
            output_air = model(state_air)

        # Outputs should be different (gating effect)
        assert not torch.allclose(output_ground, output_air, atol=1e-3), (
            "Physics gating should make outputs different for different movement states"
        )

    def test_memory_efficiency(self):
        """Test that enhanced model doesn't use excessive memory."""
        model = AttentiveStateMLP(hidden_dim=128, output_dim=128, num_heads=4)

        # Test with reasonable batch size
        batch_size = 32
        x = torch.randn(batch_size, 40)

        # Should process without memory issues
        output = model(x)
        assert output.shape == (batch_size, 128)

        # Check model size is reasonable
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 1000000, f"Model too large: {total_params} parameters"
