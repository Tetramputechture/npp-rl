"""
Unit tests for conditional edge activation system.

Tests the ConditionalEdgeMasker and PhysicsConstraintValidator components
to ensure proper physics-aware edge filtering.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from npp_rl.models.conditional_edges import ConditionalEdgeMasker, create_conditional_edge_masker
from npp_rl.models.physics_constraints import (
    PhysicsConstraintValidator, NinjaPhysicsState, TrajectoryParams,
    ValidationResult, create_physics_validator
)

# Import EdgeType for testing
from nclone.graph.graph_builder import EdgeType


class TestConditionalEdgeMasker:
    """Test cases for ConditionalEdgeMasker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.masker = create_conditional_edge_masker()
        
        # Create sample edge features [num_edges, 16]
        # Structure: [edge_type(6), direction(2), cost(1), trajectory_params(3), physics_constraints(2), requirements(2)]
        self.num_edges = 10
        self.edge_features = torch.zeros(self.num_edges, 16)
        
        # Set up different edge types
        self.edge_features[0, EdgeType.WALK] = 1.0  # Walk edge
        self.edge_features[1, EdgeType.JUMP] = 1.0  # Jump edge
        self.edge_features[2, EdgeType.WALL_SLIDE] = 1.0  # Wall slide edge
        self.edge_features[3, EdgeType.FALL] = 1.0  # Fall edge
        self.edge_features[4, EdgeType.JUMP] = 1.0  # Another jump edge
        
        # Set trajectory requirements
        self.edge_features[1, 14] = 1.0  # requires_jump
        self.edge_features[2, 15] = 1.0  # requires_wall_contact
        self.edge_features[4, 14] = 1.0  # requires_jump
        
        # Set velocity requirements
        self.edge_features[1, 12] = 0.5  # min_velocity for jump
        self.edge_features[2, 12] = 0.3  # min_velocity for wall slide
        self.edge_features[4, 12] = 1.2  # high min_velocity for difficult jump
        
        # Base edge mask (all edges initially valid)
        self.base_edge_mask = torch.ones(self.num_edges)
        
        # Sample ninja physics state [18]
        self.ninja_physics_state = torch.zeros(18)
        self.ninja_physics_state[0] = 0.8  # vx (normalized)
        self.ninja_physics_state[1] = -0.3  # vy (normalized)
        self.ninja_physics_state[2] = 0.85  # vel_magnitude
        self.ninja_physics_state[3] = 0.33  # movement_state (3/9 = jumping)
        self.ninja_physics_state[4] = 0.0  # ground_contact
        self.ninja_physics_state[5] = 0.0  # wall_contact
        self.ninja_physics_state[6] = 1.0  # airborne
        self.ninja_physics_state[9] = 0.7  # kinetic_energy
        self.ninja_physics_state[16] = 1.0  # can_jump
        self.ninja_physics_state[17] = 0.0  # can_wall_jump
    
    def test_masker_initialization(self):
        """Test that masker initializes correctly."""
        assert isinstance(self.masker, ConditionalEdgeMasker)
        assert self.masker.min_wall_jump_speed > 0
        assert self.masker.min_jump_energy > 0
    
    def test_basic_edge_masking(self):
        """Test basic edge masking functionality."""
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            self.edge_features, self.ninja_physics_state, self.base_edge_mask
        )
        
        assert dynamic_mask.shape == self.base_edge_mask.shape
        assert torch.all(dynamic_mask <= self.base_edge_mask)  # Dynamic mask should be subset
    
    def test_jump_capability_filtering(self):
        """Test that jump edges are filtered based on jump capability."""
        # Disable jump capability
        ninja_state_no_jump = self.ninja_physics_state.clone()
        ninja_state_no_jump[16] = 0.0  # can_jump = False
        
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            self.edge_features, ninja_state_no_jump, self.base_edge_mask
        )
        
        # Jump edges should be disabled
        assert dynamic_mask[1] == 0.0  # Jump edge with requires_jump
        assert dynamic_mask[4] == 0.0  # Another jump edge with requires_jump
        
        # Non-jump edges should remain enabled
        assert dynamic_mask[0] == 1.0  # Walk edge
        assert dynamic_mask[3] == 1.0  # Fall edge
    
    def test_wall_contact_filtering(self):
        """Test that wall slide edges are filtered based on wall contact."""
        # Enable wall contact
        ninja_state_wall = self.ninja_physics_state.clone()
        ninja_state_wall[5] = 1.0  # wall_contact = True
        
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            self.edge_features, ninja_state_wall, self.base_edge_mask
        )
        
        # Wall slide edge should be enabled with wall contact
        assert dynamic_mask[2] == 1.0
        
        # Test without wall contact
        ninja_state_no_wall = self.ninja_physics_state.clone()
        ninja_state_no_wall[5] = 0.0  # wall_contact = False
        
        dynamic_mask_no_wall = self.masker.compute_dynamic_edge_mask(
            self.edge_features, ninja_state_no_wall, self.base_edge_mask
        )
        
        # Wall slide edge should be disabled without wall contact
        assert dynamic_mask_no_wall[2] == 0.0
    
    def test_velocity_requirements(self):
        """Test that edges are filtered based on velocity requirements."""
        # Low velocity state
        ninja_state_slow = self.ninja_physics_state.clone()
        ninja_state_slow[2] = 0.2  # vel_magnitude = 0.2
        
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            self.edge_features, ninja_state_slow, self.base_edge_mask
        )
        
        # High velocity requirement edge should be disabled
        assert dynamic_mask[4] == 0.0  # Edge with min_velocity = 1.2
        
        # Edge with min_velocity = 0.5 should also be disabled (ninja only has 0.2 velocity)
        assert dynamic_mask[1] == 0.0  # Edge with min_velocity = 0.5, ninja has 0.2
        
        # Test with sufficient velocity
        ninja_state_fast = self.ninja_physics_state.clone()
        ninja_state_fast[2] = 0.8  # vel_magnitude = 0.8 (sufficient for edge 1)
        
        dynamic_mask_fast = self.masker.compute_dynamic_edge_mask(
            self.edge_features, ninja_state_fast, self.base_edge_mask
        )
        
        # Edge 1 should now be enabled (has sufficient velocity and can jump)
        assert dynamic_mask_fast[1] == 1.0  # Edge with min_velocity = 0.5, ninja has 0.8
        # Edge 4 should still be disabled (requires 1.2 velocity, ninja has 0.8)
        assert dynamic_mask_fast[4] == 0.0
    
    def test_batched_input(self):
        """Test masker with batched input."""
        batch_size = 3
        batched_edge_features = self.edge_features.unsqueeze(0).repeat(batch_size, 1, 1)
        batched_base_mask = self.base_edge_mask.unsqueeze(0).repeat(batch_size, 1)
        batched_physics_state = self.ninja_physics_state.unsqueeze(0).repeat(batch_size, 1)
        
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            batched_edge_features, batched_physics_state, batched_base_mask
        )
        
        assert dynamic_mask.shape == (batch_size, self.num_edges)
        assert torch.all(dynamic_mask <= batched_base_mask)
    
    def test_constraint_summary(self):
        """Test constraint summary generation."""
        summary = self.masker.get_constraint_summary(
            self.edge_features, self.ninja_physics_state, self.base_edge_mask
        )
        
        assert isinstance(summary, dict)
        assert 'base_edges' in summary
        assert 'dynamic_edges' in summary
        assert 'disabled_edges' in summary
        assert 'disable_rate' in summary
        assert 'ninja_state' in summary
        
        assert summary['base_edges'] >= summary['dynamic_edges']
        assert summary['disabled_edges'] >= 0
    
    def test_edge_features_compatibility(self):
        """Test compatibility with different edge feature dimensions."""
        # Test with minimal edge features (older format)
        minimal_edge_features = torch.zeros(self.num_edges, 9)
        minimal_edge_features[0, EdgeType.WALK] = 1.0
        minimal_edge_features[1, EdgeType.JUMP] = 1.0
        
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            minimal_edge_features, self.ninja_physics_state, self.base_edge_mask
        )
        
        assert dynamic_mask.shape == self.base_edge_mask.shape
        # Should work without trajectory requirements
    
    def test_insufficient_physics_state(self):
        """Test handling of insufficient physics state features."""
        short_physics_state = torch.zeros(10)  # Less than required 18
        
        # Should not crash and return base mask
        dynamic_mask = self.masker.compute_dynamic_edge_mask(
            self.edge_features, short_physics_state, self.base_edge_mask
        )
        
        assert torch.equal(dynamic_mask, self.base_edge_mask)


class TestPhysicsConstraintValidator:
    """Test cases for PhysicsConstraintValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = create_physics_validator()
        
        # Create sample ninja state
        self.ninja_state = NinjaPhysicsState(
            position=(100.0, 200.0),
            velocity=(1.0, -0.5),
            movement_state=3,  # Jumping
            ground_contact=False,
            wall_contact=False,
            airborne=True,
            kinetic_energy=1.5,
            potential_energy=0.8,
            can_jump=True,
            can_wall_jump=False
        )
        
        # Create sample trajectory
        self.jump_trajectory = TrajectoryParams(
            edge_type=EdgeType.JUMP,
            start_position=(100.0, 200.0),
            end_position=(120.0, 180.0),
            start_velocity=(1.0, -0.5),
            final_velocity=(1.2, 0.3),
            time_of_flight=2.0,
            energy_cost=0.8,
            success_probability=0.9,
            requires_jump=True,
            requires_wall_contact=False
        )
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        assert isinstance(self.validator, PhysicsConstraintValidator)
        assert self.validator.max_hor_speed > 0
        assert self.validator.gravity_fall > 0
    
    def test_empty_movement_chain(self):
        """Test validation of empty movement chain."""
        result = self.validator.validate_movement_sequence([], self.ninja_state)
        
        assert result.is_valid
        assert result.energy_used == 0.0
        assert result.final_velocity == self.ninja_state.velocity
        assert result.final_position == self.ninja_state.position
    
    def test_valid_single_movement(self):
        """Test validation of valid single movement."""
        result = self.validator.validate_movement_sequence(
            [self.jump_trajectory], self.ninja_state
        )
        
        assert result.is_valid
        assert result.energy_used == self.jump_trajectory.energy_cost
        assert result.final_velocity == self.jump_trajectory.final_velocity
        assert result.final_position == self.jump_trajectory.end_position
    
    def test_insufficient_energy(self):
        """Test validation failure due to insufficient energy."""
        # Create high-cost trajectory
        expensive_trajectory = TrajectoryParams(
            edge_type=EdgeType.JUMP,
            start_position=(100.0, 200.0),
            end_position=(150.0, 150.0),
            start_velocity=(1.0, -0.5),
            final_velocity=(2.0, 1.0),
            time_of_flight=3.0,
            energy_cost=10.0,  # Very high cost
            success_probability=0.5,
            requires_jump=True,
            requires_wall_contact=False
        )
        
        result = self.validator.validate_movement_sequence(
            [expensive_trajectory], self.ninja_state
        )
        
        assert not result.is_valid
        assert "Insufficient energy" in result.reason
    
    def test_jump_without_capability(self):
        """Test validation failure for jump without capability."""
        # Ninja state without jump capability
        no_jump_state = NinjaPhysicsState(
            position=(100.0, 200.0),
            velocity=(1.0, -0.5),
            movement_state=1,  # Running
            ground_contact=True,
            wall_contact=False,
            airborne=False,
            kinetic_energy=1.0,
            potential_energy=0.5,
            can_jump=False,  # Cannot jump
            can_wall_jump=False
        )
        
        result = self.validator.validate_movement_sequence(
            [self.jump_trajectory], no_jump_state
        )
        
        assert not result.is_valid
        assert "Cannot jump" in result.reason
    
    def test_position_discontinuity(self):
        """Test validation failure due to position discontinuity."""
        # Trajectory with wrong start position
        discontinuous_trajectory = TrajectoryParams(
            edge_type=EdgeType.WALK,
            start_position=(200.0, 300.0),  # Far from ninja position
            end_position=(220.0, 300.0),
            start_velocity=(1.0, 0.0),
            final_velocity=(1.0, 0.0),
            time_of_flight=1.0,
            energy_cost=0.5,
            success_probability=1.0,
            requires_jump=False,
            requires_wall_contact=False
        )
        
        result = self.validator.validate_movement_sequence(
            [discontinuous_trajectory], self.ninja_state
        )
        
        assert not result.is_valid
        assert "Position discontinuity" in result.reason
    
    def test_wall_slide_validation(self):
        """Test validation of wall slide movements."""
        wall_slide_trajectory = TrajectoryParams(
            edge_type=EdgeType.WALL_SLIDE,
            start_position=(100.0, 200.0),
            end_position=(100.0, 220.0),
            start_velocity=(1.5, 0.5),
            final_velocity=(1.2, 1.0),
            time_of_flight=1.5,
            energy_cost=0.6,
            success_probability=0.8,
            requires_jump=False,
            requires_wall_contact=True
        )
        
        # Test with wall contact
        wall_state = NinjaPhysicsState(
            position=(100.0, 200.0),
            velocity=(1.5, 0.5),
            movement_state=5,  # Wall sliding
            ground_contact=False,
            wall_contact=True,
            airborne=False,
            kinetic_energy=1.2,
            potential_energy=0.6,
            can_jump=False,
            can_wall_jump=True
        )
        
        result = self.validator.validate_movement_sequence(
            [wall_slide_trajectory], wall_state
        )
        
        assert result.is_valid
    
    def test_movement_chain_validation(self):
        """Test validation of movement chains."""
        # Create a ninja state suitable for walking (needs ground contact)
        walking_ninja_state = NinjaPhysicsState(
            position=(100.0, 200.0),
            velocity=(1.0, -0.5),
            movement_state=1,  # Running (suitable for walking)
            ground_contact=True,  # Required for walking
            wall_contact=False,
            airborne=False,
            kinetic_energy=1.5,
            potential_energy=0.8,
            can_jump=True,
            can_wall_jump=False
        )
        
        # Create a chain of movements
        walk_trajectory = TrajectoryParams(
            edge_type=EdgeType.WALK,
            start_position=(100.0, 200.0),
            end_position=(120.0, 200.0),
            start_velocity=(1.0, -0.5),
            final_velocity=(1.5, 0.0),
            time_of_flight=1.0,
            energy_cost=0.3,
            success_probability=1.0,
            requires_jump=False,
            requires_wall_contact=False
        )
        
        jump_trajectory = TrajectoryParams(
            edge_type=EdgeType.JUMP,
            start_position=(120.0, 200.0),
            end_position=(140.0, 180.0),
            start_velocity=(1.5, 0.0),
            final_velocity=(1.8, 0.5),
            time_of_flight=2.0,
            energy_cost=0.7,
            success_probability=0.9,
            requires_jump=True,
            requires_wall_contact=False
        )
        
        movement_chain = [walk_trajectory, jump_trajectory]
        
        result = self.validator.validate_movement_sequence(movement_chain, walking_ninja_state)
        
        assert result.is_valid
        assert result.energy_used == 1.0  # 0.3 + 0.7
        assert result.final_position == jump_trajectory.end_position
    
    def test_energy_calculation(self):
        """Test available energy calculation."""
        energy = self.validator.calculate_available_energy(self.ninja_state)
        
        assert energy > 0
        assert energy >= self.ninja_state.kinetic_energy  # Should include kinetic energy
    
    def test_fall_velocity_calculation(self):
        """Test fall velocity calculation."""
        initial_velocity = 1.0
        fall_distance = 50.0
        
        final_velocity = self.validator._calculate_fall_velocity(initial_velocity, fall_distance)
        
        assert final_velocity > initial_velocity  # Should increase due to gravity
        assert final_velocity > 0
    
    def test_max_jump_height_calculation(self):
        """Test maximum jump height calculation."""
        initial_upward_velocity = -2.0  # Negative is upward
        
        max_height = self.validator._calculate_max_jump_height(initial_upward_velocity)
        
        assert max_height > 0
        
        # Test with downward velocity
        downward_velocity = 1.0
        max_height_down = self.validator._calculate_max_jump_height(downward_velocity)
        assert max_height_down == 0.0


if __name__ == '__main__':
    pytest.main([__file__])