"""
Physics constraint validation for movement sequences.

This module validates movement sequences against N++ physics rules,
ensuring that planned paths are physically feasible given ninja capabilities.
"""

import math
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
from enum import IntEnum

# Import physics constants
try:
    import sys
    import os
    nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone')
    if os.path.exists(nclone_path) and nclone_path not in sys.path:
        sys.path.insert(0, nclone_path)
    from nclone.constants import (
        MAX_HOR_SPEED, GRAVITY_FALL, GRAVITY_JUMP,
        GROUND_ACCEL, AIR_ACCEL, NINJA_RADIUS
    )
    from nclone.graph.graph_builder import EdgeType
except ImportError:
    # Fallback constants
    MAX_HOR_SPEED = 3.333333333333333
    GRAVITY_FALL = 0.06666666666666665
    GRAVITY_JUMP = 0.01111111111111111
    GROUND_ACCEL = 0.06666666666666665
    AIR_ACCEL = 0.04444444444444444
    NINJA_RADIUS = 10
    
    from enum import IntEnum
    class EdgeType(IntEnum):
        WALK = 0
        JUMP = 1
        WALL_SLIDE = 2
        FALL = 3
        ONE_WAY = 4
        FUNCTIONAL = 5

# Physics validation constants
MIN_WALL_JUMP_SPEED = 1.0
MIN_JUMP_VELOCITY = 0.5
MAX_FALL_VELOCITY = 6.0  # Maximum survivable fall velocity
ENERGY_EFFICIENCY = 0.8  # Energy efficiency factor for movement chains
VELOCITY_TOLERANCE = 0.1  # Tolerance for velocity matching
POSITION_TOLERANCE = 5.0  # Tolerance for position matching (pixels)


class ValidationResult(NamedTuple):
    """Result of physics validation."""
    is_valid: bool
    reason: str
    energy_used: float
    final_velocity: Tuple[float, float]
    final_position: Tuple[float, float]


@dataclass
class TrajectoryParams:
    """Parameters describing a movement trajectory."""
    edge_type: EdgeType
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    start_velocity: Tuple[float, float]
    final_velocity: Tuple[float, float]
    time_of_flight: float
    energy_cost: float
    success_probability: float
    requires_jump: bool
    requires_wall_contact: bool


@dataclass
class NinjaPhysicsState:
    """Complete ninja physics state for validation."""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    movement_state: int  # 0-9 from sim_mechanics_doc.md
    ground_contact: bool
    wall_contact: bool
    airborne: bool
    kinetic_energy: float
    potential_energy: float
    can_jump: bool
    can_wall_jump: bool


class PhysicsConstraintValidator:
    """
    Validates movement sequences against N++ physics rules.
    
    This validator ensures that planned movement sequences are physically
    feasible given the ninja's current state and the game's physics constraints.
    """
    
    def __init__(self):
        """Initialize physics constraint validator."""
        self.max_hor_speed = MAX_HOR_SPEED
        self.gravity_fall = GRAVITY_FALL
        self.gravity_jump = GRAVITY_JUMP
        self.ground_accel = GROUND_ACCEL
        self.air_accel = AIR_ACCEL
        self.ninja_radius = NINJA_RADIUS
        
        self.min_wall_jump_speed = MIN_WALL_JUMP_SPEED
        self.min_jump_velocity = MIN_JUMP_VELOCITY
        self.max_fall_velocity = MAX_FALL_VELOCITY
        self.energy_efficiency = ENERGY_EFFICIENCY
        self.velocity_tolerance = VELOCITY_TOLERANCE
        self.position_tolerance = POSITION_TOLERANCE
    
    def validate_movement_sequence(
        self,
        movement_chain: List[TrajectoryParams],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """
        Check if a sequence of movements is physically possible.
        
        Args:
            movement_chain: List of trajectory parameters for each movement
            ninja_state: Current ninja physics state
            
        Returns:
            ValidationResult with feasibility assessment and final state
        """
        if not movement_chain:
            return ValidationResult(
                is_valid=True,
                reason="Empty movement chain",
                energy_used=0.0,
                final_velocity=ninja_state.velocity,
                final_position=ninja_state.position
            )
        
        # Initialize simulation state
        current_velocity = ninja_state.velocity
        current_position = ninja_state.position
        total_energy_used = 0.0
        available_energy = self.calculate_available_energy(ninja_state)
        
        # Validate each movement in sequence
        for i, movement in enumerate(movement_chain):
            # Check energy requirements
            if movement.energy_cost > available_energy:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Insufficient energy at step {i}: need {movement.energy_cost:.2f}, have {available_energy:.2f}",
                    energy_used=total_energy_used,
                    final_velocity=current_velocity,
                    final_position=current_position
                )
            
            # Validate individual movement
            movement_result = self._validate_single_movement(
                movement, current_position, current_velocity, ninja_state
            )
            
            if not movement_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Invalid movement at step {i}: {movement_result.reason}",
                    energy_used=total_energy_used,
                    final_velocity=current_velocity,
                    final_position=current_position
                )
            
            # Update state for next movement
            current_velocity = movement.final_velocity
            current_position = movement.end_position
            total_energy_used += movement.energy_cost
            available_energy -= movement.energy_cost
            
            # Apply energy efficiency loss for chained movements
            available_energy *= self.energy_efficiency
        
        return ValidationResult(
            is_valid=True,
            reason="Valid movement sequence",
            energy_used=total_energy_used,
            final_velocity=current_velocity,
            final_position=current_position
        )
    
    def _validate_single_movement(
        self,
        movement: TrajectoryParams,
        current_position: Tuple[float, float],
        current_velocity: Tuple[float, float],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """Validate a single movement against physics constraints."""
        
        # Check position continuity
        pos_error = math.sqrt(
            (movement.start_position[0] - current_position[0])**2 +
            (movement.start_position[1] - current_position[1])**2
        )
        if pos_error > self.position_tolerance:
            return ValidationResult(
                is_valid=False,
                reason=f"Position discontinuity: {pos_error:.1f} pixels",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=current_position
            )
        
        # Check velocity constraints based on movement type
        if movement.edge_type == EdgeType.JUMP:
            return self._validate_jump_movement(movement, current_velocity, ninja_state)
        elif movement.edge_type == EdgeType.WALL_SLIDE:
            return self._validate_wall_slide_movement(movement, current_velocity, ninja_state)
        elif movement.edge_type == EdgeType.FALL:
            return self._validate_fall_movement(movement, current_velocity, ninja_state)
        elif movement.edge_type == EdgeType.WALK:
            return self._validate_walk_movement(movement, current_velocity, ninja_state)
        else:
            # For other edge types (ONE_WAY, FUNCTIONAL), assume valid if basic checks pass
            return ValidationResult(
                is_valid=True,
                reason="Basic movement type",
                energy_used=movement.energy_cost,
                final_velocity=movement.final_velocity,
                final_position=movement.end_position
            )
    
    def _validate_jump_movement(
        self,
        movement: TrajectoryParams,
        current_velocity: Tuple[float, float],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """Validate jump movement physics."""
        
        # Check jump capability
        if not ninja_state.can_jump:
            return ValidationResult(
                is_valid=False,
                reason="Cannot jump in current state",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        # Check minimum velocity for jump initiation
        vel_magnitude = math.sqrt(current_velocity[0]**2 + current_velocity[1]**2)
        if vel_magnitude < self.min_jump_velocity:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient velocity for jump: {vel_magnitude:.2f} < {self.min_jump_velocity}",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        # Validate trajectory physics
        height_diff = movement.end_position[1] - movement.start_position[1]
        if height_diff > 0:  # Upward jump
            # Check if jump height is achievable
            max_jump_height = self._calculate_max_jump_height(current_velocity[1])
            if height_diff > max_jump_height:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Jump too high: {height_diff:.1f} > {max_jump_height:.1f}",
                    energy_used=0.0,
                    final_velocity=current_velocity,
                    final_position=ninja_state.position
                )
        
        return ValidationResult(
            is_valid=True,
            reason="Valid jump movement",
            energy_used=movement.energy_cost,
            final_velocity=movement.final_velocity,
            final_position=movement.end_position
        )
    
    def _validate_wall_slide_movement(
        self,
        movement: TrajectoryParams,
        current_velocity: Tuple[float, float],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """Validate wall slide movement physics."""
        
        # Check wall contact requirement
        if not ninja_state.wall_contact and movement.requires_wall_contact:
            return ValidationResult(
                is_valid=False,
                reason="No wall contact for wall slide",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        # Wall slides require some horizontal velocity to maintain contact
        if abs(current_velocity[0]) < self.min_wall_jump_speed * 0.5:
            return ValidationResult(
                is_valid=False,
                reason=f"Insufficient horizontal velocity for wall slide: {abs(current_velocity[0]):.2f}",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        return ValidationResult(
            is_valid=True,
            reason="Valid wall slide movement",
            energy_used=movement.energy_cost,
            final_velocity=movement.final_velocity,
            final_position=movement.end_position
        )
    
    def _validate_fall_movement(
        self,
        movement: TrajectoryParams,
        current_velocity: Tuple[float, float],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """Validate fall movement physics."""
        
        # Check fall velocity limits for survivability
        final_fall_speed = abs(movement.final_velocity[1])
        if final_fall_speed > self.max_fall_velocity:
            return ValidationResult(
                is_valid=False,
                reason=f"Fall velocity too high: {final_fall_speed:.2f} > {self.max_fall_velocity}",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        # Validate gravity-based acceleration
        height_diff = movement.end_position[1] - movement.start_position[1]
        if height_diff < 0:  # Downward fall
            expected_final_velocity = self._calculate_fall_velocity(
                current_velocity[1], abs(height_diff)
            )
            velocity_error = abs(movement.final_velocity[1] - expected_final_velocity)
            if velocity_error > self.velocity_tolerance:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Fall velocity mismatch: expected {expected_final_velocity:.2f}, got {movement.final_velocity[1]:.2f}",
                    energy_used=0.0,
                    final_velocity=current_velocity,
                    final_position=ninja_state.position
                )
        
        return ValidationResult(
            is_valid=True,
            reason="Valid fall movement",
            energy_used=movement.energy_cost,
            final_velocity=movement.final_velocity,
            final_position=movement.end_position
        )
    
    def _validate_walk_movement(
        self,
        movement: TrajectoryParams,
        current_velocity: Tuple[float, float],
        ninja_state: NinjaPhysicsState
    ) -> ValidationResult:
        """Validate walk movement physics."""
        
        # Walking requires ground contact
        if not ninja_state.ground_contact:
            return ValidationResult(
                is_valid=False,
                reason="No ground contact for walking",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        # Check horizontal speed limits
        final_horizontal_speed = abs(movement.final_velocity[0])
        if final_horizontal_speed > self.max_hor_speed:
            return ValidationResult(
                is_valid=False,
                reason=f"Horizontal speed too high: {final_horizontal_speed:.2f} > {self.max_hor_speed}",
                energy_used=0.0,
                final_velocity=current_velocity,
                final_position=ninja_state.position
            )
        
        return ValidationResult(
            is_valid=True,
            reason="Valid walk movement",
            energy_used=movement.energy_cost,
            final_velocity=movement.final_velocity,
            final_position=movement.end_position
        )
    
    def calculate_available_energy(self, ninja_state: NinjaPhysicsState) -> float:
        """
        Calculate available energy for movement based on ninja state.
        
        Args:
            ninja_state: Current ninja physics state
            
        Returns:
            Available energy for movement
        """
        # Base energy from kinetic energy
        base_energy = ninja_state.kinetic_energy
        
        # Additional energy from potential energy (height advantage)
        potential_bonus = ninja_state.potential_energy * 0.5
        
        # Contact bonuses
        ground_bonus = 1.0 if ninja_state.ground_contact else 0.0
        wall_bonus = 0.5 if ninja_state.wall_contact else 0.0
        
        # Movement state bonus (higher states have more energy available)
        state_bonus = ninja_state.movement_state / 9.0
        
        total_energy = base_energy + potential_bonus + ground_bonus + wall_bonus + state_bonus
        
        return max(total_energy, 0.1)  # Minimum energy threshold
    
    def _calculate_max_jump_height(self, initial_vertical_velocity: float) -> float:
        """Calculate maximum achievable jump height given initial velocity."""
        # Use kinematic equation: v² = u² + 2as
        # At max height, final velocity = 0
        # height = u² / (2 * gravity)
        if initial_vertical_velocity >= 0:
            return 0.0  # Already moving upward or stationary
        
        return abs(initial_vertical_velocity)**2 / (2 * self.gravity_jump)
    
    def _calculate_fall_velocity(self, initial_velocity: float, fall_distance: float) -> float:
        """Calculate final velocity after falling a given distance."""
        # Use kinematic equation: v² = u² + 2as
        return math.sqrt(initial_velocity**2 + 2 * self.gravity_fall * fall_distance)


def create_physics_validator() -> PhysicsConstraintValidator:
    """
    Create a physics constraint validator with default parameters.
    
    Returns:
        Configured PhysicsConstraintValidator instance
    """
    return PhysicsConstraintValidator()