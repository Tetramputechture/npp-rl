"""
Conditional edge activation system for physics-aware graph neural networks.

This module implements dynamic edge masking based on ninja's current physics state
and movement capabilities, enabling more realistic pathfinding decisions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

# Import EdgeType from nclone graph builder
from nclone.graph.graph_builder import EdgeType

# Physics constraint constants
MIN_WALL_JUMP_SPEED = 1.0  # Minimum horizontal speed for wall jumps
MIN_JUMP_ENERGY = 0.5      # Minimum energy for upward jumps
MIN_VELOCITY_THRESHOLD = 0.1  # Minimum velocity for movement-dependent edges
PHYSICS_STATE_THRESHOLD = 0.5  # Threshold for binary physics state flags


class ConditionalEdgeMasker(nn.Module):
    """
    Dynamic edge masker that filters edges based on ninja's current physics state.
    
    This module computes which edges are physically feasible given the ninja's
    current velocity, movement state, and contact conditions.
    """
    
    def __init__(self):
        """Initialize conditional edge masker with physics constraints."""
        super().__init__()
        
        self.min_wall_jump_speed = MIN_WALL_JUMP_SPEED
        self.min_jump_energy = MIN_JUMP_ENERGY
        self.min_velocity_threshold = MIN_VELOCITY_THRESHOLD
        self.physics_threshold = PHYSICS_STATE_THRESHOLD
        
        # Register buffers for device compatibility
        self.register_buffer('_device_check', torch.tensor(0.0))
    
    def compute_dynamic_edge_mask(
        self,
        edge_features: torch.Tensor,
        ninja_physics_state: torch.Tensor,
        base_edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute which edges are available based on current ninja state.
        
        Args:
            edge_features: Edge features [num_edges, 16] with trajectory info from Task 1.1
            ninja_physics_state: Physics features [18] from Task 1.2
            base_edge_mask: Base edge mask [num_edges] indicating valid edges
            
        Returns:
            Dynamic edge mask [num_edges] with physics constraints applied
        """
        if edge_features.dim() == 3:
            # Handle batched input [batch_size, num_edges, edge_feat_dim]
            return self._compute_batched_mask(edge_features, ninja_physics_state, base_edge_mask)
        else:
            # Handle single graph [num_edges, edge_feat_dim]
            return self._compute_single_mask(edge_features, ninja_physics_state, base_edge_mask)
    
    def _compute_single_mask(
        self,
        edge_features: torch.Tensor,
        ninja_physics_state: torch.Tensor,
        base_edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic mask for a single graph."""
        device = edge_features.device
        dynamic_mask = base_edge_mask.clone()
        
        # Extract ninja physics state components
        # Based on Task 1.2 physics state structure:
        # [vx, vy, vel_magnitude, movement_state, ground_contact, wall_contact, airborne,
        #  momentum_x, momentum_y, kinetic_energy, potential_energy,
        #  jump_buffer, floor_buffer, wall_buffer, launch_pad_buffer, input_state,
        #  can_jump, can_wall_jump]
        
        if ninja_physics_state.numel() < 18:
            logging.warning(f"Insufficient physics state features: {ninja_physics_state.numel()}, expected 18")
            return dynamic_mask
        
        vx = ninja_physics_state[0]  # Normalized velocity x
        vy = ninja_physics_state[1]  # Normalized velocity y
        vel_magnitude = ninja_physics_state[2]  # Velocity magnitude
        movement_state = ninja_physics_state[3] * 9.0  # Denormalize to 0-9
        ground_contact = ninja_physics_state[4]
        wall_contact = ninja_physics_state[5]
        airborne = ninja_physics_state[6]
        kinetic_energy = ninja_physics_state[9]
        can_jump = ninja_physics_state[16]
        can_wall_jump = ninja_physics_state[17]
        
        # Process each edge
        num_edges = edge_features.shape[0]
        for edge_idx in range(num_edges):
            if not base_edge_mask[edge_idx]:
                continue
            
            # Extract edge type (one-hot encoded in first 6 positions)
            edge_type_onehot = edge_features[edge_idx, :6]
            edge_type = torch.argmax(edge_type_onehot).item()
            
            # Extract trajectory requirements from Task 1.1 features
            # Edge features structure: [edge_type(6), direction(2), cost(1), 
            #                          trajectory_params(3), physics_constraints(2), requirements(2)]
            if edge_features.shape[1] >= 16:
                min_velocity = edge_features[edge_idx, 12]  # min_velocity requirement
                max_velocity = edge_features[edge_idx, 13]  # max_velocity requirement  
                requires_jump = edge_features[edge_idx, 14]  # requires_jump flag
                requires_wall_contact = edge_features[edge_idx, 15]  # requires_wall_contact flag
            else:
                # Fallback for older edge feature format
                min_velocity = 0.0
                max_velocity = 10.0
                requires_jump = 0.0
                requires_wall_contact = 0.0
            
            # Apply physics-based edge filtering
            should_disable = False
            
            # Check jump capability constraints
            if edge_type == EdgeType.JUMP and can_jump < self.physics_threshold:
                should_disable = True
            
            # Check wall slide constraints
            elif edge_type == EdgeType.WALL_SLIDE:
                if wall_contact < self.physics_threshold:
                    should_disable = True
                # Wall sliding requires some horizontal velocity to maintain contact
                elif vel_magnitude < self.min_velocity_threshold:
                    should_disable = True
            
            # Check velocity requirements for trajectory-based edges
            elif requires_jump > self.physics_threshold:
                if can_jump < self.physics_threshold:
                    should_disable = True
                elif vel_magnitude < min_velocity:
                    should_disable = True
            
            # Check wall contact requirements
            elif requires_wall_contact > self.physics_threshold:
                if wall_contact < self.physics_threshold:
                    should_disable = True
                # Wall jumps require sufficient horizontal velocity
                if can_wall_jump < self.physics_threshold and vel_magnitude < self.min_wall_jump_speed:
                    should_disable = True
            
            # Check energy requirements for high-cost movements
            elif edge_type == EdgeType.JUMP:
                # Extract energy cost from trajectory parameters
                if edge_features.shape[1] >= 11:
                    energy_cost = edge_features[edge_idx, 10]  # energy_cost from trajectory
                    if kinetic_energy < energy_cost * self.min_jump_energy:
                        should_disable = True
            
            # Check velocity bounds (only if max_velocity is set)
            if max_velocity > 0.0 and vel_magnitude > max_velocity:
                should_disable = True
            elif vel_magnitude < min_velocity:
                should_disable = True
            
            # Apply the mask
            if should_disable:
                dynamic_mask[edge_idx] = 0.0
        
        return dynamic_mask
    
    def _compute_batched_mask(
        self,
        edge_features: torch.Tensor,
        ninja_physics_state: torch.Tensor,
        base_edge_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute dynamic mask for batched graphs."""
        batch_size = edge_features.shape[0]
        dynamic_masks = []
        
        for b in range(batch_size):
            # Handle case where ninja_physics_state might be batched or single
            if ninja_physics_state.dim() == 2:
                batch_physics_state = ninja_physics_state[b]
            else:
                batch_physics_state = ninja_physics_state
            
            batch_mask = self._compute_single_mask(
                edge_features[b],
                batch_physics_state,
                base_edge_mask[b]
            )
            dynamic_masks.append(batch_mask)
        
        return torch.stack(dynamic_masks, dim=0)
    
    def get_constraint_summary(
        self,
        edge_features: torch.Tensor,
        ninja_physics_state: torch.Tensor,
        base_edge_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Get a summary of applied constraints for debugging and analysis.
        
        Args:
            edge_features: Edge features tensor
            ninja_physics_state: Physics state tensor
            base_edge_mask: Base edge mask tensor
            
        Returns:
            Dictionary with constraint statistics
        """
        dynamic_mask = self.compute_dynamic_edge_mask(
            edge_features, ninja_physics_state, base_edge_mask
        )
        
        base_active = base_edge_mask.sum().item()
        dynamic_active = dynamic_mask.sum().item()
        disabled_count = base_active - dynamic_active
        
        # Extract ninja state for summary
        if ninja_physics_state.numel() >= 18:
            vel_magnitude = ninja_physics_state[2].item()
            ground_contact = ninja_physics_state[4].item() > self.physics_threshold
            wall_contact = ninja_physics_state[5].item() > self.physics_threshold
            can_jump = ninja_physics_state[16].item() > self.physics_threshold
            can_wall_jump = ninja_physics_state[17].item() > self.physics_threshold
        else:
            vel_magnitude = 0.0
            ground_contact = wall_contact = can_jump = can_wall_jump = False
        
        return {
            'base_edges': int(base_active),
            'dynamic_edges': int(dynamic_active),
            'disabled_edges': int(disabled_count),
            'disable_rate': disabled_count / max(base_active, 1),
            'ninja_state': {
                'velocity_magnitude': vel_magnitude,
                'ground_contact': ground_contact,
                'wall_contact': wall_contact,
                'can_jump': can_jump,
                'can_wall_jump': can_wall_jump
            }
        }


def create_conditional_edge_masker() -> ConditionalEdgeMasker:
    """
    Create a conditional edge masker with default parameters.
    
    Returns:
        Configured ConditionalEdgeMasker instance
    """
    return ConditionalEdgeMasker()