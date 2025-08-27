"""
Test enhanced entity behavior integration for graph building.

This test validates that the graph builder correctly handles:
- Thwump directional traversability (safe from sides/back)
- Bounce block platform behavior (always traversable)
- Entity-aware sub-cell traversability
"""

import pytest
import numpy as np
import sys
import os

# Add nclone to path for graph builder access
nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nclone')
if os.path.exists(nclone_path):
    sys.path.insert(0, nclone_path)

from nclone.graph.graph_builder import GraphBuilder
from npp_rl.models.entity_type_system import EntityTypeSystem, EntityCategory


class TestEntityBehaviorImprovements:
    """Test enhanced entity behavior integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graph_builder = GraphBuilder()
        self.entity_type_system = EntityTypeSystem()
        
        # Create test level data
        self.level_data = {
            'tiles': np.zeros((23, 42), dtype=np.int32),  # Empty level
            'width': 42,
            'height': 23
        }
        
        # Test ninja position
        self.ninja_position = (240.0, 144.0)  # Center of level
        self.ninja_velocity = (50.0, 0.0)
        self.ninja_state = 0
    
    def test_thwump_directional_traversability(self):
        """Test that thwumps are safe from sides/back but dangerous from front."""
        # Create thwump entity facing right (orientation 0)
        thwump_entity = {
            'type': 20,  # Thwump
            'x': 240.0,
            'y': 144.0,
            'orientation': 0,  # Facing right
            'state': 0  # Idle
        }
        
        # Test approach from different directions
        # Approaching toward the right (into charging face - dangerous)
        approach_toward_right = (1.0, 0.0)
        is_safe_toward_right = self.graph_builder._is_thwump_side_safe(
            thwump_entity, approach_toward_right
        )
        assert not is_safe_toward_right, "Thwump should be dangerous when approached toward charging direction"
        
        # Approaching toward the left (away from charging face - safe)
        approach_toward_left = (-1.0, 0.0)
        is_safe_toward_left = self.graph_builder._is_thwump_side_safe(
            thwump_entity, approach_toward_left
        )
        assert is_safe_toward_left, "Thwump should be safe when approached away from charging direction"
        
        # From top (safe - approaching from side)
        approach_from_top = (0.0, -1.0)
        is_safe_from_top = self.graph_builder._is_thwump_side_safe(
            thwump_entity, approach_from_top
        )
        assert is_safe_from_top, "Thwump should be safe when approached from side"
        
        # From bottom (safe - approaching from side)
        approach_from_bottom = (0.0, 1.0)
        is_safe_from_bottom = self.graph_builder._is_thwump_side_safe(
            thwump_entity, approach_from_bottom
        )
        assert is_safe_from_bottom, "Thwump should be safe when approached from side"
    
    def test_thwump_different_orientations(self):
        """Test thwump safety for different orientations."""
        orientations = [
            (0, (1.0, 0.0)),   # Right-facing, dangerous from left
            (2, (0.0, 1.0)),   # Down-facing, dangerous from top
            (4, (-1.0, 0.0)),  # Left-facing, dangerous from right
            (6, (0.0, -1.0))   # Up-facing, dangerous from bottom
        ]
        
        for orientation, dangerous_approach in orientations:
            thwump_entity = {
                'type': 20,
                'x': 240.0,
                'y': 144.0,
                'orientation': orientation,
                'state': 0
            }
            
            # Test dangerous approach
            is_safe_dangerous = self.graph_builder._is_thwump_side_safe(
                thwump_entity, dangerous_approach
            )
            assert not is_safe_dangerous, f"Thwump orientation {orientation} should be dangerous from {dangerous_approach}"
            
            # Test perpendicular approaches (should be safe)
            if dangerous_approach[0] != 0:  # Horizontal dangerous direction
                safe_approaches = [(0.0, 1.0), (0.0, -1.0)]
            else:  # Vertical dangerous direction
                safe_approaches = [(1.0, 0.0), (-1.0, 0.0)]
            
            for safe_approach in safe_approaches:
                is_safe = self.graph_builder._is_thwump_side_safe(
                    thwump_entity, safe_approach
                )
                assert is_safe, f"Thwump orientation {orientation} should be safe from {safe_approach}"
    
    def test_bounce_block_traversability(self):
        """Test that bounce blocks are always traversable."""
        bounce_block_entity = {
            'type': 17,  # Bounce block
            'x': 240.0,
            'y': 144.0,
            'state': 0
        }
        
        # Test traversability from all directions
        directions = [
            (1.0, 0.0),   # From left
            (-1.0, 0.0),  # From right
            (0.0, 1.0),   # From top
            (0.0, -1.0),  # From bottom
            (0.707, 0.707),   # From diagonal
            (-0.707, -0.707)  # From opposite diagonal
        ]
        
        for direction in directions:
            is_traversable = self.graph_builder._is_entity_traversable(
                bounce_block_entity, direction
            )
            assert is_traversable, f"Bounce block should be traversable from direction {direction}"
    
    def test_entity_platform_properties(self):
        """Test entity platform properties."""
        # Test bounce block platform properties
        bounce_block = {'type': 17}
        bounce_props = self.graph_builder._get_entity_platform_properties(bounce_block)
        
        assert bounce_props['is_platform'], "Bounce block should be a platform"
        assert bounce_props['can_stand_on'], "Bounce block should be standable"
        assert bounce_props['bounce_factor'] > 1.0, "Bounce block should provide momentum boost"
        assert bounce_props['movement_type'] == 'spring', "Bounce block should have spring movement"
        
        # Test thwump platform properties
        thwump = {'type': 20}
        thwump_props = self.graph_builder._get_entity_platform_properties(thwump)
        
        assert thwump_props['is_platform'], "Thwump should be a platform"
        assert thwump_props['can_stand_on'], "Thwump should be standable"
        assert thwump_props['conditional_safe'], "Thwump should be conditionally safe"
        assert thwump_props['movement_type'] == 'moving_platform', "Thwump should be a moving platform"
    
    def test_entity_collision_radius(self):
        """Test entity collision radius calculation."""
        # Test large entities (bounce blocks, thwumps)
        large_entities = [
            {'type': 17},  # Bounce block
            {'type': 20},  # Thwump
            {'type': 28}   # Shove thwump
        ]
        
        for entity in large_entities:
            radius = self.graph_builder._get_entity_collision_radius(entity)
            assert radius == 9.0, f"Large entity {entity['type']} should have radius 9.0"
        
        # Test small entities (drones, mines)
        small_entities = [
            {'type': 12},  # Death ball
            {'type': 14},  # Drone
            {'type': 21},  # Mine
            {'type': 25},  # Death ball variant
            {'type': 26}   # Drone variant
        ]
        
        for entity in small_entities:
            radius = self.graph_builder._get_entity_collision_radius(entity)
            assert radius == 6.0, f"Small entity {entity['type']} should have radius 6.0"
    
    def test_entity_type_system_improvements(self):
        """Test entity type system improvements."""
        # Test thwump categorization
        thwump_category = self.entity_type_system.get_entity_category(20)
        assert thwump_category == EntityCategory.CONDITIONAL, "Thwump should be in CONDITIONAL category"
        
        # Test bounce block categorization
        bounce_category = self.entity_type_system.get_entity_category(17)
        assert bounce_category == EntityCategory.MOVEMENT, "Bounce block should be in MOVEMENT category"
        
        # Test directional hazard detection
        assert self.entity_type_system.is_directional_hazard(20), "Thwump should be directional hazard"
        assert not self.entity_type_system.is_directional_hazard(17), "Bounce block should not be directional hazard"
        
        # Test platform capability
        assert self.entity_type_system.is_platform_capable(20), "Thwump should be platform capable"
        assert self.entity_type_system.is_platform_capable(17), "Bounce block should be platform capable"
        
        # Test traversability
        assert self.entity_type_system.is_traversable(20), "Thwump should be conditionally traversable"
        assert self.entity_type_system.is_traversable(17), "Bounce block should be traversable"
        assert not self.entity_type_system.is_traversable(14), "Drone should not be traversable"
    
    def test_sub_cell_entity_traversability(self):
        """Test sub-cell entity traversability integration."""
        # Create entities for testing
        entities = [
            {
                'type': 20,  # Thwump facing right
                'x': 240.0,
                'y': 144.0,
                'orientation': 0,
                'state': 0
            },
            {
                'type': 17,  # Bounce block
                'x': 300.0,
                'y': 144.0,
                'state': 0
            }
        ]
        
        # Test position near thwump from safe side
        safe_position_x = 240.0 + 15.0  # Behind thwump
        safe_position_y = 144.0
        approach_direction = (-1.0, 0.0)  # Approaching from right (safe)
        
        is_safe = self.graph_builder._is_sub_cell_entity_traversable(
            safe_position_x, safe_position_y, entities, approach_direction
        )
        assert is_safe, "Position behind thwump should be safe"
        
        # Test position near thwump from dangerous side
        dangerous_position_x = 240.0 - 15.0  # In front of thwump
        dangerous_position_y = 144.0
        dangerous_approach = (1.0, 0.0)  # Approaching from left (dangerous)
        
        is_dangerous = self.graph_builder._is_sub_cell_entity_traversable(
            dangerous_position_x, dangerous_position_y, entities, dangerous_approach
        )
        assert not is_dangerous, "Position in front of thwump should be dangerous"
        
        # Test position near bounce block (should always be safe)
        bounce_position_x = 300.0 + 10.0
        bounce_position_y = 144.0
        
        is_bounce_safe = self.graph_builder._is_sub_cell_entity_traversable(
            bounce_position_x, bounce_position_y, entities, approach_direction
        )
        assert is_bounce_safe, "Position near bounce block should be safe"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])