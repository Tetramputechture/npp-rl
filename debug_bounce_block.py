#!/usr/bin/env python3
"""
Debug script to test bounce block traversal analysis directly.
"""

import sys
sys.path.append('/workspace/nclone')

from nclone.graph.hazard_system import HazardClassificationSystem
from nclone.constants.entity_types import EntityType


def test_bounce_block_analysis():
    """Test bounce block analysis directly."""
    hazard_system = HazardClassificationSystem()
    
    # Create bounce block entity
    bounce_block = {
        'id': 1,
        'type': EntityType.BOUNCE_BLOCK,
        'x': 100.0,
        'y': 100.0,
        'bounce_state': 0
    }
    
    # Create entities that form a narrow passage
    # Add walls above and below to create a narrow vertical passage
    wall_above = {'id': 2, 'type': 1, 'x': 100.0, 'y': 85.0}  # Wall 15px above
    wall_below = {'id': 3, 'type': 1, 'x': 100.0, 'y': 115.0}  # Wall 15px below
    
    all_entities = [bounce_block, wall_above, wall_below]
    
    # Test path that should be blocked
    path_start = (85.0, 100.0)
    path_end = (115.0, 100.0)
    
    print(f"Testing bounce block at ({bounce_block['x']}, {bounce_block['y']})")
    print(f"Path from {path_start} to {path_end}")
    
    result = hazard_system.analyze_bounce_block_traversal_blocking(
        bounce_block, all_entities, path_start, path_end
    )
    
    print(f"Result: {'BLOCKED' if result else 'ALLOWED'}")
    
    # Test with wider clearance
    print("\n--- Testing with wider clearance (25px) ---")
    
    # Add walls with wider clearance (25px clearance)
    wall_above_wide = {'id': 4, 'type': 1, 'x': 100.0, 'y': 75.0}  # Wall 25px above
    wall_below_wide = {'id': 5, 'type': 1, 'x': 100.0, 'y': 125.0}  # Wall 25px below
    
    entities_wide_clearance = [bounce_block, wall_above_wide, wall_below_wide]
    
    result2 = hazard_system.analyze_bounce_block_traversal_blocking(
        bounce_block, entities_wide_clearance, path_start, path_end
    )
    
    print(f"Result with 25px clearance: {'BLOCKED' if result2 else 'ALLOWED'}")
    
    # Test with even wider clearance
    print("\n--- Testing with very wide clearance (35px) ---")
    
    # Add walls with very wide clearance (35px clearance)
    wall_above_very_wide = {'id': 6, 'type': 1, 'x': 100.0, 'y': 65.0}  # Wall 35px above
    wall_below_very_wide = {'id': 7, 'type': 1, 'x': 100.0, 'y': 135.0}  # Wall 35px below
    
    entities_very_wide_clearance = [bounce_block, wall_above_very_wide, wall_below_very_wide]
    
    result3 = hazard_system.analyze_bounce_block_traversal_blocking(
        bounce_block, entities_very_wide_clearance, path_start, path_end
    )
    
    print(f"Result with 35px clearance: {'BLOCKED' if result3 else 'ALLOWED'}")


if __name__ == "__main__":
    test_bounce_block_analysis()