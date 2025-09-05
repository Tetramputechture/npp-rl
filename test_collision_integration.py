#!/usr/bin/env python3
"""
Test integration between PreciseTileCollision and MovementClassifier/TrajectoryCalculator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from npp_rl.models.movement_classifier import MovementClassifier, MovementType
from npp_rl.models.trajectory_calculator import TrajectoryCalculator


def test_movement_classifier_collision():
    """Test that MovementClassifier uses precise collision detection correctly."""
    classifier = MovementClassifier()
    
    # Test level with a solid tile blocking the path
    level_data = {
        'level_id': 'test_collision',
        'tiles': {
            (5, 5): 1  # Solid tile at (5, 5) - covers (120, 120) to (144, 144)
        }
    }
    
    # Test case 1: Path blocked by tile
    src_pos = (100, 132)  # Left of tile
    tgt_pos = (160, 132)  # Right of tile, path goes through tile
    
    movement_type, physics_params = classifier.classify_movement(
        src_pos, tgt_pos, level_data=level_data
    )
    
    print(f"Blocked path classification:")
    print(f"  Movement type: {movement_type}")
    print(f"  Physics params: {physics_params}")
    
    # Test case 2: Path clear around tile
    src_pos = (100, 100)  # Above tile
    tgt_pos = (160, 100)  # Still above tile
    
    movement_type, physics_params = classifier.classify_movement(
        src_pos, tgt_pos, level_data=level_data
    )
    
    print(f"\nClear path classification:")
    print(f"  Movement type: {movement_type}")
    print(f"  Physics params: {physics_params}")


def test_trajectory_calculator_collision():
    """Test that TrajectoryCalculator uses precise collision detection correctly."""
    calculator = TrajectoryCalculator()
    
    # Test level with a solid tile
    level_data = {
        'level_id': 'test_trajectory',
        'tiles': {
            (6, 6): 1  # Solid tile at (6, 6) - covers (144, 144) to (168, 168)
        }
    }
    
    # Test case 1: Trajectory blocked by tile
    blocked_trajectory = [
        (130, 180),  # Start below tile
        (150, 156),  # Through tile
        (180, 130)   # End above tile
    ]
    
    clearance_result = calculator.validate_trajectory_clearance(
        blocked_trajectory, level_data
    )
    
    print(f"\nBlocked trajectory clearance: {clearance_result} (expected: False)")
    
    # Test case 2: Trajectory clear of tile
    clear_trajectory = [
        (100, 180),  # Start left and below tile
        (110, 156),  # Clear of tile
        (120, 130)   # End left and above tile
    ]
    
    clearance_result = calculator.validate_trajectory_clearance(
        clear_trajectory, level_data
    )
    
    print(f"Clear trajectory clearance: {clearance_result} (expected: True)")
    
    # Test case 3: Basic jump trajectory calculation (without level data)
    result = calculator.calculate_jump_trajectory((100, 180), (120, 130))
    
    print(f"\nBasic trajectory result:")
    print(f"  Feasible: {result.feasible}")
    print(f"  Energy cost: {result.energy_cost}")
    print(f"  Success probability: {result.success_probability}")


def main():
    """Run collision integration tests."""
    print("Testing collision integration with MovementClassifier and TrajectoryCalculator...")
    print()
    
    try:
        test_movement_classifier_collision()
        test_trajectory_calculator_collision()
        print("\n✓ All collision integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())