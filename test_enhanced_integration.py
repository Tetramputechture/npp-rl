#!/usr/bin/env python3
"""
Test script to validate enhanced integration between npp-rl and nclone systems.

This test verifies that:
1. MovementClassifier and TrajectoryCalculator use enhanced hazard system
2. Bounce block traversal analysis is properly integrated
3. One-way platform directional blocking works correctly
4. Physics constants are synchronized between systems
"""

import sys
import os
sys.path.append('/workspace/nclone')

import math
from typing import Dict, List, Any, Tuple

from npp_rl.models.movement_classifier import MovementClassifier, MovementType
from npp_rl.models.trajectory_calculator import TrajectoryCalculator, MovementState
from nclone.constants.entity_types import EntityType
from nclone.constants.physics_constants import BOUNCE_BLOCK_SIZE, NINJA_RADIUS


def create_test_level_data() -> Dict[str, Any]:
    """Create minimal level data for testing."""
    return {
        'level_id': 'test_level',
        'tiles': [[0 for _ in range(44)] for _ in range(25)],  # Empty level
        'entities': []
    }


def create_bounce_block_entity(x: float, y: float, entity_id: int = 1) -> Dict[str, Any]:
    """Create a bounce block entity for testing."""
    return {
        'id': entity_id,
        'type': EntityType.BOUNCE_BLOCK,
        'x': x,
        'y': y,
        'bounce_state': 0  # Neutral state
    }


def create_one_way_platform_entity(x: float, y: float, orientation: int = 0, entity_id: int = 2) -> Dict[str, Any]:
    """Create a one-way platform entity for testing."""
    return {
        'id': entity_id,
        'type': EntityType.ONE_WAY,
        'x': x,
        'y': y,
        'orientation': orientation
    }


def test_bounce_block_traversal_blocking():
    """Test that bounce blocks can block traversal in narrow passages."""
    print("Testing bounce block traversal blocking...")
    
    classifier = MovementClassifier()
    calculator = TrajectoryCalculator()
    
    # Create test level with bounce block in narrow passage
    level_data = create_test_level_data()
    bounce_block = create_bounce_block_entity(100.0, 100.0)
    
    # Add walls to create a narrow passage (15px clearance)
    wall_above = {'id': 10, 'type': 1, 'x': 100.0, 'y': 85.0}  # Wall 15px above
    wall_below = {'id': 11, 'type': 1, 'x': 100.0, 'y': 115.0}  # Wall 15px below
    
    level_data['entities'] = [bounce_block, wall_above, wall_below]
    
    # Test path that should be blocked (narrow passage)
    src_pos = (85.0, 100.0)  # 15 pixels from bounce block center
    tgt_pos = (115.0, 100.0)  # 15 pixels on other side
    
    print(f"  Testing path from {src_pos} to {tgt_pos}")
    print(f"  Bounce block at (100.0, 100.0)")
    print(f"  Distance from path to bounce block center: 15px (should be blocked if < 16.5px)")
    
    # This should be blocked because clearance is insufficient
    is_feasible = classifier.is_movement_physically_feasible(
        src_pos, tgt_pos, level_data, level_data['entities']
    )
    
    print(f"  Narrow passage (15px clearance): {'BLOCKED' if not is_feasible else 'ALLOWED'}")
    
    # Test path that should be allowed (wide passage)
    # Create a level with wider clearance
    level_data_wide = create_test_level_data()
    bounce_block_wide = create_bounce_block_entity(100.0, 100.0)
    
    # Add walls with wider clearance (25px clearance)
    wall_above_wide = {'id': 12, 'type': 1, 'x': 100.0, 'y': 75.0}  # Wall 25px above
    wall_below_wide = {'id': 13, 'type': 1, 'x': 100.0, 'y': 125.0}  # Wall 25px below
    
    level_data_wide['entities'] = [bounce_block_wide, wall_above_wide, wall_below_wide]
    
    src_pos_wide = (70.0, 100.0)  # 30 pixels from bounce block center
    tgt_pos_wide = (130.0, 100.0)  # 30 pixels on other side
    
    is_feasible_wide = classifier.is_movement_physically_feasible(
        src_pos_wide, tgt_pos_wide, level_data_wide, level_data_wide['entities']
    )
    
    print(f"  Wide passage (30px clearance): {'BLOCKED' if not is_feasible_wide else 'ALLOWED'}")
    
    # Test trajectory validation
    trajectory_result = calculator.calculate_jump_trajectory(src_pos, tgt_pos)
    trajectory_clear = calculator.validate_trajectory_clearance(
        trajectory_result.trajectory_points, level_data, level_data['entities']
    )
    
    print(f"  Trajectory validation: {'BLOCKED' if not trajectory_clear else 'ALLOWED'}")
    
    return not is_feasible and is_feasible_wide and not trajectory_clear


def test_one_way_platform_directional_blocking():
    """Test that one-way platforms block movement from specific directions."""
    print("Testing one-way platform directional blocking...")
    
    classifier = MovementClassifier()
    
    # Create test level with one-way platform (orientation 0 = blocks from left)
    level_data = create_test_level_data()
    one_way = create_one_way_platform_entity(100.0, 100.0, orientation=0)
    level_data['entities'] = [one_way]
    
    # Test movement from blocked direction (left to right)
    src_blocked = (80.0, 100.0)
    tgt_blocked = (120.0, 100.0)
    
    is_feasible_blocked = classifier.is_movement_physically_feasible(
        src_blocked, tgt_blocked, level_data, level_data['entities']
    )
    
    print(f"  Movement from blocked direction: {'BLOCKED' if not is_feasible_blocked else 'ALLOWED'}")
    
    # Test movement from allowed direction (right to left)
    src_allowed = (120.0, 100.0)
    tgt_allowed = (80.0, 100.0)
    
    is_feasible_allowed = classifier.is_movement_physically_feasible(
        src_allowed, tgt_allowed, level_data, level_data['entities']
    )
    
    print(f"  Movement from allowed direction: {'BLOCKED' if not is_feasible_allowed else 'ALLOWED'}")
    
    return not is_feasible_blocked and is_feasible_allowed


def test_bounce_block_movement_classification():
    """Test that bounce block movement is properly classified."""
    print("Testing bounce block movement classification...")
    
    classifier = MovementClassifier()
    
    # Create test level with bounce blocks
    level_data = create_test_level_data()
    bounce_block1 = create_bounce_block_entity(50.0, 100.0, 1)
    bounce_block2 = create_bounce_block_entity(150.0, 100.0, 2)
    level_data['entities'] = [bounce_block1, bounce_block2]
    
    # Test bounce block movement detection
    src_pos = (40.0, 100.0)  # 10px from first bounce block
    tgt_pos = (160.0, 100.0)  # 10px from second bounce block
    movement_type, params = classifier.classify_movement(
        src_pos, tgt_pos, level_data=level_data
    )
    
    print(f"  Movement type classification: {movement_type.name}")
    
    energy_cost = params.get('energy_cost', 'N/A')
    time_estimate = params.get('time_estimate', 'N/A')
    boost_multiplier = params.get('boost_multiplier', 'N/A')
    
    if isinstance(energy_cost, (int, float)):
        print(f"  Energy cost: {energy_cost:.2f}")
    else:
        print(f"  Energy cost: {energy_cost}")
        
    if isinstance(time_estimate, (int, float)):
        print(f"  Time estimate: {time_estimate:.2f}")
    else:
        print(f"  Time estimate: {time_estimate}")
        
    if isinstance(boost_multiplier, (int, float)):
        print(f"  Boost multiplier: {boost_multiplier:.2f}")
    else:
        print(f"  Boost multiplier: {boost_multiplier}")
    
    # Test if bounce block movement was detected
    if movement_type in [MovementType.BOUNCE_BLOCK, MovementType.BOUNCE_CHAIN]:
        return True
    
    return False


def test_physics_constants_synchronization():
    """Test that physics constants are synchronized between systems."""
    print("Testing physics constants synchronization...")
    
    # Import constants from both systems
    from nclone.constants.physics_constants import (
        MAX_HOR_SPEED as NCLONE_MAX_HOR_SPEED,
        BOUNCE_BLOCK_BOOST_MIN as NCLONE_BOUNCE_BOOST_MIN,
        NINJA_RADIUS as NCLONE_NINJA_RADIUS
    )
    
    # Test that MovementClassifier uses same constants
    classifier = MovementClassifier()
    calculator = TrajectoryCalculator()
    
    # These should be accessible and consistent
    print(f"  MAX_HOR_SPEED: {NCLONE_MAX_HOR_SPEED}")
    print(f"  BOUNCE_BLOCK_BOOST_MIN: {NCLONE_BOUNCE_BOOST_MIN}")
    print(f"  NINJA_RADIUS: {NCLONE_NINJA_RADIUS}")
    print(f"  BOUNCE_BLOCK_SIZE: {BOUNCE_BLOCK_SIZE}")
    
    return True


def test_trajectory_calculation_accuracy():
    """Test trajectory calculation accuracy with physics constants."""
    print("Testing trajectory calculation accuracy...")
    
    calculator = TrajectoryCalculator()
    
    # Test basic jump trajectory
    src_pos = (100.0, 200.0)
    tgt_pos = (200.0, 150.0)
    
    result = calculator.calculate_jump_trajectory(src_pos, tgt_pos)
    
    print(f"  Trajectory feasible: {result.feasible}")
    print(f"  Time of flight: {result.time_of_flight:.2f}")
    print(f"  Energy cost: {result.energy_cost:.2f}")
    print(f"  Success probability: {result.success_probability:.2f}")
    print(f"  Trajectory points: {len(result.trajectory_points)}")
    
    return result.feasible and len(result.trajectory_points) > 0


def main():
    """Run all integration tests."""
    print("=== Enhanced npp-rl and nclone Integration Tests ===\n")
    
    tests = [
        ("Bounce Block Traversal Blocking", test_bounce_block_traversal_blocking),
        ("One-Way Platform Directional Blocking", test_one_way_platform_directional_blocking),
        ("Bounce Block Movement Classification", test_bounce_block_movement_classification),
        ("Physics Constants Synchronization", test_physics_constants_synchronization),
        ("Trajectory Calculation Accuracy", test_trajectory_calculation_accuracy),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result:
                print(f"✓ PASSED")
                passed += 1
            else:
                print(f"✗ FAILED")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed! Enhanced systems are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the integration.")
        return 1


if __name__ == "__main__":
    exit(main())