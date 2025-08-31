"""
Test dynamic entity caching functionality.

This module tests the dynamic entity caching system that handles entities
whose states can change during gameplay, particularly toggle mines.
"""

import unittest
import numpy as np
from npp_rl.models.movement_classifier import MovementClassifier
from npp_rl.models.physics_state_extractor import PhysicsStateExtractor
from nclone.entity_classes.entity_toggle_mine import EntityToggleMine
from nclone.entity_classes.entity_exit_switch import EntityExitSwitch
from nclone.entity_classes.entity_drone_zap import EntityDroneZap
from nclone.entity_classes.entity_thwump import EntityThwump


class TestDynamicEntityCaching(unittest.TestCase):
    """Test dynamic entity caching for state-changing entities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = MovementClassifier()
        self.extractor = PhysicsStateExtractor()
        
        # Base level data with dynamic entities
        self.level_data = {
            'tiles': np.zeros((20, 20)),
            'entities': [
                # Toggle mine - starts safe (state 1), becomes deadly (state 0)
                {
                    'type': EntityToggleMine.ENTITY_TYPE,
                    'x': 100.0,
                    'y': 100.0,
                    'state': 1  # Safe initially
                },
                # Exit switch - starts inactive (state 0), becomes active (state 1)
                {
                    'type': EntityExitSwitch.ENTITY_TYPE,
                    'x': 200.0,
                    'y': 100.0,
                    'state': 0  # Inactive initially
                },
                # Drone - position can change
                {
                    'type': EntityDroneZap.ENTITY_TYPE,
                    'x': 150.0,
                    'y': 150.0,
                    'state': 0
                },
                # Thwump - position and state can change
                {
                    'type': EntityThwump.ENTITY_TYPE,
                    'x': 250.0,
                    'y': 100.0,
                    'state': 0,
                    'orientation': 0  # Right-facing
                }
            ]
        }
        
        self.ninja_state = {
            'movement_state': 1,
            'jump_buffer': 0,
            'floor_buffer': 0,
            'wall_buffer': 0,
            'launch_pad_buffer': 0,
            'input_state': 0
        }
    
    def test_toggle_mine_state_change_detection(self):
        """Test that toggle mine state changes are properly detected."""
        # Initial state - mine is safe (state 1)
        self.classifier._update_dynamic_entities(self.level_data)
        toggle_mines = self.classifier._dynamic_entity_cache['toggle_mines']
        
        self.assertEqual(len(toggle_mines), 1)
        self.assertEqual(toggle_mines[0]['state'], 1)
        self.assertFalse(toggle_mines[0]['is_deadly'])
        
        # Change mine state to deadly (state 0)
        self.level_data['entities'][0]['state'] = 0
        self.classifier._update_dynamic_entities(self.level_data)
        toggle_mines = self.classifier._dynamic_entity_cache['toggle_mines']
        
        self.assertEqual(len(toggle_mines), 1)
        self.assertEqual(toggle_mines[0]['state'], 0)
        self.assertTrue(toggle_mines[0]['is_deadly'])
    
    def test_switch_activation_detection(self):
        """Test that switch activation state changes are properly detected."""
        # Initial state - switch is inactive (state 0)
        self.classifier._update_dynamic_entities(self.level_data)
        switches = self.classifier._dynamic_entity_cache['switches']
        
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0]['state'], 0)
        self.assertFalse(switches[0]['activated'])
        
        # Activate switch (state 1)
        self.level_data['entities'][1]['state'] = 1
        self.classifier._update_dynamic_entities(self.level_data)
        switches = self.classifier._dynamic_entity_cache['switches']
        
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0]['state'], 1)
        self.assertTrue(switches[0]['activated'])
    
    def test_drone_position_tracking(self):
        """Test that drone position changes are properly tracked."""
        # Initial position
        self.classifier._update_dynamic_entities(self.level_data)
        drones = self.classifier._dynamic_entity_cache['drones']
        
        self.assertEqual(len(drones), 1)
        self.assertEqual(drones[0]['x'], 150.0)
        self.assertEqual(drones[0]['y'], 150.0)
        
        # Change drone position
        self.level_data['entities'][2]['x'] = 175.0
        self.level_data['entities'][2]['y'] = 125.0
        self.classifier._update_dynamic_entities(self.level_data)
        drones = self.classifier._dynamic_entity_cache['drones']
        
        self.assertEqual(len(drones), 1)
        self.assertEqual(drones[0]['x'], 175.0)
        self.assertEqual(drones[0]['y'], 125.0)
    
    def test_thwump_state_tracking(self):
        """Test that thwump state and position changes are properly tracked."""
        # Initial state
        self.classifier._update_dynamic_entities(self.level_data)
        thwumps = self.classifier._dynamic_entity_cache['thwumps']
        
        self.assertEqual(len(thwumps), 1)
        self.assertEqual(thwumps[0]['x'], 250.0)
        self.assertEqual(thwumps[0]['state'], 0)
        
        # Change thwump state and position
        self.level_data['entities'][3]['x'] = 275.0
        self.level_data['entities'][3]['state'] = 1
        self.classifier._update_dynamic_entities(self.level_data)
        thwumps = self.classifier._dynamic_entity_cache['thwumps']
        
        self.assertEqual(len(thwumps), 1)
        self.assertEqual(thwumps[0]['x'], 275.0)
        self.assertEqual(thwumps[0]['state'], 1)
    
    def test_physics_extractor_dynamic_integration(self):
        """Test that physics state extractor properly integrates dynamic entities."""
        ninja_position = (120.0, 120.0)
        ninja_velocity = (5.0, 0.0)
        
        # Extract physics state with safe toggle mine
        features1 = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Change toggle mine to deadly state
        self.level_data['entities'][0]['state'] = 0
        
        # Extract physics state with deadly toggle mine
        features2 = self.extractor.extract_ninja_physics_state(
            ninja_position, ninja_velocity, self.ninja_state, self.level_data
        )
        
        # Hazard proximity should be different (higher with deadly mine)
        hazard_proximity_idx = 21  # Index for hazard proximity feature
        self.assertGreater(features2[hazard_proximity_idx], features1[hazard_proximity_idx])
    
    def test_cache_persistence_across_calls(self):
        """Test that dynamic cache is updated on each call."""
        # First call - mine is safe
        self.classifier._update_dynamic_entities(self.level_data)
        cache1 = self.classifier._dynamic_entity_cache['toggle_mines'][0]['is_deadly']
        
        # Change mine state
        self.level_data['entities'][0]['state'] = 0
        
        # Second call - mine is deadly
        self.classifier._update_dynamic_entities(self.level_data)
        cache2 = self.classifier._dynamic_entity_cache['toggle_mines'][0]['is_deadly']
        
        self.assertFalse(cache1)  # Was safe
        self.assertTrue(cache2)   # Now deadly
    
    def test_hazard_detection_with_dynamic_states(self):
        """Test hazard detection considers dynamic entity states."""
        # Test with safe toggle mine
        safe_mine = {
            'type': EntityToggleMine.ENTITY_TYPE,
            'state': 1,  # Safe state
            'x': 100.0,
            'y': 100.0
        }
        self.assertFalse(self.extractor._is_hazardous_entity(safe_mine))
        
        # Test with deadly toggle mine
        deadly_mine = {
            'type': EntityToggleMine.ENTITY_TYPE,
            'state': 0,  # Deadly state
            'x': 100.0,
            'y': 100.0
        }
        self.assertTrue(self.extractor._is_hazardous_entity(deadly_mine))
    
    def test_thwump_orientation_hazard_detection(self):
        """Test that thwump hazard detection considers orientation and ninja position."""
        ninja_position = (260.0, 100.0)  # To the right of thwump
        
        # Right-facing thwump (orientation 0) - ninja is on dangerous side
        right_thwump = {
            'type': EntityThwump.ENTITY_TYPE,
            'orientation': 0,
            'x': 250.0,
            'y': 100.0,
            'state': 0
        }
        self.assertTrue(self.extractor._is_hazardous_entity(right_thwump, ninja_position))
        
        # Left-facing thwump (orientation 2) - ninja is on safe side
        left_thwump = {
            'type': EntityThwump.ENTITY_TYPE,
            'orientation': 2,
            'x': 250.0,
            'y': 100.0,
            'state': 0
        }
        self.assertFalse(self.extractor._is_hazardous_entity(left_thwump, ninja_position))


if __name__ == '__main__':
    unittest.main()