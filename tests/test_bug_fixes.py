"""
Unit tests for bug fixes in production readiness audit.

Tests verify that critical bugs have been fixed:
1. Division by zero in exploration_metrics.py
2. Division by zero in mine_aware_curiosity.py
3. Duplicate Subtask enum definitions
4. Duplicate EdgeType enum definitions
5. HGTConfig name collision
6. Circular import issues
"""

import unittest
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDivisionByZeroFixes(unittest.TestCase):
    """Test division by zero bug fixes."""
    
    def test_exploration_metrics_empty_position_history(self):
        """Test that ExplorationMetrics handles empty position history without division by zero."""
        from npp_rl.eval.exploration_metrics import ExplorationMetrics
        
        # Create metrics tracker with empty history
        metrics = ExplorationMetrics()
        
        # This should not crash with division by zero
        entropy = metrics._compute_visitation_entropy()
        
        # Should return 0 for empty history
        self.assertEqual(entropy, 0.0, "Empty position history should return 0 entropy")
    
    def test_mine_aware_curiosity_zero_count(self):
        """Test that MineAwareCuriosityModulator handles zero count without division by zero."""
        from npp_rl.intrinsic.mine_aware_curiosity import MineAwareCuriosityModulator
        
        # Create curiosity module
        curiosity = MineAwareCuriosityModulator()
        
        # Set stats to zero count
        curiosity._stats['total_modulations'] = 0
        curiosity._stats['average_modulation'] = 0.0
        
        # This should not crash with division by zero (should early return)
        try:
            curiosity._update_running_average(2.0)
            # If we get here, it didn't crash
            self.assertTrue(True, "Zero count handled without division by zero")
        except ZeroDivisionError:
            self.fail("Division by zero occurred with zero count")
    
    def test_mine_aware_curiosity_nonzero_count(self):
        """Test that MineAwareCuriosityModulator correctly updates running average with nonzero count."""
        from npp_rl.intrinsic.mine_aware_curiosity import MineAwareCuriosityModulator
        
        curiosity = MineAwareCuriosityModulator()
        
        # Set stats to known values
        curiosity._stats['total_modulations'] = 5
        curiosity._stats['average_modulation'] = 1.0
        
        # Update with new value
        new_value = 2.0
        curiosity._update_running_average(new_value)
        
        # Should compute correct running average: (1.0 * 4 + 2.0) / 5 = 1.2
        expected = (1.0 * 4 + 2.0) / 5
        self.assertAlmostEqual(curiosity._stats['average_modulation'], expected, places=5, 
                              msg="Running average calculation should be correct")


class TestDuplicateEnumFixes(unittest.TestCase):
    """Test that duplicate enum definitions have been fixed."""
    
    def test_subtask_enum_single_definition(self):
        """Test that Subtask enum is imported from high_level_policy, not duplicated."""
        from npp_rl.hrl.high_level_policy import Subtask as SubtaskFromHighLevel
        from npp_rl.hrl.completion_controller import Subtask as SubtaskFromController
        
        # Both should be the same class (not duplicates)
        self.assertIs(SubtaskFromHighLevel, SubtaskFromController,
                     "Subtask should be imported from high_level_policy, not duplicated")
        
        # Verify all expected subtasks exist
        expected_subtasks = [
            'NAVIGATE_TO_EXIT_SWITCH',
            'NAVIGATE_TO_LOCKED_DOOR_SWITCH',
            'NAVIGATE_TO_EXIT_DOOR',
            'EXPLORE_FOR_SWITCHES'
        ]
        
        for subtask_name in expected_subtasks:
            self.assertTrue(hasattr(SubtaskFromHighLevel, subtask_name),
                          f"Subtask.{subtask_name} should exist")
    
    def test_edge_type_not_in_hgt_layer(self):
        """Test that EdgeType is not duplicated in hgt_layer.py."""
        # EdgeType should only be in conditional_edges.py
        from npp_rl.models.conditional_edges import EdgeType
        
        # Verify EdgeType has expected values
        self.assertTrue(hasattr(EdgeType, 'ADJACENT'),
                       "EdgeType.ADJACENT should exist")
        self.assertTrue(hasattr(EdgeType, 'LOGICAL'),
                       "EdgeType.LOGICAL should exist")
        self.assertTrue(hasattr(EdgeType, 'REACHABLE'),
                       "EdgeType.REACHABLE should exist")
        
        # hgt_layer should not define EdgeType
        import npp_rl.models.hgt_layer as hgt_layer_module
        
        # If EdgeType is in hgt_layer, it should be imported, not defined
        if hasattr(hgt_layer_module, 'EdgeType'):
            # Should be the same class, not a duplicate
            self.assertIs(hgt_layer_module.EdgeType, EdgeType,
                         "EdgeType in hgt_layer should be imported, not redefined")


class TestNameCollisionFixes(unittest.TestCase):
    """Test that HGTConfig name collision has been resolved."""
    
    def test_hgt_config_classes_distinct(self):
        """Test that HGTConfig and HGTFactoryConfig are distinct."""
        from npp_rl.models.hgt_config import HGTConfig
        from npp_rl.models.hgt_factory import HGTFactoryConfig
        
        # Should be different classes
        self.assertIsNot(HGTConfig, HGTFactoryConfig,
                        "HGTConfig and HGTFactoryConfig should be different classes")
        
        # HGTConfig should be a dataclass (from hgt_config.py)
        self.assertTrue(hasattr(HGTConfig, '__dataclass_fields__'),
                       "HGTConfig should be a dataclass")
        
        # HGTFactoryConfig should have class constants (from hgt_factory.py)
        self.assertTrue(hasattr(HGTFactoryConfig, 'HIDDEN_DIM'),
                       "HGTFactoryConfig should have HIDDEN_DIM constant")


class TestCircularImportFixes(unittest.TestCase):
    """Test that circular imports have been resolved."""
    
    def test_configurable_extractor_imports(self):
        """Test that ConfigurableMultimodalExtractor can be imported without circular dependency."""
        try:
            from npp_rl.feature_extractors.configurable_extractor import ConfigurableMultimodalExtractor
            
            # If we get here, import succeeded
            self.assertTrue(True, "ConfigurableMultimodalExtractor imported successfully")
            
        except ImportError as e:
            self.fail(f"Circular import detected: {e}")
    
    def test_training_module_imports(self):
        """Test that training module imports work without circular dependencies."""
        try:
            from npp_rl.training import architecture_configs
            from npp_rl.training import architecture_trainer
            
            # If we get here, imports succeeded
            self.assertTrue(True, "Training modules imported successfully")
            
        except ImportError as e:
            self.fail(f"Circular import in training modules: {e}")
    
    def test_all_core_modules_importable(self):
        """Test that all core modules can be imported without errors."""
        core_modules = [
            'npp_rl.agents.training',
            'npp_rl.models.hgt_encoder',
            'npp_rl.models.hgt_factory',
            'npp_rl.hrl.high_level_policy',
            'npp_rl.hrl.completion_controller',
            'npp_rl.feature_extractors.configurable_extractor',
        ]
        
        for module_name in core_modules:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")


class TestTorchGeometricInstalled(unittest.TestCase):
    """Test that torch_geometric is properly installed."""
    
    def test_torch_geometric_importable(self):
        """Test that torch_geometric can be imported."""
        try:
            import torch_geometric
            from torch_geometric.nn import HGTConv
            
            self.assertTrue(True, "torch_geometric imported successfully")
            
        except ImportError as e:
            self.fail(f"torch_geometric not installed or not importable: {e}")


if __name__ == '__main__':
    unittest.main()
