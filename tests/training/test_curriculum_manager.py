"""
Comprehensive tests for CurriculumManager.

Tests curriculum progression, stage advancement, performance tracking,
and level selection logic for progressive difficulty training.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from npp_rl.training.curriculum_manager import CurriculumManager


class TestCurriculumManager(unittest.TestCase):
    """Test CurriculumManager functionality."""
    
    def setUp(self):
        """Set up test fixtures with temporary dataset."""
        import pickle
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir)
        
        # Create mock level data for each stage
        self.mock_levels = {
            'simple': [
                {'level_id': f'simple_{i}', 'difficulty': 1.0}
                for i in range(10)
            ],
            'medium': [
                {'level_id': f'medium_{i}', 'difficulty': 2.0}
                for i in range(8)
            ],
            'complex': [
                {'level_id': f'complex_{i}', 'difficulty': 3.0}
                for i in range(6)
            ],
            'exploration': [
                {'level_id': f'exploration_{i}', 'difficulty': 4.0}
                for i in range(5)
            ],
            'mine_heavy': [
                {'level_id': f'mine_{i}', 'difficulty': 5.0}
                for i in range(4)
            ],
        }
        
        # Write mock level data to files in TestSuiteLoader expected format
        # TestSuiteLoader expects subdirectories with *.pkl files
        for stage, levels in self.mock_levels.items():
            stage_dir = self.dataset_path / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            for i, level in enumerate(levels):
                level_file = stage_dir / f"level_{i:03d}.pkl"
                with open(level_file, 'wb') as f:
                    pickle.dump(level, f)
    
    def test_initialization_defaults(self):
        """Test curriculum manager initializes with default parameters."""
        manager = CurriculumManager(self.dataset_path)
        
        self.assertEqual(manager.current_stage, 'simple')
        self.assertEqual(manager.current_stage_idx, 0)
        self.assertEqual(manager.advancement_threshold, 0.7)
        self.assertEqual(manager.min_episodes_per_stage, 100)
        self.assertTrue(manager.allow_stage_mixing)
    
    def test_initialization_custom_starting_stage(self):
        """Test initialization with custom starting stage."""
        manager = CurriculumManager(
            self.dataset_path,
            starting_stage='medium'
        )
        
        self.assertEqual(manager.current_stage, 'medium')
        self.assertEqual(manager.current_stage_idx, 1)
    
    def test_initialization_invalid_starting_stage(self):
        """Test that invalid starting stage raises ValueError."""
        with self.assertRaises(ValueError) as context:
            CurriculumManager(
                self.dataset_path,
                starting_stage='invalid_stage'
            )
        
        self.assertIn('Invalid starting stage', str(context.exception))
    
    def test_levels_loaded_correctly(self):
        """Test that levels are loaded correctly for all stages."""
        manager = CurriculumManager(self.dataset_path)
        
        # Verify all stages have levels loaded
        for stage in CurriculumManager.CURRICULUM_ORDER:
            self.assertIn(stage, manager.levels_by_stage)
            self.assertEqual(
                len(manager.levels_by_stage[stage]),
                len(self.mock_levels[stage])
            )
    
    def test_get_current_level_returns_from_current_stage(self):
        """Test that get_current_level returns levels from current stage."""
        manager = CurriculumManager(self.dataset_path)
        
        level = manager.get_current_level()
        
        self.assertIsNotNone(level)
        self.assertTrue(level['level_id'].startswith('simple_'))
    
    def test_record_episode_result_updates_performance(self):
        """Test recording episode results updates performance tracking."""
        manager = CurriculumManager(self.dataset_path)
        
        # Record successful episode
        manager.record_episode_result('simple', success=True)
        
        # Check performance was recorded
        self.assertEqual(len(manager.stage_performance['simple']), 1)
        self.assertTrue(manager.stage_performance['simple'][0])
        self.assertEqual(manager.stage_episode_counts['simple'], 1)
    
    def test_record_multiple_episode_results(self):
        """Test recording multiple episode results."""
        manager = CurriculumManager(self.dataset_path)
        
        # Record mix of successes and failures
        for i in range(10):
            manager.record_episode_result('simple', success=(i % 2 == 0))
        
        # Check all results recorded
        self.assertEqual(len(manager.stage_performance['simple']), 10)
        self.assertEqual(manager.stage_episode_counts['simple'], 10)
        
        # Check success rate
        success_rate = manager.get_stage_success_rate('simple')
        self.assertAlmostEqual(success_rate, 0.5, places=2)
    
    def test_performance_window_limit(self):
        """Test that performance tracking respects window size."""
        manager = CurriculumManager(
            self.dataset_path,
            performance_window=5
        )
        
        # Record more episodes than window size
        for i in range(10):
            manager.record_episode_result('simple', success=True)
        
        # Only last 5 should be kept
        self.assertEqual(len(manager.stage_performance['simple']), 5)
    
    def test_stage_advancement_when_threshold_met(self):
        """Test automatic advancement when threshold is met."""
        manager = CurriculumManager(
            self.dataset_path,
            advancement_threshold=0.7,
            min_episodes_per_stage=10
        )
        
        # Record enough successful episodes to meet threshold
        for _ in range(15):
            manager.record_episode_result('simple', success=True)
        
        # Check for advancement
        advanced = manager.check_and_advance()
        
        self.assertTrue(advanced)
        self.assertEqual(manager.current_stage, 'medium')
        self.assertEqual(manager.current_stage_idx, 1)
    
    def test_no_advancement_below_threshold(self):
        """Test no advancement when success rate below threshold."""
        manager = CurriculumManager(
            self.dataset_path,
            advancement_threshold=0.7,
            min_episodes_per_stage=10
        )
        
        # Record episodes with 50% success rate
        for i in range(20):
            manager.record_episode_result('simple', success=(i % 2 == 0))
        
        # Should not advance
        advanced = manager.check_and_advance()
        
        self.assertFalse(advanced)
        self.assertEqual(manager.current_stage, 'simple')
    
    def test_no_advancement_insufficient_episodes(self):
        """Test no advancement with insufficient episodes."""
        manager = CurriculumManager(
            self.dataset_path,
            advancement_threshold=0.7,
            min_episodes_per_stage=100
        )
        
        # Record few episodes even with high success
        for _ in range(10):
            manager.record_episode_result('simple', success=True)
        
        # Should not advance (not enough episodes)
        advanced = manager.check_and_advance()
        
        self.assertFalse(advanced)
        self.assertEqual(manager.current_stage, 'simple')
    
    def test_no_advancement_at_final_stage(self):
        """Test no advancement when already at final stage."""
        manager = CurriculumManager(
            self.dataset_path,
            starting_stage='mine_heavy',
            min_episodes_per_stage=10
        )
        
        # Record many successful episodes
        for _ in range(20):
            manager.record_episode_result('mine_heavy', success=True)
        
        # Should not advance (already at final stage)
        advanced = manager.check_and_advance()
        
        self.assertFalse(advanced)
        self.assertEqual(manager.current_stage, 'mine_heavy')


if __name__ == '__main__':
    unittest.main()
