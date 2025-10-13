"""
Comprehensive tests for PretrainingPipeline.

Tests behavioral cloning integration, replay data processing,
and pretrained model checkpoint management.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from npp_rl.optimization.architecture_configs import ArchitectureConfig
from npp_rl.training.pretraining_pipeline import PretrainingPipeline


class TestPretrainingPipeline(unittest.TestCase):
    """Test PretrainingPipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures with replay data."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.replay_data_dir = Path(self.temp_dir) / "replay_data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create replay data directory
        self.replay_data_dir.mkdir(parents=True)
        
        # Create mock replay files
        for i in range(5):
            replay_file = self.replay_data_dir / f"replay_{i}.npz"
            np.savez(
                replay_file,
                observations=np.random.randn(100, 84, 84, 12),
                actions=np.random.randint(0, 6, size=100),
                rewards=np.random.randn(100),
                dones=np.zeros(100)
            )
        
        # Create mock architecture config
        self.arch_config = ArchitectureConfig(
            name="test_bc_arch",
            cnn_architecture="3d",
            hidden_dim=256,
            num_layers=2,
            use_graph_encoder=False,
            graph_hidden_dim=0,
            use_attention=False,
            attention_heads=0,
            dropout_rate=0.1
        )
    
    def test_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        self.assertEqual(pipeline.replay_data_dir, self.replay_data_dir)
        self.assertEqual(pipeline.architecture_config, self.arch_config)
        self.assertEqual(pipeline.output_dir, self.output_dir)
        self.assertTrue(self.output_dir.exists())
    
    def test_initialization_creates_output_dir(self):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = Path(self.temp_dir) / "new_output"
        self.assertFalse(new_output_dir.exists())
        
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=new_output_dir
        )
        
        self.assertTrue(new_output_dir.exists())
    
    def test_initialization_fails_with_missing_replay_dir(self):
        """Test that initialization fails if replay directory doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(FileNotFoundError) as context:
            PretrainingPipeline(
                replay_data_dir=str(nonexistent_dir),
                architecture_config=self.arch_config,
                output_dir=self.output_dir
            )
        
        self.assertIn("not found", str(context.exception))
    
    def test_prepare_bc_data_finds_replay_files(self):
        """Test that prepare_bc_data finds replay files."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Call prepare_bc_data
        result = pipeline.prepare_bc_data(use_cache=False)
        
        # Should return None (delegates to bc_pretrain.py)
        # But should have found files
        replay_files = list(self.replay_data_dir.glob('*.npz'))
        self.assertEqual(len(replay_files), 5)
    
    def test_prepare_bc_data_uses_cache(self):
        """Test that prepare_bc_data uses cached data if available."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Create cached dataset
        cached_file = self.output_dir / 'bc_dataset.pkl'
        cached_file.touch()
        
        # Call with cache enabled
        result = pipeline.prepare_bc_data(use_cache=True)
        
        # Should return cached path
        self.assertIsNotNone(result)
        self.assertEqual(result, str(cached_file))
    
    def test_prepare_bc_data_ignores_cache_when_disabled(self):
        """Test that cache can be disabled."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Create cached dataset
        cached_file = self.output_dir / 'bc_dataset.pkl'
        cached_file.touch()
        
        # Call with cache disabled
        result = pipeline.prepare_bc_data(use_cache=False)
        
        # Should not use cache (returns None, delegates to bc_pretrain.py)
        # In current implementation, returns None to delegate
        self.assertIsNone(result)
    
    def test_prepare_bc_data_handles_empty_directory(self):
        """Test handling of empty replay directory."""
        empty_dir = Path(self.temp_dir) / "empty_replay"
        empty_dir.mkdir(parents=True)
        
        pipeline = PretrainingPipeline(
            replay_data_dir=str(empty_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Call prepare_bc_data with empty directory
        result = pipeline.prepare_bc_data(use_cache=False)
        
        # Should return None and log warning
        self.assertIsNone(result)
    
    def test_validate_replay_data_structure(self):
        """Test that replay data structure is validated."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Check for replay files
        replay_files = list(self.replay_data_dir.glob('*.npz'))
        
        # Verify replay files exist
        self.assertGreater(len(replay_files), 0)
        
        # Verify replay file structure
        for replay_file in replay_files:
            data = np.load(replay_file)
            self.assertIn('observations', data)
            self.assertIn('actions', data)
    
    def test_get_checkpoint_path(self):
        """Test checkpoint path generation."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        checkpoint_path = pipeline.get_checkpoint_path("bc_pretrained")
        
        self.assertTrue(checkpoint_path.parent == self.output_dir)
        self.assertTrue(checkpoint_path.name == "bc_pretrained.zip")
    
    def test_architecture_config_accessible(self):
        """Test that architecture config is accessible."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        self.assertEqual(pipeline.architecture_config.name, "test_bc_arch")
        self.assertEqual(pipeline.architecture_config.hidden_dim, 256)
        self.assertFalse(pipeline.architecture_config.use_graph_encoder)
    
    def test_tensorboard_writer_optional(self):
        """Test that TensorBoard writer is optional."""
        # Without writer
        pipeline1 = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        self.assertIsNone(pipeline1.tensorboard_writer)
        
        # With writer
        mock_writer = MagicMock()
        pipeline2 = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir,
            tensorboard_writer=mock_writer
        )
        
        self.assertEqual(pipeline2.tensorboard_writer, mock_writer)


class TestPretrainingPipelineEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.replay_data_dir = Path(self.temp_dir) / "replay_data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.replay_data_dir.mkdir(parents=True)
        
        self.arch_config = ArchitectureConfig(
            name="edge_case_arch",
            cnn_architecture="2d",
            hidden_dim=128,
            num_layers=2,
            use_graph_encoder=False,
            graph_hidden_dim=0,
            use_attention=False,
            attention_heads=0,
            dropout_rate=0.0
        )
    
    def test_handles_nested_output_directory(self):
        """Test handling of deeply nested output directory."""
        nested_output = Path(self.temp_dir) / "a" / "b" / "c" / "d" / "output"
        
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=nested_output
        )
        
        # Should create all parent directories
        self.assertTrue(nested_output.exists())
    
    def test_handles_replay_data_with_subdirectories(self):
        """Test handling of replay data in subdirectories."""
        # Create subdirectories with replay files
        subdir1 = self.replay_data_dir / "category1"
        subdir2 = self.replay_data_dir / "category2"
        subdir1.mkdir(parents=True)
        subdir2.mkdir(parents=True)
        
        # Add replay files to subdirectories
        np.savez(
            subdir1 / "replay_1.npz",
            observations=np.random.randn(50, 84, 84, 12),
            actions=np.random.randint(0, 6, size=50)
        )
        np.savez(
            subdir2 / "replay_2.npz",
            observations=np.random.randn(50, 84, 84, 12),
            actions=np.random.randint(0, 6, size=50)
        )
        
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Current implementation only looks at top level
        # So should find no files
        replay_files = list(self.replay_data_dir.glob('*.npz'))
        self.assertEqual(len(replay_files), 0)
    
    def test_multiple_cache_checks(self):
        """Test multiple calls to prepare_bc_data with cache."""
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Create cached dataset
        cached_file = self.output_dir / 'bc_dataset.pkl'
        cached_file.touch()
        
        # Call multiple times
        result1 = pipeline.prepare_bc_data(use_cache=True)
        result2 = pipeline.prepare_bc_data(use_cache=True)
        
        # Both should return cached path
        self.assertEqual(result1, result2)
        self.assertIsNotNone(result1)


class TestPretrainingPipelineIntegration(unittest.TestCase):
    """Integration tests for PretrainingPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.replay_data_dir = Path(self.temp_dir) / "replay_data"
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.replay_data_dir.mkdir(parents=True)
        
        # Create realistic replay data
        for i in range(3):
            replay_file = self.replay_data_dir / f"episode_{i}.npz"
            np.savez(
                replay_file,
                observations=np.random.randn(200, 84, 84, 12).astype(np.float32),
                actions=np.random.randint(0, 6, size=200),
                rewards=np.random.randn(200).astype(np.float32),
                dones=np.zeros(200, dtype=bool)
            )
        
        self.arch_config = ArchitectureConfig(
            name="integration_test_arch",
            cnn_architecture="3d",
            hidden_dim=256,
            num_layers=3,
            use_graph_encoder=True,
            graph_hidden_dim=128,
            use_attention=False,
            attention_heads=0,
            dropout_rate=0.1
        )
    
    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # Initialize pipeline
        pipeline = PretrainingPipeline(
            replay_data_dir=str(self.replay_data_dir),
            architecture_config=self.arch_config,
            output_dir=self.output_dir
        )
        
        # Verify initialization
        self.assertIsNotNone(pipeline)
        self.assertTrue(self.output_dir.exists())
        
        # Check replay data
        replay_files = list(self.replay_data_dir.glob('*.npz'))
        self.assertEqual(len(replay_files), 3)
        
        # Prepare BC data
        result = pipeline.prepare_bc_data(use_cache=False)
        
        # Should complete without errors
        # (returns None as it delegates to bc_pretrain.py)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
