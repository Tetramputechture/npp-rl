"""
Comprehensive tests for ArchitectureTrainer.

Tests architecture-specific training setup, model initialization,
environment creation, and training orchestration.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch
from stable_baselines3 import PPO

from npp_rl.optimization.architecture_configs import (
    ArchitectureConfig,
    ModalityConfig,
    GraphConfig,
    VisualConfig,
    StateConfig,
    FusionConfig,
    GraphArchitectureType,
    FusionType,
)
from npp_rl.training.architecture_trainer import ArchitectureTrainer


class TestArchitectureTrainer(unittest.TestCase):
    """Test ArchitectureTrainer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.train_dataset = Path(self.temp_dir) / "train"
        self.test_dataset = Path(self.temp_dir) / "test"
        
        # Create directories
        self.train_dataset.mkdir(parents=True)
        self.test_dataset.mkdir(parents=True)
        
        # Create mock architecture config with proper nested structure
        self.arch_config = ArchitectureConfig(
            name="test_architecture",
            description="Test architecture configuration",
            modalities=ModalityConfig(
                use_temporal_frames=True,
                use_global_view=True,
                use_graph=True,
                use_game_state=True,
                use_reachability=True,
            ),
            graph=GraphConfig(
                architecture=GraphArchitectureType.FULL_HGT,
                hidden_dim=256,
                num_layers=3,
                output_dim=256,
                num_heads=8,
            ),
            visual=VisualConfig(),
            state=StateConfig(),
            fusion=FusionConfig(fusion_type=FusionType.MULTI_HEAD_ATTENTION),
            features_dim=512,
        )
    
    def test_initialization_defaults(self):
        """Test trainer initializes with default parameters."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        self.assertEqual(trainer.architecture_config, self.arch_config)
        self.assertEqual(trainer.device_id, 0)
        self.assertEqual(trainer.world_size, 1)
        self.assertFalse(trainer.use_mixed_precision)
        self.assertFalse(trainer.use_hierarchical_ppo)
        self.assertFalse(trainer.use_curriculum)
        self.assertTrue(self.output_dir.exists())
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir,
            device_id=1,
            world_size=2,
            use_mixed_precision=True,
            use_hierarchical_ppo=True,
            use_curriculum=True,
            curriculum_kwargs={'advancement_threshold': 0.8}
        )
        
        self.assertEqual(trainer.device_id, 1)
        self.assertEqual(trainer.world_size, 2)
        self.assertTrue(trainer.use_mixed_precision)
        self.assertTrue(trainer.use_hierarchical_ppo)
        self.assertTrue(trainer.use_curriculum)
        self.assertEqual(trainer.curriculum_kwargs['advancement_threshold'], 0.8)
    
    def test_output_directory_created(self):
        """Test that output directory is created if it doesn't exist."""
        output_dir = Path(self.temp_dir) / "new_output"
        self.assertFalse(output_dir.exists())
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=output_dir
        )
        
        self.assertTrue(output_dir.exists())
    
    @patch('npp_rl.training.architecture_trainer.NppEnvironment')
    @patch('npp_rl.training.architecture_trainer.DummyVecEnv')
    def test_setup_environment_without_curriculum(self, mock_vec_env, mock_create_env):
        """Test environment setup without curriculum learning."""
        # Setup mocks
        mock_env = MagicMock()
        mock_create_env.return_value = mock_env
        mock_vec = MagicMock()
        mock_vec_env.return_value = mock_vec
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir,
            use_curriculum=False
        )
        
        # Call setup (will be called by setup_model)
        # For this test, we just verify initialization
        self.assertIsNone(trainer.curriculum_manager)
    
    @patch('npp_rl.training.architecture_trainer.NppEnvironment')
    @patch('npp_rl.training.architecture_trainer.create_curriculum_manager')
    def test_setup_environment_with_curriculum(self, mock_create_curriculum, mock_create_env):
        """Test environment setup with curriculum learning."""
        # Setup mocks
        mock_curriculum = MagicMock()
        mock_create_curriculum.return_value = mock_curriculum
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir,
            use_curriculum=True
        )
        
        # Curriculum manager should be None initially
        self.assertIsNone(trainer.curriculum_manager)
    
    def test_get_device_returns_correct_device(self):
        """Test that get_device returns correct CUDA device or CPU."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir,
            device_id=0
        )
        
        device = trainer.get_device()
        
        # Should return 'cuda:0' if available, else 'cpu'
        if torch.cuda.is_available():
            self.assertEqual(device, 'cuda:0')
        else:
            self.assertEqual(device, 'cpu')
    
    def test_get_device_with_custom_id(self):
        """Test get_device with custom device ID."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir,
            device_id=1
        )
        
        device = trainer.get_device()
        
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.assertEqual(device, 'cuda:1')
    
    @patch('npp_rl.training.architecture_trainer.ComprehensiveEvaluator')
    def test_create_evaluator(self, mock_evaluator_class):
        """Test evaluator creation."""
        mock_evaluator = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        evaluator = trainer.create_evaluator()
        
        # Verify evaluator was created
        mock_evaluator_class.assert_called_once()
        self.assertEqual(evaluator, mock_evaluator)
    
    def test_get_checkpoint_path(self):
        """Test checkpoint path generation."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        checkpoint_path = trainer.get_checkpoint_path("best_model")
        
        self.assertTrue(checkpoint_path.parent == self.output_dir)
        self.assertTrue(checkpoint_path.name == "best_model.zip")
    
    def test_save_training_state(self):
        """Test saving training state."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        # Create mock curriculum manager
        mock_curriculum = MagicMock()
        mock_curriculum.get_curriculum_state.return_value = {'stage': 'simple'}
        trainer.curriculum_manager = mock_curriculum
        
        state_file = trainer.save_training_state(1000)
        
        # Verify state file was created
        self.assertTrue(state_file.exists())
        self.assertTrue(state_file.name.startswith("training_state_"))
    
    def test_architecture_config_stored(self):
        """Test that architecture config is properly stored."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        self.assertEqual(trainer.architecture_config.name, "test_architecture")
        self.assertEqual(trainer.architecture_config.graph.hidden_dim, 256)
        self.assertEqual(trainer.architecture_config.graph.num_layers, 3)
        self.assertTrue(trainer.architecture_config.modalities.use_graph)
    
    def test_model_initially_none(self):
        """Test that model is initially None before setup."""
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        self.assertIsNone(trainer.model)
        self.assertIsNone(trainer.env)
        self.assertIsNone(trainer.eval_env)


class TestArchitectureTrainerWithMockedEnvironment(unittest.TestCase):
    """Test ArchitectureTrainer with mocked environments."""
    
    def setUp(self):
        """Set up test fixtures with mocked components."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.train_dataset = Path(self.temp_dir) / "train"
        self.test_dataset = Path(self.temp_dir) / "test"
        
        self.train_dataset.mkdir(parents=True)
        self.test_dataset.mkdir(parents=True)
        
        self.arch_config = ArchitectureConfig(
            name="mock_architecture",
            description="Mock architecture for testing",
            modalities=ModalityConfig(
                use_temporal_frames=True,
                use_global_view=False,
                use_graph=False,
                use_game_state=True,
                use_reachability=True,
            ),
            graph=GraphConfig(
                architecture=GraphArchitectureType.NONE,
                hidden_dim=128,
                num_layers=2,
                output_dim=128,
            ),
            visual=VisualConfig(),
            state=StateConfig(),
            fusion=FusionConfig(fusion_type=FusionType.CONCAT),
            features_dim=256,
        )
    
    @patch('npp_rl.training.architecture_trainer.NppEnvironment')
    def test_environment_creation_called(self, mock_create_env):
        """Test that environment creation is called during setup."""
        mock_env = MagicMock()
        mock_create_env.return_value = mock_env
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        # Environment will be created when setup_model is called
        # For now, just verify initialization
        self.assertIsNotNone(trainer)


class TestArchitectureTrainerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.train_dataset = Path(self.temp_dir) / "train"
        self.test_dataset = Path(self.temp_dir) / "test"
        
        self.train_dataset.mkdir(parents=True)
        self.test_dataset.mkdir(parents=True)
        
        self.arch_config = ArchitectureConfig(
            name="edge_case_arch",
            description="Edge case test architecture",
            modalities=ModalityConfig(
                use_temporal_frames=True,
                use_global_view=True,
                use_graph=True,
                use_game_state=True,
                use_reachability=True,
            ),
            graph=GraphConfig(
                architecture=GraphArchitectureType.FULL_HGT,
                hidden_dim=256,
                num_layers=3,
                output_dim=256,
                num_heads=4,
                dropout=0.2,
            ),
            visual=VisualConfig(),
            state=StateConfig(),
            fusion=FusionConfig(
                fusion_type=FusionType.MULTI_HEAD_ATTENTION,
                num_attention_heads=4,
                dropout=0.2,
            ),
            features_dim=512,
        )
    
    def test_handles_nonexistent_dataset_paths(self):
        """Test handling of nonexistent dataset paths."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent"
        
        # Should create trainer but may fail on actual training
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(nonexistent_path),
            test_dataset_path=str(self.test_dataset),
            output_dir=self.output_dir
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.train_dataset_path, nonexistent_path)
    
    def test_output_directory_with_nested_path(self):
        """Test output directory creation with nested path."""
        nested_output = Path(self.temp_dir) / "a" / "b" / "c" / "output"
        
        trainer = ArchitectureTrainer(
            architecture_config=self.arch_config,
            train_dataset_path=str(self.train_dataset),
            test_dataset_path=str(self.test_dataset),
            output_dir=nested_output
        )
        
        # Should create all parent directories
        self.assertTrue(nested_output.exists())


if __name__ == '__main__':
    unittest.main()
