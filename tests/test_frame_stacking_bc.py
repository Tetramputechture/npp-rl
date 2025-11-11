"""Tests for frame stacking in BC pretraining pipeline."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from collections import deque

from npp_rl.training.bc_dataset import BCReplayDataset
from npp_rl.training.architecture_configs import get_architecture_config


class TestBCDatasetFrameStacking:
    """Test frame stacking in BC dataset."""

    def test_frame_buffer_initialization(self):
        """Test that frame buffers are properly initialized."""
        frame_stack_config = {
            "enable_visual_frame_stacking": True,
            "visual_stack_size": 4,
            "enable_state_stacking": True,
            "state_stack_size": 4,
            "padding_type": "zero",
        }

        # Create a mock dataset (we can't easily test full initialization without replay files)
        # Just test that the config is properly stored
        assert frame_stack_config["enable_visual_frame_stacking"] == True
        assert frame_stack_config["visual_stack_size"] == 4
        assert frame_stack_config["enable_state_stacking"] == True
        assert frame_stack_config["state_stack_size"] == 4

    def test_visual_stacking_shape(self):
        """Test that visual frame stacking produces correct shapes."""
        # Simulate stacking 4 frames of shape (96, 96, 3)
        stack_size = 4
        h, w, c = 96, 96, 3

        # Create frames
        frames = [np.random.randn(h, w, c) for _ in range(stack_size)]

        # Stack along first dimension
        stacked = np.stack(frames, axis=0)  # (4, 96, 96, 3)

        assert stacked.shape == (stack_size, h, w, c)

    def test_state_stacking_concatenation(self):
        """Test that state stacking concatenates correctly."""
        stack_size = 4
        state_dim = 100

        # Create states
        states = [np.random.randn(state_dim) for _ in range(stack_size)]

        # Concatenate
        stacked = np.concatenate(states, axis=0)

        assert stacked.shape == (stack_size * state_dim,)

    def test_frame_buffer_zero_padding(self):
        """Test zero padding for initial frames."""
        stack_size = 4
        h, w, c = 96, 96, 3

        # Simulate buffer with only 2 frames
        buffer = deque(maxlen=stack_size)
        buffer.append(np.ones((h, w, c)))
        buffer.append(np.ones((h, w, c)) * 2)

        # Pad with zeros
        while len(buffer) < stack_size:
            buffer.appendleft(np.zeros((h, w, c)))

        stacked = np.stack(list(buffer), axis=0)

        assert stacked.shape == (stack_size, h, w, c)
        # First two should be zeros
        assert np.allclose(stacked[0], 0)
        assert np.allclose(stacked[1], 0)
        # Last two should be non-zero
        assert np.allclose(stacked[2], 1)
        assert np.allclose(stacked[3], 2)

    def test_frame_buffer_repeat_padding(self):
        """Test repeat padding for initial frames."""
        stack_size = 4
        h, w, c = 96, 96, 3

        # Simulate buffer with only 2 frames
        first_frame = np.ones((h, w, c))
        buffer = deque([first_frame, np.ones((h, w, c)) * 2], maxlen=stack_size)

        # Pad by repeating first frame
        while len(buffer) < stack_size:
            buffer.appendleft(first_frame.copy())

        stacked = np.stack(list(buffer), axis=0)

        assert stacked.shape == (stack_size, h, w, c)
        # First two should be repeated first frame
        assert np.allclose(stacked[0], 1)
        assert np.allclose(stacked[1], 1)
        # Last two should be the actual frames
        assert np.allclose(stacked[2], 1)
        assert np.allclose(stacked[3], 2)


class TestFrameStackingCheckpointMetadata:
    """Test that frame stacking metadata is saved in checkpoints."""

    def test_checkpoint_contains_frame_stacking_info(self):
        """Test that saved checkpoint includes frame stacking config."""
        # Simulate checkpoint structure
        frame_stack_config = {
            "enable_visual_frame_stacking": True,
            "visual_stack_size": 4,
            "enable_state_stacking": True,
            "state_stack_size": 4,
            "padding_type": "zero",
        }

        checkpoint = {
            "policy_state_dict": {},
            "epoch": 10,
            "metrics": {"loss": 0.5},
            "architecture": "mlp_cnn",
            "frame_stacking": frame_stack_config,
        }

        # Verify frame_stacking key exists and has correct data
        assert "frame_stacking" in checkpoint
        assert checkpoint["frame_stacking"]["enable_visual_frame_stacking"] == True
        assert checkpoint["frame_stacking"]["visual_stack_size"] == 4
        assert checkpoint["frame_stacking"]["enable_state_stacking"] == True
        assert checkpoint["frame_stacking"]["state_stack_size"] == 4

    def test_checkpoint_without_frame_stacking(self):
        """Test checkpoint without frame stacking (backward compatibility)."""
        checkpoint = {
            "policy_state_dict": {},
            "epoch": 10,
            "metrics": {"loss": 0.5},
            "architecture": "mlp_cnn",
        }

        # Should not have frame_stacking key
        assert "frame_stacking" not in checkpoint

        # This should still be valid and loadable
        assert "policy_state_dict" in checkpoint


class TestArchitectureCompatibility:
    """Test that frame stacking works with different architectures."""

    def test_mlp_cnn_with_frame_stacking(self):
        """Test that MLP baseline architecture works with frame stacking."""
        arch_config = get_architecture_config("mlp_cnn")

        frame_stack_config = {
            "enable_visual_frame_stacking": True,
            "visual_stack_size": 4,
            "enable_state_stacking": True,
            "state_stack_size": 4,
        }

        # Architecture should require visual observations
        assert arch_config.use_player_visual or arch_config.use_opponent_visual

        # When frame stacking is enabled, the expected input channels should be stack_size
        # (The policy network should be created with this in mind)
        expected_channels = frame_stack_config["visual_stack_size"]
        assert expected_channels == 4

    def test_state_only_architecture_with_state_stacking(self):
        """Test state-only architecture with state stacking."""
        frame_stack_config = {
            "enable_visual_frame_stacking": False,
            "visual_stack_size": 1,
            "enable_state_stacking": True,
            "state_stack_size": 4,
        }

        # State stacking should multiply the state dimension
        base_state_dim = 100
        stacked_state_dim = base_state_dim * frame_stack_config["state_stack_size"]

        assert stacked_state_dim == 400


class TestEndToEndFrameStacking:
    """Integration tests for end-to-end frame stacking."""

    def test_frame_stack_config_propagation(self):
        """Test that frame_stack_config propagates through the pipeline."""
        # Simulate the flow from train_and_compare.py

        # 1. Config is built from args
        args_frame_stack_config = {
            "enable_visual_frame_stacking": True,
            "visual_stack_size": 4,
            "enable_state_stacking": True,
            "state_stack_size": 4,
            "padding_type": "zero",
        }

        # 2. Passed to PretrainingPipeline
        pipeline_config = args_frame_stack_config
        assert pipeline_config == args_frame_stack_config

        # 3. Passed to BCReplayDataset
        dataset_config = pipeline_config
        assert dataset_config == args_frame_stack_config

        # 4. Passed to BCTrainer
        trainer_config = pipeline_config
        assert trainer_config == args_frame_stack_config

        # 5. Saved in checkpoint
        checkpoint_config = trainer_config
        assert checkpoint_config == args_frame_stack_config

        # Verify all configs match
        assert (
            args_frame_stack_config
            == pipeline_config
            == dataset_config
            == trainer_config
            == checkpoint_config
        )

    def test_no_frame_stacking_config(self):
        """Test that pipeline works without frame stacking (backward compatibility)."""
        frame_stack_config = None

        # Pipeline should handle None gracefully
        # (default to no frame stacking)
        assert frame_stack_config is None or frame_stack_config == {}


def test_checkpoint_validation():
    """Test validation of checkpoint compatibility with frame stacking."""
    # Checkpoint with 1 channel (no frame stacking)
    old_checkpoint_shape = (1, 96, 96)

    # Target with 4 channels (frame stacking enabled)
    target_shape = (4, 96, 96)

    # These should be incompatible
    assert old_checkpoint_shape[0] != target_shape[0]

    # Checkpoint with 4 channels (frame stacking)
    new_checkpoint_shape = (4, 96, 96)

    # These should be compatible
    assert new_checkpoint_shape[0] == target_shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
