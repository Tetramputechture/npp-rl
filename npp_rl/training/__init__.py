"""
Training utilities and classes for NPP-RL.

This package contains training-related functionality including:
- Behavioral cloning trainer
- Training utilities and helpers
- Policy creation and management
"""

from npp_rl.training.bc_trainer import BCTrainer
from npp_rl.training.training_utils import create_training_policy

__all__ = [
    'BCTrainer',
    'create_training_policy'
]
