"""
Training utilities and classes for NPP-RL.

This package contains training-related functionality including:
- Behavioral cloning trainer
- Training utilities and helpers
- Policy creation and management
- Distributed training support
"""

from npp_rl.training.training_utils import create_training_policy
from npp_rl.training.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    wrap_model_ddp,
    AMPHelper,
    distribute_environments,
    configure_cuda_for_training,
    DistributedTrainingContext,
)
from npp_rl.training.curriculum_manager import (
    CurriculumManager,
    create_curriculum_manager,
)
from npp_rl.training.architecture_trainer import ArchitectureTrainer

__all__ = [
    "create_training_policy",
    "setup_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "wrap_model_ddp",
    "AMPHelper",
    "distribute_environments",
    "configure_cuda_for_training",
    "DistributedTrainingContext",
    "CurriculumManager",
    "create_curriculum_manager",
    "ArchitectureTrainer",
]
