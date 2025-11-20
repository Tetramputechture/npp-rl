"""
Training utilities and classes for NPP-RL.

This package contains training-related functionality including:
- High-performance curriculum learning
- Behavioral cloning trainer
- Training utilities and helpers
- Policy creation and management
- Distributed training support
"""

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

# Import curriculum components
from npp_rl.training.curriculum_manager import CurriculumManager, create_curriculum_manager
from npp_rl.training.curriculum_factory import (
    create_curriculum_for_parallel_training,
    create_modular_curriculum_manager,
    create_shared_memory_curriculum_manager,
)
from npp_rl.training.bc_dataset import BCReplayDataset
from npp_rl.training.bc_trainer import BCTrainer
from npp_rl.training.policy_utils import (
    create_policy_network,
    save_policy_checkpoint,
    load_policy_checkpoint,
    count_parameters,
    log_model_info,
)
from npp_rl.training.pretraining_pipeline import (
    PretrainingPipeline,
    run_bc_pretraining_if_available,
)

__all__ = [
    # Distributed training utilities
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
    
    # Curriculum learning
    "CurriculumManager",
    "create_curriculum_manager",
    "create_curriculum_for_parallel_training",
    "create_modular_curriculum_manager",
    "create_shared_memory_curriculum_manager",
    
    # Training components
    "BCReplayDataset",
    "BCTrainer",
    "create_policy_network",
    "save_policy_checkpoint",
    "load_policy_checkpoint",
    "count_parameters",
    "log_model_info",
    "PretrainingPipeline",
    "run_bc_pretraining_if_available",
]
