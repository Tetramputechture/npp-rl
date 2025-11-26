"""Environment wrappers for enhanced RL training."""

from .curriculum_env import CurriculumEnv, CurriculumVecEnvWrapper, make_curriculum_env
# PositionTrackingWrapper removed - position tracking now integrated into BaseNppEnvironment
from .gpu_observation_wrapper import GPUObservationWrapper

__all__ = [
    "CurriculumEnv",
    "CurriculumVecEnvWrapper",
    "make_curriculum_env",
    # "PositionTrackingWrapper",  # Removed - integrated into BaseNppEnvironment
    "GPUObservationWrapper",
]
