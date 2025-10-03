"""
Hierarchical Reinforcement Learning (HRL) module for NPP-RL.

This module implements hierarchical RL components including:
- Completion controller for strategic planning
- Hierarchical policy architectures
- Subtask management and transitions
- Two-level policy architecture (Phase 2 Task 2.1)
"""

from .completion_controller import CompletionController
from .high_level_policy import (
    HighLevelPolicy,
    Subtask,
    SubtaskTransitionManager,
)
from .subtask_policies import (
    LowLevelPolicy,
    SubtaskEmbedding,
    SubtaskContextEncoder,
    ICMIntegration,
    SubtaskSpecificFeatures,
)

__all__ = [
    "CompletionController",
    "HighLevelPolicy",
    "Subtask",
    "SubtaskTransitionManager",
    "LowLevelPolicy",
    "SubtaskEmbedding",
    "SubtaskContextEncoder",
    "ICMIntegration",
    "SubtaskSpecificFeatures",
]