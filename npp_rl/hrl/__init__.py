"""
Hierarchical Reinforcement Learning (HRL) module for NPP-RL.

This module implements hierarchical RL components including:
- Completion controller for strategic planning
- Hierarchical policy architectures
- Subtask management and transitions
"""

from .completion_controller import CompletionController

__all__ = ["CompletionController"]