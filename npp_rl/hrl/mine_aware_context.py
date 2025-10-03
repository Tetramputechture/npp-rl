"""
Mine-Aware Context for Subtask Execution

This module provides utilities for mine-aware subtask execution,
including danger scoring, path safety evaluation, and priority modulation.
"""

import numpy as np
from typing import Tuple


class MineAwareSubtaskContext:
    """
    Context manager for mine-aware subtask execution.
    
    This class provides utilities for checking mine safety and
    modulating actions based on mine proximity during subtask execution.
    """
    
    DANGER_THRESHOLD = 48.0  # 2 tiles (24px per tile)
    SAFE_THRESHOLD = 96.0  # 4 tiles
    
    def __init__(self):
        """Initialize mine-aware context."""
        self._last_mine_check = 0.0
    
    @staticmethod
    def calculate_mine_danger_score(mine_proximity: float) -> float:
        """
        Calculate danger score from mine proximity.
        
        Args:
            mine_proximity: Distance to nearest dangerous mine
            
        Returns:
            Danger score in [0, 1] where 1 = extreme danger
        """
        if mine_proximity >= MineAwareSubtaskContext.SAFE_THRESHOLD:
            return 0.0
        elif mine_proximity <= MineAwareSubtaskContext.DANGER_THRESHOLD:
            return 1.0
        else:
            # Linear interpolation
            range_size = MineAwareSubtaskContext.SAFE_THRESHOLD - MineAwareSubtaskContext.DANGER_THRESHOLD
            return 1.0 - (mine_proximity - MineAwareSubtaskContext.DANGER_THRESHOLD) / range_size
    
    @staticmethod
    def should_prioritize_mine_avoidance(mine_proximity: float) -> bool:
        """
        Check if mine avoidance should override subtask actions.
        
        Args:
            mine_proximity: Distance to nearest dangerous mine
            
        Returns:
            True if mine avoidance should take priority
        """
        return mine_proximity < MineAwareSubtaskContext.DANGER_THRESHOLD
    
    @staticmethod
    def get_safe_path_score(
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        mines_blocking_path: int,
    ) -> float:
        """
        Calculate safety score for a path.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            mines_blocking_path: Number of dangerous mines blocking the path
            
        Returns:
            Safety score in [0, 1] where 1 = completely safe
        """
        if mines_blocking_path == 0:
            return 1.0
        
        # Reduce safety score based on number of blocking mines
        penalty = 0.5 ** mines_blocking_path
        return penalty
    
    @staticmethod
    def modulate_subtask_priority(
        base_priority: float,
        path_safety: float,
        mine_danger: float,
    ) -> float:
        """
        Modulate subtask priority based on mine safety.
        
        Args:
            base_priority: Base priority of subtask [0, 1]
            path_safety: Safety score of path to subtask [0, 1]
            mine_danger: Danger score from nearby mines [0, 1]
            
        Returns:
            Modulated priority [0, 1]
        """
        # Reduce priority if path is unsafe or mines are nearby
        safety_factor = (1.0 - mine_danger) * path_safety
        return base_priority * safety_factor
