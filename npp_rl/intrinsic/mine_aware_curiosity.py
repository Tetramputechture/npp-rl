"""
Mine-Aware Curiosity Modulation for ICM.

This module provides functionality to modulate ICM curiosity rewards
based on mine proximity, reducing exploration in dangerous areas and
encouraging safe exploration paths.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class MineAwareCuriosityModulator:
    """
    Modulates ICM curiosity rewards based on mine proximity.
    
    This modulator reduces curiosity near dangerous mines while maintaining
    or increasing curiosity in safe areas, helping agents learn to avoid
    mines while still exploring effectively.
    """
    
    DANGER_THRESHOLD = 48.0  # 2 tiles (24px per tile)
    SAFE_THRESHOLD = 96.0  # 4 tiles
    
    def __init__(
        self,
        min_modulation: float = 0.1,
        max_modulation: float = 1.0,
        safe_boost: float = 1.2,
        debug: bool = False,
    ):
        """
        Initialize mine-aware curiosity modulator.
        
        Args:
            min_modulation: Minimum curiosity modulation in danger zones (0.1 = 10% curiosity)
            max_modulation: Maximum curiosity modulation in safe zones (1.0 = 100% curiosity)
            safe_boost: Boost factor for safe exploration areas (>1.0 encourages safe exploration)
            debug: Enable debug output
        """
        self.min_modulation = min_modulation
        self.max_modulation = max_modulation
        self.safe_boost = safe_boost
        self.debug = debug
        
        self._stats = {
            'total_modulations': 0,
            'danger_zone_count': 0,
            'safe_zone_count': 0,
            'average_modulation': 0.0,
        }
    
    def modulate_curiosity(
        self,
        base_curiosity: float,
        mine_proximity: float,
        is_path_safe: bool = True,
    ) -> float:
        """
        Modulate curiosity reward based on mine proximity.
        
        Args:
            base_curiosity: Base ICM curiosity reward
            mine_proximity: Distance to nearest dangerous mine (in pixels)
            is_path_safe: Whether the current path is safe from mines
            
        Returns:
            Modulated curiosity reward
        """
        if mine_proximity >= self.SAFE_THRESHOLD:
            # Safe area - optionally boost curiosity to encourage exploration
            modulation = self.safe_boost if is_path_safe else self.max_modulation
            self._stats['safe_zone_count'] += 1
        elif mine_proximity <= self.DANGER_THRESHOLD:
            # Danger zone - significantly reduce curiosity
            modulation = self.min_modulation
            self._stats['danger_zone_count'] += 1
        else:
            # Transition zone - linear interpolation
            range_size = self.SAFE_THRESHOLD - self.DANGER_THRESHOLD
            normalized_distance = (mine_proximity - self.DANGER_THRESHOLD) / range_size
            modulation = self.min_modulation + (self.max_modulation - self.min_modulation) * normalized_distance
        
        # Further reduce if path is not safe
        if not is_path_safe:
            modulation *= 0.5
        
        modulated_curiosity = base_curiosity * modulation
        
        # Update statistics
        self._stats['total_modulations'] += 1
        self._update_running_average(modulation)
        
        if self.debug:
            print(f"Mine-aware curiosity: proximity={mine_proximity:.1f}px, "
                  f"modulation={modulation:.3f}, "
                  f"curiosity: {base_curiosity:.4f} -> {modulated_curiosity:.4f}")
        
        return modulated_curiosity
    
    def modulate_curiosity_batch(
        self,
        base_curiosity: np.ndarray,
        mine_proximities: np.ndarray,
        path_safety: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Modulate curiosity for a batch of samples.
        
        Args:
            base_curiosity: [batch_size] base ICM curiosity rewards
            mine_proximities: [batch_size] distances to nearest dangerous mines
            path_safety: [batch_size] optional path safety indicators
            
        Returns:
            [batch_size] modulated curiosity rewards
        """
        batch_size = base_curiosity.shape[0]
        modulated = np.zeros_like(base_curiosity)
        
        if path_safety is None:
            path_safety = np.ones(batch_size, dtype=bool)
        
        for i in range(batch_size):
            modulated[i] = self.modulate_curiosity(
                base_curiosity[i],
                mine_proximities[i],
                bool(path_safety[i])
            )
        
        return modulated
    
    def calculate_exploration_bias(
        self,
        ninja_position: Tuple[float, float],
        target_position: Tuple[float, float],
        mine_positions: list,
        mine_states: list,
    ) -> float:
        """
        Calculate exploration bias for a direction based on mine configuration.
        
        Higher bias encourages exploration in that direction, lower bias discourages it.
        
        Args:
            ninja_position: Current ninja position (x, y)
            target_position: Target exploration position (x, y)
            mine_positions: List of (x, y) mine positions
            mine_states: List of mine states (0=toggled, 1=untoggled, 2=toggling)
            
        Returns:
            Exploration bias factor [0, 1]
        """
        if not mine_positions:
            return 1.0
        
        # Calculate direction vector to target
        direction = np.array(target_position) - np.array(ninja_position)
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return 1.0
        
        direction = direction / direction_norm
        
        # Check for dangerous mines along direction
        danger_score = 0.0
        
        for mine_pos, mine_state in zip(mine_positions, mine_states):
            # Skip safe mines
            if mine_state == 1:  # Untoggled = safe
                continue
            
            # Vector from ninja to mine
            to_mine = np.array(mine_pos) - np.array(ninja_position)
            to_mine_norm = np.linalg.norm(to_mine)
            
            if to_mine_norm < 1e-6:
                danger_score += 1.0
                continue
            
            to_mine = to_mine / to_mine_norm
            
            # Calculate alignment with exploration direction
            alignment = np.dot(direction, to_mine)
            
            # If mine is in the exploration direction and close
            if alignment > 0.5 and to_mine_norm < self.SAFE_THRESHOLD:
                proximity_factor = 1.0 - (to_mine_norm / self.SAFE_THRESHOLD)
                danger_score += alignment * proximity_factor
        
        # Convert danger score to bias (less danger = more bias)
        max_danger = len(mine_positions)
        if max_danger > 0:
            normalized_danger = min(danger_score / max_danger, 1.0)
            bias = 1.0 - normalized_danger * 0.8  # Max 80% reduction
        else:
            bias = 1.0
        
        return max(bias, 0.2)  # Minimum 20% bias to allow some exploration
    
    def get_safe_exploration_zones(
        self,
        ninja_position: Tuple[float, float],
        mine_positions: list,
        mine_states: list,
        grid_resolution: int = 8,
    ) -> np.ndarray:
        """
        Generate a safety map for exploration around the ninja.
        
        Args:
            ninja_position: Current ninja position (x, y)
            mine_positions: List of (x, y) mine positions
            mine_states: List of mine states
            grid_resolution: Number of directions to check
            
        Returns:
            [grid_resolution] array of safety scores for each direction
        """
        safety_scores = np.ones(grid_resolution)
        
        if not mine_positions:
            return safety_scores
        
        # Check each direction
        for i in range(grid_resolution):
            angle = 2 * np.pi * i / grid_resolution
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Sample point in this direction
            sample_distance = self.SAFE_THRESHOLD
            sample_point = (
                ninja_position[0] + direction[0] * sample_distance,
                ninja_position[1] + direction[1] * sample_distance,
            )
            
            # Check proximity to dangerous mines
            min_mine_distance = float('inf')
            
            for mine_pos, mine_state in zip(mine_positions, mine_states):
                if mine_state == 1:  # Skip safe mines
                    continue
                
                distance = np.linalg.norm(np.array(sample_point) - np.array(mine_pos))
                min_mine_distance = min(min_mine_distance, distance)
            
            # Convert to safety score
            if min_mine_distance >= self.SAFE_THRESHOLD:
                safety_scores[i] = 1.0
            elif min_mine_distance <= self.DANGER_THRESHOLD:
                safety_scores[i] = 0.0
            else:
                range_size = self.SAFE_THRESHOLD - self.DANGER_THRESHOLD
                safety_scores[i] = (min_mine_distance - self.DANGER_THRESHOLD) / range_size
        
        return safety_scores
    
    def _update_running_average(self, modulation: float) -> None:
        """Update running average of modulation factors."""
        total = self._stats['total_modulations']
        current_avg = self._stats['average_modulation']
        
        # Incremental average update
        self._stats['average_modulation'] = (
            current_avg * (total - 1) + modulation
        ) / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about curiosity modulation.
        
        Returns:
            Dictionary with modulation statistics
        """
        stats = self._stats.copy()
        
        if stats['total_modulations'] > 0:
            stats['danger_zone_percentage'] = (
                stats['danger_zone_count'] / stats['total_modulations'] * 100
            )
            stats['safe_zone_percentage'] = (
                stats['safe_zone_count'] / stats['total_modulations'] * 100
            )
        else:
            stats['danger_zone_percentage'] = 0.0
            stats['safe_zone_percentage'] = 0.0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset modulation statistics."""
        self._stats = {
            'total_modulations': 0,
            'danger_zone_count': 0,
            'safe_zone_count': 0,
            'average_modulation': 0.0,
        }


def modulate_curiosity_for_mines(
    curiosity_reward: float,
    mine_proximity: float,
    danger_threshold: float = 48.0,
    safe_threshold: float = 96.0,
) -> float:
    """
    Simple utility function to modulate curiosity based on mine proximity.
    
    Args:
        curiosity_reward: Base curiosity reward
        mine_proximity: Distance to nearest dangerous mine
        danger_threshold: Distance threshold for danger zone
        safe_threshold: Distance threshold for safe zone
        
    Returns:
        Modulated curiosity reward
    """
    if mine_proximity >= safe_threshold:
        return curiosity_reward  # Full exploration when safe
    elif mine_proximity <= danger_threshold:
        return curiosity_reward * 0.1  # Reduce exploration near danger
    else:
        # Linear interpolation in transition zone
        range_size = safe_threshold - danger_threshold
        factor = (mine_proximity - danger_threshold) / range_size
        return curiosity_reward * (0.1 + 0.9 * factor)
