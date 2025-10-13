"""
Progress tracking utilities for hierarchical RL subtasks.

Provides classes for tracking progress toward objectives and exploration coverage.
"""

import numpy as np


class ProgressTracker:
    """
    Track progress toward a specific objective.
    
    Maintains best distance achieved and step count for a subtask.
    """
    
    def __init__(self):
        self.best_distance = None
        self.steps = 0
    
    def update_distance(self, distance: float):
        """Update best distance if this is an improvement."""
        if self.best_distance is None or distance < self.best_distance:
            self.best_distance = distance
    
    def get_best_distance(self) -> float:
        """Get best distance achieved."""
        return self.best_distance if self.best_distance is not None else float('inf')
    
    def has_previous_distance(self) -> bool:
        """Check if we have a previous distance measurement."""
        return self.best_distance is not None
    
    def increment_steps(self):
        """Increment step counter."""
        self.steps += 1
    
    def get_steps(self) -> int:
        """Get number of steps in current subtask."""
        return self.steps
    
    def reset(self):
        """Reset tracker for new episode or subtask."""
        self.best_distance = None
        self.steps = 0


class ExplorationTracker(ProgressTracker):
    """
    Track exploration progress.
    
    Extends ProgressTracker to also track visited locations for exploration
    reward calculation.
    """
    
    def __init__(self, grid_size: float = 10.0):
        """
        Initialize exploration tracker.
        
        Args:
            grid_size: Size of grid cells for discretizing visited locations
        """
        super().__init__()
        self.grid_size = grid_size
        self.visited_locations = set()
    
    def visit_new_location(self, position: np.ndarray) -> bool:
        """
        Record visit to a location and return whether it's new.
        
        Args:
            position: [x, y] position
            
        Returns:
            True if this is a new location, False otherwise
        """
        grid_x = int(position[0] / self.grid_size)
        grid_y = int(position[1] / self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        if grid_cell not in self.visited_locations:
            self.visited_locations.add(grid_cell)
            return True
        return False
    
    def reset(self):
        """Reset tracker including visited locations."""
        super().reset()
        self.visited_locations.clear()
