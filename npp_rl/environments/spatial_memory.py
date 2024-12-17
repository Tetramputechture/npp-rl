import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Configuration for spatial memory tracking"""
    position_grid_size: int = 10
    area_grid_size: int = 50
    vision_radius: int = 8
    position_memory_size: Tuple[int, int] = (88, 88)
    area_memory_size: Tuple[int, int] = (19, 19)


class SpatialMemoryTracker:
    """Tracks the agent's movement through the environment space"""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()

        # Initialize memory matrices
        self.position_memory = np.zeros(
            self.config.position_memory_size, dtype=np.float32)
        self.area_memory = np.zeros(
            self.config.area_memory_size, dtype=np.float32)
        self.visit_frequency = np.zeros(
            self.config.position_memory_size, dtype=np.float32)
        self.transition_memory = np.zeros(
            self.config.area_memory_size, dtype=np.float32)

        self.prev_area_coords = None

    def update(self, x: float, y: float) -> None:
        """Update spatial memory with new position"""
        # Convert world coordinates to grid coordinates
        grid_x = int(x / self.config.position_grid_size)
        grid_y = int(y / self.config.position_grid_size)
        area_x = int(x / self.config.area_grid_size)
        area_y = int(y / self.config.area_grid_size)

        # Update position memory (recent visits)
        decay_factor = 0.95
        self.position_memory *= decay_factor
        self.position_memory[grid_y, grid_x] = 1.0

        # Update visit frequency
        self.visit_frequency[grid_y, grid_x] += 1
        self.visit_frequency = np.clip(self.visit_frequency / 10.0, 0, 1)

        # Update area memory
        self.area_memory[area_y, area_x] = 1.0

        # Update transition memory
        if self.prev_area_coords is not None:
            if (area_x, area_y) != self.prev_area_coords:
                prev_x, prev_y = self.prev_area_coords
                self.transition_memory[prev_y, prev_x] = 1.0
                self.transition_memory[area_y, area_x] = 1.0

        self.prev_area_coords = (area_x, area_y)

    def get_centered_view(self, x: float, y: float, memory_matrix: np.ndarray,
                          grid_size: float) -> np.ndarray:
        """Create a view of the memory matrix centered on current position"""
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)

        view = np.zeros((88, 88), dtype=np.float32)
        radius = self.config.vision_radius

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                mem_x = grid_x + j
                mem_y = grid_y + i

                if 0 <= mem_y < memory_matrix.shape[0] and \
                   0 <= mem_x < memory_matrix.shape[1]:
                    obs_x = 44 + j
                    obs_y = 44 + i
                    if 0 <= obs_y < 88 and 0 <= obs_x < 88:
                        view[obs_y, obs_x] = memory_matrix[mem_y, mem_x]

        return view

    def get_exploration_maps(self, x: float, y: float) -> Dict[str, np.ndarray]:
        """Get all exploration-related maps centered on current position"""
        return {
            'recent_visits': self.get_centered_view(x, y, self.position_memory,
                                                    self.config.position_grid_size),
            'visit_frequency': self.get_centered_view(x, y, self.visit_frequency,
                                                      self.config.position_grid_size),
            'area_exploration': self.get_centered_view(x, y, self.area_memory,
                                                       self.config.area_grid_size),
            'transitions': self.get_centered_view(x, y, self.transition_memory,
                                                  self.config.area_grid_size)
        }

    def reset(self) -> None:
        """Reset all memory matrices"""
        self.position_memory.fill(0)
        self.area_memory.fill(0)
        self.visit_frequency.fill(0)
        self.transition_memory.fill(0)
        self.prev_area_coords = None
