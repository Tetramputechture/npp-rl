import numpy as np


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

        Args:
            x1, y1: Coordinates of first point
            x2, y2: Coordinates of second point

        Returns:
            float: Euclidean distance between the points
        """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
