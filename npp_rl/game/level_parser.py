import numpy as np
import cv2
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from npp_rl.game.game_window import get_game_window_frame

# Constants for the game's visual layout
CELL_SIZE = 28  # Size of each grid cell in pixels
GRID_ROWS = 23  # Number of rows in the grid
GRID_COLS = 42  # Number of columns in the grid
BACKGROUND_COLOR = np.array([79, 86, 77])  # RGB values for background
COLOR_TOLERANCE = 5  # Tolerance for color matching

# Load mine template
MINE_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), 'mine.png'))
if MINE_TEMPLATE is None:
    raise FileNotFoundError("Could not load mine template image")
# Convert to RGB to match our game frame
MINE_TEMPLATE = cv2.cvtColor(MINE_TEMPLATE, cv2.COLOR_BGR2RGB)

# Template matching parameters
MATCH_THRESHOLD = 0.8  # Confidence threshold for template matching


@dataclass
class ParsedLevel:
    """Class to store parsed level information."""
    playable_space: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    mine_coordinates: List[Tuple[int, int]]  # List of (x, y) coordinates
    width: int
    height: int

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get the level boundaries."""
        return self.playable_space

    def is_mine_at(self, x: int, y: int) -> bool:
        """Check if there is a mine at the given coordinates."""
        return (x, y) in self.mine_coordinates

    def get_nearest_mine(self, x: int, y: int, max_distance: float = float('inf')) -> Optional[Tuple[int, int]]:
        """Find the coordinates of the mine nearest to the given position within max_distance.

        Args:
            x: X coordinate of the position
            y: Y coordinate of the position
            max_distance: Maximum distance to consider mines (in pixels). Defaults to infinity.

        Returns:
            Optional[Tuple[int, int]]: Coordinates of the nearest mine within max_distance,
                                      or None if no mines are within range
        """
        if not self.mine_coordinates:
            return None

        # Calculate distances to all mines
        distances = [
            ((mine_x - x) ** 2 + (mine_y - y) ** 2, (mine_x, mine_y))
            for mine_x, mine_y in self.mine_coordinates
        ]

        # Filter mines within max_distance (comparing squared distances)
        max_distance_squared = max_distance ** 2
        valid_mines = [(d, coords)
                       for d, coords in distances if d <= max_distance_squared]

        if not valid_mines:
            return None

        # Return coordinates of the mine with minimum distance
        return min(valid_mines, key=lambda x: x[0])[1]


def parse_level() -> ParsedLevel:
    """Parse the current level and return a ParsedLevel object.

    Returns:
        ParsedLevel: Object containing level information
    """
    frame = get_game_window_frame()

    # Get playable space
    playable_space = _get_playable_space_coordinates(frame)
    width, height = playable_space[2] - \
        playable_space[0], playable_space[3] - playable_space[1]

    # Get mine coordinates
    mine_coords = _get_mine_coordinates(frame, playable_space)

    return ParsedLevel(
        playable_space=playable_space,
        mine_coordinates=mine_coords,
        width=width,
        height=height
    )


def _get_playable_space_coordinates(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the coordinates of the playable space."""
    # Verify frame dimensions and type
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be an RGB image (H, W, 3)")

    height, width = frame.shape[:2]
    active_rows = []
    active_cols = []

    # Analyze each grid cell
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y_start = row * CELL_SIZE
            y_end = y_start + CELL_SIZE
            x_start = col * CELL_SIZE
            x_end = x_start + CELL_SIZE

            if y_end > height or x_end > width:
                continue

            cell = frame[y_start:y_end, x_start:x_end]
            if not _is_background_cell(cell):
                active_rows.append(row)
                active_cols.append(col)

    if not active_rows or not active_cols:
        return (0, 0, width, height)

    # Calculate boundaries
    min_row, max_row = min(active_rows), max(active_rows) + 1
    min_col, max_col = min(active_cols), max(active_cols) + 1

    # Convert to pixel coordinates with padding
    padding = 2
    min_x = max(0, min_col * CELL_SIZE - padding)
    min_y = max(0, min_row * CELL_SIZE - padding)
    max_x = min(width, max_col * CELL_SIZE + padding)
    max_y = min(height, max_row * CELL_SIZE + padding)

    return (min_x, min_y, max_x, max_y)


def _get_mine_coordinates(frame: np.ndarray, bounds: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """Get coordinates of all mines within the playable space using template matching."""
    min_x, min_y, max_x, max_y = bounds
    mine_coords = []

    # Crop frame to bounds
    roi = frame[min_y:max_y, min_x:max_x]

    # Perform template matching
    result = cv2.matchTemplate(roi, MINE_TEMPLATE, cv2.TM_CCOEFF_NORMED)

    # Find positions where match quality exceeds our threshold
    locations = np.where(result >= MATCH_THRESHOLD)

    # Convert the locations to x,y coordinates
    # Reverse locations for x,y instead of row,col
    for pt in zip(*locations[::-1]):
        # Add the template center offset and the ROI offset
        x = min_x + pt[0] + MINE_TEMPLATE.shape[1] // 2
        y = min_y + pt[1] + MINE_TEMPLATE.shape[0] // 2
        mine_coords.append((x, y))

    return mine_coords


def _is_background_cell(cell: np.ndarray) -> bool:
    """Determine if a cell is a background cell."""
    color_diff = np.abs(cell - BACKGROUND_COLOR)
    is_background_pixel = np.all(color_diff <= COLOR_TOLERANCE, axis=2)
    return np.mean(is_background_pixel) > 0.95
