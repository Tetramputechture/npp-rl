import numpy as np
from npp_rl.game.game_window import get_game_window_frame

# Constants for the game's visual layout
CELL_SIZE = 28  # Size of each grid cell in pixels
GRID_ROWS = 23  # Number of rows in the grid
GRID_COLS = 42  # Number of columns in the grid
BACKGROUND_COLOR = np.array([79, 86, 77])  # RGB values for background
COLOR_TOLERANCE = 5  # Tolerance for color matching to account for compression artifacts


def get_playable_space_coordinates() -> tuple:
    """Get the top-left and bottom-right coordinates of the playable space.

    This method works by:
    1. Dividing the frame into a grid of cells
    2. For each cell, checking if it's a background cell
    3. Finding the boundaries of non-background cells

    Args:
        frame: numpy array of shape (H, W, 3) containing the RGB game frame

    Returns:
        tuple: ((min_x, min_y), (max_x, max_y)) coordinates of playable space
    """
    # Assumes the game is not paused so the entire level is in view
    frame = get_game_window_frame()

    # Verify frame dimensions and type
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be an RGB image (H, W, 3)")

    height, width = frame.shape[:2]

    # Initialize arrays to track non-background cells
    active_rows = []
    active_cols = []

    # Analyze each grid cell
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Calculate cell boundaries
            y_start = row * CELL_SIZE
            y_end = y_start + CELL_SIZE
            x_start = col * CELL_SIZE
            x_end = x_start + CELL_SIZE

            # Ensure we don't exceed frame boundaries
            if y_end > height or x_end > width:
                continue

            # Get the cell's content
            cell = frame[y_start:y_end, x_start:x_end]

            # Check if cell is non-background
            if not is_background_cell(cell):
                active_rows.append(row)
                active_cols.append(col)

    # If no active cells found, return full frame coordinates
    if not active_rows or not active_cols:
        return ((0, 0), (width, height))

    # Calculate boundaries of playable space
    min_row, max_row = min(active_rows), max(active_rows) + 1
    min_col, max_col = min(active_cols), max(active_cols) + 1

    # Convert grid coordinates to pixel coordinates (x1, y1, x2, y2)
    min_x = min_col * CELL_SIZE
    min_y = min_row * CELL_SIZE
    max_x = max_col * CELL_SIZE
    max_y = max_row * CELL_SIZE

    # Pad the coordinates to ensure the entire playable space is captured
    padding = 2

    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width, max_x + padding)
    max_y = min(height, max_y + padding)

    return (min_x, min_y, max_x, max_y)


def is_background_cell(cell: np.ndarray) -> bool:
    """Determine if a cell is a background cell.

    A cell is considered background if most of its pixels match
    the background color within a tolerance threshold.

    Args:
        cell: numpy array of shape (CELL_SIZE, CELL_SIZE, 3) containing RGB values

    Returns:
        bool: True if cell is background, False otherwise
    """
    # Calculate color difference from background
    color_diff = np.abs(cell - BACKGROUND_COLOR)

    # Check if pixels are within tolerance of background color
    is_background_pixel = np.all(
        color_diff <= COLOR_TOLERANCE, axis=2)

    # Cell is background if most pixels match background color
    background_pixel_ratio = np.mean(is_background_pixel)
    # 95% threshold for background classification
    return background_pixel_ratio > 0.95
