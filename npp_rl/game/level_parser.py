import numpy as np
import cv2
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from npp_rl.game.game_window import get_game_window_frame, LEVEL_FRAME

# Constants for the game's visual layout
CELL_SIZE = 28  # Size of each grid cell in pixels
GRID_ROWS = 23  # Number of rows in the grid
GRID_COLS = 42  # Number of columns in the grid
BACKGROUND_COLOR = np.array([79, 86, 77])  # RGB values for background
COLOR_TOLERANCE = 5  # Tolerance for color matching

# Load mine template
MINE_TEMPLATE = cv2.imread(os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), 'npp_rl/entity_images/mine.png'))
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
    width: int  # Width of the playable level in pixels
    height: int  # Height of the playable level in pixels
    hazard_map: np.ndarray  # Hazard map showing mine locations with gaussian falloff

    def create_hazard_map(self, image_size: int = 84) -> np.ndarray:
        """Create a hazard map showing mine locations with gaussian falloff.

        Args:
            image_size: Size of the output hazard map (default: 84x84 to match observation space)

        Returns:
            np.ndarray: Hazard map with gaussian falloff around mines
        """
        hazard_map = np.zeros((image_size, image_size), dtype=np.float32)

        if not self.mine_coordinates:
            return hazard_map

        # Scale coordinates to our observation size
        min_x, min_y, max_x, max_y = self.playable_space
        level_width = max_x - min_x
        level_height = max_y - min_y

        for mine_x, mine_y in self.mine_coordinates:
            # Convert mine coordinates to normalized space
            norm_x = (mine_x - min_x) / level_width
            norm_y = (mine_y - min_y) / level_height

            # Convert to observation space coordinates
            obs_x = int(norm_x * image_size)
            obs_y = int(norm_y * image_size)

            # Create gaussian falloff around mine (sigma=2 pixels)
            y, x = np.ogrid[-obs_y:image_size-obs_y, -obs_x:image_size-obs_x]
            mask = np.exp(-(x*x + y*y) / (2 * 2**2))
            hazard_map = np.maximum(hazard_map, mask)

        return hazard_map

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get the level boundaries."""
        return self.playable_space

    def is_mine_at(self, x: int, y: int) -> bool:
        """Check if there is a mine at the given coordinates."""
        return (x, y) in self.mine_coordinates

    def convert_game_to_level_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Convert game coordinates to level coordinates by adjusting for playable space offset.

        Args:
            x: X coordinate in game space (0,0 at top-left)
            y: Y coordinate in game space (0,0 at top-left)

        Returns:
            Tuple[int, int]: (x, y) coordinates adjusted for level space
        """
        min_x, min_y, _, _ = self.playable_space
        # Offset by both min x / min y and also the level frame offset
        return (x - min_x - LEVEL_FRAME[0], y - min_y - LEVEL_FRAME[1])

    def get_mines_in_radius(self, x: int, y: int, radius: float) -> List[Tuple[int, int]]:
        """Get all mines within a certain radius of the given coordinates."""
        return [mine for mine in self.mine_coordinates if np.linalg.norm(np.array(mine) - np.array((x, y))) <= radius]

    def get_nearest_mine(self, x: int, y: int, max_distance: float = float('inf')) -> Optional[Tuple[int, int]]:
        """Find the coordinates of the mine nearest to the given position within max_distance.

        Args:
            x: X coordinate of the position (in level coordinates)
            y: Y coordinate of the position (in level coordinates)
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

    def get_player_bounding_box(self, raw_player_x: float, raw_player_y: float) -> Tuple[int, int, int, int]:
        """Get the bounding box of the player at the given coordinates.

        Args:
            raw_player_x: Player x coordinate in game space (relative to game window)
            raw_player_y: Player y coordinate in game space (relative to game window)

        Returns:
            Tuple[int, int, int, int]: (x1, y1, x2, y2) coordinates of player bounding box in level space
        """
        # First convert raw coordinates to level space
        level_x, level_y = self.convert_game_to_level_coordinates(
            raw_player_x, raw_player_y)

        # Apply hitbox offsets
        level_x -= 2  # Shift left by 2px
        level_y += 15  # Shift down by 15px

        # Add playable space min x and y
        min_x, min_y, _, _ = self.playable_space
        level_x += min_x
        level_y += min_y

        # Return the player bounding box coordinates (11x23 pixels)

        return (level_x, level_y, level_x + 11, level_y + 23)

    def get_mine_bounding_box(self, mine_x: int, mine_y: int) -> Tuple[int, int, int, int]:
        """Get the bounding box of a mine at the given coordinates.

        Args:
            mine_x: X coordinate of the mine center
            mine_y: Y coordinate of the mine center

        Returns:
            Tuple[int, int, int, int]: (x1, y1, x2, y2) coordinates of mine bounding box
        """
        # Mine is 12x12 pixels centered on the coordinate
        return (mine_x - 6, mine_y - 6, mine_x + 6, mine_y + 6)

    def get_vector_between_rectangles(self, rect1: Tuple[int, int, int, int],
                                      rect2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Get the vector representing shortest distance between two rectangles.

        Args:
            rect1: First rectangle as (x1, y1, x2, y2)
            rect2: Second rectangle as (x1, y1, x2, y2)

        Returns:
            Tuple[int, int, int, int]: Vector as (start_x, start_y, end_x, end_y)
        """
        # Unpack rectangle coordinates
        r1_x1, r1_y1, r1_x2, r1_y2 = rect1
        r2_x1, r2_y1, r2_x2, r2_y2 = rect2

        # Find closest points on each rectangle
        if r1_x2 < r2_x1:  # rect1 is left of rect2
            x1 = r1_x2
            x2 = r2_x1
        elif r2_x2 < r1_x1:  # rect1 is right of rect2
            x1 = r1_x1
            x2 = r2_x2
        else:  # rectangles overlap in x
            x1 = max(r1_x1, r2_x1)
            x2 = x1

        if r1_y2 < r2_y1:  # rect1 is above rect2
            y1 = r1_y2
            y2 = r2_y1
        elif r2_y2 < r1_y1:  # rect1 is below rect2
            y1 = r1_y1
            y2 = r2_y2
        else:  # rectangles overlap in y
            y1 = max(r1_y1, r2_y1)
            y2 = y1

        return (x1, y1, x2, y2)

    def get_vector_to_nearest_mine(self, player_x: int, player_y: int) -> Optional[Tuple[int, int, int, int]]:
        """Get the vector from the player to the nearest mine within 50 pixels.

        Args:
            player_x: X coordinate of player center (in game coordinates)
            player_y: Y coordinate of player center (in game coordinates)

        Returns:
            Optional[Tuple[int, int, int, int]]: Vector as (start_x, start_y, end_x, end_y) 
                                                or None if no mines exist within range
        """
        if not self.mine_coordinates:
            return None

        # Convert player coordinates to level space
        player_box = self.get_player_bounding_box(player_x, player_y)

        # Quick filter mines within rough 50px radius using level coordinates
        nearby_mines = [
            (mx, my) for mx, my in self.mine_coordinates
            if abs(mx - player_box[0]) <= 50 and abs(my - player_box[1]) <= 50
        ]

        if not nearby_mines:
            return None

        shortest_vector = None
        shortest_dist = float('inf')

        # Check filtered mines
        for mine_x, mine_y in nearby_mines:
            mine_box = self.get_mine_bounding_box(mine_x, mine_y)
            vector = self.get_vector_between_rectangles(player_box, mine_box)

            # Calculate distance between vector endpoints
            dist = ((vector[2] - vector[0])**2 +
                    (vector[3] - vector[1])**2)**0.5

            if dist <= 50 and (shortest_vector is None or dist < shortest_dist):
                shortest_dist = dist
                shortest_vector = vector

        return shortest_vector


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

    # Create level object
    level = ParsedLevel(
        playable_space=playable_space,
        mine_coordinates=mine_coords,
        width=width,
        height=height,
        hazard_map=None  # Initialize as None
    )

    # Create hazard map
    level.hazard_map = level.create_hazard_map()

    return level


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
    min_x = max(0, min_col * CELL_SIZE)
    min_y = max(0, min_row * CELL_SIZE)
    max_x = min(width, max_col * CELL_SIZE)
    max_y = min(height, max_row * CELL_SIZE)

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
