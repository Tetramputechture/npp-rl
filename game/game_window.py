import win32gui
import numpy as np
from PIL import ImageGrab
from typing import Optional
GAME_WINDOW_TITLE = "NPP"


def get_game_window_handle() -> int:
    """Get the game window handle."""
    return win32gui.FindWindow(None, GAME_WINDOW_TITLE)


def get_game_window_rect() -> Optional[tuple]:
    """Get the game window rect."""
    window_handle = get_game_window_handle()
    if not window_handle:
        return None
    return win32gui.GetWindowRect(window_handle)


def get_game_window_frame() -> Optional[np.ndarray]:
    """Capture current window frame and convert to 640x480 RGB.
    Assumes a 1280x720 input."""
    window_rect = get_game_window_rect()
    if not window_rect:
        return None

    # Capture window content
    screen = ImageGrab.grab(window_rect)

    # Crop to only the level playing view
    frame = screen.crop((48, 180, 1230, 830))

    frame = np.array(frame)
    return frame


def get_center_frame(frame) -> Optional[np.ndarray]:
    """Returns the center of the game frame"""
    return frame[333:480, 350:930]
