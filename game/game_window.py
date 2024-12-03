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
    """Capture current window frame and convert to 640x480 RGB."""
    window_rect = get_game_window_rect()
    if not window_rect:
        return None

    # Capture window content
    screen = ImageGrab.grab(window_rect)

    # Resize to 640x480 and convert to grayscale
    frame = screen.resize((640, 480)).convert("RGB")

    # Preprocessing for text extraction
    frame = np.array(frame)

    return frame
