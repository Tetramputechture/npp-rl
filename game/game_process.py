# In this game, the use can press and hold any combination of the A, D, or space keys
# for any amount of time. The play space is observed as a stack of images, where each
# image is a frame of the game.

# We want to define a Game class that is responsible for:
# - Starting the game executable
# - Capturing the game window and returning the current frame
# - Sending key presses to the game
# - Stopping the game executable


from PIL import ImageGrab, Image
import win32gui
import subprocess
import time
import pyautogui
import numpy as np
import os
from typing import Optional, List
from game.game_state import GameState
from game.frame_text import FrameText
from game.config import Config


GAME_EXECUTABLE_PATH = r"D:\Games\Steam\steamapps\common\N++\N++.exe"
GAME_WINDOW_TITLE = "NPP"


class GameProcess:
    """
    Class to manage the game process and interaction.
    """

    def __init__(self):
        self.process = None
        self.window_handle = None
        self.window_rect = None
        self.state = GameState()
        self.config = Config()
        self.current_frame = None
        self.last_frame = None
        self.current_frame_text = FrameText()
        self.last_frame_text = FrameText()

    def start(self) -> bool:
        """Start the game process and get window handle."""
        print(f'Starting game process: {GAME_EXECUTABLE_PATH}')
        try:
            self.process = subprocess.Popen([GAME_EXECUTABLE_PATH])
            print('Game process started')
            # Wait for window to appear
            time.sleep(5)
            self.window_handle = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
            if not self.window_handle:
                print('Game window not found')
                return False

            # Get window dimensions
            self.window_rect = win32gui.GetWindowRect(self.window_handle)
            print(f'Game window found: {self.window_rect}')
            return True
        except Exception:
            print('Error starting game process')
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture current window frame and convert to 640x480 grayscale."""
        if not self.window_handle:
            print('Window handle not found')
            return None

        # Capture window content
        screen = ImageGrab.grab(self.window_rect)
        # Resize to 640x480 and convert to grayscale
        frame = screen.resize((640, 480)).convert("L")

        # Preprocessing for text extraction
        frame = np.array(frame)

        return frame

    def save_current_frame(self, path: str) -> None:
        """Save the current frame to the given path."""
        if self.current_frame is None:
            return

        # If the path ends in a /, our filename is the index of the file
        # in the directory, otherwise, the path is the filename
        # First, prepend the path with our working directory
        file_path = f"{os.getcwd()}/data/{path}"
        if path.endswith("/"):
            file_path = f"{file_path}{len(os.listdir(file_path))}.png"

        # Save the frame to the given path
        Image.fromarray(self.current_frame).save(file_path)

    def press_keys(self, keys: List[str], hold: bool = False) -> None:
        # Activate window before sending input
        win32gui.SetForegroundWindow(self.window_handle)

        # Press all keys
        for key in keys:
            pyautogui.keyDown(key)

        # If hold is False, release all keys
        if not hold:
            for key in keys:
                pyautogui.keyUp(key)

    def _extract_frame_text(self) -> dict:
        """Divides the frame up into three sections vertically and extracts text from each."""

        # If our current frame is None, return None
        if self.current_frame is None:
            return None

        # If our current frame is equal to our last frame, return the last frame text
        if np.array_equal(self.current_frame, self.last_frame):
            print('Static frame detected')
            return self.last_frame_text

    def loop(self) -> Optional[str]:
        """Main event loop."""
        # We want to run our event loop around 12fps,
        # so we will sleep for 1/12 seconds between each iteration.
        # This is roughly 80ms.
        try:
            if not self.check_process_running():
                # If the current game state is loading,
                # we are still loading the game. Otherwise,
                # the game has exited, and we should stop the loop
                if self.state.state == "game_loading":
                    return "game_loading"

                self.stop()
                return None

            # Process the frame and frame text
            self.last_frame = self.current_frame
            self.current_frame = self.get_frame()
            self.last_frame_text = self.current_frame_text
            self.current_frame_text.set_from_frame(self.current_frame)

            # Invoke our state transition with our frame and frame text
            new_state = self.state.transition(
                self.current_frame, self.current_frame_text)

            # If transition returns None, the game is over
            if new_state is None:
                return None
            return new_state
        except Exception as e:
            print(f'Error in game loop: {e}')

    def check_process_running(self) -> bool:
        """Check if the game process is running."""
        # Use win32gui to check if the game window is still open
        if not self.window_handle:
            return False
        if not win32gui.IsWindow(self.window_handle):
            return False
        return True

    def stop(self) -> None:
        """Stop the game process and cleanup."""
        if self.process:
            self.process = None
            self.window_handle = None
            self.window_rect = None
            self.current_frame = None
            self.current_frame_text = None
