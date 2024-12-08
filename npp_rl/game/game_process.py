# In this game, the use can press and hold any combination of the A, D, or space keys
# for any amount of time. The play space is observed as a stack of images, where each
# image is a frame of the game.

# We want to define a Game class that is responsible for:
# - Starting the game executable
# - Capturing the game window and returning the current frame
# - Sending key presses to the game
# - Stopping the game executable


from PIL import Image
import win32gui
import subprocess
import time
import numpy as np
import os
from typing import Optional
from game.state_manager import StateManager
from game.game_controller import GameController
from game.game_value_fetcher import GameValueFetcher
from game.game_window import get_game_window_handle, get_game_window_rect, get_game_window_frame

import pymem

GAME_EXECUTABLE_PATH = r"D:\Games\Steam\steamapps\common\N++\N++.exe"
GAME_EXECUTABLE_NAME = "N++.exe"


class GameProcess:
    """
    Class to manage the game process and interaction.
    """

    def __init__(self):
        self.process = None
        self.window_handle = None
        self.window_rect = None
        self.state_manager = StateManager()
        self.current_frame = None
        self.controller = None
        self.game_value_fetcher = GameValueFetcher()
        self.started = False
        self.pm = None

    def start(self) -> bool:
        """Start the game process and get window handle."""
        print(f'Starting game process: {GAME_EXECUTABLE_PATH}')
        try:
            # Do not start if the game is already running
            if not get_game_window_handle():
                self.process = subprocess.Popen(GAME_EXECUTABLE_PATH)
                # Wait for window to appear
                time.sleep(5)
            self.pm = pymem.Pymem("N++.exe")
            self.game_value_fetcher.set_pm(self.pm)
            self.window_handle = get_game_window_handle()
            self.controller = GameController(self.window_handle)
            if not self.window_handle:
                print('Game window not found')
                return False

            # Get window dimensions
            self.window_rect = get_game_window_rect()
            print(f'Game window found: {self.window_rect}')
            self.started = True
            return True
        except Exception as e:
            print(f'Error starting game process: {e}')
            return False

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
                if self.state_manager.state == "game_loading":
                    return "game_loading"

                self.stop()
                return None

            self.current_frame = get_game_window_frame()

            # Invoke our state transition with our frame
            new_state = self.state_manager.transition(self.current_frame)

            # If transition returns None, the game is over
            if new_state is None:
                return None

            # Take action based on the new state
            self.state_manager.take_action(
                self.game_value_fetcher, self.controller)

            return new_state
        except Exception as e:
            print(f'Error in game loop: {e}')

    def check_process_running(self) -> bool:
        """Check if the game process is running."""
        # Use win32gui to check if the game window is still open
        if not get_game_window_handle():
            return False
        return True

    def stop(self) -> None:
        """Stop the game process and cleanup."""
        if self.process:
            self.process.kill()
            self.process = None
            self.window_handle = None
            self.window_rect = None
            self.current_frame = None
            self.started = False
