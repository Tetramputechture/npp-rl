from game.game_window import get_game_window_handle
from game.game_value_fetcher import GameValueFetcher
from game.game_controller import GameController
import win32gui
import threading

from game.game_config import game_config

import pydirectinput
import time
pydirectinput.PAUSE = 0.01


class Trainer:
    def __init__(self, game_value_fetcher: GameValueFetcher, game_controller: GameController):
        self.game_value_fetcher = game_value_fetcher
        self.game_controller = game_controller
        self.window_focused = False
        self.training_thread = None

    def focus_window(self):
        """Focus the game window."""
        window_handle = get_game_window_handle()
        win32gui.SetForegroundWindow(window_handle)
        self.window_focused = True

    def start_training(self):
        """Start the training thread."""
        print('Starting training thread')
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.start()

    def training_loop(self):
        """The training loop."""
        print('Starting training loop in training thread')
        while True:
            if not self.window_focused:
                self.focus_window()
            if not game_config.training:
                break
            # Do training stuff hereaa
            pydirectinput.keyDown('a', _pause=False)
            pydirectinput.press('esc', _pause=False)
            # simulate training here with a delay
            print('Training...')
            time.sleep(0.5)
            pydirectinput.press('esc', _pause=False)
            pass

    def stop_training(self):
        """Stop the training thread."""
        print('Stopping training')
        if self.training_thread:
            self.training_thread.join()
            self.training_thread = None
        self.window_focused = False
