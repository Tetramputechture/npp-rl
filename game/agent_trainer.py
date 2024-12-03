from game.game_window import get_game_window_handle
from game.game_value_fetcher import GameValueFetcher
from game.game_controller import GameController
import win32gui
import threading

from game.game_config import game_config

import pydirectinput
import time
pydirectinput.PAUSE = 0.01


class AgentTrainer:
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
            # pydirectinput.keyDown('a', _pause=False)
            # pydirectinput.press('esc', _pause=False)
            # # simulate training here with a delay
            # print('Training...')
            # time.sleep(0.5)
            # pydirectinput.press('esc', _pause=False)
            # pass
            #
            # Actual training logic:
            # Pre-training:
            # 1. Focus the game window
            # 2. Reset the game state to initial
            # 3. Press the start game button (space)
            #
            # Training:
            # 1. Get the current game state
            # 2. Pause the game
            # 3. Get the action to take from the agent
            #   - The action will be a number from 1-6, where:
            #     1: Move Left (A key)
            #     2: Move Right (D key)
            #     3: Jump (Space key press)
            #     4: Jump + Left (Space + A)
            #     5: Jump + Right (Space + D)
            #     6: Do nothing
            # 4. Unpause the game
            # 5. Execute the action
            #   - If the current action includes a key from the previous action, keep the key pressed
            #   - Otherwise, keyUp on all non-common keys and keyDown on the new key
            #   - Wait 1/60th of a second (0.0167 seconds)
            # 6. Capture the frame after the action is executed
            # 7. Get the reward from the game state
            # 8. Update the agent with the reward
            # 9. Repeat from step 1

    def stop_training(self):
        """Stop the training thread."""
        print('Stopping training')
        if self.training_thread:
            self.training_thread.join()
            self.training_thread = None
        self.window_focused = False
