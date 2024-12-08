from game.game_window import get_game_window_handle
from game.game_value_fetcher import GameValueFetcher
from game.game_controller import GameController
import win32gui
import threading

from game.game_config import game_config

import pydirectinput
import time
from agents.npp_agent_ppo import start_training

pydirectinput.PAUSE = 0.01


class AgentTrainer:
    def __init__(self, game_value_fetcher: GameValueFetcher, game_controller: GameController):
        self.game_value_fetcher = game_value_fetcher
        self.game_controller = game_controller
        self.training_thread = None

    def start_training(self):
        """Start the training thread."""
        print('Starting training thread')
        self.game_controller.focus_window()
        self.training_thread = threading.Thread(target=start_training, args=(
            self.game_value_fetcher, self.game_controller))
        self.training_thread.start()

    def stop_training(self):
        """Stop the training thread."""
        print('Stopping training')
        if self.training_thread:
            self.training_thread.join()
            self.training_thread = None
