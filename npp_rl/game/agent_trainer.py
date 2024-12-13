from npp_rl.game.game_value_fetcher import GameValueFetcher
from npp_rl.game.game_controller import GameController
import threading

from npp_rl.agents.npp_agent_ppo import start_training
from npp_rl.game.training_session import training_session


class AgentTrainer:
    def __init__(self, game_value_fetcher: GameValueFetcher, game_controller: GameController):
        self.game_value_fetcher = game_value_fetcher
        self.game_controller = game_controller
        self.training_thread = None

    def start_training(self):
        """Start the training thread."""
        print('Starting training thread')
        self.game_controller.focus_window()

        # Start training session
        training_session.start_session()

        self.training_thread = threading.Thread(target=start_training, args=(
            self.game_value_fetcher, self.game_controller))
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Stop the training thread."""
        print('Stopping training')
        if self.training_thread:
            self.training_thread.join()
            self.training_thread = None

        # End training session
        training_session.end_session()
