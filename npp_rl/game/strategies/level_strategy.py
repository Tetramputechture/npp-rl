from game.strategies.base_state_strategy import BaseStateStrategy
from game.game_config import game_config
from game.agent_trainer import AgentTrainer
# Our controls:
# 1: Move Left (A key)
# 2: Move Right (D key)
# 3: Jump (Space key press)
# 4: Jump + Left (Space + A)
# 5: Jump + Right (Space + D)


class LevelStrategy(BaseStateStrategy):
    def __init__(self):
        self.training_started = False
        self.trainer = None

    def take_action(self, game_value_fetcher, game_controller, frame, frame_text) -> None:
        """Take action based on the current game state."""
        if game_config.training and not self.training_started:
            # First, make sure all our needed address are defined
            if not game_config.all_addresses_defined():
                print(
                    'Not all addresses are defined. Please define all addresses in the GUI before training')
                return

            print('Starting training')
            self.trainer = AgentTrainer(game_value_fetcher, game_controller)
            self.trainer.start_training()
            self.training_started = True
        elif not game_config.training and self.training_started:
            print('Stopping training')
            self.trainer.stop_training()
            self.training_started = False

        # Otherwise, our trainer is handling the training loop
        if game_config.training:
            return
