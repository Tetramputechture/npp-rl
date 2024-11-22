from game.strategies.base_state_strategy import BaseStateStrategy
from game.config import config


class LevelStrategy(BaseStateStrategy):
    def take_action(self) -> None:
        """Take action based on the current game state."""
        if config.training:
            self._train()

    def _train(self) -> None:
        """Train the model."""
        print("Training the model...")
        # Placeholder for training logic
