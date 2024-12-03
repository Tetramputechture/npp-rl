from abc import ABC, abstractmethod
import numpy as np
from game.game_controller import GameController
from game.game_value_fetcher import GameValueFetcher


class BaseStateStrategy(ABC):
    """Base class for game state strategies."""
    @abstractmethod
    def take_action(self, game_value_fetcher: GameValueFetcher, controller: GameController, frame: np.ndarray, frame_text: str) -> None:
        """Take action based on the current game frame and frame text."""
