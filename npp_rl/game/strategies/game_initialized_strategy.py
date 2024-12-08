from npp_rl.game.strategies.base_state_strategy import BaseStateStrategy
from npp_rl.game.game_config import game_config
import numpy as np
from npp_rl.game.game_controller import GameController


class GameInitializedStrategy(BaseStateStrategy):
    def take_action(self, game_value_fetcher, controller: GameController, frame: np.ndarray, frame_text: str) -> None:
        """Take action based on the current game state."""
        if not game_config.automate_init_screen:
            return

        print("Pressing space to continue to the main menu.")
        controller.press_keys(["space"])
