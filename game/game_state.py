import numpy as np
from game.strategies.level_strategy import LevelStrategy
from game.frame_text import FrameText


class GameState:
    def __init__(self):
        self.state = "game_loading"
        self.strategy = None
        self.state_transitions = {
            "game_loading": {
                "game_initialized": self._is_game_initialized
            },
            "game_initialized": {
                "main_menu": self._is_main_menu
            },
            "main_menu": {
                "level_select": self._is_level_select,
                "level_start": self._is_level_start
            },
            "level_select": {
                "level_start": self._is_level_start
            },
            "level_start": {
                "level_playing": self._is_level_playing
            },
            "level_playing": {
                "level_fail": self._is_level_fail,
                "level_complete": self._is_level_complete
            },
            "level_fail": {
                "intermediate_select": self._is_intermediate_select
            },
            "level_complete": {
                "intermediate_select": self._is_intermediate_select
            },
            "intermediate_select": {
                "level_select": self._is_level_select
            }
        }

    def transition(self, frame: np.ndarray, frame_text: FrameText) -> str:
        """Update the game state based on the current frame and frame text."""
        for new_state, condition in self.state_transitions[self.state].items():
            if condition(frame_text):
                print(f"Transitioning from {self.state} to {new_state}")
                self.state = new_state
                break

        if self.state == "level_playing":
            self.strategy = LevelStrategy(frame, frame_text)

        if self.strategy:
            self.strategy.take_action()

        return self.state

    def reset(self):
        """Reset the game state to the initial state."""
        self.state = "game_loading"
        self.strategy = None

    def _is_game_initialized(self, frame_text: FrameText) -> bool:
        """Check if the game has been initialized."""
        # Check if lowercase text includes "press any key"
        return "press any key" in frame_text.all_text.lower()

    def _is_main_menu(self, frame_text: FrameText) -> bool:
        """Check if the game is in the main menu."""
        # look for "play create browse" and not "solo"
        # first, lowercase the text and remove newlines
        frame_text = frame_text.all_text.lower().replace("\n", " ")
        return "play create browse" in frame_text and "solo" not in frame_text

    def _is_level_select(self, frame_text: FrameText) -> bool:
        """Check if the game is in the level select menu."""
        return False

    def _is_level_start(self, frame_text: FrameText) -> bool:
        """Check if the game is in the level start screen."""
        return "press space to begin" in frame_text.all_text.lower()

    def _is_level_playing(self, frame_text: FrameText) -> bool:
        """Check if the game is in the level playing screen."""
        return not self._is_level_fail(frame_text) and not self._is_level_complete(frame_text)

    def _is_level_fail(self, frame_text: FrameText) -> bool:
        """Check if the game is in the level fail screen."""
        return "press space to retry" in frame_text.all_text.lower()

    def _is_level_complete(self, frame_text: FrameText) -> bool:
        """Check if the game is in the level complete screen."""
        # Implement logic to check if the game is in the level complete screen
        return "press space to continue" in frame_text.all_text.lower()

    def _is_intermediate_select(self, frame_text: FrameText) -> bool:
        """Check if the game is in the intermediate select screen."""
        # Implement logic to check if the game is in the intermediate select screen
        return False
