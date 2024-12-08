from dataclasses import dataclass
import numpy as np
from typing import Callable, Dict, List, Optional, Set
from game.strategies.level_strategy import LevelStrategy
from game.strategies.game_initialized_strategy import GameInitializedStrategy
from game.frame_text import all_frame_text, main_menu_text
from game.game_controller import GameController
from game.game_value_fetcher import GameValueFetcher


@dataclass
class StateConfig:
    """Configuration for a game state including its frame processor and possible transitions."""
    frame_processor: Callable[[np.ndarray], str]
    strategy: Optional[object]
    transitions: Dict[str, 'TransitionRule']


@dataclass
class TransitionRule:
    """Rule for transitioning between states."""
    target_state: str
    condition: Callable[[str], bool]


class StateManager:
    """Manages game state transitions and actions based on frame analysis."""

    def __init__(self):
        self.state: str = "game_loading"
        self.current_frame: Optional[np.ndarray] = None
        self.current_frame_text: Optional[str] = None

        # Define state configurations
        self.state_configs = self._initialize_state_configs()

    def _initialize_state_configs(self) -> Dict[str, StateConfig]:
        """Initialize all state configurations with their frame processors and transition rules."""
        return {
            "game_loading": StateConfig(
                frame_processor=None,
                strategy=None,
                transitions={
                    "game_initialized": TransitionRule(
                        "game_initialized",
                        lambda text: "press any key" in text
                    )
                }
            ),
            "game_initialized": StateConfig(
                frame_processor=None,
                strategy=GameInitializedStrategy(),
                transitions={
                    "main_menu": TransitionRule(
                        "main_menu",
                        lambda text: "play create" in text and "solo" not in text
                    )
                }
            ),
            "main_menu": StateConfig(
                frame_processor=None,
                strategy=None,
                transitions={
                    "level_start": TransitionRule(
                        "level_start",
                        lambda text: "press space to begin" in text
                    )
                }
            ),
            "level_playing": StateConfig(
                frame_processor=None,
                strategy=LevelStrategy(),
                transitions={
                    "level_fail": TransitionRule(
                        "level_fail",
                        lambda text: "press space to retry" in text
                    ),
                    "level_complete": TransitionRule(
                        "level_complete",
                        lambda text: "press space to continue" in text
                    )
                }
            ),
            "level_start": StateConfig(
                frame_processor=None,
                strategy=None,
                transitions={
                    "level_playing": TransitionRule(
                        "level_playing",
                        lambda text: not any(phrase in text for phrase in [
                            "press space to retry",
                            "press space to continue",
                            "press space to begin"
                        ])
                    )
                }
            ),
            "level_fail": StateConfig(
                frame_processor=None,
                strategy=None,
                transitions={
                    "level_playing": TransitionRule(
                        "level_playing",
                        lambda text: not any(phrase in text for phrase in [
                            "press space to retry",
                            "press space to continue",
                            "press space to begin"
                        ])
                    )
                }
            ),
            "level_complete": StateConfig(
                frame_processor=all_frame_text,
                strategy=None,
                transitions={
                    "level_playing": TransitionRule(
                        "level_playing",
                        lambda text: not any(phrase in text for phrase in [
                            "press space to retry",
                            "press space to continue",
                            "press space to begin"
                        ])
                    )
                }
            ),
            "intermediate_select": StateConfig(
                frame_processor=all_frame_text,
                strategy=None,
                transitions={
                    "level_start": TransitionRule(
                        "level_start",
                        lambda text: "press space to begin" in text
                    )
                }
            )
        }

    def _process_frame(self, frame: np.ndarray) -> str:
        """Process the current frame using the current state's frame processor."""
        config = self.state_configs[self.state]
        text = config.frame_processor(frame) if config.frame_processor else ""
        return text.lower().replace("\n", " ")

    def transition(self, frame: np.ndarray) -> str:
        """Update the game state based on the current frame.

        Returns:
            str: The new state after processing transitions.da
        """
        self.current_frame = frame
        self.current_frame_text = self._process_frame(frame)

        # Check for valid transitions from current state
        config = self.state_configs[self.state]
        for transition in config.transitions.values():
            if transition.condition(self.current_frame_text):
                print(
                    f"Transitioning from {self.state} to {transition.target_state}")
                self.state = transition.target_state
                break

        return self.state

    def take_action(self, game_value_fetcher: GameValueFetcher, controller: GameController) -> None:
        """Execute the current state's strategy if one exists."""
        config = self.state_configs[self.state]
        if config.strategy:
            config.strategy.take_action(
                game_value_fetcher,
                controller,
                self.current_frame,
                self.current_frame_text
            )

    def force_set_state(self, state: str) -> None:
        """Set the state to the given state."""
        self.state = state
        self.current_frame = None
        self.current_frame_text = None

    def reset(self) -> None:
        """Reset the state manager to initial state."""
        self.state = "game_loading"
        self.current_frame = None
        self.current_frame_text = None
