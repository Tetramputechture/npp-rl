from abc import ABC, abstractmethod
import numpy as np


class BaseStateStrategy(ABC):
    """Base class for game state strategies."""

    def __init__(self, frame: np.ndarray, frame_text: str):
        self.frame = frame
        self.frame_text = frame_text

    @abstractmethod
    def take_action(self) -> None:
        """Take action based on the current game frame and frame text."""
