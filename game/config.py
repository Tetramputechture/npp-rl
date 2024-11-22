# Class to store runtime configuration settings,
# including:
# - Set training to ON or OFF
#   - Must be in the level_playing state

class Config:
    def __init__(self):
        self.training = False

    def set_training(self, training: bool) -> None:
        """Set the training mode."""
        self.training = training
        return self.training


config = Config()
