# Class to store runtime configuration settings,
# including:
# - Set training to ON or OFF
#   - Must be in the level_playing state

class GameConfig:
    def __init__(self):
        self.training = False
        self.automate_init_screen = False
        self.player_x_address = 0x00
        self.player_y_address = 0x00
        self.time_remaining_address = 0x00
        self.switch_activated_address = 0x00
        self.player_dead_address = 0x00
        self.exit_door_x_address = 0x00
        self.exit_door_y_address = 0x00
        self.switch_x_address = 0x00
        self.switch_y_address = 0x00

    def set_training(self, training: bool) -> None:
        """Set the training mode."""
        self.training = training
        return self.training

    def set_automate_init_screen(self, automate_init_screen: bool) -> None:
        """Set the automate_init_screen mode."""
        self.automate_init_screen = automate_init_screen
        return self.automate_init_screen

    def set_player_x_address(self, player_x_address: int) -> None:
        """Set the player_x_address."""
        self.player_x_address = player_x_address
        return self.player_x_address

    def set_player_y_address(self, player_y_address: int) -> None:
        """Set the player_y_address."""
        self.player_y_address = player_y_address
        return self.player_y_address

    def set_time_remaining_address(self, time_remaining_address: int) -> None:
        """Set the time_remaining_address."""
        self.time_remaining_address = time_remaining_address
        return self.time_remaining_address

    def set_switch_activated_address(self, switch_activated_address: int) -> None:
        """Set the switch_activated_address."""
        self.switch_activated_address = switch_activated_address
        return self.switch_activated_address

    def set_player_dead_address(self, player_dead_address: int) -> None:
        """Set the player_dead_address."""
        self.player_dead_address = player_dead_address
        return self.player_dead_address

    def set_exit_door_x_address(self, exit_door_x_address: int) -> None:
        """Set the exit_door_x_address."""
        self.exit_door_x_address = exit_door_x_address
        return self.exit_door_x_address

    def set_exit_door_y_address(self, exit_door_y_address: int) -> None:
        """Set the exit_door_y_address."""
        self.exit_door_y_address = exit_door_y_address
        return self.exit_door_y_address

    def set_switch_x_address(self, switch_x_address: int) -> None:
        """Set the switch_x_address."""
        self.switch_x_address = switch_x_address
        return self.switch_x_address

    def set_switch_y_address(self, switch_y_address: int) -> None:
        """Set the switch_y_address."""
        self.switch_y_address = switch_y_address
        return self.switch_y_address


game_config = GameConfig()
