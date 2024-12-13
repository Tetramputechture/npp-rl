

import json
import os


def valid_address(address: int) -> bool:
    """Check if the address is valid."""
    return address != 0x00


def increment_address(address: int, byte_offset: int) -> int:
    """Increment the address by a byte offset."""
    return address + byte_offset


CONFIG_PATH = 'game_config.json'


class GameConfig:
    """Class to store runtime configuration settings, including:

    - Set training to ON or OFF
      - Must be in the level_playing state

    Memory addresses for the game state:
    - Player position 
    - Time remaining
    - Switch activated
    - Player dead
    - Exit door position
    - Switch position
    - Begin retry text
    - In air

    - Current level data
    """

    def __init__(self):
        self.training = False
        self.automate_init_screen = False
        self.level_data = None  # Store the current level data
        self.player_x_address = 0x00
        self.player_y_address = 0x00
        self.time_remaining_address = 0x00
        self.switch_activated_address = 0x00
        self.player_dead_address = 0x00
        self.exit_door_x_address = 0x00
        self.exit_door_y_address = 0x00
        self.switch_x_address = 0x00
        self.switch_y_address = 0x00
        self.begin_retry_text_address = 0x00
        self.in_air_address = 0x00

    def _serialize_config(self) -> dict:
        """Serialize the configuration."""
        return {
            'training': self.training,
            'automate_init_screen': self.automate_init_screen,
            'level_data': self.level_data,
            'player_x_address': self.player_x_address,
            'player_y_address': self.player_y_address,
            'time_remaining_address': self.time_remaining_address,
            'switch_activated_address': self.switch_activated_address,
            'player_dead_address': self.player_dead_address,
            'exit_door_x_address': self.exit_door_x_address,
            'exit_door_y_address': self.exit_door_y_address,
            'switch_x_address': self.switch_x_address,
            'switch_y_address': self.switch_y_address,
            'begin_retry_text_address': self.begin_retry_text_address,
            'in_air_address': self.in_air_address,
        }

    def save_config(self) -> None:
        """Save the configuration."""
        with open(CONFIG_PATH, 'w') as f:
            json.dump(self._serialize_config(), f)

    def load_config(self) -> None:
        """Load the configuration."""
        # return if the file does not exist
        if not os.path.exists(CONFIG_PATH):
            return

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            self.training = config['training']
            self.automate_init_screen = config['automate_init_screen']
            self.level_data = config.get('level_data')
            self.player_x_address = config['player_x_address']
            self.player_y_address = config['player_y_address']
            self.time_remaining_address = config['time_remaining_address']
            self.switch_activated_address = config['switch_activated_address']
            self.player_dead_address = config['player_dead_address']
            self.exit_door_x_address = config['exit_door_x_address']
            self.exit_door_y_address = config['exit_door_y_address']
            self.switch_x_address = config['switch_x_address']
            self.switch_y_address = config['switch_y_address']
            self.begin_retry_text_address = config['begin_retry_text_address']
            self.in_air_address = config['in_air_address']

    def set_training(self, training: bool) -> None:
        """Set the training mode."""
        self.training = training
        return self.training

    def set_switch_x_address(self, switch_x_address: int) -> None:
        """Set the switch_x_address."""
        self.switch_x_address = switch_x_address
        self.switch_y_address = increment_address(switch_x_address, 4)

        # If we know the switch X address, we can infer the player X address
        # and the exit door X address
        # The player X address is the switch X address - 208
        # The exit door X address is the switch X address - 64
        self.player_x_address = switch_x_address - 208
        self.player_y_address = increment_address(self.player_x_address, 4)

        self.exit_door_x_address = switch_x_address - 64
        self.exit_door_y_address = increment_address(
            self.exit_door_x_address, 4)

        return self.switch_x_address

    def set_automate_init_screen(self, automate_init_screen: bool) -> None:
        """Set the automate_init_screen mode."""
        self.automate_init_screen = automate_init_screen
        return self.automate_init_screen

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

    def set_begin_retry_text_address(self, begin_retry_text_address: int) -> None:
        """Set the begin_retry_text_address."""
        self.begin_retry_text_address = begin_retry_text_address
        return self.begin_retry_text_address

    def set_in_air_address(self, in_air_address: int) -> None:
        """Set the in_air_address."""
        self.in_air_address = in_air_address
        return self.in_air_address

    def set_level_data(self, level_data: dict) -> dict:
        """Set the current level data.

        Args:
            level_data (dict): The level data to store

        Returns:
            dict: The stored level data
        """
        self.level_data = level_data
        return self.level_data

    def all_addresses_defined(self) -> bool:
        """Check if all addresses are defined."""
        return all([
            valid_address(self.player_x_address),
            valid_address(self.player_y_address),
            valid_address(self.time_remaining_address),
            valid_address(self.switch_activated_address),
            valid_address(self.player_dead_address),
            valid_address(self.exit_door_x_address),
            valid_address(self.exit_door_y_address),
            valid_address(self.switch_x_address),
            valid_address(self.switch_y_address),
            valid_address(self.begin_retry_text_address),
            valid_address(self.in_air_address),
        ])


game_config = GameConfig()
