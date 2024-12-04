# Class to store runtime configuration settings,
# including:
# - Set training to ON or OFF
#   - Must be in the level_playing state

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

    def _serialize_config(self) -> dict:
        """Serialize the configuration."""
        return {
            'training': self.training,
            'automate_init_screen': self.automate_init_screen,
            'player_x_address': self.player_x_address,
            'player_y_address': self.player_y_address,
            'time_remaining_address': self.time_remaining_address,
            'switch_activated_address': self.switch_activated_address,
            'player_dead_address': self.player_dead_address,
            'exit_door_x_address': self.exit_door_x_address,
            'exit_door_y_address': self.exit_door_y_address,
            'switch_x_address': self.switch_x_address,
            'switch_y_address': self.switch_y_address
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
            self.player_x_address = config['player_x_address']
            self.player_y_address = config['player_y_address']
            self.time_remaining_address = config['time_remaining_address']
            self.switch_activated_address = config['switch_activated_address']
            self.player_dead_address = config['player_dead_address']
            self.exit_door_x_address = config['exit_door_x_address']
            self.exit_door_y_address = config['exit_door_y_address']
            self.switch_x_address = config['switch_x_address']
            self.switch_y_address = config['switch_y_address']

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
        self.player_y_address = increment_address(player_x_address, 4)
        return self.player_x_address

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
        self.exit_door_y_address = increment_address(exit_door_x_address, 4)
        return self.exit_door_x_address

    def set_switch_x_address(self, switch_x_address: int) -> None:
        """Set the switch_x_address."""
        self.switch_x_address = switch_x_address
        self.switch_y_address = increment_address(switch_x_address, 4)
        return self.switch_x_address

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
            valid_address(self.switch_y_address)
        ])


game_config = GameConfig()
