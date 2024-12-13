from npp_rl.game.game_config import game_config
from pymem import Pymem


def safe_read_float(pm: Pymem, address: int) -> float:
    """Read a float from memory, returning 0.0 if there is an error."""
    if address == 0:
        return 0.0

    try:
        return pm.read_float(address)
    except Exception as e:
        print(f'Error reading float at address {address}: {e}')
        return 0.0


def safe_read_double(pm: Pymem, address: int) -> float:
    """Read a double from memory, returning 0.0 if there is an error."""
    if address == 0:
        return 0.0

    try:
        return pm.read_double(address)
    except Exception as e:
        print(f'Error reading double at address {address}: {e}')
        return 0.0


def safe_read_byte(pm: Pymem, address: int) -> bool:
    """Read a byte from memory, returning False if there is an error."""
    if address == 0:
        return False

    try:
        return pm.read_bool(address)
    except Exception as e:
        print(f'Error reading byte at address {address}: {e}')
        return False


def safe_read_string(pm: Pymem, address: int) -> str:
    """Read a string from memory, returning an empty string if there is an error."""
    if address == 0:
        return ""

    try:
        return pm.read_string(address)
    except Exception as e:
        print(f'Error reading string at address {address}: {e}')
        return ""


class GameValueFetcher:
    """Class to fetch game values from memory."""

    def __init__(self):
        self.pm = None

    def set_pm(self, pm: Pymem) -> None:
        """Set the pymem object."""
        self.pm = pm
        return self.pm

    def read_player_x(self) -> float:
        """Read the player's x position from memory."""
        return safe_read_float(self.pm, game_config.player_x_address)

    def read_player_y(self) -> float:
        """Read the player's y position from memory"""
        return safe_read_float(self.pm, game_config.player_y_address)

    def read_time_remaining(self) -> float:
        """Read the time remaining from memory."""
        return safe_read_double(self.pm, game_config.time_remaining_address)

    def read_switch_activated(self) -> bool:
        """Read the switch activated status from memory."""
        return safe_read_byte(self.pm, game_config.switch_activated_address)

    def read_player_dead(self) -> bool:
        """Read the player dead status from memory."""
        return safe_read_byte(self.pm, game_config.player_dead_address)

    def read_exit_door_x(self) -> float:
        """Read the exit door's x position from memory."""
        return safe_read_float(self.pm, game_config.exit_door_x_address)

    def read_exit_door_y(self) -> float:
        """Read the exit door's y position from memory."""
        return safe_read_float(self.pm, game_config.exit_door_y_address)

    def read_switch_x(self) -> float:
        """Read the switch's x position from memory."""
        return safe_read_float(self.pm, game_config.switch_x_address)

    def read_switch_y(self) -> float:
        """Read the switch's y position from memory."""
        return safe_read_float(self.pm, game_config.switch_y_address)

    def read_begin_retry_text(self) -> str:
        """Read the begin retry label from memory."""
        return safe_read_string(self.pm, game_config.begin_retry_text_address)

    def read_in_air(self) -> bool:
        """Read the in air status from memory."""
        return safe_read_byte(self.pm, game_config.in_air_address)
