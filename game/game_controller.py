# This class reads from the GameProcess's current state's stategy
# and takes action based on the current game state.
import win32gui
import pygetwindow as gw
import pydirectinput


class GameController:
    def __init__(self, window_handle):
        self.window_handle = window_handle

    def press_keys(self, keys, hold=False):
        """Press and hold or release a list of keys."""
        # Set focus to the game window
        # win32gui.SetForegroundWindow(self.window_handle)

        # # Get the game window
        # game_window = gw.getWindowsWithTitle("NPP")[0]
        # game_window.activate()

        # Press all keys
        for key in keys:
            pydirectinput.keyDown(key)

        # If hold is False, release all keys
        if not hold:
            for key in keys:
                pydirectinput.keyUp(key)

    def release_keys(self, keys):
        """Release a list of keys."""
        self.press_keys(keys, hold=False)

    def press_key_then_pause(self, key):
        """Presses a key immediately followed by a pause.
        A pause is the ESC key."""
        self.press_keys([key], hold=True)
        self.press_keys(["esc"])
        self.press_keys([key], hold=False)
