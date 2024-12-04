# This class reads from the GameProcess's current state's stategy
# and takes action based on the current game state.
import win32gui
import pygetwindow as gw
import pydirectinput
import time

MOVEMENT_KEYS = {
    'left': 'a',
    'right': 'd',
    'jump': 'space',
    'space': 'space',
    'pause': 'esc',
    'reset': 'k',
}


class GameController:
    def __init__(self, window_handle):
        self.window_handle = window_handle
        self.window_focused = False
        self.held_keys = []

    def focus_window(self):
        """Focus the game window, if it's not already focused."""
        if self.window_focused:
            return

        print("Focusing window.")

        win32gui.SetForegroundWindow(self.window_handle)

        game_window = gw.getWindowsWithTitle("NPP")[0]
        game_window.activate()
        self.window_focused = True
        # Sometimes, taking focus can take a moment
        time.sleep(1)

    def set_window_focused(self, focused):
        self.window_focused = focused

    def _press(self, key, pause=False):
        self.focus_window()

        pydirectinput.press([key], _pause=pause)

    def _key_down(self, key):
        self.focus_window()

        pydirectinput.keyDown(key, _pause=False)
        self.held_keys.append(key)

    def _key_up(self, key):
        self.focus_window()

        pydirectinput.keyUp(key, _pause=False)
        self.held_keys.remove(key)

    def release_all_keys(self):
        self.focus_window()

        for key in self.held_keys:
            pydirectinput.keyUp(key, _pause=False)
        self.held_keys = []

    def move_left_key_down(self):
        self._key_down(MOVEMENT_KEYS['left'])

    def move_left_key_up(self):
        self._key_up(MOVEMENT_KEYS['left'])

    def move_right_key_down(self):
        self._key_down(MOVEMENT_KEYS['right'])

    def move_right_key_up(self):
        self._key_up(MOVEMENT_KEYS['right'])

    def jump_key_down(self):
        self._key_down(MOVEMENT_KEYS['jump'])

    def jump_key_up(self):
        self._key_up(MOVEMENT_KEYS['jump'])

    def press_reset_key(self):
        self._press(MOVEMENT_KEYS['reset'], pause=True)

    def press_space_key(self):
        self._press(MOVEMENT_KEYS['space'])

    def press_pause_key(self):
        self._press(MOVEMENT_KEYS['pause'])

    def reset_level(self):
        print("Resetting level.")

        # Press reset key to restart level
        self.press_reset_key()

        # Press space twice to start level
        self.press_space_key()
        time.sleep(0.01)  # Small delay to ensure game registers input
        self.press_space_key()
