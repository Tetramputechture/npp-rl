# This class reads from the GameProcess's current state's stategy
# and takes action based on the current game state.
import win32gui
import pygetwindow as gw
import pydirectinput

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

    def _focus_window(self):
        """Focus the game window, if it's not already focused."""
        if self.window_focused:
            return

        win32gui.SetForegroundWindow(self.window_handle)

        game_window = gw.getWindowsWithTitle("NPP")[0]
        game_window.activate()
        self.window_focused = True

    def _press(self, key):
        self._focus_window()

        pydirectinput.press(key, _pause=False)

    def _key_down(self, key):
        self._focus_window()

        pydirectinput.keyDown(key, _pause=False)
        self.held_keys.append(key)

    def _key_up(self, key):
        self._focus_window()

        pydirectinput.keyUp(key, _pause=False)
        self.held_keys.remove(key)

    def release_all_keys(self):
        self._focus_window()

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

    def pause_key_down(self):
        self._key_down(MOVEMENT_KEYS['pause'])

    def pause_key_up(self):
        self._key_up(MOVEMENT_KEYS['pause'])

    def press_reset_key(self):
        self._press(MOVEMENT_KEYS['reset'])

    def press_space_key(self):
        self._press(MOVEMENT_KEYS['space'])
