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
        """Focus the game window."""
        print('Focusing game window')
        win32gui.SetForegroundWindow(self.window_handle)
        # game_window = gw.getWindowsWithTitle("NPP")[0]
        # game_window.activate()
        self.window_focused = True
        time.sleep(1)

    def maybe_focus_window(self):
        """Focus the game window, if it's not already focused."""
        if self.window_focused:
            return

        self.focus_window()

    def _press(self, keys, pause=False):
        self.maybe_focus_window()

        return pydirectinput.press(keys, _pause=pause)

    def space_with_hold(self):
        self.maybe_focus_window()

        pydirectinput.keyDown('space', _pause=False)
        time.sleep(0.01)
        pydirectinput.keyUp('space', _pause=False)

    def _key_down(self, key):
        self.maybe_focus_window()

        pydirectinput.keyDown(key, _pause=False)
        self.held_keys.append(key) if key not in self.held_keys else None

    def _key_up(self, key):
        self.maybe_focus_window()

        pydirectinput.keyUp(key, _pause=False)
        self.held_keys.remove(key) if key in self.held_keys else None

    def release_all_keys(self):
        self.maybe_focus_window()

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
        self._press([MOVEMENT_KEYS['reset']])

    def press_space_key(self, pause=True):
        return self._press([MOVEMENT_KEYS['space']], pause=pause)

    def space_key_up(self):
        self._key_up(MOVEMENT_KEYS['space'])

    def press_pause_key(self):
        return self._press([MOVEMENT_KEYS['pause']], pause=True)

    def press_success_reset_key_combo(self):
        # We press ctrl+k to reset the level after succeeding
        self._key_down('ctrl')
        self._key_down('k')
        time.sleep(0.1)
        self._key_up('k')
        self._key_up('ctrl')

    def reset_level(self):
        time.sleep(0.5)

        # Press space to start level
        self.press_space_key()
        self.press_space_key()
