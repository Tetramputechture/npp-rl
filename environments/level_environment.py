"""
This is an abstraction of a level in the game N++, implemented as a Gym-like environment
for reinforcement learning. The environment simulates a platformer level where an agent
must navigate through obstacles, collect items, and reach specific goals.

Environment Description:
----------------------
The game level is represented as a 640x480 RGB image where:
- The player character can move left/right and jump
- Controls are mapped to discrete actions (A: left, D: right, Space: jump)
- Jump height is proportional to button hold duration
- The level contains:
    * Player character (starting position)
    * Exit door (goal)
    * Switch (must be activated to open exit)
    * Optional gold pieces (time bonuses)
    * Possible hazards (traps, enemies)
    * Terrain (walls, platforms)
- The player must reach the exit door after activating the switch to complete the level
- The player is either alive or dead, with a health bar that depletes every second
  and each gold piece collected adds 1 second to the timer
- The entire game state and player information is observable from the screen
- Failure is indicated by player death or time running out
- Success is indicated by reaching the 'Level Complete' screen, which appears after the player reaches the exit door
    when the switch is activated

Action Space:
------------
Discrete(6):
    0: No action (NOOP)
    1: Move Left (A key)
    2: Move Right (D key)
    3: Jump (Space key press)
    4: Jump + Left (Space + A)
    5: Jump + Right (Space + D)

Observation Space:
----------------
Since the entire game can be observed from the screen, the observation space
is a dictionary containing the following elements:
- 'screen': RGB image of the game screen (480x640x3)
- 'player_x': Player's x-coordinate in the level (float, pixels)
- 'player_y': Player's y-coordinate in the level (float, pixels)
- 'time_remaining': Time remaining in the level (int, milliseconds)
- 'switch_activated': Whether the switch has been activated (bool)
- 'exit_door_x': X-coordinate of the exit door (float, pixels)
- 'exit_door_y': Y-coordinate of the exit door (float, pixels)
- 'switch_x': X-coordinate of the switch (float, pixels)
- 'switch_y': Y-coordinate of the switch (float, pixels)

Rewards:
-------
Each frame, the agent receives a reward based on the following conditions:
- If time remaining increases since the past frame, reward += 10 * seconds gained
- If time remaining decreases, reward -= 10 * seconds lost
- If switch is not activated, reward += the change in distance to the switch from the previous frame (encourage movement)
    - More positive reward for moving closer to the switch
        - Calculation: reward += (distance_to_switch_prev - distance_to_switch_curr)
- If switch is activated, reward += 10000
- If switch is activated, reward -= the change in distance to the exit from the previous frame (penalize distance)
    - The more the player moves away from the exit, the more negative the reward.
- If no movement is detected, reward -= 1
    - This encourages the agent to explore the level, but we make it low because sometimes the agent should stay still.
- If player reaches exit, reward += 20000 and episode ends
- If player dies, reward -= 10000 and episode ends

The episode ends when:
- The player reaches the exit after activating the switch (success)
- The player health reaches 0 (collision with hazards/enemies)
- The time remaining reaches 0

Our episode reward will be the sum of all rewards received during the episode.
We do not want to train on individual rewards, as they may not be indicative of the
agent's performance throughout the level. Instead, we will use the episode reward
to evaluate the agent's performance on the level. Take for example a level where the player
has to take a roundabout way to reach the exit. The agent may receive negative rewards
for moving away from the exit, but this is necessary to reach the switch and open the exit.
In this case, the episode reward will be positive, indicating that the agent performed well.

Starting State:
-------------
- Player spawns at predetermined position in level
- Timer starts at level-specific value (typically 90 seconds)
- Switch is inactive
- Exit door is closed
- Player is grounded

Episode Termination:
------------------
The episode terminates under the following conditions:

1. Success:
    - Player reaches the exit after activating the switch

2. Failure:
    - Player is dead
    - Time remaining reaches 0 (this results in a player death)

Additional Info:
--------------
1. The state of the player is the only part of the observation space that is directly controllable.
   The rest of the state is normally ran at a random FPS, so we have to pause the game to get the game state.
   We pause the game at a fixed timestep of 1/60th of a second (0.0167 seconds) to get the game state after an action
   is taken.
   The game (an episode) should start by pressing the space key to start the game. After this,
   the observation should be built by getting the frame data of the game and the game state
   in memory. Then, the game should be paused to wait for the agent to decide on what action to take.
   When a decision is made, the game should be unpaused, the action should be taken, and the game should be paused
   again after 0.0167 seconds to get the new observation space. This should be repeated until the episode ends.
   The episode ends when the player reaches the exit while the switch is activated, the player dies, or the time runs out (time remaining = 0).

2. Version:
    - v1.0

3. Render modes:
    - 'human': Displays game window

4. Max episode steps:
    - Since the entire game state is not in direct controler, we have a fixed timestep of 1/60th of a second (0.0167 seconds)
      that we will run the game for after taking an action. After this time, we will capture the frame, pause the game, and
      get the reward. This will be repeated until the episode ends. The max episode steps should be around 600 seconds (36000 frames).
    
5. Game Value Fetcher class: Class to fetch game values. These should be read when the game is paused.
    Methods:
        set_pm: Set the pymem object.
        read_player_x: Read the player's x position
        read_player_y: Read the player's y position
        read_time_remaining: Read the time remaining in seconds as a float
        read_switch_activated: Read the switch activated status
        read_player_dead: Read the player dead status
        read_exit_door_x: Read the exit door's x position
        read_exit_door_y: Read the exit door's y position.
        read_switch_x: Read the switch's x position.
        read_switch_y: Read the switch's y position.
        
6. Game Controller class: Class to control the game. This should be used to take actions based on the game state.
    Methods:
        focus_window: Focus the game window.
        
7. get_game_window_frame: Function to get the current RGB window frame as a numpy array.

8. We can assume the env is instantiated when the game is in a level_playing state.
   In this state, we can either be waiting to start the level, or waiting to retry the level after failing or succeeding.
   In these cases, no episode is running, and we should start a new episode by pressing the space key to start the level.
"""
import gym
from gym import spaces
import numpy as np
from game.game_controller import GameController
from game.game_window import get_game_window_frame
from game.game_value_fetcher import GameValueFetcher
import time
from typing import Tuple, Dict, Any


class NPlusPlus(gym.Env):
    """Custom Environment that follows gym interface for N++ game.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. The environment handles game state
    management, reward calculation, and action execution while maintaining proper
    timing and synchronization with the game.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, gvf: GameValueFetcher, gc: GameController):
        """Initialize the N++ environment.

        Args:
            gvf (GameValueFetcher): Instance of GameValueFetcher to read game state
        """
        super().__init__()

        # Store game value fetcher instance
        self.gvf = gvf

        # Store game controller instance
        self.gc = gc

        # Define timestep duration (in seconds)
        self.timestep = 1/60  # 60 FPS

        # Track previous state for reward calculation
        self.prev_distance_to_switch = None
        self.prev_distance_to_exit = None
        self.prev_time_remaining = None

        # Define action space
        # Discrete space with 6 possible actions:
        # 0: NOOP, 1: Left, 2: Right, 3: Jump, 4: Jump+Left, 5: Jump+Right
        self.action_space = spaces.Discrete(6)

        # Define observation space
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0,
                high=255,
                shape=(480, 640, 3),
                dtype=np.uint8
            ),
            'player_x': spaces.Box(
                low=63,
                high=1217,
                shape=(),
                dtype=np.float32
            ),
            'player_y': spaces.Box(
                low=171,
                high=791,
                shape=(),
                dtype=np.float32
            ),
            'time_remaining': spaces.Box(
                low=0,
                high=999,
                shape=(),
                dtype=np.float32
            ),
            'switch_activated': spaces.Discrete(2),
            'exit_door_x': spaces.Box(
                low=52,
                high=1258,
                shape=(),
                dtype=np.float32
            ),
            'exit_door_y': spaces.Box(
                low=158,
                high=802,
                shape=(),
                dtype=np.float32
            ),
            'switch_x': spaces.Box(
                low=52,
                high=1258,
                shape=(),
                dtype=np.float32
            ),
            'switch_y': spaces.Box(
                low=158,
                high=802,
                shape=(),
                dtype=np.float32
            )
        })

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from game state.

        Returns:
            Dict containing the current observation space values
        """
        return {
            'screen': get_game_window_frame(),
            'player_x': self.gvf.read_player_x(),
            'player_y': self.gvf.read_player_y(),
            'time_remaining': self.gvf.read_time_remaining(),
            'switch_activated': self.gvf.read_switch_activated(),
            'exit_door_x': self.gvf.read_exit_door_x(),
            'exit_door_y': self.gvf.read_exit_door_y(),
            'switch_x': self.gvf.read_switch_x(),
            'switch_y': self.gvf.read_switch_y()
        }

    def _calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            x1, y1: Coordinates of first point
            x2, y2: Coordinates of second point

        Returns:
            float: Euclidean distance between the points
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Calculate reward based on current and previous observations.

        Args:
            obs: Current observation dictionary
            prev_obs: Previous observation dictionary

        Returns:
            float: Calculated reward value
        """
        reward = 0.0

        # Time-based rewards
        time_diff = obs['time_remaining'] - prev_obs['time_remaining']
        if time_diff > 0:  # Collected gold (gained time)
            reward += 10 * time_diff
        else:  # Normal time decrease
            reward += time_diff

        # Calculate distances
        curr_distance_to_switch = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['switch_x'], obs['switch_y']
        )
        curr_distance_to_exit = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['exit_door_x'], obs['exit_door_y']
        )

        # Switch-related rewards
        if not obs['switch_activated']:
            # Reward for moving closer to switch
            distance_diff = self.prev_distance_to_switch - curr_distance_to_switch
            reward += distance_diff
        else:
            # One-time reward for activating switch
            if not prev_obs['switch_activated']:
                reward += 10000

            # Penalty for moving away from exit
            distance_diff = curr_distance_to_exit - self.prev_distance_to_exit
            reward -= distance_diff

        # Update previous distances
        self.prev_distance_to_switch = curr_distance_to_switch
        self.prev_distance_to_exit = curr_distance_to_exit

        # Small penalty for no movement (encourage exploration)
        if obs['player_x'] == prev_obs['player_x'] and obs['player_y'] == prev_obs['player_y']:
            reward -= 1

        return reward

    def _execute_action(self, action: int):
        """Execute the specified action using the game controller.

        Args:
            action (int): Action to execute (0-5)

        The action mapping is:
        0: NOOP - No action
        1: Left - Press 'A' key
        2: Right - Press 'D' key
        3: Jump - Press Space key
        4: Jump + Left - Press Space + 'A' keys
        5: Jump + Right - Press Space + 'D' keys
        """
        # First release any previously held keys
        self.gc.release_all_keys()

        # Execute the new action
        if action == 0:  # NOOP
            pass
        elif action == 1:  # Left
            self.gc.move_left_key_down()
        elif action == 2:  # Right
            self.gc.move_right_key_down()
        elif action == 3:  # Jump
            self.gc.jump_key_down()
        elif action == 4:  # Jump + Left
            self.gc.jump_key_down()
            self.gc.move_left_key_down()
        elif action == 5:  # Jump + Right
            self.gc.jump_key_down()
            self.gc.move_right_key_down()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action and return new state of the environment.

        Args:
            action: Integer representing the action to take (0-5)

        Returns:
            tuple containing:
            - observation: Dict of current game state
            - reward: Float reward value
            - terminated: Boolean indicating if episode ended
            - truncated: Boolean indicating if episode was truncated
            - info: Dict containing auxiliary information
        """
        # Store previous observation for reward calculation
        prev_obs = self._get_observation()

        # Execute action
        self._execute_action(action)

        # Wait for timestep duration
        time.sleep(self.timestep)

        # Release all keys after the timestep
        self.gc.release_all_keys()

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(observation, prev_obs)

        # Check termination conditions
        terminated = False
        if observation['time_remaining'] <= 0 or self.gvf.read_player_dead():
            terminated = True
            reward -= 10000  # Death penalty
        elif (self.gvf.read_switch_activated() and
              abs(observation['player_x'] - observation['exit_door_x']) < 5 and
              abs(observation['player_y'] - observation['exit_door_y']) < 5):
            terminated = True
            reward += 20000  # Success reward

        # We don't use truncation in this environment
        truncated = False

        # Additional info for debugging and monitoring
        info = {
            'time_remaining': observation['time_remaining'],
            'switch_activated': observation['switch_activated'],
            'distance_to_switch': self.prev_distance_to_switch,
            'distance_to_exit': self.prev_distance_to_exit
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed
            options: Optional dict with additional options

        Returns:
            tuple containing:
            - observation: Dict of initial game state
            - info: Dict of auxiliary information
        """
        # Optional: Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Release any held keys
        self.gc.release_all_keys()

        # Press reset key to restart level
        self.gc.press_reset_key()

        # Press space to start level
        self.gc.press_space_key()
        time.sleep(0.1)  # Small delay to ensure game registers input

        # Get initial observation
        observation = self._get_observation()

        # Initialize previous distances
        self.prev_distance_to_switch = self._calculate_distance(
            observation['player_x'], observation['player_y'],
            observation['switch_x'], observation['switch_y']
        )
        self.prev_distance_to_exit = self._calculate_distance(
            observation['player_x'], observation['player_y'],
            observation['exit_door_x'], observation['exit_door_y']
        )
        self.prev_time_remaining = observation['time_remaining']

        return observation, {}

    def render(self, mode='human'):
        """Render the environment.

        Since the game window is already visible, we don't need additional rendering.
        """
        pass

    def close(self):
        """Clean up environment resources."""
        # Ensure all keys are released when environment is closed
        self.gc.release_all_keys()
