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
- 'screen': 4-frame stack of grayscale images (84x84 pixels) representing the game window
- 'player_x': Player's x-coordinate in the level (float, pixels)
- 'player_y': Player's y-coordinate in the level (float, pixels)
- 'time_remaining': Time remaining in the level (int, milliseconds)
- 'switch_activated': Whether the switch has been activated (bool)
- 'exit_door_x': X-coordinate of the exit door (float, pixels)
- 'exit_door_y': Y-coordinate of the exit door (float, pixels)
- 'switch_x': X-coordinate of the switch (float, pixels)
- 'switch_y': Y-coordinate of the switch (float, pixels)

When processing, the observation space is preprocessed as follows:
- The screen is resized to 84x84 pixels and normalized to [0, 1]
- The player's x and y coordinates are normalized to the level dimensions
- Time remaining is normalized to [0, 1]
- Switch activated status is converted to a float (0 or 1)
- Exit door and switch coordinates are normalized to the level dimensions

When building the input tensor for the agent, the observation space is stacked
with the last 4 frames to provide temporal information. We have 84x84 resolution
gray-scale images, so the final observation space is (84, 84, 4 + 8) where 4 is the
frame stack and 8 is the numerical features.


Rewards:
-------
Each frame, the agent receives a reward based on the following conditions:
- If time remaining increases since the past frame, reward += 10 * seconds gained
- If time remaining decreases, reward -= 10 * seconds lost
- If switch is not activated, reward += the change in distance to the switch from the previous frame (encourage movement)
    - More positive reward for moving closer to the switch
        - Calculation: reward += (distance_to_switch_prev - distance_to_switch_curr)
    - We also subtract the distance to the switch overall to encourage the agent to move towards the switch
- If switch is activated, reward += 10000
- If switch is activated, reward += the change in distance to the exit from the previous frame
    - Reward for moving closer to the exit after switch is activated
    - Calculation: reward += (distance_to_exit_prev - distance_to_exit_curr)
    - We also subtract the distance to the exit overall to encourage the agent to move towards the exit
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

9. We keep a frame stack of 4 frames to provide the agent with temporal information.

10. When taking an action that includes the same key or keys as the previous action, we don't want to release the keys
    before pressing them again. This is because the agent may want to continue holding the key(s) for the next action.
"""
import gym
from gymnasium.spaces import discrete, box
import numpy as np
from game.game_controller import GameController
from game.game_window import get_game_window_frame, get_center_frame
from game.frame_text import extract_text
from game.game_value_fetcher import GameValueFetcher
import time
from typing import Tuple, Dict, Any
from collections import deque
import cv2
import os


class NPlusPlus(gym.Env):
    """Custom Gym envjr onment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. The environment handles game state
    management, reward calculation, and action execution while maintaining proper
    timing and synchronization with the game.
    """
    metadata = {'render.modes': ['human']}

    # Constants for game time speed increase
    GAME_SPEED_FACTOR = 2.0  # Speed factor for game time
    # Default game speed in frames per second
    GAME_DEFAULT_SPEED_FRAMES_PER_SECOND = 60.0
    # Game speed after speed increase
    GAME_SPEED_FRAMES_PER_SECOND = GAME_DEFAULT_SPEED_FRAMES_PER_SECOND * GAME_SPEED_FACTOR

    # Constants for reward scaling
    BASE_TIME_PENALTY = -0.01  # Small constant penalty per timestep
    GOLD_COLLECTION_REWARD = 1.0  # Reward for collecting gold (time increase)
    SWITCH_ACTIVATION_REWARD = 10.0  # One-time reward for activating switch
    TERMINAL_REWARD = 20.0  # Reward for completing level
    DEATH_PENALTY = -10.0  # Penalty for dying
    TIMEOUT_PENALTY = -10.0  # Penalty for time running out

    # Constants for distance-based rewards
    DISTANCE_SCALE = 0.1  # Scale factor for distance-based rewards
    APPROACH_REWARD_SCALE = 0.5  # Scale factor for approaching objectives
    RETREAT_PENALTY_SCALE = 0.2  # Scale factor for moving away from objectives

    # Movement assessment constants
    MIN_MOVEMENT_THRESHOLD = 0.1  # Minimum movement to avoid penalty
    MOVEMENT_PENALTY = -0.01  # Penalty for being too static
    MAX_MOVEMENT_REWARD = 0.05  # Cap on movement reward

    # We take our observations at the game speed * 5
    # this way our observations are more accurate
    TIMESTEP = 1/(GAME_SPEED_FRAMES_PER_SECOND * 5)

    # Movement truncation constants
    MOVEMENT_THRESHOLD = 1.0
    MOVEMENT_CHECK_DURATION = 20.0
    MOVEMENT_CHECK_FRAMES = int(
        MOVEMENT_CHECK_DURATION / (1/GAME_SPEED_FRAMES_PER_SECOND))

    def __init__(self, gvf: GameValueFetcher, gc: GameController, frame_stack: int = 4):
        """Initialize the N++ environment.

        Args:
            gvf (GameValueFetcher): Instance of GameValueFetcher to read game state
        """
        super().__init__()

        self.gvf = gvf
        self.gc = gc
        self.frame_stack = frame_stack

        # Initialize frame stacking
        self.frames = deque(maxlen=frame_stack)

        # Initialize movement and velocity tracking
        self.position_history = deque(maxlen=self.MOVEMENT_CHECK_FRAMES)
        self.velocity_history = deque(maxlen=10)

        # Track previous state for reward calculation
        self.prev_distance_to_switch = None
        self.prev_distance_to_exit = None
        self.prev_time_remaining = None
        self.prev_position = None

        # Track previous action
        self.prev_action = None

        # Initialize spaces
        self.action_space = discrete.Discrete(6)
        self.observation_space = box.Box(
            low=0,
            high=1,
            shape=(84, 84, frame_stack + 8),
            dtype=np.float32
        )

        # Initialize position log folder
        self.position_log_folder_name = f'training_logs/position_log_{time.strftime("%m-%d-%Y-%H-%M-%S")}'
        os.makedirs(self.position_log_folder_name)

        # Initialize position log file string
        self.position_log_file_string = 'PlayerX,PlayerY\n'

        # Initialize action log folder
        self.action_log_folder_name = f'training_logs/action_log_{time.strftime("%m-%d-%Y-%H-%M-%S")}'
        os.makedirs(self.action_log_folder_name)

        # Initialize action log file string
        self.action_log_file_string = 'Action\n'

        # Initialize episode counter
        self.episode_counter = 0

    def _preprocess_frame(self, frame):
        """Preprocess raw frame for CNN input.

        Returns:
            numpy.ndarray: Preprocessed frame of shape (84, 84, 1)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1] and ensure correct shape (84, 84, 1)
        frame = frame.astype(np.float32) / 255.0

        # Add channel dimension if not present
        if len(frame.shape) == 2:
            frame = frame[..., np.newaxis]

        return frame

    def _get_numerical_features(self, obs):
        """Extract and normalize numerical features.

        Returns:
            numpy.ndarray: Feature array of shape (84, 84, 8)
        """
        # Create normalized feature vector
        features = np.array([
            (obs['player_x'] - 63) / (1217 - 63),
            (obs['player_y'] - 171) / (791 - 171),
            obs['time_remaining'] / 999.0,
            float(obs['switch_activated']),
            (obs['exit_door_x'] - 52) / (1258 - 52),
            (obs['exit_door_y'] - 158) / (802 - 158),
            (obs['switch_x'] - 52) / (1258 - 52),
            (obs['switch_y'] - 158) / (802 - 158)
        ], dtype=np.float32)

        # Reshape to (1, 1, 8) then broadcast to (84, 84, 8)
        features = features.reshape((1, 1, 8))
        features = np.broadcast_to(features, (84, 84, 8))

        return features

    def _get_stacked_observation(self, obs):
        """Combine frame stack with numerical features properly.

        Returns:
            numpy.ndarray: Combined observation of shape (84, 84, frame_stack + 8)
        """
        # Preprocess current frame
        frame = self._preprocess_frame(obs['screen'])  # Shape: (84, 84, 1)

        # Update our frame stack
        self.frames.append(frame)

        # If stack isn't full, duplicate the first frame
        while len(self.frames) < self.frame_stack:
            self.frames.append(frame)

        # Concatenate frames along channel dimension
        # Shape: (84, 84, frame_stack)
        stacked_frames = np.concatenate(list(self.frames), axis=2)

        # Get and process numerical features
        features = self._get_numerical_features(obs)  # Shape: (84, 84, 8)

        # Combine frames and features
        final_observation = np.concatenate([stacked_frames, features], axis=2)

        # Ensure observation is in correct range [0, 1]
        final_observation = np.clip(final_observation, 0, 1)

        return final_observation.astype(np.float32)

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

    def _shaped_distance_reward(self, current_distance: float, previous_distance: float,
                                is_primary_objective: bool = True) -> float:
        """Calculate shaped reward based on distance to objective.

        Args:
            current_distance: Current distance to objective
            previous_distance: Previous distance to objective
            is_primary_objective: Whether this is the current primary objective

        Returns:
            float: Shaped reward value
        """
        # Calculate basic progress reward
        distance_progress = previous_distance - current_distance

        # Scale factors based on objective priority
        approach_scale = self.APPROACH_REWARD_SCALE * \
            (1.5 if is_primary_objective else 1.0)
        retreat_scale = self.RETREAT_PENALTY_SCALE * \
            (1.5 if is_primary_objective else 1.0)

        # Calculate progressive reward
        if distance_progress > 0:  # Moving closer
            progress_reward = distance_progress * approach_scale
        else:  # Moving away
            progress_reward = distance_progress * retreat_scale

        # Add inverse distance component for continuous guidance
        inverse_distance = 1.0 / (1.0 + current_distance)

        return progress_reward + (inverse_distance * self.DISTANCE_SCALE)

    def _movement_quality_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Calculate reward based on movement quality.

        Evaluates movement based on:
        - Magnitude of movement
        - Consistency of movement
        - Progress toward objectives
        """
        current_pos = np.array([obs['player_x'], obs['player_y']])

        if self.prev_position is None:
            self.prev_position = current_pos
            return 0.0

        # Calculate velocity
        velocity = current_pos - self.prev_position
        velocity_magnitude = np.linalg.norm(velocity)

        # Update velocity history
        self.velocity_history.append(velocity)

        # Calculate movement reward
        if velocity_magnitude < self.MIN_MOVEMENT_THRESHOLD:
            movement_reward = self.MOVEMENT_PENALTY
        else:
            # Reward smooth, purposeful movement
            movement_reward = min(velocity_magnitude * 0.1,
                                  self.MAX_MOVEMENT_REWARD)

            # Add bonus for consistent movement direction if we have enough history
            if len(self.velocity_history) >= 2:
                prev_velocity = self.velocity_history[-2]
                if np.linalg.norm(prev_velocity) != 0:
                    direction_consistency = np.dot(velocity, prev_velocity) / (
                        np.linalg.norm(velocity) * np.linalg.norm(prev_velocity))
                    movement_reward *= (1.0 + direction_consistency * 0.5)

        self.prev_position = current_pos
        return movement_reward

    def _calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Calculate reward based on current and previous observations.

        Implements an improved reward structure with:
        1. Shaped distance-based rewards
        2. Normalized reward scales
        3. Movement quality assessment
        4. Continuous time penalties
        5. Balanced terminal rewards
        """
        reward = 0.0

        # Calculate current distances
        curr_distance_to_switch = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['switch_x'], obs['switch_y']
        )
        curr_distance_to_exit = self._calculate_distance(
            obs['player_x'], obs['player_y'],
            obs['exit_door_x'], obs['exit_door_y']
        )

        # Base time penalty to encourage efficiency
        reward += self.BASE_TIME_PENALTY

        # Time and gold collection rewards
        time_diff = obs['time_remaining'] - prev_obs['time_remaining']
        if time_diff > 0:  # Collected gold
            reward += self.GOLD_COLLECTION_REWARD * time_diff

        # Objective-based rewards
        if not obs['switch_activated']:
            # Primary objective: Switch activation
            reward += self._shaped_distance_reward(
                curr_distance_to_switch,
                self.prev_distance_to_switch,
                is_primary_objective=True
            )
        else:
            # Switch activation reward (one-time)
            if not prev_obs['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD

            # Secondary objective: Reach exit
            reward += self._shaped_distance_reward(
                curr_distance_to_exit,
                self.prev_distance_to_exit,
                is_primary_objective=True
            )

        # Movement quality reward
        reward += self._movement_quality_reward(obs, prev_obs)

        # Terminal state rewards
        if obs['time_remaining'] <= 0:
            reward += self.TIMEOUT_PENALTY

        if self.gvf.read_player_dead():
            reward += self.DEATH_PENALTY

        # Success reward
        if 'retry level' in self.gvf.read_begin_retry_text().lower():
            reward += self.TERMINAL_REWARD

        # Update previous distances
        self.prev_distance_to_switch = curr_distance_to_switch
        self.prev_distance_to_exit = curr_distance_to_exit

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

    def _action_to_string(self, action: int) -> str:
        """Convert action index to human-readable string."""
        action_names = {
            0: 'NOOP',
            1: 'Left',
            2: 'Right',
            3: 'Jump',
            4: 'Jump + Left',
            5: 'Jump + Right'
        }
        return action_names.get(action, 'Unknown')

    def _check_movement_truncation(self, curr_x: float, curr_y: float) -> bool:
        """Check if the episode should be truncated due to lack of movement.
        Lack of movement is defined as the player not having an absolute
        distance traveled (sum of all movements, positive or negative) greater
        than a threshold over a fixed time window.

        Args:
            curr_x: Current x position of the player
            curr_y: Current y position of the player

        Returns:
            bool: True if episode should be truncated, False otherwise
        """
        # Add current position to history
        self.position_history.append((curr_x, curr_y))

        # We need enough history to make a decision
        if len(self.position_history) < self.MOVEMENT_CHECK_FRAMES:
            return False

        # Calculate total distance traveled over the window
        # We want to check the sum of all movements, so we calculate the
        # distance between each pair of consecutive positions
        total_distance = 0.0

        for i in range(1, len(self.position_history)):
            prev_x, prev_y = self.position_history[i - 1]
            curr_x, curr_y = self.position_history[i]
            total_distance += self._calculate_distance(
                prev_x, prev_y, curr_x, curr_y)

        print(
            f'Total distance traveled in last {self.MOVEMENT_CHECK_DURATION} seconds: {total_distance}')
        return total_distance < self.MOVEMENT_THRESHOLD

    def _store_prev_action(self, action: int):
        """Store the previous action taken."""
        self.prev_action = action

    def _check_termination(self, observation: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if the episode should be terminated.

        Args:
            observation: Current observation dictionary

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
        """
        terminated = self.gvf.read_player_dead(
        ) or 'retry level' in self.gvf.read_begin_retry_text().lower()

        if terminated:
            print("Episode terminated")

        truncated = self._check_movement_truncation(
            observation['player_x'],
            observation['player_y']
        )

        if truncated:
            print("Episode truncated due to lack of movement")
            # Manually hit  the reset key to reset the level
            self.gc.press_reset_key()

        return terminated, truncated

    def step(self, action):
        """Execute action and return new state."""
        # Get previous observation
        prev_obs = self._get_observation()

        # Track position
        self.position_log_file_string += f'{prev_obs["player_x"]},{prev_obs["player_y"]}\n'

        # Track action
        self.action_log_file_string += f'{self._action_to_string(action)}\n'

        # Our reset method ensures the game is paused, and the end of our step method
        # also pauses the game. So, we press the pause key to unpause the game.
        # self.gc.press_pause_key()

        # Now we are in the 'level playing' state, we can execute the action
        self._execute_action(action)

        # Store this action for the next step
        self._store_prev_action(action)

        # Wait our timestep for the action to take effect and the game state to update
        time.sleep(self.TIMESTEP)

        # Get the new observation
        observation = self._get_observation()

        # Check if episode should be ended
        terminated, truncated = self._check_termination(observation)

        # If the episode has not been terminated, pause the game
        # if not terminated or truncated:
        #     # self.gc.press_pause_key()
        # else:
        #     print("Episode done. Not pausing. The environment reset should handle this.")

        # Process the observation
        processed_obs = self._get_stacked_observation(observation)

        # Calculate reward
        reward = self._calculate_reward(observation, prev_obs)

        info = {
            'raw_reward': reward,
            'time_remaining': observation['time_remaining'],
            'switch_activated': observation['switch_activated'],
            'player_dead': self.gvf.read_player_dead(),
        }

        # Log new position to file training_logs.txt

        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state.
        Assumes the game is in a level_playing state.

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

        # Reset frame stack
        self.frames.clear()

        # Reset position and velocity history
        self.position_history.clear()
        self.velocity_history.clear()

        # Reset previous state values
        self.prev_distance_to_switch = None
        self.prev_distance_to_exit = None
        self.prev_time_remaining = None
        self.prev_position = None

        # Reset previous action
        self.prev_action = None

        # Write the position log file to our position log folder with the episode number
        with open(f'{self.position_log_folder_name}/position_log_{self.episode_counter}.csv', 'w') as f:
            f.write(self.position_log_file_string)

        # Write the action log file to our action log folder with the episode number
        with open(f'{self.action_log_folder_name}/action_log_{self.episode_counter}.csv', 'w') as f:
            f.write(self.action_log_file_string)

        # Reset position log file string
        self.position_log_file_string = 'PlayerX,PlayerY\n'

        # Reset action log file string
        self.action_log_file_string = 'Action\n'

        # Increment episode counter
        self.episode_counter += 1

        # Release all keys
        self.gc.release_all_keys()

        print("Resetting environment...")

        # center_frame = get_center_frame(get_game_window_frame())
        # center_text = extract_text(center_frame)

        # print(center_text)

        # retry_max = 5
        # retries = 0

        # Case when level is completed -- read_begin_retry_text contains 'retry level'
        # In this case, we want to press the success reset key
        if 'retry level' in self.gvf.read_begin_retry_text().lower():
            print("Level completed. Pressing success reset key combo...")
            self.gc.press_success_reset_key_combo()

        # Press space twice just in case we are at the 'level retry' screen already
        self.gc.press_space_key()
        self.gc.press_space_key()

        while 'ret' not in self.gvf.read_begin_retry_text().lower() and not \
                (self.gvf.read_player_dead() or 'begi' in self.gvf.read_begin_retry_text().lower()):
            print("Killing player...")
            self.gc.press_reset_key()
            time.sleep(0.2)

        while 'begi' not in self.gvf.read_begin_retry_text().lower():
            print("Pressing space to go to 'press space to begin' screen...")
            # # if this loops, 'pause' and 'retry' are shown at the same time
            self.gc.press_space_key()
            time.sleep(0.1)
            # center_frame = get_center_frame(get_game_window_frame())
            # center_text = extract_text(center_frame)

        # # Get initial player position in level
        # initial_obs = self._get_observation()

        # print(
        #     f"Initial player position: ({initial_obs['player_x']}, {initial_obs['player_y']})")

        # # We are sure the player is not dead, and we are at the 'press space to begin' screen
        # # To assure this, we want to get the text from the center of the screen
        # # and check if it is 'begin'
        # center_frame = get_center_frame(get_game_window_frame())
        # center_text = extract_text(center_frame)

        # observation = self._get_observation()

        # # While either begin is in the center text, OR the player is not moving, press space
        # position_diff_x = initial_obs['player_x'] - observation['player_x']
        # position_diff_y = initial_obs['player_y'] - observation['player_y']

        print("Pressing space to go to the 'level playing' state...")
        self.gc.press_space_key()
        # center_frame = get_center_frame(get_game_window_frame())
        # center_text = extract_text(center_frame)

        # # Get the new observation
        # observation = self._get_observation()

        # # Calculate the difference in player position
        # position_diff_x = initial_obs['player_x'] - observation['player_x']
        # position_diff_y = initial_obs['player_y'] - observation['player_y']

        # Get initial observation
        observation = self._get_observation()

        # # Press pause key to pause the game. We are sure we are in the 'level playing' state,
        # # since we verified the player has moved.
        # print(f'Center text before pause: {center_text}')
        # # Print player position before pause
        # print(
        #     f"Player position before pause: ({observation['player_x']}, {observation['player_y']})")
        # # pause_key_success = self.gc.press_pause_key()
        # # print("Pause key press success:", pause_key_success)

        print('Game started')

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

        # Process observation for A2C
        processed_obs = self._get_stacked_observation(observation)

        return processed_obs, {}

    def render(self):
        """Render the environment.

        Since the game window is already visible, we don't need additional rendering.
        We can just return the current game window frame.
        """
        return get_game_window_frame()

    def close(self):
        """Clean up environment resources."""
        # Ensure all keys are released when environment is closed
        self.gc.release_all_keys()
