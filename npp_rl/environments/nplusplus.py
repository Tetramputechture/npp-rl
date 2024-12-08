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
- 'in_air': Whether the player is in the air (bool)

When processing, the observation space is preprocessed as follows:
- The screen is resized to 84x84 pixels and normalized to [0, 1]
- The player's x and y coordinates are normalized to the level dimensions
- Time remaining is normalized to [0, 1]
- Switch activated status is converted to a float (0 or 1)
- Exit door and switch coordinates are normalized to the level dimensions
- In-air status is converted to a float (0 or 1)

When building the input tensor for the agent, the observation space is stacked
with the last 4 frames to provide temporal information. We have 84x84 resolution
gray-scale images, so the final observation space is (84, 84, 4 + 9) where 4 is the
frame stack and 9 is the numerical features.


Rewards:
-------
See _calculate_reward method for detailed reward structure.

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
import gymnasium
from gymnasium.spaces import discrete, box
import numpy as np
from npp_rl.game.game_controller import GameController
from npp_rl.game.game_window import get_game_window_frame
from npp_rl.game.game_value_fetcher import GameValueFetcher
from npp_rl.environments.reward_calculator import RewardCalculator
from npp_rl.util.util import calculate_distance
from npp_rl.environments.movement_evaluator import MovementEvaluator
from npp_rl.environments.constants import TIMESTEP, GAME_SPEED_FRAMES_PER_SECOND
import time
from typing import Tuple, Dict, Any
from collections import deque
import cv2
import os

NUM_NUMERICAL_FEATURES = 11


class NPlusPlus(gymnasium.Env):
    """Custom Gym environment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. The environment handles game state
    management, reward calculation, and action execution while maintaining proper
    timing and synchronization with the game. The environment also tracks
    overall success in the level and progress towards sub-goals in the level,
    such as activating the switch or reaching the exit door, to help the agent
    learn efficient sub-paths through the level.

    Success metrics include:
    1. Overall level completion
    2. Progress towards switch
    3. Progress towards exit
    4. Gold collection efficiency
    5. Time management
    6. Movement efficiency

    """
    metadata = {'render.modes': ['human']}

    # Movement truncation constants
    MOVEMENT_THRESHOLD = 1.0
    MOVEMENT_CHECK_DURATION = 20.0
    MOVEMENT_CHECK_FRAMES = int(
        MOVEMENT_CHECK_DURATION / (1/GAME_SPEED_FRAMES_PER_SECOND))

    # Progress thresholds for sub-goals
    MOVEMENT_SUCCESS_THRESHOLD = 0.85
    PROGRESS_SUCCESS_THRESHOLD = 0.7
    TIME_GAIN_THRESHOLD = 0.5

    # The max absolute velocity of the player
    MAX_VELOCITY = 20000.0

    def __init__(self, gvf: GameValueFetcher, gc: GameController, movement_evaluator: MovementEvaluator, frame_stack: int = 4):
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

        # Initialize spaces
        self.action_space = discrete.Discrete(6)
        self.observation_space = box.Box(
            low=0,
            high=1,
            shape=(84, 84, frame_stack + NUM_NUMERICAL_FEATURES),
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

        # Initialize movement evaluator to determine movement success rates
        # (not directly related to reward calculation)
        self.movement_evaluator = movement_evaluator

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(movement_evaluator)

        # Success tracking
        # Basic success flags
        self.current_episode_success = False
        self.switch_activated = False
        self.died = False
        self.time_expired = False
        self.completed_level = False

        # Progressive success tracking
        self.initial_time = None
        self.max_time_gained = 0
        self.previous_time = None
        self.best_switch_distance = float('inf')
        self.best_exit_distance = float('inf')

        # Movement tracking for success
        # 1 second of positions at (GAME_SPEED_FRAMES_PER_SECOND) frames per second
        self.position_history = deque(maxlen=GAME_SPEED_FRAMES_PER_SECOND)
        self.previous_position = None

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

    def _get_numerical_features(self, obs, prev_obs):
        """
        Process numerical features into grouped format for our enhanced network architecture.

        The features are organized into three logical groups:
        1. Position features: Current position and velocity (4 features)
        2. Objective features: Distances to goals (4 features)
        3. State features: Game state information (3 features)
        """
        # Calculate normalized positions and distances first
        max_level_width = 1258
        max_level_height = 802
        max_time = 600.0

        # Calculate relative distances to objectives
        exit_door_dist_x = obs['exit_door_x'] - obs['player_x']
        exit_door_dist_y = obs['exit_door_y'] - obs['player_y']
        switch_dist_x = obs['switch_x'] - obs['player_x']
        switch_dist_y = obs['switch_y'] - obs['player_y']

        # Calculate normalized velocities
        vx, vy = self._calculate_velocity(prev_obs, obs)
        normalized_vx = (vx + self.MAX_VELOCITY) / (2 * self.MAX_VELOCITY)
        normalized_vy = (vy + self.MAX_VELOCITY) / (2 * self.MAX_VELOCITY)

        # Group 1: Position features (player position and velocity)
        position_features = np.array([
            # Normalized x position
            (obs['player_x'] - 63) / (1217 - 63),
            # Normalized y position
            (obs['player_y'] - 171) / (791 - 171),
            normalized_vx,                                 # Normalized x velocity
            normalized_vy                                  # Normalized y velocity
        ], dtype=np.float32)

        # Group 2: Objective features (distances to goals)
        objective_features = np.array([
            (exit_door_dist_x + max_level_width) /
            (2 * max_level_width),   # Exit x distance
            (exit_door_dist_y + max_level_height) / \
            (2 * max_level_height),  # Exit y distance
            (switch_dist_x + max_level_width) / \
            (2 * max_level_width),      # Switch x distance
            (switch_dist_y + max_level_height) / \
            (2 * max_level_height)     # Switch y distance
        ], dtype=np.float32)

        # Group 3: State features (game state information)
        state_features = np.array([
            obs['time_remaining'] / max_time,              # Normalized time
            float(obs['switch_activated']),                # Switch status
            float(obs['in_air'])                          # Air status
        ], dtype=np.float32)

        # Combine all features in the correct order
        features = np.concatenate([
            position_features,    # First 4 features
            objective_features,   # Next 4 features
            state_features       # Last 3 features
        ])

        # Assert our features are NUM_NUMERICAL_FEATURES long
        assert len(features) == NUM_NUMERICAL_FEATURES

        # Reshape to match network expectations
        features = features.reshape((1, 1, NUM_NUMERICAL_FEATURES))
        features = np.broadcast_to(features, (84, 84, NUM_NUMERICAL_FEATURES))

        return features.astype(np.float32)

    def _get_stacked_observation(self, obs, prev_obs):
        """Combine frame stack with numerical features properly.

        Returns:
            numpy.ndarray: Combined observation of shape (84, 84, frame_stack + NUM_NUMERICAL_FEATURES)
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
        # Shape: (84, 84, NUM_NUMERICAL_FEATURES)
        features = self._get_numerical_features(obs, prev_obs)

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
            'switch_y': self.gvf.read_switch_y(),
            'in_air': self.gvf.read_in_air(),
            # Note: velocity values are calculated indirectly in _get_numerical_features
            # because we don't current have direct access to velocity
        }

    def _calculate_velocity(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate player velocity based on previous and current observations.

        Args:
            prev_obs: Previous observation dictionary
            curr_obs: Current observation dictionary
        """
        # Calculate time elapsed between observations

        # Calculate distance traveled in x and y directions
        dx = curr_obs['player_x'] - prev_obs['player_x']
        dy = curr_obs['player_y'] - prev_obs['player_y']

        # Calculate velocity components
        vx = dx / TIMESTEP
        vy = dy / TIMESTEP

        return vx, vy

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
            total_distance += calculate_distance(
                prev_x, prev_y, curr_x, curr_y)

        print(
            f'Total distance traveled in last {self.MOVEMENT_CHECK_DURATION} seconds: {total_distance}')
        return total_distance < self.MOVEMENT_THRESHOLD

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

    def _calculate_movement_efficiency(self, curr_state, prev_state, action):
        """
        Calculate movement efficiency using the MovementEvaluator's comprehensive analysis.

        This method leverages the MovementEvaluator's deep understanding of platforming
        mechanics to evaluate movement quality across multiple dimensions.
        """
        # Get full movement evaluation
        movement_results = self.movement_evaluator.evaluate_movement_success(
            current_state=curr_state,
            previous_state=prev_state,
            action_taken=action
        )

        # Extract metrics
        metrics = movement_results['metrics']

        # Calculate weighted efficiency score
        movement_efficiency = (
            metrics['precision'] * 0.3 +      # Control precision
            metrics['landing'] * 0.25 +       # Landing quality
            metrics['momentum'] * 0.25 +      # Momentum management
            metrics['progress'] * 0.2         # Progress toward objectives
        )

        return {
            'efficiency': movement_efficiency,
            'has_meaningful_movement': movement_results['has_meaningful_movement'],
            'precision': metrics['precision'],
            'landing': metrics['landing'],
            'momentum': metrics['momentum'],
            'progress_score': metrics['progress'],
            'segment_success': metrics['segment_success']
        }

    def _calculate_objective_progress(self, curr_state, prev_state):
        """Calculate progress toward level objectives with enhanced metrics."""
        # Track progress toward primary objectives
        if not curr_state['switch_activated']:
            # Calculate progress toward switch
            current_distance = np.sqrt(
                (curr_state['player_x'] - curr_state['switch_x'])**2 +
                (curr_state['player_y'] - curr_state['switch_y'])**2
            )
            self.best_switch_distance = min(
                self.best_switch_distance, current_distance)

            switch_progress = 1.0 - \
                (current_distance / (self.best_switch_distance + 1e-6))
            progress_score = max(0, switch_progress)
        else:
            # Calculate progress toward exit
            current_distance = np.sqrt(
                (curr_state['player_x'] - curr_state['exit_door_x'])**2 +
                (curr_state['player_y'] - curr_state['exit_door_y'])**2
            )
            self.best_exit_distance = min(
                self.best_exit_distance, current_distance)

            exit_progress = 1.0 - (current_distance /
                                   (self.best_exit_distance + 1e-6))
            progress_score = max(0, exit_progress)

        # Calculate time management success
        current_time = curr_state['time_remaining']
        time_gained = current_time - \
            self.previous_time if self.previous_time is not None else 0
        self.max_time_gained = max(self.max_time_gained, time_gained)
        self.previous_time = current_time

        time_efficiency = current_time / (self.initial_time + 1e-6)
        time_efficiency = np.clip(time_efficiency, 0, 1)

        return {
            'objective_progress': progress_score,
            'time_efficiency': time_efficiency,
            'time_gained': float(time_gained > self.TIME_GAIN_THRESHOLD)
        }

    def _get_success_metrics(self, curr_state, prev_state, action):
        """Compile comprehensive success metrics combining movement and objectives."""
        # Get movement efficiency metrics
        movement_metrics = self._calculate_movement_efficiency(
            curr_state, prev_state, action)

        # Get objective progress metrics
        progress_metrics = self._calculate_objective_progress(
            curr_state, prev_state)

        # Update success tracking
        self.switch_activated = curr_state.get('switch_activated', False)
        self.died = self.gvf.read_player_dead()
        self.time_expired = curr_state.get('time_remaining', 0) <= 0
        self.completed_level = 'retry level' in self.gvf.read_begin_retry_text().lower()

        # Calculate overall success conditions
        movement_success = movement_metrics['efficiency'] > self.MOVEMENT_SUCCESS_THRESHOLD
        progress_success = progress_metrics['objective_progress'] > self.PROGRESS_SUCCESS_THRESHOLD

        self.current_episode_success = (
            self.completed_level and
            self.switch_activated and
            not self.died and
            not self.time_expired and
            movement_success and
            progress_success
        )

        return {
            # Overall success metrics
            'success': float(self.current_episode_success),
            'switch_activated': float(self.switch_activated),
            'died': float(self.died),
            'time_expired': float(self.time_expired),
            'completed_level': float(self.completed_level),

            # Movement quality metrics
            'movement_efficiency': movement_metrics['efficiency'],
            'movement_precision': movement_metrics['precision'],
            'landing_quality': movement_metrics['landing'],
            'momentum_efficiency': movement_metrics['momentum'],
            'movement_progress': movement_metrics['progress_score'],
            'segment_success': movement_metrics['segment_success'],
            'meaningful_movement': float(movement_metrics['has_meaningful_movement']),

            # Objective progress metrics
            'objective_progress': progress_metrics['objective_progress'],
            'time_efficiency': progress_metrics['time_efficiency'],
            'time_gained': progress_metrics['time_gained']
        }

    def step(self, action):
        """
        Execute one environment step with tracking and evaluation.

        Provides:
        1. Success tracking
        2. Movement evaluation
        3. Progressive reward calculation
        4. Detailed metrics and monitoring
        """
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

        # Wait our timestep for the action to take effect and the game state to update
        time.sleep(TIMESTEP)

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
        processed_obs = self._get_stacked_observation(observation, prev_obs)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            observation, prev_obs, action)

        # Prepare info
        info = {
            'raw_reward': reward,
            'time_remaining': observation['time_remaining'],
            'switch_activated': observation['switch_activated'],
            'player_dead': self.gvf.read_player_dead(),
            'begin_retry_text': self.gvf.read_begin_retry_text()
        }

        # Calculate movement success metrics
        movement_results = self.movement_evaluator.evaluate_movement_success(
            current_state=observation,
            previous_state=prev_obs,
            action_taken=action
        )

        # Add movement metrics to info
        info.update({
            'movement_efficiency': movement_results['metrics']['precision'],
            'landing_quality': movement_results['metrics']['landing'],
            'momentum_efficiency': movement_results['metrics']['momentum'],
            'movement_progress': movement_results['metrics']['progress'],
            'segment_success': movement_results['metrics']['segment_success'],
            'has_meaningful_movement': movement_results['has_meaningful_movement']
        })

        # Calculate distance-based metrics
        curr_switch_dist = calculate_distance(
            observation['player_x'], observation['player_y'],
            observation['switch_x'], observation['switch_y']
        )

        curr_exit_dist = calculate_distance(
            observation['player_x'], observation['player_y'],
            observation['exit_door_x'], observation['exit_door_y']
        )

        # Add progress metrics
        if not observation['switch_activated']:
            progress = max(0, 1 - (curr_switch_dist /
                           (self.best_switch_distance + 1e-6)))
        else:
            progress = max(
                0, 1 - (curr_exit_dist / (self.best_exit_distance + 1e-6)))

        info.update({
            'objective_progress': progress,
            'time_efficiency': observation['time_remaining'] / self.initial_time if self.initial_time > 0 else 0,
            'time_gained': float(observation['time_remaining'] > prev_obs['time_remaining'])
        })

        # Update best distances
        if not observation['switch_activated']:
            self.best_switch_distance = min(
                self.best_switch_distance, curr_switch_dist)
        else:
            self.best_exit_distance = min(
                self.best_exit_distance, curr_exit_dist)

        # Calculate overall success metrics
        success_metrics = {
            'success': float(
                not terminated and
                not truncated and
                observation['switch_activated'] and
                'retry level' in self.gvf.read_begin_retry_text().lower()
            ),
            'switch_activated': float(observation['switch_activated']),
            'died': float(self.gvf.read_player_dead()),
            'time_expired': float(observation['time_remaining'] <= 0),
            'completed_level': float('retry level' in self.gvf.read_begin_retry_text().lower())
        }

        # Add success metrics to info
        info.update(success_metrics)

        if terminated or truncated:
            print("\nEpisode finished!")
            print(f"Success: {success_metrics['success']}")
            print(f"Switch Activated: {success_metrics['switch_activated']}")
            print(f"Died: {success_metrics['died']}")
            print(f"Time Expired: {success_metrics['time_expired']}")
            print(f"Time Gained: {info['time_gained']}")
            print(f"Objective Progress: {info['objective_progress']}")
            print(f"Level Completed: {success_metrics['completed_level']}\n")

            # Update progression metrics for reward scaling
            print("Updating progression metrics...")
            self.reward_calculator.update_progression_metrics()

            # Log final positions and actions
            print("Logging final positions and actions...")
            with open(f'{self.position_log_folder_name}/position_log_{self.episode_counter}.csv', 'w') as f:
                f.write(self.position_log_file_string)

            with open(f'{self.action_log_folder_name}/action_log_{self.episode_counter}.csv', 'w') as f:
                f.write(self.action_log_file_string)

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

        # Reset success flags
        self.current_episode_success = False
        self.switch_activated = False
        self.died = False
        self.time_expired = False
        self.completed_level = False

        # Reset progress tracking
        self.initial_time = None
        self.max_time_gained = 0
        self.previous_time = None
        self.best_switch_distance = float('inf')
        self.best_exit_distance = float('inf')
        self.position_history.clear()
        self.previous_position = None

        # Reset frame stack
        self.frames.clear()

        # Reset position and velocity history
        self.position_history.clear()

        # Reset movement evaluator
        self.movement_evaluator.reset()

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

        # Initialize time tracking
        self.initial_time = observation.get('time_remaining', 0)
        self.previous_time = self.initial_time

        # For initial success metrics, we should:
        # 1. Only include metrics that make sense at episode start
        # 2. Use appropriate default values for comparative metrics
        initial_metrics = {
            'success': 0.0,
            'switch_activated': 0.0,
            'died': 0.0,
            'time_expired': 0.0,
            'completed_level': 0.0,
            'movement_efficiency': 1.0,  # Start with perfect efficiency
            'movement_precision': 1.0,   # Start with perfect precision
            'landing_quality': 1.0,      # Start with perfect landing
            'momentum_efficiency': 1.0,  # Start with perfect momentum
            'movement_progress': 0.0,    # No progress yet
            'segment_success': 0.0,      # No segments yet
            'meaningful_movement': 0.0,  # No movement yet
            'objective_progress': 0.0,   # No progress yet
            'time_efficiency': 1.0,      # Start with full time
            'time_gained': 0.0          # No time gained yet
        }

        # Update info with initial metrics
        info = initial_metrics

        # # Press pause key to pause the game. We are sure we are in the 'level playing' state,
        # # since we verified the player has moved.
        # print(f'Center text before pause: {center_text}')
        # # Print player position before pause
        # print(
        #     f"Player position before pause: ({observation['player_x']}, {observation['player_y']})")
        # # pause_key_success = self.gc.press_pause_key()
        # # print("Pause key press success:", pause_key_success)

        print('Game started')

        # Process observation for A2C
        processed_obs = self._get_stacked_observation(observation, observation)

        return processed_obs, info

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
