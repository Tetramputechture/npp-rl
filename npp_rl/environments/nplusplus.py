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
    * Mines (hazards)
    * Terrain (walls, platforms)
- The player must reach the exit door after activating the switch to complete the level
- The player has a timer that depletes every second, with gold pieces adding time
- The entire game state is observable through memory reading and screen capture
- Failure occurs through player death or time expiration
- Success occurs when reaching the exit door with the switch activated

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
Dictionary containing:
- 'screen': 4-frame stack of preprocessed grayscale images (84x84 pixels)
- 'player_x': Normalized player x-coordinate
- 'player_y': Normalized player y-coordinate
- 'time_remaining': Normalized time remaining
- 'switch_activated': Boolean as float (0 or 1)
- 'exit_door_x': Normalized exit door x-coordinate
- 'exit_door_y': Normalized exit door y-coordinate
- 'switch_x': Normalized switch x-coordinate
- 'switch_y': Normalized switch y-coordinate
- 'in_air': Boolean as float (0 or 1)

The observation is processed through ObservationProcessor which:
- Resizes and normalizes screen frames to 84x84 pixels
- Maintains a frame stack of 4 frames
- Normalizes all numerical features to [0, 1]
- Combines visual and numerical features into a single tensor

Rewards:
-------
The reward system is composed of multiple components:
1. Main Reward: Base rewards for key achievements (switch activation, level completion)
2. Movement Reward: Evaluates efficiency and quality of movement
3. Navigation Reward: Rewards progress towards objectives
4. Time Reward: Penalties for time usage and rewards for gold collection

Each component is calculated by specialized reward calculators that consider:
- Distance to objectives
- Movement efficiency
- Time management
- Gold collection
- Hazard avoidance

Game State Management:
--------------------
The environment manages game state through:
1. GameValueFetcher: Reads game memory for precise state information
2. GameController: Handles game input and window management
3. TrainingSession: Manages training progress and configuration
4. Fixed timestep execution (1/60th second) for consistent physics simulation

Additional Features:
-----------------
1. Position Logging: Tracks player movement for analysis
2. Movement Evaluation: Assesses movement quality and efficiency
3. Spatial Memory: Tracks visited areas and exploration
4. Mine Avoidance: Special handling of hazardous areas
5. Training Session Management: Configurable training parameters

Version: v2.0

Render Modes:
------------
- 'human': Displays game window

Max Episode Steps:
----------------
~36000 frames (600 seconds at 60 FPS)

Dependencies:
-----------
- gymnasium
- numpy
- pymem (for memory reading)
- win32gui (for window management)
- pygame (for input simulation)
"""
import gymnasium
from gymnasium.spaces import discrete, box
import numpy as np
from npp_rl.game.game_controller import GameController
from npp_rl.game.game_window import get_game_window_frame
from npp_rl.game.game_value_fetcher import GameValueFetcher
from npp_rl.environments.reward_calculation import RewardCalculator
from npp_rl.util.util import calculate_distance
from npp_rl.environments.movement_evaluator import MovementEvaluator
from npp_rl.environments.constants import TIMESTEP, GAME_SPEED_FRAMES_PER_SECOND, TOTAL_OBSERVATION_CHANNELS, OBSERVATION_IMAGE_SIZE
from npp_rl.game.game_config import game_config
import time
from typing import Tuple, Dict, Any, List
import os
from npp_rl.environments.observation_processor import ObservationProcessor
from collections import deque
from npp_rl.game.training_session import training_session


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

    def __init__(self, gvf: GameValueFetcher, gc: GameController, frame_stack: int = 4):
        """Initialize the environment.

        Args:
            gvf (GameValueFetcher): Game value fetcher instance
            gc (GameController): Game controller instance
            frame_stack (int, optional): Number of frames to stack. Defaults to 4.
        """
        super().__init__()

        # Game interfaces
        self.gvf = gvf
        self.gc = gc

        # Use global training session
        self.training_session = training_session

        self.frame_stack = frame_stack
        self.mine_coords: List[Tuple[float, float]] = []

        # Initialize observation processing
        self.observation_processor = ObservationProcessor(
            frame_stack=frame_stack)

        # Initialize movement evaluator and reward calculator
        self.movement_evaluator = MovementEvaluator()
        self.reward_calculator = RewardCalculator(self.movement_evaluator)

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize observation space
        self.observation_space = box.Box(
            low=0,
            high=1,
            shape=(OBSERVATION_IMAGE_SIZE, OBSERVATION_IMAGE_SIZE,
                   TOTAL_OBSERVATION_CHANNELS),
            dtype=np.float32
        )

        # Initialize position log folder
        self.position_log_folder_name = f'training_logs/{time.strftime("%m-%d-%Y_%H-%M-%S")}/position_log'
        os.makedirs(self.position_log_folder_name)

        # Initialize position log file string
        self.position_log_file_string = 'PlayerX,PlayerY\n'

        # Initialize action log folder
        self.action_log_folder_name = f'training_logs/{time.strftime("%m-%d-%Y_%H-%M-%S")}/action_log'
        os.makedirs(self.action_log_folder_name)

        # Initialize action log file string
        self.action_log_file_string = 'Action\n'

        # Initialize episode counter
        self.episode_counter = 0

        # Initialize current level playable space (x1, y1, x2, y2)        # Initialize position history
        self.position_history = deque(maxlen=self.MOVEMENT_CHECK_FRAMES)

        # Initialize success tracking
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

        self.level_data = None

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        player_level_x, player_level_y = self.level_data.convert_game_to_level_coordinates(
            self.gvf.read_player_x(),
            self.gvf.read_player_y()
        )

        level_x, level_y, level_x2, level_y2 = self.level_data.playable_space
        level_width = level_x2 - level_x
        level_height = level_y2 - level_y

        obs = {
            'screen': get_game_window_frame(self.level_data.playable_space),
            'time_remaining': self.gvf.read_time_remaining(),
            'player_x': self.gvf.read_player_x(),
            'player_y': self.gvf.read_player_y(),
            'switch_activated': self.gvf.read_switch_activated(),
            'switch_x': self.gvf.read_switch_x(),
            'switch_y': self.gvf.read_switch_y(),
            'exit_door_x': self.gvf.read_exit_door_x(),
            'exit_door_y': self.gvf.read_exit_door_y(),
            'player_dead': self.gvf.read_player_dead(),
            'begin_retry_text': self.gvf.read_begin_retry_text(),
            'in_air': self.gvf.read_in_air(),
            'level_width': level_width,
            'level_height': level_height,
            'closest_mine_distance': self._get_closest_mine_distance(
                player_level_x, player_level_y),
            'closest_mine_vector': self._get_closest_mine_vector(
                player_level_x, player_level_y),
            'level_data': self.level_data
        }
        return obs

    def _get_closest_mine_vector(self, level_x: float, level_y: float) -> Tuple[float, float]:
        """Calculate vector to closest mine using ParsedLevel utility.

        Args:
            level_x: Player x position in level coordinates
            level_y: Player y position in level coordinates

        Returns:
            Tuple[float, float]: Vector from player to closest mine
        """
        nearest_mine = self.level_data.get_nearest_mine(level_x, level_y)
        if nearest_mine is None:
            return (0.0, 0.0)

        return (nearest_mine[0] - level_x, nearest_mine[1] - level_y)

    def _get_closest_mine_distance(self, level_x: float, level_y: float) -> float:
        """Calculate distance to closest mine using ParsedLevel utility.

        Args:
            level_x: Player x position in level coordinates
            level_y: Player y position in level coordinates

        Returns:
            float: Distance to nearest mine, or infinity if no mines
        """
        nearest_mine = self.level_data.get_nearest_mine(level_x, level_y)
        if nearest_mine is None:
            return float('inf')

        mine_x, mine_y = nearest_mine
        return np.sqrt((mine_x - level_x)**2 + (mine_y - level_y)**2)

    def _count_mines_in_radius(self, level_x: float, level_y: float, radius: float) -> int:
        """Count mines within radius using ParsedLevel utility.

        Args:
            level_x: Player x position in level coordinates
            level_y: Player y position in level coordinates
            radius: Radius to check for mines

        Returns:
            int: Number of mines within radius
        """
        mines = self.level_data.get_mines_in_radius(level_x, level_y, radius)
        return len(mines)

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
            # Manually hit the reset key to reset the level
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

        player_x = self.gvf.read_player_x()
        player_y = self.gvf.read_player_y()

        self.position_log_file_string += f'{player_x},{player_y}\n'
        self.training_session.add_position(player_x, player_y)
        self.action_log_file_string += f'{self._action_to_string(action)}\n'

        # Continue game with advanced continue
        # self.gc.press_advanced_continue_key()

        # Execute action
        self._execute_action(action)

        # Wait scaled timestep
        print(f"Waiting for {TIMESTEP} seconds")
        time.sleep(TIMESTEP)

        # Pause game with advanced pause
        # self.gc.press_advanced_pause_key()

        # Get new observation
        observation = self._get_observation()

        # Check termination
        terminated, truncated = self._check_termination(observation)

        # Process observation and calculate reward
        processed_obs = self.observation_processor.process_observation(
            observation)
        reward = self.reward_calculator.calculate_reward(
            observation, prev_obs, action)

        # Add action and reward in sync (action first, then its resulting reward)
        self.training_session.add_action(action)
        self.training_session.add_reward(reward)

        # Update best reward at episode end
        if terminated or truncated:
            self.training_session.update_best_reward(
                self.training_session.current_episode_reward)

        return processed_obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode.

        This method:
        1. Resets game state with proper key presses
        2. Gets initial observation
        3. Updates mine coordinates for the new level
        4. Returns initial observation

        Returns:
            tuple: Initial observation and info dict
        """
        # Update training session episode counter - this will store current episode data
        # in historical arrays and reset current episode arrays
        self.training_session.increment_episode()

        # Reset success tracking
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

        # Reset position history for movement tracking
        self.position_history = []

        # Increment episode counter
        self.episode_counter += 1

        # Reset reward calculator and movement evaluator
        self.reward_calculator.reset()
        self.movement_evaluator.reset()

        # Reset observation processor
        self.observation_processor.reset()

        # Reset position and action logs
        self.position_log_file_string = 'PlayerX,PlayerY\n'
        self.action_log_file_string = 'Action\n'

        # Release all held keys
        self.gc.release_all_keys()

        # Make sure game is continued before we start
        self.gc.press_advanced_continue_key()

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
            # if this loops, 'pause' and 'retry' are shown at the same time
            self.gc.press_space_key()
            time.sleep(0.1)

        print("Pressing space to go to the 'level playing' state...")
        self.gc.press_space_key()

        # Get level data from game config
        self.level_data = game_config.level_data
        if self.level_data is None:
            raise ValueError(
                "Level data not set in game config. Please set level data first.")

        # Get mine coordinates from level data
        self.mine_coords = self.level_data.mine_coordinates

        # Update reward calculator with mine coordinates
        self.reward_calculator.navigation_calculator.set_mine_coordinates(
            self.mine_coords)

        # Get initial observation
        initial_obs = self._get_observation()

        self.initial_time = initial_obs['time_remaining']

        # Get processed observation
        processed_obs = self.observation_processor.process_observation(
            initial_obs)

        return processed_obs, {}

    def render(self):
        """Render the environment.

        Since the game window is already visible, we don't need additional rendering.
        We can just return the current game window frame.
        """
        return get_game_window_frame(self.level_data.playable_space)

    def close(self):
        """Clean up environment resources."""
        # Ensure all keys are released when environment is closed
        self.gc.release_all_keys()
