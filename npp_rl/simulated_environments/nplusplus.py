import gymnasium
from gymnasium.spaces import discrete, box
import numpy as np
from npp_rl.simulated_environments.reward_calculation.main_reward_calculator import RewardCalculator
import time
from typing import Tuple, Dict, Any
import os
from npp_rl.simulated_environments.observation_processor import ObservationProcessor
from nplay_headless import NPlayHeadless
import uuid

OBSERVATION_IMAGE_SIZE = 84


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

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, frame_stack: int = 4):
        """Initialize the environment.

        Args:
            frame_stack (int, optional): Number of frames to stack. Defaults to 4.
        """
        super().__init__()

        self.nplay_headless = NPlayHeadless()
        self.nplay_headless.load_map("../nclone/map_data")

        self.frame_stack = frame_stack

        self.reward_calculator = RewardCalculator()

        # Initialize observation processing
        self.observation_processor = ObservationProcessor(
            frame_stack=frame_stack)

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize observation space
        self.observation_space = box.Box(
            low=0,
            high=1,
            shape=(OBSERVATION_IMAGE_SIZE, OBSERVATION_IMAGE_SIZE,
                   7),
            dtype=np.float32
        )

        # Initialize position log folder
        # add uuid to folder name
        self.position_log_folder_name = f'training_logs/{time.strftime("%m-%d-%Y_%H-%M-%S-")}/{uuid.uuid4()}/position_log'
        os.makedirs(self.position_log_folder_name)

        # Initialize position log file string
        self.position_log_file_string = 'PlayerX,PlayerY\n'

        # Initialize action log folder
        self.action_log_folder_name = f'training_logs/{time.strftime("%m-%d-%Y_%H-%M-%S-")}/{uuid.uuid4()}/action_log'
        os.makedirs(self.action_log_folder_name)

        # Initialize action log file string
        self.action_log_file_string = 'Action\n'

        # Initialize episode counter
        self.episode_counter = 0

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        obs = {
            'screen': self.nplay_headless.render(),
            'player_x': self.nplay_headless.ninja_position()[0],
            'player_y': self.nplay_headless.ninja_position()[1],
            'player_dead': self.nplay_headless.ninja_has_died(),
            'player_won': self.nplay_headless.ninja_has_won()
        }
        return obs

    def _actions_to_execute(self, action: int) -> Tuple[int, int]:
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
        hoz_input = 0
        jump_input = 0

        # Execute the new action
        if action == 0:  # NOOP
            pass
        elif action == 1:  # Left
            hoz_input = -1
        elif action == 2:  # Right
            hoz_input = 1
        elif action == 3:  # Jump
            jump_input = 1
        elif action == 4:  # Jump + Left
            jump_input = 1
            hoz_input = -1
        elif action == 5:  # Jump + Right
            jump_input = 1
            hoz_input = 1

        return hoz_input, jump_input

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

    def _check_termination(self, observation: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if the episode should be terminated.

        Args:
            observation: Current observation dictionary

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
        """
        terminated = observation.get(
            'player_won', False) or observation.get('player_dead', False)

        if terminated:
            print(
                f"Episode terminated at frame {self.nplay_headless.sim.frame}")

        # Check truncation
        # Truncation is when the current simulation frame is greater than 30000
        truncated = self.nplay_headless.sim.frame > 500

        if truncated:
            print(
                f"Episode truncated at frame {self.nplay_headless.sim.frame}")

        return terminated, truncated

    def _get_success_metrics(self):
        """Compile comprehensive success metrics combining movement and objectives."""
        return {
            # Overall success metrics
            'success': float(self.nplay_headless.ninja_has_won())
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
        player_x, player_y = self.nplay_headless.ninja_position()

        self.position_log_file_string += f'{player_x},{player_y}\n'
        self.action_log_file_string += f'{self._action_to_string(action)}\n'

        # Execute action
        action_hoz, action_jump = self._actions_to_execute(action)
        self.nplay_headless.tick(action_hoz, action_jump)

        # Get new observation
        observation = self._get_observation()

        # Check termination
        terminated, truncated = self._check_termination(observation)

        # Process observation and calculate reward
        processed_obs = self.observation_processor.process_observation(
            observation)
        reward = self.reward_calculator.calculate_reward(observation)

        success_metrics = self._get_success_metrics()

        # Print reward if terminated or truncated
        if terminated or truncated:
            print(f"Episode reward: {reward}")

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
        # Increment episode counter
        self.episode_counter += 1

        # Reset observation processor
        self.observation_processor.reset()

        # Reset position and action logs
        self.position_log_file_string = 'PlayerX,PlayerY\n'
        self.action_log_file_string = 'Action\n'

        # reset level
        self.nplay_headless.reset()

        # Get initial observation
        initial_obs = self._get_observation()

        # Get processed observation
        processed_obs = self.observation_processor.process_observation(
            initial_obs)

        return processed_obs, {}

    def render(self):
        """Render the environment.

        Since the game window is already visible, we don't need additional rendering.
        We can just return the current game window frame.
        """
        return self.nplay_headless.render()

    def close(self):
        """Clean up environment resources."""
        self.nplay_headless.exit()
