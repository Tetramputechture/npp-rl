import gymnasium
from gymnasium.spaces import discrete, box, Dict
import numpy as np
from npp_rl.environments.reward_calculation.main_reward_calculator import RewardCalculator
from npp_rl.environments.reward_calculation.planning_reward_calculator import PlanningRewardCalculator
from npp_rl.environments.planning.path_planner import PathPlanner
from npp_rl.environments.planning.waypoint_manager import WaypointManager
import time
from typing import Tuple
import os
from npp_rl.environments.movement_evaluator import MovementEvaluator
from nplay_headless import NPlayHeadless
import uuid
from npp_rl.environments.constants import (
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,
    OBSERVATION_IMAGE_HEIGHT,
    OBSERVATION_IMAGE_WIDTH,
    TEMPORAL_FRAMES,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT
)
from npp_rl.environments.visualization.path_visualizer import PathVisualizer
from npp_rl.environments.constants import MAX_TIME_IN_FRAMES
from npp_rl.environments.observation_processor import ObservationProcessor

MAP_DATA_PATH = "../nclone/maps/map_data_complex"


class NPlusPlus(gymnasium.Env):
    """Custom Gym environment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: str = 'rgb_array', enable_frame_stack: bool = False):
        """Initialize the environment."""
        super().__init__()

        self.nplay_headless = NPlayHeadless(render_mode=render_mode)
        self.nplay_headless.load_random_map()
        # self.nplay_headless.load_map(MAP_DATA_PATH)

        self.render_mode = render_mode

        # Initialize observation processor
        self.observation_processor = ObservationProcessor(
            enable_frame_stack=enable_frame_stack)

        # Initialize planning components
        self.path_planner = PathPlanner()
        self.waypoint_manager = WaypointManager(self.path_planner)
        self.planning_reward_calculator = PlanningRewardCalculator(
            waypoint_manager=self.waypoint_manager)

        # Initialize movement evaluator
        self.movement_evaluator = MovementEvaluator()

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            movement_evaluator=self.movement_evaluator)

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize observation space as a Dict space with player_frame, base_frame, and game_state
        player_frame_dimension_count = TEMPORAL_FRAMES if enable_frame_stack else 1
        self.observation_space = Dict({
            # Player-centered frame
            'player_frame': box.Box(
                low=0,
                high=255,
                shape=(PLAYER_FRAME_HEIGHT, PLAYER_FRAME_WIDTH,
                       player_frame_dimension_count),
                dtype=np.uint8
            ),
            # Base frame (full screen)
            'base_frame': box.Box(
                low=0,
                high=255,
                shape=(OBSERVATION_IMAGE_WIDTH, OBSERVATION_IMAGE_HEIGHT, 1),
                dtype=np.uint8
            ),
            # Game state features
            'game_state': box.Box(
                low=-1,
                high=1,
                # + 4 for vectors to objectives
                shape=(GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH + 4,),
                dtype=np.float32
            )
        })

        # Initialize path visualizer
        self.path_visualizer = PathVisualizer()
        self.current_path = []

        # Initialize frame buffer for path visualization
        self.path_viz_buffer = []

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (MAX_TIME_IN_FRAMES -
                          self.nplay_headless.sim.frame) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()
        entity_states = self.nplay_headless.get_entity_states(
            only_one_exit_and_switch=True)
        game_state = np.concatenate([ninja_state, entity_states])

        return {
            'screen': self.render(),
            'game_state': game_state,
            'player_dead': self.nplay_headless.ninja_has_died(),
            'player_won': self.nplay_headless.ninja_has_won(),
            'player_x': self.nplay_headless.ninja_position()[0],
            'player_y': self.nplay_headless.ninja_position()[1],
            'switch_activated': self.nplay_headless.exit_switch_activated(),
            'switch_x': self.nplay_headless.exit_switch_position()[0],
            'switch_y': self.nplay_headless.exit_switch_position()[1],
            'exit_door_x': self.nplay_headless.exit_door_position()[0],
            'exit_door_y': self.nplay_headless.exit_door_position()[1],
            'time_remaining': time_remaining,
            'sim_frame': self.nplay_headless.sim.frame,
        }

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

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if the episode should be terminated.

        Args:
            observation: Current observation array

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
        """
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.ninja_has_died()
        terminated = player_won or player_dead

        # Check truncation
        # Truncation is when the current simulation frame is greater than 5000
        truncated = self.nplay_headless.sim.frame > MAX_TIME_IN_FRAMES

        return terminated, truncated, player_won

    def _get_success_metrics(self):
        """Compile comprehensive success metrics combining movement and objectives."""
        return {
            # Overall success metrics
            'success': float(self.nplay_headless.ninja_has_won())
        }

    def _update_planning(self, observation: np.ndarray):
        """Update path planning based on current state."""
        # Get current position
        current_pos = (observation['player_x'], observation['player_y'])

        # Get target position based on game state
        if not observation['switch_activated']:
            target_pos = (observation['switch_x'], observation['switch_y'])
        else:
            target_pos = (observation['exit_door_x'],
                          observation['exit_door_y'])

        # Get tile and segment data from simulator
        tile_data = self.nplay_headless.get_tile_data()
        segment_data = self.nplay_headless.get_segment_data()
        grid_edges = self.nplay_headless.get_grid_edges()
        segment_edges = self.nplay_headless.get_segment_edges()

        # Update collision grid with all available data
        self.path_planner.update_collision_grid(
            tile_data=tile_data,
            segment_data=segment_data,
            grid_edges=grid_edges,
            segment_edges=segment_edges
        )

        # Find path to target
        path = self.path_planner.find_path(current_pos, target_pos)

        # Store current path for visualization
        self.current_path = path

        # Update waypoint manager
        if path:
            self.waypoint_manager.update_path(path)
            self.planning_reward_calculator.update_path(path)

    def _save_path_visualization(self, observation: np.ndarray, prev_obs: np.ndarray):
        """Save visualization of current path and agent state to buffer."""
        return
        if not self.current_path:
            return

        # Get current frame
        frame = self.render()

        # Get current position and goal
        current_pos = (observation['player_x'], observation['player_y'])
        if not observation['switch_activated']:
            goal_pos = (observation['switch_x'], observation['switch_y'])
        else:
            goal_pos = (observation['exit_door_x'], observation['exit_door_y'])

        prev_pos = (prev_obs['player_x'], prev_obs['player_y'])

        # Get current waypoint
        current_waypoint = self.waypoint_manager.get_current_waypoint()

        # Get path deviation
        metrics = self.waypoint_manager.calculate_metrics(
            current_pos, prev_pos)
        path_deviation = metrics.path_deviation if metrics else 0.0

        # Render path visualization
        viz_frame = self.path_visualizer.render_path(
            frame=frame,
            planned_path=self.current_path,
            current_pos=current_pos,
            current_waypoint=current_waypoint,
            goal_pos=goal_pos,
            path_deviation=path_deviation,
            distance_to_waypoint=metrics.distance_to_waypoint,
            episode_reward=self.current_episode_reward
        )

        # Add frame to buffer
        self.path_viz_buffer.append(viz_frame)

    def _save_path_video(self):
        """Save path visualization buffer as video."""
        if not self.path_viz_buffer:
            return

        # # Create video filename with HMS timestamp
        # timestamp = time.strftime("%H-%M-%S")
        # video_path = os.path.join(
        #     self.path_visualizer.session_dir, f'paths_{timestamp}.mp4')

        # # Save video using imageio
        # if len(self.path_viz_buffer) > 0:
        #     imageio.mimsave(video_path, self.path_viz_buffer, fps=30)

        # Clear buffer
        self.path_viz_buffer = []

    def step(self, action: int):
        """Execute one environment step with planning and visualization."""
        # Get previous observation
        prev_obs = self._get_observation()

        # Execute action
        action_hoz, action_jump = self._actions_to_execute(action)
        self.nplay_headless.tick(action_hoz, action_jump)

        # Get current observation
        curr_obs = self._get_observation()
        terminated, truncated, player_won = self._check_termination()

        # Calculate reward
        movement_reward = self.reward_calculator.calculate_reward(
            curr_obs, prev_obs, action)
        reward = movement_reward

        # Process observation using ObservationProcessor
        processed_obs = self.observation_processor.process_observation(
            curr_obs)

        ep_info = {'is_success': player_won}

        return processed_obs, reward, terminated, truncated, ep_info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Reset observation processor
        self.observation_processor.reset()

        # Reset reward calculator
        self.reward_calculator.reset()

        # Reset level and load random map
        self.nplay_headless.reset()
        self.nplay_headless.load_random_map()
        # self.nplay_headless.load_map(MAP_DATA_PATH)

        # Get initial observation and process it
        initial_obs = self._get_observation()
        processed_obs = self.observation_processor.process_observation(
            initial_obs)

        return processed_obs, {}

    def render(self):
        """Render the environment."""
        return self.nplay_headless.render()

    def close(self):
        """Clean up environment resources."""
        # self.nplay_headless.exit()
