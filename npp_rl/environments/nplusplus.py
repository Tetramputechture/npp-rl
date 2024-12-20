import gymnasium
from gymnasium.spaces import discrete, box
import numpy as np
from npp_rl.environments.reward_calculation.main_reward_calculator import RewardCalculator
from npp_rl.environments.reward_calculation.planning_reward_calculator import PlanningRewardCalculator
from npp_rl.environments.planning.path_planner import PathPlanner
from npp_rl.environments.planning.waypoint_manager import WaypointManager
import time
from typing import Tuple, Dict, Any
import os
from npp_rl.environments.observation_processor import ObservationProcessor
from npp_rl.environments.movement_evaluator import MovementEvaluator
from nplay_headless import NPlayHeadless
import uuid
import random
from npp_rl.environments.constants import (
    OBSERVATION_IMAGE_SIZE,
    NUM_TEMPORAL_FRAMES,
    NUM_PLAYER_STATE_CHANNELS
)
from npp_rl.environments.visualization.path_visualizer import PathVisualizer
import imageio

MAP_DATA_PATH = "../nclone/maps/map_data_simple"


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
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode: str = 'rgb_array'):
        """Initialize the environment."""
        super().__init__()

        self.nplay_headless = NPlayHeadless(render_mode=render_mode)
        self.nplay_headless.load_random_map(seed=42)

        self.render_mode = render_mode

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

        # Initialize observation processing
        self.observation_processor = ObservationProcessor()

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize observation space as a box for visual features
        # self.observation_space = box.Box(
        #     low=0,
        #     high=255,
        #     shape=(
        #         OBSERVATION_IMAGE_SIZE,
        #         OBSERVATION_IMAGE_SIZE,
        #         NUM_TEMPORAL_FRAMES
        #     ),
        #     dtype=np.uint8
        # )

        # Initialize observation space as a Dict space with all features
        self.observation_space = gymnasium.spaces.Dict({
            # Frame stack of 4 grayscale images
            'visual': box.Box(
                low=0,
                high=255,
                shape=(
                    OBSERVATION_IMAGE_SIZE,
                    OBSERVATION_IMAGE_SIZE,
                    NUM_TEMPORAL_FRAMES
                ),
                dtype=np.uint8
            ),
            'player_state': box.Box(
                low=0,
                high=1,
                # pos_x, pos_y, vel_x, vel_y, in_air, walled
                shape=(NUM_PLAYER_STATE_CHANNELS,),
                dtype=np.float32
            ),
            'goal_features': box.Box(
                low=0,
                high=1,
                shape=(3,),  # switch_dist, exit_dist, switch_activated
                dtype=np.float32
            ),
        })

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

        # Initialize current episode reward
        self.current_episode_reward = 0.0

        # Initialize path visualizer
        self.path_visualizer = PathVisualizer()
        self.current_path = []
        self.episode_step_counter = 0

        # Initialize frame buffer for path visualization
        self.path_viz_buffer = []

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        obs = {
            'screen': self.nplay_headless.render(),
            'player_x': self.nplay_headless.ninja_position()[0],
            'player_y': self.nplay_headless.ninja_position()[1],
            'player_vx': self.nplay_headless.ninja_velocity()[0],
            'player_vy': self.nplay_headless.ninja_velocity()[1],
            'player_dead': self.nplay_headless.ninja_has_died(),
            'player_won': self.nplay_headless.ninja_has_won(),
            'switch_activated': self.nplay_headless.exit_switch_activated(),
            'switch_x': self.nplay_headless.exit_switch_position()[0],
            'switch_y': self.nplay_headless.exit_switch_position()[1],
            'exit_door_x': self.nplay_headless.exit_door_position()[0],
            'exit_door_y': self.nplay_headless.exit_door_position()[1],
            'in_air': self.nplay_headless.ninja_is_in_air(),
            'walled': self.nplay_headless.ninja_is_walled(),
            'level_width': 1032.0,
            'level_height': 576.0,
            # Add collision and pathfinding data
            'tile_data': self.nplay_headless.get_tile_data(),
            'segment_data': self.nplay_headless.get_segment_data(),
            'grid_edges': self.nplay_headless.get_grid_edges(),
            'segment_edges': self.nplay_headless.get_segment_edges(),
            'dynamic_objects': self.nplay_headless.get_dynamic_objects() if hasattr(self.nplay_headless, 'get_dynamic_objects') else []
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
        # Truncation is when the current simulation frame is greater than 5000
        truncated = self.nplay_headless.sim.frame > 5000

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

    def _update_planning(self, observation: Dict[str, Any]):
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

    def _save_path_visualization(self, observation: Dict[str, Any], prev_obs: Dict[str, Any]):
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

        player_x, player_y = self.nplay_headless.ninja_position()

        self.position_log_file_string += f'{player_x},{player_y}\n'
        self.action_log_file_string += f'{self._action_to_string(action)}\n'

        # Randomly choose number of frames to repeat action (2-4)
        num_repeat_frames = random.randint(2, 4)

        # Execute action for the chosen number of frames
        action_hoz, action_jump = self._actions_to_execute(action)

        terminated = False
        truncated = False

        # Execute the same action for num_repeat_frames
        for _ in range(num_repeat_frames):
            self.nplay_headless.tick(action_hoz, action_jump)

            # Check termination after each frame
            curr_obs = self._get_observation()
            terminated, truncated = self._check_termination(curr_obs)

            # Break early if episode ended
            if terminated or truncated:
                break

        # Get final observation after all repeated frames
        observation = self._get_observation()

        self.episode_step_counter += 1

        # Process observation and calculate rewards
        processed_obs = self.observation_processor.process_observation(
            observation)

        # Calculate combined reward
        movement_reward = self.reward_calculator.calculate_reward(
            observation, prev_obs, action)

        reward = movement_reward

        self.current_episode_reward += reward

        # Print reward if terminated or truncated
        if terminated or truncated:
            print(f"Episode reward: {self.current_episode_reward}")

        return processed_obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Save video from previous episode if buffer is not empty
        self._save_path_video()

        # Increment episode counter
        self.episode_counter += 1

        # Reset current episode reward
        self.current_episode_reward = 0.0

        # Reset reward calculator and movement evaluator
        self.reward_calculator.reset()
        self.movement_evaluator.reset()
        # self.reward_calculator.update_progression_metrics()

        # Reset observation processor
        self.observation_processor.reset()

        # Reset position and action logs
        self.position_log_file_string = 'PlayerX,PlayerY\n'
        self.action_log_file_string = 'Action\n'

        # reset level
        self.nplay_headless.reset()

        # load random map
        self.nplay_headless.load_random_map()

        # Reset planning components
        self.path_planner = PathPlanner()
        self.waypoint_manager = WaypointManager(
            path_planner=self.path_planner)
        self.planning_reward_calculator.reset(
            waypoint_manager=self.waypoint_manager)

        # Reset episode step counter
        self.episode_step_counter = 0
        self.current_path = []

        # Get initial observation
        initial_obs = self._get_observation()

        # Update initial planning
        # self._update_planning(initial_obs)

        # Save initial planned path
        # self._save_path_visualization(initial_obs, initial_obs)

        # Get processed observation with planning features
        processed_obs = self.observation_processor.process_observation(
            initial_obs)
        # processed_obs['planning_features'] = self.planning_reward_calculator.get_planning_features()

        return processed_obs, {}

    def render(self):
        """Render the environment."""
        return self.nplay_headless.render()

    def close(self):
        """Clean up environment resources."""
        self.nplay_headless.exit()
