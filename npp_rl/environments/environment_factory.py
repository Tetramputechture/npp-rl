"""
Environment factory functions for creating NPP environments with various configurations.

This module provides factory functions for creating NPP environments with
reachability features, hierarchical wrappers, and completion planning integration.
"""

from typing import Optional, Dict, Any
from nclone.gym_environment.npp_environment import NppEnvironment
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from npp_rl.hrl.completion_controller import CompletionController


def create_reachability_aware_env(
    render_mode: str = "rgb_array",
    level_set: str = "intro",
    max_episode_steps: int = 2000,
    enable_reachability: bool = True,
    enable_monitoring: bool = True,
    debug: bool = False,
    **kwargs
) -> NppEnvironment:
    """
    Create an NPP environment with reachability features enabled.
    
    This function creates the standard NPP environment used in the current
    training pipeline with reachability analysis integration.
    
    Args:
        render_mode: Rendering mode ('rgb_array' or 'human')
        level_set: Level set to use ('intro', 'advanced', etc.)
        max_episode_steps: Maximum steps per episode
        enable_reachability: Whether to enable reachability features
        enable_monitoring: Whether to enable performance monitoring
        debug: Whether to enable debug mode
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Configured NPP environment
    """
    env = NppEnvironment(
        render_mode=render_mode,
        level_set=level_set,
        max_episode_steps=max_episode_steps,
        enable_reachability=enable_reachability,
        **kwargs
    )
    
    # Add intrinsic reward wrapper if monitoring is enabled
    if enable_monitoring:
        env = IntrinsicRewardWrapper(env)
    
    return env


def create_hierarchical_env(
    render_mode: str = "rgb_array",
    level_set: str = "intro", 
    max_episode_steps: int = 2000,
    completion_controller: Optional[CompletionController] = None,
    enable_subtask_rewards: bool = True,
    subtask_reward_scale: float = 0.1,
    **kwargs
) -> "HierarchicalNppWrapper":
    """
    Create an NPP environment with hierarchical wrapper for completion-focused training.
    
    This function creates an NPP environment wrapped with hierarchical functionality
    including completion controller integration and subtask-specific reward shaping.
    
    Args:
        render_mode: Rendering mode ('rgb_array' or 'human')
        level_set: Level set to use ('intro', 'advanced', etc.)
        max_episode_steps: Maximum steps per episode
        completion_controller: Optional completion controller instance
        enable_subtask_rewards: Whether to enable subtask-specific reward shaping
        subtask_reward_scale: Scaling factor for subtask rewards
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Hierarchical NPP environment wrapper
    """
    # Create base environment
    base_env = create_reachability_aware_env(
        render_mode=render_mode,
        level_set=level_set,
        max_episode_steps=max_episode_steps,
        **kwargs
    )
    
    # Wrap with hierarchical functionality
    hierarchical_env = HierarchicalNppWrapper(
        base_env,
        completion_controller=completion_controller,
        enable_subtask_rewards=enable_subtask_rewards,
        subtask_reward_scale=subtask_reward_scale,
    )
    
    return hierarchical_env


class HierarchicalNppWrapper:
    """
    Wrapper for NPP environment that adds hierarchical functionality.
    
    This wrapper integrates the completion controller with the environment,
    providing subtask-specific reward shaping and state augmentation for
    hierarchical RL training.
    """
    
    def __init__(
        self,
        env: NppEnvironment,
        completion_controller: Optional[CompletionController] = None,
        enable_subtask_rewards: bool = True,
        subtask_reward_scale: float = 0.1,
    ):
        """
        Initialize hierarchical wrapper.
        
        Args:
            env: Base NPP environment
            completion_controller: Completion controller for subtask management
            enable_subtask_rewards: Whether to add subtask-specific rewards
            subtask_reward_scale: Scaling factor for subtask rewards
        """
        self.env = env
        self.completion_controller = completion_controller or CompletionController()
        self.enable_subtask_rewards = enable_subtask_rewards
        self.subtask_reward_scale = subtask_reward_scale
        
        # Track subtask progress for reward shaping
        self.last_subtask = None
        self.subtask_step_count = 0
        
        # Expose environment attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata
    
    def reset(self, **kwargs):
        """Reset environment and hierarchical state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset completion controller
        self.completion_controller.reset()
        self.last_subtask = None
        self.subtask_step_count = 0
        
        # Update subtask based on initial state
        current_subtask = self.completion_controller.get_current_subtask(
            self._obs_to_dict(obs), info
        )
        self.last_subtask = current_subtask
        
        # Augment info with hierarchical information
        info['hierarchical'] = {
            'current_subtask': current_subtask.name,
            'subtask_features': self.completion_controller.get_subtask_features(),
            'subtask_metrics': self.completion_controller.get_subtask_metrics(),
        }
        
        return obs, info
    
    def step(self, action):
        """Step environment with hierarchical reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update completion controller
        obs_dict = self._obs_to_dict(obs)
        current_subtask = self.completion_controller.get_current_subtask(obs_dict, info)
        self.completion_controller.step(obs_dict, info)
        
        # Add subtask-specific reward shaping
        if self.enable_subtask_rewards:
            subtask_reward = self._calculate_subtask_reward(
                current_subtask, obs_dict, info, terminated
            )
            reward += subtask_reward * self.subtask_reward_scale
        
        # Track subtask transitions
        if self.last_subtask != current_subtask:
            info['subtask_transition'] = {
                'from': self.last_subtask.name if self.last_subtask else None,
                'to': current_subtask.name,
                'step_count': self.subtask_step_count
            }
            self.subtask_step_count = 0
        else:
            self.subtask_step_count += 1
        
        self.last_subtask = current_subtask
        
        # Augment info with hierarchical information
        info['hierarchical'] = {
            'current_subtask': current_subtask.name,
            'subtask_features': self.completion_controller.get_subtask_features(),
            'subtask_metrics': self.completion_controller.get_subtask_metrics(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def _obs_to_dict(self, obs) -> Dict[str, Any]:
        """Convert observation to dictionary format for completion controller."""
        # This would convert the observation to the format expected by the controller
        # For now, return a simple wrapper
        return {'observation': obs}
    
    def _calculate_subtask_reward(
        self, 
        current_subtask, 
        obs_dict: Dict[str, Any], 
        info: Dict[str, Any], 
        terminated: bool
    ) -> float:
        """
        Calculate subtask-specific reward shaping.
        
        Args:
            current_subtask: Current subtask enum
            obs_dict: Observation dictionary
            info: Environment info
            terminated: Whether episode terminated
            
        Returns:
            Subtask-specific reward
        """
        from npp_rl.hrl.completion_controller import Subtask
        
        reward = 0.0
        
        # Reward based on subtask progress
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Reward getting closer to exit switch
            if 'switch_distance' in info:
                # Negative distance as reward (closer = higher reward)
                reward += -info['switch_distance'] * 0.01
                
        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Reward getting closer to locked door switches
            if 'locked_door_distance' in info:
                reward += -info['locked_door_distance'] * 0.01
                
        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Reward getting closer to exit door
            if 'exit_distance' in info:
                reward += -info['exit_distance'] * 0.01
                
        elif current_subtask == Subtask.AVOID_MINE:
            # Reward staying away from mines
            if 'mine_distance' in info:
                reward += info['mine_distance'] * 0.005  # Positive distance reward
        
        # Bonus for subtask completion
        if 'subtask_transition' in info:
            reward += 0.5  # Bonus for successful subtask transition
        
        # Penalty for taking too long on a subtask
        if self.subtask_step_count > 500:  # 500 steps without progress
            reward -= 0.1
        
        return reward
    
    def render(self, *args, **kwargs):
        """Render the environment."""
        return self.env.render(*args, **kwargs)
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def __getattr__(self, name):
        """Delegate attribute access to base environment."""
        return getattr(self.env, name)