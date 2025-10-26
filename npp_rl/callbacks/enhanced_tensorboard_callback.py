"""Enhanced TensorBoard callback for comprehensive training metrics.

This callback provides detailed logging of training metrics, including:
- Episode statistics (rewards, lengths, success rates)
- Agent behavior metrics (action distributions, value estimates)
- Learning progress indicators (loss components, gradient norms)
- Environment interaction statistics
- Performance metrics (FPS, rollout time)
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

logger = logging.getLogger(__name__)


class EnhancedTensorBoardCallback(BaseCallback):
    """Enhanced TensorBoard callback with comprehensive metrics logging.
    
    This callback logs detailed metrics to TensorBoard including:
    - Episode statistics (rewards, lengths, success rates, completion times)
    - Action distribution histograms
    - Value function estimates and advantages
    - Policy and value loss components
    - Learning rate schedule
    - Gradient norms and clip ratios
    - Environment statistics
    - FPS and performance metrics
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        histogram_freq: int = 1000,
        verbose: int = 0,
        log_gradients: bool = True,
        log_weights: bool = False,
    ):
        """Initialize enhanced TensorBoard callback.
        
        Args:
            log_freq: Frequency (in steps) to log scalar metrics
            histogram_freq: Frequency (in steps) to log histograms (more expensive)
            verbose: Verbosity level
            log_gradients: Whether to log gradient norms
            log_weights: Whether to log model weight histograms (very expensive)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.histogram_freq = histogram_freq
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        
        # TensorBoard writer
        self.tb_writer = None
        
        # Tracking buffers
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episode_completion_times = deque(maxlen=100)
        
        # Reward component tracking (for intrinsic/hierarchical rewards)
        self.episode_intrinsic_rewards = deque(maxlen=100)
        self.episode_extrinsic_rewards = deque(maxlen=100)
        self.episode_hierarchical_rewards = deque(maxlen=100)
        
        # PBRS reward component tracking (step-level)
        self.pbrs_navigation_rewards = deque(maxlen=1000)
        self.pbrs_exploration_rewards = deque(maxlen=1000)
        self.pbrs_shaping_rewards = deque(maxlen=1000)
        self.pbrs_total_rewards = deque(maxlen=1000)
        
        # PBRS potential tracking
        self.pbrs_objective_potentials = deque(maxlen=1000)
        self.pbrs_hazard_potentials = deque(maxlen=1000)
        self.pbrs_impact_potentials = deque(maxlen=1000)
        self.pbrs_exploration_potentials = deque(maxlen=1000)
        
        # Action tracking
        self.action_counts = defaultdict(int)
        self.total_actions = 0
        
        # Action name mapping for N++ (indices must match environment action space)
        self.action_names = {
            0: "NOOP",
            1: "Left",
            2: "Right", 
            3: "Jump",
            4: "Jump+Left",
            5: "Jump+Right"
        }
        
        # Action transition tracking (for behavior analysis)
        self.last_actions = None  # Track previous actions per environment
        self.action_transitions = defaultdict(lambda: defaultdict(int))  # [prev_action][next_action] = count
        
        # Value function tracking
        self.value_estimates = deque(maxlen=1000)
        self.advantages = deque(maxlen=1000)
        
        # Performance tracking
        self.rollout_times = deque(maxlen=10)
        self.update_times = deque(maxlen=10)
        self.fps_history = deque(maxlen=100)
        
        # Training progress
        self.last_log_step = 0
        self.last_histogram_step = 0
        self.start_time = None
        
        # Episode tracking for detailed metrics
        self.current_episode_data = defaultdict(list)
        
    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        # Find TensorBoard writer
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break
        
        if self.tb_writer is None:
            logger.warning("TensorBoard writer not found - enhanced logging disabled")
            return
            
        self.start_time = time.time()
        logger.info("Enhanced TensorBoard callback initialized")
        
    def _on_step(self) -> bool:
        """Called after each environment step.
        
        Returns:
            bool: If False, training will be stopped
        """
        if self.tb_writer is None:
            return True
            
        # Track episode completions and step-level PBRS components
        if 'dones' in self.locals:
            dones = self.locals['dones']
            if 'infos' in self.locals:
                infos = self.locals['infos']
                for i, (done, info) in enumerate(zip(dones, infos)):
                    # Track PBRS components at every step
                    self._track_pbrs_components(info)
                    
                    # Process episode end metrics
                    if done:
                        self._process_episode_end(info)
        
        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            
            # Initialize last_actions on first step
            if self.last_actions is None:
                self.last_actions = np.zeros(len(actions), dtype=int)
            
            # Track action counts and transitions
            for env_idx, action in enumerate(actions):
                action_idx = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
                self.action_counts[action_idx] += 1
                self.total_actions += 1
                
                # Track action transitions
                prev_action = int(self.last_actions[env_idx])
                self.action_transitions[prev_action][action_idx] += 1
                self.last_actions[env_idx] = action_idx
        
        # Track value estimates from rollout buffer (more reliable than trying to access obs_tensor)
        # Note: This captures values after they've been computed during rollout collection
        if hasattr(self.model, 'rollout_buffer'):
            try:
                # Only access if buffer has data
                if hasattr(self.model.rollout_buffer, 'values') and hasattr(self.model.rollout_buffer, 'pos'):
                    buffer_pos = self.model.rollout_buffer.pos
                    if buffer_pos > 0:  # Buffer has new data
                        # Get recently added values (last position)
                        recent_idx = max(0, buffer_pos - 1)
                        values = self.model.rollout_buffer.values[recent_idx]
                        if values is not None and len(values) > 0:
                            self.value_estimates.extend(values.flatten().tolist())
            except Exception as e:
                logger.debug(f"Could not track value estimates from rollout buffer: {e}")
        
        # Log scalar metrics at regular intervals
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_scalar_metrics()
            self.last_log_step = self.num_timesteps
        
        # Log histograms at regular intervals
        if self.num_timesteps - self.last_histogram_step >= self.histogram_freq:
            self._log_histogram_metrics()
            self.last_histogram_step = self.num_timesteps
            
        return True
    
    def _track_pbrs_components(self, info: Dict[str, Any]) -> None:
        """Track PBRS reward components from step info.
        
        Args:
            info: Step info dictionary that may contain PBRS components
        """
        if 'pbrs_components' not in info:
            return
            
        pbrs_data = info['pbrs_components']
        
        # Track reward components
        if 'navigation_reward' in pbrs_data:
            self.pbrs_navigation_rewards.append(float(pbrs_data['navigation_reward']))
        if 'exploration_reward' in pbrs_data:
            self.pbrs_exploration_rewards.append(float(pbrs_data['exploration_reward']))
        if 'pbrs_reward' in pbrs_data:
            self.pbrs_shaping_rewards.append(float(pbrs_data['pbrs_reward']))
        if 'total_reward' in pbrs_data:
            self.pbrs_total_rewards.append(float(pbrs_data['total_reward']))
        
        # Track potential components
        if 'pbrs_components' in pbrs_data:
            potentials = pbrs_data['pbrs_components']
            if isinstance(potentials, dict):
                if 'objective' in potentials:
                    self.pbrs_objective_potentials.append(float(potentials['objective']))
                if 'hazard' in potentials:
                    self.pbrs_hazard_potentials.append(float(potentials['hazard']))
                if 'impact' in potentials:
                    self.pbrs_impact_potentials.append(float(potentials['impact']))
                if 'exploration' in potentials:
                    self.pbrs_exploration_potentials.append(float(potentials['exploration']))
    
    def _process_episode_end(self, info: Dict[str, Any]) -> None:
        """Process episode completion and extract metrics.
        
        Args:
            info: Episode info dictionary
        """
        # Standard episode metrics
        if 'episode' in info:
            episode_info = info['episode']
            self.episode_rewards.append(episode_info['r'])
            self.episode_lengths.append(episode_info['l'])
        
        # Intrinsic reward tracking (from IntrinsicRewardWrapper)
        if 'r_int_episode' in info:
            self.episode_intrinsic_rewards.append(info['r_int_episode'])
        if 'r_ext_episode' in info:
            self.episode_extrinsic_rewards.append(info['r_ext_episode'])
        
        # Hierarchical reward tracking (from HierarchicalRewardWrapper)
        if 'hierarchical_reward_episode' in info:
            self.episode_hierarchical_rewards.append(info['hierarchical_reward_episode'])
        
        # Success/failure tracking
        if 'success' in info:
            self.episode_successes.append(float(info['success']))
        elif 'is_success' in info:
            self.episode_successes.append(float(info['is_success']))
        elif 'episode' in info and 'r' in info['episode']:
            # Infer success from reward (N++ gives 1.0 for completion)
            self.episode_successes.append(float(info['episode']['r'] > 0.9))
        
        # Completion time tracking
        if 'completion_time' in info:
            self.episode_completion_times.append(info['completion_time'])
        elif 'episode' in info and 'l' in info['episode']:
            # Use episode length as proxy
            self.episode_completion_times.append(info['episode']['l'])
    
    def _log_scalar_metrics(self) -> None:
        """Log scalar metrics to TensorBoard."""
        step = self.num_timesteps
        
        # Episode statistics
        if self.episode_rewards:
            self.tb_writer.add_scalar('episode/reward_mean', np.mean(self.episode_rewards), step)
            self.tb_writer.add_scalar('episode/reward_std', np.std(self.episode_rewards), step)
            self.tb_writer.add_scalar('episode/reward_max', np.max(self.episode_rewards), step)
            self.tb_writer.add_scalar('episode/reward_min', np.min(self.episode_rewards), step)
        
        if self.episode_lengths:
            self.tb_writer.add_scalar('episode/length_mean', np.mean(self.episode_lengths), step)
            self.tb_writer.add_scalar('episode/length_std', np.std(self.episode_lengths), step)
        
        if self.episode_successes:
            success_rate = np.mean(self.episode_successes)
            self.tb_writer.add_scalar('episode/success_rate', success_rate, step)
            self.tb_writer.add_scalar('episode/failure_rate', 1.0 - success_rate, step)
        
        if self.episode_completion_times:
            self.tb_writer.add_scalar('episode/completion_time_mean', 
                                     np.mean(self.episode_completion_times), step)
        
        # Reward component statistics
        if self.episode_intrinsic_rewards:
            self.tb_writer.add_scalar('rewards/intrinsic_mean', 
                                     np.mean(self.episode_intrinsic_rewards), step)
            self.tb_writer.add_scalar('rewards/intrinsic_std', 
                                     np.std(self.episode_intrinsic_rewards), step)
        
        if self.episode_extrinsic_rewards:
            self.tb_writer.add_scalar('rewards/extrinsic_mean', 
                                     np.mean(self.episode_extrinsic_rewards), step)
            self.tb_writer.add_scalar('rewards/extrinsic_std', 
                                     np.std(self.episode_extrinsic_rewards), step)
        
        if self.episode_hierarchical_rewards:
            self.tb_writer.add_scalar('rewards/hierarchical_mean', 
                                     np.mean(self.episode_hierarchical_rewards), step)
            self.tb_writer.add_scalar('rewards/hierarchical_std', 
                                     np.std(self.episode_hierarchical_rewards), step)
        
        # Reward component ratios for analysis
        if self.episode_intrinsic_rewards and self.episode_extrinsic_rewards:
            int_rewards = np.array(self.episode_intrinsic_rewards)
            ext_rewards = np.array(self.episode_extrinsic_rewards)
            # Calculate ratio safely avoiding division by zero
            total_abs = np.abs(int_rewards) + np.abs(ext_rewards)
            valid_indices = total_abs > 1e-6
            if valid_indices.any():
                ratio = np.abs(int_rewards[valid_indices]) / total_abs[valid_indices]
                self.tb_writer.add_scalar('rewards/intrinsic_ratio', 
                                         np.mean(ratio), step)
        
        # PBRS reward component statistics (step-level aggregation)
        if self.pbrs_navigation_rewards:
            self.tb_writer.add_scalar('pbrs_rewards/navigation_mean', 
                                     np.mean(self.pbrs_navigation_rewards), step)
            self.tb_writer.add_scalar('pbrs_rewards/navigation_std', 
                                     np.std(self.pbrs_navigation_rewards), step)
        
        if self.pbrs_exploration_rewards:
            self.tb_writer.add_scalar('pbrs_rewards/exploration_mean', 
                                     np.mean(self.pbrs_exploration_rewards), step)
            self.tb_writer.add_scalar('pbrs_rewards/exploration_std', 
                                     np.std(self.pbrs_exploration_rewards), step)
        
        if self.pbrs_shaping_rewards:
            self.tb_writer.add_scalar('pbrs_rewards/pbrs_mean', 
                                     np.mean(self.pbrs_shaping_rewards), step)
            self.tb_writer.add_scalar('pbrs_rewards/pbrs_std', 
                                     np.std(self.pbrs_shaping_rewards), step)
            # Log min/max for debugging
            self.tb_writer.add_scalar('pbrs_rewards/pbrs_min', 
                                     np.min(self.pbrs_shaping_rewards), step)
            self.tb_writer.add_scalar('pbrs_rewards/pbrs_max', 
                                     np.max(self.pbrs_shaping_rewards), step)
        
        if self.pbrs_total_rewards:
            self.tb_writer.add_scalar('pbrs_rewards/total_mean', 
                                     np.mean(self.pbrs_total_rewards), step)
            self.tb_writer.add_scalar('pbrs_rewards/total_std', 
                                     np.std(self.pbrs_total_rewards), step)
        
        # PBRS potential statistics
        if self.pbrs_objective_potentials:
            self.tb_writer.add_scalar('pbrs_potentials/objective_mean', 
                                     np.mean(self.pbrs_objective_potentials), step)
            self.tb_writer.add_scalar('pbrs_potentials/objective_std', 
                                     np.std(self.pbrs_objective_potentials), step)
        
        if self.pbrs_hazard_potentials:
            self.tb_writer.add_scalar('pbrs_potentials/hazard_mean', 
                                     np.mean(self.pbrs_hazard_potentials), step)
        
        if self.pbrs_impact_potentials:
            self.tb_writer.add_scalar('pbrs_potentials/impact_mean', 
                                     np.mean(self.pbrs_impact_potentials), step)
        
        if self.pbrs_exploration_potentials:
            self.tb_writer.add_scalar('pbrs_potentials/exploration_mean', 
                                     np.mean(self.pbrs_exploration_potentials), step)
        
        # PBRS contribution analysis
        if self.pbrs_shaping_rewards and self.pbrs_total_rewards:
            pbrs_vals = np.array(self.pbrs_shaping_rewards)
            total_vals = np.array(self.pbrs_total_rewards)
            # Calculate PBRS contribution ratio
            if len(pbrs_vals) == len(total_vals):
                # Safe division avoiding division by zero
                valid_indices = np.abs(total_vals) > 1e-6
                if valid_indices.any():
                    contribution = np.abs(pbrs_vals[valid_indices]) / np.abs(total_vals[valid_indices])
                    self.tb_writer.add_scalar('pbrs_summary/pbrs_contribution_ratio', 
                                             np.mean(contribution), step)
        
        # Action distribution with descriptive names
        if self.total_actions > 0:
            # Get action space size dynamically
            n_actions = self.model.action_space.n if hasattr(self.model.action_space, 'n') else 6
            
            # Log individual action frequencies with descriptive names
            for action_idx in range(n_actions):
                count = self.action_counts.get(action_idx, 0)
                action_freq = count / self.total_actions
                action_name = self.action_names.get(action_idx, f"Action{action_idx}")
                self.tb_writer.add_scalar(f'actions/frequency/{action_name}', 
                                         action_freq, step)
            
            # Calculate action probabilities for entropy and analysis
            action_probs = np.array([self.action_counts.get(i, 0) for i in range(n_actions)]) / max(self.total_actions, 1)
            action_probs = action_probs + 1e-10  # Avoid log(0)
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            self.tb_writer.add_scalar('actions/entropy', action_entropy, step)
            
            # Movement-specific metrics
            # Horizontal movement: Left (1) + Right (2) + Jump+Left (4) + Jump+Right (5)
            left_actions = self.action_counts.get(1, 0) + self.action_counts.get(4, 0)
            right_actions = self.action_counts.get(2, 0) + self.action_counts.get(5, 0)
            horizontal_movement = left_actions + right_actions
            
            if horizontal_movement > 0:
                left_bias = left_actions / horizontal_movement
                right_bias = right_actions / horizontal_movement
                self.tb_writer.add_scalar('actions/movement/left_bias', left_bias, step)
                self.tb_writer.add_scalar('actions/movement/right_bias', right_bias, step)
            
            # Calculate movement vs stationary time
            noop_freq = action_probs[0]
            movement_freq = 1.0 - noop_freq
            self.tb_writer.add_scalar('actions/movement/stationary_pct', noop_freq, step)
            self.tb_writer.add_scalar('actions/movement/active_pct', movement_freq, step)
            
            # Jump analysis
            # Jump actions: Jump (3) + Jump+Left (4) + Jump+Right (5)
            jump_only = self.action_counts.get(3, 0)
            jump_left = self.action_counts.get(4, 0)
            jump_right = self.action_counts.get(5, 0)
            total_jumps = jump_only + jump_left + jump_right
            
            if total_jumps > 0:
                # How often jumps are combined with directional movement
                directional_jump_pct = (jump_left + jump_right) / total_jumps
                self.tb_writer.add_scalar('actions/jump/directional_pct', directional_jump_pct, step)
                self.tb_writer.add_scalar('actions/jump/vertical_only_pct', jump_only / total_jumps, step)
            
            # Overall jump frequency (useful for understanding agent behavior)
            jump_freq = total_jumps / self.total_actions
            self.tb_writer.add_scalar('actions/jump/frequency', jump_freq, step)
        
        # Value function statistics
        if self.value_estimates:
            self.tb_writer.add_scalar('value/estimate_mean', np.mean(self.value_estimates), step)
            self.tb_writer.add_scalar('value/estimate_std', np.std(self.value_estimates), step)
            self.tb_writer.add_scalar('value/estimate_max', np.max(self.value_estimates), step)
            self.tb_writer.add_scalar('value/estimate_min', np.min(self.value_estimates), step)
        
        # Learning progress from model logger
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            log_data = self.model.logger.name_to_value
            
            # Policy loss
            if 'train/policy_loss' in log_data:
                self.tb_writer.add_scalar('loss/policy', log_data['train/policy_loss'], step)
            
            # Value loss
            if 'train/value_loss' in log_data:
                self.tb_writer.add_scalar('loss/value', log_data['train/value_loss'], step)
            
            # Entropy loss
            if 'train/entropy_loss' in log_data:
                self.tb_writer.add_scalar('loss/entropy', log_data['train/entropy_loss'], step)
            
            # Total loss
            if 'train/loss' in log_data:
                self.tb_writer.add_scalar('loss/total', log_data['train/loss'], step)
            
            # Clip fraction (important for PPO stability)
            if 'train/clip_fraction' in log_data:
                self.tb_writer.add_scalar('training/clip_fraction', log_data['train/clip_fraction'], step)
            
            # Explained variance (value function quality)
            if 'train/explained_variance' in log_data:
                self.tb_writer.add_scalar('training/explained_variance', 
                                         log_data['train/explained_variance'], step)
            
            # Learning rate
            if 'train/learning_rate' in log_data:
                self.tb_writer.add_scalar('training/learning_rate', 
                                         log_data['train/learning_rate'], step)
            
            # Approximate KL divergence
            if 'train/approx_kl' in log_data:
                self.tb_writer.add_scalar('training/approx_kl', log_data['train/approx_kl'], step)
        
        # Performance metrics
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.tb_writer.add_scalar('performance/elapsed_time_minutes', 
                                     elapsed_time / 60, step)
            
            # Steps per second
            sps = step / elapsed_time if elapsed_time > 0 else 0
            self.tb_writer.add_scalar('performance/steps_per_second', sps, step)
        
        # FPS from rollout
        if hasattr(self.model, '_last_obs') and hasattr(self.model, 'num_timesteps'):
            try:
                if hasattr(self, '_last_time'):
                    current_time = time.time()
                    time_delta = current_time - self._last_time
                    if time_delta > 0:
                        steps_delta = self.num_timesteps - getattr(self, '_last_timesteps', 0)
                        fps = steps_delta / time_delta
                        self.fps_history.append(fps)
                        self.tb_writer.add_scalar('performance/fps_instant', fps, step)
                        if len(self.fps_history) > 1:
                            self.tb_writer.add_scalar('performance/fps_mean', 
                                                     np.mean(self.fps_history), step)
                
                self._last_time = time.time()
                self._last_timesteps = self.num_timesteps
            except Exception as e:
                logger.debug(f"Could not compute FPS: {e}")
        
        self.tb_writer.flush()
    
    def _log_histogram_metrics(self) -> None:
        """Log histogram metrics to TensorBoard (more expensive)."""
        step = self.num_timesteps
        
        # Episode reward distribution
        if self.episode_rewards:
            self.tb_writer.add_histogram('episode/reward_distribution', 
                                         np.array(self.episode_rewards), step)
        
        # Episode length distribution
        if self.episode_lengths:
            self.tb_writer.add_histogram('episode/length_distribution',
                                         np.array(self.episode_lengths), step)
        
        # Value estimates distribution
        if self.value_estimates:
            self.tb_writer.add_histogram('value/estimate_distribution',
                                         np.array(self.value_estimates), step)
        
        # Action distribution histogram
        if self.total_actions > 0:
            n_actions = self.model.action_space.n if hasattr(self.model.action_space, 'n') else 6
            action_dist = np.array([self.action_counts.get(i, 0) for i in range(n_actions)])
            self.tb_writer.add_histogram('actions/distribution', action_dist, step)
            
            # Action transition matrix visualization
            # This shows patterns like "after moving left, what action is most common?"
            if self.action_transitions:
                # Create transition matrix
                transition_matrix = np.zeros((n_actions, n_actions))
                for prev_action in range(n_actions):
                    total_from_prev = sum(self.action_transitions[prev_action].values())
                    if total_from_prev > 0:
                        for next_action in range(n_actions):
                            count = self.action_transitions[prev_action][next_action]
                            transition_matrix[prev_action, next_action] = count / total_from_prev
                
                # Log most common transitions as scalars (easier to track)
                for prev_action in range(n_actions):
                    for next_action in range(n_actions):
                        prob = transition_matrix[prev_action, next_action]
                        if prob > 0.01:  # Only log significant transitions
                            prev_name = self.action_names.get(prev_action, f"A{prev_action}")
                            next_name = self.action_names.get(next_action, f"A{next_action}")
                            self.tb_writer.add_scalar(
                                f'actions/transitions/{prev_name}_to_{next_name}',
                                prob, step
                            )
        
        # Gradient norms (if enabled)
        if self.log_gradients and hasattr(self.model, 'policy'):
            try:
                total_norm = 0.0
                for name, param in self.model.policy.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        
                        # Log individual layer gradient norms (sample)
                        if 'features_extractor' in name or 'mlp' in name:
                            self.tb_writer.add_scalar(f'gradients/{name}_norm', 
                                                     param_norm.item(), step)
                
                total_norm = total_norm ** 0.5
                self.tb_writer.add_scalar('gradients/total_norm', total_norm, step)
            except Exception as e:
                logger.debug(f"Could not log gradient norms: {e}")
        
        # Model weight distributions (if enabled, very expensive)
        if self.log_weights and hasattr(self.model, 'policy'):
            try:
                for name, param in self.model.policy.named_parameters():
                    if 'weight' in name:
                        self.tb_writer.add_histogram(f'weights/{name}', 
                                                     param.data.cpu().numpy(), step)
            except Exception as e:
                logger.debug(f"Could not log weight histograms: {e}")
        
        self.tb_writer.flush()
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        # Track rollout time
        if hasattr(self, '_rollout_start_time'):
            rollout_time = time.time() - self._rollout_start_time
            self.rollout_times.append(rollout_time)
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('performance/rollout_time_seconds', 
                                         rollout_time, self.num_timesteps)
                if len(self.rollout_times) > 1:
                    self.tb_writer.add_scalar('performance/rollout_time_mean',
                                             np.mean(self.rollout_times), 
                                             self.num_timesteps)
    
    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout."""
        self._rollout_start_time = time.time()
