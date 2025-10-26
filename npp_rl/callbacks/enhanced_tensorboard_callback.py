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
        
        # Action tracking
        self.action_counts = defaultdict(int)
        self.total_actions = 0
        
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
            
        # Track episode completions
        if 'dones' in self.locals:
            dones = self.locals['dones']
            if 'infos' in self.locals:
                infos = self.locals['infos']
                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        self._process_episode_end(info)
        
        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            for action in actions:
                action_idx = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
                self.action_counts[action_idx] += 1
                self.total_actions += 1
        
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
        
        # Action distribution
        if self.total_actions > 0:
            for action_idx, count in self.action_counts.items():
                action_freq = count / self.total_actions
                self.tb_writer.add_scalar(f'actions/frequency_action_{action_idx}', 
                                         action_freq, step)
            
            # Action entropy (measure of policy exploration)
            # Get action space size dynamically
            n_actions = self.model.action_space.n if hasattr(self.model.action_space, 'n') else 6
            action_probs = np.array([self.action_counts.get(i, 0) for i in range(n_actions)]) / max(self.total_actions, 1)
            action_probs = action_probs + 1e-10  # Avoid log(0)
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            self.tb_writer.add_scalar('actions/entropy', action_entropy, step)
        
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
            action_dist = np.array([self.action_counts.get(i, 0) for i in range(6)])
            self.tb_writer.add_histogram('actions/distribution', action_dist, step)
        
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
