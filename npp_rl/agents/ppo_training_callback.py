from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, Optional
import torch
from collections import deque


class PPOTrainingCallback(BaseCallback):
    """
    Training callback with monitoring and adaptation mechanisms.

    This callback implements:
    1. Dynamic entropy adaptation through policy modification
    2. Comprehensive training metrics tracking
    3. Adaptive learning rate adjustment
    4. Performance-based model saving
    5. Detailed logging and monitoring
    """

    def __init__(self,
                 check_freq: int,
                 log_dir: Path,
                 verbose: int = 1,
                 save_freq: int = 10000,
                 min_ent_coef: float = 0.005,
                 max_ent_coef: float = 0.02,
                 moving_average_window: int = 100):
        """
        Initialize the callback with specified parameters.

        Args:
            check_freq: How often to check training metrics
            log_dir: Directory for saving logs and models
            verbose: Verbosity level
            save_freq: How often to save model checkpoints
            min_ent_coef: Minimum entropy coefficient
            max_ent_coef: Maximum entropy coefficient
            moving_average_window: Window size for calculating moving averages
        """
        super().__init__(verbose)

        # Basic configuration
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = str(log_dir / 'best_model')

        # Entropy coefficient bounds
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.current_ent_coef = max_ent_coef  # Start with maximum exploration

        # Performance tracking
        self.best_mean_reward = -np.inf
        self.episode_rewards = deque(maxlen=moving_average_window)
        self.episode_lengths = deque(maxlen=moving_average_window)
        self.moving_avg_rewards = deque(maxlen=moving_average_window)

        # Training stability metrics
        self.loss_values = deque(maxlen=moving_average_window)
        self.entropy_values = deque(maxlen=moving_average_window)
        self.value_loss_values = deque(maxlen=moving_average_window)
        self.policy_loss_values = deque(maxlen=moving_average_window)

        # Success tracking
        self.success_rate = deque(maxlen=moving_average_window)
        self.training_progress = 0.0

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training log
        self.training_log = {
            'rewards': [],
            'lengths': [],
            'entropy': [],
            'losses': [],
            'success_rates': []
        }

    def _init_callback(self) -> None:
        """Initialize callback and store original network parameters."""
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Store original network parameters
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'action_net'):
            # Store the original parameters of the action network
            self.original_action_weights = self.model.policy.action_net.weight.data.clone()
            if hasattr(self.model.policy.action_net, 'bias'):
                self.original_action_bias = self.model.policy.action_net.bias.data.clone()

    def _adjust_entropy_coefficient(self) -> None:
        """
        Dynamically adjusts the entropy coefficient based on comprehensive success metrics.

        In N++, success is hierarchical:
        1. Basic Movement: Precision, landing, and momentum control
        2. Navigation: Reaching and activating the switch
        3. Completion: Reaching the exit after switch activation

        The entropy adjustment considers all these levels to maintain appropriate
        exploration throughout the learning process.
        """
        print('Attempting to adjust entropy coefficient')

        # We need enough history to make informed adjustments
        if len(self.moving_avg_rewards) < 2 or len(self.loss_values) < 2:
            print('Not enough history to adjust entropy coefficient')
            return

        # Get the latest episode info for detailed success metrics
        if len(self.model.ep_info_buffer) > 0:
            last_info = self.model.ep_info_buffer[-1]

            # Extract all success components with defaults
            movement_success = last_info.get('movement_efficiency', 0.0)
            landing_success = last_info.get('landing_quality', 0.0)
            momentum_success = last_info.get('momentum_efficiency', 0.0)
            switch_progress = last_info.get('switch_activated', 0.0)
            objective_progress = last_info.get('objective_progress', 0.0)
            level_success = last_info.get('success', 0.0)

            # Calculate composite success rates
            movement_mastery = np.mean(
                [movement_success, landing_success, momentum_success])
            navigation_success = np.mean([switch_progress, objective_progress])

            # Calculate overall progress (weighted average)
            overall_progress = (0.4 * movement_mastery +
                                0.4 * navigation_success +
                                0.2 * level_success)

            # Update success rate with granular progress
            self.success_rate.append(overall_progress)
        else:
            movement_mastery = 0.0
            navigation_success = 0.0
            level_success = 0.0
            overall_progress = 0.0

        print(f'Movement Mastery: {movement_mastery:.3f}')
        print(f'Navigation Success: {navigation_success:.3f}')
        print(f'Level Success: {level_success:.3f}')
        print(f'Overall Progress: {overall_progress:.3f}')

        # Base entropy adjustment on overall progress
        if overall_progress < 0.3:
            # Early learning - keep high exploration
            target_entropy = self.max_ent_coef
        elif overall_progress < 0.6:
            # Mid learning - gradual reduction
            progress_ratio = (overall_progress - 0.3) / 0.3
            target_entropy = self.max_ent_coef - \
                (progress_ratio * (self.max_ent_coef - self.min_ent_coef) * 0.5)
        else:
            # Late learning - low exploration
            progress_ratio = (overall_progress - 0.6) / 0.4
            target_entropy = self.max_ent_coef - \
                (progress_ratio * (self.max_ent_coef - self.min_ent_coef) * 0.8)

        # Smooth the transition
        adjustment_rate = 0.1
        new_ent_coef = self.current_ent_coef * \
            (1 - adjustment_rate) + target_entropy * adjustment_rate

        # Ensure we stay within bounds
        new_ent_coef = np.clip(
            new_ent_coef, self.min_ent_coef, self.max_ent_coef)

        # Apply the new entropy coefficient if it's significantly different
        if abs(new_ent_coef - self.current_ent_coef) > 0.001:
            self._update_policy_entropy(new_ent_coef)

            # Log the adjustment if verbose
            if self.verbose > 0:
                print(f"\nAdjusting entropy coefficient:")
                print(f"Previous: {self.current_ent_coef:.4f}")
                print(f"New: {new_ent_coef:.4f}")
                print(f"Target: {target_entropy:.4f}")
                print(f"Progress metrics:")
                print(f"- Movement mastery: {movement_mastery:.3f}")
                print(f"- Navigation success: {navigation_success:.3f}")
                print(f"- Level success: {level_success:.3f}")
                print(f"- Overall progress: {overall_progress:.3f}")

    def _update_policy_entropy(self, new_ent_coef: float) -> None:
        """
        Update the policy's entropy by scaling the action network parameters.

        Args:
            new_ent_coef: New entropy coefficient to apply
        """
        if not hasattr(self.model, 'policy') or not hasattr(self.model.policy, 'action_net'):
            print(
                'WARNING: Policy does not have an action network. Cannot adjust entropy.')
            return

        if self.original_action_weights is None:
            # Store original parameters if not already stored
            self.original_action_weights = self.model.policy.action_net.weight.data.clone()
            if hasattr(self.model.policy.action_net, 'bias'):
                self.original_action_bias = self.model.policy.action_net.bias.data.clone()

        # Calculate scale factor based on entropy coefficient
        scale_factor = np.sqrt(new_ent_coef / self.max_ent_coef)

        with torch.no_grad():
            # Scale the weights based on original values
            self.model.policy.action_net.weight.data = self.original_action_weights * scale_factor

            # Scale the bias if it exists
            if hasattr(self.model.policy.action_net, 'bias') and self.original_action_bias is not None:
                self.model.policy.action_net.bias.data = self.original_action_bias * scale_factor

        self.current_ent_coef = new_ent_coef

    def _on_step(self) -> bool:
        """
        Update callback statistics and adjust training parameters.
        """
        # Record episode statistics
        if len(self.model.ep_info_buffer) > 0:
            last_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(last_info['r'])
            self.episode_lengths.append(last_info['l'])

            # Update success rate
            if 'success' in last_info:
                self.success_rate.append(float(last_info['success']))

        # Calculate moving average reward
        if self.episode_rewards:
            current_mean_reward = np.mean(list(self.episode_rewards))
            self.moving_avg_rewards.append(current_mean_reward)

        # Record training metrics
        if hasattr(self.model, 'logger'):
            logs = self.model.logger.name_to_value
            if 'train/loss' in logs:
                self.loss_values.append(logs['train/loss'])
            if 'train/entropy' in logs:
                self.entropy_values.append(logs['train/entropy'])
            if 'train/value_loss' in logs:
                self.value_loss_values.append(logs['train/value_loss'])
            if 'train/policy_loss' in logs:
                self.policy_loss_values.append(logs['train/policy_loss'])

        # Periodic checks and adjustments
        if self.n_calls % self.check_freq == 0:
            # Adjust entropy coefficient
            self._adjust_entropy_coefficient()

            # Log training progress
            self._log_training_progress()

            # Save best model
            if self.moving_avg_rewards and np.mean(list(self.moving_avg_rewards)) > self.best_mean_reward:
                self.best_mean_reward = np.mean(list(self.moving_avg_rewards))
                self.model.save(self.save_path)

        # Periodic model saving
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = str(
                self.log_dir / f'checkpoint_{self.n_calls}_steps')
            self.model.save(checkpoint_path)

            # Save training log
            self._save_training_log()

        return True

    def _log_training_progress(self) -> None:
        """
        Log detailed training progress information.
        """
        # Calculate current metrics
        current_reward = np.mean(
            list(self.episode_rewards)) if self.episode_rewards else 0
        current_length = np.mean(
            list(self.episode_lengths)) if self.episode_lengths else 0
        current_success = np.mean(
            list(self.success_rate)) if self.success_rate else 0

        # Update training log
        self.training_log['rewards'].append(current_reward)
        self.training_log['lengths'].append(current_length)
        self.training_log['entropy'].append(self.current_ent_coef)
        self.training_log['success_rates'].append(current_success)
        if self.loss_values:
            self.training_log['losses'].append(self.loss_values[-1])

        if self.verbose > 0:
            print("\n" + "="*50)
            print(f"Training Progress at Step {self.n_calls}")
            print(f"Mean reward: {current_reward:.2f}")
            print(f"Mean episode length: {current_length:.2f}")
            print(f"Current entropy coefficient: {self.current_ent_coef:.4f}")
            print(f"Success rate: {current_success:.2%}")
            if self.loss_values:
                print(f"Current loss: {self.loss_values[-1]:.4f}")
            print(f"Best mean reward: {self.best_mean_reward:.2f}")
            print("="*50 + "\n")

    def _save_training_log(self) -> None:
        """
        Save training log to disk.
        """
        log_path = self.log_dir / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f)

    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get current training statistics.

        Returns:
            Dictionary containing various training metrics
        """
        return {
            'mean_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
            'mean_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0,
            'success_rate': np.mean(list(self.success_rate)) if self.success_rate else 0,
            'current_entropy': self.current_ent_coef,
            'best_reward': self.best_mean_reward,
            'recent_losses': list(self.loss_values) if self.loss_values else []
        }
