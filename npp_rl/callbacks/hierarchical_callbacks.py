"""
Hierarchical Training Callbacks for Stability Monitoring

This module implements comprehensive callbacks for monitoring and ensuring
stable training of hierarchical PPO policies with two-level architecture.

Key Features:
- Policy gradient norm tracking for both levels
- Value function loss monitoring
- Subtask transition tracking
- Exploration efficiency metrics
- Mine avoidance success rates
- Adaptive hyperparameter adjustment
- Training stability detection

Task 2.4 Requirements:
- Continuous stability metrics tracking
- Adaptive learning rate adjustment
- Policy balance monitoring
- Curriculum progression tracking
"""

import numpy as np
import torch
from collections import deque, defaultdict
from typing import List, Optional
from stable_baselines3.common.callbacks import BaseCallback
import warnings


class HierarchicalStabilityCallback(BaseCallback):
    """
    Monitor training stability for hierarchical PPO.

    Tracks:
    - Policy gradient norms (high-level and low-level)
    - Value function loss convergence
    - Policy loss trends
    - Entropy trends (exploration)
    - Learning rate schedules

    Detects:
    - Training instability (exploding gradients, diverging losses)
    - Training stagnation (no improvement)
    - Policy imbalance (one level dominating)
    """

    def __init__(
        self,
        instability_window: int = 1000,
        stagnation_window: int = 10000,
        gradient_norm_threshold: float = 10.0,
        value_loss_threshold: float = 5.0,
        log_freq: int = 100,
        verbose: int = 1,
    ):
        """
        Initialize stability monitoring callback.

        Args:
            instability_window: Steps to check for instability
            stagnation_window: Steps to check for stagnation
            gradient_norm_threshold: Max gradient norm before flagging instability
            value_loss_threshold: Max value loss before flagging instability
            log_freq: Logging frequency (steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.instability_window = instability_window
        self.stagnation_window = stagnation_window
        self.gradient_norm_threshold = gradient_norm_threshold
        self.value_loss_threshold = value_loss_threshold
        self.log_freq = log_freq

        # History buffers
        self.high_level_gradient_norms = deque(maxlen=instability_window)
        self.low_level_gradient_norms = deque(maxlen=instability_window)
        self.high_level_value_losses = deque(maxlen=instability_window)
        self.low_level_value_losses = deque(maxlen=instability_window)
        self.high_level_policy_losses = deque(maxlen=instability_window)
        self.low_level_policy_losses = deque(maxlen=instability_window)
        self.episode_rewards = deque(maxlen=stagnation_window)

        # Stability flags
        self.is_stable = True
        self.instability_reasons = []
        self.stagnation_detected = False

        # Step counter
        self.step_count = 0

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1

        # Extract metrics from model if available
        if hasattr(self.model, "policy"):
            # Handle DDP-wrapped policies
            from npp_rl.training.distributed_utils import is_model_wrapped_ddp

            policy = (
                self.model.policy.module
                if is_model_wrapped_ddp(self.model.policy)
                else self.model.policy
            )

            # Track gradient norms
            if hasattr(policy, "mlp_extractor"):
                mlp = policy.mlp_extractor

                # High-level gradients
                if hasattr(mlp, "high_level_policy"):
                    hl_grad_norm = self._compute_gradient_norm(mlp.high_level_policy)
                    if hl_grad_norm is not None:
                        self.high_level_gradient_norms.append(hl_grad_norm)

                # Low-level gradients
                if hasattr(mlp, "low_level_policy"):
                    ll_grad_norm = self._compute_gradient_norm(mlp.low_level_policy)
                    if ll_grad_norm is not None:
                        self.low_level_gradient_norms.append(ll_grad_norm)

        # Log metrics periodically
        if self.step_count % self.log_freq == 0:
            self._log_stability_metrics()
            self._detect_instability()
            self._detect_stagnation()

        return True  # Continue training

    def _compute_gradient_norm(self, module: torch.nn.Module) -> Optional[float]:
        """Compute L2 norm of gradients for a module."""
        total_norm = 0.0
        param_count = 0

        for param in module.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            return None

        return total_norm**0.5

    def _log_stability_metrics(self):
        """Log stability metrics to tensorboard."""
        if not self.logger:
            return

        # Gradient norms
        if self.high_level_gradient_norms:
            self.logger.record(
                "stability/high_level_gradient_norm_mean",
                np.mean(self.high_level_gradient_norms),
            )
            self.logger.record(
                "stability/high_level_gradient_norm_max",
                np.max(self.high_level_gradient_norms),
            )

        if self.low_level_gradient_norms:
            self.logger.record(
                "stability/low_level_gradient_norm_mean",
                np.mean(self.low_level_gradient_norms),
            )
            self.logger.record(
                "stability/low_level_gradient_norm_max",
                np.max(self.low_level_gradient_norms),
            )

        # Gradient norm ratio (policy balance indicator)
        if self.high_level_gradient_norms and self.low_level_gradient_norms:
            hl_mean = np.mean(self.high_level_gradient_norms)
            ll_mean = np.mean(self.low_level_gradient_norms)
            if ll_mean > 0:
                ratio = hl_mean / ll_mean
                self.logger.record("stability/gradient_norm_ratio", ratio)

        # Stability flag
        self.logger.record("stability/is_stable", 1.0 if self.is_stable else 0.0)

        # Instability reasons
        if self.instability_reasons:
            self.logger.record(
                "stability/instability_count", len(self.instability_reasons)
            )

    def _detect_instability(self):
        """Detect training instability."""
        self.instability_reasons = []

        # Check high-level gradient norms
        if self.high_level_gradient_norms:
            max_hl_grad = np.max(self.high_level_gradient_norms)
            if max_hl_grad > self.gradient_norm_threshold:
                self.instability_reasons.append(
                    f"High-level gradient norm {max_hl_grad:.2f} exceeds threshold {self.gradient_norm_threshold}"
                )

        # Check low-level gradient norms
        if self.low_level_gradient_norms:
            max_ll_grad = np.max(self.low_level_gradient_norms)
            if max_ll_grad > self.gradient_norm_threshold:
                self.instability_reasons.append(
                    f"Low-level gradient norm {max_ll_grad:.2f} exceeds threshold {self.gradient_norm_threshold}"
                )

        # Check value losses
        if self.high_level_value_losses:
            max_hl_vloss = np.max(self.high_level_value_losses)
            if max_hl_vloss > self.value_loss_threshold:
                self.instability_reasons.append(
                    f"High-level value loss {max_hl_vloss:.2f} exceeds threshold {self.value_loss_threshold}"
                )

        if self.low_level_value_losses:
            max_ll_vloss = np.max(self.low_level_value_losses)
            if max_ll_vloss > self.value_loss_threshold:
                self.instability_reasons.append(
                    f"Low-level value loss {max_ll_vloss:.2f} exceeds threshold {self.value_loss_threshold}"
                )

        # Update stability flag
        self.is_stable = len(self.instability_reasons) == 0

        # Log warnings
        if not self.is_stable and self.verbose > 0:
            warnings.warn(
                f"Training instability detected: {'; '.join(self.instability_reasons)}"
            )

    def _detect_stagnation(self):
        """Detect training stagnation (no improvement)."""
        if len(self.episode_rewards) < self.stagnation_window:
            return

        # Split into first and second half
        mid = len(self.episode_rewards) // 2
        first_half = list(self.episode_rewards)[:mid]
        second_half = list(self.episode_rewards)[mid:]

        # Check improvement
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        improvement = (second_mean - first_mean) / (abs(first_mean) + 1e-8)

        # Detection threshold: less than 1% improvement
        self.stagnation_detected = improvement < 0.01

        if self.stagnation_detected and self.verbose > 0:
            warnings.warn(
                f"Training stagnation detected: improvement {improvement:.4f} below threshold 0.01"
            )


class SubtaskTransitionCallback(BaseCallback):
    """
    Track subtask transitions and hierarchical coordination.

    Logs:
    - Subtask transition frequencies
    - Average subtask durations
    - Subtask success rates
    - Coordination efficiency
    """

    def __init__(
        self,
        log_freq: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Transition tracking
        self.subtask_history = []
        self.current_subtask = None
        self.subtask_start_step = 0
        self.subtask_durations = defaultdict(list)
        self.subtask_successes = defaultdict(lambda: {"success": 0, "total": 0})
        self.transition_count = 0

        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        # Extract current subtask from environment
        if hasattr(self.training_env, "get_attr"):
            try:
                current_subtasks = self.training_env.get_attr("current_subtask")
                if current_subtasks:
                    subtask = current_subtasks[0]  # From first env

                    # Track transition
                    if subtask != self.current_subtask:
                        self._log_transition(subtask)
                        self.current_subtask = subtask
                        self.transition_count += 1
                        self.subtask_start_step = self.step_count
            except AttributeError:
                pass

        # Periodic logging
        if self.step_count % self.log_freq == 0:
            self._log_subtask_metrics()

        return True

    def _log_transition(self, new_subtask):
        """Log a subtask transition."""
        if self.current_subtask is not None:
            duration = self.step_count - self.subtask_start_step
            self.subtask_durations[self.current_subtask].append(duration)
            self.subtask_history.append(
                {
                    "from": self.current_subtask,
                    "to": new_subtask,
                    "duration": duration,
                    "step": self.step_count,
                }
            )

    def _log_subtask_metrics(self):
        """Log subtask metrics to tensorboard."""
        if not self.logger:
            return

        # Transition count
        self.logger.record("hierarchical/total_transitions", self.transition_count)

        # Average durations per subtask
        for subtask, durations in self.subtask_durations.items():
            if durations:
                avg_duration = np.mean(durations)
                self.logger.record(f"hierarchical/avg_duration_{subtask}", avg_duration)

        # Success rates per subtask
        for subtask, stats in self.subtask_successes.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                self.logger.record(f"hierarchical/success_rate_{subtask}", success_rate)


class ExplorationMetricsCallback(BaseCallback):
    """
    Track exploration efficiency and ICM performance.

    Logs:
    - Intrinsic reward statistics
    - Curiosity module losses
    - Exploration coverage
    - Mine avoidance success rates
    """

    def __init__(
        self,
        log_freq: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Exploration tracking
        self.intrinsic_rewards = deque(maxlen=1000)
        self.icm_forward_losses = deque(maxlen=1000)
        self.icm_inverse_losses = deque(maxlen=1000)
        self.mine_proximity_events = deque(maxlen=1000)
        self.mine_avoidance_successes = 0
        self.mine_avoidance_failures = 0

        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        # Extract ICM metrics if available
        if hasattr(self.model, "intrinsic_reward_module"):
            icm = self.model.intrinsic_reward_module

            if hasattr(icm, "latest_intrinsic_reward"):
                self.intrinsic_rewards.append(icm.latest_intrinsic_reward)

            if hasattr(icm, "latest_forward_loss"):
                self.icm_forward_losses.append(icm.latest_forward_loss)

            if hasattr(icm, "latest_inverse_loss"):
                self.icm_inverse_losses.append(icm.latest_inverse_loss)

        # Periodic logging
        if self.step_count % self.log_freq == 0:
            self._log_exploration_metrics()

        return True

    def _log_exploration_metrics(self):
        """Log exploration metrics to tensorboard."""
        if not self.logger:
            return

        # Intrinsic rewards
        if self.intrinsic_rewards:
            self.logger.record(
                "icm/intrinsic_reward_mean", np.mean(self.intrinsic_rewards)
            )
            self.logger.record(
                "icm/intrinsic_reward_std", np.std(self.intrinsic_rewards)
            )
            self.logger.record(
                "icm/intrinsic_reward_max", np.max(self.intrinsic_rewards)
            )

        # ICM losses
        if self.icm_forward_losses:
            self.logger.record(
                "icm/forward_loss_mean", np.mean(self.icm_forward_losses)
            )

        if self.icm_inverse_losses:
            self.logger.record(
                "icm/inverse_loss_mean", np.mean(self.icm_inverse_losses)
            )

        # Mine avoidance
        if self.mine_avoidance_successes + self.mine_avoidance_failures > 0:
            success_rate = self.mine_avoidance_successes / (
                self.mine_avoidance_successes + self.mine_avoidance_failures
            )
            self.logger.record("exploration/mine_avoidance_success_rate", success_rate)


class AdaptiveLearningRateCallback(BaseCallback):
    """
    Dynamically adjust learning rates based on training metrics.

    Adjusts:
    - High-level policy learning rate
    - Low-level policy learning rate
    - ICM learning rate

    Based on:
    - Training stability (reduce if unstable)
    - Training stagnation (increase if stagnating)
    - Policy balance (adjust ratios if imbalanced)
    """

    def __init__(
        self,
        stability_callback: HierarchicalStabilityCallback,
        lr_decrease_factor: float = 0.8,
        lr_increase_factor: float = 1.1,
        lr_min: float = 1e-6,
        lr_max: float = 3e-4,
        adjustment_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stability_callback = stability_callback
        self.lr_decrease_factor = lr_decrease_factor
        self.lr_increase_factor = lr_increase_factor
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.adjustment_freq = adjustment_freq

        self.step_count = 0
        self.current_lr = None

    def _on_step(self) -> bool:
        self.step_count += 1

        # Periodic adjustment
        if self.step_count % self.adjustment_freq == 0:
            self._adjust_learning_rates()

        return True

    def _adjust_learning_rates(self):
        """Adjust learning rates based on training state."""
        if not hasattr(self.model, "lr_schedule"):
            return

        # Check stability
        if not self.stability_callback.is_stable:
            # Reduce learning rate
            new_lr = self._scale_learning_rate(self.lr_decrease_factor)
            if self.verbose > 0:
                print(f"Reducing learning rate to {new_lr:.6f} due to instability")

        elif self.stability_callback.stagnation_detected:
            # Increase learning rate
            new_lr = self._scale_learning_rate(self.lr_increase_factor)
            if self.verbose > 0:
                print(f"Increasing learning rate to {new_lr:.6f} due to stagnation")

    def _scale_learning_rate(self, factor: float) -> float:
        """Scale learning rate by factor, respecting bounds."""
        if self.current_lr is None:
            # Get initial LR
            self.current_lr = self.model.learning_rate

        # Scale
        new_lr = self.current_lr * factor

        # Clip to bounds
        new_lr = np.clip(new_lr, self.lr_min, self.lr_max)

        # Update model
        self.model.learning_rate = new_lr
        self.current_lr = new_lr

        return new_lr


class CurriculumProgressionCallback(BaseCallback):
    """
    Coordinate curriculum progression with CurriculumManager.

    This callback serves as a bridge between the training loop and the
    CurriculumManager, providing automatic progression monitoring and
    environment updates during training.

    Responsibilities:
    - Record episode outcomes to CurriculumManager
    - Periodically check for curriculum advancement readiness
    - Update environment wrappers when advancing stages
    - Log curriculum metrics to TensorBoard

    Note: This callback delegates all curriculum logic to CurriculumManager.
    It does not reimplement progression logic - it coordinates it.
    """

    def __init__(
        self,
        curriculum_manager=None,
        check_freq: int = 10000,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        """
        Initialize curriculum progression callback.

        Args:
            curriculum_manager: CurriculumManager instance to coordinate with.
                               If None, callback operates in standalone mode.
            check_freq: Frequency (in steps) to check for advancement readiness
            log_freq: Frequency (in steps) to log curriculum metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.step_count = 0
        self.last_stage_idx = -1

        # Validate curriculum manager if provided
        if curriculum_manager is not None:
            from npp_rl.training.curriculum_manager import CurriculumManager

            if not isinstance(curriculum_manager, CurriculumManager):
                warnings.warn(
                    f"curriculum_manager should be CurriculumManager instance, got {type(curriculum_manager)}"
                )

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1

        # Record episode outcomes if curriculum manager available
        if self.curriculum_manager is not None:
            self._record_episodes()

        # Check for advancement periodically
        if self.step_count % self.check_freq == 0:
            if self.curriculum_manager is not None:
                advanced = self.curriculum_manager.check_advancement()
                if advanced:
                    self._handle_advancement()

        # Log metrics periodically
        if self.step_count % self.log_freq == 0:
            self._log_curriculum_metrics()

        return True

    def _record_episodes(self):
        """Record episode outcomes to curriculum manager."""
        # Extract episode outcomes from training data
        if hasattr(self.locals, "dones"):
            dones = self.locals.get("dones", [])
            infos = self.locals.get("infos", [])

            for done, info in zip(dones, infos):
                if done:
                    # Try multiple keys for episode success
                    success = (
                        info.get("is_success", False)
                        or info.get("episode_success", False)
                        or info.get("success", False)
                    )

                    # Record to curriculum manager
                    self.curriculum_manager.record_episode(success)

    def _handle_advancement(self):
        """Handle curriculum stage advancement."""
        new_stage = self.curriculum_manager.get_current_stage()
        new_stage_idx = self.curriculum_manager.current_stage_idx

        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("CURRICULUM ADVANCEMENT!")
            print(f"New stage: {new_stage} (stage {new_stage_idx + 1}/6)")
            print("=" * 60 + "\n")

        # Update environment wrappers if possible
        self._update_environments()

    def _update_environments(self):
        """Update environment wrappers with new curriculum stage."""
        if not hasattr(self, "training_env"):
            return

        new_stage = self.curriculum_manager.get_current_stage()

        # Try to update vectorized environment attributes
        try:
            if hasattr(self.training_env, "set_attr"):
                self.training_env.set_attr("current_curriculum_stage", new_stage)
            elif hasattr(self.training_env, "envs"):
                # Direct attribute setting for each environment
                for env in self.training_env.envs:
                    if hasattr(env, "current_curriculum_stage"):
                        env.current_curriculum_stage = new_stage
        except Exception as e:
            if self.verbose > 0:
                warnings.warn(
                    f"Could not update environment curriculum stage: {e}. "
                    f"Manual intervention may be required."
                )

    def _log_curriculum_metrics(self):
        """Log curriculum metrics to TensorBoard."""
        if not self.logger or self.curriculum_manager is None:
            return

        # Current stage info
        current_stage = self.curriculum_manager.get_current_stage()
        current_stage_idx = self.curriculum_manager.current_stage_idx

        # Get performance for current stage
        perf = self.curriculum_manager.get_stage_performance(current_stage)

        # Log metrics
        self.logger.record("curriculum/current_stage_idx", current_stage_idx)
        self.logger.record("curriculum/current_stage_name", current_stage)
        self.logger.record("curriculum/success_rate", perf["success_rate"])
        self.logger.record("curriculum/episodes_in_stage", perf["episodes"])
        self.logger.record(
            "curriculum/can_advance", 1.0 if perf["can_advance"] else 0.0
        )
        self.logger.record(
            "curriculum/advancement_threshold", perf["advancement_threshold"]
        )

        # Log performance for all stages (for comparison)
        for stage_name in self.curriculum_manager.CURRICULUM_ORDER:
            stage_perf = self.curriculum_manager.get_stage_performance(stage_name)
            self.logger.record(
                f"curriculum_stages/{stage_name}_success_rate",
                stage_perf["success_rate"],
            )
            self.logger.record(
                f"curriculum_stages/{stage_name}_episodes", stage_perf["episodes"]
            )


def create_hierarchical_callbacks(
    log_freq: int = 100,
    adjustment_freq: int = 10000,
    curriculum_manager=None,
    verbose: int = 1,
) -> List[BaseCallback]:
    """
    Create a comprehensive set of hierarchical training callbacks.

    Args:
        log_freq: Logging frequency (steps)
        adjustment_freq: Adaptive adjustment frequency (steps)
        curriculum_manager: Optional CurriculumManager for curriculum progression
        verbose: Verbosity level

    Returns:
        List of configured callbacks
    """
    # Create stability callback (needed by adaptive LR callback)
    stability_callback = HierarchicalStabilityCallback(
        log_freq=log_freq,
        verbose=verbose,
    )

    callbacks = [
        stability_callback,
        SubtaskTransitionCallback(log_freq=log_freq, verbose=verbose),
        ExplorationMetricsCallback(log_freq=log_freq, verbose=verbose),
        AdaptiveLearningRateCallback(
            stability_callback=stability_callback,
            adjustment_freq=adjustment_freq,
            verbose=verbose,
        ),
    ]

    # Add curriculum callback if curriculum manager provided
    if curriculum_manager is not None:
        callbacks.append(
            CurriculumProgressionCallback(
                curriculum_manager=curriculum_manager,
                check_freq=adjustment_freq,  # Check advancement at same freq as LR adjustments
                log_freq=log_freq * 10,
                verbose=verbose,
            )
        )

    return callbacks
