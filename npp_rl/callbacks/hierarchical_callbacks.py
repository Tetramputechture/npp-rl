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

import logging
from stable_baselines3.common.callbacks import BaseCallback
import warnings
from npp_rl.training.curriculum_components import CURRICULUM_ORDER

logger = logging.getLogger(__name__)


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
        self.use_vec_wrapper = False  # Will be set in _init_callback()

        # Validate curriculum manager if provided
        if curriculum_manager is not None:
            from npp_rl.training.curriculum_manager import CurriculumManager

            if not isinstance(curriculum_manager, CurriculumManager):
                warnings.warn(
                    f"curriculum_manager should be CurriculumManager instance, got {type(curriculum_manager)}"
                )

    def _init_callback(self) -> None:
        """Initialize callback after training environment is set.

        This is called by BaseCallback after the training_env is available.
        We use it to detect if CurriculumVecEnvWrapper is in use.
        """
        super()._init_callback()

        # Check if training environment is wrapped with CurriculumVecEnvWrapper
        # If so, disable duplicate episode recording (VecWrapper handles it)
        if self.training_env is not None:
            from npp_rl.wrappers.curriculum_env import CurriculumVecEnvWrapper

            # Walk through wrapper chain to find CurriculumVecEnvWrapper
            env = self.training_env
            while env is not None:
                if isinstance(env, CurriculumVecEnvWrapper):
                    self.use_vec_wrapper = True
                    if self.verbose > 0:
                        logger.info(
                            "[CurriculumCallback] CurriculumVecEnvWrapper detected - "
                            "episode recording handled by wrapper, callback will skip recording"
                        )
                    break
                # Try to get wrapped env
                env = getattr(env, "venv", None) or getattr(env, "env", None)

    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1

        # Record episode outcomes if curriculum manager available
        # (skipped if VecEnvWrapper handles it)
        if self.curriculum_manager is not None:
            self._record_episodes()

        # Check for advancement periodically
        # Skip if VecEnvWrapper handles it (avoids race conditions)
        if self.step_count % self.check_freq == 0:
            if self.curriculum_manager is not None and not self.use_vec_wrapper:
                # Check for automatic threshold adjustment (Week 3-4)
                if hasattr(self.curriculum_manager, "check_auto_adjustment"):
                    adjusted = self.curriculum_manager.check_auto_adjustment(
                        self.num_timesteps
                    )
                    if adjusted:
                        self.logger.record("curriculum/auto_adjustment_event", 1.0)

                advanced = self.curriculum_manager.check_advancement()
                if advanced:
                    self._handle_advancement()
            elif self.use_vec_wrapper and self.step_count % (self.check_freq * 10) == 0:
                # Log curriculum status less frequently when VecWrapper manages it
                if self.verbose > 0:
                    current_stage = self.curriculum_manager.get_current_stage()
                    logger.info(
                        f"[CurriculumCallback] Curriculum status: {current_stage} "
                        f"(managed by VecEnvWrapper)"
                    )

        # Log metrics periodically
        if self.step_count % self.log_freq == 0:
            self._log_curriculum_metrics()

        return True

    def _record_episodes(self):
        """Record episode outcomes to curriculum manager.

        Note: If CurriculumVecEnvWrapper is in use, skip recording here since
        the wrapper already handles all episode recording with full stage/generator info.
        """
        # Skip recording if VecEnvWrapper handles it
        if self.use_vec_wrapper:
            return  # VecEnvWrapper records episodes with full context

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

                    # Extract stage and generator info if available
                    stage = info.get("curriculum_stage", None)
                    generator_type = info.get("curriculum_generator", None)
                    frames = info.get("l", None)  # Frame count from episode

                    # Record to curriculum manager with full context
                    if stage and stage != "unknown":
                        self.curriculum_manager.record_episode(
                            stage, success, generator_type, frames
                        )
                    else:
                        # Fallback: record without stage (legacy behavior)
                        # This path shouldn't be reached if curriculum is properly configured
                        print(
                            "[CurriculumCallback] Recording episode without stage info - "
                            "this may indicate a configuration issue"
                        )

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
        """Log comprehensive curriculum metrics to TensorBoard.

        Delegates to curriculum_manager.log_curriculum_metrics() for detailed
        per-generator logging and sampling distribution tracking.
        """
        if not self.logger or self.curriculum_manager is None:
            return

        # Current stage info
        current_stage = self.curriculum_manager.get_current_stage()
        current_stage_idx = self.curriculum_manager.current_stage_idx

        # Get performance for current stage
        perf = self.curriculum_manager.get_stage_performance(current_stage)

        # Log basic metrics using stable_baselines3 logger
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
        for stage_name in CURRICULUM_ORDER:
            stage_perf = self.curriculum_manager.get_stage_performance(stage_name)
            self.logger.record(
                f"curriculum_stages/{stage_name}_success_rate",
                stage_perf["success_rate"],
            )

        # Call comprehensive logging method for per-generator metrics
        # This logs sampling distribution, balance variance, and success rates
        # per generator type (e.g., maze:small, jump_required:medium, etc.)
        if hasattr(self.curriculum_manager, "log_curriculum_metrics"):
            # Get TensorBoard writer from model logger if available
            writer = None
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "writer"):
                writer = self.model.logger.writer
            elif hasattr(self.logger, "writer"):
                writer = self.logger.writer

            if writer is not None:
                self.curriculum_manager.log_curriculum_metrics(
                    writer, self.step_count, current_stage_only=False
                )
