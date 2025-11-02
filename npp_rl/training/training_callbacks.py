"""Verbose training callback for debugging and monitoring training progress."""

import logging
import time
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class VerboseTrainingCallback(BaseCallback):
    """Callback for verbose logging during training to help debug hangs."""

    def __init__(self, log_freq: int = 1, verbose: int = 1):
        """
        Args:
            log_freq: How often to log (in number of rollouts/updates)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.update_count = 0
        self.start_time = None
        self.last_log_time = None

    def _on_training_start(self) -> None:
        """Called before the first rollout."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        logger.info("=" * 60)
        logger.info("VerboseTrainingCallback: Training started")
        logger.info("Beginning first environment reset and rollout collection...")
        logger.info("=" * 60)

    def _on_rollout_start(self) -> None:
        """Called before collecting a new rollout."""
        if self.update_count % self.log_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            since_last = current_time - self.last_log_time
            logger.info(
                f"[Update {self.update_count}] Starting rollout collection "
                f"(elapsed: {elapsed:.1f}s, since last: {since_last:.1f}s)"
            )
            self.last_log_time = current_time

    def _on_step(self) -> bool:
        """Called after each environment step during rollout."""
        return True

    def _on_rollout_end(self) -> None:
        """Called after rollout is collected."""
        if self.update_count % self.log_freq == 0:
            logger.info(
                f"[Update {self.update_count}] Rollout complete - "
                f"timesteps: {self.num_timesteps}, starting gradient update..."
            )
        self.update_count += 1

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"VerboseTrainingCallback: Training ended after {total_time:.1f}s")
        logger.info("=" * 60)
