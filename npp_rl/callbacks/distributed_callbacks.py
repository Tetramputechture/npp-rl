"""Callbacks for distributed training coordination.

This module provides callbacks that handle distributed training scenarios,
ensuring that logging and progress tracking work correctly across multiple GPUs.
"""

from stable_baselines3.common.callbacks import BaseCallback
from npp_rl.training.distributed_utils import get_rank, get_world_size, is_main_process


class DistributedProgressCallback(BaseCallback):
    """Log distributed training progress on main process only.
    
    This callback ensures that distributed training metrics are properly logged
    without conflicts between multiple GPU processes. It tracks:
    - Current rank and world size
    - Global step count (accounting for all GPUs)
    - Per-process and global throughput
    
    Only the main process (rank 0) logs these metrics to avoid conflicts.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        """Initialize distributed progress callback.
        
        Args:
            log_freq: How often to log distributed metrics (in timesteps)
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rank = get_rank()
        self.world_size = get_world_size()

    def _on_step(self) -> bool:
        """Called after each environment step.
        
        Returns:
            True to continue training, False to stop
        """
        if not is_main_process():
            return True  # Silent on worker processes

        if self.num_timesteps % self.log_freq == 0:
            # Log progress with multi-GPU context
            if self.world_size > 1:
                self.logger.record("distributed/rank", self.rank)
                self.logger.record("distributed/world_size", self.world_size)
                self.logger.record(
                    "distributed/global_steps", self.num_timesteps * self.world_size
                )

                # Log effective throughput
                if hasattr(self, "t_start") and self.num_timesteps > 0:
                    import time

                    elapsed = time.time() - self.t_start
                    if elapsed > 0:
                        fps_per_gpu = self.num_timesteps / elapsed
                        global_fps = fps_per_gpu * self.world_size
                        self.logger.record("distributed/fps_per_gpu", fps_per_gpu)
                        self.logger.record("distributed/global_fps", global_fps)

        return True

    def _on_training_start(self) -> None:
        """Called at the start of training."""
        import time

        self.t_start = time.time()

        if is_main_process() and self.world_size > 1:
            if self.verbose > 0:
                print(f"Distributed training started on {self.world_size} GPUs")
                print(f"Main process (rank {self.rank}) will handle logging")

