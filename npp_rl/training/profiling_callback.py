"""Callback for runtime profiling during training."""

import logging
import time

from stable_baselines3.common.callbacks import BaseCallback

from npp_rl.training.runtime_profiler import RuntimeProfiler, ComponentTiming

logger = logging.getLogger(__name__)


class RuntimeProfilingCallback(BaseCallback):
    """Callback that tracks detailed timing during training loop.

    Integrates with RuntimeProfiler to track:
    - Model forward/backward passes
    - Environment steps
    - Data collection
    - Policy updates
    - Value function computation
    - Advantage estimation
    - Gradient computation
    """

    def __init__(
        self,
        profiler: RuntimeProfiler,
        log_freq: int = 1000,
        verbose: int = 0,
        track_detailed_components: bool = True,
        memory_snapshot_freq: int = 5000,
    ):
        """Initialize profiling callback.

        Args:
            profiler: RuntimeProfiler instance
            log_freq: Frequency to log profiling updates
            verbose: Verbosity level
            track_detailed_components: Track detailed per-step component timings
            memory_snapshot_freq: Frequency of memory snapshots during training (in timesteps)
        """
        super().__init__(verbose)
        self.profiler = profiler
        self.log_freq = log_freq
        self.track_detailed_components = track_detailed_components
        self.memory_snapshot_freq = memory_snapshot_freq

        # Track step timing
        self.step_start_time = None
        self.last_log_step = 0
        self.last_memory_snapshot_step = 0
        self.rollout_start_time = None
        self.train_start_time = None

    def _on_training_start(self) -> None:
        """Called when training starts."""
        logger.info("Runtime profiling callback initialized")
        if self.profiler and self.profiler.enable_pytorch_profiler:
            self.profiler.start_pytorch_profiler("training_loop")

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.profiler:
            if self.profiler.enable_pytorch_profiler:
                self.profiler.stop_pytorch_profiler()
            # Save profiling data when training ends (may be incomplete)
            try:
                self.profiler.save_summary(force=True)
            except Exception as e:
                logger.error(f"Failed to save profiling data in callback: {e}")
        logger.info("Runtime profiling callback finished")

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Step PyTorch profiler if enabled
        if self.profiler and self.profiler.enable_pytorch_profiler:
            self.profiler.step_pytorch_profiler()

        # Record periodic memory snapshots
        if (
            self.profiler
            and self.num_timesteps - self.last_memory_snapshot_step
            >= self.memory_snapshot_freq
        ):
            self.last_memory_snapshot_step = self.num_timesteps
            self.profiler.record_memory_snapshot(f"training_step_{self.num_timesteps}")
            if self.verbose >= 1:
                logger.info(f"Memory snapshot recorded at step {self.num_timesteps}")

        # Log periodic updates
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self.last_log_step = self.num_timesteps
            if self.verbose >= 1 and self.profiler:
                logger.debug(
                    f"Profiling: step={self.num_timesteps}, "
                    f"components_tracked={len(self.profiler.component_timings)}"
                )

        return True

    def _on_rollout_start(self) -> None:
        """Called when rollout collection starts."""
        if self.profiler and self.track_detailed_components:
            self.rollout_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """Called when rollout collection ends."""
        if (
            self.profiler
            and self.track_detailed_components
            and self.rollout_start_time is not None
        ):
            duration = time.time() - self.rollout_start_time
            self._add_component_timing("rollout_collection", duration)
            self.rollout_start_time = None

    def _add_component_timing(self, component_name: str, duration: float) -> None:
        """Helper method to add timing for a component.

        Args:
            component_name: Name of the component
            duration: Duration in seconds
        """
        if not self.profiler:
            return

        with self.profiler.component_lock:
            if component_name not in self.profiler.component_timings:
                self.profiler.component_timings[component_name] = ComponentTiming(
                    name=component_name
                )
            self.profiler.component_timings[component_name].add_timing(duration)
