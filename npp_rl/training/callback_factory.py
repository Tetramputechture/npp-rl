"""Factory for creating training callbacks."""

import logging
from pathlib import Path
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from npp_rl.training.training_callbacks import VerboseTrainingCallback
from npp_rl.callbacks import (
    RouteVisualizationCallback,
    EnhancedTensorBoardCallback,
    PBRSLoggingCallback,
)

logger = logging.getLogger(__name__)


class CallbackFactory:
    """Creates and configures training callbacks."""

    def __init__(
        self,
        output_dir: Path,
        use_curriculum: bool = False,
        curriculum_manager=None,
        use_distributed: bool = False,
        world_size: int = 1,
        enable_early_stopping: bool = False,
        early_stopping_patience: int = 10,
    ):
        """Initialize callback factory.

        Args:
            output_dir: Output directory for logs and visualizations
            use_curriculum: Whether using curriculum learning
            curriculum_manager: Curriculum manager instance
            use_distributed: Whether using distributed training
            world_size: Number of GPUs for distributed training
            enable_early_stopping: Enable early stopping callback (default: False)
            early_stopping_patience: Patience for early stopping (default: 10)
        """
        self.output_dir = Path(output_dir)
        self.use_curriculum = use_curriculum
        self.curriculum_manager = curriculum_manager
        self.use_distributed = use_distributed
        self.world_size = world_size
        self.enable_early_stopping = enable_early_stopping
        self.early_stopping_patience = early_stopping_patience

    def create_callbacks(
        self, user_callback: Optional[BaseCallback] = None
    ) -> List[BaseCallback]:
        """Create list of training callbacks.

        Args:
            user_callback: Optional user-provided callback (profiling callback
                should be passed here to be added first)

        Returns:
            List of configured callbacks
        """
        callbacks = []

        # Add user callback first (typically profiling callback)
        # This ensures it tracks all other callbacks
        if user_callback is not None:
            callbacks.append(user_callback)
            logger.info("Added user-provided callback (first in chain)")

        # Add verbose callback for training progress monitoring
        verbose_callback = VerboseTrainingCallback(log_freq=1)
        callbacks.append(verbose_callback)
        logger.info("Added verbose training callback")

        # Add enhanced TensorBoard metrics callback
        enhanced_tb_callback = EnhancedTensorBoardCallback(
            log_freq=200,  # Log scalars every 200 steps (reduced overhead)
            verbose=1,
        )
        callbacks.append(enhanced_tb_callback)
        logger.info(
            "Added enhanced TensorBoard callback (includes PBRS and curriculum metrics)"
        )

        # Add PBRS logging callback for detailed episode-level PBRS metrics
        pbrs_callback = PBRSLoggingCallback(verbose=0)
        callbacks.append(pbrs_callback)
        logger.info(
            "Added PBRS logging callback (episode-level reward component analysis)"
        )

        routes_dir = self.output_dir / "route_visualizations"
        # Sample 10% of episodes by default to reduce overhead
        # This significantly reduces per-step overhead while still providing useful visualizations
        episode_sampling_rate = 0.75
        route_callback = RouteVisualizationCallback(
            save_dir=str(routes_dir),
            max_routes_per_checkpoint=50,
            visualization_freq=50000,
            max_stored_routes=200,
            async_save=True,
            image_size=(800, 600),
            episode_sampling_rate=episode_sampling_rate,
            verbose=2,
        )
        callbacks.append(route_callback)
        logger.info(
            f"Added route visualization callback (saving to {routes_dir}, "
            f"sampling {episode_sampling_rate * 100:.1f}% of episodes)"
        )

        # Add curriculum progression callback if curriculum learning is enabled
        if self.use_curriculum and self.curriculum_manager is not None:
            self._add_curriculum_callback(callbacks)

            # Add early stopping if enabled (Week 3-4)
            if self.enable_early_stopping:
                from npp_rl.callbacks.early_stopping import (
                    CurriculumEarlyStoppingCallback,
                )

                early_stop_callback = CurriculumEarlyStoppingCallback(
                    patience=self.early_stopping_patience,
                    min_delta=0.01,  # 1% improvement required
                    min_evaluations=5,
                    curriculum_manager=self.curriculum_manager,
                    eval_freq=25000,  # Match default eval freq
                    verbose=1,
                )
                callbacks.append(early_stop_callback)
                logger.info(
                    f"Added early stopping callback (patience={self.early_stopping_patience})"
                )

        # Add distributed progress callback if using multi-GPU training
        if self.use_distributed and self.world_size > 1:
            self._add_distributed_callback(callbacks)

        return callbacks

    def _add_curriculum_callback(self, callbacks: List[BaseCallback]) -> None:
        """Add curriculum progression callback.

        Args:
            callbacks: List to add callbacks to
        """
        from npp_rl.callbacks.hierarchical_callbacks import (
            CurriculumProgressionCallback,
        )

        curriculum_callback = CurriculumProgressionCallback(
            curriculum_manager=self.curriculum_manager,
            check_freq=10000,  # Check advancement every 10K steps
            log_freq=1000,  # Log metrics every 1K steps
            verbose=1,
        )
        callbacks.append(curriculum_callback)
        logger.info(
            f"Added curriculum progression callback (current stage: {self.curriculum_manager.get_current_stage()})"
        )

    def _add_distributed_callback(self, callbacks: List[BaseCallback]) -> None:
        """Add distributed progress callback.

        Args:
            callbacks: List to add callbacks to
        """
        from npp_rl.callbacks import DistributedProgressCallback

        distributed_callback = DistributedProgressCallback(log_freq=1000, verbose=1)
        callbacks.append(distributed_callback)
        logger.info("Added distributed progress callback for multi-GPU coordination")
