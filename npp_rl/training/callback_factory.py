"""Factory for creating training callbacks."""

import logging
from pathlib import Path
from typing import List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from npp_rl.training.training_callbacks import VerboseTrainingCallback

logger = logging.getLogger(__name__)


class CallbackFactory:
    """Creates and configures training callbacks."""

    def __init__(
        self,
        output_dir: Path,
        use_hierarchical_ppo: bool = False,
        use_curriculum: bool = False,
        curriculum_manager=None,
        use_distributed: bool = False,
        world_size: int = 1,
    ):
        """Initialize callback factory.

        Args:
            output_dir: Output directory for logs and visualizations
            use_hierarchical_ppo: Whether using hierarchical PPO
            use_curriculum: Whether using curriculum learning
            curriculum_manager: Curriculum manager instance
            use_distributed: Whether using distributed training
            world_size: Number of GPUs for distributed training
        """
        self.output_dir = Path(output_dir)
        self.use_hierarchical_ppo = use_hierarchical_ppo
        self.use_curriculum = use_curriculum
        self.curriculum_manager = curriculum_manager
        self.use_distributed = use_distributed
        self.world_size = world_size

    def create_callbacks(
        self, user_callback: Optional[BaseCallback] = None
    ) -> List[BaseCallback]:
        """Create list of training callbacks.

        Args:
            user_callback: Optional user-provided callback

        Returns:
            List of configured callbacks
        """
        callbacks = []

        # Add verbose callback for training progress monitoring
        verbose_callback = VerboseTrainingCallback(log_freq=1)
        callbacks.append(verbose_callback)
        logger.info("Added verbose training callback")

        # Add user callback if provided
        if user_callback is not None:
            callbacks.append(user_callback)
            logger.info("Added user-provided callback")

        # Add enhanced TensorBoard metrics callback
        # Note: PBRS logging integrated into this callback to avoid duplication
        from npp_rl.callbacks import EnhancedTensorBoardCallback

        enhanced_tb_callback = EnhancedTensorBoardCallback(
            log_freq=200,  # Log scalars every 200 steps (reduced overhead)
            histogram_freq=5000,  # Log histograms every 5000 steps (expensive operation)
            verbose=1,
            log_gradients=False,  # Gradient logging disabled by default
            log_weights=False,  # Weight logging disabled by default (expensive)
        )
        callbacks.append(enhanced_tb_callback)
        logger.info(
            "Added enhanced TensorBoard callback (includes PBRS and curriculum metrics)"
        )

        # Add route visualization callback
        from npp_rl.callbacks import RouteVisualizationCallback

        routes_dir = self.output_dir / "route_visualizations"
        route_callback = RouteVisualizationCallback(
            save_dir=str(routes_dir),
            max_routes_per_checkpoint=10,
            visualization_freq=100,
            max_stored_routes=100,
            async_save=True,
            image_size=(800, 600),
            verbose=2,
        )
        callbacks.append(route_callback)
        logger.info(f"Added route visualization callback (saving to {routes_dir})")

        # Add hierarchical PPO callbacks if using hierarchical training
        if self.use_hierarchical_ppo:
            self._add_hierarchical_callbacks(callbacks)

        # Add curriculum progression callback if curriculum learning is enabled
        if self.use_curriculum and self.curriculum_manager is not None:
            self._add_curriculum_callback(callbacks)

        # Add distributed progress callback if using multi-GPU training
        if self.use_distributed and self.world_size > 1:
            self._add_distributed_callback(callbacks)

        return callbacks

    def _add_hierarchical_callbacks(self, callbacks: List[BaseCallback]) -> None:
        """Add hierarchical PPO specific callbacks.

        Args:
            callbacks: List to add callbacks to
        """
        from npp_rl.callbacks.hierarchical_callbacks import (
            HierarchicalStabilityCallback,
            SubtaskTransitionCallback,
        )

        # Add stability monitoring
        stability_callback = HierarchicalStabilityCallback(
            instability_window=1000,
            stagnation_window=10000,
            gradient_norm_threshold=10.0,
            value_loss_threshold=5.0,
            log_freq=100,
            verbose=1,
        )
        callbacks.append(stability_callback)
        logger.info("Added hierarchical stability callback for training monitoring")

        # Add subtask transition tracking
        subtask_callback = SubtaskTransitionCallback(
            log_freq=100,
            verbose=1,
        )
        callbacks.append(subtask_callback)
        logger.info("Added subtask transition callback for HRL metrics")

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
