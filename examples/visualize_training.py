#!/usr/bin/env python3
"""Example: Training with Real-Time Visualization

This script demonstrates how to programmatically enable visualization
during training, without using the command-line interface.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import get_architecture_config
from npp_rl.training.architecture_trainer import ArchitectureTrainer
from npp_rl.callbacks import TrainingVisualizationCallback
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_visualization(
    architecture_name: str = "simple_mlp",
    train_dataset: str = "data/train",
    test_dataset: str = "data/test",
    num_envs: int = 2,
    total_timesteps: int = 50000,
    render_freq: int = 100,
    target_fps: int = 60,
):
    """Train a model with real-time visualization.

    Args:
        architecture_name: Name of architecture to train
        train_dataset: Path to training dataset
        test_dataset: Path to test dataset
        num_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        render_freq: How often to render (in timesteps)
        target_fps: Target frame rate for visualization
    """

    logger.info("=" * 70)
    logger.info("Training with Real-Time Visualization Example")
    logger.info("=" * 70)

    # Create output directory
    output_dir = Path("experiments") / f"viz_example_{architecture_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get architecture configuration
    logger.info(f"Loading architecture: {architecture_name}")
    arch_config = get_architecture_config(architecture_name)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = ArchitectureTrainer(
        architecture_config=arch_config,
        train_dataset_path=train_dataset,
        test_dataset_path=test_dataset,
        output_dir=output_dir,
        device_id=0,
        world_size=1,
        use_mixed_precision=False,
        use_curriculum=False,
    )

    # Setup model with default hyperparameters
    logger.info("Setting up model...")
    trainer.setup_model()

    # Setup environments with visualization enabled
    logger.info(f"Setting up {num_envs} environments with visualization...")
    trainer.setup_environments(
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        enable_visualization=True,  # Enable visualization
        vis_env_idx=0,  # Visualize first environment
    )

    # Create visualization callback
    logger.info("Creating visualization callback...")
    vis_callback = TrainingVisualizationCallback(
        render_freq=render_freq,
        render_mode="timesteps",
        env_idx=0,
        target_fps=target_fps,
        window_title=f"Training: {architecture_name}",
        verbose=1,
    )

    logger.info("=" * 70)
    logger.info("Starting training with visualization")
    logger.info(f"  Architecture: {architecture_name}")
    logger.info(f"  Environments: {num_envs}")
    logger.info(f"  Total timesteps: {total_timesteps:,}")
    logger.info(f"  Render frequency: every {render_freq} timesteps")
    logger.info(f"  Target FPS: {target_fps}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Visualization Controls:")
    logger.info("  SPACE - Pause/unpause visualization")
    logger.info("  ESC/Q - Close visualization window")
    logger.info("=" * 70)

    # Train with visualization callback
    try:
        training_results = trainer.train(
            total_timesteps=total_timesteps,
            eval_freq=10000,  # Evaluate every 10K steps
            save_freq=25000,  # Save every 25K steps
            callback_fn=vis_callback,
        )

        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info(f"Results: {training_results}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info(f"Partial results saved to: {output_dir}")

    finally:
        # Clean up
        trainer.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train NPP-RL model with real-time visualization"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="simple_mlp",
        help="Architecture to train (simple_mlp, gnn, transformer, etc.)",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="data/train",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default="data/test",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="Number of parallel environments (use ≤4 for best visualization)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--render-freq",
        type=int,
        default=100,
        help="Render every N timesteps",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target FPS for visualization (0 = unlimited)",
    )

    args = parser.parse_args()

    # Run training with visualization
    train_with_visualization(
        architecture_name=args.architecture,
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        render_freq=args.render_freq,
        target_fps=args.fps,
    )
