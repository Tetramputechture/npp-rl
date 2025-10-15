#!/usr/bin/env python3
"""Master training and comparison script for NPP-RL.

Orchestrates training, evaluation, and comparison of multiple architectures
with optional pretraining and multi-GPU support.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import (
    get_architecture_config,
    ARCHITECTURE_REGISTRY,
)
from npp_rl.utils import (
    setup_experiment_logging,
    save_experiment_config,
    create_s3_uploader,
    TensorBoardManager,
)
from npp_rl.training.architecture_trainer import ArchitectureTrainer
from npp_rl.training.pretraining_pipeline import run_bc_pretraining_if_available


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and compare NPP-RL architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Unique experiment name"
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        required=True,
        choices=list(ARCHITECTURE_REGISTRY.keys()),
        help="Architecture names to train",
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, help="Path to test dataset"
    )

    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="experiments/", help="Base output directory"
    )

    # Pretraining options
    parser.add_argument(
        "--test-pretraining",
        action="store_true",
        help="Compare with and without pretraining",
    )
    parser.add_argument(
        "--no-pretraining", action="store_true", help="Skip pretraining entirely"
    )
    parser.add_argument(
        "--replay-data-dir",
        type=str,
        default=None,
        help="Directory containing replay data for BC",
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=10, help="Number of BC training epochs"
    )
    parser.add_argument(
        "--bc-batch-size", type=int, default=64, help="BC training batch size"
    )

    # Training options
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--num-envs", type=int, default=64, help="Number of parallel environments"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=100000, help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=500000,
        help="Checkpoint save frequency (timesteps)",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=250,
        help="Number of episodes for final evaluation (reduce for quick testing)",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip final evaluation (for quick CPU validation tests)",
    )

    # Multi-GPU options
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable mixed precision training",
    )

    # Hierarchical PPO options
    parser.add_argument(
        "--use-hierarchical-ppo",
        action="store_true",
        help="Use hierarchical PPO with high/low level policies",
    )
    parser.add_argument(
        "--high-level-update-freq",
        type=int,
        default=50,
        help="High-level policy update frequency (hierarchical PPO)",
    )

    # Curriculum learning options
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Enable curriculum learning from simple to complex levels",
    )
    parser.add_argument(
        "--curriculum-start-stage",
        type=str,
        default="simple",
        choices=["simple", "medium", "complex", "exploration", "mine_heavy"],
        help="Starting curriculum stage",
    )
    parser.add_argument(
        "--curriculum-threshold",
        type=float,
        default=0.7,
        help="Success rate threshold for curriculum advancement",
    )
    parser.add_argument(
        "--curriculum-min-episodes",
        type=int,
        default=100,
        help="Minimum episodes per curriculum stage",
    )

    # S3 options
    parser.add_argument(
        "--s3-bucket", type=str, default=None, help="S3 bucket for artifact upload"
    )
    parser.add_argument(
        "--s3-prefix", type=str, default="experiments/", help="S3 prefix for uploads"
    )
    parser.add_argument(
        "--s3-sync-freq", type=int, default=500000, help="S3 sync frequency (timesteps)"
    )

    # Video recording options
    parser.add_argument(
        "--record-eval-videos",
        action="store_true",
        help="Record videos during final evaluation",
    )
    parser.add_argument(
        "--max-videos-per-category",
        type=int,
        default=10,
        help="Maximum videos to record per category",
    )
    parser.add_argument(
        "--video-fps", type=int, default=30, help="Video framerate (default: 30)"
    )

    # Resumption
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from existing experiment directory",
    )

    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def create_experiment_directory(base_dir: Path, experiment_name: str) -> Path:
    """Create experiment directory with timestamp.

    Args:
        base_dir: Base output directory
        experiment_name: Experiment name

    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train_architecture(
    architecture_name: str,
    architecture_config,
    output_dir: Path,
    args,
    pretrained_checkpoint: str = None,
    condition_name: str = "",
) -> dict:
    """Train a single architecture configuration.

    Args:
        architecture_name: Architecture name
        architecture_config: Architecture configuration object
        output_dir: Output directory for this architecture
        args: Command-line arguments
        pretrained_checkpoint: Optional pretrained checkpoint path
        condition_name: Condition name (e.g., "with_pretrain")

    Returns:
        Training results dictionary
    """
    logger = logging.getLogger("npp_rl.training")

    condition_suffix = f" ({condition_name})" if condition_name else ""
    logger.info("=" * 70)
    logger.info(f"Training: {architecture_name}{condition_suffix}")
    logger.info("=" * 70)

    # Create condition-specific output directory
    if condition_name:
        arch_output_dir = output_dir / architecture_name / condition_name
    else:
        arch_output_dir = output_dir / architecture_name

    arch_output_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    tb_writer = TensorBoardManager(arch_output_dir / "tensorboard")

    try:
        # Prepare curriculum kwargs if enabled
        curriculum_kwargs = {}
        if args.use_curriculum:
            curriculum_kwargs = {
                "starting_stage": args.curriculum_start_stage,
                "advancement_threshold": args.curriculum_threshold,
                "min_episodes_per_stage": args.curriculum_min_episodes,
            }

        # Create trainer
        trainer = ArchitectureTrainer(
            architecture_config=architecture_config,
            train_dataset_path=args.train_dataset,
            test_dataset_path=args.test_dataset,
            output_dir=arch_output_dir,
            device_id=0,  # TODO: Support multi-GPU
            world_size=args.num_gpus,
            tensorboard_writer=tb_writer.get_writer("training"),
            use_mixed_precision=args.mixed_precision,
            use_hierarchical_ppo=args.use_hierarchical_ppo,
            use_curriculum=args.use_curriculum,
            curriculum_kwargs=curriculum_kwargs,
        )

        # Setup model
        trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

        # Setup environments (pass total_timesteps to allow adjustment for minimal training)
        trainer.setup_environments(
            num_envs=args.num_envs, total_timesteps=args.total_timesteps
        )

        # Train
        training_results = trainer.train(
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )

        # Evaluate (skip if requested)
        if args.skip_final_eval:
            logger.info("Skipping final evaluation (--skip-final-eval flag set)")
            eval_results = {"success_rate": 0.0, "level_types": {}, "skipped": True}
        else:
            # Setup video recording if enabled
            video_output_dir = None
            if args.record_eval_videos:
                video_output_dir = str(arch_output_dir / "videos")
                logger.info(f"Video recording enabled: {video_output_dir}")

            eval_results = trainer.evaluate(
                num_episodes=args.num_eval_episodes,
                record_videos=args.record_eval_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=args.max_videos_per_category,
                video_fps=args.video_fps,
            )

        tb_writer.close_all()

        return {
            "architecture": architecture_name,
            "condition": condition_name,
            "training": training_results,
            "evaluation": eval_results,
            "output_dir": str(arch_output_dir),
        }

    except Exception as e:
        logger.error(f"Training failed for {architecture_name}{condition_suffix}: {e}")
        tb_writer.close_all()
        return {
            "architecture": architecture_name,
            "condition": condition_name,
            "status": "failed",
            "error": str(e),
        }


def main():
    """Main execution function."""
    args = parse_args()

    # Create experiment directory
    base_output_dir = Path(args.output_dir)
    exp_dir = create_experiment_directory(base_output_dir, args.experiment_name)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_experiment_logging(
        output_dir=exp_dir, experiment_name=args.experiment_name, log_level=log_level
    )

    logger.info("=" * 70)
    logger.info("NPP-RL Training and Comparison Experiment")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info("=" * 70)

    # Save configuration
    config = vars(args)
    config["experiment_dir"] = str(exp_dir)
    config["start_time"] = datetime.now().isoformat()
    save_experiment_config(config, exp_dir)

    # Setup S3 uploader if requested
    s3_uploader = create_s3_uploader(
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
        experiment_name=args.experiment_name,
    )

    # Upload configuration
    if s3_uploader:
        s3_uploader.upload_file(str(exp_dir / "config.json"), "config.json")

    # Track all results
    all_results = []

    # Train each architecture
    for arch_name in args.architectures:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Processing architecture: {arch_name}")
        logger.info(f"{'=' * 70}\n")

        # Get architecture config
        try:
            arch_config = get_architecture_config(arch_name)
        except Exception as e:
            logger.error(f"Failed to load architecture config for '{arch_name}': {e}")
            continue

        # Determine pretraining conditions
        if args.no_pretraining:
            # No pretraining at all
            conditions = [("no_pretrain", None)]
        elif args.test_pretraining:
            # Test both conditions
            # Try to get pretrained checkpoint
            pretrained_ckpt = run_bc_pretraining_if_available(
                replay_data_dir=args.replay_data_dir,
                architecture_config=arch_config,
                output_dir=exp_dir / arch_name / "pretrain",
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
            )
            conditions = [("no_pretrain", None), ("with_pretrain", pretrained_ckpt)]
        else:
            # Try pretraining, use if available
            pretrained_ckpt = run_bc_pretraining_if_available(
                replay_data_dir=args.replay_data_dir,
                architecture_config=arch_config,
                output_dir=exp_dir / arch_name / "pretrain",
                epochs=args.bc_epochs,
                batch_size=args.bc_batch_size,
            )
            if pretrained_ckpt:
                conditions = [("with_pretrain", pretrained_ckpt)]
            else:
                conditions = [("no_pretrain", None)]

        # Train each condition
        for condition_name, pretrained_checkpoint in conditions:
            result = train_architecture(
                architecture_name=arch_name,
                architecture_config=arch_config,
                output_dir=exp_dir,
                args=args,
                pretrained_checkpoint=pretrained_checkpoint,
                condition_name=condition_name if args.test_pretraining else "",
            )
            all_results.append(result)

            # Upload to S3 if configured
            if s3_uploader and result.get("status") != "failed":
                output_dir = Path(result["output_dir"])

                # Upload checkpoints
                s3_uploader.upload_directory(
                    str(output_dir / "checkpoints"),
                    f"{arch_name}/{condition_name}/checkpoints",
                )

                # Upload videos if they exist
                videos_dir = output_dir / "videos"
                if videos_dir.exists() and videos_dir.is_dir():
                    logger.info(
                        f"Uploading videos to S3 for {arch_name}/{condition_name}"
                    )
                    s3_uploader.upload_directory(
                        str(videos_dir),
                        f"{arch_name}/{condition_name}/videos",
                        pattern="*.mp4",
                    )

    # Save all results
    results_file = exp_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("=" * 70)

    # Upload final manifest
    if s3_uploader:
        s3_uploader.save_manifest(str(exp_dir / "s3_manifest.json"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
