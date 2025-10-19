#!/usr/bin/env python3
"""Master training and comparison script for NPP-RL.

Orchestrates training, evaluation, and comparison of multiple architectures
with optional pretraining and multi-GPU support.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


import torch
import torch.distributed as dist

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
from npp_rl.training.hardware_profiles import (
    get_hardware_profile,
    auto_detect_profile,
    HARDWARE_PROFILES,
)


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

    # Multi-GPU and hardware profile options
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default=None,
        choices=list(HARDWARE_PROFILES.keys()) + ["auto"],
        help=(
            "Hardware profile for optimized settings. Use 'auto' for auto-detection. "
            "If specified, overrides --num-gpus, --num-envs, and batch size settings."
        ),
    )
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
        choices=[
            "very_simple",
            "simple",
            "medium",
            "complex",
            "exploration",
            "mine_heavy",
        ],
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


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    """Setup distributed training process group.

    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes (GPUs)
        backend: Distributed backend (nccl for GPU, gloo for CPU)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_architecture(
    architecture_name: str,
    architecture_config,
    output_dir: Path,
    args,
    pretrained_checkpoint: str = None,
    condition_name: str = "",
    device_id: int = 0,
) -> dict:
    """Train a single architecture configuration.

    Args:
        architecture_name: Architecture name
        architecture_config: Architecture configuration object
        output_dir: Output directory for this architecture
        args: Command-line arguments
        pretrained_checkpoint: Optional pretrained checkpoint path
        condition_name: Condition name (e.g., "with_pretrain")
        device_id: GPU device ID for this process

    Returns:
        Training results dictionary
    """
    logger = logging.getLogger("npp_rl.training")

    condition_suffix = f" ({condition_name})" if condition_name else ""
    logger.info("=" * 70)
    logger.info(f"Training: {architecture_name}{condition_suffix} on GPU {device_id}")
    logger.info("=" * 70)

    # Create condition-specific output directory
    if condition_name:
        arch_output_dir = output_dir / architecture_name / condition_name
    else:
        arch_output_dir = output_dir / architecture_name

    arch_output_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer (only on rank 0 to avoid conflicts)
    tb_writer = None
    if device_id == 0:
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

        # Calculate environments per GPU
        if args.num_gpus > 0:
            envs_per_gpu = args.num_envs // args.num_gpus
            logger.info(f"GPU {device_id}: Using {envs_per_gpu} environments")
        else:
            # CPU mode: use all environments on single device
            envs_per_gpu = args.num_envs
            logger.info(f"CPU mode: Using {envs_per_gpu} environments")

        # Create trainer
        trainer = ArchitectureTrainer(
            architecture_config=architecture_config,
            train_dataset_path=args.train_dataset,
            test_dataset_path=args.test_dataset,
            output_dir=arch_output_dir,
            device_id=device_id,
            world_size=args.num_gpus,
            tensorboard_writer=tb_writer.get_writer("training") if tb_writer else None,
            use_mixed_precision=args.mixed_precision,
            use_hierarchical_ppo=args.use_hierarchical_ppo,
            use_curriculum=args.use_curriculum,
            curriculum_kwargs=curriculum_kwargs,
        )

        # Setup model
        trainer.setup_model(pretrained_checkpoint=pretrained_checkpoint)

        # Setup environments (use per-GPU environment count)
        trainer.setup_environments(
            num_envs=envs_per_gpu, total_timesteps=args.total_timesteps
        )

        # Train
        training_results = trainer.train(
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
        )

        # Evaluate (skip if requested, or only run on rank 0 for multi-GPU)
        if args.skip_final_eval:
            logger.info("Skipping final evaluation (--skip-final-eval flag set)")
            eval_results = {"success_rate": 0.0, "level_types": {}, "skipped": True}
        elif device_id == 0:
            # Only evaluate on primary GPU to avoid conflicts
            logger.info("Running final evaluation on primary GPU...")
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
        else:
            logger.info(f"GPU {device_id}: Skipping evaluation (handled by rank 0)")
            eval_results = {
                "success_rate": 0.0,
                "level_types": {},
                "skipped_on_worker": True,
            }

        # Clean up environments
        trainer.cleanup()
        if tb_writer:
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
        # Clean up environments even on failure (if trainer was initialized)
        if "trainer" in locals():
            trainer.cleanup()
        if tb_writer:
            tb_writer.close_all()
        return {
            "architecture": architecture_name,
            "condition": condition_name,
            "device_id": device_id,
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

    # Apply hardware profile if specified
    hardware_profile = None
    if args.hardware_profile:
        if args.hardware_profile.lower() == "auto":
            logger.info("Auto-detecting hardware profile...")
            hardware_profile = auto_detect_profile()
        else:
            hardware_profile = get_hardware_profile(args.hardware_profile)

        if hardware_profile:
            logger.info(f"\nApplying hardware profile: {hardware_profile.name}")
            logger.info(f"Description: {hardware_profile.description}")

            # Override settings with profile values
            args.num_gpus = hardware_profile.num_gpus
            args.num_envs = hardware_profile.num_envs
            args.mixed_precision = hardware_profile.mixed_precision

            logger.info("Profile settings applied:")
            logger.info(f"  GPUs: {args.num_gpus}")
            logger.info(f"  Environments: {args.num_envs}")
            logger.info(f"  Batch size: {hardware_profile.batch_size}")
            logger.info(f"  Learning rate: {hardware_profile.learning_rate:.2e}")
            logger.info(f"  Mixed precision: {args.mixed_precision}")

    # Log GPU configuration with detailed diagnostics
    logger.info("\n" + "=" * 70)
    logger.info("CUDA/GPU Diagnostics")
    logger.info("=" * 70)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"GPU count: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = gpu_props.total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"    Compute capability: {gpu_props.major}.{gpu_props.minor}")

        # Validate num_gpus argument
        if args.num_gpus > num_gpus:
            logger.warning(
                f"Requested {args.num_gpus} GPUs but only {num_gpus} available. "
                f"Using {num_gpus} GPUs."
            )
            args.num_gpus = num_gpus
    else:
        logger.warning("No GPUs available via torch.cuda.is_available()")
        logger.info("Possible reasons:")
        logger.info("  1. PyTorch not built with CUDA support")
        logger.info("  2. CUDA drivers not installed or incompatible")
        logger.info("  3. Environment variable issues (check CUDA_VISIBLE_DEVICES)")

        # Try to provide more diagnostics
        import subprocess

        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if nvidia_smi.returncode == 0:
                logger.warning("nvidia-smi found GPUs but PyTorch cannot access them!")
                logger.info("nvidia-smi output (first 10 lines):")
                for line in nvidia_smi.stdout.split("\n")[:10]:
                    logger.info(f"  {line}")
            else:
                logger.info("nvidia-smi not available or failed")
        except Exception as e:
            logger.info(f"Could not run nvidia-smi: {e}")

        args.num_gpus = 0

    logger.info("=" * 70)

    # Save configuration
    config = vars(args)
    config["experiment_dir"] = str(exp_dir)
    config["start_time"] = datetime.now().isoformat()
    if hardware_profile:
        config["hardware_profile_used"] = hardware_profile.name
        config["hardware_profile_settings"] = {
            "num_gpus": hardware_profile.num_gpus,
            "gpu_memory_gb": hardware_profile.gpu_memory_gb,
            "num_envs": hardware_profile.num_envs,
            "batch_size": hardware_profile.batch_size,
            "n_steps": hardware_profile.n_steps,
            "learning_rate": hardware_profile.learning_rate,
        }
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
