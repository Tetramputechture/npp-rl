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


import torch
import torch.multiprocessing as mp

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
from npp_rl.training.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    configure_cuda_for_training,
    barrier,
)
from npp_rl.training.runtime_profiler import create_profiler
from npp_rl.training.profiling_callback import RuntimeProfilingCallback
from stable_baselines3.common.callbacks import BaseCallback
import traceback
import gc
import os


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
        "--bc-epochs",
        type=int,
        default=10,  # Updated from 10 - but reduced from previous 50 to avoid overfitting
        help="Number of BC training epochs (light initialization, avoid overfitting)",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=128,  # Updated from 64 for faster convergence
        help="BC training batch size",
    )
    parser.add_argument(
        "--bc-num-workers",
        type=int,
        default=4,
        help="Number of workers for BC dataset processing",
    )

    # Training options
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=128,  # INCREASED from 64 with optimized hyperparameters
        help="Number of parallel environments (optimized for memory efficiency)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25000,
        help="Evaluation frequency (timesteps) - increased 4x for better monitoring",
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

    # Deep ResNet policy options (deprecated - automatically enabled for 'attention' architecture)
    parser.add_argument(
        "--use-deep-resnet-policy",
        action="store_true",
        help="Use deep ResNet policy with separate feature extractors (DEPRECATED: automatically enabled for 'attention' architecture)",
    )
    parser.add_argument(
        "--deep-resnet-use-residual",
        action="store_true",
        default=True,
        help="Use residual connections in deep ResNet policy (default: True)",
    )
    parser.add_argument(
        "--deep-resnet-use-layer-norm",
        action="store_true",
        default=True,
        help="Use LayerNorm in deep ResNet policy (default: True)",
    )
    parser.add_argument(
        "--deep-resnet-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for deep ResNet policy (default: 0.1)",
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
        default="simplest",
        choices=[
            "simplest",
            "simplest_few_mines",
            "simplest_with_mines",
            "simpler",
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
        default=0.8,
        help="Success rate threshold for curriculum advancement (lowered based on analysis)",
    )
    parser.add_argument(
        "--curriculum-min-episodes",
        type=int,
        default=50,
        help="Minimum episodes per curriculum stage",
    )

    # Reward shaping options
    parser.add_argument(
        "--pbrs-gamma",
        type=float,
        default=0.995,
        help="Discount factor for PBRS (always enabled, must match PPO gamma for policy invariance)",
    )

    # Curriculum safety options
    parser.add_argument(
        "--disable-trend-advancement",
        action="store_true",
        help="Disable trend-based curriculum advancement for more conservative progression",
    )
    parser.add_argument(
        "--disable-early-advancement",
        action="store_true",
        help="Disable early advancement even with high performance",
    )

    # Automatic curriculum adjustment (Week 3-4)
    parser.add_argument(
        "--enable-auto-curriculum-adjustment",
        action="store_true",
        help="Automatically reduce curriculum thresholds when agent stuck (5%% reduction, 40%% floor)",
    )
    parser.add_argument(
        "--curriculum-adjustment-freq",
        type=int,
        default=50000,
        help="Steps between automatic curriculum threshold checks",
    )
    parser.add_argument(
        "--curriculum-min-threshold",
        type=float,
        default=0.40,
        help="Minimum curriculum threshold floor for auto-adjustment",
    )

    # Early stopping options (Week 3-4)
    parser.add_argument(
        "--enable-early-stopping",
        action="store_true",
        help="Enable early stopping when training plateaus (default: 10 evals patience)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Number of evaluations without improvement before early stopping",
    )

    # Learning rate options
    parser.add_argument(
        "--enable-lr-annealing",
        action="store_true",
        help="Enable learning rate annealing (linear decay to 0) for better convergence",
    )
    parser.add_argument(
        "--initial-lr",
        type=float,
        default=None,
        help="Initial learning rate (if not using hardware profile). Default: 3e-4",
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

    # Frame stacking options
    parser.add_argument(
        "--enable-visual-frame-stacking",
        action="store_true",
        default=False,
        help="Enable frame stacking for visual observations (RECOMMENDED for temporal info)",
    )
    parser.add_argument(
        "--visual-stack-size",
        type=int,
        default=4,
        choices=[2, 3, 4, 6, 8, 12],
        help="Number of visual frames to stack (2-12)",
    )
    parser.add_argument(
        "--enable-state-stacking",
        action="store_true",
        help="Enable frame stacking for game state observations (RECOMMENDED: provides temporal context for physics)",
    )
    parser.add_argument(
        "--state-stack-size",
        type=int,
        default=4,
        choices=[2, 3, 4, 6, 8, 12],
        help="Number of game states to stack (2-12)",
    )
    parser.add_argument(
        "--frame-stack-padding",
        type=str,
        default="zero",
        choices=["zero", "repeat"],
        help="Padding type for initial frames",
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

    # Visualization options
    parser.add_argument(
        "--visualize-training",
        action="store_true",
        help="Enable real-time visualization of training",
    )
    parser.add_argument(
        "--vis-render-freq",
        type=int,
        default=100,
        help="Visualization render frequency (in timesteps)",
    )
    parser.add_argument(
        "--vis-env-idx",
        type=int,
        default=0,
        help="Which environment to visualize (from vectorized environments)",
    )
    parser.add_argument(
        "--vis-fps",
        type=int,
        default=60,
        help="Target FPS for visualization (0 = unlimited)",
    )

    # Debugging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Runtime profiling options
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable comprehensive runtime profiling",
    )
    parser.add_argument(
        "--enable-pytorch-profiler",
        action="store_true",
        default=True,
        help="Enable PyTorch profiler for detailed GPU/CPU traces (default: True if profiling enabled)",
    )
    parser.add_argument(
        "--profiler-wait",
        type=int,
        default=1,
        help="PyTorch profiler: steps to wait before profiling",
    )
    parser.add_argument(
        "--profiler-warmup",
        type=int,
        default=1,
        help="PyTorch profiler: steps to warmup before profiling",
    )
    parser.add_argument(
        "--profiler-active",
        type=int,
        default=3,
        help="PyTorch profiler: steps to actively profile",
    )
    parser.add_argument(
        "--profiler-repeat",
        type=int,
        default=1,
        help="PyTorch profiler: number of profiling cycles",
    )

    # Memory profiling options
    parser.add_argument(
        "--enable-memory-profiling",
        action="store_true",
        help="Enable comprehensive memory profiling (CPU + GPU tracking)",
    )
    parser.add_argument(
        "--memory-snapshot-freq",
        type=int,
        default=10000,
        help="Frequency of memory snapshots during training (in timesteps)",
    )

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


def upload_training_artifacts(
    s3_uploader, output_dir: Path, s3_prefix: str, logger
) -> None:
    """Upload all training artifacts for a single training run to S3.

    This uploads the complete set of training outputs including:
    - checkpoints/: All model checkpoints saved during training
    - final_model.zip: Final trained model
    - tensorboard/: All TensorBoard event files and logs
    - eval_results.json: Evaluation metrics and results
    - videos/*.mp4: Recorded evaluation videos
    - route_visualizations/*.png: Route visualization images
    - training_config.json: Training configuration
    - training_state_*.json: Training state snapshots

    Args:
        s3_uploader: S3Uploader instance
        output_dir: Local output directory for this training run
        s3_prefix: S3 prefix for this training run
        logger: Logger instance
    """
    logger.info(f"Uploading artifacts to S3 for {s3_prefix}")

    # 1. Upload checkpoints
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        count = s3_uploader.upload_directory(
            str(checkpoints_dir),
            f"{s3_prefix}/checkpoints",
        )
        logger.info(f"  ✓ Uploaded {count} checkpoint files")

    # 2. Upload final model
    final_model = output_dir / "final_model.zip"
    if final_model.exists():
        s3_uploader.upload_file(
            str(final_model),
            f"{s3_prefix}/final_model.zip",
        )
        logger.info("  ✓ Uploaded final model")

    # 3. Upload TensorBoard logs
    tensorboard_dir = output_dir / "tensorboard"
    if tensorboard_dir.exists() and tensorboard_dir.is_dir():
        count = s3_uploader.sync_tensorboard_logs(
            str(tensorboard_dir),
            f"{s3_prefix}/tensorboard",
        )
        logger.info(f"  ✓ Uploaded {count} TensorBoard event files")

    # 4. Upload evaluation results
    eval_results = output_dir / "eval_results.json"
    if eval_results.exists():
        s3_uploader.upload_file(
            str(eval_results),
            f"{s3_prefix}/eval_results.json",
        )
        logger.info("  ✓ Uploaded evaluation results")

    # 5. Upload videos
    videos_dir = output_dir / "videos"
    if videos_dir.exists() and videos_dir.is_dir():
        count = s3_uploader.upload_directory(
            str(videos_dir),
            f"{s3_prefix}/videos",
            pattern="*.mp4",
        )
        logger.info(f"  ✓ Uploaded {count} evaluation videos")

    # 6. Upload route visualizations
    routes_dir = output_dir / "route_visualizations"
    if routes_dir.exists() and routes_dir.is_dir():
        count = s3_uploader.upload_directory(
            str(routes_dir),
            f"{s3_prefix}/route_visualizations",
            pattern="*.png",
        )
        logger.info(f"  ✓ Uploaded {count} route visualization images")

    # 7. Upload training config
    training_config = output_dir / "training_config.json"
    if training_config.exists():
        s3_uploader.upload_file(
            str(training_config),
            f"{s3_prefix}/training_config.json",
        )
        logger.info("  ✓ Uploaded training config")

    # 8. Upload training state files
    for state_file in output_dir.glob("training_state_*.json"):
        s3_uploader.upload_file(
            str(state_file),
            f"{s3_prefix}/{state_file.name}",
        )
        logger.info(f"  ✓ Uploaded {state_file.name}")

    # 9. Upload profiling results if available
    profiling_dir = output_dir / "profiling"
    if profiling_dir.exists() and profiling_dir.is_dir():
        count = s3_uploader.upload_directory(
            str(profiling_dir),
            f"{s3_prefix}/profiling",
        )
        logger.info(f"  ✓ Uploaded {count} profiling files")

    logger.info(f"S3 upload complete for {s3_prefix}")


def upload_experiment_level_artifacts(
    s3_uploader,
    exp_dir: Path,
    experiment_name: str,
    architectures: list,
    results_file: Path,
    logger,
) -> None:
    """Upload experiment-level artifacts to S3.

    This uploads experiment-wide files including:
    - all_results.json: Aggregated results from all training runs
    - {experiment_name}.log: Main experiment log file
    - {arch}/pretrain/bc_checkpoint.pt: BC pretraining checkpoints
    - {arch}/pretrain/tensorboard/: Pretraining TensorBoard logs
    - {arch}/pretrain/*.json: Pretraining configuration files
    - {arch}/pretrain/*.log: Pretraining log files
    - s3_manifest.json: Manifest of all uploaded files

    Args:
        s3_uploader: S3Uploader instance
        exp_dir: Experiment directory
        experiment_name: Experiment name
        architectures: List of architecture names
        results_file: Path to all_results.json
        logger: Logger instance
    """
    logger.info("Uploading experiment-level artifacts to S3...")

    # Upload all_results.json
    s3_uploader.upload_file(
        str(results_file),
        "all_results.json",
    )
    logger.info("  ✓ Uploaded aggregated results")

    # Upload experiment log file
    log_file = exp_dir / f"{experiment_name}.log"
    if log_file.exists():
        s3_uploader.upload_file(
            str(log_file),
            f"{experiment_name}.log",
        )
        logger.info("  ✓ Uploaded experiment log file")

    # Upload pretraining artifacts if they exist
    for arch_name in architectures:
        pretrain_dir = exp_dir / arch_name / "pretrain"
        if pretrain_dir.exists() and pretrain_dir.is_dir():
            logger.info(f"Uploading pretraining artifacts for {arch_name}...")

            # Upload pretrain checkpoint
            pretrain_ckpt = pretrain_dir / "bc_checkpoint.pt"
            if pretrain_ckpt.exists():
                s3_uploader.upload_file(
                    str(pretrain_ckpt),
                    f"{arch_name}/pretrain/bc_checkpoint.pt",
                )
                logger.info(f"  ✓ Uploaded BC checkpoint for {arch_name}")

            # Upload pretrain tensorboard logs
            pretrain_tb = pretrain_dir / "tensorboard"
            if pretrain_tb.exists():
                count = s3_uploader.sync_tensorboard_logs(
                    str(pretrain_tb),
                    f"{arch_name}/pretrain/tensorboard",
                )
                logger.info(
                    f"  ✓ Uploaded {count} pretraining TensorBoard files for {arch_name}"
                )

            # Upload pretrain config/logs
            for pretrain_file in pretrain_dir.glob("*.json"):
                s3_uploader.upload_file(
                    str(pretrain_file),
                    f"{arch_name}/pretrain/{pretrain_file.name}",
                )
                logger.info(f"  ✓ Uploaded {pretrain_file.name} for {arch_name}")

            for pretrain_log in pretrain_dir.glob("*.log"):
                s3_uploader.upload_file(
                    str(pretrain_log),
                    f"{arch_name}/pretrain/{pretrain_log.name}",
                )
                logger.info(f"  ✓ Uploaded {pretrain_log.name} for {arch_name}")

    # Upload final manifest
    s3_uploader.save_manifest(str(exp_dir / "s3_manifest.json"))
    logger.info("  ✓ Uploaded S3 manifest")


class MemorySnapshotCallback(BaseCallback):
    """Periodic memory snapshots during training.

    Takes memory snapshots at regular intervals and compares them
    to detect memory leaks during training.
    """

    def __init__(self, profiler, freq=10000, verbose=0):
        """Initialize memory snapshot callback.

        Args:
            profiler: MemoryProfiler instance
            freq: Frequency of snapshots in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.profiler = profiler
        self.freq = freq

    def _on_step(self) -> bool:
        """Take snapshot if frequency reached."""
        if self.n_calls % self.freq == 0:
            self.profiler.snapshot(f"training_step_{self.n_calls}")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compare to previous snapshot if available
            if len(self.profiler.snapshots) >= 2:
                self.profiler.compare_snapshots(
                    self.profiler.snapshots[-2]["label"],
                    self.profiler.snapshots[-1]["label"],
                    top_n=10,
                )
        return True


def train_architecture(
    architecture_name: str,
    architecture_config,
    output_dir: Path,
    args,
    pretrained_checkpoint: str = None,
    condition_name: str = "",
    device_id: int = 0,
    hardware_profile=None,
    use_distributed: bool = False,
    profiler=None,
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
        hardware_profile: Optional hardware profile with optimized hyperparameters
        use_distributed: If True, enable DistributedDataParallel mode for multi-GPU

    Returns:
        Training results dictionary
    """
    logger = logging.getLogger("npp_rl.training")

    condition_suffix = f" ({condition_name})" if condition_name else ""
    logger.info("=" * 70)
    logger.info(f"Training: {architecture_name}{condition_suffix} on GPU {device_id}")
    logger.info("=" * 70)

    # AUTOMATIC DETECTION: Enable ObjectiveAttentionActorCriticPolicy for 'attention' architecture
    use_objective_attention_policy = architecture_name == "attention"
    if use_objective_attention_policy:
        logger.info("🎯 Detected 'attention' architecture")
        logger.info("   Automatically enabling: ObjectiveAttentionActorCriticPolicy")
        logger.info("   - Deep ResNet MLP (5-layer policy, 3-layer value)")
        logger.info("   - Objective-specific attention over 1-16 locked doors")
        logger.info("   - Dueling value architecture (always enabled)")
        logger.info("   - Residual connections + LayerNorm + SiLU")
        logger.info("   Total parameters: ~15-18M")

    # Create condition-specific output directory
    if condition_name:
        arch_output_dir = output_dir / architecture_name / condition_name
    else:
        arch_output_dir = output_dir / architecture_name

    arch_output_dir.mkdir(parents=True, exist_ok=True)

    # Create profiler for this architecture if enabled
    arch_profiler = None
    if profiler is None and args.enable_profiling:
        profiler_output_dir = arch_output_dir / "profiling"
        arch_profiler = create_profiler(
            output_dir=profiler_output_dir,
            enable=True,
            enable_pytorch_profiler=args.enable_pytorch_profiler,
            pytorch_profiler_wait=args.profiler_wait,
            pytorch_profiler_warmup=args.profiler_warmup,
            pytorch_profiler_active=args.profiler_active,
            pytorch_profiler_repeat=args.profiler_repeat,
        )
        logger.info(f"Runtime profiler enabled: output_dir={profiler_output_dir}")
    elif profiler is not None:
        arch_profiler = profiler

    # Create memory profiler if enabled
    mem_profiler = None
    if args.enable_memory_profiling:
        from npp_rl.utils.memory_profiler import MemoryProfiler

        mem_profiler = MemoryProfiler(
            output_dir=arch_output_dir,
            enable=True,
        )
        logger.info("Memory profiler enabled")

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
                "enable_auto_adjustment": args.enable_auto_curriculum_adjustment,
                "auto_adjustment_freq": args.curriculum_adjustment_freq,
                "auto_adjustment_min_threshold": args.curriculum_min_threshold,
            }

        # Calculate environments per GPU
        if args.num_gpus > 0:
            envs_per_gpu = args.num_envs // args.num_gpus
            logger.info(f"GPU {device_id}: Using {envs_per_gpu} environments")
        else:
            # CPU mode: use all environments on single device
            envs_per_gpu = args.num_envs
            logger.info(f"CPU mode: Using {envs_per_gpu} environments")

        # Build frame stacking configuration
        frame_stack_config = None
        if args.enable_visual_frame_stacking or args.enable_state_stacking:
            frame_stack_config = {
                "enable_visual_frame_stacking": args.enable_visual_frame_stacking,
                "visual_stack_size": args.visual_stack_size,
                "enable_state_stacking": args.enable_state_stacking,
                "state_stack_size": args.state_stack_size,
                "padding_type": args.frame_stack_padding,
            }
            logger.info(f"Frame stacking configuration: {frame_stack_config}")

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
            use_objective_attention_policy=use_objective_attention_policy,
            use_curriculum=args.use_curriculum,
            curriculum_kwargs=curriculum_kwargs,
            use_distributed=use_distributed,
            frame_stack_config=frame_stack_config,
            pbrs_gamma=args.pbrs_gamma,
            enable_early_stopping=args.enable_early_stopping,
            early_stopping_patience=args.early_stopping_patience,
            debug_mode=args.debug,
        )

        # Build PPO hyperparameters from hardware profile
        ppo_kwargs = {}
        if hardware_profile:
            logger.info(
                f"Applying hardware profile hyperparameters: {hardware_profile.name}"
            )
            ppo_kwargs["batch_size"] = hardware_profile.batch_size
            ppo_kwargs["n_steps"] = hardware_profile.n_steps

            # Learning rate: Apply annealing if requested
            base_lr = hardware_profile.learning_rate
            if args.enable_lr_annealing:
                # Linear annealing: lr(t) = initial_lr * (1 - t/T) where t=progress, T=1
                ppo_kwargs["learning_rate"] = lambda f: f * base_lr
                logger.info(f"  Learning rate: {base_lr:.2e} (with linear annealing)")
            else:
                ppo_kwargs["learning_rate"] = base_lr
                logger.info(f"  Learning rate: {base_lr:.2e} (constant)")

            logger.info(f"  Batch size: {ppo_kwargs['batch_size']}")
            logger.info(f"  N steps: {ppo_kwargs['n_steps']}")
        elif args.initial_lr is not None:
            # Manual learning rate provided
            base_lr = args.initial_lr
            if args.enable_lr_annealing:
                ppo_kwargs["learning_rate"] = lambda f: f * base_lr
                logger.info(f"Learning rate: {base_lr:.2e} (with linear annealing)")
            else:
                ppo_kwargs["learning_rate"] = base_lr
                logger.info(f"Learning rate: {base_lr:.2e} (constant)")

        # Add deep ResNet policy kwargs if enabled (or for attention architecture)
        if args.use_deep_resnet_policy or use_objective_attention_policy:
            ppo_kwargs["use_residual"] = args.deep_resnet_use_residual
            ppo_kwargs["use_layer_norm"] = args.deep_resnet_use_layer_norm
            ppo_kwargs["dropout"] = args.deep_resnet_dropout

            # Note: dueling is always enabled for ObjectiveAttentionActorCriticPolicy
            if not use_objective_attention_policy:
                ppo_kwargs["dueling"] = True  # Force dueling for consistency

            if not use_objective_attention_policy:
                # Only log for manual deep ResNet policy (attention arch already logged above)
                logger.info("Deep ResNet policy enabled:")
                logger.info(f"  Residual connections: {args.deep_resnet_use_residual}")
                logger.info(f"  LayerNorm: {args.deep_resnet_use_layer_norm}")
                logger.info(f"  Dropout: {args.deep_resnet_dropout}")

        # Setup model (with profiling)
        if mem_profiler:
            mem_profiler.snapshot("before_model_setup")
        if arch_profiler:
            arch_profiler.record_memory_snapshot("before_model_setup")
            with arch_profiler.phase(
                "model_setup", {"architecture": architecture_name}
            ):
                trainer.setup_model(
                    pretrained_checkpoint=pretrained_checkpoint, **ppo_kwargs
                )
            arch_profiler.record_memory_snapshot("after_model_setup")
        else:
            trainer.setup_model(
                pretrained_checkpoint=pretrained_checkpoint, **ppo_kwargs
            )
        if mem_profiler:
            mem_profiler.snapshot("after_model_setup")
            mem_profiler.compare_snapshots("before_model_setup", "after_model_setup")

        # Setup environments (use per-GPU environment count)
        if mem_profiler:
            mem_profiler.snapshot("before_env_setup")
        if arch_profiler:
            with arch_profiler.phase("environment_setup", {"num_envs": envs_per_gpu}):
                trainer.setup_environments(
                    num_envs=envs_per_gpu,
                    total_timesteps=args.total_timesteps,
                    enable_visualization=args.visualize_training,
                    vis_env_idx=args.vis_env_idx,
                )
            arch_profiler.record_memory_snapshot("after_env_setup")
        else:
            trainer.setup_environments(
                num_envs=envs_per_gpu,
                total_timesteps=args.total_timesteps,
                enable_visualization=args.visualize_training,
                vis_env_idx=args.vis_env_idx,
            )
        if mem_profiler:
            mem_profiler.snapshot("after_env_setup")
            mem_profiler.compare_snapshots("before_env_setup", "after_env_setup")

        # Create visualization callback if enabled
        vis_callback = None
        if args.visualize_training:
            from npp_rl.callbacks import TrainingVisualizationCallback

            vis_callback = TrainingVisualizationCallback(
                render_freq=args.vis_render_freq,
                render_mode="timesteps",
                env_idx=args.vis_env_idx,
                target_fps=args.vis_fps,
                window_title=f"NPP-RL Training: {architecture_name}",
                verbose=1 if args.debug else 0,
            )
            logger.info(
                f"Visualization enabled: rendering env {args.vis_env_idx} "
                f"every {args.vis_render_freq} timesteps at {args.vis_fps} FPS"
            )

        # Create profiling callback if enabled
        profiling_callback = None
        if arch_profiler:
            profiling_callback = RuntimeProfilingCallback(
                profiler=arch_profiler,
                log_freq=1000,
                verbose=1 if args.debug else 0,
            )
            logger.info("Runtime profiling callback enabled")

        # Combine callbacks (profiling callback should be first to track everything)
        combined_callback = None
        if profiling_callback and vis_callback:
            # Use profiling callback as base, it will handle both
            combined_callback = profiling_callback
            # Note: We can't easily chain callbacks, so we'll pass profiling
            # and let the callback factory handle visualization
            # For now, prioritize profiling
            combined_callback = profiling_callback
        elif profiling_callback:
            combined_callback = profiling_callback
        elif vis_callback:
            combined_callback = vis_callback

        # Add memory snapshot callback if enabled
        callbacks_list = [combined_callback] if combined_callback else []
        if mem_profiler:
            mem_callback = MemorySnapshotCallback(
                mem_profiler,
                freq=args.memory_snapshot_freq,
                verbose=1 if args.debug else 0,
            )
            callbacks_list.append(mem_callback)
            logger.info(
                f"Memory snapshot callback enabled (freq={args.memory_snapshot_freq})"
            )

        # Use CallbackList if multiple callbacks, otherwise single callback
        from stable_baselines3.common.callbacks import CallbackList

        final_callback = (
            CallbackList(callbacks_list)
            if len(callbacks_list) > 1
            else (callbacks_list[0] if callbacks_list else None)
        )

        # Train (with profiling)
        if mem_profiler:
            mem_profiler.snapshot("before_training")
        if arch_profiler:
            arch_profiler.record_memory_snapshot("before_training")
            with arch_profiler.phase(
                "training",
                {
                    "total_timesteps": args.total_timesteps,
                    "eval_freq": args.eval_freq,
                    "num_envs": envs_per_gpu,
                },
            ):
                training_results = trainer.train(
                    total_timesteps=args.total_timesteps,
                    eval_freq=args.eval_freq,
                    save_freq=args.save_freq,
                    callback_fn=final_callback,
                )
            arch_profiler.record_memory_snapshot("after_training")
        else:
            training_results = trainer.train(
                total_timesteps=args.total_timesteps,
                eval_freq=args.eval_freq,
                save_freq=args.save_freq,
                callback_fn=final_callback,
            )
        if mem_profiler:
            mem_profiler.snapshot("after_training")
            mem_profiler.compare_snapshots("before_training", "after_training")

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

            if arch_profiler:
                arch_profiler.record_memory_snapshot("before_evaluation")
                with arch_profiler.phase(
                    "evaluation", {"num_episodes": args.num_eval_episodes}
                ):
                    eval_results = trainer.evaluate(
                        num_episodes=args.num_eval_episodes,
                        record_videos=args.record_eval_videos,
                        video_output_dir=video_output_dir,
                        max_videos_per_category=args.max_videos_per_category,
                        video_fps=args.video_fps,
                    )
                arch_profiler.record_memory_snapshot("after_evaluation")
            else:
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

        # Save profiling results
        if arch_profiler:
            arch_profiler.print_summary()
            summary_path = arch_profiler.save_summary()
            logger.info(f"Profiling summary saved to {summary_path}")
            trace_path = arch_profiler.export_trace("training_trace")
            if trace_path:
                logger.info(f"PyTorch trace available in {trace_path}")

        # Save memory profiling results
        if mem_profiler:
            report_path = mem_profiler.save_report()
            logger.info(f"Memory profiling report saved to {report_path}")
            mem_profiler.cleanup()

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

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt)")
        # Save profiling data before exiting
        if arch_profiler:
            logger.info("Saving profiling data before exit...")
            try:
                arch_profiler.save_summary(force=True)
                if arch_profiler.enable_pytorch_profiler:
                    arch_profiler.stop_pytorch_profiler()
                logger.info("Profiling data saved successfully")
            except Exception as save_error:
                logger.error(f"Failed to save profiling data: {save_error}")
        # Clean up environments
        if "trainer" in locals():
            trainer.cleanup()
        if tb_writer:
            tb_writer.close_all()
        # Re-raise to allow normal cleanup
        raise
    except Exception as e:
        logger.error(f"Training failed for {architecture_name}{condition_suffix}: {e}")
        logger.error(traceback.format_exc())
        # Save profiling data even on failure
        if arch_profiler:
            logger.info("Saving profiling data after failure...")
            try:
                arch_profiler.save_summary(force=True)
                if arch_profiler.enable_pytorch_profiler:
                    arch_profiler.stop_pytorch_profiler()
            except Exception as save_error:
                logger.error(f"Failed to save profiling data: {save_error}")
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
    finally:
        # Ensure profiling data is saved even if something goes wrong
        if arch_profiler:
            try:
                if not arch_profiler._saved:
                    logger.info("Saving profiling data in finally block...")
                    arch_profiler.save_summary(force=True)
            except Exception as save_error:
                logger.error(f"Failed to save profiling data in finally: {save_error}")


def train_worker(
    rank: int,
    world_size: int,
    args,
    exp_dir: Path,
    hardware_profile,
    s3_bucket: str,
    s3_prefix: str,
    experiment_name: str,
):
    """Training worker for single GPU in distributed training setup.

    This function runs in a separate process for each GPU. It initializes
    distributed training, creates its own environments, and trains models
    using DistributedDataParallel for gradient synchronization.

    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        args: Parsed command-line arguments
        exp_dir: Experiment directory
        hardware_profile: Hardware profile (if any)
        s3_bucket: S3 bucket name (or None)
        s3_prefix: S3 prefix for uploads
        experiment_name: Name of experiment
    """
    try:
        setup_distributed(
            rank=rank, world_size=world_size, backend=args.distributed_backend
        )
        configure_cuda_for_training(rank)

        # Setup logging for this worker
        logger = logging.getLogger("npp_rl.training")

        if is_main_process():
            logger.info(
                f"[Rank {rank}] Main process - will handle logging/checkpointing/evaluation"
            )
        else:
            logger.info(f"[Rank {rank}] Worker process - training only, no I/O")

        # Setup S3 uploader (only on rank 0 to avoid conflicts)
        s3_uploader = None
        if is_main_process() and s3_bucket:
            s3_uploader = create_s3_uploader(
                bucket=s3_bucket,
                prefix=s3_prefix,
                experiment_name=experiment_name,
            )
            if s3_uploader:
                s3_uploader.upload_file(str(exp_dir / "config.json"), "config.json")

        # Track results (only on rank 0)
        all_results = []

        # Train each architecture
        for arch_name in args.architectures:
            logger.info(f"\n[Rank {rank}] {'=' * 70}")
            logger.info(f"[Rank {rank}] Processing architecture: {arch_name}")
            logger.info(f"[Rank {rank}] {'=' * 70}\n")

            # Get architecture config
            try:
                arch_config = get_architecture_config(arch_name)
            except Exception as e:
                print(
                    f"[Rank {rank}] Failed to load architecture config for '{arch_name}': {e}"
                )
                continue

            # Build frame stacking configuration for BC pretraining
            bc_frame_stack_config = None
            if args.enable_visual_frame_stacking or args.enable_state_stacking:
                bc_frame_stack_config = {
                    "enable_visual_frame_stacking": args.enable_visual_frame_stacking,
                    "visual_stack_size": args.visual_stack_size,
                    "enable_state_stacking": args.enable_state_stacking,
                    "state_stack_size": args.state_stack_size,
                    "padding_type": args.frame_stack_padding,
                }

            # Create profiler for pretraining if enabled
            pretrain_profiler = None
            if args.enable_profiling and is_main_process():
                pretrain_output_dir = exp_dir / arch_name / "pretrain" / "profiling"
                pretrain_profiler = create_profiler(
                    output_dir=pretrain_output_dir,
                    enable=True,
                    enable_pytorch_profiler=False,  # Disable PyTorch profiler for pretraining
                )

            # Determine pretraining conditions (only on rank 0 to avoid conflicts)
            if is_main_process():
                if args.no_pretraining:
                    conditions = [("no_pretrain", None)]
                elif args.test_pretraining:
                    if pretrain_profiler:
                        pretrain_profiler.record_memory_snapshot("before_pretraining")
                        with pretrain_profiler.phase(
                            "bc_pretraining",
                            {
                                "epochs": args.bc_epochs,
                                "batch_size": args.bc_batch_size,
                            },
                        ):
                            pretrained_ckpt = run_bc_pretraining_if_available(
                                replay_data_dir=args.replay_data_dir,
                                architecture_config=arch_config,
                                output_dir=exp_dir / arch_name / "pretrain",
                                epochs=args.bc_epochs,
                                batch_size=args.bc_batch_size,
                                frame_stack_config=bc_frame_stack_config,
                                tensorboard_writer=TensorBoardManager(
                                    exp_dir / arch_name / "pretrain" / "tensorboard"
                                ),
                                test_dataset_path=str(args.test_dataset),
                                dataset_num_workers=args.bc_num_workers,
                            )
                        pretrain_profiler.record_memory_snapshot("after_pretraining")
                        if pretrain_profiler:
                            pretrain_profiler.print_summary()
                            pretrain_profiler.save_summary()
                    else:
                        pretrained_ckpt = run_bc_pretraining_if_available(
                            replay_data_dir=args.replay_data_dir,
                            architecture_config=arch_config,
                            output_dir=exp_dir / arch_name / "pretrain",
                            epochs=args.bc_epochs,
                            batch_size=args.bc_batch_size,
                            frame_stack_config=bc_frame_stack_config,
                            tensorboard_writer=TensorBoardManager(
                                exp_dir / arch_name / "pretrain" / "tensorboard"
                            ),
                            test_dataset_path=str(args.test_dataset),
                            dataset_num_workers=args.bc_num_workers,
                        )
                    conditions = [
                        ("no_pretrain", None),
                        ("with_pretrain", pretrained_ckpt),
                    ]
                else:
                    if pretrain_profiler:
                        pretrain_profiler.record_memory_snapshot("before_pretraining")
                        with pretrain_profiler.phase(
                            "bc_pretraining",
                            {
                                "epochs": args.bc_epochs,
                                "batch_size": args.bc_batch_size,
                            },
                        ):
                            pretrained_ckpt = run_bc_pretraining_if_available(
                                replay_data_dir=args.replay_data_dir,
                                architecture_config=arch_config,
                                output_dir=exp_dir / arch_name / "pretrain",
                                epochs=args.bc_epochs,
                                batch_size=args.bc_batch_size,
                                frame_stack_config=bc_frame_stack_config,
                                tensorboard_writer=TensorBoardManager(
                                    exp_dir / arch_name / "pretrain" / "tensorboard"
                                ),
                                test_dataset_path=str(args.test_dataset),
                                dataset_num_workers=args.bc_num_workers,
                            )
                        pretrain_profiler.record_memory_snapshot("after_pretraining")
                        if pretrain_profiler:
                            pretrain_profiler.print_summary()
                            pretrain_profiler.save_summary()
                    else:
                        pretrained_ckpt = run_bc_pretraining_if_available(
                            replay_data_dir=args.replay_data_dir,
                            architecture_config=arch_config,
                            output_dir=exp_dir / arch_name / "pretrain",
                            epochs=args.bc_epochs,
                            batch_size=args.bc_batch_size,
                            frame_stack_config=bc_frame_stack_config,
                            tensorboard_writer=TensorBoardManager(
                                exp_dir / arch_name / "pretrain" / "tensorboard"
                            ),
                            test_dataset_path=str(args.test_dataset),
                            dataset_num_workers=args.bc_num_workers,
                        )
                    if pretrained_ckpt:
                        conditions = [("with_pretrain", pretrained_ckpt)]
                    else:
                        conditions = [("no_pretrain", None)]
            else:
                # Workers skip pretraining
                conditions = [("no_pretrain", None)]

            # Synchronize all processes before training
            barrier()

            # Train each condition
            for condition_name, pretrained_checkpoint in conditions:
                result = train_architecture(
                    architecture_name=arch_name,
                    architecture_config=arch_config,
                    output_dir=exp_dir,
                    args=args,
                    pretrained_checkpoint=pretrained_checkpoint,
                    condition_name=condition_name if args.test_pretraining else "",
                    device_id=rank,
                    hardware_profile=hardware_profile,
                    use_distributed=True,  # Enable DDP mode
                )

                # Only rank 0 collects results and uploads to S3
                if is_main_process():
                    all_results.append(result)

                    # Upload ALL artifacts to S3 if configured
                    if s3_uploader and result.get("status") != "failed":
                        output_dir = Path(result["output_dir"])
                        s3_prefix = (
                            f"{arch_name}/{condition_name}"
                            if condition_name
                            else arch_name
                        )
                        upload_training_artifacts(
                            s3_uploader, output_dir, s3_prefix, logger
                        )

                # Synchronize after each training run
                barrier()

        # Save all results (only rank 0)
        if is_main_process():
            results_file = exp_dir / "all_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            logger.info("\n" + "=" * 70)
            logger.info("Experiment complete!")
            logger.info(f"Results saved to: {exp_dir}")
            logger.info("=" * 70)

            # Upload final experiment-level artifacts
            if s3_uploader:
                upload_experiment_level_artifacts(
                    s3_uploader,
                    exp_dir,
                    experiment_name,
                    args.architectures,
                    results_file,
                    logger,
                )
                logger.info(
                    f"All artifacts uploaded to s3://{s3_bucket}/{s3_prefix}/{experiment_name}"
                )

    finally:
        # CRITICAL: Clean up distributed training
        cleanup_distributed()


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
            # Use first architecture for auto-detection if available
            # This provides architecture-aware memory estimation
            architecture_name = (
                args.architectures[0] if args.architectures else "full_hgt"
            )
            logger.info(
                f"Using architecture '{architecture_name}' for memory estimation"
            )
            hardware_profile = auto_detect_profile(architecture_name=architecture_name)
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

        # Configure GPU memory allocator to reduce fragmentation
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            logger.info("Configured PyTorch CUDA allocator: max_split_size_mb=512")

        # Clear GPU cache before training
        torch.cuda.empty_cache()
        logger.info("Cleared GPU cache")

        # Validate num_gpus argument
        if args.num_gpus > num_gpus:
            print(
                f"Requested {args.num_gpus} GPUs but only {num_gpus} available. "
                f"Using {num_gpus} GPUs."
            )
            args.num_gpus = num_gpus
    else:
        print("No GPUs available via torch.cuda.is_available()")
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
                print("nvidia-smi found GPUs but PyTorch cannot access them!")
                logger.info("nvidia-smi output (first 10 lines):")
                for line in nvidia_smi.stdout.split("\n")[:10]:
                    logger.info(f"  {line}")
            else:
                logger.info("nvidia-smi not available or failed")
        except Exception as e:
            logger.info(f"Could not run nvidia-smi: {e}")

        args.num_gpus = 0

    logger.info("=" * 70)

    # Validate configuration for BC pretraining with MLP baseline
    if "mlp_cnn" in args.architectures and args.replay_data_dir:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Configuration Validation for MLP Baseline")
        logger.info("=" * 70)

        # Check 2: Validate environment count
        if args.num_envs and args.num_envs < 128:
            print(f"⚠️  WARNING: Only {args.num_envs} environments specified")
            print(
                "   Recommendation: Use --num-envs 128 or higher for better data diversity "
                "(optimized hyperparameters enable 2x more environments)"
            )

        logger.info("=" * 70)
        logger.info("")

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

    # CRITICAL: Detect multi-GPU scenario and spawn workers
    if args.num_gpus > 1:
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-GPU TRAINING DETECTED")
        logger.info(f"Spawning {args.num_gpus} worker processes (one per GPU)")
        logger.info("Each worker will use distributed_utils for coordination")
        logger.info("Using DistributedDataParallel for gradient synchronization")
        logger.info("=" * 70 + "\n")

        # Spawn one process per GPU using torch.multiprocessing
        mp.spawn(
            train_worker,
            args=(
                args.num_gpus,  # world_size
                args,  # parsed args
                exp_dir,  # experiment directory
                hardware_profile,  # hardware profile
                args.s3_bucket,  # s3 bucket
                args.s3_prefix,  # s3 prefix
                args.experiment_name,  # experiment name
            ),
            nprocs=args.num_gpus,
            join=True,
        )

        logger.info("\n" + "=" * 70)
        logger.info("All GPU workers completed successfully")
        logger.info("=" * 70)

        return 0

    # Single GPU/CPU training (existing code path)
    logger.info("Single GPU/CPU training - no distributed coordination needed")

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
            print(f"Failed to load architecture config for '{arch_name}': {e}")
            continue

        # Build frame stacking configuration for BC pretraining
        bc_frame_stack_config = None
        if args.enable_visual_frame_stacking or args.enable_state_stacking:
            bc_frame_stack_config = {
                "enable_visual_frame_stacking": args.enable_visual_frame_stacking,
                "visual_stack_size": args.visual_stack_size,
                "enable_state_stacking": args.enable_state_stacking,
                "state_stack_size": args.state_stack_size,
                "padding_type": args.frame_stack_padding,
            }

        # Create profiler for pretraining if enabled
        pretrain_profiler = None
        if args.enable_profiling:
            pretrain_output_dir = exp_dir / arch_name / "pretrain" / "profiling"
            pretrain_profiler = create_profiler(
                output_dir=pretrain_output_dir,
                enable=True,
                enable_pytorch_profiler=False,  # Disable PyTorch profiler for pretraining
            )

        # Determine pretraining conditions
        if args.no_pretraining:
            # No pretraining at all
            conditions = [("no_pretrain", None)]
        elif args.test_pretraining:
            # Test both conditions
            # Try to get pretrained checkpoint
            if pretrain_profiler:
                pretrain_profiler.record_memory_snapshot("before_pretraining")
                with pretrain_profiler.phase(
                    "bc_pretraining",
                    {"epochs": args.bc_epochs, "batch_size": args.bc_batch_size},
                ):
                    pretrained_ckpt = run_bc_pretraining_if_available(
                        replay_data_dir=args.replay_data_dir,
                        architecture_config=arch_config,
                        output_dir=exp_dir / arch_name / "pretrain",
                        epochs=args.bc_epochs,
                        batch_size=args.bc_batch_size,
                        frame_stack_config=bc_frame_stack_config,
                        test_dataset_path=str(args.test_dataset),
                        dataset_num_workers=args.bc_num_workers,
                    )
                pretrain_profiler.record_memory_snapshot("after_pretraining")
                if pretrain_profiler:
                    pretrain_profiler.print_summary()
                    pretrain_profiler.save_summary()
            else:
                pretrained_ckpt = run_bc_pretraining_if_available(
                    replay_data_dir=args.replay_data_dir,
                    architecture_config=arch_config,
                    output_dir=exp_dir / arch_name / "pretrain",
                    epochs=args.bc_epochs,
                    batch_size=args.bc_batch_size,
                    frame_stack_config=bc_frame_stack_config,
                    test_dataset_path=str(args.test_dataset),
                    dataset_num_workers=args.bc_num_workers,
                )
            conditions = [("no_pretrain", None), ("with_pretrain", pretrained_ckpt)]
        else:
            # Try pretraining, use if available
            if pretrain_profiler:
                pretrain_profiler.record_memory_snapshot("before_pretraining")
                with pretrain_profiler.phase(
                    "bc_pretraining",
                    {"epochs": args.bc_epochs, "batch_size": args.bc_batch_size},
                ):
                    pretrained_ckpt = run_bc_pretraining_if_available(
                        replay_data_dir=args.replay_data_dir,
                        architecture_config=arch_config,
                        output_dir=exp_dir / arch_name / "pretrain",
                        epochs=args.bc_epochs,
                        batch_size=args.bc_batch_size,
                        frame_stack_config=bc_frame_stack_config,
                        test_dataset_path=str(args.test_dataset),
                        dataset_num_workers=args.bc_num_workers,
                    )
                pretrain_profiler.record_memory_snapshot("after_pretraining")
                if pretrain_profiler:
                    pretrain_profiler.print_summary()
                    pretrain_profiler.save_summary()
            else:
                pretrained_ckpt = run_bc_pretraining_if_available(
                    replay_data_dir=args.replay_data_dir,
                    architecture_config=arch_config,
                    output_dir=exp_dir / arch_name / "pretrain",
                    epochs=args.bc_epochs,
                    batch_size=args.bc_batch_size,
                    frame_stack_config=bc_frame_stack_config,
                    test_dataset_path=str(args.test_dataset),
                    dataset_num_workers=args.bc_num_workers,
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
                hardware_profile=hardware_profile,
            )
            all_results.append(result)

            # Upload ALL artifacts to S3 if configured
            if s3_uploader and result.get("status") != "failed":
                output_dir = Path(result["output_dir"])
                s3_prefix = (
                    f"{arch_name}/{condition_name}" if condition_name else arch_name
                )
                upload_training_artifacts(s3_uploader, output_dir, s3_prefix, logger)

    # Save all results
    results_file = exp_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("=" * 70)

    # Upload final experiment-level artifacts
    if s3_uploader:
        upload_experiment_level_artifacts(
            s3_uploader,
            exp_dir,
            args.experiment_name,
            args.architectures,
            results_file,
            logger,
        )
        logger.info(
            f"All artifacts uploaded to s3://{args.s3_bucket}/{args.s3_prefix}/{args.experiment_name}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
