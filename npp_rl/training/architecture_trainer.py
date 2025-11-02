"""Architecture trainer for single architecture training runs.

Handles training for a specific architecture configuration including
setup, training loop, evaluation, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from npp_rl.evaluation.comprehensive_evaluator import ComprehensiveEvaluator
from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
from npp_rl.training.curriculum_manager import create_curriculum_manager
from npp_rl.training.bc_weight_loader import BCWeightLoader
from npp_rl.training.environment_factory import EnvironmentFactory
from npp_rl.training.callback_factory import CallbackFactory

logger = logging.getLogger(__name__)


class ArchitectureTrainer:
    """Manages training for a specific architecture configuration."""

    def __init__(
        self,
        architecture_config: ArchitectureConfig,
        train_dataset_path: str,
        test_dataset_path: str,
        output_dir: Path,
        device_id: int = 0,
        world_size: int = 1,
        tensorboard_writer: Optional[SummaryWriter] = None,
        use_mixed_precision: bool = False,
        use_hierarchical_ppo: bool = False,
        use_curriculum: bool = False,
        curriculum_kwargs: Optional[Dict[str, Any]] = None,
        use_distributed: bool = False,
        frame_stack_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = 0.99,
        enable_mine_avoidance_reward: bool = True,
        use_icm: bool = False,
        icm_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize architecture trainer.

        Args:
            architecture_config: Architecture configuration
            train_dataset_path: Path to training dataset
            test_dataset_path: Path to test dataset
            output_dir: Output directory for checkpoints/logs
            device_id: GPU device ID
            world_size: Number of GPUs (for distributed training)
            tensorboard_writer: DEPRECATED - Not used. SB3's built-in tensorboard
                logging is used instead. Kept for backward compatibility.
            use_mixed_precision: Enable mixed precision training
            use_hierarchical_ppo: Use hierarchical PPO instead of standard PPO
            use_curriculum: Enable curriculum learning
            curriculum_kwargs: Curriculum manager configuration
            use_distributed: Enable DistributedDataParallel mode for multi-GPU
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str
            pbrs_gamma: Discount factor for PBRS (always enabled)
            enable_mine_avoidance_reward: Enable mine avoidance component in hierarchical rewards
            use_icm: Enable Intrinsic Curiosity Module (ICM)
            icm_config: ICM configuration dict (eta, alpha, etc.)
        """
        self.architecture_config = architecture_config
        self.train_dataset_path = Path(train_dataset_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.output_dir = Path(output_dir)
        self.device_id = device_id
        self.world_size = world_size
        self.tensorboard_writer = tensorboard_writer
        self.use_mixed_precision = use_mixed_precision
        self.use_hierarchical_ppo = use_hierarchical_ppo
        self.use_curriculum = use_curriculum
        self.curriculum_kwargs = curriculum_kwargs or {}
        self.use_distributed = use_distributed
        self.frame_stack_config = frame_stack_config or {}
        self.pbrs_gamma = pbrs_gamma
        self.enable_mine_avoidance_reward = enable_mine_avoidance_reward
        self.use_icm = use_icm
        self.icm_config = icm_config or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.curriculum_manager = None
        self.environment_factory = None
        self.callback_factory = None
        self.pretrained_checkpoint = None
        self.policy_kwargs = None
        self.policy_class = None
        self.hyperparams = None

        logger.info(f"Initialized trainer for architecture: {architecture_config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: cuda:{device_id}")
        logger.info(f"Hierarchical PPO: {use_hierarchical_ppo}")
        logger.info(f"Curriculum learning: {use_curriculum}")
        logger.info("PBRS: ALWAYS ENABLED (mandatory)")
        logger.info(f"  PBRS gamma: {pbrs_gamma}")
        logger.info(f"  Mine avoidance reward: {enable_mine_avoidance_reward}")
        if frame_stack_config:
            logger.info(f"Frame stacking: {frame_stack_config}")

    def setup_model(self, pretrained_checkpoint: Optional[str] = None, **ppo_kwargs):
        """Initialize model configuration from architecture config.

        Args:
            pretrained_checkpoint: Optional path to pretrained weights
            **ppo_kwargs: Additional PPO hyperparameters

        Returns:
            None (model will be created when environments are set up)
        """
        logger.info("Setting up model configuration...")

        # Store pretrained checkpoint path for later loading
        self.pretrained_checkpoint = pretrained_checkpoint

        # Set up policy kwargs with architecture config
        self.policy_kwargs = {
            "features_extractor_class": ConfigurableMultimodalExtractor,
            "features_extractor_kwargs": {
                "config": self.architecture_config,
                "frame_stack_config": self.frame_stack_config,
            },
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
            "optimizer_kwargs": {"eps": 1e-5},  # PPO standard from openai/baselines
        }

        # Set policy class based on hierarchical PPO flag
        if self.use_hierarchical_ppo:
            from npp_rl.agents.hierarchical_ppo import HierarchicalActorCriticPolicy

            self.policy_class = HierarchicalActorCriticPolicy

            # Add hierarchical PPO specific kwargs
            self.policy_kwargs.update(
                {
                    "high_level_update_frequency": ppo_kwargs.pop(
                        "high_level_update_frequency", 50
                    ),
                    "max_steps_per_subtask": ppo_kwargs.pop(
                        "max_steps_per_subtask", 500
                    ),
                    "use_icm": ppo_kwargs.pop("use_icm", True),
                }
            )
            logger.info(
                "Using HierarchicalActorCriticPolicy with hierarchical parameters"
            )
        else:
            self.policy_class = "MultiInputPolicy"

        # Configure hyperparameters
        self._configure_hyperparameters(ppo_kwargs)

        logger.info(
            f"Model configuration prepared for architecture: {self.architecture_config.name}"
        )
        logger.info("Model will be instantiated when environments are set up")

        return None

    def _configure_hyperparameters(self, ppo_kwargs: Dict[str, Any]) -> None:
        """Configure PPO hyperparameters with automatic scaling for multi-GPU.

        Args:
            ppo_kwargs: User-provided PPO hyperparameters
        """
        # Only apply automatic scaling if hyperparameters not explicitly provided
        if "batch_size" in ppo_kwargs and "learning_rate" in ppo_kwargs:
            # Use explicitly provided hyperparameters
            logger.info(
                "Using explicitly provided hyperparameters "
                "(automatic multi-GPU scaling skipped)"
            )
            default_batch_size = ppo_kwargs.get("batch_size", 256)
            default_learning_rate = ppo_kwargs.get("learning_rate", 3e-4)
        else:
            # Apply automatic scaling for multi-GPU DDP training
            base_batch_size = 256
            base_learning_rate = 3e-4

            if self.world_size > 1 and self.use_distributed:
                default_batch_size = base_batch_size * self.world_size
                default_learning_rate = base_learning_rate * (self.world_size**0.5)
                logger.info(
                    f"DDP hyperparameter scaling for {self.world_size} GPUs: "
                    f"batch_size={default_batch_size}, lr={default_learning_rate:.2e}"
                )
            else:
                default_batch_size = base_batch_size
                default_learning_rate = base_learning_rate

        # Default PPO hyperparameters
        default_hyperparams = {
            "learning_rate": default_learning_rate,
            "n_steps": 2048,
            "batch_size": default_batch_size,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": 1.0,  # Tighter clipping for value function stability
            "ent_coef": 0.001,  # Lower entropy for faster convergence
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": str(self.output_dir / "tensorboard"),
            "device": f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        }

        # Merge with provided hyperparameters
        self.hyperparams = {**default_hyperparams, **ppo_kwargs}

        # Log final hyperparameters
        logger.info("Final PPO hyperparameters:")
        logger.info(f"  Learning rate: {self.hyperparams['learning_rate']:.2e}")
        logger.info(f"  Batch size: {self.hyperparams['batch_size']}")
        logger.info(f"  N steps: {self.hyperparams['n_steps']}")
        logger.info(f"  Gamma: {self.hyperparams['gamma']}")
        logger.info(f"  GAE lambda: {self.hyperparams['gae_lambda']}")

    def setup_environments(
        self,
        num_envs: int = 64,
        total_timesteps: int = None,
        enable_visualization: bool = False,
        vis_env_idx: int = 0,
    ) -> None:
        """Create vectorized training and eval environments.

        Args:
            num_envs: Number of parallel environments
            total_timesteps: Total training timesteps (used to adjust n_steps for small runs)
            enable_visualization: If True, create one environment with human rendering
            vis_env_idx: Index of environment to enable visualization for
        """
        logger.info("=" * 60)
        logger.info(f"Setting up {num_envs} training environments...")
        logger.info(f"Frame stacking config: {self.frame_stack_config}")

        try:
            # Adjust n_steps if total_timesteps is very small (for CPU testing)
            if total_timesteps is not None:
                self._adjust_hyperparams_for_small_runs(total_timesteps, num_envs)

            # Set up curriculum manager if enabled
            if self.use_curriculum:
                self.curriculum_manager = create_curriculum_manager(
                    dataset_path=str(self.train_dataset_path), **self.curriculum_kwargs
                )
                logger.info("Curriculum learning enabled")
                logger.info(
                    f"Starting stage: {self.curriculum_manager.get_current_stage()}"
                )

            # Create environment factory
            logger.info("Creating environment factory...")
            self.environment_factory = EnvironmentFactory(
                use_curriculum=self.use_curriculum,
                curriculum_manager=self.curriculum_manager,
                frame_stack_config=self.frame_stack_config,
                pbrs_gamma=self.pbrs_gamma,
                enable_mine_avoidance_reward=self.enable_mine_avoidance_reward,
                output_dir=self.output_dir,
                pretrained_checkpoint=self.pretrained_checkpoint,
                enable_icm=self.use_icm,
                icm_config=self.icm_config,
            )

            # Create training environment
            logger.info(f"Creating {num_envs} training environments...")
            self.env = self.environment_factory.create_training_env(
                num_envs=num_envs,
                gamma=self.hyperparams.get("gamma", 0.99),
                enable_visualization=enable_visualization,
                vis_env_idx=vis_env_idx,
            )
            logger.info(f"✓ Training environment created: {type(self.env).__name__}")

            # Create evaluation environment
            logger.info("Creating evaluation environment...")
            self.eval_env = self.environment_factory.create_eval_env()
            logger.info(f"✓ Eval environment created: {type(self.eval_env).__name__}")

            # Create callback factory
            logger.info("Creating callback factory...")
            self.callback_factory = CallbackFactory(
                output_dir=self.output_dir,
                use_hierarchical_ppo=self.use_hierarchical_ppo,
                use_curriculum=self.use_curriculum,
                curriculum_manager=self.curriculum_manager,
                use_distributed=self.use_distributed,
                world_size=self.world_size,
            )

            # Create the model with the environment
            logger.info("Creating model with environment...")
            self._create_model()

            logger.info(f"✓ Model fully initialized with {num_envs} environments")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("!" * 60)
            logger.error("CRITICAL: Environment setup failed!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Num envs: {num_envs}")
            logger.error(f"Frame stack config: {self.frame_stack_config}")
            logger.error("Full traceback:", exc_info=True)
            logger.error("!" * 60)
            raise

    def _adjust_hyperparams_for_small_runs(
        self, total_timesteps: int, num_envs: int
    ) -> None:
        """Adjust hyperparameters for small test runs.

        Args:
            total_timesteps: Total training timesteps
            num_envs: Number of environments
        """
        max_n_steps = max(total_timesteps // num_envs, 1)
        if self.hyperparams["n_steps"] > max_n_steps:
            old_n_steps = self.hyperparams["n_steps"]
            self.hyperparams["n_steps"] = max_n_steps
            # Adjust batch_size to be compatible
            if self.hyperparams["batch_size"] > max_n_steps:
                self.hyperparams["batch_size"] = max_n_steps
            logger.info(
                f"Adjusted n_steps from {old_n_steps} to {max_n_steps} "
                f"for total_timesteps={total_timesteps}, num_envs={num_envs}"
            )
            logger.info(f"Adjusted batch_size to {self.hyperparams['batch_size']}")

    def _create_model(self) -> None:
        """Create PPO or HierarchicalPPO model instance."""
        logger.info("=" * 60)
        logger.info("Creating model...")

        try:
            if self.use_hierarchical_ppo:
                self._create_hierarchical_model()
            else:
                self._create_standard_model()

            logger.info(f"✓ Model is on device: {self.model.device}")
            logger.info("=" * 60)

            # Load pretrained weights if provided
            if self.pretrained_checkpoint:
                self._load_pretrained_weights()

            # Wrap policy with DistributedDataParallel if using multi-GPU
            if self.use_distributed and self.world_size > 1:
                self._wrap_model_ddp()

        except Exception as e:
            logger.error("!" * 60)
            logger.error("CRITICAL: Model creation failed!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Architecture: {self.architecture_config.name}")
            logger.error(f"Policy class: {self.policy_class}")
            logger.error(f"Device: {self.hyperparams.get('device', 'unknown')}")
            logger.error("Full traceback:", exc_info=True)
            logger.error("!" * 60)
            raise

    def _create_hierarchical_model(self) -> None:
        """Create HierarchicalPPO model."""
        from npp_rl.agents.hierarchical_ppo import HierarchicalPPO

        logger.info("Creating HierarchicalPPO model with training environment...")
        logger.info(f"Policy class: {self.policy_class}")
        logger.info(f"Device: {self.hyperparams.get('device')}")
        logger.info("Feature extractor: ConfigurableMultimodalExtractor")
        logger.info(f"Network architecture: {self.policy_kwargs.get('net_arch')}")
        logger.info(
            f"High-level update frequency: {self.policy_kwargs.get('high_level_update_frequency')}"
        )
        logger.info(
            f"Max steps per subtask: {self.policy_kwargs.get('max_steps_per_subtask')}"
        )
        logger.info(f"Using ICM: {self.policy_kwargs.get('use_icm')}")
        logger.info("Initializing hierarchical policy networks and moving to device...")

        hierarchical_ppo = HierarchicalPPO(
            policy_class=self.policy_class,
            high_level_update_frequency=self.policy_kwargs.get(
                "high_level_update_frequency", 50
            ),
            max_steps_per_subtask=self.policy_kwargs.get("max_steps_per_subtask", 500),
            use_icm=self.policy_kwargs.get("use_icm", True),
            policy_kwargs=self.policy_kwargs,
            **self.hyperparams,
        )

        self.model = hierarchical_ppo.create_model(env=self.env)
        logger.info("✓ HierarchicalPPO model created successfully")

    def _create_standard_model(self) -> None:
        """Create standard PPO model."""
        logger.info("Creating PPO model with training environment...")
        logger.info(f"Policy class: {self.policy_class}")
        logger.info(f"Device: {self.hyperparams.get('device')}")
        logger.info("Feature extractor: ConfigurableMultimodalExtractor")
        logger.info(f"Network architecture: {self.policy_kwargs.get('net_arch')}")
        logger.info("Initializing policy networks and moving to device...")

        self.model = PPO(
            policy=self.policy_class,
            env=self.env,
            policy_kwargs=self.policy_kwargs,
            **self.hyperparams,
        )
        logger.info("✓ PPO model created successfully")

        # Note: Mixed precision training (use_mixed_precision flag) is not yet
        # supported by stable-baselines3 PPO. To enable it, would require custom
        # PPO implementation with AMP integration. Current training uses FP32.
        if self.use_mixed_precision:
            logger.warning(
                "Mixed precision requested but not supported by stable-baselines3 PPO. "
                "Training will use FP32. To enable mixed precision, implement custom "
                "PPO with AMPHelper integration."
            )

    def _load_pretrained_weights(self) -> None:
        """Load BC pretrained weights into model."""
        logger.info("=" * 60)
        logger.info(f"Loading pretrained weights from {self.pretrained_checkpoint}")
        logger.info(f"Architecture: {self.architecture_config.name}")
        if self.frame_stack_config:
            logger.info(f"Frame stacking config: {self.frame_stack_config}")

        try:
            weight_loader = BCWeightLoader(
                model=self.model,
                architecture_name=self.architecture_config.name,
                frame_stack_config=self.frame_stack_config,
            )
            weight_loader.load_weights(self.pretrained_checkpoint)
            logger.info("=" * 60)
        except Exception as e:
            logger.error("!" * 60)
            logger.error("CRITICAL: Failed to load pretrained weights!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Checkpoint path: {self.pretrained_checkpoint}")
            logger.error(f"Architecture: {self.architecture_config.name}")
            logger.error(f"Frame stack config: {self.frame_stack_config}")
            logger.error("Full traceback:", exc_info=True)
            logger.error("!" * 60)
            logger.warning(
                "⚠️  Continuing with RANDOM INITIALIZATION - training will start from scratch!"
            )
            logger.warning("⚠️  BC pretraining benefits will be LOST!")
            logger.error("!" * 60)

    def _wrap_model_ddp(self) -> None:
        """Wrap model policy with DistributedDataParallel."""
        from npp_rl.training.distributed_utils import wrap_model_ddp

        logger.info("=" * 60)
        logger.info(
            f"Setting up DistributedDataParallel (DDP) for rank {self.device_id}/{self.world_size}"
        )
        logger.info("DDP will synchronize gradients across all GPUs during training")

        self.model.policy = wrap_model_ddp(
            self.model.policy,
            device_id=self.device_id,
            find_unused_parameters=False,
        )

        logger.info(f"✓ Policy wrapped with DDP on GPU {self.device_id}")
        logger.info("✓ Multi-GPU distributed training setup complete")
        logger.info("=" * 60)

    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 100000,
        save_freq: int = 500000,
        callback_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Main training loop with evaluation and checkpointing.

        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Evaluation frequency (timesteps)
            save_freq: Checkpoint save frequency (timesteps)
            callback_fn: Optional callback function

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first")

        if self.env is None:
            raise RuntimeError(
                "Environments not initialized. Call setup_environments() first"
            )

        self._log_training_start(total_timesteps, eval_freq, save_freq)

        try:
            # Create callbacks
            callbacks = self.callback_factory.create_callbacks(
                user_callback=callback_fn
            )

            logger.info("Starting model.learn() with callbacks...")

            # Only show progress bar on main process (rank 0)
            from npp_rl.training.distributed_utils import is_main_process

            show_progress = is_main_process() if self.use_distributed else True

            # Train model
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=show_progress,
            )

            logger.info("Training completed successfully")

            # Save final model
            self._save_final_model()

            return {"status": "completed", "total_timesteps": total_timesteps}

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            logger.error("=" * 60)
            logger.error("TRAINING FAILURE DETAILS:")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error(f"  Error message: {str(e)}")
            logger.error(f"  Total timesteps attempted: {total_timesteps}")
            logger.error(
                f"  Model device: {self.model.device if self.model else 'N/A'}"
            )
            logger.error(
                f"  Num environments: {self.env.num_envs if self.env else 'N/A'}"
            )
            logger.error("=" * 60)
            return {"status": "failed", "error": str(e), "error_type": type(e).__name__}

    def _log_training_start(
        self, total_timesteps: int, eval_freq: int, save_freq: int
    ) -> None:
        """Log training start information.

        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Evaluation frequency
            save_freq: Save frequency
        """
        logger.info("=" * 60)
        logger.info(f"Starting training: {self.architecture_config.name}")
        logger.info(f"Total timesteps: {total_timesteps:,}")

        # Add distributed training info
        if self.use_distributed and self.world_size > 1:
            from npp_rl.training.distributed_utils import is_main_process

            logger.info(f"Distributed training: {self.world_size} GPUs")
            logger.info(f"Current rank: {self.device_id}")
            logger.info(
                f"Effective batch size: {self.hyperparams.get('batch_size', 'N/A')} (per GPU)"
            )
            logger.info(
                f"Global batch size: {self.hyperparams.get('batch_size', 0) * self.world_size}"
            )
            if not is_main_process():
                logger.info("Worker process - progress bar disabled to avoid conflicts")

        logger.info(f"Eval frequency: {eval_freq:,}")
        logger.info(f"Save frequency: {save_freq:,}")
        logger.info("=" * 60)

        # Log model and environment details
        logger.info(f"Model device: {self.model.device}")
        logger.info(f"Number of environments: {self.env.num_envs}")
        logger.info(f"Policy architecture: {self.policy_class}")
        logger.info(
            f"PPO hyperparameters: n_steps={self.hyperparams.get('n_steps')}, "
            f"batch_size={self.hyperparams.get('batch_size')}, "
            f"learning_rate={self.hyperparams.get('learning_rate')}"
        )

        logger.info("Initializing environment reset...")
        logger.info(
            "Calling model.learn() - this will reset environments and start rollout collection"
        )
        logger.info(
            "First rollout collection may take time - collecting experience from all environments"
        )

    def _save_final_model(self) -> None:
        """Save final trained model."""
        final_path = self.output_dir / "final_model.zip"

        from npp_rl.training.distributed_utils import is_model_wrapped_ddp

        policy_was_wrapped = is_model_wrapped_ddp(self.model.policy)
        original_policy = self.model.policy

        if policy_was_wrapped:
            self.model.policy = self.model.policy.module
            logger.debug("Unwrapped DDP policy for final model saving")

        try:
            self.model.save(str(final_path))
            logger.info(f"Saved final model to {final_path}")
        finally:
            if policy_was_wrapped:
                self.model.policy = original_policy
                logger.debug("Re-wrapped policy with DDP after saving")

    def evaluate(
        self,
        num_episodes: int = 250,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
        timeout_per_episode: float = 200.0,
    ) -> Dict[str, float]:
        """Evaluate model on test dataset.

        Args:
            num_episodes: Number of episodes to evaluate (per category)
            record_videos: Whether to record videos of episodes
            video_output_dir: Directory to save videos (required if record_videos=True)
            max_videos_per_category: Maximum number of videos per category
            video_fps: Video framerate
            timeout_per_episode: Timeout in seconds per episode (default: 200.0)

        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Only evaluate on rank 0 to avoid conflicts
        if self.use_distributed:
            from npp_rl.training.distributed_utils import is_main_process

            if not is_main_process():
                logger.info(
                    f"[Rank {self.device_id}] Skipping evaluation (only rank 0 evaluates)"
                )
                return {"success_rate": 0.0, "skipped_on_worker": True}

        logger.info(
            f"Evaluating model on test suite ({num_episodes} episodes per category)..."
        )

        try:
            evaluator = ComprehensiveEvaluator(
                test_dataset_path=str(self.test_dataset_path),
                device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
            )

            # Create episodes per category dict
            num_episodes_per_category = {
                category: num_episodes for category in evaluator.test_levels.keys()
            }

            results = evaluator.evaluate_model(
                model=self.model,
                num_episodes_per_category=num_episodes_per_category,
                max_steps_per_episode=10000,
                deterministic=True,
                timeout_per_episode=timeout_per_episode,
                record_videos=record_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=max_videos_per_category,
                video_fps=video_fps,
                frame_stack_config=self.frame_stack_config,
            )

            # Save results
            results_path = self.output_dir / "eval_results.json"
            evaluator.save_results(results, str(results_path))

            logger.info("Evaluation complete")
            logger.info(f"Success rate: {results['overall']['success_rate']:.2%}")

            return results["overall"]

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"success_rate": 0.0, "error": str(e)}

    def cleanup(self) -> None:
        """Clean up environments and release resources."""
        if self.env is not None:
            try:
                self.env.close()
                logger.info("Closed training environments")
            except Exception as e:
                logger.warning(f"Error closing training environments: {e}")
            self.env = None

        if self.eval_env is not None:
            try:
                self.eval_env.close()
                logger.info("Closed evaluation environment")
            except Exception as e:
                logger.warning(f"Error closing evaluation environment: {e}")
            self.eval_env = None

    def save_checkpoint(self, timestep: int, is_final: bool = False) -> Path:
        """Save model checkpoint with metadata.

        Args:
            timestep: Current training timestep
            is_final: Whether this is the final checkpoint

        Returns:
            Path to saved checkpoint (or None on worker processes)
        """
        # Only save on rank 0 to avoid conflicts
        if self.use_distributed:
            from npp_rl.training.distributed_utils import is_main_process, barrier

            if not is_main_process():
                barrier()  # Wait for rank 0 to finish saving
                return None

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / "final_model.zip"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{timestep}.zip"

        # Unwrap DDP-wrapped policy before saving
        from npp_rl.training.distributed_utils import is_model_wrapped_ddp

        policy_was_wrapped = is_model_wrapped_ddp(self.model.policy)
        original_policy = self.model.policy

        if policy_was_wrapped:
            self.model.policy = self.model.policy.module
            logger.debug("Unwrapped DDP policy for checkpoint saving")

        try:
            self.model.save(str(checkpoint_path))
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        finally:
            if policy_was_wrapped:
                self.model.policy = original_policy
                logger.debug("Re-wrapped policy with DDP after saving")

        # Signal to other processes that save is complete
        if self.use_distributed:
            from npp_rl.training.distributed_utils import barrier

            barrier()

        return checkpoint_path

    def get_device(self) -> str:
        """Get the device string for this trainer.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        return "cpu"

    def get_checkpoint_path(self, name: str) -> Path:
        """Get path for a checkpoint file.

        Args:
            name: Checkpoint name (without extension)

        Returns:
            Path to checkpoint file
        """
        return self.output_dir / f"{name}.zip"

    def create_evaluator(self) -> ComprehensiveEvaluator:
        """Create comprehensive evaluator for this architecture.

        Returns:
            Configured ComprehensiveEvaluator instance
        """
        return ComprehensiveEvaluator(
            test_dataset_path=str(self.test_dataset_path),
            device=f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu",
        )

    def save_training_state(self, timestep: int) -> Path:
        """Save complete training state including curriculum progress.

        Args:
            timestep: Current training timestep

        Returns:
            Path to saved state file
        """
        import json
        from datetime import datetime

        state = {
            "timestep": timestep,
            "architecture": self.architecture_config.name,
            "timestamp": datetime.now().isoformat(),
        }

        # Add curriculum state if available
        if self.curriculum_manager is not None:
            state["curriculum"] = self.curriculum_manager.get_curriculum_state()

        # Save state file
        state_file = self.output_dir / f"training_state_{timestep}.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved training state to {state_file}")
        return state_file
