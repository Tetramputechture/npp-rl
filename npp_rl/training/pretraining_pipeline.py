"""Pretraining pipeline for behavioral cloning.

Automates the process of preparing replay data and running BC pretraining
for architecture comparison experiments.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from npp_rl.training.architecture_configs import ArchitectureConfig
from npp_rl.training.bc_dataset import BCReplayDataset
from npp_rl.training.bc_trainer import BCTrainer


logger = logging.getLogger(__name__)


class PretrainingPipeline:
    """Manages the full pretraining pipeline from replay data to checkpoint.

    Workflow:
    1. Validate replay data directory
    2. Generate BC training data (if needed)
    3. Run behavioral cloning training
    4. Validate pretrained model
    5. Save checkpoint for RL fine-tuning
    """

    def __init__(
        self,
        replay_data_dir: str,
        architecture_config: ArchitectureConfig,
        output_dir: Path,
        tensorboard_writer: Optional[SummaryWriter] = None,
        frame_stack_config: Optional[Dict] = None,
        test_dataset_path: Optional[str] = None,
        checkpoint_frame_stack_config: Optional[Dict] = None,
    ):
        """Initialize pretraining pipeline.

        Args:
            replay_data_dir: Directory containing replay files
            architecture_config: Architecture configuration
            output_dir: Output directory for BC checkpoints
            tensorboard_writer: Optional TensorBoard writer
            frame_stack_config: Frame stacking configuration dict for training with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str ('zero' or 'repeat')
            test_dataset_path: Path to test dataset
            checkpoint_frame_stack_config: Frame stacking config to save in checkpoint.
                If None, uses frame_stack_config. This allows saving the original
                RL config even when BC training uses a modified config (e.g., disabled
                state stacking for AttentiveStateMLP architectures).
        """
        self.replay_data_dir = Path(replay_data_dir)
        self.architecture_config = architecture_config
        self.output_dir = Path(output_dir)
        self.tensorboard_writer = tensorboard_writer
        self.frame_stack_config = frame_stack_config or {}
        self.test_dataset_path = test_dataset_path
        self.checkpoint_frame_stack_config = (
            checkpoint_frame_stack_config or self.frame_stack_config
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate replay data exists
        if not self.replay_data_dir.exists():
            raise FileNotFoundError(
                f"Replay data directory not found: {self.replay_data_dir}"
            )

        logger.info(f"Initialized pretraining pipeline for {architecture_config.name}")
        logger.info(f"Replay data: {self.replay_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Log frame stacking configuration
        if frame_stack_config:
            logger.info("Frame stacking configuration:")
            logger.info(
                f"  Visual: {frame_stack_config.get('enable_visual_frame_stacking', False)} "
                f"(size: {frame_stack_config.get('visual_stack_size', 4)})"
            )
            logger.info(
                f"  State: {frame_stack_config.get('enable_state_stacking', False)} "
                f"(size: {frame_stack_config.get('state_stack_size', 4)})"
            )
            logger.info(f"  Padding: {frame_stack_config.get('padding_type', 'zero')}")

    def prepare_bc_data(
        self,
        use_cache: bool = True,
        max_replays: Optional[int] = None,
        filter_successful_only: bool = True,
        num_workers: Optional[int] = None,
    ) -> Optional[BCReplayDataset]:
        """Process replay data into BC training format.

        Creates a BCReplayDataset from replay files, with optional caching.

        Args:
            use_cache: If True, use cached processed data if available
            max_replays: Maximum number of replays to load (None for all)
            filter_successful_only: Only include successful replays
            num_workers: Number of parallel workers for processing replays.
                If None, auto-detects to min(len(replay_files), 4).
                Set to 1 for sequential processing.

        Returns:
            BCReplayDataset instance, or None if no data available
        """
        # Look for replay files
        replay_files = list(self.replay_data_dir.glob("*.replay"))

        if not replay_files:
            print(
                f"No .replay files found in {self.replay_data_dir}. "
                "Skipping BC pretraining."
            )
            return None

        logger.info(f"Found {len(replay_files)} replay files")

        try:
            # Create BC dataset with architecture config and frame stacking
            dataset = BCReplayDataset(
                replay_dir=str(self.replay_data_dir),
                cache_dir=str(self.output_dir / "cache"),
                use_cache=use_cache,
                filter_successful_only=filter_successful_only,
                max_replays=max_replays,
                architecture_config=self.architecture_config,
                frame_stack_config=self.frame_stack_config,
                num_workers=num_workers,
            )

            if len(dataset) == 0:
                print("No training samples generated from replays")
                return None

            logger.info(f"BC dataset ready with {len(dataset)} training samples")
            return dataset

        except Exception:
            # print(f"Failed to create BC dataset: {e}", exc_info=True)
            return None

    def run_pretraining(
        self,
        bc_dataset: Optional[BCReplayDataset],
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        num_workers: int = 4,
        device: str = "auto",
    ) -> Optional[str]:
        """Run behavioral cloning training.

        Args:
            bc_dataset: BC replay dataset (if None, skips pretraining)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_workers: Number of data loading workers
            device: Device to train on ('auto', 'cpu', 'cuda')

        Returns:
            Path to pretrained checkpoint, or None if skipped
        """
        if bc_dataset is None:
            logger.info("No BC dataset available, skipping pretraining")
            return None

        logger.info("=" * 60)
        logger.info(f"Starting BC pretraining for {self.architecture_config.name}")
        logger.info(f"Architecture: {self.architecture_config.name}")
        logger.info(f"Dataset size: {len(bc_dataset)} samples")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info("=" * 60)

        try:
            # Create BC trainer
            trainer = BCTrainer(
                architecture_config=self.architecture_config,
                dataset=bc_dataset,
                output_dir=str(self.output_dir),
                device=device,
                validation_split=0.1,
                tensorboard_writer=self.tensorboard_writer,
                frame_stack_config=self.frame_stack_config,
                test_dataset_path=self.test_dataset_path,
                checkpoint_frame_stack_config=self.checkpoint_frame_stack_config,
            )

            # Run training
            best_checkpoint_path = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_workers=num_workers,
                save_frequency=5,
                early_stopping_patience=5,
            )

            logger.info(f"BC pretraining completed: {best_checkpoint_path}")
            return best_checkpoint_path

        except Exception:
            # print(f"BC pretraining failed: {e}", exc_info=True)
            print("Continuing without pretrained checkpoint")
            return None

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """Validate pretrained checkpoint loads correctly.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        if checkpoint_path is None:
            return False

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # Try to load checkpoint
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            # Check for expected keys
            required_keys = ["policy_state_dict"]

            if isinstance(checkpoint, dict):
                has_keys = all(key in checkpoint for key in required_keys)
                if has_keys:
                    logger.info(f"Checkpoint validated: {checkpoint_path}")
                    return True
                else:
                    print(f"Checkpoint missing required keys: {required_keys}")
                    return False
            else:
                print("Checkpoint is not a dictionary")
                return False

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def get_checkpoint_path(self) -> Optional[Path]:
        """Get path to BC checkpoint if it exists.

        Returns:
            Path to checkpoint or None if not found
        """
        checkpoint_path = self.output_dir / "bc_checkpoint.pth"

        if checkpoint_path.exists():
            return checkpoint_path

        # Check for alternative names
        for alt_name in ["bc_best.pth", "checkpoint.pth", "model.pth"]:
            alt_path = self.output_dir / alt_name
            if alt_path.exists():
                return alt_path

        return None


def run_bc_pretraining_if_available(
    replay_data_dir: Optional[str],
    architecture_config: ArchitectureConfig,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    num_workers: int = 4,
    device: str = "auto",
    max_replays: Optional[int] = None,
    tensorboard_writer: Optional[SummaryWriter] = None,
    frame_stack_config: Optional[Dict] = None,
    test_dataset_path: Optional[str] = None,
    dataset_num_workers: Optional[int] = None,
) -> Optional[str]:
    """Convenience function to run BC pretraining if replay data available.

    Args:
        replay_data_dir: Directory containing replay data (None to skip)
        architecture_config: Architecture configuration
        output_dir: Output directory for checkpoints
        epochs: Number of BC epochs
        batch_size: BC batch size
        learning_rate: Learning rate
        num_workers: Number of data loading workers (for DataLoader)
        device: Device to train on
        max_replays: Maximum number of replays to use (None for all)
        tensorboard_writer: Optional TensorBoard writer
        frame_stack_config: Frame stacking configuration dict
        test_dataset_path: Path to test dataset
        dataset_num_workers: Number of parallel workers for processing replays.
            If None, auto-detects to min(len(replay_files), 4).
            Set to 1 for sequential processing.

    Returns:
        Path to pretrained checkpoint, or None if skipped/failed
    """
    if replay_data_dir is None:
        logger.info("No replay data directory specified, skipping BC pretraining")
        return None

    replay_data_dir = Path(replay_data_dir)

    if not replay_data_dir.exists():
        print(
            f"Replay data directory not found: {replay_data_dir}. "
            "Skipping BC pretraining."
        )
        return None

    try:
        # Use the same frame stacking config for BC as RL
        # AttentiveStateMLP handles stacked states by extracting the most recent frame
        pipeline = PretrainingPipeline(
            replay_data_dir=str(replay_data_dir),
            architecture_config=architecture_config,
            output_dir=output_dir,
            tensorboard_writer=tensorboard_writer,
            frame_stack_config=frame_stack_config,
            test_dataset_path=test_dataset_path,
            checkpoint_frame_stack_config=frame_stack_config,  # Same config for BC and RL
        )

        # Check for existing checkpoint
        existing_checkpoint = pipeline.get_checkpoint_path()

        if existing_checkpoint:
            logger.info(f"Found existing BC checkpoint: {existing_checkpoint}")
            if pipeline.validate_checkpoint(str(existing_checkpoint)):
                return str(existing_checkpoint)

        # Prepare BC data
        bc_dataset = pipeline.prepare_bc_data(
            use_cache=True,
            max_replays=max_replays,
            filter_successful_only=True,
            num_workers=dataset_num_workers,
        )

        # Run pretraining
        checkpoint_path = pipeline.run_pretraining(
            bc_dataset=bc_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=num_workers,
            device=device,
        )

        if checkpoint_path and pipeline.validate_checkpoint(checkpoint_path):
            logger.info(f"BC pretraining completed: {checkpoint_path}")
            return checkpoint_path
        else:
            print("BC pretraining did not produce valid checkpoint")
            return None

    except Exception as e:
        print(f"BC pretraining pipeline failed: {e}", exc_info=True)
        print("Continuing without pretraining")
        return None
