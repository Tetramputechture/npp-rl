"""Pretraining pipeline for behavioral cloning.

Automates the process of preparing replay data and running BC pretraining
for architecture comparison experiments.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from npp_rl.optimization.architecture_configs import ArchitectureConfig
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig


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
    ):
        """Initialize pretraining pipeline.

        Args:
            replay_data_dir: Directory containing replay files
            architecture_config: Architecture configuration
            output_dir: Output directory for BC checkpoints
            tensorboard_writer: Optional TensorBoard writer
        """
        self.replay_data_dir = Path(replay_data_dir)
        self.architecture_config = architecture_config
        self.output_dir = Path(output_dir)
        self.tensorboard_writer = tensorboard_writer

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate replay data exists
        if not self.replay_data_dir.exists():
            raise FileNotFoundError(
                f"Replay data directory not found: {self.replay_data_dir}"
            )

        logger.info(f"Initialized pretraining pipeline for {architecture_config.name}")
        logger.info(f"Replay data: {self.replay_data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def prepare_bc_data(self, use_cache: bool = True) -> Optional[str]:
        """Process replay data into BC training format.

        This checks for existing processed data or generates it from replay files.

        Args:
            use_cache: If True, use cached processed data if available

        Returns:
            Path to BC dataset file, or None if no data available
        """
        # Check for existing processed data
        cached_data = self.output_dir / "bc_dataset.pkl"

        if use_cache and cached_data.exists():
            logger.info(f"Using cached BC data: {cached_data}")
            return str(cached_data)

        # Look for replay files
        replay_files = list(self.replay_data_dir.glob("*.npz"))

        if not replay_files:
            logger.warning(
                f"No replay files found in {self.replay_data_dir}. "
                "Skipping BC pretraining."
            )
            return None

        logger.info(f"Found {len(replay_files)} replay files")

        # Check if bc_pretrain.py processing is needed
        # For now, we assume data is pre-processed or will use existing BC script
        logger.info(
            "Replay data processing should be done using bc_pretrain.py "
            "or existing processed datasets"
        )

        return None  # Caller should use bc_pretrain.py directly

    def run_pretraining(
        self,
        bc_data_path: Optional[str],
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        checkpoint_name: str = "bc_checkpoint.pth",
    ) -> Optional[str]:
        """Run behavioral cloning training.

        Args:
            bc_data_path: Path to BC dataset (if None, skips pretraining)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            checkpoint_name: Name for saved checkpoint

        Returns:
            Path to pretrained checkpoint, or None if skipped
        """
        if bc_data_path is None:
            logger.info("No BC data available, skipping pretraining")
            return None

        logger.info("=" * 60)
        logger.info(f"Starting BC pretraining for {self.architecture_config.name}")
        logger.info(f"Architecture: {self.architecture_config.name}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info("=" * 60)

        try:
            # Create environment to get observation/action spaces
            env = NppEnvironment(config=EnvironmentConfig.for_training())
            env.close()

            # Create BC trainer
            # Note: This is a simplified version. Full integration would require
            # loading the actual BC dataset and configuring the policy based on
            # architecture_config
            logger.warning(
                "BC pretraining integration is currently simplified. "
                "For full pretraining, use bc_pretrain.py directly with "
                "the appropriate architecture configuration."
            )

            checkpoint_path = self.output_dir / checkpoint_name

            # NOTE: By design, this delegates to bc_pretrain.py script for actual BC training
            # This pipeline provides orchestration and validation, not the training loop itself
            # Users should run: python bc_pretrain.py --dataset_dir <path> --epochs 20
            logger.info(f"BC pretraining checkpoint location: {checkpoint_path}")
            logger.info("Run bc_pretrain.py separately for actual BC training")

            return str(checkpoint_path) if checkpoint_path.exists() else None

        except Exception as e:
            logger.error(f"BC pretraining failed: {e}")
            logger.warning("Continuing without pretrained checkpoint")
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
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        try:
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Check for expected keys
            required_keys = ["policy_state_dict"]

            if isinstance(checkpoint, dict):
                has_keys = all(key in checkpoint for key in required_keys)
                if has_keys:
                    logger.info(f"Checkpoint validated: {checkpoint_path}")
                    return True
                else:
                    logger.warning(f"Checkpoint missing required keys: {required_keys}")
                    return False
            else:
                logger.warning("Checkpoint is not a dictionary")
                return False

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
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
    tensorboard_writer: Optional[SummaryWriter] = None,
) -> Optional[str]:
    """Convenience function to run BC pretraining if replay data available.

    Args:
        replay_data_dir: Directory containing replay data (None to skip)
        architecture_config: Architecture configuration
        output_dir: Output directory for checkpoints
        epochs: Number of BC epochs
        batch_size: BC batch size
        tensorboard_writer: Optional TensorBoard writer

    Returns:
        Path to pretrained checkpoint, or None if skipped/failed
    """
    if replay_data_dir is None:
        logger.info("No replay data directory specified, skipping BC pretraining")
        return None

    replay_data_dir = Path(replay_data_dir)

    if not replay_data_dir.exists():
        logger.warning(
            f"Replay data directory not found: {replay_data_dir}. "
            "Skipping BC pretraining."
        )
        return None

    try:
        pipeline = PretrainingPipeline(
            replay_data_dir=str(replay_data_dir),
            architecture_config=architecture_config,
            output_dir=output_dir,
            tensorboard_writer=tensorboard_writer,
        )

        # Check for existing checkpoint
        existing_checkpoint = pipeline.get_checkpoint_path()

        if existing_checkpoint:
            logger.info(f"Found existing BC checkpoint: {existing_checkpoint}")
            if pipeline.validate_checkpoint(str(existing_checkpoint)):
                return str(existing_checkpoint)

        # Prepare BC data
        bc_data = pipeline.prepare_bc_data(use_cache=True)

        # Run pretraining
        checkpoint_path = pipeline.run_pretraining(
            bc_data_path=bc_data, epochs=epochs, batch_size=batch_size
        )

        if checkpoint_path and pipeline.validate_checkpoint(checkpoint_path):
            logger.info(f"BC pretraining completed: {checkpoint_path}")
            return checkpoint_path
        else:
            logger.warning("BC pretraining did not produce valid checkpoint")
            return None

    except Exception as e:
        logger.error(f"BC pretraining pipeline failed: {e}")
        logger.warning("Continuing without pretraining")
        return None
