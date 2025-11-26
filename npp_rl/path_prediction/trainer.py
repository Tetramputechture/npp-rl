"""Trainer for path predictor network with offline and online learning modes.

This module provides:
1. Offline training from expert replay demonstrations
2. Online fine-tuning during RL agent training
3. Integration with pattern extraction and graph validation
"""

import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .multipath_predictor import ProbabilisticPathPredictor, CandidatePath
from .pattern_extractor import GeneralizedPatternExtractor
from .training_losses import compute_path_prediction_loss
from .graph_observation_utils import extract_graph_observation_batch
from ..path_prediction.replay_dataset import PathReplayDataset, collate_replay_batch

logger = logging.getLogger(__name__)


class PathPredictorTrainer:
    """Trainer for ProbabilisticPathPredictor with offline and online modes.

    Offline Mode:
    - Trains from expert replay demonstrations
    - Learns to predict waypoint sequences
    - Builds pattern database from demonstrations

    Online Mode:
    - Fine-tunes predictor during RL training
    - Updates path preferences based on outcomes
    - Adapts to agent's discovered paths
    """

    def __init__(
        self,
        predictor: ProbabilisticPathPredictor,
        pattern_extractor: Optional[GeneralizedPatternExtractor] = None,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize path predictor trainer.

        Args:
            predictor: ProbabilisticPathPredictor model
            pattern_extractor: Optional pattern extractor for building database
            learning_rate: Learning rate for optimizer
            device: Device for training
            loss_weights: Weights for loss components
        """
        self.predictor = predictor.to(device)
        self.pattern_extractor = pattern_extractor
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.predictor.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Loss weights
        self.loss_weights = loss_weights or {
            "waypoint": 1.0,
            "confidence": 0.5,
            "diversity": 0.3,
        }

        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_trained = 0

        logger.info(f"Initialized PathPredictorTrainer on {device}")
        logger.info(f"Loss weights: {self.loss_weights}")

    def train_offline(
        self,
        train_dataset: PathReplayDataset,
        val_dataset: Optional[PathReplayDataset] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        num_workers: int = 4,
        save_dir: Optional[str] = None,
        save_freq: int = 10,
    ) -> Dict[str, Any]:
        """Train predictor offline from replay demonstrations.

        Args:
            train_dataset: Training replay dataset
            val_dataset: Validation replay dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            num_workers: Number of dataloader workers
            save_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N epochs

        Returns:
            Training statistics dictionary
        """
        logger.info(f"Starting offline training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_replay_batch,
            pin_memory=True,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_replay_batch,
                pin_memory=True,
            )

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'=' * 60}")

            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            logger.info(f"Train Loss: {train_loss:.4f}")

            # Validation epoch
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                logger.info(f"Val Loss: {val_loss:.4f}")

                # Update learning rate
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if save_dir:
                        self._save_checkpoint(
                            save_dir,
                            epoch,
                            is_best=True,
                        )
                    logger.info(f"New best validation loss: {val_loss:.4f}")

            # Periodic checkpoint
            if save_dir and (epoch + 1) % save_freq == 0:
                self._save_checkpoint(save_dir, epoch, is_best=False)

            self.epochs_trained += 1

        # Save final checkpoint
        if save_dir:
            logger.info("\nSaving final checkpoint...")
            self._save_checkpoint(
                save_dir, num_epochs - 1, is_best=False, is_final=True
            )

        logger.info("\nTraining complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
        }

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            Average loss for epoch
        """
        self.predictor.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Skip empty batches
            if batch["batch_size"] == 0:
                continue

            # Forward pass
            loss_dict = self._compute_batch_loss(batch)

            if loss_dict is None:
                continue

            loss = loss_dict["total_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "waypoint": f"{loss_dict['waypoint_loss'].item():.4f}",
                    "confidence": f"{loss_dict['confidence_loss'].item():.4f}",
                    "diversity": f"{loss_dict['diversity_loss'].item():.4f}",
                }
            )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch.

        Args:
            dataloader: Validation dataloader

        Returns:
            Average validation loss
        """
        self.predictor.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Skip empty batches
                if batch["batch_size"] == 0:
                    continue

                # Forward pass
                loss_dict = self._compute_batch_loss(batch)

                if loss_dict is None:
                    continue

                loss = loss_dict["total_loss"]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _compute_batch_loss(
        self, batch: Dict[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Compute loss for a batch.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Dictionary with loss components or None if batch invalid
        """
        # Extract features from batch
        tile_patterns = batch["tile_patterns"].to(self.device)
        entity_features = batch["entity_features"].to(self.device)
        expert_waypoints = batch["expert_waypoints_tensor"].to(self.device)
        waypoint_masks = batch["waypoint_masks"].to(self.device)

        # Extract graph observations from batch using consistent utility
        # graph_obs is a list of dicts with node/edge features
        graph_obs_list = batch["graph_obs"]

        # Use centralized graph observation extraction for consistency
        graph_obs = extract_graph_observation_batch(
            graph_obs_list, target_dim=256, device=self.device
        )

        # DEFENSIVE: Validate input features before forward pass
        if torch.isnan(graph_obs).any() or torch.isinf(graph_obs).any():
            logger.warning("NaN/Inf detected in graph_obs input features")
            logger.warning(
                f"  graph_obs stats: min={graph_obs.min()}, max={graph_obs.max()}"
            )
            return None

        if torch.isnan(tile_patterns).any() or torch.isinf(tile_patterns).any():
            logger.warning("NaN/Inf detected in tile_patterns input features")
            logger.warning(
                f"  tile_patterns stats: min={tile_patterns.min()}, max={tile_patterns.max()}"
            )
            return None

        if torch.isnan(entity_features).any() or torch.isinf(entity_features).any():
            logger.warning("NaN/Inf detected in entity_features input features")
            logger.warning(
                f"  entity_features stats: min={entity_features.min()}, max={entity_features.max()}"
            )
            return None

        # Forward pass through predictor
        outputs = self.predictor(
            graph_obs=graph_obs,
            tile_patterns=tile_patterns,
            entity_features=entity_features,
        )

        # DEFENSIVE: Check outputs for NaN/Inf after forward pass
        predicted_waypoints = outputs["waypoints"]
        if (
            torch.isnan(predicted_waypoints).any()
            or torch.isinf(predicted_waypoints).any()
        ):
            logger.warning("NaN/Inf detected in model outputs (predicted_waypoints)")
            logger.warning(
                f"  predicted_waypoints stats: min={predicted_waypoints.min()}, max={predicted_waypoints.max()}"
            )
            logger.warning("  Input feature stats:")
            logger.warning(
                f"    graph_obs: min={graph_obs.min()}, max={graph_obs.max()}, mean={graph_obs.mean()}"
            )
            logger.warning(
                f"    tile_patterns: min={tile_patterns.min()}, max={tile_patterns.max()}, mean={tile_patterns.mean()}"
            )
            logger.warning(
                f"    entity_features: min={entity_features.min()}, max={entity_features.max()}, mean={entity_features.mean()}"
            )
            return None

        # Validate predicted waypoints are in reasonable normalized range (sample first batch item)
        if self.predictor.training and predicted_waypoints.size(0) > 0:
            from ..path_prediction.coordinate_utils import validate_normalized_coords

            is_valid, msg = validate_normalized_coords(
                predicted_waypoints[0], tolerance=0.2
            )
            if not is_valid:
                logger.warning(f"Predicted waypoints validation: {msg}")
            else:
                logger.debug(f"Predicted waypoints validation: {msg}")

        # Extract predicted paths and confidences
        predicted_waypoints = outputs[
            "waypoints"
        ]  # [batch, num_heads, max_waypoints, 2]
        confidences = outputs["confidences"]  # [batch, num_heads]

        # Create masks for predicted waypoints based on coordinate validity
        # Consider a waypoint valid if its coordinates are reasonable (not NaN/Inf, within bounds)
        pred_masks = (predicted_waypoints.abs() < 10000).all(dim=-1) & torch.isfinite(
            predicted_waypoints
        ).all(dim=-1)  # [batch, num_heads, max_waypoints]

        # Compute loss
        loss_result = compute_path_prediction_loss(
            predicted_paths=predicted_waypoints,
            path_confidences=confidences,
            expert_waypoints=expert_waypoints,
            pred_masks=pred_masks,
            expert_masks=waypoint_masks,
            waypoint_loss_weight=self.loss_weights["waypoint"],
            confidence_loss_weight=self.loss_weights["confidence"],
            diversity_loss_weight=self.loss_weights["diversity"],
        )

        return {
            "total_loss": loss_result.total_loss,
            "waypoint_loss": loss_result.waypoint_loss,
            "confidence_loss": loss_result.confidence_loss,
            "diversity_loss": loss_result.diversity_loss,
        }

    def train_online(
        self,
        attempted_paths: List[CandidatePath],
        outcomes: List[float],
        num_updates: int = 1,
    ) -> Dict[str, float]:
        """Update predictor online based on path outcomes.

        Args:
            attempted_paths: Paths attempted by RL agent
            outcomes: Success scores for each path (0.0-1.0)
            num_updates: Number of gradient updates to perform

        Returns:
            Dictionary with update statistics
        """
        if len(attempted_paths) != len(outcomes):
            logger.warning("Mismatch between attempted paths and outcomes")
            return {"update_success": False}

        # Update path preferences (no gradient update)
        self.predictor.update_path_preferences(attempted_paths, outcomes)

        # Optionally perform gradient updates if we have enough samples
        # This would require building a mini-batch from recent experiences
        # For now, just update preferences

        logger.debug(f"Updated path preferences for {len(attempted_paths)} paths")

        return {
            "update_success": True,
            "paths_updated": len(attempted_paths),
            "avg_outcome": sum(outcomes) / len(outcomes) if outcomes else 0.0,
        }

    def _save_checkpoint(
        self,
        save_dir: str,
        epoch: int,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint after training
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "loss_weights": self.loss_weights,
        }

        # Save final checkpoint
        if is_final:
            final_path = save_path / "final_model.pt"
            torch.save(checkpoint, final_path)
            logger.info(f"Saved final model to {final_path}")
        else:
            # Save regular periodic checkpoint
            checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = save_path / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.predictor.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.epochs_trained = checkpoint.get("epoch", 0) + 1

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.epochs_trained}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        return {
            "epochs_trained": self.epochs_trained,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "current_lr": self.optimizer.param_groups[0]["lr"],
            "predictor_stats": self.predictor.get_statistics(),
        }
