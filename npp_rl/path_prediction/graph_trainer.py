"""Trainer for graph-based path predictor using GNN + Pointer Networks.

This trainer handles:
1. Offline training from expert replay demonstrations
2. Graph-based loss computation (node classification, connectivity, diversity)
3. Training/validation loops with comprehensive logging
4. Model checkpointing and evaluation
"""

import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

from .graph_path_predictor import GraphPathPredictor
from .graph_losses import compute_graph_path_loss, compute_path_accuracy
from ..path_prediction.replay_dataset import PathReplayDataset, collate_replay_batch

logger = logging.getLogger(__name__)


class GraphPathPredictorTrainer:
    """Trainer for graph-based path prediction."""

    def __init__(
        self,
        predictor: GraphPathPredictor,
        learning_rate: float = 3e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize graph path predictor trainer.

        Args:
            predictor: GraphPathPredictor model
            learning_rate: Learning rate for optimizer
            device: Device for training
            loss_weights: Weights for loss components (node, connectivity, diversity)
        """
        self.predictor = predictor.to(device)
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
            "node": 1.0,
            "connectivity": 0.5,
            "diversity": 0.3,
        }

        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_trained = 0

        logger.info(f"Initialized GraphPathPredictorTrainer on {device}")
        logger.info(f"Loss weights: {self.loss_weights}")

    def train_offline(
        self,
        train_dataset: PathReplayDataset,
        val_dataset: Optional[PathReplayDataset] = None,
        num_epochs: int = 50,
        batch_size: int = 32,
        num_workers: int = 4,
        save_dir: Optional[str] = None,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        """Train predictor offline from expert demonstrations.

        Args:
            train_dataset: Training dataset with expert replays
            val_dataset: Optional validation dataset
            num_epochs: Number of training epochs
            batch_size: Batch size
            num_workers: Number of dataloader workers
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs

        Returns:
            Training statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting offline training")
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}")
        logger.info("=" * 60)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_replay_batch,
            pin_memory=True if self.device == "cuda" else False,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_replay_batch,
                pin_memory=True if self.device == "cuda" else False,
            )

        # Create save directory
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoints will be saved to {save_path}")

        # Training loop
        for epoch in range(num_epochs):
            self.epochs_trained = epoch + 1

            logger.info("=" * 60)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info("=" * 60)

            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics["loss"])

            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(
                f"  Node: {train_metrics['node_loss']:.4f}, "
                f"Connectivity: {train_metrics['connectivity_loss']:.4f}, "
                f"Diversity: {train_metrics['diversity_loss']:.4f}"
            )

            # Validate
            if val_loader:
                val_metrics = self._validate(val_loader)
                self.val_losses.append(val_metrics["loss"])

                logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
                logger.info(
                    f"  Node: {val_metrics['node_loss']:.4f}, "
                    f"Connectivity: {val_metrics['connectivity_loss']:.4f}, "
                    f"Diversity: {val_metrics['diversity_loss']:.4f}"
                )
                logger.info(
                    f"  Accuracy: {val_metrics['accuracy']:.3f}, "
                    f"Best Head Acc: {val_metrics['best_head_accuracy']:.3f}"
                )

                # Update scheduler
                self.scheduler.step(val_metrics["loss"])

                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    if save_dir:
                        best_path = save_path / "best_model.pt"
                        self.save_checkpoint(str(best_path), epoch, val_metrics)
                        logger.info(
                            f"  ✓ Saved best model (val_loss: {val_metrics['loss']:.4f})"
                        )

            # Periodic checkpoint
            if save_dir and (epoch + 1) % save_every == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(str(checkpoint_path), epoch, train_metrics)
                logger.info(f"  ✓ Saved checkpoint at epoch {epoch + 1}")

        logger.info("=" * 60)
        logger.info("Training complete")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epochs_trained,
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.predictor.train()

        total_loss = 0.0
        total_node_loss = 0.0
        total_connectivity_loss = 0.0
        total_diversity_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in pbar:
            # Move data to device
            node_features = batch["node_features"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            node_mask = batch["node_mask"].to(self.device)
            edge_mask = batch["edge_mask"].to(self.device)
            expert_node_ids = batch["expert_node_ids"].to(self.device)
            expert_node_mask = batch["expert_node_mask"].to(self.device)

            # Multimodal fusion inputs (for start/goal/physics conditioning)
            start_node_ids = (
                batch["start_node_ids"].to(self.device)
                if "start_node_ids" in batch
                else None
            )
            goal_node_ids = (
                batch["goal_node_ids"].to(self.device)
                if "goal_node_ids" in batch
                else None
            )
            ninja_states = (
                batch["ninja_states"].to(self.device)
                if "ninja_states" in batch
                else None
            )

            # Forward pass with multimodal fusion
            outputs = self.predictor(
                node_features,
                edge_index,
                node_mask,
                edge_mask,
                start_node_ids=start_node_ids,
                goal_node_ids=goal_node_ids,
                ninja_states=ninja_states,
                temperature=1.0,
                use_sampling=False,  # Start with argmax
            )

            predicted_logits = outputs["logits"]
            predicted_node_ids = outputs["node_indices"]

            # Compute loss with start/goal awareness
            loss_obj = compute_graph_path_loss(
                predicted_logits=predicted_logits,
                predicted_node_ids=predicted_node_ids,
                expert_node_ids=expert_node_ids,
                expert_masks=expert_node_mask,
                edge_index=edge_index,
                edge_mask=edge_mask,
                node_mask=node_mask,
                start_node_ids=start_node_ids,
                goal_node_ids=goal_node_ids,
                node_loss_weight=self.loss_weights.get("node", 1.0),
                connectivity_loss_weight=self.loss_weights.get("connectivity", 0.5),
                diversity_loss_weight=self.loss_weights.get("diversity", 0.3),
                start_goal_weight=self.loss_weights.get("start_goal", 1.0),
                goal_reaching_weight=self.loss_weights.get("goal_reaching", 0.5),
                use_start_goal_weighting=True,
                use_goal_reaching=True,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss_obj.total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_obj.total_loss.item()
            total_node_loss += loss_obj.node_classification_loss.item()
            total_connectivity_loss += loss_obj.connectivity_loss.item()
            total_diversity_loss += loss_obj.diversity_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss_obj.total_loss.item(),
                    "node": loss_obj.node_classification_loss.item(),
                    "conn": loss_obj.connectivity_loss.item(),
                    "div": loss_obj.diversity_loss.item(),
                }
            )

        return {
            "loss": total_loss / num_batches,
            "node_loss": total_node_loss / num_batches,
            "connectivity_loss": total_connectivity_loss / num_batches,
            "diversity_loss": total_diversity_loss / num_batches,
        }

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.predictor.eval()

        total_loss = 0.0
        total_node_loss = 0.0
        total_connectivity_loss = 0.0
        total_diversity_loss = 0.0
        total_accuracy = 0.0
        total_best_head_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                node_features = batch["node_features"].to(self.device)
                edge_index = batch["edge_index"].to(self.device)
                node_mask = batch["node_mask"].to(self.device)
                edge_mask = batch["edge_mask"].to(self.device)
                expert_node_ids = batch["expert_node_ids"].to(self.device)
                expert_node_mask = batch["expert_node_mask"].to(self.device)

                # Multimodal fusion inputs
                start_node_ids = (
                    batch["start_node_ids"].to(self.device)
                    if "start_node_ids" in batch
                    else None
                )
                goal_node_ids = (
                    batch["goal_node_ids"].to(self.device)
                    if "goal_node_ids" in batch
                    else None
                )
                ninja_states = (
                    batch["ninja_states"].to(self.device)
                    if "ninja_states" in batch
                    else None
                )

                # Forward pass with multimodal fusion
                outputs = self.predictor(
                    node_features,
                    edge_index,
                    node_mask,
                    edge_mask,
                    start_node_ids=start_node_ids,
                    goal_node_ids=goal_node_ids,
                    ninja_states=ninja_states,
                    temperature=1.0,
                    use_sampling=False,
                )

                predicted_logits = outputs["logits"]
                predicted_node_ids = outputs["node_indices"]

                # Compute loss with start/goal awareness
                loss_obj = compute_graph_path_loss(
                    predicted_logits=predicted_logits,
                    predicted_node_ids=predicted_node_ids,
                    expert_node_ids=expert_node_ids,
                    expert_masks=expert_node_mask,
                    edge_index=edge_index,
                    edge_mask=edge_mask,
                    node_mask=node_mask,
                    start_node_ids=start_node_ids,
                    goal_node_ids=goal_node_ids,
                    node_loss_weight=self.loss_weights.get("node", 1.0),
                    connectivity_loss_weight=self.loss_weights.get("connectivity", 0.5),
                    diversity_loss_weight=self.loss_weights.get("diversity", 0.3),
                    start_goal_weight=self.loss_weights.get("start_goal", 1.0),
                    goal_reaching_weight=self.loss_weights.get("goal_reaching", 0.5),
                    use_start_goal_weighting=True,
                    use_goal_reaching=True,
                )

                # Compute accuracy
                accuracy_metrics = compute_path_accuracy(
                    predicted_node_ids, expert_node_ids, expert_node_mask
                )

                # Accumulate metrics
                total_loss += loss_obj.total_loss.item()
                total_node_loss += loss_obj.node_classification_loss.item()
                total_connectivity_loss += loss_obj.connectivity_loss.item()
                total_diversity_loss += loss_obj.diversity_loss.item()
                total_accuracy += accuracy_metrics["exact_match_accuracy"]
                total_best_head_acc += accuracy_metrics["best_head_accuracy"]
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "node_loss": total_node_loss / num_batches,
            "connectivity_loss": total_connectivity_loss / num_batches,
            "diversity_loss": total_diversity_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "best_head_accuracy": total_best_head_acc / num_batches,
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional training metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.predictor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_weights": self.loss_weights,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.predictor.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.loss_weights = checkpoint["loss_weights"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_trained = checkpoint["epoch"] + 1

        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

        return checkpoint
