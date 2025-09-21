"""
Behavioral Cloning trainer for policy pretraining.

This module provides the BCTrainer class which implements supervised learning
on human replay data to provide good initialization for subsequent RL training.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from gymnasium.spaces import Discrete, Dict as SpacesDict
from stable_baselines3 import PPO

from npp_rl.training.training_utils import create_training_policy, setup_device
from nclone.gym_environment.graph_observation import (
    create_graph_enhanced_env,
)


class BCTrainer:
    """
    Behavioral Cloning trainer for policy pretraining.

    This trainer uses supervised learning to train a policy on expert demonstrations,
    providing a good initialization for subsequent reinforcement learning.
    """

    def __init__(
        self,
        observation_space: SpacesDict,
        action_space: Discrete,
        policy_class: str = "npp",
        use_graph_obs: bool = False,
        learning_rate: float = 3e-4,
        entropy_coef: float = 0.01,
        freeze_backbone_steps: int = 0,
        device: str = "auto",
    ):
        """
        Initialize BC trainer.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            policy_class: Policy architecture ('npp' or 'simple')
            use_graph_obs: Whether to use graph observations
            learning_rate: Learning rate for optimization
            entropy_coef: Entropy regularization coefficient
            freeze_backbone_steps: Number of steps to freeze backbone for
            device: Device to train on ('auto', 'cpu', 'cuda')
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_graph_obs = use_graph_obs
        self.entropy_coef = entropy_coef
        self.freeze_backbone_steps = freeze_backbone_steps

        # Set up device
        self.device = setup_device(device)
        print(f"Using device: {self.device}")

        # Create policy
        self.policy = create_training_policy(
            observation_space=observation_space,
            action_space=action_space,
            policy_class=policy_class,
            use_graph_obs=use_graph_obs,
        )
        self.policy.to(self.device)

        # Create optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.policy.parameters(), lr=learning_rate, weight_decay=1e-4
        )

        # Learning rate scheduler for training stability
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-6
        )

        # Training state tracking
        self.step = 0
        self.epoch = 0
        self.best_accuracy = 0.0

        # Metrics storage
        self.train_metrics = {"loss": [], "accuracy": [], "entropy": []}

    def train_epoch(
        self, dataloader, writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            writer: Optional TensorBoard writer for logging

        Returns:
            Dictionary of training metrics for the epoch
        """
        self.policy.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_entropy = 0.0
        num_batches = 0

        # Apply backbone freezing if specified
        if self.step < self.freeze_backbone_steps:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")

        for batch_idx, (observations, actions) in enumerate(pbar):
            # Move data to device
            observations = {
                key: value.to(self.device) for key, value in observations.items()
            }
            actions = actions.to(self.device)

            # Forward pass with error handling
            try:
                logits = self.policy(observations)
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(
                    f"Observation shapes: {[(k, v.shape) for k, v in observations.items()]}"
                )
                continue

            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(logits, actions)

            # Add entropy regularization to encourage exploration
            if self.entropy_coef > 0:
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                loss = loss - self.entropy_coef * entropy
                epoch_entropy += entropy.item()

            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == actions).float().mean()

            # Update metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            self.step += 1

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy.item():.3f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # Log to TensorBoard
            if writer and self.step % 100 == 0:
                writer.add_scalar("train/loss", loss.item(), self.step)
                writer.add_scalar("train/accuracy", accuracy.item(), self.step)
                writer.add_scalar(
                    "train/learning_rate", self.scheduler.get_last_lr()[0], self.step
                )
                if self.entropy_coef > 0:
                    writer.add_scalar("train/entropy", entropy.item(), self.step)

        # Compute epoch averages
        metrics = {
            "loss": epoch_loss / num_batches,
            "accuracy": epoch_accuracy / num_batches,
            "entropy": epoch_entropy / num_batches if self.entropy_coef > 0 else 0.0,
        }

        self.epoch += 1
        return metrics

    def validate(
        self, dataloader, writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """
        Validate the model on a validation dataset.

        Args:
            dataloader: Validation data loader
            writer: Optional TensorBoard writer for logging

        Returns:
            Dictionary of validation metrics
        """
        self.policy.eval()

        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for observations, actions in tqdm(dataloader, desc="Validation"):
                # Move data to device
                observations = {
                    key: value.to(self.device) for key, value in observations.items()
                }
                actions = actions.to(self.device)

                # Forward pass with error handling
                try:
                    logits = self.policy(observations)
                except Exception as e:
                    print(f"Error in validation forward pass: {e}")
                    continue

                # Compute loss and accuracy
                loss = nn.functional.cross_entropy(logits, actions)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == actions).float().mean()

                val_loss += loss.item()
                val_accuracy += accuracy.item()
                num_batches += 1

        metrics = {
            "loss": val_loss / num_batches if num_batches > 0 else float("inf"),
            "accuracy": val_accuracy / num_batches if num_batches > 0 else 0.0,
        }

        # Log to TensorBoard
        if writer:
            writer.add_scalar("val/loss", metrics["loss"], self.epoch)
            writer.add_scalar("val/accuracy", metrics["accuracy"], self.epoch)

        return metrics

    def _freeze_backbone(self):
        """
        Freeze backbone encoder parameters to stabilize early training.

        This can be useful when training on limited data or when fine-tuning
        pre-trained feature extractors.
        """
        if hasattr(self.policy[0], "player_encoder"):
            for param in self.policy[0].player_encoder.parameters():
                param.requires_grad = False
        if hasattr(self.policy[0], "global_encoder"):
            for param in self.policy[0].global_encoder.parameters():
                param.requires_grad = False
        if hasattr(self.policy[0], "graph_encoder"):
            for param in self.policy[0].graph_encoder.parameters():
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all model parameters for full training."""
        for param in self.policy.parameters():
            param.requires_grad = True

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save training checkpoint with full training state.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "train_metrics": self.train_metrics,
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = str(Path(path).parent / "best_model.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint and restore training state.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_accuracy = checkpoint["best_accuracy"]
        self.train_metrics = checkpoint["train_metrics"]

    def export_for_sb3(self, path: str):
        """
        Export policy in Stable-Baselines3 compatible format.

        This allows the pretrained policy to be used as initialization
        for subsequent RL training with SB3.

        Args:
            path: Path to save the exported policy
        """
        # Create a dummy environment to get the right structure
        dummy_env = create_graph_enhanced_env(use_graph_obs=self.use_graph_obs)

        try:
            # Create PPO model with matching architecture for compatibility
            ppo_model = PPO(
                policy="MultiInputPolicy",
                env=dummy_env,
                policy_kwargs={
                    "features_extractor_class": type(self.policy[0])
                    if hasattr(self.policy, "__getitem__")
                    else None,
                    "features_extractor_kwargs": {
                        "features_dim": 512,
                        "use_graph_obs": self.use_graph_obs,
                    },
                },
            )

            # Save policy state dict with metadata for loading
            torch.save(
                {
                    "policy_state_dict": self.policy.state_dict(),
                    "observation_space": self.observation_space,
                    "action_space": self.action_space,
                    "use_graph_obs": self.use_graph_obs,
                },
                path,
            )

            print(f"Exported BC policy to {path}")

        except Exception as e:
            print(f"Error exporting policy: {e}")
            # Fallback: save raw policy state dict
            torch.save(self.policy.state_dict(), path)

        finally:
            dummy_env.close()
