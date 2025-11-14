"""Callback to compute and log auxiliary death prediction losses.

This callback extracts auxiliary predictions from the policy and computes
losses using death_context observations. Currently logs losses for monitoring.

Note: To integrate auxiliary loss into PPO training, extend PPO.train() method
to add auxiliary loss to the total loss. This callback provides monitoring only.
"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch


class AuxiliaryLossCallback(BaseCallback):
    """Callback to compute and log auxiliary death prediction losses.

    This callback extracts auxiliary predictions from the policy and computes
    losses using death_context observations. Currently logs losses for monitoring.

    Note: To integrate auxiliary loss into PPO training, extend PPO.train() method
    to add auxiliary loss to the total loss. This callback provides monitoring only.
    """

    def __init__(
        self, log_freq: int = 100, verbose: int = 0, auxiliary_weight: float = 0.1
    ):
        """Initialize auxiliary loss callback.

        Args:
            log_freq: Frequency (in steps) to log auxiliary losses
            verbose: Verbosity level
            auxiliary_weight: Weight for auxiliary loss (for logging, not actual loss integration)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.auxiliary_weight = auxiliary_weight
        self.death_loss_history = deque(maxlen=1000)

    def _on_rollout_end(self) -> bool:
        """Called after rollout collection, before policy update.

        Computes auxiliary losses from rollout buffer data and logs them.
        """
        # Check if policy has auxiliary heads
        policy = self.model.policy
        if not hasattr(policy, "get_auxiliary_predictions"):
            return True

        # Get rollout buffer data
        rollout_buffer = self.model.rollout_buffer
        if rollout_buffer is None or rollout_buffer.size() == 0:
            return True

        # Extract observations (may be dict with death_context)
        try:
            observations = rollout_buffer.observations
        except (AttributeError, KeyError):
            # Rollout buffer observations not accessible or structured differently
            return True

        # Check if death_context is available
        # Handle both dict observations and cases where death_context might not exist
        death_context = None
        if isinstance(observations, dict):
            death_context = observations.get("death_context", None)
        elif hasattr(observations, "get"):
            # Try dict-like access
            death_context = observations.get("death_context", None)

        if death_context is not None:
            try:
                # Compute death labels with lookahead

                # Convert to tensor if needed
                if isinstance(death_context, np.ndarray):
                    death_context_tensor = torch.from_numpy(death_context).float()
                elif isinstance(death_context, torch.Tensor):
                    death_context_tensor = death_context.float()
                else:
                    # Unknown type, skip
                    return True

                # Debug: log death_context shape
                if self.verbose > 1:
                    self.logger.info(
                        f"death_context shape: {death_context_tensor.shape}, "
                        f"dim: {death_context_tensor.dim()}"
                    )

                # Handle different shapes: [batch, 9] or [9] (single timestep)
                if death_context_tensor.dim() == 1:
                    # Single timestep: shape [9]
                    # Treat as batch_size=1
                    batch_size = 1
                    death_context_tensor = death_context_tensor.unsqueeze(0)  # [1, 9]
                elif death_context_tensor.dim() == 2:
                    # Batch of timesteps: shape [batch, 9]
                    batch_size = death_context_tensor.shape[0]
                else:
                    if self.verbose > 0:
                        self.logger.warn(
                            f"Unexpected death_context shape: {death_context_tensor.shape}, "
                            f"dim: {death_context_tensor.dim()}"
                        )
                    return True

                # Extract death flags (index 0)
                # death_context_tensor[:, 0] extracts first feature for all batch elements -> [batch]
                death_flags = death_context_tensor[:, 0] > 0.5

                # Debug: verify death_flags shape
                if death_flags.dim() != 1 or death_flags.shape[0] != batch_size:
                    if self.verbose > 0:
                        self.logger.warn(
                            f"death_flags shape mismatch: expected [batch={batch_size}], "
                            f"got {death_flags.shape}"
                        )
                    return True

                # Compute labels: for each timestep, check if death occurs within horizon
                horizon = 10
                death_labels = torch.zeros(
                    batch_size, dtype=torch.float32, device=death_context_tensor.device
                )

                for i in range(batch_size):
                    # Verify death_flags[i] is a scalar before calling .item()
                    flag_tensor = death_flags[i]
                    if flag_tensor.numel() != 1:
                        if self.verbose > 0:
                            self.logger.warn(
                                f"death_flags[{i}] is not a scalar: shape={flag_tensor.shape}, "
                                f"numel={flag_tensor.numel()}"
                            )
                        continue
                    if flag_tensor.item():
                        # Mark previous steps within horizon
                        start_idx = max(0, i - horizon + 1)
                        death_labels[start_idx : i + 1] = 1.0

                # Log death label statistics
                death_rate = death_labels.mean().item()
                if self.num_timesteps % self.log_freq == 0:
                    self.logger.record("auxiliary/death_label_rate", death_rate)
                    self.logger.record(
                        "auxiliary/death_labels_sum", death_labels.sum().item()
                    )
            except (
                IndexError,
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as e:
                # Handle shape mismatches or other errors gracefully
                import traceback

                if self.verbose > 0:
                    self.logger.warn(
                        f"Error computing auxiliary death labels: {e}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )

        return True

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Check if policy has auxiliary predictions from last forward pass
        try:
            policy = self.model.policy
            if hasattr(policy, "get_auxiliary_predictions"):
                auxiliary_preds = policy.get_auxiliary_predictions()
                if auxiliary_preds is not None and "death_prob" in auxiliary_preds:
                    # Log death probability prediction
                    death_prob = auxiliary_preds["death_prob"].mean().item()
                    if self.num_timesteps % self.log_freq == 0:
                        self.logger.record("auxiliary/death_prob_mean", death_prob)
        except (AttributeError, KeyError, TypeError):
            # Policy doesn't have auxiliary predictions or error accessing them
            pass

        return True

