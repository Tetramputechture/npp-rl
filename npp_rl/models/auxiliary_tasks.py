"""Auxiliary prediction tasks for multi-task learning.

Implements auxiliary prediction heads for death prediction, time-to-goal
estimation, and subgoal classification to improve representation learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AuxiliaryTaskHeads(nn.Module):
    """Multi-task prediction heads for auxiliary learning.

    These auxiliary tasks help the policy learn better representations by
    providing additional learning signals beyond the primary RL objective:

    1. Death Prediction: Predict probability of death in next N steps
    2. Time-to-Goal: Predict steps needed to reach current objective
    3. Next Subgoal: Classify which objective to pursue next

    These tasks encourage the network to learn features that capture:
    - Safety/danger (death prediction)
    - Progress and efficiency (time-to-goal)
    - Strategic planning (subgoal selection)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        max_objectives: int = 34,  # 1 exit + 16 locked doors + 16 switches
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize auxiliary task heads.

        Args:
            feature_dim: Dimension of input policy features
            max_objectives: Maximum number of objectives for subgoal classification
            hidden_dim: Hidden dimension for prediction heads
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.max_objectives = max_objectives

        # Death prediction head (binary classification)
        # Predicts: will the agent die in the next 10 steps?
        self.death_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output probability in [0, 1]
        )

        # Time-to-goal prediction head (regression)
        # Predicts: how many steps until reaching the current objective?
        self.time_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive output
        )

        # Next subgoal classification head (multi-class classification)
        # Predicts: which objective should be pursued next?
        # Classes: exit_switch, exit_door, locked_door_0, ..., locked_door_15,
        #          locked_switch_0, ..., locked_switch_15
        self.subgoal_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, max_objectives),  # One logit per objective
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all auxiliary heads.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Dictionary with keys:
                - death_prob: Death probability [batch, 1]
                - time_to_goal: Estimated steps to goal [batch, 1]
                - next_subgoal_logits: Subgoal logits [batch, max_objectives]
        """
        return {
            "death_prob": self.death_head(features),
            "time_to_goal": self.time_head(features),
            "next_subgoal_logits": self.subgoal_head(features),
        }

    def predict_death(self, features: torch.Tensor) -> torch.Tensor:
        """Predict death probability.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Death probabilities [batch, 1]
        """
        return self.death_head(features)

    def predict_time_to_goal(self, features: torch.Tensor) -> torch.Tensor:
        """Predict time to goal.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Estimated steps to goal [batch, 1]
        """
        return self.time_head(features)

    def predict_next_subgoal(self, features: torch.Tensor) -> torch.Tensor:
        """Predict next subgoal.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Subgoal logits [batch, max_objectives]
        """
        return self.subgoal_head(features)


def compute_death_labels_from_context(
    observations: Dict[str, torch.Tensor],
    horizon: int = 10,
) -> torch.Tensor:
    """Compute death labels from death_context observation with lookahead.

    For each timestep t, label is 1 if agent dies within next horizon steps.
    Uses death_context observation to detect actual deaths.

    Args:
        observations: Dictionary with "death_context" key containing [batch, 9] array
        horizon: Number of steps to look ahead for death prediction

    Returns:
        Binary labels [batch] (1=will die, 0=won't die)
    """
    if "death_context" not in observations:
        # No death context available, return zeros
        batch_size = observations.get("game_state", torch.zeros(1, 64)).shape[0]
        device = observations.get("game_state", torch.zeros(1, 64)).device
        return torch.zeros(batch_size, dtype=torch.float32, device=device)

    death_context = observations["death_context"]  # Expected: [batch, 9]
    device = death_context.device

    # Handle different shapes: [batch, 9] or [9] (single timestep)
    if death_context.dim() == 1:
        # Single timestep: shape [9]
        # Treat as batch_size=1
        batch_size = 1
        death_context = death_context.unsqueeze(0)  # [1, 9]
    elif death_context.dim() == 2:
        # Batch of timesteps: shape [batch, 9]
        batch_size = death_context.shape[0]
    else:
        raise ValueError(
            f"death_context must be 1D [9] or 2D [batch, 9], got shape {death_context.shape}"
        )

    # Extract death_occurred flags (index 0)
    # death_context[:, 0] extracts first feature for all batch elements -> [batch]
    death_flags = death_context[:, 0] > 0.5  # [batch]

    # Debug: verify death_flags shape
    if death_flags.dim() != 1 or death_flags.shape[0] != batch_size:
        raise ValueError(
            f"death_flags shape mismatch: expected [batch={batch_size}], got {death_flags.shape}"
        )

    # For each position, check if death occurs within horizon
    # This is a simplified version - in practice, would use rollout buffer
    # to look ahead across timesteps
    death_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # Simple approach: if death occurred at this step, mark previous horizon steps
    # In full implementation, would use rollout buffer to look ahead
    for i in range(batch_size):
        # Verify death_flags[i] is a scalar before calling .item()
        flag_tensor = death_flags[i]
        if flag_tensor.numel() != 1:
            raise ValueError(
                f"death_flags[{i}] is not a scalar: shape={flag_tensor.shape}, numel={flag_tensor.numel()}"
            )
        if flag_tensor.item():
            # Mark previous steps within horizon
            start_idx = max(0, i - horizon + 1)
            death_labels[start_idx : i + 1] = 1.0

    return death_labels


def compute_auxiliary_labels(
    trajectory: Dict[str, torch.Tensor],
    death_horizon: int = 10,
    max_time: int = 1000,
) -> Dict[str, torch.Tensor]:
    """Compute auxiliary task labels from trajectory data.

    This function processes trajectory data to generate labels for
    auxiliary tasks using hindsight information.

    Args:
        trajectory: Dictionary containing:
            - observations: [T, obs_dim] or dict with "death_context"
            - actions: [T] (optional)
            - returns: [T] (can be used instead of rewards)
            - rewards: [T] (optional, returns takes precedence)
            - dones: [T] (optional)
            - infos: List of T info dicts (optional)
        death_horizon: Number of steps to look ahead for death prediction
        max_time: Maximum time value for time-to-goal

    Returns:
        Dictionary with keys:
            - death_labels: [T] (1 if died within horizon, 0 otherwise)
            - time_labels: [T] (steps to reach goal, from hindsight)
            - subgoal_labels: [T] (optimal next subgoal indices)
    """
    # Get trajectory length - handle both tensor and list cases
    # Try to infer T from available fields
    if "dones" in trajectory:
        dones = trajectory["dones"]
        if isinstance(dones, torch.Tensor):
            T = dones.shape[0] if dones.dim() > 0 else 1
            device = dones.device
        else:
            T = len(dones)
            device = (
                dones[0].device
                if len(dones) > 0 and isinstance(dones[0], torch.Tensor)
                else torch.device("cpu")
            )
    elif "returns" in trajectory:
        # Infer from returns if dones not available
        returns = trajectory["returns"]
        if isinstance(returns, torch.Tensor):
            T = returns.shape[0] if returns.dim() > 0 else 1
            device = returns.device
        else:
            T = len(returns)
            device = torch.device("cpu")
    elif "observations" in trajectory:
        # Try to infer from observations
        obs = trajectory["observations"]
        if isinstance(obs, dict):
            # Get any tensor from obs dict
            for key, val in obs.items():
                if isinstance(val, torch.Tensor):
                    T = val.shape[0] if val.dim() > 0 else 1
                    device = val.device
                    break
            else:
                # No tensor found, default
                T = 1
                device = torch.device("cpu")
        elif isinstance(obs, torch.Tensor):
            T = obs.shape[0] if obs.dim() > 0 else 1
            device = obs.device
        else:
            T = len(obs) if hasattr(obs, "__len__") else 1
            device = torch.device("cpu")
    else:
        # Last resort defaults
        T = 1
        device = torch.device("cpu")

    # 1. Death prediction labels
    # Handle case where observations might be a dict or tensor
    observations = trajectory["observations"]
    if isinstance(observations, dict):
        death_labels = compute_death_labels_from_context(
            observations,
            horizon=death_horizon,
        )
        # Ensure death_labels matches T
        if death_labels.shape[0] != T:
            # Resize to match T
            if death_labels.shape[0] < T:
                # Pad with zeros
                padding = torch.zeros(
                    T - death_labels.shape[0],
                    dtype=death_labels.dtype,
                    device=device,
                )
                death_labels = torch.cat([death_labels, padding])
            else:
                # Truncate
                death_labels = death_labels[:T]
    else:
        # If observations is a tensor, create a dummy dict and return zeros
        # This shouldn't happen in normal usage, but handle gracefully
        death_labels = torch.zeros(T, dtype=torch.float32, device=device)

    # 2. Time-to-goal labels
    # Use hindsight: count steps until next positive reward (goal reached)
    # Try to use rewards if available, otherwise use returns as proxy
    time_labels = torch.full((T,), max_time, dtype=torch.float32, device=device)

    # Check if we have rewards or returns
    reward_signal = None
    if "rewards" in trajectory:
        reward_signal = trajectory["rewards"]
    elif "returns" in trajectory:
        # Returns are available from rollout samples
        # Use return increases as proxy for reward events
        reward_signal = trajectory["returns"]

    if reward_signal is not None:
        # Ensure reward_signal is a tensor
        if not isinstance(reward_signal, torch.Tensor):
            reward_signal = torch.as_tensor(reward_signal, device=device)

        # Handle shape mismatch - reward_signal might be from a minibatch
        if reward_signal.shape[0] == T:
            for t in range(T):
                # Find next positive reward (goal/subgoal completion)
                positive_rewards = (reward_signal[t:] > 0.5).nonzero(as_tuple=True)[0]
                if len(positive_rewards) > 0:
                    time_labels[t] = float(positive_rewards[0].item())
        else:
            # Size mismatch - use conservative default
            # This happens when processing mini-batches
            # Set time labels to gradually decreasing values as heuristic
            for t in range(T):
                time_labels[t] = float(min(max_time, T - t))

    # 3. Next subgoal labels (placeholder - would require A* or expert planning)
    # For now, use a simple heuristic based on switch/door states
    subgoal_labels = torch.zeros(T, dtype=torch.long, device=device)
    for t in range(T):
        info = trajectory["infos"][t] if "infos" in trajectory else {}
        switch_activated = info.get("switch_activated", False)

        # Simple heuristic:
        # - If switch not activated → go to exit switch (objective 0)
        # - If switch activated → go to exit door (objective 1)
        subgoal_labels[t] = 1 if switch_activated else 0

    return {
        "death_labels": death_labels,
        "time_labels": time_labels,
        "subgoal_labels": subgoal_labels,
    }


def compute_auxiliary_losses(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    weights: Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute auxiliary task losses.

    Args:
        predictions: Dictionary with prediction outputs
        labels: Dictionary with ground truth labels
        weights: Optional loss weights for each task

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    if weights is None:
        weights = {
            "death": 0.1,
            "time": 0.1,
            "subgoal": 0.1,
        }

    losses = {}

    # Death prediction loss (binary cross-entropy)
    if "death_prob" in predictions and "death_labels" in labels:
        death_pred = predictions["death_prob"].squeeze(-1)
        death_target = labels["death_labels"]
        losses["death"] = nn.functional.binary_cross_entropy(
            death_pred, death_target, reduction="mean"
        )

    # Time-to-goal loss (smooth L1 loss)
    if "time_to_goal" in predictions and "time_labels" in labels:
        time_pred = predictions["time_to_goal"].squeeze(-1)
        time_target = labels["time_labels"]
        losses["time"] = nn.functional.smooth_l1_loss(
            time_pred, time_target, reduction="mean"
        )

    # Next subgoal loss (cross-entropy)
    if "next_subgoal_logits" in predictions and "subgoal_labels" in labels:
        subgoal_logits = predictions["next_subgoal_logits"]
        subgoal_target = labels["subgoal_labels"]
        losses["subgoal"] = nn.functional.cross_entropy(
            subgoal_logits, subgoal_target, reduction="mean"
        )

    # Compute weighted total loss
    total_loss = sum(weights.get(k, 0.1) * v for k, v in losses.items())

    return total_loss, losses


class MultiTaskPolicy(nn.Module):
    """Policy with integrated auxiliary task heads.

    This wraps a base policy with auxiliary prediction heads for
    multi-task learning.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        feature_dim: int = 256,
        max_objectives: int = 34,
        enable_auxiliary: bool = True,
    ):
        """Initialize multi-task policy.

        Args:
            base_policy: Base policy network
            feature_dim: Dimension of policy features
            max_objectives: Maximum number of objectives
            enable_auxiliary: Whether to enable auxiliary tasks
        """
        super().__init__()

        self.base_policy = base_policy
        self.enable_auxiliary = enable_auxiliary

        if enable_auxiliary:
            self.auxiliary_heads = AuxiliaryTaskHeads(
                feature_dim=feature_dim,
                max_objectives=max_objectives,
            )
        else:
            self.auxiliary_heads = None

    def forward(
        self,
        observations: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with optional auxiliary predictions.

        Args:
            observations: Input observations
            return_auxiliary: Whether to return auxiliary predictions

        Returns:
            Tuple of (action_logits, auxiliary_predictions)
        """
        # Get policy features and action logits
        policy_features = self.base_policy.get_policy_latent(observations)
        action_logits = self.base_policy.action_net(policy_features)

        # Compute auxiliary predictions if requested
        auxiliary_predictions = {}
        if return_auxiliary and self.auxiliary_heads is not None:
            auxiliary_predictions = self.auxiliary_heads(policy_features)

        return action_logits, auxiliary_predictions
