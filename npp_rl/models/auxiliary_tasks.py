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
            - observations: [T, obs_dim]
            - actions: [T]
            - rewards: [T]
            - dones: [T]
            - infos: List of T info dicts
        death_horizon: Number of steps to look ahead for death prediction
        max_time: Maximum time value for time-to-goal
    
    Returns:
        Dictionary with keys:
            - death_labels: [T] (1 if died within horizon, 0 otherwise)
            - time_labels: [T] (steps to reach goal, from hindsight)
            - subgoal_labels: [T] (optimal next subgoal indices)
    """
    T = len(trajectory["dones"])
    device = trajectory["dones"].device
    
    # 1. Death prediction labels
    # Look ahead death_horizon steps to see if agent dies
    death_labels = torch.zeros(T, dtype=torch.float32, device=device)
    for t in range(T):
        # Check if agent dies within next death_horizon steps
        end_idx = min(t + death_horizon, T)
        if trajectory["dones"][t:end_idx].any():
            # Find first death
            death_idx = (trajectory["dones"][t:end_idx] == 1).nonzero(as_tuple=True)[0][0]
            # Check if death was negative reward (actual death, not completion)
            if t + death_idx < T and trajectory["rewards"][t + death_idx] < 0:
                death_labels[t] = 1.0
    
    # 2. Time-to-goal labels
    # Use hindsight: count steps until next positive reward (goal reached)
    time_labels = torch.full((T,), max_time, dtype=torch.float32, device=device)
    for t in range(T):
        # Find next positive reward (goal/subgoal completion)
        positive_rewards = (trajectory["rewards"][t:] > 0.5).nonzero(as_tuple=True)[0]
        if len(positive_rewards) > 0:
            time_labels[t] = float(positive_rewards[0])
    
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

