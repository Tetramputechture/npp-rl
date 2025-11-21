"""Auxiliary prediction tasks for multi-task learning.

Implements auxiliary prediction head for death prediction to improve representation learning.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AuxiliaryTaskHeads(nn.Module):
    """Death prediction head for auxiliary learning.

    This auxiliary task helps the policy learn better representations by
    providing additional learning signals beyond the primary RL objective:

    1. Death Prediction: Predict probability of death in next N steps

    This task encourages the network to learn features that capture:
    - Safety/danger patterns from game physics and state
    """

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize auxiliary task head.

        Args:
            feature_dim: Dimension of input policy features
            hidden_dim: Hidden dimension for prediction head
            dropout: Dropout rate
        """
        super().__init__()

        self.feature_dim = feature_dim

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

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through death prediction head.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Dictionary with keys:
                - death_prob: Death probability [batch, 1]
        """
        return {
            "death_prob": self.death_head(features),
        }

    def predict_death(self, features: torch.Tensor) -> torch.Tensor:
        """Predict death probability.

        Args:
            features: Policy features [batch, feature_dim]

        Returns:
            Death probabilities [batch, 1]
        """
        return self.death_head(features)


def compute_death_labels_from_physics(
    observations: Dict[str, torch.Tensor],
    horizon: int = 10,
) -> torch.Tensor:
    """Compute death labels from physics-based forward prediction using actual game mechanics.

    Uses the actual terminal impact detection logic from ninja.py and mine collision
    detection from entity_toggle_mine.py to predict death risk.

    Terminal Impact Logic (from ninja.py lines 457-477, 507-525):
    - impact_vel = -(normal_x * speed_x + normal_y * speed_y)
    - Dies if: impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)
    - MAX_SURVIVABLE_IMPACT = 6, only when ninja was airborne

    Mine Collision Logic (from entity_toggle_mine.py):
    - Only toggled mines (state 0) are deadly with radius 4.0 pixels
    - NINJA_RADIUS = 10 pixels
    - Collision when distance < (NINJA_RADIUS + mine_radius) = 14.0 pixels

    Args:
        observations: Dictionary with game state and position information
        horizon: Number of steps to look ahead for death prediction

    Returns:
        Binary labels [batch] (1=will die, 0=won't die)
    """
    # Get batch size and device from available observations
    if "game_state" in observations:
        game_state = observations["game_state"]
        batch_size = game_state.shape[0] if game_state.dim() > 1 else 1
        device = game_state.device
    elif "entity_positions" in observations:
        entity_pos = observations["entity_positions"]
        batch_size = entity_pos.shape[0] if entity_pos.dim() > 1 else 1
        device = entity_pos.device
    else:
        # Fallback: no useful observations available
        return torch.zeros(1, dtype=torch.float32)

    death_labels = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # Actual game constants (from nclone physics_constants.py)
    MAX_SURVIVABLE_IMPACT = 6  # From ninja.py terminal impact logic
    NINJA_RADIUS = 10  # Ninja collision radius in pixels
    TOGGLE_MINE_RADIUS_DEADLY = 4.0  # Toggled mine radius (state 0)
    MINE_COLLISION_DISTANCE = NINJA_RADIUS + TOGGLE_MINE_RADIUS_DEADLY  # 14.0 pixels
    TERMINAL_IMPACT_SAFE_VELOCITY = 3  # Below this, skip expensive checks
    MAX_HOR_SPEED = 3.333  # Maximum horizontal speed from physics constants
    LEVEL_WIDTH = 1056  # Full level width in pixels
    LEVEL_HEIGHT = 600  # Full level height in pixels

    # Physics-based death risk calculation using actual game mechanics
    for i in range(batch_size):
        death_risk = 0.0

        # Extract game state for this batch element
        if "game_state" in observations:
            if batch_size > 1:
                state = game_state[i]  # [state_dim]
            else:
                state = game_state.squeeze(0) if game_state.dim() > 1 else game_state

            # Extract velocity components (indices from get_ninja_state())
            if state.shape[0] >= 3:
                velocity_mag_norm = state[
                    0
                ].item()  # Normalized velocity magnitude [-1, 1]
                velocity_dir_x = state[1].item()  # Velocity direction x [-1, 1]
                velocity_dir_y = state[2].item()  # Velocity direction y [-1, 1]

                # Convert normalized velocity back to actual physics units
                # From get_ninja_state(): velocity_mag is normalized by MAX_HOR_SPEED * 2
                max_velocity = MAX_HOR_SPEED * 2  # 6.666
                actual_velocity_mag = (
                    (velocity_mag_norm + 1) * max_velocity / 2
                )  # [0, 6.666]
                actual_speed_x = velocity_dir_x * actual_velocity_mag
                actual_speed_y = velocity_dir_y * actual_velocity_mag

                # Extract additional physics state (indices 4-7 from get_ninja_state())
                if state.shape[0] >= 8:
                    ground_movement = state[
                        4
                    ].item()  # Ground movement category [-1, 1]
                    air_movement = state[5].item()  # Air movement category [-1, 1]
                    airborne_status = state[7].item()  # Airborne status [-1, 1]

                    # Only check terminal impact if ninja is/was airborne (like actual game logic)
                    was_airborne = airborne_status > 0.0 or air_movement > 0.0

                    if (
                        was_airborne
                        and actual_velocity_mag > TERMINAL_IMPACT_SAFE_VELOCITY
                    ):
                        # ACTUAL TERMINAL IMPACT LOGIC (from ninja.py lines 457-477, 507-525)

                        # Extract surface normals (indices from get_ninja_state())
                        if (
                            state.shape[0] >= 20
                        ):  # floor_normalized_x at index 17, floor_normalized_y at index 15
                            floor_normal_x = (
                                state[17].item() if state.shape[0] > 17 else 0.0
                            )  # Approximate
                            floor_normal_y = (
                                state[15].item() if state.shape[0] > 15 else -1.0
                            )  # Default down

                            # Calculate impact velocity using actual game formula
                            # impact_vel = -(normal_x * speed_x + normal_y * speed_y)
                            impact_vel_floor = -(
                                floor_normal_x * actual_speed_x
                                + floor_normal_y * actual_speed_y
                            )

                            # Apply actual death threshold from ninja.py
                            # Dies if: impact_vel > MAX_SURVIVABLE_IMPACT - 4/3 * abs(normal_y)
                            death_threshold = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(
                                floor_normal_y
                            )

                            if impact_vel_floor > death_threshold:
                                death_risk += (
                                    0.8  # High confidence - uses actual game logic
                                )

                        # Simplified ceiling impact check (similar logic)
                        if (
                            actual_speed_y < -3.0
                        ):  # Fast upward movement (negative is up)
                            ceiling_normal_y = 1.0  # Ceiling points down
                            impact_vel_ceiling = -(
                                -ceiling_normal_y * actual_speed_y
                            )  # Upward impact
                            ceiling_threshold = MAX_SURVIVABLE_IMPACT - (4 / 3) * abs(
                                ceiling_normal_y
                            )

                            if impact_vel_ceiling > ceiling_threshold:
                                death_risk += 0.7  # High ceiling impact risk

        # ACTUAL MINE COLLISION DETECTION using mine_data (from gather_entities_from_neighbourhood approach)
        if "mine_data" in observations:
            mine_data = observations["mine_data"]
            ninja_pos = observations.get("entity_positions", None)

            if ninja_pos is not None and ninja_pos.numel() >= 2:
                # Get ninja position
                if batch_size > 1:
                    pos = ninja_pos[i][:2]  # [ninja_x, ninja_y]
                else:
                    pos = (
                        ninja_pos.squeeze(0)[:2]
                        if ninja_pos.dim() > 1
                        else ninja_pos[:2]
                    )

                ninja_x_norm = pos[0].item()  # Normalized position [0, 1]
                ninja_y_norm = pos[1].item()  # Normalized position [0, 1]

                # Convert to actual pixel coordinates
                ninja_x = ninja_x_norm * LEVEL_WIDTH  # [0, 1056]
                ninja_y = ninja_y_norm * LEVEL_HEIGHT  # [0, 600]

                # ACTUAL MINE COLLISION LOGIC (from entity_toggle_mine.py logical_collision)
                # Check collision with each mine using actual game collision detection
                if isinstance(mine_data, dict) and "positions" in mine_data:
                    mine_positions = mine_data.get("positions", [])
                    mine_states = mine_data.get("states", [])
                    mine_radii = mine_data.get("radii", [])

                    if len(mine_positions) > 0:
                        for mine_idx in range(len(mine_positions)):
                            mine_x = mine_positions[mine_idx][0]
                            mine_y = mine_positions[mine_idx][1]
                            mine_state = (
                                mine_states[mine_idx]
                                if mine_idx < len(mine_states)
                                else 0
                            )
                            mine_radius = (
                                mine_radii[mine_idx]
                                if mine_idx < len(mine_radii)
                                else 4.0
                            )

                            # Only deadly mines (state 0) can kill
                            if mine_state == 0:  # Toggled/deadly mine
                                # Use actual game collision formula: overlap_circle_vs_circle
                                distance = (
                                    (ninja_x - mine_x) ** 2 + (ninja_y - mine_y) ** 2
                                ) ** 0.5
                                collision_distance = NINJA_RADIUS + mine_radius

                                if distance < collision_distance:
                                    # Direct collision with deadly mine = certain death
                                    death_risk += 0.95  # Very high confidence - actual game collision
                                elif (
                                    distance < collision_distance + 20
                                ):  # Near-miss danger zone
                                    # Close to deadly mine - moderate risk based on velocity
                                    proximity_factor = 1.0 - (
                                        (distance - collision_distance) / 20.0
                                    )
                                    velocity_factor = min(
                                        actual_velocity_mag / 5.0, 1.0
                                    )
                                    death_risk += (
                                        proximity_factor * velocity_factor * 0.4
                                    )

                # Edge/out-of-bounds risk (common death cause)
                edge_buffer = 30  # Pixels from edge
                if (
                    ninja_x < edge_buffer
                    or ninja_x > LEVEL_WIDTH - edge_buffer
                    or ninja_y < edge_buffer
                    or ninja_y > LEVEL_HEIGHT - edge_buffer
                ):
                    death_risk += (
                        0.3  # Edge risk (reduced since we have actual mine data now)
                    )

        # Fallback: if mine_data not available, use entity_positions for basic risk assessment
        elif "entity_positions" in observations:
            entity_pos = observations["entity_positions"]
            if batch_size > 1:
                pos = entity_pos[i][:2] if entity_pos[i].numel() >= 2 else entity_pos[i]
            else:
                pos = (
                    entity_pos.squeeze(0)[:2]
                    if entity_pos.dim() > 1 and entity_pos.numel() >= 2
                    else entity_pos[:2]
                )

            if pos.numel() >= 2:
                ninja_x_norm = pos[0].item()
                ninja_y_norm = pos[1].item()
                ninja_x = ninja_x_norm * LEVEL_WIDTH
                ninja_y = ninja_y_norm * LEVEL_HEIGHT

                # Basic edge risk only (without mine data)
                edge_buffer = 40
                if (
                    ninja_x < edge_buffer
                    or ninja_x > LEVEL_WIDTH - edge_buffer
                    or ninja_y < edge_buffer
                    or ninja_y > LEVEL_HEIGHT - edge_buffer
                ):
                    death_risk += 0.2  # Basic edge risk

        # Additional physics-based risk factors using actual game state
        if "game_state" in observations and "state" in locals() and state.shape[0] >= 8:
            # Movement state analysis (indices from get_ninja_state())
            # ground_movement = state[4].item()  # Ground movement category [-1, 1] - not used
            air_movement = state[5].item()  # Air movement category [-1, 1]
            wall_interaction = state[6].item()  # Wall interaction [-1, 1]
            airborne_status = state[7].item()  # Airborne status [-1, 1]

            # Extract more physics state if available (buffer states, contact info)
            if state.shape[0] >= 16:
                floor_contact = (
                    state[14].item() if state.shape[0] > 14 else -1.0
                )  # Floor contact
                wall_contact = (
                    state[15].item() if state.shape[0] > 15 else -1.0
                )  # Wall contact

                # Dangerous combinations from actual game physics
                # High speed + no surface contact = potential terminal impact
                if (
                    actual_velocity_mag > 5.0
                    and airborne_status > 0.0
                    and floor_contact < 0.0
                    and wall_contact < 0.0
                ):
                    free_fall_risk = min(actual_velocity_mag / 8.0, 1.0)
                    death_risk += free_fall_risk * 0.5

                # Wall sliding at high speed can lead to dangerous situations
                if wall_interaction > 0.0 and actual_velocity_mag > 4.0:
                    wall_slide_risk = min((actual_velocity_mag - 4.0) / 4.0, 1.0)
                    death_risk += wall_slide_risk * 0.3

        # Apply death risk with horizon scaling and confidence adjustment
        # Higher risk = higher probability of death within horizon steps
        death_probability = min(
            death_risk, 0.9
        )  # Cap at 90% (actual physics can be confident)

        # Apply horizon scaling: risk decreases with longer prediction horizon
        horizon_factor = max(
            0.3, 1.0 - (horizon - 5) * 0.1
        )  # Reduce confidence for longer horizons
        adjusted_probability = death_probability * horizon_factor

        # Only label if significant risk (threshold based on actual game mechanics)
        if (
            adjusted_probability > 0.15
        ):  # Lower threshold since we're using actual physics
            death_labels[i] = adjusted_probability

    return death_labels


def compute_auxiliary_labels(
    trajectory: Dict[str, torch.Tensor],
    death_horizon: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute death prediction labels from trajectory data.

    This function processes trajectory data to generate labels for
    death prediction using hindsight information.

    Args:
        trajectory: Dictionary containing:
            - observations: [T, obs_dim] or dict with "death_context"
            - actions: [T] (optional)
            - returns: [T] (optional)
            - rewards: [T] (optional)
            - dones: [T] (optional)
            - infos: List of T info dicts (optional)
        death_horizon: Number of steps to look ahead for death prediction

    Returns:
        Dictionary with keys:
            - death_labels: [T] (1 if died within horizon, 0 otherwise)
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

    # 1. Death prediction labels using physics-based forward prediction
    # Handle case where observations might be a dict or tensor
    observations = trajectory["observations"]
    if isinstance(observations, dict):
        death_labels = compute_death_labels_from_physics(
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

    return {
        "death_labels": death_labels,
    }


def compute_auxiliary_losses(
    predictions: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    weights: Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute death prediction loss.

    Args:
        predictions: Dictionary with prediction outputs
        labels: Dictionary with ground truth labels
        weights: Optional loss weights for death prediction

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    if weights is None:
        weights = {
            "death": 0.01,  # Reduced weight for stability
        }

    losses = {}

    # Death prediction loss (binary cross-entropy)
    if "death_prob" in predictions and "death_labels" in labels:
        death_pred = predictions["death_prob"].squeeze(-1)
        death_target = labels["death_labels"]
        losses["death"] = nn.functional.binary_cross_entropy(
            death_pred, death_target, reduction="mean"
        )

    # Compute weighted total loss
    total_loss = sum(weights.get(k, 0.01) * v for k, v in losses.items())

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
