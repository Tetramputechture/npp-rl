"""Loss functions for training the path predictor network.

This module implements loss functions for:
1. Waypoint prediction - matching expert trajectories
2. Path confidence calibration - ranking expert paths highest
3. Path diversity - encouraging diverse candidate paths

IMPORTANT: All loss functions operate on NORMALIZED [0, 1] coordinates.
Distance thresholds and diversity metrics must be scaled accordingly.
For reference: 50 pixels ≈ 0.047 in normalized space (1056px width)
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PathPredictionLoss:
    """Container for path prediction loss components."""

    total_loss: torch.Tensor
    waypoint_loss: torch.Tensor
    confidence_loss: torch.Tensor
    diversity_loss: torch.Tensor

    def backward(self):
        """Convenience method to call backward on total loss."""
        self.total_loss.backward()

    def item_dict(self) -> Dict[str, float]:
        """Convert losses to dictionary of scalars."""
        return {
            "total_loss": self.total_loss.item(),
            "waypoint_loss": self.waypoint_loss.item(),
            "confidence_loss": self.confidence_loss.item(),
            "diversity_loss": self.diversity_loss.item(),
        }


def chamfer_distance(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    pred_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute Chamfer distance between two point sets.

    Chamfer distance is permutation-invariant and handles variable-length sequences.
    For each predicted point, finds nearest target point, and vice versa.

    Note: Coordinates are expected to be in NORMALIZED [0, 1] space.
    Distances will be in normalized units (e.g., 0.05 ≈ 50 pixels on 1000px scale).

    Args:
        pred_points: Predicted waypoints [batch, num_pred, 2] (normalized [0,1] coordinates)
        target_points: Target waypoints [batch, num_target, 2] (normalized [0,1] coordinates)
        pred_mask: Mask for valid predicted points [batch, num_pred]
        target_mask: Mask for valid target points [batch, num_target]

    Returns:
        Chamfer distance averaged over batch (returns large constant instead of inf if all masked)
    """
    # DEFENSIVE: Check if all masks are False (no valid waypoints)
    if pred_mask is not None and not pred_mask.any():
        logger.warning(
            "chamfer_distance: All predicted waypoints masked out (pred_mask all False)"
        )
        return torch.tensor(10.0, device=pred_points.device, dtype=pred_points.dtype)

    if target_mask is not None and not target_mask.any():
        logger.warning(
            "chamfer_distance: All target waypoints masked out (target_mask all False)"
        )
        return torch.tensor(10.0, device=pred_points.device, dtype=pred_points.dtype)

    # Compute pairwise distances [batch, num_pred, num_target]
    pred_expanded = pred_points.unsqueeze(2)  # [batch, num_pred, 1, 2]
    target_expanded = target_points.unsqueeze(1)  # [batch, 1, num_target, 2]

    pairwise_dist = torch.sum(
        (pred_expanded - target_expanded) ** 2, dim=-1
    )  # [batch, num_pred, num_target]

    # Apply masks if provided
    if pred_mask is not None:
        pred_mask = pred_mask.unsqueeze(2)  # [batch, num_pred, 1]
        pairwise_dist = pairwise_dist.masked_fill(~pred_mask, float("inf"))

    if target_mask is not None:
        target_mask = target_mask.unsqueeze(1)  # [batch, 1, num_target]
        pairwise_dist = pairwise_dist.masked_fill(~target_mask, float("inf"))

    # Chamfer distance: min over target for each pred + min over pred for each target
    pred_to_target = torch.min(pairwise_dist, dim=2)[0]  # [batch, num_pred]
    target_to_pred = torch.min(pairwise_dist, dim=1)[0]  # [batch, num_target]

    # DEFENSIVE: Check for inf values ONLY in valid points (invalid points are expected to have inf)
    if pred_mask is not None:
        pred_mask_squeezed = pred_mask.squeeze(2)  # [batch, num_pred]
        # Check if any VALID predicted points have inf distance
        valid_pred_has_inf = torch.isinf(pred_to_target) & pred_mask_squeezed
        if valid_pred_has_inf.any():
            logger.warning(
                "chamfer_distance: Inf detected in VALID predicted points (no valid targets reachable)"
            )
            num_valid_pred = pred_mask_squeezed.sum().item()
            num_inf_valid = valid_pred_has_inf.sum().item()
            logger.warning(
                f"  {num_inf_valid}/{num_valid_pred} valid predicted points have inf distance"
            )
            return torch.tensor(
                10.0,
                device=pred_points.device,
                dtype=pred_points.dtype,
                requires_grad=True,
            )

    if target_mask is not None:
        target_mask_squeezed = target_mask.squeeze(1)  # [batch, num_target]
        # Check if any VALID target points have inf distance
        valid_target_has_inf = torch.isinf(target_to_pred) & target_mask_squeezed
        if valid_target_has_inf.any():
            logger.warning(
                "chamfer_distance: Inf detected in VALID target points (no valid preds reachable)"
            )
            num_valid_target = target_mask_squeezed.sum().item()
            num_inf_valid = valid_target_has_inf.sum().item()
            logger.warning(
                f"  {num_inf_valid}/{num_valid_target} valid target points have inf distance"
            )
            return torch.tensor(
                10.0,
                device=pred_points.device,
                dtype=pred_points.dtype,
                requires_grad=True,
            )

    # Average over valid points
    if pred_mask is not None:
        pred_mask = pred_mask.squeeze(2)  # [batch, num_pred]
        pred_to_target = pred_to_target.masked_fill(~pred_mask, 0.0)
        num_valid_pred = pred_mask.sum(dim=1).clamp(min=1)
        pred_to_target = pred_to_target.sum(dim=1) / num_valid_pred  # [batch]
    else:
        pred_to_target = pred_to_target.mean(dim=1)  # [batch]

    if target_mask is not None:
        target_mask = target_mask.squeeze(1)  # [batch, num_target]
        target_to_pred = target_to_pred.masked_fill(~target_mask, 0.0)
        num_valid_target = target_mask.sum(dim=1).clamp(min=1)
        target_to_pred = target_to_pred.sum(dim=1) / num_valid_target  # [batch]
    else:
        target_to_pred = target_to_pred.mean(dim=1)  # [batch]

    # Combine both directions - both should now be [batch]
    chamfer_dist = pred_to_target + target_to_pred  # [batch]

    # DEFENSIVE: Final check for nan/inf
    mean_dist = chamfer_dist.mean()
    if torch.isnan(mean_dist) or torch.isinf(mean_dist):
        logger.warning(
            f"chamfer_distance: Final result is {mean_dist}, returning large constant"
        )
        return torch.tensor(10.0, device=pred_points.device, dtype=pred_points.dtype)

    return mean_dist  # scalar


def waypoint_prediction_loss(
    predicted_paths: torch.Tensor,
    expert_waypoints: torch.Tensor,
    pred_masks: torch.Tensor,
    expert_masks: torch.Tensor,
    use_chamfer: bool = True,
) -> torch.Tensor:
    """Compute waypoint prediction loss.

    Compares predicted waypoint sequences to expert trajectories.
    Can use either Chamfer distance (permutation-invariant) or MSE (ordered).

    Args:
        predicted_paths: Predicted waypoints [batch, num_candidates, max_waypoints, 2]
        expert_waypoints: Expert waypoints [batch, max_waypoints, 2]
        pred_masks: Mask for valid predicted waypoints [batch, num_candidates, max_waypoints]
        expert_masks: Mask for valid expert waypoints [batch, max_waypoints]
        use_chamfer: If True, use Chamfer distance. If False, use MSE.

    Returns:
        Waypoint prediction loss (scalar)
    """
    num_candidates = predicted_paths.size(1)

    # For each candidate path, compute distance to expert
    losses = []

    for cand_idx in range(num_candidates):
        cand_waypoints = predicted_paths[:, cand_idx, :, :]  # [batch, max_waypoints, 2]
        cand_mask = pred_masks[:, cand_idx, :]  # [batch, max_waypoints]

        if use_chamfer:
            # Permutation-invariant Chamfer distance
            loss = chamfer_distance(
                cand_waypoints,
                expert_waypoints,
                cand_mask,
                expert_masks,
            )
        else:
            # Ordered MSE loss
            diff = (cand_waypoints - expert_waypoints) ** 2
            diff = diff.sum(dim=-1)  # [batch, max_waypoints]

            # Apply masks
            valid_mask = cand_mask & expert_masks
            diff = diff.masked_fill(~valid_mask, 0.0)

            # Average over valid points
            loss = diff.sum() / valid_mask.sum().clamp(min=1)

        losses.append(loss)

    # Take mean loss across all candidates to ensure all heads are trained
    # This provides gradient signal to all heads, allowing diversity loss to differentiate them
    mean_loss = torch.stack(losses).mean()

    return mean_loss


def confidence_calibration_loss(
    path_confidences: torch.Tensor,
    predicted_paths: torch.Tensor,
    expert_waypoints: torch.Tensor,
    pred_masks: torch.Tensor,
    expert_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute confidence calibration loss.

    Ensures the path closest to expert has highest confidence.
    Uses ranking loss to encourage proper confidence ordering.

    Args:
        path_confidences: Confidence scores [batch, num_candidates]
        predicted_paths: Predicted waypoints [batch, num_candidates, max_waypoints, 2]
        expert_waypoints: Expert waypoints [batch, max_waypoints, 2]
        pred_masks: Mask for predicted waypoints [batch, num_candidates, max_waypoints]
        expert_masks: Mask for expert waypoints [batch, max_waypoints]

    Returns:
        Confidence calibration loss (scalar)
    """
    batch_size, num_candidates = path_confidences.size()

    # Compute distance of each candidate to expert for each batch item
    all_distances = []

    for cand_idx in range(num_candidates):
        cand_waypoints = predicted_paths[:, cand_idx, :, :]  # [batch, max_waypoints, 2]
        cand_mask = pred_masks[:, cand_idx, :]  # [batch, max_waypoints]

        # Compute Chamfer distance per batch item
        batch_distances = []
        for batch_idx in range(batch_size):
            dist = chamfer_distance(
                cand_waypoints[batch_idx : batch_idx + 1],  # [1, max_waypoints, 2]
                expert_waypoints[batch_idx : batch_idx + 1],  # [1, max_waypoints, 2]
                cand_mask[batch_idx : batch_idx + 1],  # [1, max_waypoints]
                expert_masks[batch_idx : batch_idx + 1],  # [1, max_waypoints]
            )
            batch_distances.append(dist)

        # Stack to [batch]
        cand_distances = torch.stack(batch_distances)
        all_distances.append(cand_distances)

    # Stack to [num_candidates, batch] then transpose to [batch, num_candidates]
    distances = torch.stack(all_distances, dim=0).transpose(
        0, 1
    )  # [batch, num_candidates]

    # Find best candidate per batch item
    best_candidate_indices = torch.argmin(distances, dim=1)  # [batch]

    # Create target: one-hot encoding for best candidate per batch item
    target = torch.zeros(batch_size, num_candidates, device=path_confidences.device)
    target[torch.arange(batch_size), best_candidate_indices] = 1.0

    # Cross-entropy loss (confidence should be highest for best match)
    log_probs = F.log_softmax(path_confidences, dim=1)
    loss = -(target * log_probs).sum(dim=1).mean()

    return loss


def path_diversity_loss(
    predicted_paths: torch.Tensor,
    pred_masks: torch.Tensor,
    min_diversity: float = 0.05,
) -> torch.Tensor:
    """Compute path diversity loss.

    Encourages different candidate paths to be spatially diverse.
    Uses multiple diversity metrics:
    1. Trajectory diversity (Chamfer distance between full paths)
    2. Endpoint diversity (distance between path endpoints)
    3. Direction diversity (variation in path directions)

    IMPORTANT: min_diversity is in NORMALIZED [0, 1] coordinate space.
    Default 0.05 ≈ 50 pixels on 1056px width level.

    Args:
        predicted_paths: Predicted waypoints [batch, num_candidates, max_waypoints, 2]
                        in NORMALIZED [0, 1] coordinates
        pred_masks: Mask for predicted waypoints [batch, num_candidates, max_waypoints]
        min_diversity: Minimum distance between path pairs in normalized space
                      (default: 0.05 ≈ 50 pixels)

    Returns:
        Path diversity loss (scalar)
    """
    batch_size, num_candidates = predicted_paths.size(0), predicted_paths.size(1)

    if num_candidates < 2:
        return torch.tensor(0.0, device=predicted_paths.device)

    # Compute pairwise distances between all candidate paths
    trajectory_penalties = []
    endpoint_penalties = []

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            path_i = predicted_paths[:, i, :, :]
            path_j = predicted_paths[:, j, :, :]
            mask_i = pred_masks[:, i, :]
            mask_j = pred_masks[:, j, :]

            # 1. Trajectory diversity: Chamfer distance between full paths
            traj_dist = chamfer_distance(path_i, path_j, mask_i, mask_j)
            traj_penalty = F.relu(min_diversity - traj_dist)
            trajectory_penalties.append(traj_penalty)

            # 2. Endpoint diversity: Distance between path endpoints
            # Find last valid waypoint for each path
            # Sum valid waypoints to get indices
            valid_counts_i = mask_i.sum(dim=1)  # [batch]
            valid_counts_j = mask_j.sum(dim=1)  # [batch]

            # Get endpoints for each batch item
            endpoint_dists = []
            for b in range(batch_size):
                if valid_counts_i[b] > 0 and valid_counts_j[b] > 0:
                    last_idx_i = int(valid_counts_i[b].item()) - 1
                    last_idx_j = int(valid_counts_j[b].item()) - 1
                    endpoint_i = path_i[b, last_idx_i]  # [2]
                    endpoint_j = path_j[b, last_idx_j]  # [2]
                    ep_dist = torch.sqrt(((endpoint_i - endpoint_j) ** 2).sum())
                    endpoint_dists.append(ep_dist)

            if endpoint_dists:
                avg_endpoint_dist = torch.stack(endpoint_dists).mean()
                endpoint_penalty = F.relu(min_diversity * 0.5 - avg_endpoint_dist)
                endpoint_penalties.append(endpoint_penalty)

    # Combine diversity penalties
    total_penalty = torch.tensor(0.0, device=predicted_paths.device)

    if trajectory_penalties:
        total_penalty = total_penalty + torch.stack(trajectory_penalties).mean()

    if endpoint_penalties:
        # Endpoint diversity is secondary but still important
        total_penalty = total_penalty + 0.5 * torch.stack(endpoint_penalties).mean()

    return total_penalty


def compute_path_prediction_loss(
    predicted_paths: torch.Tensor,
    path_confidences: torch.Tensor,
    expert_waypoints: torch.Tensor,
    pred_masks: torch.Tensor,
    expert_masks: torch.Tensor,
    waypoint_loss_weight: float = 1.0,
    confidence_loss_weight: float = 0.5,
    diversity_loss_weight: float = 0.3,
) -> PathPredictionLoss:
    """Compute complete path prediction loss.

    Combines waypoint prediction, confidence calibration, and path diversity.

    Args:
        predicted_paths: Predicted waypoints [batch, num_candidates, max_waypoints, 2]
        path_confidences: Confidence scores [batch, num_candidates]
        expert_waypoints: Expert waypoints [batch, max_waypoints, 2]
        pred_masks: Mask for predicted waypoints [batch, num_candidates, max_waypoints]
        expert_masks: Mask for expert waypoints [batch, max_waypoints]
        waypoint_loss_weight: Weight for waypoint prediction loss
        confidence_loss_weight: Weight for confidence calibration loss
        diversity_loss_weight: Weight for path diversity loss

    Returns:
        PathPredictionLoss with all components (returns large fallback loss if data invalid)
    """
    # DEFENSIVE: Validate inputs before computing loss
    # Check for nan/inf in coordinates (normalized [0,1] space)
    if torch.isnan(predicted_paths).any() or torch.isinf(predicted_paths).any():
        logger.warning(
            "compute_path_prediction_loss: NaN/Inf detected in predicted_paths"
        )
        logger.warning(
            f"  predicted_paths stats: min={predicted_paths.min()}, max={predicted_paths.max()}"
        )
        fallback_loss = torch.tensor(
            10.0, device=predicted_paths.device, requires_grad=True
        )
        zero_loss = torch.tensor(0.0, device=predicted_paths.device, requires_grad=True)
        return PathPredictionLoss(
            total_loss=fallback_loss,
            waypoint_loss=fallback_loss,
            confidence_loss=fallback_loss,
            diversity_loss=zero_loss,
        )

    if torch.isnan(expert_waypoints).any() or torch.isinf(expert_waypoints).any():
        logger.warning(
            "compute_path_prediction_loss: NaN/Inf detected in expert_waypoints"
        )
        logger.warning(
            f"  expert_waypoints stats: min={expert_waypoints.min()}, max={expert_waypoints.max()}"
        )
        fallback_loss = torch.tensor(
            10.0, device=predicted_paths.device, requires_grad=True
        )
        zero_loss = torch.tensor(0.0, device=predicted_paths.device, requires_grad=True)
        return PathPredictionLoss(
            total_loss=fallback_loss,
            waypoint_loss=fallback_loss,
            confidence_loss=fallback_loss,
            diversity_loss=zero_loss,
        )

    # Check if expert masks are all False (no valid expert waypoints)
    if not expert_masks.any():
        logger.warning(
            "compute_path_prediction_loss: All expert waypoints masked out (expert_masks all False)"
        )
        logger.warning(
            f"  Batch size: {expert_masks.shape[0]}, Max waypoints: {expert_masks.shape[1]}"
        )
        logger.warning(f"  Expert waypoints sample: {expert_waypoints[0, :5]}")
        fallback_loss = torch.tensor(
            10.0, device=predicted_paths.device, requires_grad=True
        )
        zero_loss = torch.tensor(0.0, device=predicted_paths.device, requires_grad=True)
        return PathPredictionLoss(
            total_loss=fallback_loss,
            waypoint_loss=fallback_loss,
            confidence_loss=fallback_loss,
            diversity_loss=zero_loss,
        )

    # Check if pred masks are all False (no valid predictions)
    if not pred_masks.any():
        logger.warning(
            "compute_path_prediction_loss: All predicted waypoints masked out (pred_masks all False)"
        )
        fallback_loss = torch.tensor(
            10.0, device=predicted_paths.device, requires_grad=True
        )
        zero_loss = torch.tensor(0.0, device=predicted_paths.device, requires_grad=True)
        return PathPredictionLoss(
            total_loss=fallback_loss,
            waypoint_loss=fallback_loss,
            confidence_loss=fallback_loss,
            diversity_loss=zero_loss,
        )

    # Compute individual loss components
    waypoint_loss = waypoint_prediction_loss(
        predicted_paths,
        expert_waypoints,
        pred_masks,
        expert_masks,
        use_chamfer=True,
    )

    confidence_loss = confidence_calibration_loss(
        path_confidences,
        predicted_paths,
        expert_waypoints,
        pred_masks,
        expert_masks,
    )

    diversity_loss = path_diversity_loss(
        predicted_paths,
        pred_masks,
        min_diversity=0.05,  # 0.05 in normalized space ≈ 50 pixels
    )

    # Weighted combination
    total_loss = (
        waypoint_loss_weight * waypoint_loss
        + confidence_loss_weight * confidence_loss
        + diversity_loss_weight * diversity_loss
    )

    # DEFENSIVE: Final check for nan/inf in total loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.warning(f"compute_path_prediction_loss: total_loss is {total_loss}")
        logger.warning(f"  waypoint_loss: {waypoint_loss}")
        logger.warning(f"  confidence_loss: {confidence_loss}")
        logger.warning(f"  diversity_loss: {diversity_loss}")
        fallback_loss = torch.tensor(
            10.0, device=predicted_paths.device, requires_grad=True
        )
        zero_loss = torch.tensor(0.0, device=predicted_paths.device, requires_grad=True)
        return PathPredictionLoss(
            total_loss=fallback_loss,
            waypoint_loss=waypoint_loss
            if torch.isfinite(waypoint_loss)
            else fallback_loss,
            confidence_loss=confidence_loss
            if torch.isfinite(confidence_loss)
            else fallback_loss,
            diversity_loss=diversity_loss
            if torch.isfinite(diversity_loss)
            else zero_loss,
        )

    return PathPredictionLoss(
        total_loss=total_loss,
        waypoint_loss=waypoint_loss,
        confidence_loss=confidence_loss,
        diversity_loss=diversity_loss,
    )


def compute_graph_validation_loss(
    predicted_paths: torch.Tensor,
    pred_masks: torch.Tensor,
    adjacency_graphs: List[Dict],
    weight: float = 0.1,
) -> torch.Tensor:
    """Compute loss for graph-based path validation.

    Penalizes paths that traverse non-adjacent graph nodes.
    This encourages physically feasible paths.

    Args:
        predicted_paths: Predicted waypoints [batch, num_candidates, max_waypoints, 2]
        pred_masks: Mask for predicted waypoints [batch, num_candidates, max_waypoints]
        adjacency_graphs: List of adjacency dicts (one per batch item)
        weight: Loss weight

    Returns:
        Graph validation loss (scalar)
    """
    if not adjacency_graphs:
        return torch.tensor(0.0, device=predicted_paths.device)

    batch_size, num_candidates, max_waypoints = predicted_paths.size()[:3]
    device = predicted_paths.device

    total_violations = 0
    total_edges = 0

    # For each sample in batch
    for batch_idx in range(batch_size):
        adjacency = adjacency_graphs[batch_idx]
        if adjacency is None:
            continue

        # For each candidate path
        for cand_idx in range(num_candidates):
            waypoints = predicted_paths[batch_idx, cand_idx]  # [max_waypoints, 2]
            mask = pred_masks[batch_idx, cand_idx]  # [max_waypoints]

            # Check consecutive waypoints
            for i in range(max_waypoints - 1):
                if not mask[i] or not mask[i + 1]:
                    continue

                # Check if edge exists in graph
                node_i = tuple(waypoints[i].detach().cpu().numpy().astype(int))
                node_j = tuple(waypoints[i + 1].detach().cpu().numpy().astype(int))

                # Check adjacency (both directions)
                is_adjacent = False
                if node_i in adjacency:
                    neighbors = adjacency[node_i]
                    for neighbor_info in neighbors:
                        if isinstance(neighbor_info, tuple):
                            neighbor_pos = (neighbor_info[0], neighbor_info[1])
                            if neighbor_pos == node_j:
                                is_adjacent = True
                                break

                total_edges += 1
                if not is_adjacent:
                    total_violations += 1

    if total_edges == 0:
        return torch.tensor(0.0, device=device)

    # Violation rate as loss
    violation_rate = total_violations / total_edges
    loss = weight * violation_rate

    return torch.tensor(loss, device=device, requires_grad=False)
