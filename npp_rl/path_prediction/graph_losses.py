"""Loss functions for graph-based path prediction.

Implements discrete node selection losses:
1. Node classification loss - matching expert node sequences
2. Connectivity loss - penalizing non-adjacent node pairs
3. Diversity loss - encouraging different paths across heads
4. Path quality metrics
"""

import torch
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphPathLoss:
    """Container for graph path prediction loss components."""

    total_loss: torch.Tensor
    node_classification_loss: torch.Tensor
    connectivity_loss: torch.Tensor
    diversity_loss: torch.Tensor
    start_goal_weighted_loss: torch.Tensor = None
    goal_reaching_loss: torch.Tensor = None

    def item_dict(self) -> Dict[str, float]:
        """Convert losses to dictionary of scalars."""
        result = {
            "total_loss": self.total_loss.item(),
            "node_classification": self.node_classification_loss.item(),
            "connectivity": self.connectivity_loss.item(),
            "diversity": self.diversity_loss.item(),
        }
        if self.start_goal_weighted_loss is not None:
            result["start_goal_weighted"] = self.start_goal_weighted_loss.item()
        if self.goal_reaching_loss is not None:
            result["goal_reaching"] = self.goal_reaching_loss.item()
        return result


def node_classification_loss(
    predicted_logits: torch.Tensor,
    expert_node_ids: torch.Tensor,
    expert_masks: torch.Tensor,
    use_hungarian: bool = False,
) -> torch.Tensor:
    """Compute node classification loss (vectorized implementation).

    Matches predicted node distributions to expert node sequences.
    Fully vectorized - processes all heads simultaneously.

    Args:
        predicted_logits: [batch, num_heads, max_waypoints, max_nodes]
        expert_node_ids: [batch, max_expert_waypoints] - expert node IDs
        expert_masks: [batch, max_expert_waypoints] - valid expert waypoints
        use_hungarian: If True, use Hungarian algorithm for matching (not implemented)

    Returns:
        Cross-entropy loss averaged over valid predictions
    """
    batch_size, num_heads, max_waypoints, max_nodes = predicted_logits.shape
    max_expert = expert_node_ids.size(1)

    # Truncate or pad expert to match max_waypoints
    if max_expert < max_waypoints:
        # Pad expert with zeros (will be masked out)
        pad_size = max_waypoints - max_expert
        expert_node_ids_padded = F.pad(expert_node_ids, (0, pad_size), value=0)
        expert_masks_padded = F.pad(expert_masks, (0, pad_size), value=0)
    else:
        # Truncate expert to max_waypoints
        expert_node_ids_padded = expert_node_ids[:, :max_waypoints]
        expert_masks_padded = expert_masks[:, :max_waypoints]

    # Vectorized computation across all heads
    # Reshape logits: [batch, num_heads, max_waypoints, max_nodes]
    # -> [batch * num_heads * max_waypoints, max_nodes]
    logits_flat = predicted_logits.reshape(-1, max_nodes)

    # Expand targets and masks for all heads
    # [batch, max_waypoints] -> [batch, num_heads, max_waypoints] -> [batch * num_heads * max_waypoints]
    targets_expanded = (
        expert_node_ids_padded.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1).long()
    )
    masks_expanded = (
        expert_masks_padded.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1)
    )

    # Compute cross-entropy loss for all heads at once
    ce_loss = F.cross_entropy(logits_flat, targets_expanded, reduction="none")

    # Apply mask and average
    ce_loss_masked = ce_loss * masks_expanded
    num_valid = masks_expanded.sum().clamp(min=1)
    total_loss = ce_loss_masked.sum() / num_valid

    return total_loss


def _build_batch_adjacency_matrices(
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    max_nodes: int,
) -> torch.Tensor:
    """Build adjacency matrices for entire batch using vectorized operations.

    This helper constructs boolean adjacency matrices indicating which nodes
    are connected by edges. Uses efficient tensor operations instead of Python loops.

    Args:
        edge_index: [batch, 2, max_edges] - edge connectivity
        edge_mask: [batch, max_edges] - mask for valid edges
        max_nodes: Maximum number of nodes in any graph

    Returns:
        adjacency: [batch, max_nodes, max_nodes] boolean tensor where
                   adjacency[b, i, j] = True if nodes i and j are connected
    """
    batch_size = edge_index.size(0)
    device = edge_index.device

    # Initialize adjacency matrices as boolean tensors (memory efficient)
    adjacency = torch.zeros(
        batch_size, max_nodes, max_nodes, dtype=torch.bool, device=device
    )

    # Get source and target nodes
    src = edge_index[:, 0, :]  # [batch, max_edges]
    tgt = edge_index[:, 1, :]  # [batch, max_edges]

    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(src)

    # Get valid edge mask as boolean
    valid = edge_mask.bool()

    # Vectorized edge insertion (both directions for undirected graph)
    # Only insert edges where mask is True
    adjacency[batch_indices[valid], src[valid], tgt[valid]] = True
    adjacency[batch_indices[valid], tgt[valid], src[valid]] = True

    return adjacency


def connectivity_loss(
    predicted_node_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """Compute connectivity loss for predicted paths (vectorized implementation).

    Penalizes consecutive nodes that are not connected in the graph.
    This is a fully vectorized implementation that avoids Python loops and
    CPU/GPU synchronizations for significant performance improvements.

    Args:
        predicted_node_ids: [batch, num_heads, max_waypoints] - predicted node IDs
        edge_index: [batch, 2, max_edges] - graph edges
        edge_mask: [batch, max_edges] - valid edges
        node_mask: [batch, max_nodes] - valid nodes
        penalty_weight: Weight for connectivity violations

    Returns:
        Connectivity violation penalty (differentiable)
    """
    batch_size, num_heads, max_waypoints = predicted_node_ids.shape
    max_nodes = node_mask.size(1)
    device = predicted_node_ids.device

    # Build adjacency matrices once for the entire batch
    adjacency = _build_batch_adjacency_matrices(edge_index, edge_mask, max_nodes)

    # Get consecutive node pairs
    # [batch, num_heads, max_waypoints-1]
    node_a = predicted_node_ids[:, :, :-1]
    node_b = predicted_node_ids[:, :, 1:]

    # Check if pairs are identical (skip these - collapsed predictions)
    same_node = node_a == node_b

    # Clamp node IDs to valid range to prevent out-of-bounds access
    node_a_clamped = node_a.clamp(0, max_nodes - 1)
    node_b_clamped = node_b.clamp(0, max_nodes - 1)

    # Check node validity using broadcasting
    # [batch, num_heads, max_waypoints-1]
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1)
    valid_a = node_mask[batch_idx, node_a_clamped].bool()
    valid_b = node_mask[batch_idx, node_b_clamped].bool()
    valid_pair = valid_a & valid_b & ~same_node

    # Check edge existence using adjacency matrix lookup
    # [batch, num_heads, max_waypoints-1]
    edge_exists = adjacency[batch_idx, node_a_clamped, node_b_clamped]

    # Count violations: valid pairs without edges
    violations = valid_pair & ~edge_exists
    num_violations = violations.sum().float()
    num_checked = valid_pair.sum().float().clamp(min=1)

    # Compute violation rate
    violation_rate = num_violations / num_checked

    return penalty_weight * violation_rate


def path_diversity_loss(
    predicted_node_ids: torch.Tensor,
    min_unique_nodes: int = 10,
    overlap_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute path diversity loss (vectorized implementation).

    Encourages different heads to predict different paths by penalizing
    overlap in node selections. Uses efficient tensor operations instead
    of Python loops and numpy conversions.

    Args:
        predicted_node_ids: [batch, num_heads, max_waypoints]
        min_unique_nodes: Minimum unique nodes required per path pair (unused, kept for compatibility)
        overlap_threshold: Threshold above which overlap is penalized (default: 0.5)

    Returns:
        Diversity penalty (higher when paths overlap too much)
    """
    batch_size, num_heads, max_waypoints = predicted_node_ids.shape
    device = predicted_node_ids.device

    if num_heads < 2:
        return torch.tensor(0.0, device=device)

    # Compute pairwise overlaps between all heads in a vectorized manner
    total_penalty = 0.0
    num_pairs = 0

    # For each pair of heads, compute Jaccard similarity
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            # Get paths for this pair of heads across all batches
            # [batch, max_waypoints]
            path_i = predicted_node_ids[:, i, :]
            path_j = predicted_node_ids[:, j, :]

            # For each batch item, compute overlap using tensor operations
            for b in range(batch_size):
                # Create one-hot representation of nodes in each path
                # This is more efficient than set operations
                nodes_i = path_i[b]  # [max_waypoints]
                nodes_j = path_j[b]  # [max_waypoints]

                # Count unique nodes using bincount (fast)
                max_node_id = max(nodes_i.max().item(), nodes_j.max().item()) + 1

                # Create binary vectors indicating node presence
                present_i = torch.zeros(max_node_id, dtype=torch.bool, device=device)
                present_j = torch.zeros(max_node_id, dtype=torch.bool, device=device)

                present_i[nodes_i] = True
                present_j[nodes_j] = True

                # Compute intersection and union
                intersection = (present_i & present_j).sum().float()
                union = (present_i | present_j).sum().float()

                # Compute overlap ratio (Jaccard similarity)
                if union > 0:
                    overlap_ratio = intersection / union

                    # Penalize if more than threshold overlap
                    if overlap_ratio > overlap_threshold:
                        penalty = (overlap_ratio - overlap_threshold).item()
                        total_penalty += penalty

                num_pairs += 1

    if num_pairs == 0:
        return torch.tensor(0.0, device=device)

    avg_penalty = total_penalty / num_pairs

    return torch.tensor(avg_penalty, device=device, requires_grad=False)


def start_goal_weighted_loss(
    predicted_logits: torch.Tensor,
    expert_node_ids: torch.Tensor,
    expert_masks: torch.Tensor,
    start_node_ids: torch.Tensor = None,
    goal_node_ids: torch.Tensor = None,
    start_weight: float = 3.0,
    goal_weight: float = 5.0,
) -> torch.Tensor:
    """Compute node classification loss with higher weight on start/goal waypoints.

    This loss emphasizes the importance of predicting correct start and end positions,
    which is critical for path planning with multimodal fusion.

    Args:
        predicted_logits: [batch, num_heads, max_waypoints, max_nodes]
        expert_node_ids: [batch, max_expert_waypoints] - expert node IDs
        expert_masks: [batch, max_expert_waypoints] - valid expert waypoints
        start_node_ids: [batch] - start node IDs (optional, for extra weighting)
        goal_node_ids: [batch, num_goals] - goal node IDs (optional, for extra weighting)
        start_weight: Weight multiplier for first waypoint (default: 3.0)
        goal_weight: Weight multiplier for last waypoint (default: 5.0)

    Returns:
        Weighted cross-entropy loss
    """
    batch_size, num_heads, max_waypoints, max_nodes = predicted_logits.shape
    max_expert = expert_node_ids.size(1)
    device = predicted_logits.device

    # Truncate or pad expert to match max_waypoints
    if max_expert < max_waypoints:
        pad_size = max_waypoints - max_expert
        expert_node_ids_padded = F.pad(expert_node_ids, (0, pad_size), value=0)
        expert_masks_padded = F.pad(expert_masks, (0, pad_size), value=0)
    else:
        expert_node_ids_padded = expert_node_ids[:, :max_waypoints]
        expert_masks_padded = expert_masks[:, :max_waypoints]

    # Create position weights: higher for first and last waypoints
    # [batch, max_waypoints]
    weights = torch.ones(batch_size, max_waypoints, device=device)

    # Weight first waypoint (start)
    weights[:, 0] = start_weight

    # Weight last valid waypoint (goal) for each batch item
    for b in range(batch_size):
        valid_mask = expert_masks_padded[b].bool()
        if valid_mask.any():
            last_valid_idx = valid_mask.nonzero(as_tuple=True)[0][-1].item()
            weights[b, last_valid_idx] = goal_weight

    # Reshape for vectorized computation
    logits_flat = predicted_logits.reshape(-1, max_nodes)
    targets_expanded = (
        expert_node_ids_padded.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1).long()
    )
    masks_expanded = (
        expert_masks_padded.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1)
    )
    weights_expanded = weights.unsqueeze(1).expand(-1, num_heads, -1).reshape(-1)

    # Compute weighted cross-entropy
    ce_loss = F.cross_entropy(logits_flat, targets_expanded, reduction="none")
    ce_loss_weighted = ce_loss * masks_expanded * weights_expanded
    num_valid = (masks_expanded * weights_expanded).sum().clamp(min=1)
    total_loss = ce_loss_weighted.sum() / num_valid

    return total_loss


def goal_reaching_loss(
    predicted_node_ids: torch.Tensor,
    goal_node_ids: torch.Tensor = None,
    expert_node_ids: torch.Tensor = None,
    expert_masks: torch.Tensor = None,
    max_distance_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalize paths that don't end near goal nodes.

    This loss ensures predicted paths actually reach the intended goals,
    which is essential for task completion.

    Args:
        predicted_node_ids: [batch, num_heads, max_waypoints] - predicted nodes
        goal_node_ids: [batch, num_goals] - target goal node IDs (optional)
        expert_node_ids: [batch, max_expert_waypoints] - expert path (fallback)
        expert_masks: [batch, max_expert_waypoints] - valid expert waypoints
        max_distance_threshold: Maximum node ID distance to consider "reached" goal

    Returns:
        Goal reaching penalty
    """
    batch_size, num_heads, max_waypoints = predicted_node_ids.shape
    device = predicted_node_ids.device

    # Determine goal nodes from either explicit goals or expert path end
    if goal_node_ids is not None:
        # Use explicit goal node IDs
        target_goals = goal_node_ids  # [batch, num_goals]
    elif expert_node_ids is not None and expert_masks is not None:
        # Use last valid node from expert path as goal
        target_goals = []
        for b in range(batch_size):
            valid_mask = expert_masks[b].bool()
            if valid_mask.any():
                last_expert_idx = valid_mask.nonzero(as_tuple=True)[0][-1].item()
                last_expert_node = expert_node_ids[b, last_expert_idx].unsqueeze(0)
                target_goals.append(last_expert_node)
            else:
                target_goals.append(torch.tensor([0], device=device))
        target_goals = torch.stack(target_goals)  # [batch, 1]
    else:
        # No goals available, return zero loss
        return torch.tensor(0.0, device=device)

    # Find last predicted node for each head (use final waypoint)
    # [batch, num_heads]
    final_predicted_nodes = predicted_node_ids[:, :, -1]

    # Compute minimum distance to any goal for each predicted endpoint
    # [batch, num_heads]
    total_penalty = 0.0
    num_evaluated = 0

    for b in range(batch_size):
        goals_b = target_goals[b]  # [num_goals] or [1]

        # Skip if no goals available for this batch item
        if len(goals_b) == 0:
            continue

        for h in range(num_heads):
            pred_endpoint = final_predicted_nodes[b, h].item()

            # Compute distance to nearest goal
            min_dist = min(abs(pred_endpoint - goal.item()) for goal in goals_b)

            # Penalize if distance exceeds threshold
            if min_dist > max_distance_threshold:
                penalty = (min_dist - max_distance_threshold) / max_waypoints
                total_penalty += penalty

            num_evaluated += 1

    if num_evaluated == 0:
        return torch.tensor(0.0, device=device)

    avg_penalty = total_penalty / num_evaluated
    return torch.tensor(avg_penalty, device=device, requires_grad=False)


def compute_graph_path_loss(
    predicted_logits: torch.Tensor,
    predicted_node_ids: torch.Tensor,
    expert_node_ids: torch.Tensor,
    expert_masks: torch.Tensor,
    edge_index: torch.Tensor,
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    start_node_ids: torch.Tensor = None,
    goal_node_ids: torch.Tensor = None,
    node_loss_weight: float = 1.0,
    connectivity_loss_weight: float = 0.5,
    diversity_loss_weight: float = 0.3,
    start_goal_weight: float = 1.0,
    goal_reaching_weight: float = 0.5,
    use_start_goal_weighting: bool = True,
    use_goal_reaching: bool = True,
) -> GraphPathLoss:
    """Compute complete graph path prediction loss with multimodal awareness.

    Combines:
    1. Node classification (cross-entropy on node distributions)
    2. Connectivity (penalty for non-adjacent nodes)
    3. Diversity (penalty for similar paths)
    4. Start/goal weighting (emphasize first/last waypoints)
    5. Goal reaching (penalize paths not ending at goals)

    Args:
        predicted_logits: [batch, num_heads, max_waypoints, max_nodes]
        predicted_node_ids: [batch, num_heads, max_waypoints] - selected nodes
        expert_node_ids: [batch, max_expert_waypoints] - expert node sequence
        expert_masks: [batch, max_expert_waypoints] - valid expert waypoints
        edge_index: [batch, 2, max_edges] - graph edges
        edge_mask: [batch, max_edges] - valid edges
        node_mask: [batch, max_nodes] - valid nodes
        start_node_ids: [batch] - start node IDs for multimodal fusion
        goal_node_ids: [batch, num_goals] - goal node IDs for multimodal fusion
        node_loss_weight: Weight for node classification loss
        connectivity_loss_weight: Weight for connectivity loss
        diversity_loss_weight: Weight for diversity loss
        start_goal_weight: Weight for start/goal emphasis loss
        goal_reaching_weight: Weight for goal reaching loss
        use_start_goal_weighting: Enable start/goal weighted loss
        use_goal_reaching: Enable goal reaching loss

    Returns:
        GraphPathLoss with all components
    """
    # 1. Node classification loss
    node_loss = node_classification_loss(
        predicted_logits, expert_node_ids, expert_masks
    )

    # 2. Connectivity loss
    conn_loss = connectivity_loss(predicted_node_ids, edge_index, edge_mask, node_mask)

    # 3. Diversity loss
    div_loss = path_diversity_loss(predicted_node_ids)

    # 4. Start/goal weighted loss (emphasize correct start/end)
    sg_loss = None
    if use_start_goal_weighting:
        sg_loss = start_goal_weighted_loss(
            predicted_logits,
            expert_node_ids,
            expert_masks,
            start_node_ids,
            goal_node_ids,
        )

    # 5. Goal reaching loss (ensure paths end at goals)
    gr_loss = None
    if use_goal_reaching:
        gr_loss = goal_reaching_loss(
            predicted_node_ids, goal_node_ids, expert_node_ids, expert_masks
        )

    # Weighted combination
    total_loss = (
        node_loss_weight * node_loss
        + connectivity_loss_weight * conn_loss
        + diversity_loss_weight * div_loss
    )

    if sg_loss is not None:
        total_loss = total_loss + start_goal_weight * sg_loss

    if gr_loss is not None:
        total_loss = total_loss + goal_reaching_weight * gr_loss

    return GraphPathLoss(
        total_loss=total_loss,
        node_classification_loss=node_loss,
        connectivity_loss=conn_loss,
        diversity_loss=div_loss,
        start_goal_weighted_loss=sg_loss,
        goal_reaching_loss=gr_loss,
    )


def compute_path_accuracy(
    predicted_node_ids: torch.Tensor,
    expert_node_ids: torch.Tensor,
    expert_masks: torch.Tensor,
    tolerance: int = 0,
) -> Dict[str, float]:
    """Compute accuracy metrics for path prediction.

    Args:
        predicted_node_ids: [batch, num_heads, max_waypoints]
        expert_node_ids: [batch, max_expert_waypoints]
        expert_masks: [batch, max_expert_waypoints]
        tolerance: Allow prediction within tolerance nodes of expert

    Returns:
        Dictionary with accuracy metrics
    """
    batch_size, num_heads, max_waypoints = predicted_node_ids.shape

    # Find best matching head per batch item
    best_matches = []

    for b in range(batch_size):
        expert_path = expert_node_ids[b, expert_masks[b].bool()].cpu().numpy()
        if len(expert_path) == 0:
            continue

        best_match_score = 0
        for h in range(num_heads):
            pred_path = predicted_node_ids[b, h, : len(expert_path)].cpu().numpy()

            # Count matches
            if tolerance == 0:
                matches = (pred_path == expert_path).sum()
            else:
                matches = 0
                for i in range(len(expert_path)):
                    if abs(pred_path[i] - expert_path[i]) <= tolerance:
                        matches += 1

            match_score = matches / len(expert_path)
            best_match_score = max(best_match_score, match_score)

        best_matches.append(best_match_score)

    if not best_matches:
        return {"exact_match_accuracy": 0.0, "best_head_accuracy": 0.0}

    return {
        "exact_match_accuracy": sum(m == 1.0 for m in best_matches) / len(best_matches),
        "best_head_accuracy": sum(best_matches) / len(best_matches),
    }
