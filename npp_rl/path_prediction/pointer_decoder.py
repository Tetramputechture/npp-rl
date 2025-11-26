"""Pointer Network decoder for selecting graph nodes as path waypoints.

Implements attention-based pointer mechanism to select sequences of graph nodes,
ensuring predictions are on valid discrete positions rather than continuous space.

Based on Vinyals et al. (2015) "Pointer Networks" but adapted for graph node selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PointerDecoder(nn.Module):
    """Pointer Network decoder for selecting graph nodes as waypoints.

    Uses attention mechanism to "point" to graph nodes, outputting a distribution
    over nodes for each waypoint position in the path.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: Optional[int] = None,
        max_waypoints: int = 20,
        dropout: float = 0.1,
    ):
        """Initialize pointer decoder.

        Args:
            hidden_dim: Dimension of node embeddings from GNN
            context_dim: Dimension of fused context (if None, defaults to hidden_dim)
            max_waypoints: Maximum number of waypoints to predict
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.max_waypoints = max_waypoints

        # Query generator: converts fused context to K query vectors
        self.query_generator = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim * max_waypoints),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism for pointing to nodes
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_scale = hidden_dim**-0.5

        # Optional: Learnable positional embeddings for waypoint positions
        self.position_embeddings = nn.Parameter(
            torch.randn(max_waypoints, hidden_dim) * 0.02
        )

        # Confidence prediction (optional, for ranking paths)
        # Uses context_dim as input since it receives fused_context
        self.confidence_head = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim // 2),
            nn.ReLU(),
            nn.Linear(self.context_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize weights for better training stability
        self._init_weights()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        fused_context: torch.Tensor,
        node_mask: torch.Tensor,
        temperature: float = 1.0,
        return_probs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass to select sequence of graph nodes.

        Args:
            node_embeddings: Node features from GNN [batch, max_nodes, hidden_dim]
            fused_context: Fused multimodal context [batch, context_dim]
            node_mask: Valid node mask [batch, max_nodes]
            temperature: Softmax temperature for sampling (default: 1.0)
            return_probs: If True, return probability distributions

        Returns:
            Tuple of:
            - node_indices: Selected node IDs [batch, max_waypoints]
            - logits: Attention logits [batch, max_waypoints, max_nodes]
            - confidence: Path confidence score [batch] (optional)
        """
        batch_size = fused_context.size(0)
        max_nodes = node_embeddings.size(1)

        # Generate query vectors from fused context
        # [batch, context_dim] -> [batch, max_waypoints * hidden_dim]
        queries_flat = self.query_generator(fused_context)

        # Reshape to [batch, max_waypoints, hidden_dim]
        queries = queries_flat.view(batch_size, self.max_waypoints, self.hidden_dim)

        # Add positional embeddings
        queries = queries + self.position_embeddings.unsqueeze(0)  # [batch, K, hidden]

        # Project queries and keys
        Q = self.query_proj(queries)  # [batch, max_waypoints, hidden_dim]
        K = self.key_proj(node_embeddings)  # [batch, max_nodes, hidden_dim]

        # Compute attention scores: Q @ K^T
        # [batch, max_waypoints, hidden_dim] @ [batch, hidden_dim, max_nodes]
        # -> [batch, max_waypoints, max_nodes]
        logits = torch.bmm(Q, K.transpose(1, 2)) * self.attention_scale

        # Apply temperature
        logits = logits / temperature

        # Mask invalid nodes (set to -inf so softmax gives 0 probability)
        # Expand node_mask: [batch, max_nodes] -> [batch, 1, max_nodes]
        node_mask_expanded = node_mask.unsqueeze(1).bool()
        logits = logits.masked_fill(~node_mask_expanded, float("-inf"))

        # Select nodes (argmax or sample)
        # For now, use argmax for deterministic selection
        node_indices = torch.argmax(logits, dim=-1)  # [batch, max_waypoints]

        # Compute confidence score (based on fused context)
        confidence = self.confidence_head(fused_context).squeeze(-1)  # [batch]

        if return_probs:
            return node_indices, logits, confidence
        else:
            return node_indices, logits, confidence

    def _init_weights(self):
        """Initialize weights for better training stability.

        Critical improvements:
        1. Xavier/Glorot initialization for query/key projections
        2. Scaled initialization for positional embeddings
        3. Smaller initial weights to prevent attention collapse
        """
        # Initialize query and key projections with Xavier uniform
        # This helps with gradient flow and prevents attention collapse
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
        nn.init.zeros_(self.query_proj.bias)

        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
        nn.init.zeros_(self.key_proj.bias)

        # Initialize positional embeddings with smaller variance
        # This prevents them from dominating early training
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.01)

        # Initialize query generator with careful scaling
        for module in self.query_generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize confidence head
        for module in self.confidence_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        logger.debug("Pointer decoder weights initialized with improved scheme")

    def forward_with_sampling(
        self,
        node_embeddings: torch.Tensor,
        fused_context: torch.Tensor,
        node_mask: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with sampling for exploration during training.

        Args:
            node_embeddings: Node features [batch, max_nodes, hidden_dim]
            fused_context: Fused multimodal context [batch, context_dim]
            node_mask: Valid nodes [batch, max_nodes]
            temperature: Sampling temperature
            top_k: If set, sample from top-k nodes only

        Returns:
            Tuple of (node_indices, logits, confidence)
        """
        batch_size = fused_context.size(0)

        # Get logits
        _, logits, confidence = self.forward(
            node_embeddings,
            fused_context,
            node_mask,
            temperature=temperature,
            return_probs=True,
        )

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)  # [batch, max_waypoints, max_nodes]

        if top_k is not None:
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # Sample from top-k
            sampled_k_indices = torch.multinomial(
                top_k_probs.view(-1, top_k), num_samples=1
            ).view(batch_size, self.max_waypoints)

            # Map back to original node indices
            node_indices = torch.gather(
                top_k_indices, -1, sampled_k_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            # Sample from full distribution
            node_indices = torch.multinomial(
                probs.view(-1, probs.size(-1)), num_samples=1
            ).view(batch_size, self.max_waypoints)

        return node_indices, logits, confidence


class MultiHeadPointerDecoder(nn.Module):
    """Multiple pointer decoders for diverse path generation.

    Each head independently selects a path, with diversity encouraged
    through different initializations and optional diversity loss.
    """

    def __init__(
        self,
        num_heads: int = 4,
        hidden_dim: int = 256,
        max_waypoints: int = 20,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        """Initialize multi-head pointer decoder.

        Args:
            num_heads: Number of path candidates to generate
            hidden_dim: Dimension of node embeddings
            max_waypoints: Maximum waypoints per path
            dropout: Dropout rate
            context_dim: Dimension of fused context (if None, defaults to hidden_dim)
        """
        super().__init__()

        self.num_heads = num_heads
        self.max_waypoints = max_waypoints

        # Create separate decoder for each head
        self.decoders = nn.ModuleList(
            [
                PointerDecoder(
                    hidden_dim=hidden_dim,
                    context_dim=context_dim,
                    max_waypoints=max_waypoints,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

        # Head-specific context projections (for diversity)
        # Note: These should operate on context_dim, not hidden_dim
        effective_context_dim = context_dim if context_dim is not None else hidden_dim
        self.head_projections = nn.ModuleList(
            [
                nn.Linear(effective_context_dim, effective_context_dim)
                for _ in range(num_heads)
            ]
        )

        # Initialize head projections for diversity
        self._init_head_projections()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        fused_context: torch.Tensor,
        node_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate multiple diverse path candidates.

        Args:
            node_embeddings: Node features [batch, max_nodes, hidden_dim]
            fused_context: Fused multimodal context [batch, context_dim]
            node_mask: Valid nodes [batch, max_nodes]
            temperature: Sampling temperature

        Returns:
            Tuple of:
            - all_node_indices: [batch, num_heads, max_waypoints]
            - all_logits: [batch, num_heads, max_waypoints, max_nodes]
            - all_confidences: [batch, num_heads]
        """
        batch_size = fused_context.size(0)
        max_nodes = node_embeddings.size(1)

        all_node_indices = []
        all_logits = []
        all_confidences = []

        for head_idx in range(self.num_heads):
            # Project fused context (for diversity)
            context_head = self.head_projections[head_idx](fused_context)

            # Generate path with this head
            node_indices, logits, confidence = self.decoders[head_idx](
                node_embeddings, context_head, node_mask, temperature
            )

            all_node_indices.append(node_indices)
            all_logits.append(logits)
            all_confidences.append(confidence)

        # Stack outputs
        all_node_indices = torch.stack(all_node_indices, dim=1)  # [batch, heads, K]
        all_logits = torch.stack(all_logits, dim=1)  # [batch, heads, K, nodes]
        all_confidences = torch.stack(all_confidences, dim=1)  # [batch, heads]

        return all_node_indices, all_logits, all_confidences

    def forward_with_sampling(
        self,
        node_embeddings: torch.Tensor,
        fused_context: torch.Tensor,
        node_mask: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate paths with sampling for training exploration.

        Args:
            node_embeddings: Node features [batch, max_nodes, hidden_dim]
            fused_context: Fused multimodal context [batch, context_dim]
            node_mask: Valid nodes [batch, max_nodes]
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Tuple of (all_node_indices, all_logits, all_confidences)
        """
        batch_size = fused_context.size(0)

        all_node_indices = []
        all_logits = []
        all_confidences = []

        for head_idx in range(self.num_heads):
            context_head = self.head_projections[head_idx](fused_context)

            node_indices, logits, confidence = self.decoders[
                head_idx
            ].forward_with_sampling(
                node_embeddings, context_head, node_mask, temperature, top_k
            )

            all_node_indices.append(node_indices)
            all_logits.append(logits)
            all_confidences.append(confidence)

        all_node_indices = torch.stack(all_node_indices, dim=1)
        all_logits = torch.stack(all_logits, dim=1)
        all_confidences = torch.stack(all_confidences, dim=1)

        return all_node_indices, all_logits, all_confidences

    def _init_head_projections(self):
        """Initialize head projections with diversity-inducing noise.

        Each head gets slightly different initialization to encourage
        diverse path predictions even with the same input.
        """
        for head_idx, projection in enumerate(self.head_projections):
            # Initialize with Xavier but add small head-specific bias
            nn.init.xavier_uniform_(projection.weight, gain=1.0)

            # Add small head-specific offset to bias
            # This creates initial diversity between heads
            bias_offset = (head_idx - self.num_heads / 2) * 0.01
            nn.init.constant_(projection.bias, bias_offset)

        logger.debug(
            f"Initialized {self.num_heads} head projections with diversity-inducing offsets"
        )
