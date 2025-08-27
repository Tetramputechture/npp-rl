"""
Multi-scale feature fusion for hierarchical graph neural networks.

This module implements advanced fusion mechanisms that combine features
from multiple resolution levels with attention-based weighting and
context-aware processing for N++ level understanding.

Key Components:
- AdaptiveScaleFusion: Context-aware weighting based on ninja physics state
- HierarchicalFeatureAggregator: Learned routing between resolution levels  
- ContextAwareScaleSelector: Dynamic attention to appropriate scales
- UnifiedMultiScaleFusion: Integrated fusion combining all mechanisms

The fusion mechanisms enable the model to dynamically focus on the most
relevant resolution level based on the current task requirements and
ninja state, improving both local precision and strategic planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any

from .diffpool_gnn import MultiScaleGraphAttention


class AdaptiveScaleFusion(nn.Module):
    """
    Adaptive fusion of multi-scale graph features with context-aware weighting.
    
    Dynamically weights different resolution levels based on ninja state,
    level complexity, and current task requirements.
    """
    
    def __init__(
        self,
        scale_dims: Dict[str, int],
        fusion_dim: int = 256,
        context_dim: int = 18,  # Ninja physics state dimension
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize adaptive scale fusion.
        
        Args:
            scale_dims: Feature dimensions for each scale {'sub_cell': dim, 'tile': dim, 'region': dim}
            fusion_dim: Output fusion dimension
            context_dim: Context dimension (ninja physics state)
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.scale_dims = scale_dims
        self.fusion_dim = fusion_dim
        self.context_dim = context_dim
        self.num_scales = len(scale_dims)
        
        # Project each scale to common dimension
        self.scale_projections = nn.ModuleDict()
        for scale_name, dim in scale_dims.items():
            self.scale_projections[scale_name] = nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(fusion_dim)
            )
        
        # Context encoder for ninja physics state
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim // 2)
        )
        
        # Scale importance predictor
        self.scale_importance = nn.Sequential(
            nn.Linear(fusion_dim + fusion_dim // 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.num_scales),
            nn.Softmax(dim=-1)
        )
        
        # Cross-scale attention mechanism
        self.cross_scale_attention = MultiScaleGraphAttention(
            feature_dims={name: fusion_dim for name in scale_dims.keys()},
            hidden_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Hierarchical feature integration
        self.hierarchical_integration = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_attention_heads,
                dim_feedforward=fusion_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        scale_features: Dict[str, torch.Tensor],
        ninja_physics_state: Optional[torch.Tensor] = None,
        level_complexity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through adaptive scale fusion.
        
        Args:
            scale_features: Features from different scales
            ninja_physics_state: Current ninja physics state [batch_size, context_dim]
            level_complexity: Optional level complexity metrics
            
        Returns:
            Tuple of (fused_features, attention_weights)
            
        Raises:
            ValueError: If scale_features is empty or contains invalid tensors
        """
        # Input validation
        if not scale_features:
            raise ValueError("scale_features cannot be empty")
        
        # Check that all scale features are valid tensors
        for scale_name, features in scale_features.items():
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Features for scale '{scale_name}' must be a torch.Tensor")
            if features.numel() == 0:
                raise ValueError(f"Features for scale '{scale_name}' cannot be empty")
        
        batch_size = next(iter(scale_features.values())).shape[0]
        device = next(iter(scale_features.values())).device
        
        # Project all scales to common dimension
        projected_features = {}
        for scale_name, features in scale_features.items():
            projected_features[scale_name] = self.scale_projections[scale_name](features)
        
        # Encode context (ninja physics state)
        if ninja_physics_state is not None:
            if ninja_physics_state.dim() == 1:
                ninja_physics_state = ninja_physics_state.unsqueeze(0).expand(batch_size, -1)
            context_features = self.context_encoder(ninja_physics_state)
        else:
            context_features = torch.zeros(batch_size, self.fusion_dim // 2, device=device)
        
        # Compute scale importance weights
        scale_names = list(projected_features.keys())
        stacked_features = torch.stack([projected_features[name] for name in scale_names], dim=1)
        
        # Use mean of scale features for importance computation
        mean_scale_features = torch.mean(stacked_features, dim=1)
        importance_input = torch.cat([mean_scale_features, context_features], dim=-1)
        scale_weights = self.scale_importance(importance_input)
        
        # Apply cross-scale attention
        attended_features = self.cross_scale_attention(
            projected_features, ninja_physics_state
        )
        
        # Hierarchical integration using transformer layers
        # Prepare features for transformer (add positional encoding for scales)
        transformer_input = stacked_features  # [batch_size, num_scales, fusion_dim]
        
        # Add scale positional encoding
        scale_positions = torch.arange(self.num_scales, device=device).float()
        scale_positions = scale_positions.unsqueeze(0).expand(batch_size, -1)
        
        # Simple sinusoidal positional encoding for scales
        pos_encoding = self._get_positional_encoding(scale_positions, self.fusion_dim)
        transformer_input = transformer_input + pos_encoding
        
        # Apply transformer layers
        for transformer_layer in self.hierarchical_integration:
            transformer_input = transformer_layer(transformer_input)
        
        # Weighted combination of transformer output
        weighted_transformer = transformer_input * scale_weights.unsqueeze(-1)
        transformer_fused = torch.sum(weighted_transformer, dim=1)
        
        # Combine attended features and transformer output
        combined_features = self.residual_weight * attended_features + (1 - self.residual_weight) * transformer_fused
        
        # Final fusion
        fused_features = self.fusion_layers(combined_features)
        
        # Prepare attention weights for output
        attention_weights = {
            'scale_importance': {scale_names[i]: scale_weights[:, i] for i in range(len(scale_names))},
            'residual_weight': self.residual_weight.item()
        }
        
        return fused_features, attention_weights
    
    def _get_positional_encoding(self, positions: torch.Tensor, d_model: int) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding for scale positions.
        
        Args:
            positions: Scale positions [batch_size, num_scales]
            d_model: Model dimension
            
        Returns:
            Positional encoding [batch_size, num_scales, d_model]
        """
        batch_size, seq_len = positions.shape
        device = positions.device
        
        # Create position encoding matrix
        pe = torch.zeros(batch_size, seq_len, d_model, device=device)
        
        # Compute div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sinusoidal encoding
        pe[:, :, 0::2] = torch.sin(positions.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(positions.unsqueeze(-1) * div_term)
        
        return pe


class HierarchicalFeatureAggregator(nn.Module):
    """
    Aggregates features across hierarchical levels with learned routing.
    
    Implements sophisticated feature routing that learns to send information
    between different resolution levels based on content and context.
    """
    
    def __init__(
        self,
        level_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_routing_iterations: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical feature aggregator.
        
        Args:
            level_dims: Feature dimensions for each level
            hidden_dim: Hidden dimension for processing
            num_routing_iterations: Number of routing iterations
            dropout: Dropout probability
        """
        super().__init__()
        
        self.level_dims = level_dims
        self.hidden_dim = hidden_dim
        self.num_routing_iterations = num_routing_iterations
        self.level_names = list(level_dims.keys())
        self.num_levels = len(self.level_names)
        
        # Feature projections for each level
        self.level_projections = nn.ModuleDict()
        for level_name, dim in level_dims.items():
            self.level_projections[level_name] = nn.Linear(dim, hidden_dim)
        
        # Routing networks for each level pair
        self.routing_networks = nn.ModuleDict()
        for i, src_level in enumerate(self.level_names):
            for j, tgt_level in enumerate(self.level_names):
                if i != j:  # No self-routing
                    key = f"{src_level}_to_{tgt_level}"
                    self.routing_networks[key] = nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                        nn.Sigmoid()
                    )
        
        # Level-specific processing networks
        self.level_processors = nn.ModuleDict()
        for level_name in self.level_names:
            self.level_processors[level_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # Final aggregation network
        self.final_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * self.num_levels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        level_features: Dict[str, torch.Tensor],
        cross_level_edges: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical feature aggregator.
        
        Args:
            level_features: Features for each level
            cross_level_edges: Optional cross-level connectivity information
            
        Returns:
            Tuple of (aggregated_features, routing_weights)
        """
        batch_size = next(iter(level_features.values())).shape[0]
        device = next(iter(level_features.values())).device
        
        # Project all levels to common dimension
        projected_features = {}
        for level_name, features in level_features.items():
            projected_features[level_name] = self.level_projections[level_name](features)
        
        # Initialize routing weights
        routing_weights = {}
        
        # Iterative routing process
        current_features = projected_features.copy()
        
        for iteration in range(self.num_routing_iterations):
            # Compute routing weights for this iteration
            iteration_routing = {}
            
            for src_level in self.level_names:
                for tgt_level in self.level_names:
                    if src_level != tgt_level:
                        key = f"{src_level}_to_{tgt_level}"
                        
                        # Concatenate source and target features
                        src_feat = current_features[src_level]
                        tgt_feat = current_features[tgt_level]
                        combined_feat = torch.cat([src_feat, tgt_feat], dim=-1)
                        
                        # Compute routing weight
                        routing_weight = self.routing_networks[key](combined_feat)
                        iteration_routing[key] = routing_weight
            
            # Update features based on routing
            updated_features = {}
            for level_name in self.level_names:
                # Start with current features
                updated_feat = current_features[level_name].clone()
                
                # Add routed features from other levels
                for src_level in self.level_names:
                    if src_level != level_name:
                        key = f"{src_level}_to_{level_name}"
                        if key in iteration_routing:
                            routing_weight = iteration_routing[key]
                            routed_feat = routing_weight * current_features[src_level]
                            updated_feat = updated_feat + routed_feat
                
                # Process updated features
                updated_feat = self.level_processors[level_name](updated_feat)
                updated_features[level_name] = updated_feat
            
            current_features = updated_features
            
            # Store routing weights from final iteration
            if iteration == self.num_routing_iterations - 1:
                routing_weights = iteration_routing
        
        # Final aggregation
        all_features = [current_features[level_name] for level_name in self.level_names]
        concatenated_features = torch.cat(all_features, dim=-1)
        aggregated_features = self.final_aggregator(concatenated_features)
        
        return aggregated_features, routing_weights


class ContextAwareScaleSelector(nn.Module):
    """
    Context-aware scale selection for dynamic attention to resolution levels.
    
    Learns to focus on appropriate resolution levels based on ninja state,
    level characteristics, and current task phase.
    """
    
    def __init__(
        self,
        scale_dims: Dict[str, int],
        context_dim: int = 18,
        hidden_dim: int = 128,
        num_context_types: int = 4,  # local_movement, global_planning, hazard_avoidance, exploration
        dropout: float = 0.1
    ):
        """
        Initialize context-aware scale selector.
        
        Args:
            scale_dims: Feature dimensions for each scale
            context_dim: Context dimension (ninja physics state)
            hidden_dim: Hidden dimension
            num_context_types: Number of context types to consider
            dropout: Dropout probability
        """
        super().__init__()
        
        self.scale_dims = scale_dims
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_context_types = num_context_types
        self.scale_names = list(scale_dims.keys())
        self.num_scales = len(self.scale_names)
        
        # Context type classifier
        self.context_classifier = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_context_types),
            nn.Softmax(dim=-1)
        )
        
        # Scale preference for each context type
        self.context_scale_preferences = nn.Parameter(
            torch.randn(num_context_types, self.num_scales)
        )
        
        # Dynamic scale attention based on features
        self.feature_scale_attention = nn.ModuleDict()
        for scale_name, dim in scale_dims.items():
            self.feature_scale_attention[scale_name] = nn.Sequential(
                nn.Linear(dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Temporal consistency regularization
        self.temporal_smoother = nn.Parameter(torch.tensor(0.8))  # Momentum for temporal smoothing
        self.register_buffer('prev_scale_weights', torch.ones(1, self.num_scales) / self.num_scales)
        
    def forward(
        self,
        scale_features: Dict[str, torch.Tensor],
        ninja_physics_state: Optional[torch.Tensor] = None,
        apply_temporal_smoothing: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through context-aware scale selector.
        
        Args:
            scale_features: Features from different scales
            ninja_physics_state: Ninja physics state for context
            apply_temporal_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Tuple of (scale_weights, context_info)
        """
        batch_size = next(iter(scale_features.values())).shape[0]
        device = next(iter(scale_features.values())).device
        
        # Initialize scale weights
        scale_weights = torch.ones(batch_size, self.num_scales, device=device) / self.num_scales
        context_info = {}
        
        if ninja_physics_state is not None:
            # Expand ninja physics state if needed
            if ninja_physics_state.dim() == 1:
                ninja_physics_state = ninja_physics_state.unsqueeze(0).expand(batch_size, -1)
            
            # Classify context type
            context_probs = self.context_classifier(ninja_physics_state)
            context_info['context_probabilities'] = context_probs
            
            # Get context-based scale preferences
            context_scale_weights = torch.mm(context_probs, self.context_scale_preferences)
            context_scale_weights = F.softmax(context_scale_weights, dim=-1)
            
            # Compute feature-based attention for each scale
            feature_attentions = []
            for i, scale_name in enumerate(self.scale_names):
                attention = self.feature_scale_attention[scale_name](scale_features[scale_name])
                feature_attentions.append(attention.squeeze(-1))
            
            feature_attentions = torch.stack(feature_attentions, dim=-1)
            feature_attentions = F.softmax(feature_attentions, dim=-1)
            
            # Combine context and feature-based weights
            combined_weights = 0.6 * context_scale_weights + 0.4 * feature_attentions
            scale_weights = F.softmax(combined_weights, dim=-1)
            
            context_info['context_scale_weights'] = context_scale_weights
            context_info['feature_attentions'] = feature_attentions
        
        # Apply temporal smoothing if requested
        if apply_temporal_smoothing and hasattr(self, 'prev_scale_weights'):
            # Expand previous weights to match batch size
            prev_weights = self.prev_scale_weights.expand(batch_size, -1)
            
            # Smooth with previous weights
            scale_weights = (self.temporal_smoother * prev_weights + 
                           (1 - self.temporal_smoother) * scale_weights)
            
            # Update previous weights (use mean across batch for next iteration)
            self.prev_scale_weights = torch.mean(scale_weights, dim=0, keepdim=True).detach()
        
        context_info['final_scale_weights'] = scale_weights
        
        return scale_weights, context_info
    
    def reset_temporal_state(self):
        """Reset temporal smoothing state."""
        self.prev_scale_weights.fill_(1.0 / self.num_scales)


class UnifiedMultiScaleFusion(nn.Module):
    """
    Unified multi-scale fusion combining all fusion mechanisms.
    
    Integrates adaptive scale fusion, hierarchical aggregation, and
    context-aware selection for comprehensive multi-resolution processing.
    """
    
    def __init__(
        self,
        scale_dims: Dict[str, int],
        fusion_dim: int = 256,
        context_dim: int = 18,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize unified multi-scale fusion.
        
        Args:
            scale_dims: Feature dimensions for each scale
            fusion_dim: Final fusion dimension
            context_dim: Context dimension
            hidden_dim: Hidden dimension for processing
            dropout: Dropout probability
        """
        super().__init__()
        
        self.scale_dims = scale_dims
        self.fusion_dim = fusion_dim
        
        # Component fusion modules
        self.adaptive_fusion = AdaptiveScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=fusion_dim,
            context_dim=context_dim,
            dropout=dropout
        )
        
        self.hierarchical_aggregator = HierarchicalFeatureAggregator(
            level_dims=scale_dims,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.scale_selector = ContextAwareScaleSelector(
            scale_dims=scale_dims,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Final fusion of different fusion approaches
        self.meta_fusion = nn.Sequential(
            nn.Linear(fusion_dim + hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Learnable combination weights
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(
        self,
        scale_features: Dict[str, torch.Tensor],
        ninja_physics_state: Optional[torch.Tensor] = None,
        cross_level_edges: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through unified multi-scale fusion.
        
        Args:
            scale_features: Features from different scales
            ninja_physics_state: Ninja physics state
            cross_level_edges: Cross-level connectivity
            
        Returns:
            Tuple of (unified_features, fusion_info)
        """
        # Apply adaptive fusion
        adaptive_features, adaptive_weights = self.adaptive_fusion(
            scale_features, ninja_physics_state
        )
        
        # Apply hierarchical aggregation
        hierarchical_features, routing_weights = self.hierarchical_aggregator(
            scale_features, cross_level_edges
        )
        
        # Get scale selection weights
        scale_weights, context_info = self.scale_selector(
            scale_features, ninja_physics_state
        )
        
        # Normalize fusion weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Combine different fusion approaches
        combined_features = torch.cat([
            adaptive_features * fusion_weights[0],
            hierarchical_features * fusion_weights[1]
        ], dim=-1)
        
        # Final meta-fusion
        unified_features = self.meta_fusion(combined_features)
        
        # Collect fusion information
        fusion_info = {
            'adaptive_weights': adaptive_weights,
            'routing_weights': routing_weights,
            'scale_weights': scale_weights,
            'context_info': context_info,
            'fusion_weights': fusion_weights.detach().cpu().numpy()
        }
        
        return unified_features, fusion_info