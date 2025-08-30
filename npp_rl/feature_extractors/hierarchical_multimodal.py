"""
Hierarchical Multimodal Feature Extractor for N++ RL Agent

This module implements an advanced multimodal feature extractor that integrates
hierarchical graph neural networks with traditional visual and symbolic processing.

Key features:
- Multi-resolution graph processing with DiffPool
- Adaptive scale fusion based on ninja physics state
- Integration with existing CNN and MLP encoders
- End-to-end training with auxiliary losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

from ..models.diffpool_gnn import HierarchicalDiffPoolGNN
from ..models.multi_scale_fusion import UnifiedMultiScaleFusion
from .multimodal import MultimodalGraphExtractor


class HierarchicalMultimodalExtractor(BaseFeaturesExtractor):
    """
    Advanced multimodal feature extractor with hierarchical graph processing.
    
    Combines traditional visual/symbolic processing with multi-resolution
    graph neural networks for comprehensive N++ level understanding.
    
    Architecture:
    1. Visual processing: 3D CNNs for temporal frames, 2D CNNs for global view
    2. Symbolic processing: MLPs for game state
    3. Hierarchical graph processing: Multi-resolution GNNs with DiffPool
    4. Multi-scale fusion: Adaptive fusion based on ninja physics state
    5. Final integration: Combined multimodal representation
    """
    
    def __init__(
        self,
        observation_space: SpacesDict,
        features_dim: int = 512,
        use_hierarchical_graph: bool = True,
        hierarchical_hidden_dim: int = 128,
        fusion_dim: int = 256,
        enable_auxiliary_losses: bool = True,
        **kwargs
    ):
        """
        Initialize hierarchical multimodal feature extractor.
        
        Args:
            observation_space: Gym observation space dictionary
            features_dim: Final output feature dimension
            use_hierarchical_graph: Whether to use hierarchical graph processing
            hierarchical_hidden_dim: Hidden dimension for hierarchical GNN
            fusion_dim: Dimension for multi-scale fusion
            enable_auxiliary_losses: Whether to compute auxiliary losses
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, features_dim)
        
        self.use_hierarchical_graph = use_hierarchical_graph
        self.enable_auxiliary_losses = enable_auxiliary_losses
        self.fusion_dim = fusion_dim
        
        # Check for hierarchical graph observations
        self.has_hierarchical_graph = use_hierarchical_graph and self._check_hierarchical_graph_obs(observation_space)
        
        # Initialize base multimodal extractor for visual/symbolic processing
        self.base_extractor = MultimodalGraphExtractor(
            observation_space=observation_space,
            features_dim=256,  # Intermediate dimension
            use_graph_obs=False,  # We'll handle graph processing separately
            **kwargs
        )
        
        # Initialize hierarchical graph processing
        if self.has_hierarchical_graph:
            self._init_hierarchical_graph_processing(observation_space)
        
        # Initialize final fusion network
        self._init_final_fusion()
        
        # Storage for auxiliary losses
        self.auxiliary_losses = {}
    
    def _check_hierarchical_graph_obs(self, observation_space: SpacesDict) -> bool:
        """Check if hierarchical graph observations are available."""
        required_keys = [
            'sub_cell_node_features', 'sub_cell_edge_index', 'sub_cell_node_mask', 'sub_cell_edge_mask',
            'tile_node_features', 'tile_edge_index', 'tile_node_mask', 'tile_edge_mask',
            'region_node_features', 'region_edge_index', 'region_node_mask', 'region_edge_mask'
        ]
        
        return all(key in observation_space.spaces for key in required_keys)
    
    def _init_hierarchical_graph_processing(self, observation_space: SpacesDict):
        """Initialize hierarchical graph neural network components."""
        # Extract dimensions for each resolution level
        scale_dims = {
            'sub_cell': observation_space['sub_cell_node_features'].shape[1],
            'tile': observation_space['tile_node_features'].shape[1],
            'region': observation_space['region_node_features'].shape[1]
        }
        
        # Initialize hierarchical DiffPool GNN
        self.hierarchical_gnn = HierarchicalDiffPoolGNN(
            input_dims=scale_dims,
            hidden_dim=128,
            output_dim=self.fusion_dim,
            num_levels=3,
            pooling_ratios=[0.25, 0.25, 0.5],
            gnn_layers_per_level=2,
            dropout=0.1
        )
        
        # Initialize multi-scale fusion
        self.multi_scale_fusion = UnifiedMultiScaleFusion(
            scale_dims=scale_dims,
            fusion_dim=self.fusion_dim,
            context_dim=18,  # Ninja physics state dimension
            hidden_dim=128,
            dropout=0.1
        )
        
        self.hierarchical_feature_dim = self.fusion_dim
    
    def _init_final_fusion(self):
        """Initialize final fusion network combining all modalities."""
        # Calculate total input dimension
        base_dim = 256  # From base extractor
        hierarchical_dim = self.fusion_dim if self.has_hierarchical_graph else 0
        
        total_dim = base_dim + hierarchical_dim
        
        # Final fusion network
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            
            nn.Linear(256, self.features_dim),
            nn.ReLU()
        )
        
        # Adaptive weighting for different modalities
        self.modality_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # [base, hierarchical]
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hierarchical multimodal extractor.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Extracted features tensor
        """
        batch_size = next(iter(observations.values())).shape[0]
        device = next(iter(observations.values())).device
        
        # Reset auxiliary losses
        self.auxiliary_losses = {}
        
        # Process base modalities (visual + symbolic)
        base_features = self.base_extractor(observations)
        
        # Process hierarchical graph if available
        if self.has_hierarchical_graph:
            hierarchical_features, fusion_info = self._process_hierarchical_graph(observations)
            
            # Store fusion information as auxiliary data
            if self.enable_auxiliary_losses:
                self.auxiliary_losses.update(fusion_info.get('auxiliary_losses', {}))
        else:
            hierarchical_features = torch.zeros(batch_size, self.fusion_dim, device=device)
        
        # Combine modalities with learned weights
        modality_weights = F.softmax(self.modality_weights, dim=0)
        
        if self.has_hierarchical_graph:
            combined_features = torch.cat([
                base_features * modality_weights[0],
                hierarchical_features * modality_weights[1]
            ], dim=-1)
        else:
            combined_features = base_features
        
        # Final fusion
        final_features = self.final_fusion(combined_features)
        
        return final_features
    
    def _process_hierarchical_graph(
        self, 
        observations: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process hierarchical graph observations.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Tuple of (hierarchical_features, fusion_info)
            
        Raises:
            KeyError: If required hierarchical graph observations are missing
            RuntimeError: If hierarchical graph processing fails
        """
        # Validate required keys are present
        required_keys = [
            'sub_cell_node_features', 'sub_cell_edge_index', 'sub_cell_node_mask', 'sub_cell_edge_mask',
            'tile_node_features', 'tile_edge_index', 'tile_node_mask', 'tile_edge_mask',
            'region_node_features', 'region_edge_index', 'region_node_mask', 'region_edge_mask'
        ]
        
        missing_keys = [key for key in required_keys if key not in observations]
        if missing_keys:
            raise KeyError(f"Missing required hierarchical graph observations: {missing_keys}")
        
        # Extract ninja physics state if available
        ninja_physics_state = observations.get('ninja_physics_state', None)
        
        # Prepare hierarchical graph data
        hierarchical_graph_data = {}
        
        # Sub-cell level
        hierarchical_graph_data['sub_cell_node_features'] = observations['sub_cell_node_features']
        hierarchical_graph_data['sub_cell_edge_index'] = observations['sub_cell_edge_index']
        hierarchical_graph_data['sub_cell_node_mask'] = observations['sub_cell_node_mask']
        hierarchical_graph_data['sub_cell_edge_mask'] = observations['sub_cell_edge_mask']
        
        # Tile level
        hierarchical_graph_data['tile_node_features'] = observations['tile_node_features']
        hierarchical_graph_data['tile_edge_index'] = observations['tile_edge_index']
        hierarchical_graph_data['tile_node_mask'] = observations['tile_node_mask']
        hierarchical_graph_data['tile_edge_mask'] = observations['tile_edge_mask']
        
        # Region level
        hierarchical_graph_data['region_node_features'] = observations['region_node_features']
        hierarchical_graph_data['region_edge_index'] = observations['region_edge_index']
        hierarchical_graph_data['region_node_mask'] = observations['region_node_mask']
        hierarchical_graph_data['region_edge_mask'] = observations['region_edge_mask']
        
        # Process through hierarchical GNN
        gnn_features, gnn_auxiliary_losses = self.hierarchical_gnn(
            hierarchical_graph_data, ninja_physics_state
        )
        
        # Prepare scale features for multi-scale fusion
        scale_features = {
            'sub_cell': observations['sub_cell_node_features'].mean(dim=1),  # Global pool
            'tile': observations['tile_node_features'].mean(dim=1),
            'region': observations['region_node_features'].mean(dim=1)
        }
        
        # Apply multi-scale fusion
        fused_features, fusion_info = self.multi_scale_fusion(
            scale_features, ninja_physics_state
        )
        
        # Combine GNN and fusion features
        combined_hierarchical = 0.7 * gnn_features + 0.3 * fused_features
        
        # Prepare comprehensive fusion info
        comprehensive_fusion_info = {
            'gnn_features': gnn_features,
            'fused_features': fused_features,
            'fusion_info': fusion_info,
            'auxiliary_losses': gnn_auxiliary_losses
        }
        
        return combined_hierarchical, comprehensive_fusion_info
    
    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary losses for training.
        
        Returns:
            Dictionary of auxiliary losses
        """
        return self.auxiliary_losses
    
    def compute_total_loss(
        self,
        main_loss: torch.Tensor,
        aux_loss_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute total loss including auxiliary losses.
        
        Args:
            main_loss: Main task loss (e.g., RL policy loss)
            aux_loss_weights: Weights for auxiliary losses
            
        Returns:
            Total weighted loss
        """
        if not self.enable_auxiliary_losses or not self.auxiliary_losses:
            return main_loss
        
        if aux_loss_weights is None:
            aux_loss_weights = {
                'link_prediction_loss': 0.1,
                'entropy_loss': 0.01,
                'orthogonality_loss': 0.1
            }
        
        total_loss = main_loss
        
        for loss_name, loss_value in self.auxiliary_losses.items():
            weight = aux_loss_weights.get(loss_name, 0.0)
            total_loss += weight * loss_value
        
        return total_loss


class HierarchicalGraphObservationWrapper:
    """
    Wrapper to convert standard graph observations to hierarchical format.
    
    This wrapper takes standard graph observations and creates hierarchical
    representations by applying the hierarchical graph builder.
    """
    
    def __init__(
        self,
        enable_hierarchical: bool = True,
        cache_graphs: bool = True
    ):
        """
        Initialize hierarchical graph observation wrapper.
        
        Args:
            enable_hierarchical: Whether to enable hierarchical processing
            cache_graphs: Whether to cache graph computations
        """
        self.enable_hierarchical = enable_hierarchical
        self.cache_graphs = cache_graphs
        
        # Initialize hierarchical graph builder
        if enable_hierarchical:
            # Import here to avoid circular imports
            import sys
            import os
            nclone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'nclone')
            if os.path.exists(nclone_path) and nclone_path not in sys.path:
                sys.path.insert(0, nclone_path)
            
            try:
                from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder
                self.hierarchical_builder = HierarchicalGraphBuilder()
            except ImportError:
                print("Warning: Could not import HierarchicalGraphBuilder, disabling hierarchical processing")
                self.enable_hierarchical = False
                self.hierarchical_builder = None
        
        # Cache for graph computations
        self.graph_cache = {} if cache_graphs else None
    
    def process_observations(
        self,
        observations: Dict[str, Any],
        level_data: Optional[Dict[str, Any]] = None,
        ninja_position: Optional[Tuple[float, float]] = None,
        entities: Optional[list] = None,
        ninja_velocity: Optional[Tuple[float, float]] = None,
        ninja_state: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process observations to add hierarchical graph data.
        
        Args:
            observations: Original observations
            level_data: Level structure data
            ninja_position: Ninja position
            entities: Entity list
            ninja_velocity: Ninja velocity
            ninja_state: Ninja movement state
            
        Returns:
            Observations with hierarchical graph data
        """
        if not self.enable_hierarchical or self.hierarchical_builder is None:
            return observations
        
        # Create cache key if caching is enabled
        cache_key = None
        if self.cache_graphs and level_data is not None:
            cache_key = (
                id(level_data),
                ninja_position,
                ninja_velocity,
                ninja_state
            )
            
            if cache_key in self.graph_cache:
                hierarchical_data = self.graph_cache[cache_key]
                observations.update(hierarchical_data)
                return observations
        
        # Build hierarchical graph
        if level_data is not None and ninja_position is not None:
            hierarchical_graph = self.hierarchical_builder.build_hierarchical_graph(
                level_data, ninja_position, entities or [], ninja_velocity, ninja_state
            )
            
            # Convert to tensor format
            hierarchical_data = self._convert_to_tensors(hierarchical_graph)
            
            # Cache if enabled
            if self.cache_graphs and cache_key is not None:
                self.graph_cache[cache_key] = hierarchical_data
            
            # Add to observations
            observations.update(hierarchical_data)
        
        return observations
    
    def _convert_to_tensors(self, hierarchical_graph) -> Dict[str, torch.Tensor]:
        """Convert hierarchical graph data to PyTorch tensors."""
        tensor_data = {}
        
        # Sub-cell level
        tensor_data['sub_cell_node_features'] = torch.from_numpy(
            hierarchical_graph.sub_cell_graph.node_features
        ).float()
        tensor_data['sub_cell_edge_index'] = torch.from_numpy(
            hierarchical_graph.sub_cell_graph.edge_index
        ).long()
        tensor_data['sub_cell_node_mask'] = torch.from_numpy(
            hierarchical_graph.sub_cell_graph.node_mask
        ).float()
        tensor_data['sub_cell_edge_mask'] = torch.from_numpy(
            hierarchical_graph.sub_cell_graph.edge_mask
        ).float()
        
        # Tile level
        tensor_data['tile_node_features'] = torch.from_numpy(
            hierarchical_graph.tile_graph.node_features
        ).float()
        tensor_data['tile_edge_index'] = torch.from_numpy(
            hierarchical_graph.tile_graph.edge_index
        ).long()
        tensor_data['tile_node_mask'] = torch.from_numpy(
            hierarchical_graph.tile_graph.node_mask
        ).float()
        tensor_data['tile_edge_mask'] = torch.from_numpy(
            hierarchical_graph.tile_graph.edge_mask
        ).float()
        
        # Region level
        tensor_data['region_node_features'] = torch.from_numpy(
            hierarchical_graph.region_graph.node_features
        ).float()
        tensor_data['region_edge_index'] = torch.from_numpy(
            hierarchical_graph.region_graph.edge_index
        ).long()
        tensor_data['region_node_mask'] = torch.from_numpy(
            hierarchical_graph.region_graph.node_mask
        ).float()
        tensor_data['region_edge_mask'] = torch.from_numpy(
            hierarchical_graph.region_graph.edge_mask
        ).float()
        
        return tensor_data


def create_hierarchical_multimodal_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    use_hierarchical_graph: bool = True,
    **kwargs
) -> HierarchicalMultimodalExtractor:
    """
    Factory function to create hierarchical multimodal feature extractor.
    
    Args:
        observation_space: Gym observation space
        features_dim: Output feature dimension
        use_hierarchical_graph: Whether to use hierarchical graph processing
        **kwargs: Additional arguments
        
    Returns:
        Configured hierarchical multimodal extractor
    """
    return HierarchicalMultimodalExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        use_hierarchical_graph=use_hierarchical_graph,
        **kwargs
    )