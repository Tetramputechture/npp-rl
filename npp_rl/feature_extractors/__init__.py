"""
Feature Extractors Package for N++ RL Agent

This package provides the state-of-the-art hierarchical multimodal feature extractor
for the N++ RL environment, implementing multi-resolution graph processing with
DiffPool GNNs and adaptive fusion mechanisms.

Key Components:
    
    Primary Extractor:
        - HierarchicalMultimodalExtractor: Advanced multimodal extractor with:
          * Multi-resolution graph processing (6px, 24px, 96px)
          * DiffPool GNN with learnable hierarchical representations
          * Context-aware attention mechanisms
          * Auxiliary loss training for improved representations
    
    Factory Functions:
        - create_hierarchical_multimodal_extractor: Factory for the primary extractor

Usage Examples:

    # Primary hierarchical multimodal extractor (recommended)
    from npp_rl.feature_extractors import HierarchicalMultimodalExtractor
    extractor = HierarchicalMultimodalExtractor(
        observation_space,
        features_dim=512,
        use_hierarchical_graph=True
    )
    
    # Factory function
    from npp_rl.feature_extractors import create_hierarchical_multimodal_extractor
    extractor = create_hierarchical_multimodal_extractor(
        observation_space=env.observation_space,
        features_dim=512,
        use_hierarchical_graph=True
    )

Note: Legacy extractors (temporal, multimodal) have been moved to archive/
for reference. The hierarchical multimodal extractor provides superior
accuracy, robustness, and sample efficiency.
"""

from typing import Union, Literal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

# Import primary extractor
from .hierarchical_multimodal import (
    HierarchicalMultimodalExtractor,
    HierarchicalGraphObservationWrapper,
    create_hierarchical_multimodal_extractor
)


def create_feature_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    use_hierarchical_graph: bool = True,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Factory function to create the hierarchical multimodal feature extractor.
    
    This function creates the state-of-the-art hierarchical multimodal extractor
    with multi-resolution graph processing capabilities.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension (default: 512)
        use_hierarchical_graph: Whether to use hierarchical graph processing
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured hierarchical multimodal feature extractor instance
        
    Examples:
        # Standard hierarchical extractor
        extractor = create_feature_extractor(obs_space)
        
        # Without hierarchical graphs (fallback mode)
        extractor = create_feature_extractor(
            obs_space, use_hierarchical_graph=False
        )
    """
    return create_hierarchical_multimodal_extractor(
        observation_space=observation_space,
        features_dim=features_dim,
        use_hierarchical_graph=use_hierarchical_graph,
        **kwargs
    )


# Export all public components
__all__ = [
    # Primary hierarchical extractor
    'HierarchicalMultimodalExtractor',
    'HierarchicalGraphObservationWrapper',
    
    # Factory functions
    'create_feature_extractor',
    'create_hierarchical_multimodal_extractor',
]
