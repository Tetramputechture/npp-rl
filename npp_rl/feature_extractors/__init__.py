"""
Feature Extractors Package for N++ RL Agent

This package provides state-of-the-art feature extractors for the N++ RL environment,
with the primary architecture using Heterogeneous Graph Transformers (HGT) for
optimal performance on complex spatial reasoning tasks.

Key Components:
    
    Primary Extractor (RECOMMENDED):
        - HGTMultimodalExtractor: State-of-the-art HGT-based extractor with:
          * Heterogeneous Graph Transformers with type-specific attention
          * Specialized processing for different node/edge types
          * Advanced multimodal fusion with cross-modal attention
          * Superior performance on complex spatial reasoning
    
    Secondary Extractor:
        - HierarchicalMultimodalExtractor: Hierarchical multimodal extractor with:
          * Multi-resolution graph processing (6px, 24px, 96px)
          * DiffPool GNN with learnable hierarchical representations
          * Context-aware attention mechanisms
    
    Factory Functions:
        - create_hgt_multimodal_extractor: Factory for the primary HGT extractor
        - create_hierarchical_multimodal_extractor: Factory for hierarchical extractor

Usage Examples:

    # PRIMARY: HGT multimodal extractor (RECOMMENDED)
    from npp_rl.feature_extractors import HGTMultimodalExtractor
    extractor = HGTMultimodalExtractor(
        observation_space,
        features_dim=512,
        hgt_hidden_dim=256,
        hgt_num_layers=3
    )
    
    # Factory function for HGT extractor
    from npp_rl.feature_extractors import create_hgt_multimodal_extractor
    extractor = create_hgt_multimodal_extractor(
        observation_space=env.observation_space,
        features_dim=512
    )
    
    # SECONDARY: Hierarchical multimodal extractor
    from npp_rl.feature_extractors import HierarchicalMultimodalExtractor
    extractor = HierarchicalMultimodalExtractor(
        observation_space,
        features_dim=512,
        use_hierarchical_graph=True
    )

Note: Legacy extractors (temporal, basic multimodal) have been moved to archive/
for reference. The HGT multimodal extractor provides superior accuracy, robustness,
and sample efficiency for complex spatial reasoning tasks.
"""

from typing import Union, Literal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

# Import primary HGT extractor
from .hgt_multimodal import (
    HGTMultimodalExtractor,
    create_hgt_multimodal_extractor
)

# Import secondary hierarchical extractor
from .hierarchical_multimodal import (
    HierarchicalMultimodalExtractor,
    HierarchicalGraphObservationWrapper,
    create_hierarchical_multimodal_extractor
)


def create_feature_extractor(
    observation_space: SpacesDict,
    features_dim: int = 512,
    extractor_type: Literal['hgt', 'hierarchical'] = 'hgt',
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Factory function to create feature extractors.
    
    This function creates the appropriate feature extractor based on the specified type.
    The HGT extractor is recommended for optimal performance.
    
    Args:
        observation_space: Gym observation space dictionary
        features_dim: Output feature dimension (default: 512)
        extractor_type: Type of extractor ('hgt' or 'hierarchical')
        **kwargs: Additional arguments passed to the extractor
        
    Returns:
        Configured feature extractor instance
        
    Examples:
        # PRIMARY: HGT extractor (recommended)
        extractor = create_feature_extractor(obs_space, extractor_type='hgt')
        
        # SECONDARY: Hierarchical extractor
        extractor = create_feature_extractor(obs_space, extractor_type='hierarchical')
    """
    if extractor_type == 'hgt':
        return create_hgt_multimodal_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            **kwargs
        )
    elif extractor_type == 'hierarchical':
        return create_hierarchical_multimodal_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}. "
                        f"Must be 'hgt' or 'hierarchical'.")


# Export all public components
__all__ = [
    # PRIMARY: HGT extractor (RECOMMENDED)
    'HGTMultimodalExtractor',
    'create_hgt_multimodal_extractor',
    
    # SECONDARY: Hierarchical extractor
    'HierarchicalMultimodalExtractor',
    'HierarchicalGraphObservationWrapper',
    'create_hierarchical_multimodal_extractor',
    
    # Factory functions
    'create_feature_extractor',
]
