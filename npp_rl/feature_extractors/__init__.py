"""
Feature Extractors Package for N++ RL Agent

This package provides a comprehensive collection of feature extractors for
different types of observations and modeling needs in the N++ RL environment.

The package is organized into specialized modules:
- `temporal`: 3D CNN-based extractors for temporal modeling
- `multimodal`: Advanced extractors supporting multiple observation modalities

Key Components:
    
    Temporal Extractors:
        - TemporalFeatureExtractor: 3D CNN for frame-stacked observations
        - FeatureExtractor: Alias for backward compatibility
    
    Multimodal Extractors:
        - MultimodalGraphExtractor: Full multimodal with GNN support
        - MultimodalExtractor: Simplified multimodal without graphs
        - NppMultimodalGraphExtractor: Backward compatibility alias
        - NppMultimodalExtractor: Backward compatibility alias
    
    Factory Functions:
        - create_feature_extractor: Smart factory for any extractor type
        - create_multimodal_extractor: Factory for multimodal extractors

Usage Examples:

    # Basic temporal modeling (recommended for most use cases)
    from npp_rl.feature_extractors import TemporalFeatureExtractor
    extractor = TemporalFeatureExtractor(observation_space, features_dim=512)
    
    # Advanced multimodal with graph support
    from npp_rl.feature_extractors import MultimodalGraphExtractor
    extractor = MultimodalGraphExtractor(
        observation_space,
        features_dim=512,
        use_graph_obs=True
    )
    
    # Factory function (automatic selection)
    from npp_rl.feature_extractors import create_feature_extractor
    extractor = create_feature_extractor(
        observation_space,
        extractor_type='temporal',  # or 'multimodal'
        features_dim=512
    )
    
    # Backward compatibility
    from npp_rl.feature_extractors import FeatureExtractor  # Original name
    from npp_rl.feature_extractors import create_feature_extractor as create_fe
"""

from typing import Union, Literal
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict as SpacesDict

# Import all extractors
from .temporal import TemporalFeatureExtractor, FeatureExtractor
from .multimodal import (
    MultimodalGraphExtractor,
    MultimodalExtractor,
    create_multimodal_extractor,
    # Backward compatibility aliases
    NppMultimodalGraphExtractor,
    NppMultimodalExtractor,
)


def create_feature_extractor(
    observation_space: SpacesDict,
    extractor_type: Literal['temporal', 'multimodal'] = 'temporal',
    features_dim: int = 512,
    use_graph_obs: bool = False,
    **kwargs
) -> BaseFeaturesExtractor:
    """
    Smart factory function to create appropriate feature extractor.
    
    This function provides a unified interface for creating any type of
    feature extractor based on the specified type and parameters.
    
    Args:
        observation_space: Gym observation space dictionary
        extractor_type: Type of extractor to create:
            - 'temporal': TemporalFeatureExtractor with 3D CNNs
            - 'multimodal': MultimodalExtractor with optional GNN support
        features_dim: Output feature dimension (default: 512)
        use_graph_obs: Whether to use graph observations (multimodal only)
        **kwargs: Additional arguments passed to the specific extractor
        
    Returns:
        Configured feature extractor instance
        
    Raises:
        ValueError: If extractor_type is not recognized
        
    Examples:
        # Basic temporal extractor
        extractor = create_feature_extractor(obs_space, 'temporal')
        
        # Multimodal without graphs
        extractor = create_feature_extractor(obs_space, 'multimodal')
        
        # Multimodal with graphs
        extractor = create_feature_extractor(
            obs_space, 'multimodal', use_graph_obs=True
        )
    """
    if extractor_type == 'temporal':
        return TemporalFeatureExtractor(
            observation_space=observation_space,
            features_dim=features_dim,
            **kwargs
        )
    elif extractor_type == 'multimodal':
        return create_multimodal_extractor(
            observation_space=observation_space,
            features_dim=features_dim,
            use_graph_obs=use_graph_obs,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown extractor_type: {extractor_type}. "
            f"Expected 'temporal' or 'multimodal'."
        )


# Export all public components
__all__ = [
    # Temporal extractors
    'TemporalFeatureExtractor',
    'FeatureExtractor',  # Backward compatibility
    
    # Multimodal extractors
    'MultimodalGraphExtractor',
    'MultimodalExtractor',
    'NppMultimodalGraphExtractor',  # Backward compatibility
    'NppMultimodalExtractor',       # Backward compatibility
    
    # Factory functions
    'create_feature_extractor',
    'create_multimodal_extractor',
]


# Backward compatibility - direct access to old names
# This allows existing code to continue working without modification
def create_feature_extractor_legacy(*args, **kwargs):
    """Legacy factory function for backward compatibility."""
    return create_multimodal_extractor(*args, **kwargs)


# Maintain old import path compatibility
create_feature_extractor_old = create_multimodal_extractor
