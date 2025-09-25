"""
Feature Extractors Package for N++ RL Agent

This package provides feature extractors for the N++ RL environment,
with production-ready architectures using Heterogeneous Graph Transformers (HGT) for
optimal performance on complex spatial reasoning tasks.

Key Components:

    - RobustHGTMultimodalExtractor: Production-ready HGT extractor with:
        * 3D CNN for temporal processing (12-frame stacks)
        * 2D CNN with spatial attention for global view
        * Full Heterogeneous Graph Transformers with type-specific attention
        * Advanced cross-modal fusion with attention mechanisms
        * Designed for generalizability across diverse NPP levels

    - HGTMultimodalExtractor: Simplified HGT extractor (legacy):
        * Basic 2D CNN for visual processing
        * Simplified graph processing
        * Maintained for compatibility

    Factory Functions:
        - create_robust_hgt_extractor: Factory for production-ready extractor
        - create_hgt_multimodal_extractor: Factory for legacy extractor

Usage Examples:

    # Production-ready robust extractor (recommended)
    from npp_rl.feature_extractors import RobustHGTMultimodalExtractor
    extractor = RobustHGTMultimodalExtractor(
        observation_space,
        features_dim=512,
        debug=False
    )

    # Factory function for robust extractor
    from npp_rl.feature_extractors import create_robust_hgt_extractor
    extractor = create_robust_hgt_extractor(
        observation_space=env.observation_space,
        features_dim=512
    )

    # Legacy simplified extractor
    from npp_rl.feature_extractors import HGTMultimodalExtractor
    extractor = HGTMultimodalExtractor(
        observation_space,
        features_dim=512
    )

Architecture Features:
- 3D CNN: Temporal processing of 12-frame stacks for movement pattern recognition
- 2D CNN: Spatial processing with attention mechanisms for global level understanding
- HGT: Full heterogeneous graph processing for entity relationships and reachability
- Cross-modal Fusion: Advanced attention mechanisms for optimal feature integration
- Generalizability: Robust architecture designed for diverse NPP level completion
"""

__all__ = [
    "RobustHGTMultimodalExtractor",
    "HGTMultimodalExtractor", 
    "create_robust_hgt_extractor",
    "create_hgt_multimodal_extractor",
]

# Import extractors
from .robust_hgt_multimodal import RobustHGTMultimodalExtractor
from .hgt_multimodal import HGTMultimodalExtractor


def create_robust_hgt_extractor(observation_space, features_dim: int = 512, debug: bool = False):
    """
    Factory function for creating production-ready robust HGT multimodal extractor.
    
    Args:
        observation_space: Environment observation space
        features_dim: Output feature dimension
        debug: Enable debug output
        
    Returns:
        RobustHGTMultimodalExtractor instance
    """
    return RobustHGTMultimodalExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        debug=debug
    )


def create_hgt_multimodal_extractor(observation_space, features_dim: int = 512, debug: bool = False):
    """
    Factory function for creating legacy simplified HGT multimodal extractor.
    
    Args:
        observation_space: Environment observation space
        features_dim: Output feature dimension
        debug: Enable debug output
        
    Returns:
        HGTMultimodalExtractor instance
    """
    return HGTMultimodalExtractor(
        observation_space=observation_space,
        features_dim=features_dim,
        debug=debug
    )
