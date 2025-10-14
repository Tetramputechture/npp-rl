"""
Feature Extractors Package for N++ RL Agent

This package provides feature extractors for the N++ RL environment,
using Heterogeneous Graph Transformers (HGT) for multimodal processing.

Key Components:

    - HGTMultimodalExtractor: Enhanced HGT extractor with:
        * 3D CNN for temporal processing (12-frame stacks)
        * 2D CNN with spatial attention for global view
        * Full Heterogeneous Graph Transformers with type-specific attention
        * Cross-modal fusion with attention mechanisms
        * Designed for generalizability across diverse NPP levels

    - HGTMultimodalExtractor: Alternative HGT implementation:
        * Similar architecture with different implementation approach
        * Available for comparison and experimentation

Usage Examples:

    # Main HGT extractor (recommended)
    from npp_rl.feature_extractors import HGTMultimodalExtractor
    extractor = HGTMultimodalExtractor(
        observation_space,
        features_dim=512,
        debug=False
    )

    # Alternative implementation
    from npp_rl.feature_extractors import HGTMultimodalExtractor
    extractor = HGTMultimodalExtractor(
        observation_space,
        features_dim=512,
        debug=False
    )

Architecture Features:
- 3D CNN: Temporal processing of 12-frame stacks for movement pattern recognition
- 2D CNN: Spatial processing with attention mechanisms for global level understanding
- HGT: Full heterogeneous graph processing for entity relationships and reachability
- Cross-modal Fusion: Attention mechanisms for optimal feature integration
- Generalizability: Architecture designed for diverse NPP level completion
"""

__all__ = [
    "HGTMultimodalExtractor",
    "VisionFreeExtractor",
    "MinimalStateExtractor",
]

# Import extractors
from .hgt_multimodal import HGTMultimodalExtractor
from .vision_free_extractor import VisionFreeExtractor, MinimalStateExtractor
