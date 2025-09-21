"""
Feature Extractors Package for N++ RL Agent

This package provides feature extractors for the N++ RL environment,
with the primary architecture using Heterogeneous Graph Transformers (HGT) for
optimal performance on complex spatial reasoning tasks.

Key Components:

    - HGTMultimodalExtractor: HGT-based extractor with:
        * Heterogeneous Graph Transformers with type-specific attention
        * Specialized processing for different node/edge types
        * Advanced multimodal fusion with cross-modal attention
        * Superior performance on complex spatial reasoning

    Factory Functions:
        - create_hgt_multimodal_extractor: Factory for the primary HGT extractor
        - create_hierarchical_multimodal_extractor: Factory for hierarchical extractor

Usage Examples:

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


Note: Legacy extractors (temporal, basic multimodal) have been moved to archive/
for reference. The HGT multimodal extractor provides superior accuracy, robustness,
and sample efficiency for complex spatial reasoning tasks.
"""

__all__ = [
    "HGTMultimodalExtractor",
    "create_hgt_multimodal_extractor",
]
