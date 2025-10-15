"""
Feature Extractors Package for N++ RL Agent

This package provides feature extractors for the N++ RL environment,
using various architectures for multimodal processing.

## Recommended Extractors (Architecture System)

For systematic architecture comparison and production training, use:

    **ConfigurableMultimodalExtractor** (npp_rl.optimization.configurable_extractor)
    - Unified system supporting 8 validated architectures
    - Use with ArchitectureTrainer for best results
    - Supports: full_hgt, simplified_hgt, gat, gcn, mlp_baseline, vision_free, etc.
    
    from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
    from npp_rl.optimization.architecture_configs import get_architecture_config
    
    config = get_architecture_config("full_hgt")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)

## Legacy Extractors (Maintained for Compatibility)

These extractors are maintained for backward compatibility and specific use cases:

    **HGTMultimodalExtractor** [LEGACY - Use ConfigurableMultimodalExtractor instead]
    - Original HGT implementation
    - Replaced by ConfigurableMultimodalExtractor with "full_hgt" config
    - Still used by: train_hierarchical_stable.py, npp_rl/agents/training.py
    
    **VisionFreeExtractor** [SPECIAL PURPOSE]
    - For environments WITHOUT graph observations
    - Uses only: game_state, reachability_features, entity_positions
    - Suitable for CPU training and rapid prototyping
    - Different from "vision_free" architecture config (which uses graphs)
    
    **MinimalStateExtractor** [SPECIAL PURPOSE]
    - Minimal state-only processing (game_state + reachability)
    - Fastest option for CPU training and debugging
    - No visual or graph processing

Usage Examples:

    # RECOMMENDED: Configurable system with validated architectures
    from npp_rl.training.architecture_trainer import ArchitectureTrainer
    trainer = ArchitectureTrainer(config_name="full_hgt", env_id="NPP-v0")
    trainer.train()

    # Legacy HGT extractor (use ConfigurableMultimodalExtractor instead)
    from npp_rl.feature_extractors import HGTMultimodalExtractor
    extractor = HGTMultimodalExtractor(observation_space, features_dim=512)
    
    # Special purpose: CPU training without graph obs
    from npp_rl.feature_extractors import VisionFreeExtractor
    extractor = VisionFreeExtractor(observation_space, features_dim=256)
"""

__all__ = [
    "HGTMultimodalExtractor",
    "VisionFreeExtractor", 
    "MinimalStateExtractor",
]

# Import extractors
from .hgt_multimodal import HGTMultimodalExtractor
from .vision_free_extractor import VisionFreeExtractor, MinimalStateExtractor
