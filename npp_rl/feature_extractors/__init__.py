"""
Feature Extractors Package for N++ RL Agent

This package previously contained legacy feature extractors that have been
replaced by the unified ConfigurableMultimodalExtractor system.

## Current Approach (Use This)

For all training and architecture comparison, use:

    **ConfigurableMultimodalExtractor** (npp_rl.optimization.configurable_extractor)
    - Unified system supporting 8 validated architectures
    - Use with ArchitectureTrainer or directly with PPO
    - Architectures: full_hgt, simplified_hgt, gat, gcn, mlp_baseline, 
                     vision_free, no_global_view, local_frames_only
    
    from npp_rl.optimization.configurable_extractor import ConfigurableMultimodalExtractor
    from npp_rl.optimization.architecture_configs import get_architecture_config
    
    # Example: Full HGT architecture
    config = get_architecture_config("full_hgt")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)
    
    # Example: MLP baseline (no graph processing)
    config = get_architecture_config("mlp_baseline")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)
    
    # Example: Vision-free (no visual processing)
    config = get_architecture_config("vision_free")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)

## Removed Legacy Extractors

The following extractors have been removed and replaced by ConfigurableMultimodalExtractor:

    - HGTMultimodalExtractor → Use get_architecture_config("full_hgt")
    - VisionFreeExtractor → Use get_architecture_config("vision_free")
    - MinimalStateExtractor → Use get_architecture_config("mlp_baseline")

All training scripts have been updated to use the new unified system.

Usage with Training Scripts:

    # Using npp_rl/agents/training.py
    python -m npp_rl.agents.training --architecture full_hgt --num_envs 64
    
    # Using ArchitectureTrainer for systematic comparison
    from npp_rl.training.architecture_trainer import ArchitectureTrainer
    trainer = ArchitectureTrainer(config_name="full_hgt", env_id="NPP-v0")
    trainer.train()
"""

__all__ = []  # No extractors exported from this package anymore
