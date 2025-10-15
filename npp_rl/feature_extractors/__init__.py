"""
Feature Extractors Package for N++ RL Agent

This package provides the unified ConfigurableMultimodalExtractor system for
multimodal feature extraction with 8 validated architecture variants.

## Current System

**ConfigurableMultimodalExtractor** - Unified feature extraction system
- Supports 8 validated architectures
- Use with ArchitectureTrainer or directly with PPO
- Architectures: full_hgt, simplified_hgt, gat, gcn, mlp_baseline,
                 vision_free, no_global_view, local_frames_only

## Usage Examples

### Basic Usage with Architecture Configs

    from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
    from npp_rl.training.architecture_configs import get_architecture_config

    # Example: Full HGT architecture
    config = get_architecture_config("full_hgt")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)

    # Example: MLP baseline (no graph processing)
    config = get_architecture_config("mlp_baseline")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)

    # Example: Vision-free (no visual processing)
    config = get_architecture_config("vision_free")
    extractor = ConfigurableMultimodalExtractor(observation_space, config)

### Usage with Training Scripts

    # Using npp_rl/agents/training.py
    python -m npp_rl.agents.training --architecture full_hgt --num_envs 64

    # Using ArchitectureTrainer for systematic comparison
    from npp_rl.training.architecture_trainer import ArchitectureTrainer
    trainer = ArchitectureTrainer(config_name="full_hgt", env_id="NPP-v0")
    trainer.train()

### Direct PPO Integration

    from stable_baselines3 import PPO
    from npp_rl.feature_extractors import ConfigurableMultimodalExtractor
    from npp_rl.training.architecture_configs import get_architecture_config

    config = get_architecture_config("full_hgt")
    policy_kwargs = {
        'features_extractor_class': ConfigurableMultimodalExtractor,
        'features_extractor_kwargs': {'config': config},
    }

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)

## Migration Notes

Legacy extractors (HGTMultimodalExtractor, VisionFreeExtractor, MinimalStateExtractor)
have been removed and replaced by ConfigurableMultimodalExtractor:

    - HGTMultimodalExtractor → Use get_architecture_config("full_hgt")
    - VisionFreeExtractor → Use get_architecture_config("vision_free")
    - MinimalStateExtractor → Use get_architecture_config("mlp_baseline")

All training scripts have been updated to use the unified system.
"""

from .configurable_extractor import ConfigurableMultimodalExtractor

__all__ = ["ConfigurableMultimodalExtractor"]
