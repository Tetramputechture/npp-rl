"""
Configuration system for Phase 2 features.

This module provides configuration classes and utilities for managing
the various Phase 2 components (ICM, GNN, BC, etc.).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


@dataclass
class ICMConfig:
    """Configuration for Intrinsic Curiosity Module."""
    enabled: bool = True
    feature_dim: int = 512
    action_dim: int = 6
    hidden_dim: int = 256
    eta: float = 0.01  # Intrinsic reward scaling
    lambda_inv: float = 0.1  # Inverse model loss weight
    lambda_fwd: float = 0.9  # Forward model loss weight
    learning_rate: float = 1e-3
    alpha: float = 0.1  # Intrinsic reward weight
    r_int_clip: float = 1.0  # Max intrinsic reward
    update_frequency: int = 1  # Update ICM every N steps
    buffer_size: int = 10000  # Experience buffer size
    share_backbone: bool = True  # Share backbone with policy


@dataclass
class GraphConfig:
    """Configuration for Graph Neural Network observations."""
    enabled: bool = False
    node_feature_dim: int = 85  # From GraphBuilder (67 + 18 physics features)
    edge_feature_dim: int = 16  # From GraphBuilder (updated for trajectory features)
    hidden_dim: int = 128
    num_layers: int = 3
    output_dim: int = 256
    aggregator: str = 'mean'  # 'mean', 'max', 'sum'
    global_pool: str = 'mean_max'  # 'mean', 'max', 'mean_max'
    dropout: float = 0.1
    normalize_features: str = 'l2'  # 'l2', 'batch', 'none'


@dataclass
class BCConfig:
    """Configuration for Behavioral Cloning pretraining."""
    enabled: bool = False
    dataset_dir: str = 'datasets/shards'
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 10
    entropy_coef: float = 0.01
    freeze_backbone_steps: int = 0
    max_episodes: Optional[int] = None
    quality_filter: Optional[Dict[str, Any]] = None
    level_filter: Optional[List[str]] = None
    stratify_by_level: bool = True
    validation_split: float = 0.1


@dataclass
class FeatureExtractorConfig:
    """Configuration for multimodal feature extractor."""
    type: str = 'multimodal'  # 'multimodal', 'enhanced_3d', 'enhanced_cnn'
    features_dim: int = 512
    use_3d_conv: bool = True
    use_graph_obs: bool = False
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_output_dim: int = 256


@dataclass
class ExplorationConfig:
    """Configuration for exploration metrics and evaluation."""
    enabled: bool = True
    grid_width: int = 42
    grid_height: int = 23
    cell_size: int = 24
    window_size: int = 100  # Rolling metrics window
    log_frequency: int = 1000  # Log metrics every N steps
    track_level_complexity: bool = True


@dataclass
class Phase2Config:
    """Main configuration for Phase 2 features."""
    # Component configurations
    icm: ICMConfig = field(default_factory=ICMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    
    # Global settings
    device: str = 'auto'
    seed: int = 42
    experiment_name: str = 'phase2_experiment'
    output_dir: str = 'experiments'
    
    # Training settings
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    
    # Logging
    use_tensorboard: bool = True
    use_wandb: bool = False
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure consistency between components
        if self.graph.enabled:
            self.feature_extractor.use_graph_obs = True
        
        # Set feature extractor dimensions based on graph config
        if self.feature_extractor.use_graph_obs:
            self.feature_extractor.gnn_hidden_dim = self.graph.hidden_dim
            self.feature_extractor.gnn_num_layers = self.graph.num_layers
            self.feature_extractor.gnn_output_dim = self.graph.output_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'icm': self.icm.__dict__,
            'graph': self.graph.__dict__,
            'bc': self.bc.__dict__,
            'feature_extractor': self.feature_extractor.__dict__,
            'exploration': self.exploration.__dict__,
            'device': self.device,
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'total_timesteps': self.total_timesteps,
            'eval_frequency': self.eval_frequency,
            'save_frequency': self.save_frequency,
            'use_tensorboard': self.use_tensorboard,
            'use_wandb': self.use_wandb,
            'log_level': self.log_level
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Phase2Config':
        """Create config from dictionary."""
        # Extract component configs
        icm_config = ICMConfig(**config_dict.get('icm', {}))
        graph_config = GraphConfig(**config_dict.get('graph', {}))
        bc_config = BCConfig(**config_dict.get('bc', {}))
        feature_config = FeatureExtractorConfig(**config_dict.get('feature_extractor', {}))
        exploration_config = ExplorationConfig(**config_dict.get('exploration', {}))
        
        # Extract global settings
        global_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['icm', 'graph', 'bc', 'feature_extractor', 'exploration']}
        
        return cls(
            icm=icm_config,
            graph=graph_config,
            bc=bc_config,
            feature_extractor=feature_config,
            exploration=exploration_config,
            **global_settings
        )
    
    def save(self, path: str):
        """Save config to JSON file."""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Phase2Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_default_config() -> Phase2Config:
    """Create default Phase 2 configuration."""
    return Phase2Config()


def create_icm_only_config() -> Phase2Config:
    """Create config with only ICM enabled."""
    config = Phase2Config()
    config.icm.enabled = True
    config.graph.enabled = False
    config.bc.enabled = False
    config.feature_extractor.use_graph_obs = False
    return config


def create_graph_only_config() -> Phase2Config:
    """Create config with only graph observations enabled."""
    config = Phase2Config()
    config.icm.enabled = False
    config.graph.enabled = True
    config.bc.enabled = False
    config.feature_extractor.use_graph_obs = True
    return config


def create_full_phase2_config() -> Phase2Config:
    """Create config with all Phase 2 features enabled."""
    config = Phase2Config()
    config.icm.enabled = True
    config.graph.enabled = True
    config.bc.enabled = True
    config.feature_extractor.use_graph_obs = True
    return config


def create_bc_pretrain_config() -> Phase2Config:
    """Create config optimized for BC pretraining."""
    config = Phase2Config()
    config.icm.enabled = False
    config.graph.enabled = False  # Can be enabled if graph data available
    config.bc.enabled = True
    config.bc.epochs = 20
    config.bc.batch_size = 128
    config.bc.learning_rate = 1e-3
    config.feature_extractor.use_graph_obs = False
    return config


def get_config_presets() -> Dict[str, Phase2Config]:
    """Get dictionary of configuration presets."""
    return {
        'default': create_default_config(),
        'icm_only': create_icm_only_config(),
        'graph_only': create_graph_only_config(),
        'full_phase2': create_full_phase2_config(),
        'bc_pretrain': create_bc_pretrain_config()
    }


def validate_config(config: Phase2Config) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check for conflicting settings
    if config.feature_extractor.use_graph_obs and not config.graph.enabled:
        messages.append("Warning: Graph observations enabled in feature extractor but graph config disabled")
    
    if config.bc.enabled and not Path(config.bc.dataset_dir).exists():
        messages.append(f"Warning: BC dataset directory does not exist: {config.bc.dataset_dir}")
    
    # Check reasonable parameter ranges
    if config.icm.eta > 1.0:
        messages.append("Warning: ICM eta parameter is very large (>1.0)")
    
    if config.icm.alpha > 0.5:
        messages.append("Warning: ICM alpha parameter is large (>0.5), may dominate extrinsic rewards")
    
    if config.graph.num_layers > 5:
        messages.append("Warning: Many GNN layers (>5) may cause over-smoothing")
    
    # Check device availability
    if config.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            messages.append("Warning: CUDA requested but not available")
    
    return messages