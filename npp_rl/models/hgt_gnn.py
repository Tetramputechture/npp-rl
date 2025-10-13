"""
Heterogeneous Graph Transformer (HGT) Main Module for NPP-RL.

This module provides the main interface for HGT functionality in the NPP-RL
system. It imports and re-exports components from the split HGT modules.

The HGT implementation is split into focused modules:
- hgt_layer.py: Core HGT layer implementation
- hgt_encoder.py: Multi-layer HGT encoder
- attention_mechanisms.py: Specialized attention mechanisms
- hgt_factory.py: Factory functions and production utilities
"""

# Import core HGT components
from .hgt_layer import (
    HGTLayer,
    EdgeType,
    create_hgt_layer
)

from .hgt_encoder import (
    HGTEncoder,
    create_hgt_encoder
)

from .attention_mechanisms import (
    TypeSpecificAttention,
    CrossModalAttention,
    create_type_specific_attention,
    create_cross_modal_attention
)
from .entity_type_system import HazardAwareAttention

from .hgt_factory import (
    HGTFactory,
    ProductionHGTConfig,
    MultimodalHGTSystem,
    create_production_hgt_layer,
    create_production_hgt_encoder,
    create_production_multimodal_hgt,
    get_production_hgt_config,
    validate_hgt_installation,
    default_hgt_factory
)

# Public API exports
__all__ = [
    # Core components
    "HGTLayer",
    "HGTEncoder", 
    "EdgeType",
    
    # Attention mechanisms
    "TypeSpecificAttention",
    "HazardAwareAttention", 
    "CrossModalAttention",
    
    # Factory and configuration
    "HGTFactory",
    "ProductionHGTConfig",
    "MultimodalHGTSystem",
    
    # Factory functions
    "create_hgt_layer",
    "create_hgt_encoder",
    "create_type_specific_attention",
 
    "create_cross_modal_attention",
    "create_production_hgt_layer",
    "create_production_hgt_encoder",
    "create_production_multimodal_hgt",
    
    # Utilities
    "get_production_hgt_config",
    "validate_hgt_installation",
    "default_hgt_factory"
]


# Production-ready defaults for NPP-RL
def get_npp_rl_hgt_config() -> ProductionHGTConfig:
    """Get HGT configuration optimized for NPP-RL production use."""
    return ProductionHGTConfig()


def create_npp_rl_hgt_encoder(**kwargs) -> HGTEncoder:
    """Create HGT encoder optimized for NPP-RL with simplified features."""
    config = get_npp_rl_hgt_config()
    
    # NPP-RL specific defaults
    defaults = {
        "node_feature_dim": config.NODE_FEATURE_DIM,  # 8 simplified features
        "edge_feature_dim": config.EDGE_FEATURE_DIM,  # 4 simplified features
        "hidden_dim": config.HIDDEN_DIM,              # 128 for efficiency
        "num_layers": config.NUM_LAYERS,              # 3 layers
        "num_heads": config.NUM_HEADS,                # 8 heads
        "output_dim": config.OUTPUT_DIM,              # 256 output
        "num_node_types": config.NUM_NODE_TYPES,      # 6 entity types
        "num_edge_types": config.NUM_EDGE_TYPES,      # 3 simplified edge types
        "dropout": config.DROPOUT,                    # 0.1 dropout
        "global_pool": config.GLOBAL_POOL,            # mean_max pooling
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return create_hgt_encoder(**defaults)


def create_npp_rl_multimodal_hgt(
    visual_dim: int = 512,
    state_dim: int = 64,
    **kwargs
) -> MultimodalHGTSystem:
    """Create complete multimodal HGT system for NPP-RL production."""
    return create_production_multimodal_hgt(
        visual_dim=visual_dim,
        state_dim=state_dim,
        **kwargs
    )


# Module-level validation
def validate_hgt_components():
    """Validate that all HGT components are properly installed."""
    try:
        results = validate_hgt_installation()
        if not results["production_ready"]:
            raise ImportError(f"HGT components not ready: {results.get('error', 'Unknown error')}")
        return True
    except Exception as e:
        raise ImportError(f"HGT validation failed: {e}")


# Perform validation on import
validate_hgt_components()


# Version information
__version__ = "2.0.0"  # Split architecture version
__author__ = "NPP-RL Team"
__description__ = "Production-ready Heterogeneous Graph Transformer for NPP-RL"