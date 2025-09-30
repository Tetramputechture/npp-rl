"""
Constants for Heterogeneous Graph Transformer (HGT) mechanisms in multimodal feature extraction.

This module centralizes configuration constants to avoid magic numbers
and provide consistent defaults across HGT components.
"""

from dataclasses import dataclass
from typing import Tuple


# === Configuration Data Structures ===


@dataclass(frozen=True)
class ConvLayerConfig:
    """Configuration for a single convolutional layer."""

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]


@dataclass(frozen=True)
class CNNArchitectureConfig:
    """Complete CNN architecture configuration for multimodal processing."""

    # 3D CNN layers for temporal visual processing
    conv3d_layer1: ConvLayerConfig
    conv3d_layer2: ConvLayerConfig
    conv3d_layer3: ConvLayerConfig

    # 2D CNN layers for global view processing
    conv2d_layer1: ConvLayerConfig
    conv2d_layer2: ConvLayerConfig
    conv2d_layer3: ConvLayerConfig


@dataclass(frozen=True)
class PoolingConfig:
    """Pooling and feature sizing configuration."""

    adaptive_pool3d_output_size: Tuple[int, int, int]
    adaptive_pool2d_output_size: Tuple[int, int]
    cnn_final_channels: int

    @property
    def cnn_flattened_size(self) -> int:
        """Calculate flattened size after pooling."""
        return (
            self.cnn_final_channels
            * self.adaptive_pool2d_output_size[0]
            * self.adaptive_pool2d_output_size[1]
        )


@dataclass(frozen=True)
class DefaultConfig:
    """Default configuration values for HGT multimodal extractor."""

    embed_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 256
    num_attention_heads: int = 8
    dropout_rate: float = 0.1

    # Visual processing defaults
    visual_hidden_dim: int = 256
    global_hidden_dim: int = 128

    # State processing defaults
    state_hidden_dim: int = 128

    # Reachability processing defaults (8-dimensional features)
    reachability_dim: int = 8
    reachability_hidden_dim: int = 16  # 8 * 2

    # Spatial attention defaults
    spatial_height: int = 16
    spatial_width: int = 16
    guidance_dim: int = 64


@dataclass(frozen=True)
class FactoryConfig:
    """Default configuration for factory function."""

    features_dim: int = 512
    hgt_hidden_dim: int = 256
    hgt_num_layers: int = 3
    hgt_output_dim: int = 256


@dataclass(frozen=True)
class MultiplierConfig:
    """Feature dimension multipliers and factors."""

    hgt_output_multiplier: int = 2  # For mean_max pooling concatenation
    fusion_expansion_factor: int = 2  # For fusion network intermediate layer
    feature_expansion_factor: int = 2  # General feature expansion
    guidance_dim_reduction_factor: int = 2
    attention_head_reduction_factor: int = 2


@dataclass(frozen=True)
class HGTConfig:
    """Configuration for Heterogeneous Graph Transformer (SIMPLIFIED)."""

    # SIMPLIFIED: Node feature dimensions reduced from 31 to 8
    node_feat_dim: int = (
        8  # Simplified node features (position, tile, entity, distances)
    )
    edge_feat_dim: int = 4  # Simplified edge features (type + weight)

    # HGT architecture
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1

    # SIMPLIFIED: Node and edge type counts
    num_node_types: int = 6  # From entity_type_system.py (unchanged)
    num_edge_types: int = 3  # Simplified: ADJACENT, LOGICAL, REACHABLE


# === Configuration Instances ===

# CNN Architecture Configuration
CNN_CONFIG = CNNArchitectureConfig(
    # 3D CNN for temporal visual processing
    conv3d_layer1=ConvLayerConfig(
        in_channels=1,
        out_channels=32,
        kernel_size=(4, 7, 7),
        stride=(2, 2, 2),
        padding=(1, 3, 3),
    ),
    conv3d_layer2=ConvLayerConfig(
        in_channels=32,
        out_channels=64,
        kernel_size=(3, 5, 5),
        stride=(1, 2, 2),
        padding=(1, 2, 2),
    ),
    conv3d_layer3=ConvLayerConfig(
        in_channels=64,
        out_channels=128,
        kernel_size=(2, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
    ),
    # 2D CNN for global view processing
    conv2d_layer1=ConvLayerConfig(
        in_channels=1, out_channels=32, kernel_size=(7,), stride=(2,), padding=(3,)
    ),
    conv2d_layer2=ConvLayerConfig(
        in_channels=32, out_channels=64, kernel_size=(5,), stride=(2,), padding=(2,)
    ),
    conv2d_layer3=ConvLayerConfig(
        in_channels=64, out_channels=128, kernel_size=(3,), stride=(2,), padding=(1,)
    ),
)

# Pooling Configuration
POOLING_CONFIG = PoolingConfig(
    adaptive_pool3d_output_size=(1, 4, 4),
    adaptive_pool2d_output_size=(4, 4),
    cnn_final_channels=128,
)

# Default Configuration
DEFAULT_CONFIG = DefaultConfig()

# Factory Configuration
FACTORY_CONFIG = FactoryConfig()

# Multiplier Configuration
MULTIPLIER_CONFIG = MultiplierConfig()

# HGT Configuration (SIMPLIFIED)
HGT_CONFIG = HGTConfig()


# === Additional Constants (not in configs) ===

# Network architecture constants
CONV_KERNEL_SIZE_3x3 = 3
CONV_KERNEL_SIZE_1x1 = 1
CONV_PADDING_SAME = 1

# Dropout and regularization defaults
DEFAULT_DROPOUT_RATE = 0.1

# Spatial attention defaults
DEFAULT_GUIDANCE_DIM = 128
DEFAULT_SPATIAL_HEIGHT = 32
DEFAULT_SPATIAL_WIDTH = 32
DEFAULT_NUM_ATTENTION_HEADS = 8

# Feature expansion factors
FEATURE_EXPANSION_FACTOR = 2
GUIDANCE_DIM_REDUCTION_FACTOR = 2
ATTENTION_HEAD_REDUCTION_FACTOR = 2

# Residual connection defaults
DEFAULT_RESIDUAL_WEIGHT = 0.7

# Minimum values for validation
MIN_ATTENTION_HEADS = 1
MIN_SPATIAL_DIM = 4
MIN_FEATURE_DIM = 1

# Multi-scale attention defaults
DEFAULT_SCALES = [8, 16, 32]

# Interpolation settings
BILINEAR_MODE = "bilinear"
ALIGN_CORNERS = False

# Normalization epsilon
LAYER_NORM_EPS = 1e-5
