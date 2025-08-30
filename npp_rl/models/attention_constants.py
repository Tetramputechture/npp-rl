"""
Constants for attention mechanisms in multimodal feature extraction.

This module centralizes configuration constants to avoid magic numbers
and provide consistent defaults across attention components.
"""

# Default attention configuration
DEFAULT_EMBED_DIM = 512
DEFAULT_NUM_ATTENTION_HEADS = 8
DEFAULT_DROPOUT_RATE = 0.1

# Spatial attention defaults
DEFAULT_SPATIAL_HEIGHT = 16
DEFAULT_SPATIAL_WIDTH = 16
DEFAULT_GUIDANCE_DIM = 64

# Network architecture constants
CONV_KERNEL_SIZE_3x3 = 3
CONV_KERNEL_SIZE_1x1 = 1
CONV_PADDING_SAME = 1

# Feature processing constants
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
BILINEAR_MODE = 'bilinear'
ALIGN_CORNERS = False

# Normalization epsilon
LAYER_NORM_EPS = 1e-5