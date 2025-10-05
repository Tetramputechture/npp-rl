"""
Simplified Graph Neural Network architectures for Task 3.1 comparison.

DEPRECATED: This module has been split into separate files for better organization.
Please import from the specific modules:
- npp_rl.models.gcn: GCNLayer, GCNEncoder
- npp_rl.models.gat: GATLayer, GATEncoder
- npp_rl.models.simplified_hgt: SimplifiedHGTEncoder

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

# Import from new locations
from .gcn import GCNLayer, GCNEncoder
from .gat import GATLayer, GATEncoder
from .simplified_hgt import SimplifiedHGTEncoder

# Emit deprecation warning when this module is imported
warnings.warn(
    "Importing from 'simplified_gnn' is deprecated. "
    "Please import from specific modules: "
    "'npp_rl.models.gcn', 'npp_rl.models.gat', or 'npp_rl.models.simplified_hgt'.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "GCNLayer",
    "GCNEncoder",
    "GATLayer",
    "GATEncoder",
    "SimplifiedHGTEncoder",
]
