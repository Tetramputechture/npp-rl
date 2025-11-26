"""Path prediction system for generalized N++ navigation.

This package contains all components for predicting and learning optimal paths
in N++ levels, including pattern extraction, multi-path prediction, and 
online route discovery for zero-shot generalization to unseen levels.
"""

from .pattern_extractor import (
    GeneralizedPatternExtractor,
    TilePattern,
    EntityContext,
    WaypointPreference
)

from .multipath_predictor import (
    ProbabilisticPathPredictor,
    CandidatePath,
    create_multipath_predictor
)

from .route_discovery import (
    OnlineRouteDiscovery,
    PathOutcome,
    BanditArm,
    LevelTopologyClassifier
)

from . import graph_utils

__all__ = [
    # Pattern extraction
    'GeneralizedPatternExtractor',
    'TilePattern',
    'EntityContext', 
    'WaypointPreference',
    
    # Multi-path prediction
    'ProbabilisticPathPredictor',
    'CandidatePath',
    'create_multipath_predictor',
    
    # Route discovery
    'OnlineRouteDiscovery',
    'PathOutcome',
    'BanditArm',
    'LevelTopologyClassifier',
    
    # Graph utilities
    'graph_utils',
]
