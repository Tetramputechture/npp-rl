"""Meta-learning components for generalized N++ navigation.

This package implements meta-learning systems for N++ reinforcement learning.
Path prediction components have been moved to npp_rl.path_prediction.
"""

# Path prediction components moved to npp_rl.path_prediction
# Import them from there for compatibility:
try:
    from ..path_prediction import (
        GeneralizedPatternExtractor,
        TilePattern,
        EntityContext,
        WaypointPreference,
        OnlineRouteDiscovery,
        PathOutcome,
        BanditArm,
        LevelTopologyClassifier
    )
    
    __all__ = [
        'GeneralizedPatternExtractor',
        'TilePattern', 
        'EntityContext',
        'WaypointPreference',
        'OnlineRouteDiscovery',
        'PathOutcome',
        'BanditArm',
        'LevelTopologyClassifier'
    ]
except ImportError:
    # Fallback if path_prediction package has issues
    __all__ = []
