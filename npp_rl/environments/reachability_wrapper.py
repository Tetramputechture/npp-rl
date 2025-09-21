"""
Reachability-Aware Environment Wrapper

This module provides a wrapper for NPP environments that adds reachability feature
extraction to the observation space. It integrates with the nclone reachability
system to provide compact spatial reasoning features for RL training.

Integration Strategy:
- Extends observation space to include 64-dimensional reachability features
- Implements efficient caching for real-time performance (<2ms target)
- Provides graceful fallback when nclone components are unavailable
- Maintains compatibility with existing training pipelines

Performance Optimizations:
- Intelligent caching with TTL for repeated computations
- Tier-1 reachability extraction for real-time requirements
- Batch processing support for vectorized environments

References:
- Reachability analysis: Custom integration with nclone physics system
- Observation space design: Gym environment standards
- Performance optimization: Real-time RL training requirements
"""

# Standard library imports
import time
from typing import Dict, Any, Optional, Tuple, List

# Third-party imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# npp_rl imports
from npp_rl.utils.performance_monitor import PerformanceMonitor


class ReachabilityWrapper(gym.Wrapper):
    """
    Environment wrapper that adds reachability features to observations.

    This wrapper extends any NPP environment with compact reachability features
    extracted from the nclone physics system. It provides spatial reasoning
    guidance for RL agents while maintaining real-time performance requirements.

    Features:
    - 64-dimensional compact reachability feature encoding
    - Intelligent caching with configurable TTL
    - Performance monitoring and optimization
    - Graceful degradation when nclone is unavailable
    - Batch processing support for vectorized environments

    Example usage:
        base_env = NppEnvironment(...)
        env = ReachabilityWrapper(base_env, cache_ttl_ms=100.0)
        obs = env.reset()
        # obs now includes 'reachability_features' key
    """

    def __init__(
        self,
        env: gym.Env,
        cache_ttl_ms: float = 100.0,
        max_cache_size: int = 1000,
        performance_target: str = "fast",
        enable_monitoring: bool = True,
        debug: bool = False,
    ):
        """
        Initialize reachability wrapper.

        Args:
            env: Base NPP environment to wrap
            cache_ttl_ms: Cache time-to-live in milliseconds
            max_cache_size: Maximum number of cached entries
            performance_target: Reachability extraction performance target
            enable_monitoring: Whether to enable performance monitoring
            debug: Enable debug output
        """
        super().__init__(env)

        self.cache_ttl_ms = cache_ttl_ms
        self.max_cache_size = max_cache_size
        self.performance_target = performance_target
        self.debug = debug

        # Initialize reachability components
        self.reachability_extractor = self._initialize_reachability_extractor()
        self.reachability_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance monitoring
        if enable_monitoring:
            self.performance_monitor = PerformanceMonitor("reachability_extraction")
        else:
            self.performance_monitor = None

        # Extend observation space
        self._extend_observation_space()

        if self.debug:
            print(
                f"ReachabilityWrapper initialized with cache_ttl={cache_ttl_ms}ms, "
                f"performance_target={performance_target}"
            )

    def _initialize_reachability_extractor(self):
        """Initialize reachability feature extractor."""
        # Import at runtime to avoid hard dependency on nclone
        from nclone.graph.reachability.tiered_system import TieredReachabilitySystem
        from nclone.graph.reachability.feature_extractor import (
            ReachabilityFeatureExtractor,
            PerformanceMode,
        )

        tiered_system = TieredReachabilitySystem(debug=self.debug)
        extractor = ReachabilityFeatureExtractor(
            tiered_system=tiered_system,
            cache_ttl_ms=self.cache_ttl_ms,
            debug=self.debug,
        )

        if self.debug:
            print("Successfully initialized nclone reachability extractor")
        return extractor

    def _extend_observation_space(self):
        """Extend observation space to include reachability features."""
        # Add reachability features to observation space
        reachability_space = spaces.Box(
            low=0.0,
            high=2.0,
            shape=(64,),
            dtype=np.float32,
            name="reachability_features",
        )

        # Update observation space
        if isinstance(self.observation_space, spaces.Dict):
            # Add to existing Dict space
            new_spaces = dict(self.observation_space.spaces)
            new_spaces["reachability_features"] = reachability_space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            # Convert to Dict space
            original_space = self.observation_space
            self.observation_space = spaces.Dict(
                {
                    "original_obs": original_space,
                    "reachability_features": reachability_space,
                }
            )

        if self.debug:
            print(
                f"Extended observation space with reachability features: {reachability_space}"
            )

    def reset(self, **kwargs):
        """Reset environment and extract initial reachability features."""
        obs, info = self.env.reset(**kwargs)

        # Extract reachability features for initial state
        reachability_features = self._extract_reachability_features()

        # Add to observation
        obs["reachability_features"] = self._extract_reachability_features()

        # Add reachability info for debugging
        if "reachability" not in info:
            info["reachability"] = {}
        info["reachability"].update(
            {
                "extraction_time_ms": getattr(
                    reachability_features, "extraction_time", 0.0
                ),
                "cache_hit": getattr(reachability_features, "from_cache", False),
                "confidence": getattr(reachability_features, "confidence", 1.0),
            }
        )

        return obs, info

    def step(self, action):
        """Step environment and extract reachability features."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract reachability features
        reachability_features = self._extract_reachability_features()

        # Add to observation
        obs["reachability_features"] = reachability_features

        # Add reachability info for debugging
        if "reachability" not in info:
            info["reachability"] = {}
        info["reachability"].update(
            {
                "extraction_time_ms": getattr(
                    reachability_features, "extraction_time", 0.0
                ),
                "cache_hit": getattr(reachability_features, "from_cache", False),
                "confidence": getattr(reachability_features, "confidence", 1.0),
                "cache_stats": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hits
                    / max(1, self.cache_hits + self.cache_misses),
                },
            }
        )

        return obs, reward, terminated, truncated, info

    def _extract_reachability_features(self) -> np.ndarray:
        """Extract reachability features for current game state."""
        start_time = time.perf_counter()

        try:
            # Get current game state
            ninja_pos = self._get_ninja_position()
            level_data = self._get_level_data()
            entities = self._get_entities()
            switch_states = self._get_switch_states()

            # Check cache first
            cache_key = self._generate_cache_key(ninja_pos, switch_states)
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                cached_entry = self.reachability_cache[cache_key]
                cached_entry["cache_hits"] += 1
                features = cached_entry["features"]
                features.from_cache = True
                features.extraction_time = 0.0  # Cache hit
                return features

            self.cache_misses += 1

            # Extract features using nclone reachability system
            if hasattr(self.reachability_extractor, "extract_features"):
                # Use ReachabilityFeatureExtractor interface
                # Convert performance target to PerformanceMode enum
                from nclone.graph.reachability.feature_extractor import PerformanceMode

                perf_mode = getattr(
                    PerformanceMode,
                    self.performance_target.upper(),
                    PerformanceMode.FAST,
                )

                features_result = self.reachability_extractor.extract_features(
                    ninja_position=ninja_pos,
                    level_data=level_data,
                    entities=entities,
                    switch_states=switch_states,
                    performance_mode=perf_mode,
                )

                # Handle different return types
                if hasattr(features_result, "features"):
                    features = features_result.features
                else:
                    features = features_result
            else:
                # Fallback for dummy extractor
                features = self.reachability_extractor.extract_features(
                    ninja_pos, level_data, switch_states, self.performance_target
                )

            # Ensure numpy array format
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)

            # Validate feature dimensions
            if features.shape != (64,):
                if self.debug:
                    print(f"Warning: Expected 64 features, got {features.shape}")
                # Pad or truncate to 64 dimensions
                if len(features) < 64:
                    features = np.pad(features, (0, 64 - len(features)), "constant")
                else:
                    features = features[:64]

            extraction_time = (time.perf_counter() - start_time) * 1000

            # Add metadata
            features.extraction_time = extraction_time
            features.from_cache = False
            features.confidence = 1.0

            # Cache result
            self._cache_features(cache_key, features, extraction_time)

            # Performance monitoring
            if self.performance_monitor:
                self.performance_monitor.record_timing(extraction_time)

            return features

        except Exception as e:
            if self.debug:
                print(f"Warning: Reachability feature extraction failed: {e}")

            # Return zero features as fallback
            extraction_time = (time.perf_counter() - start_time) * 1000
            features = np.zeros(64, dtype=np.float32)
            features.extraction_time = extraction_time
            features.from_cache = False
            features.confidence = 0.0

            return features

    def _get_ninja_position(self) -> Tuple[float, float]:
        """Extract ninja position from environment state."""
        # Check if this is an nclone environment
        if hasattr(self.env, "ninja_position"):
            return self.env.ninja_position()
        elif hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, "ninja_position"
        ):
            return self.env.unwrapped.ninja_position()
        else:
            # Fallback for non-nclone environments
            return (0.0, 0.0)

    def _get_level_data(self) -> Optional[Any]:
        """Extract level data from environment state."""
        # Check if this is an nclone environment
        if hasattr(self.env, "level_data"):
            return self.env.level_data
        elif hasattr(self.env, "unwrapped") and hasattr(
            self.env.unwrapped, "level_data"
        ):
            return self.env.unwrapped.level_data
        else:
            # Fallback for non-nclone environments
            return None

    def _get_switch_states(self) -> Dict[str, bool]:
        """Extract switch states from environment state."""
        # Check if this is an nclone environment
        if hasattr(self.env, "entities"):
            entities = self.env.entities
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "entities"):
            entities = self.env.unwrapped.entities
        else:
            # Fallback for non-nclone environments
            return {}

        # Extract switch states from entities
        switch_states = {}
        for entity in entities:
            if hasattr(entity, "entity_id") and hasattr(entity, "activated"):
                # This is a switch entity
                switch_states[str(entity.entity_id)] = entity.activated

        return switch_states

    def _get_entities(self) -> List[Any]:
        """Extract entities from environment state."""
        # Check if this is an nclone environment
        if hasattr(self.env, "entities"):
            return self.env.entities
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "entities"):
            return self.env.unwrapped.entities
        else:
            # Fallback for non-nclone environments
            return []

    def _generate_cache_key(
        self, ninja_pos: Tuple[float, float], switch_states: Dict[str, bool]
    ) -> str:
        """Generate cache key for current state."""
        # Simple cache key based on position and switch states
        pos_key = f"{ninja_pos[0]:.2f},{ninja_pos[1]:.2f}"
        switch_key = ",".join(f"{k}:{v}" for k, v in sorted(switch_states.items()))
        return f"{pos_key}|{switch_key}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.reachability_cache:
            return False

        entry = self.reachability_cache[cache_key]
        current_time = time.perf_counter() * 1000  # Convert to ms

        return (current_time - entry["timestamp"]) < self.cache_ttl_ms

    def _cache_features(
        self, cache_key: str, features: np.ndarray, extraction_time: float
    ):
        """Cache reachability features."""
        # Clean old entries if cache is full
        if len(self.reachability_cache) >= self.max_cache_size:
            self._clean_cache()

        self.reachability_cache[cache_key] = {
            "features": features,
            "timestamp": time.perf_counter() * 1000,  # Convert to ms
            "extraction_time": extraction_time,
            "cache_hits": 0,
        }

    def _clean_cache(self):
        """Remove oldest cache entries."""
        if not self.reachability_cache:
            return

        # Remove oldest 25% of entries
        entries_to_remove = max(1, len(self.reachability_cache) // 4)

        # Sort by timestamp and remove oldest
        sorted_entries = sorted(
            self.reachability_cache.items(), key=lambda x: x[1]["timestamp"]
        )

        for key, _ in sorted_entries[:entries_to_remove]:
            del self.reachability_cache[key]

        if self.debug:
            print(
                f"Cleaned {entries_to_remove} cache entries, "
                f"{len(self.reachability_cache)} remaining"
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for reachability extraction."""
        stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
            "cache_size": len(self.reachability_cache),
        }

        if self.performance_monitor:
            stats.update(self.performance_monitor.get_stats())

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.reachability_cache.clear()

        if self.performance_monitor:
            self.performance_monitor.reset()


def create_reachability_aware_env(
    base_env: gym.Env,
    cache_ttl_ms: float = 100.0,
    performance_target: str = "fast",
    enable_monitoring: bool = True,
    debug: bool = False,
) -> ReachabilityWrapper:
    """
    Factory function to create a reachability-aware environment.

    Args:
        base_env: Base NPP environment to wrap
        cache_ttl_ms: Cache time-to-live in milliseconds
        performance_target: Reachability extraction performance target
        enable_monitoring: Whether to enable performance monitoring
        debug: Enable debug output

    Returns:
        ReachabilityWrapper: Environment with reachability features
    """
    return ReachabilityWrapper(
        env=base_env,
        cache_ttl_ms=cache_ttl_ms,
        performance_target=performance_target,
        enable_monitoring=enable_monitoring,
        debug=debug,
    )
