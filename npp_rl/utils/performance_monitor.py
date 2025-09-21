"""
Performance monitoring utilities for RL training components.

This module provides lightweight performance monitoring tools for tracking
timing, memory usage, and other metrics during RL training. It's designed
to have minimal overhead while providing useful insights for optimization.
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from collections import deque


class PerformanceMonitor:
    """
    Lightweight performance monitor for tracking timing and metrics.
    
    This class provides efficient tracking of performance metrics with
    configurable history size and statistical analysis capabilities.
    """
    
    def __init__(self, name: str, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            name: Name of the monitored component
            max_history: Maximum number of timing records to keep
        """
        self.name = name
        self.max_history = max_history
        
        # Timing data
        self.timings = deque(maxlen=max_history)
        self.total_calls = 0
        self.total_time = 0.0
        
        # Statistics cache
        self._stats_cache = {}
        self._cache_valid = False
    
    def record_timing(self, duration_ms: float):
        """Record a timing measurement."""
        self.timings.append(duration_ms)
        self.total_calls += 1
        self.total_time += duration_ms
        self._cache_valid = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._cache_valid:
            self._update_stats_cache()
        return self._stats_cache.copy()
    
    def _update_stats_cache(self):
        """Update cached statistics."""
        if not self.timings:
            self._stats_cache = {
                'name': self.name,
                'total_calls': 0,
                'avg_time_ms': 0.0,
                'min_time_ms': 0.0,
                'max_time_ms': 0.0,
                'p50_time_ms': 0.0,
                'p95_time_ms': 0.0,
                'p99_time_ms': 0.0,
                'total_time_ms': 0.0
            }
        else:
            timings_list = list(self.timings)
            
            self._stats_cache = {
                'name': self.name,
                'total_calls': self.total_calls,
                'avg_time_ms': statistics.mean(timings_list),
                'min_time_ms': min(timings_list),
                'max_time_ms': max(timings_list),
                'p50_time_ms': statistics.median(timings_list),
                'p95_time_ms': self._percentile(timings_list, 0.95),
                'p99_time_ms': self._percentile(timings_list, 0.99),
                'total_time_ms': self.total_time
            }
        
        self._cache_valid = True
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]
    
    def reset(self):
        """Reset all statistics."""
        self.timings.clear()
        self.total_calls = 0
        self.total_time = 0.0
        self._cache_valid = False
    
    def __str__(self) -> str:
        """String representation of performance stats."""
        stats = self.get_stats()
        return (f"PerformanceMonitor({stats['name']}): "
                f"calls={stats['total_calls']}, "
                f"avg={stats['avg_time_ms']:.2f}ms, "
                f"p95={stats['p95_time_ms']:.2f}ms")


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, monitor: PerformanceMonitor):
        """Initialize timing context."""
        self.monitor = monitor
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record result."""
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.monitor.record_timing(duration_ms)


def time_function(monitor: PerformanceMonitor):
    """Decorator for timing function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(monitor):
                return func(*args, **kwargs)
        return wrapper
    return decorator