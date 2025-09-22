"""
Integration utilities for dynamic graph wrapper.

This module provides utilities to integrate the dynamic graph wrapper
with existing environments and training pipelines.
"""

import logging
from typing import Dict, Any, Optional, Callable
import gymnasium as gym
from gymnasium.spaces import Dict as SpacesDict

from .dynamic_graph_wrapper import DynamicGraphWrapper, UpdateBudget
from .vectorization_wrapper import VectorizationWrapper


def create_dynamic_graph_env(
    env_kwargs: Optional[Dict[str, Any]] = None,
    enable_dynamic_updates: bool = True,
    update_budget: Optional[UpdateBudget] = None,
    performance_mode: str = "balanced",  # "fast", "balanced", "accurate"
) -> gym.Env:
    """
    Create an environment with dynamic graph capabilities.

    Args:
        env_kwargs: Environment configuration parameters
        enable_dynamic_updates: Whether to enable dynamic graph updates
        update_budget: Custom update budget, or None for defaults
        performance_mode: Performance/accuracy tradeoff mode

    Returns:
        Environment with dynamic graph wrapper applied
    """
    # Set default environment kwargs
    if env_kwargs is None:
        env_kwargs = {
            "render_mode": "rgb_array",
            "enable_pbrs": True,
            "pbrs_weights": {
                "objective_weight": 1.0,
                "hazard_weight": 0.5,
                "impact_weight": 0.3,
                "exploration_weight": 0.2,
            },
            "pbrs_gamma": 0.99,
        }

    # Create base environment
    base_env = VectorizationWrapper(env_kwargs)

    # Configure update budget based on performance mode
    if update_budget is None:
        if performance_mode == "fast":
            update_budget = UpdateBudget(
                max_time_ms=15.0,
                max_edge_updates=500,
                max_node_updates=250,
                priority_threshold=0.7,
            )
        elif performance_mode == "balanced":
            update_budget = UpdateBudget(
                max_time_ms=25.0,
                max_edge_updates=1000,
                max_node_updates=500,
                priority_threshold=0.5,
            )
        elif performance_mode == "accurate":
            update_budget = UpdateBudget(
                max_time_ms=40.0,
                max_edge_updates=2000,
                max_node_updates=1000,
                priority_threshold=0.3,
            )
        else:
            raise ValueError(f"Unknown performance mode: {performance_mode}")

    # Apply dynamic graph wrapper
    dynamic_env = DynamicGraphWrapper(
        env=base_env,
        enable_dynamic_updates=enable_dynamic_updates,
        update_budget=update_budget,
        event_buffer_size=100,
        temporal_window_size=10.0,
    )

    return dynamic_env


def add_dynamic_graph_monitoring(
    env: DynamicGraphWrapper,
    log_interval: int = 100,
    performance_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> DynamicGraphWrapper:
    """
    Add monitoring capabilities to a dynamic graph environment.

    Args:
        env: Dynamic graph environment to monitor
        log_interval: How often to log performance stats (in steps)
        performance_callback: Optional callback for performance data

    Returns:
        Environment with monitoring wrapper applied
    """

    class DynamicGraphMonitor(gym.Wrapper):
        """Monitor dynamic graph performance."""

        def __init__(self, env: DynamicGraphWrapper):
            super().__init__(env)
            self.step_count = 0
            self.log_interval = log_interval
            self.performance_callback = performance_callback

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.step_count += 1

            # Log performance stats periodically
            if self.step_count % self.log_interval == 0:
                stats = self.env.get_performance_stats()

                logging.info(
                    f"Dynamic Graph Stats (step {self.step_count}): "
                    f"avg_update_time={stats['avg_update_time_ms']:.2f}ms, "
                    f"events_processed={stats['events_processed']}, "
                    f"budget_exceeded={stats['budget_exceeded_count']}"
                )

                # Call performance callback if provided
                if self.performance_callback:
                    self.performance_callback(stats)

            return obs, reward, terminated, truncated, info

    return DynamicGraphMonitor(env)


def validate_dynamic_graph_environment(env: gym.Env) -> bool:
    """
    Validate that an environment is properly configured for dynamic graphs.

    Args:
        env: Environment to validate

    Returns:
        True if environment is valid, False otherwise
    """
    # Check if environment has dynamic graph wrapper
    if not isinstance(env, DynamicGraphWrapper):
        current_env = env
        while hasattr(current_env, "env"):
            current_env = current_env.env
            if isinstance(current_env, DynamicGraphWrapper):
                break
        else:
            logging.error("Environment does not have DynamicGraphWrapper")
            return False

    # Check observation space
    if not hasattr(env, "observation_space"):
        logging.error("Environment missing observation_space")
        return False

    # Test environment reset and step
    obs, info = env.reset()
    if "dynamic_graph_metadata" in obs:
        metadata = obs["dynamic_graph_metadata"]
        if len(metadata) != 10:
            logging.error(f"Invalid dynamic graph metadata shape: {metadata.shape}")
            return False

    # Test a single step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    logging.info("Dynamic graph environment validation passed")
    return True


def benchmark_dynamic_graph_performance(
    env: DynamicGraphWrapper, num_steps: int = 1000, target_fps: float = 60.0
) -> Dict[str, Any]:
    """
    Benchmark dynamic graph performance.

    Args:
        env: Dynamic graph environment to benchmark
        num_steps: Number of steps to run
        target_fps: Target FPS for performance evaluation

    Returns:
        Performance benchmark results
    """
    import time

    # Reset environment
    env.reset()

    # Warm up
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)

    # Benchmark
    start_time = time.time()
    update_times = []

    for step in range(num_steps):
        step_start = time.time()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        step_time = time.time() - step_start
        update_times.append(step_time * 1000)  # Convert to ms

        if terminated or truncated:
            env.reset()

    total_time = time.time() - start_time

    # Calculate statistics
    avg_step_time_ms = sum(update_times) / len(update_times)
    max_step_time_ms = max(update_times)
    min_step_time_ms = min(update_times)

    target_step_time_ms = 1000.0 / target_fps
    performance_ratio = target_step_time_ms / avg_step_time_ms

    # Get dynamic graph stats
    graph_stats = env.get_performance_stats()

    results = {
        "total_steps": num_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": avg_step_time_ms,
        "max_step_time_ms": max_step_time_ms,
        "min_step_time_ms": min_step_time_ms,
        "target_step_time_ms": target_step_time_ms,
        "performance_ratio": performance_ratio,
        "meets_target_fps": performance_ratio >= 1.0,
        "graph_stats": graph_stats,
    }

    logging.info(
        f"Dynamic Graph Benchmark Results:\n"
        f"  Average step time: {avg_step_time_ms:.2f}ms\n"
        f"  Target step time: {target_step_time_ms:.2f}ms\n"
        f"  Performance ratio: {performance_ratio:.2f}\n"
        f"  Meets target FPS: {results['meets_target_fps']}\n"
        f"  Graph update time: {graph_stats['avg_update_time_ms']:.2f}ms"
    )

    return results


class DynamicGraphProfiler:
    """
    Profiler for dynamic graph operations.

    This class provides detailed profiling of dynamic graph update operations
    to help identify performance bottlenecks.
    """

    def __init__(self):
        self.profiles = {}
        self.current_profile = None

    def start_profile(self, name: str):
        """Start profiling a named operation."""
        import time

        self.current_profile = {
            "name": name,
            "start_time": time.time(),
            "operations": [],
        }

    def record_operation(
        self,
        operation_name: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a profiled operation."""
        if self.current_profile:
            self.current_profile["operations"].append(
                {
                    "name": operation_name,
                    "duration_ms": duration_ms,
                    "metadata": metadata or {},
                }
            )

    def end_profile(self):
        """End current profiling session."""
        if self.current_profile:
            import time

            self.current_profile["total_time"] = (
                time.time() - self.current_profile["start_time"]
            )
            self.profiles[self.current_profile["name"]] = self.current_profile
            self.current_profile = None

    def get_profile_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary of a named profile."""
        if name not in self.profiles:
            return None

        profile = self.profiles[name]
        operations = profile["operations"]

        if not operations:
            return profile

        # Calculate operation statistics
        total_op_time = sum(op["duration_ms"] for op in operations)
        avg_op_time = total_op_time / len(operations)
        max_op_time = max(op["duration_ms"] for op in operations)

        # Group by operation type
        op_groups = {}
        for op in operations:
            op_name = op["name"]
            if op_name not in op_groups:
                op_groups[op_name] = []
            op_groups[op_name].append(op["duration_ms"])

        op_summaries = {}
        for op_name, times in op_groups.items():
            op_summaries[op_name] = {
                "count": len(times),
                "total_time_ms": sum(times),
                "avg_time_ms": sum(times) / len(times),
                "max_time_ms": max(times),
            }

        return {
            **profile,
            "total_operation_time_ms": total_op_time,
            "avg_operation_time_ms": avg_op_time,
            "max_operation_time_ms": max_op_time,
            "operation_summaries": op_summaries,
        }
