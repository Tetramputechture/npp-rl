"""Performance benchmarks for curriculum environment wrappers.

Compares original vs optimized curriculum environment wrappers to measure
the performance improvements in the hot paths (reset/step methods).
"""

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

import numpy as np
import gymnasium as gym

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.training.curriculum_manager import CurriculumManager
from npp_rl.training.optimized_curriculum_manager import OptimizedCurriculumManager
from npp_rl.wrappers.curriculum_env import CurriculumEnv, CurriculumVecEnvWrapper
from npp_rl.wrappers.optimized_curriculum_env import (
    OptimizedCurriculumEnv,
    OptimizedCurriculumVecEnvWrapper,
)

# Import CurriculumBenchmark functionality inline to avoid module issues
import pickle

logger = logging.getLogger(__name__)


class MockEnvironment(gym.Env):
    """Mock environment for benchmarking wrappers without N++ overhead."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "tiles": gym.spaces.Box(
                    low=0, high=255, shape=(20, 20), dtype=np.uint8
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(6,), dtype=np.bool_
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(6)

        # Mock nplay_headless for compatibility (access via unwrapped)
        self._mock_nplay = Mock()
        self._mock_nplay.sim = Mock()
        self._mock_nplay.sim.sim_config = Mock()
        self._mock_nplay.sim.sim_config.debug = False
        self._mock_nplay.load_map_from_map_data = Mock()

        self._episode_count = 0
        self._step_count = 0

    @property
    def unwrapped(self):
        """Return self as unwrapped with mock nplay_headless."""
        # Create a simple object to hold the mock
        unwrapped = type("MockUnwrapped", (), {})()
        unwrapped.nplay_headless = self._mock_nplay
        return unwrapped

    def reset(self, **kwargs):
        """Mock reset method."""
        obs = {
            "tiles": np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            "action_mask": np.ones(6, dtype=np.bool_),
        }
        info = {"l": 0, "player_won": False}
        self._step_count = 0
        return obs, info

    def step(self, action):
        """Mock step method."""
        self._step_count += 1

        obs = {
            "tiles": np.random.randint(0, 255, (20, 20), dtype=np.uint8),
            "action_mask": np.ones(6, dtype=np.bool_),
        }

        # Simulate episode completion after 100 steps
        terminated = self._step_count >= 100
        truncated = False
        reward = 1.0 if terminated else 0.0

        info = {
            "l": self._step_count,
            "player_won": terminated and np.random.random() < 0.3,  # 30% win rate
            "has_won": terminated and np.random.random() < 0.3,
        }

        if terminated:
            self._episode_count += 1

        return obs, reward, terminated, truncated, info


class MockVectorizedEnv:
    """Mock vectorized environment for benchmarking VecEnv wrappers."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.envs = [MockEnvironment() for _ in range(num_envs)]
        self._step_counts = np.zeros(num_envs, dtype=np.int32)

        # Required attributes for VecEnv compatibility
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """Mock vectorized reset."""
        obs_list = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)

        # Stack observations
        stacked_obs = {
            "tiles": np.stack([obs["tiles"] for obs in obs_list]),
            "action_mask": np.stack([obs["action_mask"] for obs in obs_list]),
        }

        return stacked_obs, infos

    def step_async(self, actions):
        """Mock async step (not used in benchmark)."""
        self.pending_actions = actions

    def step_wait(self):
        """Mock step wait."""
        obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, self.pending_actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

        # Stack results
        stacked_obs = {
            "tiles": np.stack([obs["tiles"] for obs in obs_list]),
            "action_mask": np.stack([obs["action_mask"] for obs in obs_list]),
        }

        return stacked_obs, np.array(rewards), np.array(dones), infos

    def step(self, actions):
        """Mock synchronous step."""
        self.step_async(actions)
        return self.step_wait()

    def env_method(self, method_name, *args, **kwargs):
        """Mock env_method for subprocess communication simulation."""
        results = []
        for env in self.envs:
            if hasattr(env, method_name):
                method = getattr(env, method_name)
                result = method(*args, **kwargs)
                results.append(result)
            else:
                results.append(None)
        return results


class CurriculumEnvBenchmark:
    """Benchmark curriculum environment wrappers performance."""

    def __init__(self):
        pass

    def create_mock_dataset(self, tmp_dir: Path, levels_per_stage: int = 50) -> str:
        """Create a mock dataset for consistent benchmarking.

        Args:
            tmp_dir: Temporary directory for mock data
            levels_per_stage: Number of levels per curriculum stage

        Returns:
            Path to mock dataset directory
        """
        mock_dataset_dir = tmp_dir / "mock_curriculum_dataset"

        stages = [
            "simplest",
            "simplest_few_mines",
            "simplest_with_mines",
            "simpler",
            "simple",
            "medium",
            "complex",
            "exploration",
            "mine_heavy",
        ]

        generators = ["manual", "procedural", "mixed"]

        for stage in stages:
            stage_dir = mock_dataset_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            for i in range(levels_per_stage):
                level_data = {
                    "level_id": f"{stage}_{i:04d}",
                    "map_data": {"tiles": np.random.randint(0, 10, (20, 20)).tolist()},
                    "metadata": {
                        "generator": generators[i % len(generators)],
                        "category": stage,
                        "difficulty": np.random.uniform(0.1, 1.0),
                    },
                }

                level_file = stage_dir / f"level_{i:04d}.pkl"
                with open(level_file, "wb") as f:
                    pickle.dump(level_data, f)

        return str(mock_dataset_dir)

    def benchmark_single_env_wrappers(self, episodes: int = 1000) -> Dict[str, Any]:
        """Benchmark single environment wrappers.

        Args:
            episodes: Number of episodes to run

        Returns:
            Benchmark results comparing original vs optimized wrappers
        """
        print("=" * 70)
        print("CURRICULUM ENVIRONMENT WRAPPER BENCHMARKS")
        print("=" * 70)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir), levels_per_stage=50)

            print(
                f"\nBenchmarking single environment wrappers ({episodes} episodes)..."
            )

            # Test original wrapper
            print("\n1. ORIGINAL CURRICULUM ENV")
            print("-" * 30)
            original_results = self._benchmark_single_wrapper(
                CurriculumEnv, CurriculumManager, mock_dataset, episodes, "Original"
            )

            # Test optimized wrapper with original manager
            print("\n2. OPTIMIZED WRAPPER + ORIGINAL MANAGER")
            print("-" * 40)
            opt_wrapper_results = self._benchmark_single_wrapper(
                OptimizedCurriculumEnv,
                CurriculumManager,
                mock_dataset,
                episodes,
                "OptWrapper+OrigMgr",
            )

            # Test optimized wrapper with optimized manager
            print("\n3. OPTIMIZED WRAPPER + OPTIMIZED MANAGER")
            print("-" * 42)
            fully_opt_results = self._benchmark_single_wrapper(
                OptimizedCurriculumEnv,
                OptimizedCurriculumManager,
                mock_dataset,
                episodes,
                "FullyOptimized",
            )

            # Calculate improvements
            print("\n4. PERFORMANCE IMPROVEMENTS")
            print("-" * 30)
            improvements = self._calculate_wrapper_improvements(
                original_results, opt_wrapper_results, fully_opt_results
            )

            return {
                "original": original_results,
                "optimized_wrapper": opt_wrapper_results,
                "fully_optimized": fully_opt_results,
                "improvements": improvements,
            }

    def benchmark_vec_env_wrappers(
        self, episodes: int = 1000, num_envs: int = 4
    ) -> Dict[str, Any]:
        """Benchmark vectorized environment wrappers.

        Args:
            episodes: Number of total episodes to run across all environments
            num_envs: Number of parallel environments

        Returns:
            Benchmark results comparing original vs optimized VecEnv wrappers
        """
        print(
            f"\nBenchmarking vectorized environment wrappers ({episodes} episodes, {num_envs} envs)..."
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir), levels_per_stage=50)

            # Test original VecEnv wrapper
            print("\n5. ORIGINAL CURRICULUM VEC ENV")
            print("-" * 35)
            original_vec_results = self._benchmark_vec_wrapper(
                CurriculumVecEnvWrapper,
                CurriculumManager,
                mock_dataset,
                episodes,
                num_envs,
                "OriginalVec",
            )

            # Test optimized VecEnv wrapper
            print("\n6. OPTIMIZED CURRICULUM VEC ENV")
            print("-" * 35)
            optimized_vec_results = self._benchmark_vec_wrapper(
                OptimizedCurriculumVecEnvWrapper,
                OptimizedCurriculumManager,
                mock_dataset,
                episodes,
                num_envs,
                "OptimizedVec",
            )

            # Calculate VecEnv improvements
            print("\n7. VEC ENV IMPROVEMENTS")
            print("-" * 25)
            vec_improvements = self._calculate_vec_improvements(
                original_vec_results, optimized_vec_results
            )

            return {
                "original_vec": original_vec_results,
                "optimized_vec": optimized_vec_results,
                "vec_improvements": vec_improvements,
            }

    def _benchmark_single_wrapper(
        self, wrapper_class, manager_class, dataset_path: str, episodes: int, name: str
    ) -> Dict[str, Any]:
        """Benchmark a single environment wrapper implementation."""

        # Create curriculum manager
        manager = manager_class(
            dataset_path=dataset_path,
            starting_stage="simpler",
            performance_window=50,
        )

        # Add some performance data for realistic mixing
        for _ in range(50):
            manager.record_episode("simplest", np.random.random() < 0.8)
            manager.record_episode("simpler", np.random.random() < 0.6)

        # Create wrapped environment
        base_env = MockEnvironment()
        wrapped_env = wrapper_class(base_env, manager, check_advancement_freq=10)

        # Benchmark reset performance
        reset_times = []
        for i in range(100):
            start = time.perf_counter()
            wrapped_env.reset()
            end = time.perf_counter()
            reset_times.append(end - start)

        # Benchmark step performance
        step_times = []
        wrapped_env.reset()
        for i in range(100):
            start = time.perf_counter()
            wrapped_env.step(0)  # Always action 0 for consistency
            end = time.perf_counter()
            step_times.append(end - start)

        # Benchmark full episode performance
        episode_start = time.perf_counter()
        episodes_completed = 0
        total_steps = 0

        while episodes_completed < episodes:
            wrapped_env.reset()
            done = False

            while not done:
                action = np.random.randint(0, 6)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                done = terminated or truncated
                total_steps += 1

            episodes_completed += 1

        episode_end = time.perf_counter()
        total_episode_time = episode_end - episode_start

        results = {
            "reset_mean_ms": np.mean(reset_times) * 1000,
            "reset_std_ms": np.std(reset_times) * 1000,
            "step_mean_ms": np.mean(step_times) * 1000,
            "step_std_ms": np.std(step_times) * 1000,
            "episodes_per_second": episodes / total_episode_time,
            "steps_per_second": total_steps / total_episode_time,
            "total_time": total_episode_time,
        }

        print(f"  {name}:")
        print(
            f"    Reset time: {results['reset_mean_ms']:.3f}ms ± {results['reset_std_ms']:.3f}ms"
        )
        print(
            f"    Step time: {results['step_mean_ms']:.3f}ms ± {results['step_std_ms']:.3f}ms"
        )
        print(f"    Episodes/sec: {results['episodes_per_second']:.1f}")
        print(f"    Steps/sec: {results['steps_per_second']:.1f}")

        return results

    def _benchmark_vec_wrapper(
        self,
        wrapper_class,
        manager_class,
        dataset_path: str,
        episodes: int,
        num_envs: int,
        name: str,
    ) -> Dict[str, Any]:
        """Benchmark a vectorized environment wrapper implementation."""

        # Create curriculum manager
        manager = manager_class(
            dataset_path=dataset_path,
            starting_stage="simpler",
            performance_window=50,
        )

        # Add performance data
        for _ in range(50):
            manager.record_episode("simplest", np.random.random() < 0.8)
            manager.record_episode("simpler", np.random.random() < 0.6)

        # Create wrapped vectorized environment
        base_venv = MockVectorizedEnv(num_envs=num_envs)
        wrapped_venv = wrapper_class(base_venv, manager, check_advancement_freq=10)

        # Benchmark vectorized performance
        start_time = time.perf_counter()
        episodes_completed = 0
        total_steps = 0

        # Reset all environments
        wrapped_venv.reset()

        while episodes_completed < episodes:
            actions = np.random.randint(0, 6, size=num_envs)
            obs, rewards, dones, infos = wrapped_venv.step(actions)

            # Count completed episodes
            for i, done in enumerate(dones):
                if done:
                    episodes_completed += 1

            total_steps += num_envs

        end_time = time.perf_counter()
        total_time = end_time - start_time

        results = {
            "episodes_per_second": episodes / total_time,
            "steps_per_second": total_steps / total_time,
            "total_time": total_time,
            "num_envs": num_envs,
        }

        print(f"  {name}:")
        print(f"    Episodes/sec: {results['episodes_per_second']:.1f}")
        print(f"    Steps/sec: {results['steps_per_second']:.1f}")
        print(
            f"    Throughput per env: {results['steps_per_second'] / num_envs:.1f} steps/sec/env"
        )

        return results

    def _calculate_wrapper_improvements(
        self,
        original: Dict[str, Any],
        opt_wrapper: Dict[str, Any],
        fully_opt: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate performance improvements for single environment wrappers."""

        improvements = {}

        # Wrapper-only improvements (original manager)
        wrapper_reset_speedup = original["reset_mean_ms"] / opt_wrapper["reset_mean_ms"]
        wrapper_step_speedup = original["step_mean_ms"] / opt_wrapper["step_mean_ms"]
        wrapper_episode_speedup = (
            opt_wrapper["episodes_per_second"] / original["episodes_per_second"]
        )

        improvements["wrapper_only"] = {
            "reset_speedup": wrapper_reset_speedup,
            "step_speedup": wrapper_step_speedup,
            "episode_speedup": wrapper_episode_speedup,
        }

        # Full optimization improvements
        full_reset_speedup = original["reset_mean_ms"] / fully_opt["reset_mean_ms"]
        full_step_speedup = original["step_mean_ms"] / fully_opt["step_mean_ms"]
        full_episode_speedup = (
            fully_opt["episodes_per_second"] / original["episodes_per_second"]
        )

        improvements["full_optimization"] = {
            "reset_speedup": full_reset_speedup,
            "step_speedup": full_step_speedup,
            "episode_speedup": full_episode_speedup,
        }

        # Print results
        print("Wrapper-only improvements:")
        print(f"  Reset speedup: {wrapper_reset_speedup:.2f}x")
        print(f"  Step speedup: {wrapper_step_speedup:.2f}x")
        print(f"  Episode speedup: {wrapper_episode_speedup:.2f}x")

        print("Full optimization improvements:")
        print(f"  Reset speedup: {full_reset_speedup:.2f}x")
        print(f"  Step speedup: {full_step_speedup:.2f}x")
        print(f"  Episode speedup: {full_episode_speedup:.2f}x")

        # Overall assessment
        avg_speedup = (
            full_reset_speedup + full_step_speedup + full_episode_speedup
        ) / 3
        print(f"\n📊 OVERALL WRAPPER IMPROVEMENT: {avg_speedup:.2f}x speedup")

        if avg_speedup > 1.5:
            print("🎯 EXCELLENT IMPROVEMENT: >50% speedup achieved!")
        elif avg_speedup > 1.2:
            print("✅ GOOD IMPROVEMENT: >20% speedup achieved")
        else:
            print("⚠️  MODEST IMPROVEMENT: <20% speedup")

        improvements["overall_speedup"] = avg_speedup

        return improvements

    def _calculate_vec_improvements(
        self, original: Dict[str, Any], optimized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance improvements for vectorized environment wrappers."""

        episode_speedup = (
            optimized["episodes_per_second"] / original["episodes_per_second"]
        )
        step_speedup = optimized["steps_per_second"] / original["steps_per_second"]

        improvements = {
            "episode_speedup": episode_speedup,
            "step_speedup": step_speedup,
        }

        print(f"VecEnv episode speedup: {episode_speedup:.2f}x")
        print(f"VecEnv step speedup: {step_speedup:.2f}x")

        avg_speedup = (episode_speedup + step_speedup) / 2
        print(f"\n📊 OVERALL VEC ENV IMPROVEMENT: {avg_speedup:.2f}x speedup")

        improvements["overall_speedup"] = avg_speedup

        return improvements


def main():
    """Run curriculum environment wrapper benchmarks."""
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)

    benchmark = CurriculumEnvBenchmark()

    # Benchmark single environment wrappers
    single_results = benchmark.benchmark_single_env_wrappers(episodes=500)

    # Benchmark vectorized environment wrappers
    vec_results = benchmark.benchmark_vec_env_wrappers(episodes=500, num_envs=4)

    print("\n" + "=" * 70)
    print("CURRICULUM ENVIRONMENT WRAPPER BENCHMARKS COMPLETED!")
    print("=" * 70)

    return {
        "single_env": single_results,
        "vec_env": vec_results,
    }


if __name__ == "__main__":
    results = main()
