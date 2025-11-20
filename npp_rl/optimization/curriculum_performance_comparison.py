"""Performance comparison between original and optimized CurriculumManager.

This script provides a direct comparison of performance improvements
achieved by the optimized implementation.
"""

import gc
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.training.curriculum_manager import CurriculumManager
from npp_rl.training.optimized_curriculum_manager import OptimizedCurriculumManager
from npp_rl.optimization.curriculum_benchmarks import CurriculumBenchmark

logger = logging.getLogger(__name__)


class CurriculumPerformanceComparison:
    """Direct performance comparison between original and optimized implementations."""

    def __init__(self):
        self.benchmark_suite = CurriculumBenchmark("", "")

    def run_comparison(self, samples: int = 10000) -> Dict[str, Any]:
        """Run comprehensive performance comparison.

        Args:
            samples: Number of samples for throughput test

        Returns:
            Comparison results dictionary
        """
        print("=" * 70)
        print("CURRICULUM MANAGER PERFORMANCE COMPARISON")
        print("=" * 70)

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.benchmark_suite.create_mock_dataset(
                Path(tmp_dir), levels_per_stage=100
            )

            # Test original implementation
            print("\n1. ORIGINAL IMPLEMENTATION")
            print("-" * 35)
            original_results = self._benchmark_implementation(
                CurriculumManager, mock_dataset, samples, "Original"
            )

            # Test optimized implementation
            print("\n2. OPTIMIZED IMPLEMENTATION")
            print("-" * 35)
            optimized_results = self._benchmark_implementation(
                OptimizedCurriculumManager, mock_dataset, samples, "Optimized"
            )

            # Calculate improvements
            print("\n3. PERFORMANCE IMPROVEMENTS")
            print("-" * 35)
            improvements = self._calculate_improvements(
                original_results, optimized_results
            )

            return {
                "original": original_results,
                "optimized": optimized_results,
                "improvements": improvements,
            }

    def _benchmark_implementation(
        self, manager_class, dataset_path: str, samples: int, name: str
    ) -> Dict[str, Any]:
        """Benchmark a specific CurriculumManager implementation.

        Args:
            manager_class: CurriculumManager class to test
            dataset_path: Path to dataset
            samples: Number of samples for throughput test
            name: Name for logging

        Returns:
            Benchmark results dictionary
        """
        results = {}

        # 1. Initialization benchmark
        print(f"  {name} - Initialization...")
        init_times = []
        memory_deltas = []

        for i in range(5):
            gc.collect()
            initial_memory = self._get_memory_mb()

            start = time.perf_counter()
            manager = manager_class(
                dataset_path=dataset_path,
                starting_stage="simpler",
                performance_window=50,
                enable_adaptive_mixing=True,
            )
            end = time.perf_counter()

            final_memory = self._get_memory_mb()

            init_times.append(end - start)
            memory_deltas.append(final_memory - initial_memory)

            del manager
            gc.collect()

        results["initialization"] = {
            "mean_time": np.mean(init_times),
            "std_time": np.std(init_times),
            "mean_memory_mb": np.mean(memory_deltas),
        }

        print(
            f"    Init time: {results['initialization']['mean_time']:.4f}s ± {results['initialization']['std_time']:.4f}s"
        )
        print(f"    Memory: {results['initialization']['mean_memory_mb']:.1f}MB")

        # 2. Sampling throughput benchmark
        print(f"  {name} - Sampling throughput...")

        manager = manager_class(
            dataset_path=dataset_path,
            starting_stage="simpler",
            performance_window=50,
            enable_adaptive_mixing=True,
        )

        # Add performance data for mixing
        for _ in range(50):
            manager.record_episode("simplest", np.random.random() < 0.8)
            manager.record_episode("simpler", np.random.random() < 0.6)

        # Warmup
        for _ in range(100):
            manager.sample_level()

        # Benchmark sampling
        start = time.perf_counter()
        for i in range(samples):
            level = manager.sample_level()
            if level is None:
                raise RuntimeError(f"Got None level at sample {i}")
        end = time.perf_counter()

        sampling_time = end - start
        samples_per_second = samples / sampling_time

        results["sampling"] = {
            "total_time": sampling_time,
            "samples_per_second": samples_per_second,
            "mean_sample_time_ms": (sampling_time / samples) * 1000,
        }

        print(f"    Rate: {samples_per_second:.1f} samples/sec")
        print(
            f"    Mean time: {results['sampling']['mean_sample_time_ms']:.3f}ms per sample"
        )

        # 3. Memory usage during operations
        print(f"  {name} - Memory usage...")

        initial_memory = self._get_memory_mb()

        # Simulate training operations
        for i in range(500):
            level = manager.sample_level()
            stage = level.get("sampled_stage", "simpler")
            success = np.random.random() < 0.7
            manager.record_episode(stage, success)

            if i % 50 == 0:
                manager.check_advancement()

        final_memory = self._get_memory_mb()

        results["memory"] = {
            "operational_growth_mb": final_memory - initial_memory,
        }

        print(
            f"    Operational growth: {results['memory']['operational_growth_mb']:.2f}MB"
        )

        del manager
        gc.collect()

        return results

    def _calculate_improvements(
        self, original: Dict[str, Any], optimized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance improvements."""
        improvements = {}

        # Initialization improvements
        init_speedup = (
            original["initialization"]["mean_time"]
            / optimized["initialization"]["mean_time"]
        )
        init_memory_reduction = (
            (
                original["initialization"]["mean_memory_mb"]
                - optimized["initialization"]["mean_memory_mb"]
            )
            / original["initialization"]["mean_memory_mb"]
            * 100
        )

        improvements["initialization"] = {
            "speedup": init_speedup,
            "memory_reduction_percent": init_memory_reduction,
        }

        # Sampling improvements
        sampling_speedup = (
            optimized["sampling"]["samples_per_second"]
            / original["sampling"]["samples_per_second"]
        )
        time_reduction = (
            (
                original["sampling"]["mean_sample_time_ms"]
                - optimized["sampling"]["mean_sample_time_ms"]
            )
            / original["sampling"]["mean_sample_time_ms"]
            * 100
        )

        improvements["sampling"] = {
            "speedup": sampling_speedup,
            "time_reduction_percent": time_reduction,
        }

        # Memory improvements
        if original["memory"]["operational_growth_mb"] > 0:
            memory_improvement = (
                (
                    original["memory"]["operational_growth_mb"]
                    - optimized["memory"]["operational_growth_mb"]
                )
                / original["memory"]["operational_growth_mb"]
                * 100
            )
        else:
            memory_improvement = 0.0

        improvements["memory"] = {
            "operational_memory_reduction_percent": memory_improvement,
        }

        # Print improvements
        print(
            f"Initialization speedup: {init_speedup:.2f}x ({init_memory_reduction:+.1f}% memory)"
        )
        print(
            f"Sampling speedup: {sampling_speedup:.2f}x ({time_reduction:+.1f}% time reduction)"
        )
        print(
            f"Memory efficiency: {memory_improvement:+.1f}% operational growth reduction"
        )

        # Overall performance score
        overall_speedup = (init_speedup + sampling_speedup) / 2
        print(f"\n📊 OVERALL PERFORMANCE IMPROVEMENT: {overall_speedup:.2f}x speedup")

        if init_speedup > 1.3 and sampling_speedup > 1.3:
            print("🎯 TARGET ACHIEVED: >30% improvement in key metrics!")
        elif overall_speedup > 1.2:
            print("✅ GOOD IMPROVEMENT: >20% overall speedup")
        else:
            print("⚠️  MODEST IMPROVEMENT: <20% overall speedup")

        improvements["overall_speedup"] = overall_speedup

        return improvements

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


def main():
    """Run the performance comparison."""
    # Reduce logging noise
    logging.basicConfig(level=logging.WARNING)

    comparison = CurriculumPerformanceComparison()
    results = comparison.run_comparison(samples=5000)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
