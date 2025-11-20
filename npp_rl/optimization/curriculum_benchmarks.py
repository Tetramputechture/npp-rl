"""Comprehensive performance benchmarks for CurriculumManager optimization.

This module provides detailed benchmarks to measure performance bottlenecks
before and after optimizations, focusing on:
1. Initialization time and memory overhead
2. Hot path performance (sample_level throughput)
3. Memory usage patterns
4. Parallelization overhead
"""

import gc
import json
import logging
import os
import pickle
import psutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Add the project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from npp_rl.training.curriculum_manager import CurriculumManager

logger = logging.getLogger(__name__)


class CurriculumBenchmark:
    """Comprehensive benchmarking suite for CurriculumManager performance."""

    def __init__(self, dataset_path: str, output_dir: Optional[str] = None):
        """Initialize benchmark suite.

        Args:
            dataset_path: Path to curriculum dataset
            output_dir: Directory to save benchmark results
        """
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)

        # Results storage
        self.results = {
            "initialization": {},
            "sampling": {},
            "memory": {},
            "parallelization": {},
        }

        # Process for memory monitoring
        self.process = psutil.Process(os.getpid())

    def create_mock_dataset(self, tmp_dir: Path, levels_per_stage: int = 100) -> str:
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
                    "tiles": np.random.randint(0, 10, (20, 20)).tolist(),
                    "objects": [],
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

    def benchmark_initialization(self, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark CurriculumManager initialization performance.

        Args:
            iterations: Number of initialization runs

        Returns:
            Dictionary with initialization benchmark results
        """
        print(f"Benchmarking initialization ({iterations} iterations)...")

        init_times = []
        memory_usage = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir))

            for i in range(iterations):
                # Force garbage collection before each run
                gc.collect()
                initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

                start_time = time.perf_counter()

                # Initialize curriculum manager
                manager = CurriculumManager(
                    dataset_path=mock_dataset,
                    starting_stage="simplest",
                    performance_window=50,
                )

                end_time = time.perf_counter()
                final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

                init_time = end_time - start_time
                memory_delta = final_memory - initial_memory

                init_times.append(init_time)
                memory_usage.append(memory_delta)

                print(
                    f"  Iteration {i + 1:2d}: {init_time:.4f}s, memory: +{memory_delta:.1f}MB"
                )

                # Clean up
                del manager
                gc.collect()

        results = {
            "iterations": iterations,
            "mean_time": np.mean(init_times),
            "std_time": np.std(init_times),
            "min_time": np.min(init_times),
            "max_time": np.max(init_times),
            "mean_memory_mb": np.mean(memory_usage),
            "std_memory_mb": np.std(memory_usage),
            "raw_times": init_times,
            "raw_memory": memory_usage,
        }

        self.results["initialization"] = results
        return results

    def benchmark_sampling_throughput(
        self, samples: int = 10000, stages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Benchmark sample_level() throughput.

        Args:
            samples: Number of samples to generate
            stages: Specific stages to test, or None for current stage

        Returns:
            Dictionary with sampling benchmark results
        """
        print(f"Benchmarking sampling throughput ({samples} samples)...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir), levels_per_stage=50)

            manager = CurriculumManager(
                dataset_path=mock_dataset,
                starting_stage="simpler",  # Start at a stage that allows mixing
                performance_window=50,
            )

            # Add some performance data to enable mixing
            for _ in range(100):
                manager.record_episode("simplest", np.random.random() < 0.8)
                manager.record_episode("simpler", np.random.random() < 0.6)

            # Warm up JIT/caches
            for _ in range(100):
                manager.sample_level()

            # Benchmark sampling
            start_time = time.perf_counter()
            sample_times = []

            for i in range(samples):
                sample_start = time.perf_counter()
                level = manager.sample_level()
                sample_end = time.perf_counter()

                sample_times.append(sample_end - sample_start)

                if level is None:
                    raise RuntimeError(f"Got None level at sample {i}")

                if i % 1000 == 0 and i > 0:
                    elapsed = sample_end - start_time
                    rate = i / elapsed
                    print(f"  {i:5d} samples: {rate:.1f} samples/sec")

            end_time = time.perf_counter()
            total_time = end_time - start_time

            results = {
                "total_samples": samples,
                "total_time": total_time,
                "samples_per_second": samples / total_time,
                "mean_sample_time": np.mean(sample_times),
                "std_sample_time": np.std(sample_times),
                "p95_sample_time": np.percentile(sample_times, 95),
                "p99_sample_time": np.percentile(sample_times, 99),
                "max_sample_time": np.max(sample_times),
            }

            self.results["sampling"] = results
            return results

    def benchmark_memory_usage(self, samples: int = 1000) -> Dict[str, Any]:
        """Benchmark memory usage patterns during operation.

        Args:
            samples: Number of operations to monitor

        Returns:
            Dictionary with memory benchmark results
        """
        print(f"Benchmarking memory usage ({samples} operations)...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir))

            gc.collect()
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            manager = CurriculumManager(
                dataset_path=mock_dataset,
                starting_stage="simplest",
                performance_window=50,
            )

            post_init_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            init_overhead = post_init_memory - initial_memory

            # Memory tracking during operations
            memory_samples = []
            sample_points = np.linspace(0, samples, 20, dtype=int)

            for i in range(samples):
                # Sample level
                level = manager.sample_level()

                # Record episode (simulate training)
                stage = level.get("sampled_stage", "simplest")
                success = np.random.random() < 0.7
                manager.record_episode(stage, success)

                # Check advancement occasionally
                if i % 50 == 0:
                    manager.check_advancement()

                # Sample memory at specific points
                if i in sample_points:
                    current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(
                        {
                            "operation": i,
                            "memory_mb": current_memory,
                            "delta_from_start": current_memory - initial_memory,
                            "delta_from_init": current_memory - post_init_memory,
                        }
                    )
                    print(
                        f"  Operation {i:4d}: {current_memory:.1f}MB "
                        f"(+{current_memory - post_init_memory:.1f}MB from init)"
                    )

            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            results = {
                "initial_memory_mb": initial_memory,
                "post_init_memory_mb": post_init_memory,
                "final_memory_mb": final_memory,
                "init_overhead_mb": init_overhead,
                "operational_growth_mb": final_memory - post_init_memory,
                "total_growth_mb": final_memory - initial_memory,
                "memory_samples": memory_samples,
            }

            self.results["memory"] = results
            return results

    def benchmark_parallelization_overhead(
        self, num_processes: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Any]:
        """Benchmark parallelization overhead with multiple processes.

        Args:
            num_processes: List of process counts to test

        Returns:
            Dictionary with parallelization benchmark results
        """
        print("Benchmarking parallelization overhead...")

        # Note: Due to pickling limitations with nested functions in multiprocessing,
        # we'll simulate parallelization overhead by measuring initialization costs
        # and provide theoretical scaling analysis

        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_dataset = self.create_mock_dataset(Path(tmp_dir))

            parallel_results = {}
            samples_per_test = 1000

            # Measure single process baseline
            manager = CurriculumManager(
                dataset_path=mock_dataset,
                starting_stage="simpler",
                performance_window=50,
            )

            # Add some performance data
            for _ in range(20):
                manager.record_episode("simplest", np.random.random() < 0.8)
                manager.record_episode("simpler", np.random.random() < 0.6)

            # Measure sampling performance
            start_time = time.perf_counter()
            for _ in range(samples_per_test):
                level = manager.sample_level()
                if level is None:
                    raise RuntimeError("Got None level during parallel benchmark")
            end_time = time.perf_counter()

            baseline_time = end_time - start_time
            baseline_rate = samples_per_test / baseline_time

            # Simulate parallel overhead by measuring manager creation cost
            init_times = []
            for _ in range(5):
                init_start = time.perf_counter()
                test_manager = CurriculumManager(
                    dataset_path=mock_dataset,
                    starting_stage="simpler",
                    performance_window=50,
                )
                init_end = time.perf_counter()
                init_times.append(init_end - init_start)
                del test_manager

            avg_init_time = np.mean(init_times)

            # Calculate theoretical parallel performance
            for n_proc in num_processes:
                print(f"  Simulating {n_proc} processes...")

                if n_proc == 1:
                    # Single process baseline
                    parallel_results[n_proc] = {
                        "total_time": baseline_time,
                        "total_samples": samples_per_test,
                        "aggregate_samples_per_second": baseline_rate,
                        "initialization_overhead": 0.0,
                        "efficiency": 1.0,
                        "speedup": 1.0,
                    }
                else:
                    # Parallel simulation with initialization overhead
                    parallel_init_time = (
                        avg_init_time * n_proc
                    )  # Each process needs initialization
                    parallel_sample_time = (
                        baseline_time / n_proc
                    )  # Perfect parallel scaling
                    total_parallel_time = max(parallel_init_time, parallel_sample_time)

                    effective_rate = (samples_per_test * n_proc) / (
                        total_parallel_time + parallel_init_time
                    )
                    efficiency = effective_rate / (baseline_rate * n_proc)
                    speedup = effective_rate / baseline_rate

                    parallel_results[n_proc] = {
                        "total_time": total_parallel_time,
                        "total_samples": samples_per_test * n_proc,
                        "aggregate_samples_per_second": effective_rate,
                        "initialization_overhead": parallel_init_time,
                        "efficiency": efficiency,
                        "speedup": speedup,
                    }

                result = parallel_results[n_proc]
                print(
                    f"    {n_proc} processes: {result['aggregate_samples_per_second']:.1f} samples/sec "
                    f"(speedup: {result['speedup']:.2f}x, efficiency: {result['efficiency']:.1%})"
                )

        self.results["parallelization"] = parallel_results
        return parallel_results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite.

        Returns:
            Complete benchmark results dictionary
        """
        print("=" * 60)
        print("COMPREHENSIVE CURRICULUM MANAGER BENCHMARKS")
        print("=" * 60)

        # 1. Initialization benchmark
        print("\n1. INITIALIZATION PERFORMANCE")
        print("-" * 30)
        init_results = self.benchmark_initialization(iterations=5)
        print(
            f"Average initialization time: {init_results['mean_time']:.4f}s ± {init_results['std_time']:.4f}s"
        )
        print(
            f"Average memory overhead: {init_results['mean_memory_mb']:.1f}MB ± {init_results['std_memory_mb']:.1f}MB"
        )

        # 2. Sampling throughput benchmark
        print("\n2. SAMPLING THROUGHPUT")
        print("-" * 30)
        sampling_results = self.benchmark_sampling_throughput(samples=5000)
        print(
            f"Sampling rate: {sampling_results['samples_per_second']:.1f} samples/sec"
        )
        print(f"Mean sample time: {sampling_results['mean_sample_time'] * 1000:.3f}ms")
        print(f"P95 sample time: {sampling_results['p95_sample_time'] * 1000:.3f}ms")
        print(f"Max sample time: {sampling_results['max_sample_time'] * 1000:.3f}ms")

        # 3. Memory usage benchmark
        print("\n3. MEMORY USAGE PATTERNS")
        print("-" * 30)
        memory_results = self.benchmark_memory_usage(samples=500)
        print(f"Initialization overhead: {memory_results['init_overhead_mb']:.1f}MB")
        print(f"Operational growth: {memory_results['operational_growth_mb']:.1f}MB")
        print(f"Total memory growth: {memory_results['total_growth_mb']:.1f}MB")

        # 4. Parallelization benchmark
        print("\n4. PARALLELIZATION PERFORMANCE")
        print("-" * 30)
        parallel_results = self.benchmark_parallelization_overhead()

        for n_proc in sorted(parallel_results.keys()):
            result = parallel_results[n_proc]
            print(
                f"  {n_proc} processes: {result['aggregate_samples_per_second']:.1f} samples/sec "
                f"(speedup: {result['speedup']:.2f}x, efficiency: {result['efficiency']:.1%})"
            )

        # Save results
        self.save_results()

        print(f"\n✓ Benchmark results saved to: {self.output_dir}")
        print("=" * 60)

        return self.results

    def save_results(self):
        """Save benchmark results to files."""
        # Save raw results as JSON
        results_file = self.output_dir / "curriculum_benchmark_results.json"

        # Make results JSON serializable
        json_results = {}
        for category, data in self.results.items():
            if isinstance(data, dict):
                json_results[category] = self._make_json_serializable(data)
            else:
                json_results[category] = data

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)

        # Save summary report
        self._generate_summary_report()

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-JSON types to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    def _generate_summary_report(self):
        """Generate human-readable summary report."""
        report_file = self.output_dir / "curriculum_benchmark_summary.txt"

        with open(report_file, "w") as f:
            f.write("CURRICULUM MANAGER PERFORMANCE BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Initialization
            if "initialization" in self.results:
                init = self.results["initialization"]
                f.write("INITIALIZATION PERFORMANCE:\n")
                f.write(
                    f"  Average time: {init['mean_time']:.4f}s ± {init['std_time']:.4f}s\n"
                )
                f.write(f"  Range: {init['min_time']:.4f}s - {init['max_time']:.4f}s\n")
                f.write(
                    f"  Memory overhead: {init['mean_memory_mb']:.1f}MB ± {init['std_memory_mb']:.1f}MB\n\n"
                )

            # Sampling
            if "sampling" in self.results:
                sampling = self.results["sampling"]
                f.write("SAMPLING THROUGHPUT:\n")
                f.write(f"  Rate: {sampling['samples_per_second']:.1f} samples/sec\n")
                f.write(
                    f"  Mean time per sample: {sampling['mean_sample_time'] * 1000:.3f}ms\n"
                )
                f.write(f"  P95 time: {sampling['p95_sample_time'] * 1000:.3f}ms\n")
                f.write(f"  P99 time: {sampling['p99_sample_time'] * 1000:.3f}ms\n\n")

            # Memory
            if "memory" in self.results:
                memory = self.results["memory"]
                f.write("MEMORY USAGE:\n")
                f.write(
                    f"  Initialization overhead: {memory['init_overhead_mb']:.1f}MB\n"
                )
                f.write(
                    f"  Operational growth: {memory['operational_growth_mb']:.1f}MB\n"
                )
                f.write(f"  Total growth: {memory['total_growth_mb']:.1f}MB\n\n")

            # Parallelization
            if "parallelization" in self.results:
                parallel = self.results["parallelization"]
                f.write("PARALLELIZATION EFFICIENCY:\n")
                for n_proc in sorted(parallel.keys()):
                    result = parallel[n_proc]
                    f.write(
                        f"  {n_proc:2d} processes: {result['aggregate_samples_per_second']:6.1f} samples/sec "
                        f"(speedup: {result['speedup']:4.2f}x, efficiency: {result['efficiency']:5.1%})\n"
                    )


def run_benchmark():
    """Run curriculum manager benchmark with mock data."""
    # Use a temporary directory for mock dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        benchmark = CurriculumBenchmark(
            dataset_path="",  # Will be created by mock
            output_dir=str(Path.cwd() / "curriculum_benchmark_results"),
        )

        results = benchmark.run_full_benchmark()
        return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during benchmarking

    # Run benchmark
    results = run_benchmark()

    print("\nBenchmark completed successfully!")
    print("Results saved to: curriculum_benchmark_results/")
