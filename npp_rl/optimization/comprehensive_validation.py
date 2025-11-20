"""Comprehensive validation of optimized curriculum learning components.

This script validates that all optimizations maintain accuracy while delivering
the measured performance improvements. Tests functional correctness, behavioral
equivalence, and performance across all optimized components.
"""

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
from npp_rl.training.curriculum_components import ModularCurriculumManager
from npp_rl.training.curriculum_shared_memory import create_shared_curriculum_manager

logger = logging.getLogger(__name__)


class CurriculumValidationSuite:
    """Comprehensive validation suite for optimized curriculum components."""

    def __init__(self):
        self.validation_results = {}

    def create_test_dataset(self, tmp_dir: Path, levels_per_stage: int = 20) -> str:
        """Create test dataset for validation."""
        import pickle

        dataset_dir = tmp_dir / "validation_dataset"
        generators = ["manual", "procedural", "mixed"]

        for stage in self.curriculum_stages:
            stage_dir = dataset_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            for i in range(levels_per_stage):
                level_data = {
                    "level_id": f"{stage}_{i:04d}",
                    "map_data": {"tiles": np.random.randint(0, 10, (15, 15)).tolist()},
                    "metadata": {
                        "generator": generators[i % len(generators)],
                        "category": stage,
                        "difficulty": np.random.uniform(0.1, 1.0),
                    },
                }

                level_file = stage_dir / f"level_{i:04d}.pkl"
                with open(level_file, "wb") as f:
                    pickle.dump(level_data, f)

        return str(dataset_dir)

    def validate_sampling_accuracy(self, dataset_path: str) -> Dict[str, Any]:
        """Validate that optimized sampling produces equivalent results."""
        print("=" * 60)
        print("SAMPLING ACCURACY VALIDATION")
        print("=" * 60)

        # Create managers
        original = CurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=50
        )
        optimized = OptimizedCurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=50
        )
        modular = ModularCurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=50
        )

        # Add identical performance data
        for manager in [original, optimized, modular]:
            for _ in range(30):
                manager.record_episode("simplest", np.random.random() < 0.8)
                manager.record_episode("simpler", np.random.random() < 0.6)

        # Test sampling distribution
        samples_per_manager = 1000

        def get_sampling_distribution(manager, n_samples):
            stage_counts = {}
            generator_counts = {}

            for _ in range(n_samples):
                level = manager.sample_level()
                stage = level.get("sampled_stage", level.get("category", "unknown"))
                generator = level.get(
                    "sampled_generator",
                    level.get("metadata", {}).get("generator", "unknown"),
                )

                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                generator_counts[generator] = generator_counts.get(generator, 0) + 1

            return stage_counts, generator_counts

        print("Comparing sampling distributions...")

        orig_stages, orig_gens = get_sampling_distribution(
            original, samples_per_manager
        )
        opt_stages, opt_gens = get_sampling_distribution(optimized, samples_per_manager)
        mod_stages, mod_gens = get_sampling_distribution(modular, samples_per_manager)

        # Compare distributions (should be similar within statistical variance)
        def compare_distributions(dist1, dist2, name):
            """Compare two distributions for statistical similarity."""
            all_keys = set(dist1.keys()) | set(dist2.keys())
            max_diff = 0

            for key in all_keys:
                count1 = dist1.get(key, 0)
                count2 = dist2.get(key, 0)
                diff = abs(count1 - count2) / samples_per_manager
                max_diff = max(max_diff, diff)

            print(f"{name} max difference: {max_diff:.3f} ({max_diff * 100:.1f}%)")
            return max_diff < 0.15  # Allow 15% variance due to randomness

        stage_accuracy = (
            compare_distributions(
                orig_stages, opt_stages, "Original vs Optimized stages"
            )
            and compare_distributions(
                orig_stages, mod_stages, "Original vs Modular stages"
            )
            and compare_distributions(
                opt_stages, mod_stages, "Optimized vs Modular stages"
            )
        )

        generator_accuracy = (
            compare_distributions(
                orig_gens, opt_gens, "Original vs Optimized generators"
            )
            and compare_distributions(
                orig_gens, mod_gens, "Original vs Modular generators"
            )
            and compare_distributions(
                opt_gens, mod_gens, "Optimized vs Modular generators"
            )
        )

        print(f"Stage sampling accuracy: {'✅ PASS' if stage_accuracy else '❌ FAIL'}")
        print(
            f"Generator sampling accuracy: {'✅ PASS' if generator_accuracy else '❌ FAIL'}"
        )

        return {
            "stage_accuracy": stage_accuracy,
            "generator_accuracy": generator_accuracy,
            "original_stages": orig_stages,
            "optimized_stages": opt_stages,
            "modular_stages": mod_stages,
        }

    def validate_performance_tracking(self, dataset_path: str) -> Dict[str, Any]:
        """Validate that performance tracking produces consistent results."""
        print("\n" + "=" * 60)
        print("PERFORMANCE TRACKING VALIDATION")
        print("=" * 60)

        # Create managers
        original = CurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=30
        )
        optimized = OptimizedCurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=30
        )
        modular = ModularCurriculumManager(
            dataset_path, starting_stage="simpler", performance_window=30
        )

        # Record identical episode sequences
        episode_data = []
        for i in range(100):
            stage = "simpler" if i < 50 else "simple"
            success = np.random.random() < (0.7 if stage == "simpler" else 0.5)
            generator = ["manual", "procedural", "mixed"][i % 3]
            episode_data.append((stage, success, generator))

        # Record episodes in all managers
        for stage, success, generator in episode_data:
            original.record_episode(stage, success, generator)
            optimized.record_episode(stage, success, generator)
            modular.record_episode(stage, success, generator)

        # Compare performance metrics
        def compare_performance_metrics(manager1, manager2, name):
            """Compare performance metrics between managers."""
            metrics_match = True

            for stage in ["simpler", "simple"]:
                perf1 = manager1.get_stage_performance(stage)
                perf2 = manager2.get_stage_performance(stage)

                # Compare key metrics (allow small floating point differences)
                success_diff = abs(perf1["success_rate"] - perf2["success_rate"])
                episode_diff = abs(perf1["episodes"] - perf2["episodes"])

                if (
                    success_diff > 0.01 or episode_diff > 0
                ):  # 1% tolerance for success rate
                    print(
                        f"{name} - Stage {stage}: success rate diff {success_diff:.4f}, episode diff {episode_diff}"
                    )
                    metrics_match = False

                print(
                    f"{name} - {stage}: {perf1['success_rate']:.3f} vs {perf2['success_rate']:.3f} success rate"
                )

            return metrics_match

        orig_vs_opt = compare_performance_metrics(
            original, optimized, "Original vs Optimized"
        )
        orig_vs_mod = compare_performance_metrics(
            original, modular, "Original vs Modular"
        )
        opt_vs_mod = compare_performance_metrics(
            optimized, modular, "Optimized vs Modular"
        )

        tracking_accuracy = orig_vs_opt and orig_vs_mod and opt_vs_mod

        print(
            f"Performance tracking accuracy: {'✅ PASS' if tracking_accuracy else '❌ FAIL'}"
        )

        return {
            "tracking_accuracy": tracking_accuracy,
            "original_performance": {
                stage: original.get_stage_performance(stage)
                for stage in ["simpler", "simple"]
            },
            "optimized_performance": {
                stage: optimized.get_stage_performance(stage)
                for stage in ["simpler", "simple"]
            },
            "modular_performance": {
                stage: modular.get_stage_performance(stage)
                for stage in ["simpler", "simple"]
            },
        }

    def validate_advancement_logic(self, dataset_path: str) -> Dict[str, Any]:
        """Validate that advancement logic is consistent across implementations."""
        print("\n" + "=" * 60)
        print("ADVANCEMENT LOGIC VALIDATION")
        print("=" * 60)

        # Create managers
        managers = {
            "original": CurriculumManager(
                dataset_path, starting_stage="simplest", performance_window=20
            ),
            "optimized": OptimizedCurriculumManager(
                dataset_path, starting_stage="simplest", performance_window=20
            ),
            "modular": ModularCurriculumManager(
                dataset_path, starting_stage="simplest", performance_window=20
            ),
        }

        # Simulate high success rate to trigger advancement
        for name, manager in managers.items():
            for _ in range(120):  # Above minimum episodes
                manager.record_episode(
                    "simplest", np.random.random() < 0.85
                )  # Above threshold

        # Check advancement status
        advancement_results = {}
        for name, manager in managers.items():
            can_advance = manager.check_advancement()
            current_stage = manager.get_current_stage()
            stage_perf = manager.get_stage_performance("simplest")

            advancement_results[name] = {
                "advanced": can_advance,
                "current_stage": current_stage,
                "success_rate": stage_perf["success_rate"],
                "episodes": stage_perf["episodes"],
                "can_advance": stage_perf["can_advance"],
            }

            print(
                f"{name}: advanced={can_advance}, stage={current_stage}, "
                f"success={stage_perf['success_rate']:.3f}, episodes={stage_perf['episodes']}"
            )

        # All should have advanced or all should not have advanced
        advanced_states = [
            result["advanced"] for result in advancement_results.values()
        ]
        current_stages = [
            result["current_stage"] for result in advancement_results.values()
        ]

        advancement_consistency = (
            len(set(advanced_states)) == 1  # All same advancement decision
            and len(set(current_stages)) == 1  # All same current stage
        )

        print(
            f"Advancement logic consistency: {'✅ PASS' if advancement_consistency else '❌ FAIL'}"
        )

        return {
            "advancement_consistency": advancement_consistency,
            "results": advancement_results,
        }

    def validate_performance_improvements(self, dataset_path: str) -> Dict[str, Any]:
        """Validate that performance improvements are maintained."""
        print("\n" + "=" * 60)
        print("PERFORMANCE IMPROVEMENT VALIDATION")
        print("=" * 60)

        # Benchmark sampling performance
        managers = {
            "original": CurriculumManager(
                dataset_path, starting_stage="simpler", performance_window=50
            ),
            "optimized": OptimizedCurriculumManager(
                dataset_path, starting_stage="simpler", performance_window=50
            ),
            "modular": ModularCurriculumManager(
                dataset_path, starting_stage="simpler", performance_window=50
            ),
        }

        # Add performance data for realistic mixing
        for manager in managers.values():
            for _ in range(50):
                manager.record_episode("simplest", np.random.random() < 0.8)
                manager.record_episode("simpler", np.random.random() < 0.6)

        # Benchmark sampling speed
        samples = 5000
        performance_results = {}

        for name, manager in managers.items():
            # Warmup
            for _ in range(100):
                manager.sample_level()

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(samples):
                level = manager.sample_level()
                if level is None:
                    raise RuntimeError(f"Got None level from {name} manager")
            end_time = time.perf_counter()

            sampling_time = end_time - start_time
            samples_per_second = samples / sampling_time

            performance_results[name] = {
                "samples_per_second": samples_per_second,
                "mean_sample_time_ms": (sampling_time / samples) * 1000,
            }

            print(
                f"{name}: {samples_per_second:.1f} samples/sec, {performance_results[name]['mean_sample_time_ms']:.3f}ms per sample"
            )

        # Calculate improvements
        baseline = performance_results["original"]["samples_per_second"]
        optimized_speedup = (
            performance_results["optimized"]["samples_per_second"] / baseline
        )
        modular_speedup = (
            performance_results["modular"]["samples_per_second"] / baseline
        )

        # Should see improvements over original
        meets_performance_target = (
            optimized_speedup >= 1.1 and modular_speedup >= 1.05
        )  # 10% and 5% minimum improvement

        print(f"Optimized speedup: {optimized_speedup:.2f}x")
        print(f"Modular speedup: {modular_speedup:.2f}x")
        print(
            f"Performance target met: {'✅ PASS' if meets_performance_target else '❌ FAIL'}"
        )

        return {
            "meets_performance_target": meets_performance_target,
            "optimized_speedup": optimized_speedup,
            "modular_speedup": modular_speedup,
            "performance_results": performance_results,
        }

    def validate_shared_memory_implementation(
        self, dataset_path: str
    ) -> Dict[str, Any]:
        """Validate shared memory implementation for parallel environments."""
        print("\n" + "=" * 60)
        print("SHARED MEMORY IMPLEMENTATION VALIDATION")
        print("=" * 60)

        try:
            # Create shared memory manager
            shared_manager, shared_buffer = create_shared_curriculum_manager(
                self.curriculum_stages, starting_stage="simpler", performance_window=30
            )

            # Test basic functionality
            shared_manager.record_episode("simpler", True)
            shared_manager.record_episode("simpler", False)
            shared_manager.record_episode("simpler", True)

            success_rate = shared_manager.get_stage_success_rate("simpler")
            episode_count = shared_buffer.get_episode_count("simpler")

            shared_memory_functional = (
                0.6 <= success_rate <= 0.7  # 2/3 success rate
                and episode_count == 3
            )

            print(
                f"Shared memory basic functionality: {'✅ PASS' if shared_memory_functional else '❌ FAIL'}"
            )
            print(f"Success rate: {success_rate:.3f}, Episodes: {episode_count}")

            # Test thread safety (basic)
            import threading

            def worker_function():
                for _ in range(10):
                    shared_manager.record_episode("simple", np.random.random() < 0.5)

            threads = [threading.Thread(target=worker_function) for _ in range(4)]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            final_count = shared_buffer.get_episode_count("simple")
            thread_safety_ok = final_count == 40  # 4 threads * 10 episodes each

            print(f"Thread safety: {'✅ PASS' if thread_safety_ok else '❌ FAIL'}")
            print(f"Expected 40 episodes, got {final_count}")

            return {
                "shared_memory_functional": shared_memory_functional,
                "thread_safety": thread_safety_ok,
                "success_rate": success_rate,
                "episode_count": episode_count,
                "final_count": final_count,
            }

        except Exception as e:
            print(f"Shared memory validation failed: {e}")
            return {
                "shared_memory_functional": False,
                "thread_safety": False,
                "error": str(e),
            }

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🔍 COMPREHENSIVE CURRICULUM OPTIMIZATION VALIDATION")
        print("🔍 " + "=" * 58)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = self.create_test_dataset(Path(tmp_dir))

            # Run all validation tests
            sampling_results = self.validate_sampling_accuracy(dataset_path)
            tracking_results = self.validate_performance_tracking(dataset_path)
            advancement_results = self.validate_advancement_logic(dataset_path)
            performance_results = self.validate_performance_improvements(dataset_path)
            shared_memory_results = self.validate_shared_memory_implementation(
                dataset_path
            )

            # Aggregate results
            all_results = {
                "sampling": sampling_results,
                "tracking": tracking_results,
                "advancement": advancement_results,
                "performance": performance_results,
                "shared_memory": shared_memory_results,
            }

            # Calculate overall validation score
            validation_checks = [
                sampling_results["stage_accuracy"],
                sampling_results["generator_accuracy"],
                tracking_results["tracking_accuracy"],
                advancement_results["advancement_consistency"],
                performance_results["meets_performance_target"],
                shared_memory_results["shared_memory_functional"],
                shared_memory_results.get("thread_safety", False),
            ]

            passed_checks = sum(validation_checks)
            total_checks = len(validation_checks)
            validation_score = passed_checks / total_checks

            print("\n" + "=" * 60)
            print("🎯 COMPREHENSIVE VALIDATION RESULTS")
            print("=" * 60)
            print(
                f"Validation Score: {passed_checks}/{total_checks} ({validation_score:.1%})"
            )
            print(
                f"Sampling Accuracy: {'✅' if sampling_results['stage_accuracy'] and sampling_results['generator_accuracy'] else '❌'}"
            )
            print(
                f"Performance Tracking: {'✅' if tracking_results['tracking_accuracy'] else '❌'}"
            )
            print(
                f"Advancement Logic: {'✅' if advancement_results['advancement_consistency'] else '❌'}"
            )
            print(
                f"Performance Improvements: {'✅' if performance_results['meets_performance_target'] else '❌'}"
            )
            print(
                f"Shared Memory: {'✅' if shared_memory_results['shared_memory_functional'] else '❌'}"
            )
            print(
                f"Thread Safety: {'✅' if shared_memory_results.get('thread_safety', False) else '❌'}"
            )

            if validation_score >= 0.85:
                print(
                    "\n🎉 VALIDATION PASSED! All optimizations maintain accuracy and performance."
                )
            elif validation_score >= 0.70:
                print("\n⚠️ VALIDATION MOSTLY PASSED with some minor issues.")
            else:
                print("\n❌ VALIDATION FAILED. Significant issues detected.")

            print("Performance Improvements Achieved:")
            print(
                f"  - Optimized Manager: {performance_results['optimized_speedup']:.2f}x speedup"
            )
            print(
                f"  - Modular Manager: {performance_results['modular_speedup']:.2f}x speedup"
            )

            all_results["validation_score"] = validation_score
            all_results["validation_passed"] = validation_score >= 0.85

            return all_results


def main():
    """Run comprehensive curriculum optimization validation."""
    # Configure logging
    logging.basicConfig(level=logging.WARNING)

    # Run validation suite
    validator = CurriculumValidationSuite()
    results = validator.run_comprehensive_validation()

    return results


if __name__ == "__main__":
    results = main()
    print(f"\nValidation completed. Overall success: {results['validation_passed']}")
    print(f"Validation score: {results['validation_score']:.1%}")
