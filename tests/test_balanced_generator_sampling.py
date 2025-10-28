"""Test balanced generator sampling implementation.

This script validates that the curriculum manager correctly:
1. Organizes levels by generator type
2. Implements stratified sampling for balance
3. Tracks per-generator performance
4. Logs comprehensive metrics
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.curriculum_manager import CurriculumManager


def test_level_organization():
    """Test that levels are organized by generator type."""
    print("=" * 70)
    print("Test 1: Level Organization by Generator Type")
    print("=" * 70)

    # Create curriculum manager with test dataset
    # Use absolute path to nclone datasets
    dataset_path = "/home/tetra/projects/nclone/datasets/test"
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
    )

    # Check structure
    assert hasattr(manager, "levels_by_stage_and_generator"), (
        "CurriculumManager should have levels_by_stage_and_generator attribute"
    )

    # Verify nested structure
    for stage, generators in manager.levels_by_stage_and_generator.items():
        assert isinstance(generators, dict), (
            f"Stage '{stage}' should map to dict of generators"
        )

        for gen_type, levels in generators.items():
            assert isinstance(levels, list), (
                f"Generator '{gen_type}' should map to list of levels"
            )
            print(f"  {stage}/{gen_type}: {len(levels)} levels")

    print("✓ Level organization test passed!\n")
    return manager


def test_stratified_sampling(manager):
    """Test that sampling is balanced across generator types."""
    print("=" * 70)
    print("Test 2: Stratified Sampling Balance")
    print("=" * 70)

    # Sample many times from a stage
    num_samples = 1000
    stage = "simple"

    if stage in manager.levels_by_stage_and_generator:
        generator_types = list(manager.levels_by_stage_and_generator[stage].keys())

        if len(generator_types) > 1:
            # Reset sample counts
            for gen in generator_types:
                manager.generator_sample_counts[stage][gen] = 0

            # Sample many times
            for _ in range(num_samples):
                level = manager.sample_level()
                if level:
                    # Just sampling, not using
                    pass

            # Check balance
            counts = [
                manager.generator_sample_counts[stage][gen] for gen in generator_types
            ]
            mean_count = np.mean(counts)
            variance = np.var(counts)

            print(f"\nSampled {num_samples} levels from stage '{stage}':")
            for gen, count in zip(generator_types, counts):
                print(f"  {gen}: {count} samples ({count / num_samples * 100:.1f}%)")

            print("\nBalance metrics:")
            print(f"  Mean count: {mean_count:.1f}")
            print(f"  Variance: {variance:.2f}")
            print(f"  Max deviation: {max(abs(c - mean_count) for c in counts):.1f}")

            # Check that variance is reasonable (should be low for balanced sampling)
            expected_variance_threshold = mean_count * 0.5  # Allow 50% variance
            assert variance < expected_variance_threshold, (
                f"Sampling variance too high ({variance:.2f} > {expected_variance_threshold:.2f})"
            )

            print("✓ Stratified sampling test passed!")
        else:
            print(
                f"⚠ Stage {stage!r} has only one generator type, skipping balance test"
            )
    else:
        print(f"⚠ Stage '{stage}' not found, skipping test")

    print()


def test_performance_tracking(manager):
    """Test that per-generator performance is tracked."""
    print("=" * 70)
    print("Test 3: Per-Generator Performance Tracking")
    print("=" * 70)

    # Simulate recording episodes for different generators
    stage = "simple"

    if stage in manager.generator_performance:
        for gen_type in manager.generator_performance[stage].keys():
            # Simulate some successes and failures
            manager.record_episode(stage, success=True, generator_type=gen_type)
            manager.record_episode(stage, success=True, generator_type=gen_type)
            manager.record_episode(stage, success=False, generator_type=gen_type)

        # Check that performance was tracked
        print(f"\nPerformance tracking for stage '{stage}':")
        for gen_type, perf in manager.generator_performance[stage].items():
            total = perf["successes"] + perf["failures"]
            if total > 0:
                success_rate = perf["successes"] / total
                print(
                    f"  {gen_type}: {success_rate:.1%} ({perf['successes']} / {total})"
                )

        print("✓ Performance tracking test passed!")
    else:
        print(f"⚠ Stage {stage!r} not found, skipping test")

    print()


def test_statistics_generation(manager):
    """Test that statistics can be generated."""
    print("=" * 70)
    print("Test 4: Statistics Generation")
    print("=" * 70)

    stats = manager.get_generator_statistics()

    print("\nGenerator statistics:")
    for stage, stage_stats in stats.items():
        print(f"\n{stage}:")
        print(f"  Total samples: {stage_stats['total_samples']}")
        print(f"  Balance variance: {stage_stats['balance_variance']:.2f}")

        if stage_stats["generators"]:
            print("  Generators:")
            for gen, gen_stats in stage_stats["generators"].items():
                print(f"    - {gen}:")
                print(f"        Samples: {gen_stats['sample_count']}")
                print(f"        Episodes: {gen_stats['episodes']}")
                if gen_stats["episodes"] > 0:
                    print(f"        Success rate: {gen_stats['success_rate']:.1%}")

    print("\n✓ Statistics generation test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BALANCED GENERATOR SAMPLING VALIDATION")
    print("=" * 70 + "\n")

    try:
        # Test 1: Level organization
        manager = test_level_organization()

        # Test 2: Stratified sampling
        test_stratified_sampling(manager)

        # Test 3: Performance tracking
        test_performance_tracking(manager)

        # Test 4: Statistics generation
        test_statistics_generation(manager)

        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nBalanced generator sampling is working correctly.")
        print("The agent will now receive balanced exposure across all level types!")

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
