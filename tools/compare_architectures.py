"""
Architecture comparison tool.

This script systematically compares different model architectures by:
1. Building each architecture variant
2. Benchmarking inference time, memory, and complexity
3. Comparing results across architectures
4. Generating comparison reports

Usage:
    python tools/compare_architectures.py --architectures full_hgt vision_free simplified_hgt
    python tools/compare_architectures.py --all  # Compare all architectures
    python tools/compare_architectures.py --save-results results/  # Save benchmark results

Note: This uses mock observation data. For full evaluation with real training,
      use the training scripts after selecting an architecture.
"""

import argparse
import sys
from pathlib import Path
import torch
import json
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.optimization import (
    get_architecture_config,
    list_available_architectures,
    ConfigurableMultimodalExtractor,
    ArchitectureBenchmark,
    BenchmarkResults,
    create_mock_observations,
)
from gymnasium.spaces import Dict as SpacesDict, Box


def create_observation_space() -> SpacesDict:
    """
    Create observation space matching NPP-RL environment.

    TODO: This should be replaced with actual training set levels once available.
    For now, using mock observation space for architecture comparison.
    """
    return SpacesDict(
        {
            "player_frame": Box(low=0, high=255, shape=(84, 84, 1), dtype="uint8"),
            "global_view": Box(low=0, high=255, shape=(176, 100, 1), dtype="uint8"),
            "game_state": Box(
                low=-float("inf"), high=float("inf"), shape=(30,), dtype="float32"
            ),
            "reachability_features": Box(
                low=-float("inf"), high=float("inf"), shape=(8,), dtype="float32"
            ),
            "graph_obs": SpacesDict(
                {
                    "node_features": Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(100, 67),
                        dtype="float32",
                    ),
                    "edge_index": Box(low=0, high=100, shape=(2, 200), dtype="int64"),
                    "node_mask": Box(low=0, high=1, shape=(100,), dtype="bool"),
                    "edge_mask": Box(low=0, high=1, shape=(200,), dtype="bool"),
                    "node_types": Box(low=0, high=6, shape=(100,), dtype="int64"),
                }
            ),
        }
    )


def build_and_benchmark_architecture(
    arch_name: str,
    observation_space: SpacesDict,
    sample_observations: dict,
    benchmark: ArchitectureBenchmark,
    num_iterations: int = 100,
) -> Optional[BenchmarkResults]:
    """
    Build architecture and run benchmarks.

    Args:
        arch_name: Name of architecture to test
        observation_space: Gymnasium observation space
        sample_observations: Sample observations for benchmarking
        benchmark: ArchitectureBenchmark instance
        num_iterations: Number of inference iterations

    Returns:
        BenchmarkResults or None if architecture failed to build
    """
    try:
        # Get configuration
        config = get_architecture_config(arch_name)
        print(f"\n{'=' * 60}")
        print(f"Building architecture: {arch_name}")
        print(f"{'=' * 60}")

        # Create feature extractor
        extractor = ConfigurableMultimodalExtractor(observation_space, config)

        # Print summary
        num_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
        print("✓ Model built successfully")
        print(f"  - Parameters: {num_params:,}")
        print(
            f"  - Modalities: {', '.join(config.modalities.get_enabled_modalities())}"
        )

        # Run benchmark
        print(f"\nRunning benchmark ({num_iterations} iterations)...")
        results = benchmark.benchmark_model(
            model=extractor,
            sample_observations=sample_observations,
            num_iterations=num_iterations,
            warmup_iterations=10,
            architecture_name=arch_name,
            config=config,
        )

        print("✓ Benchmark complete")
        print(
            f"  - Inference time: {results.mean_inference_time_ms:.2f} ± {results.std_inference_time_ms:.2f} ms"
        )
        print(f"  - Memory: {results.parameter_memory_mb:.2f} MB")

        return results

    except Exception as e:
        print(f"\n✗ Failed to build/benchmark {arch_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare different model architectures for NPP-RL"
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        help="Architecture names to compare (e.g., full_hgt vision_free)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Compare all available architectures"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available architectures and exit"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run benchmarks on (default: auto)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Directory to save benchmark results (JSON format)",
    )

    args = parser.parse_args()

    # List architectures if requested
    if args.list:
        available = list_available_architectures()
        print("\nAvailable architectures:")
        for arch in available:
            config = get_architecture_config(arch)
            print(f"  - {arch}: {config.description}")
        print()
        return

    # Determine which architectures to test
    if args.all:
        architectures_to_test = list_available_architectures()
    elif args.architectures:
        architectures_to_test = args.architectures
    else:
        print("Error: Must specify --architectures or --all")
        parser.print_help()
        return

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'=' * 60}")
    print("Architecture Comparison Tool")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Architectures to test: {len(architectures_to_test)}")
    print(f"{'=' * 60}\n")

    # Create observation space and sample data
    print("Creating mock observation space...")
    print("NOTE: Using mock data. Replace with actual training levels once available.")
    observation_space = create_observation_space()
    sample_observations = create_mock_observations(
        batch_size=args.batch_size, device=device
    )

    # Initialize benchmark
    benchmark = ArchitectureBenchmark(device=device)

    # Run benchmarks for each architecture
    all_results = []
    for arch_name in architectures_to_test:
        result = build_and_benchmark_architecture(
            arch_name=arch_name,
            observation_space=observation_space,
            sample_observations=sample_observations,
            benchmark=benchmark,
            num_iterations=args.iterations,
        )

        if result is not None:
            all_results.append(result)

    # Print comparison
    if all_results:
        print(f"\n{'=' * 80}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 80}\n")

        benchmark.print_comparison_table(all_results)

        # Print detailed comparison
        comparison = benchmark.compare_architectures(all_results)

        print("\nKey Findings:")
        print(
            f"  Fastest Architecture: {comparison['fastest']['name']} "
            f"({comparison['fastest']['time_ms']:.2f} ms)"
        )
        print(
            f"  Most Memory Efficient: {comparison['most_memory_efficient']['name']} "
            f"({comparison['most_memory_efficient']['memory_mb']:.2f} MB)"
        )
        print(
            f"  Smallest Model: {comparison['smallest']['name']} "
            f"({comparison['smallest']['num_params']:,} parameters)"
        )

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Find best architecture based on weighted criteria
        # Performance weight: 40%, Efficiency weight: 30%, Training speed: 20%, Generalization: 10%
        # For now, we can only measure efficiency metrics (inference time, memory)

        print("\nEfficiency Analysis:")

        # Fastest with acceptable memory
        fast_archs = sorted(all_results, key=lambda r: r.mean_inference_time_ms)[:3]
        print("\nTop 3 Fastest Architectures:")
        for i, r in enumerate(fast_archs, 1):
            speedup = (
                all_results[0].mean_inference_time_ms / r.mean_inference_time_ms
                if r != all_results[0]
                else 1.0
            )
            print(
                f"  {i}. {r.architecture_name}: {r.mean_inference_time_ms:.2f} ms "
                f"({speedup:.2f}x baseline)"
            )

        # Memory efficient
        mem_archs = sorted(all_results, key=lambda r: r.parameter_memory_mb)[:3]
        print("\nTop 3 Most Memory Efficient:")
        for i, r in enumerate(mem_archs, 1):
            print(
                f"  {i}. {r.architecture_name}: {r.parameter_memory_mb:.2f} MB "
                f"({r.num_parameters:,} params)"
            )

        print("\nNext Steps:")
        print(
            "  1. Train selected architectures on actual N++ levels (once training set is ready)"
        )
        print("  2. Evaluate final performance and convergence speed")
        print("  3. Select architecture based on weighted criteria")
        print("  4. Fine-tune selected architecture for production use")

        # Save results if requested
        if args.save_results:
            save_dir = Path(args.save_results)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save individual results
            for result in all_results:
                result_file = save_dir / f"{result.architecture_name}_benchmark.json"
                result.save(result_file)
                print(f"\n  Saved: {result_file}")

            # Save comparison
            comparison_file = save_dir / "comparison_summary.json"
            with open(comparison_file, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"  Saved: {comparison_file}")

    else:
        print("\n✗ No successful benchmarks completed")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
