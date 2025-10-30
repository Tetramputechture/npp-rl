#!/usr/bin/env python3
"""Profile GPU memory usage per environment for different architectures.

This script measures actual GPU memory footprint of environments and models
to inform auto-detection parameters in hardware_profiles.py.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gc
import numpy as np
import psutil
import torch

from npp_rl.training.architecture_configs import (
    ARCHITECTURE_REGISTRY,
    get_architecture_config,
)
from npp_rl.training.environment_factory import EnvironmentFactory
from npp_rl.training.architecture_trainer import ArchitectureTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def get_gpu_memory_peak_gb() -> float:
    """Get peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def get_cpu_memory_gb() -> float:
    """Get current CPU memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1e9


def measure_memory_at_checkpoint(
    checkpoint_name: str, reset_peak: bool = False
) -> Dict[str, float]:
    """Measure GPU and CPU memory at a checkpoint.

    Args:
        checkpoint_name: Name of the checkpoint
        reset_peak: Whether to reset peak memory before measurement

    Returns:
        Dictionary with memory metrics
    """
    if reset_peak:
        reset_gpu_memory_stats()

    torch.cuda.synchronize()
    gpu_allocated = get_gpu_memory_gb()
    gpu_peak = get_gpu_memory_peak_gb()
    cpu_memory = get_cpu_memory_gb()

    return {
        "checkpoint": checkpoint_name,
        "allocated_gb": gpu_allocated,  # GPU memory for backward compatibility
        "peak_gb": gpu_peak,
        "gpu_memory_gb": gpu_allocated,
        "cpu_memory_gb": cpu_memory,
        "total_memory_gb": gpu_allocated + cpu_memory,  # Note: rough estimate
    }


def profile_architecture_memory(
    architecture_name: str,
    num_envs_list: List[int],
    batch_size: int = 256,
    n_steps: int = 1024,
    train_dataset_path: Optional[str] = None,
    test_dataset_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, any]:
    """Profile memory usage for a specific architecture.

    Args:
        architecture_name: Architecture name to profile
        num_envs_list: List of environment counts to test incrementally
        batch_size: Batch size for buffer measurement
        n_steps: Number of steps per rollout
        train_dataset_path: Path to training dataset (required for EnvironmentFactory)
        test_dataset_path: Path to test dataset (required for EnvironmentFactory)
        output_dir: Output directory for trainer

    Returns:
        Dictionary with memory profiling results
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Profiling architecture: {architecture_name}")
    logger.info(f"{'=' * 70}")

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot profile GPU memory.")
        return {}

    # Get architecture config
    try:
        architecture_config = get_architecture_config(architecture_name)
    except ValueError as e:
        logger.error(f"Failed to get architecture config: {e}")
        return {}

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_memory_total_gb:.1f} GB)")

    # Default paths if not provided
    if train_dataset_path is None:
        train_dataset_path = "data/train"
    if test_dataset_path is None:
        test_dataset_path = "data/test"
    if output_dir is None:
        output_dir = Path("/tmp/memory_profiling")

    results = {
        "architecture": architecture_name,
        "gpu_name": gpu_name,
        "gpu_memory_total_gb": gpu_memory_total_gb,
        "checkpoints": [],
        "per_env_measurements": [],
    }

    # Baseline measurement
    reset_gpu_memory_stats()
    gc.collect()  # Force garbage collection for accurate CPU measurement
    baseline = measure_memory_at_checkpoint("baseline", reset_peak=True)
    results["checkpoints"].append(baseline)
    baseline_memory = baseline["allocated_gb"]
    baseline_cpu_memory = baseline["cpu_memory_gb"]

    # Test incremental environment counts
    prev_env_memory = baseline_memory
    prev_cpu_memory = baseline_cpu_memory
    prev_num_envs = 0

    for num_envs in num_envs_list:
        logger.info(f"\nTesting with {num_envs} environments...")

        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            reset_gpu_memory_stats()

            # Create environment factory
            env_factory = EnvironmentFactory(
                use_curriculum=False,
                frame_stack_config=None,
                enable_pbrs=False,
            )

            # Create environments separately for memory measurement
            # Note: setup_environments() will create environments again, but this
            # gives us a baseline measurement of just environment memory
            env = env_factory.create_training_env(
                num_envs=num_envs,
                gamma=0.99,
                enable_visualization=False,
            )

            gc.collect()  # Force garbage collection before measurement
            env_memory = measure_memory_at_checkpoint(f"envs_{num_envs}")
            env_memory["delta_gb"] = env_memory["allocated_gb"] - baseline_memory
            env_memory["delta_cpu_gb"] = (
                env_memory["cpu_memory_gb"] - baseline_cpu_memory
            )
            results["checkpoints"].append(env_memory)

            # Calculate memory per environment (need at least 2 measurements)
            # Use CPU memory as primary metric since environments are CPU-resident
            # GPU memory is primarily for model weights/activations
            if prev_num_envs > 0:
                # Calculate GPU memory per env
                gpu_memory_per_env = (env_memory["allocated_gb"] - prev_env_memory) / (
                    num_envs - prev_num_envs
                )
                # Calculate CPU memory per env (more accurate for environments)
                cpu_memory_per_env = (env_memory["cpu_memory_gb"] - prev_cpu_memory) / (
                    num_envs - prev_num_envs
                )
                # Total memory per env (GPU + CPU)
                total_memory_per_env = gpu_memory_per_env + cpu_memory_per_env

                if total_memory_per_env > 0:  # Sanity check
                    results["per_env_measurements"].append(
                        {
                            "num_envs": num_envs,
                            "gpu_memory_per_env_gb": gpu_memory_per_env,
                            "cpu_memory_per_env_gb": cpu_memory_per_env,
                            "memory_per_env_gb": total_memory_per_env,  # Total for backward compatibility
                            "total_gpu_memory_gb": env_memory["allocated_gb"],
                            "total_cpu_memory_gb": env_memory["cpu_memory_gb"],
                            "delta_gpu_memory_gb": env_memory["allocated_gb"]
                            - prev_env_memory,
                            "delta_cpu_memory_gb": env_memory["cpu_memory_gb"]
                            - prev_cpu_memory,
                        }
                    )
                    logger.info(
                        f"  Memory per environment: {total_memory_per_env:.3f} GB "
                        f"(GPU: {gpu_memory_per_env:.3f} GB, CPU: {cpu_memory_per_env:.3f} GB)"
                    )
                else:
                    logger.warning(
                        f"  Invalid memory per environment calculation: total={total_memory_per_env:.3f}"
                    )

            # Create trainer
            trainer = ArchitectureTrainer(
                architecture_config=architecture_config,
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                output_dir=output_dir,
                device_id=0,
                world_size=1,
                use_mixed_precision=False,
                use_hierarchical_ppo=False,
                use_curriculum=False,
                use_distributed=False,
                frame_stack_config=None,
                enable_pbrs=False,
            )

            # Setup model first (this initializes hyperparams needed by setup_environments)
            trainer.setup_model(batch_size=batch_size, n_steps=n_steps)

            # Setup environments (will create new environments, but that's okay for memory profiling)
            trainer.setup_environments(
                num_envs=num_envs,
                total_timesteps=None,
                enable_visualization=False,
            )

            # Measure memory after model creation
            model_memory = measure_memory_at_checkpoint(f"model_with_{num_envs}_envs")
            model_memory["delta_gb"] = model_memory["allocated_gb"] - baseline_memory
            results["checkpoints"].append(model_memory)

            logger.info(
                f"  Memory after model creation: {model_memory['allocated_gb']:.2f} GB "
                f"(delta: {model_memory['delta_gb']:.2f} GB)"
            )

            # Measure peak memory during forward pass
            reset_gpu_memory_stats()
            try:
                obs = trainer.env.reset()

                # Convert observation to torch tensors if needed
                # Stable-baselines3 models expect torch tensors, not numpy arrays
                if isinstance(obs, dict):
                    obs_tensors = {}
                    device = next(trainer.model.policy.parameters()).device
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            # Convert numpy to tensor and move to device
                            obs_tensors[key] = torch.from_numpy(value).to(device)
                        elif isinstance(value, torch.Tensor):
                            obs_tensors[key] = value.to(device)
                        else:
                            obs_tensors[key] = torch.tensor(value, device=device)
                    obs = obs_tensors
                elif isinstance(obs, np.ndarray):
                    device = next(trainer.model.policy.parameters()).device
                    obs = torch.from_numpy(obs).to(device)

                with torch.no_grad():
                    _ = trainer.model.policy(obs)
                torch.cuda.synchronize()

                forward_memory = measure_memory_at_checkpoint(
                    f"forward_pass_{num_envs}_envs"
                )
                forward_memory["delta_gb"] = (
                    forward_memory["allocated_gb"] - baseline_memory
                )
                results["checkpoints"].append(forward_memory)

                logger.info(
                    f"  Peak memory during forward pass: {forward_memory['peak_gb']:.2f} GB"
                )
            except Exception as e:
                logger.warning(f"  Could not measure forward pass memory: {e}")
                # Still measure memory even if forward pass failed
                forward_memory = measure_memory_at_checkpoint(
                    f"forward_pass_{num_envs}_envs_failed"
                )
                forward_memory["delta_gb"] = (
                    forward_memory["allocated_gb"] - baseline_memory
                )
                forward_memory["error"] = str(e)
                results["checkpoints"].append(forward_memory)

            # Calculate model overhead (difference between env only and env + model)
            model_overhead = model_memory["allocated_gb"] - env_memory["allocated_gb"]
            logger.info(f"  Model overhead: {model_overhead:.2f} GB")

            # Cleanup
            del trainer.model
            del trainer.env
            del trainer.eval_env
            del env  # Clean up the separately created env too
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection after cleanup

            prev_env_memory = env_memory["allocated_gb"]
            prev_cpu_memory = env_memory["cpu_memory_gb"]
            prev_num_envs = num_envs

        except Exception as e:
            logger.error(f"Error profiling {num_envs} environments: {e}")
            logger.exception("Full traceback:")
            break

    # Calculate summary statistics
    if results["per_env_measurements"]:
        memory_values = [
            m["memory_per_env_gb"] for m in results["per_env_measurements"]
        ]
        cpu_memory_values = [
            m["cpu_memory_per_env_gb"] for m in results["per_env_measurements"]
        ]
        gpu_memory_values = [
            m["gpu_memory_per_env_gb"] for m in results["per_env_measurements"]
        ]

        if memory_values and len(memory_values) > 0:
            avg_memory = sum(memory_values) / len(memory_values)
            avg_cpu_memory = sum(cpu_memory_values) / len(cpu_memory_values)
            avg_gpu_memory = sum(gpu_memory_values) / len(gpu_memory_values)

            if avg_memory > 0:
                results["summary"] = {
                    "avg_memory_per_env_gb": avg_memory,
                    "avg_cpu_memory_per_env_gb": avg_cpu_memory,
                    "avg_gpu_memory_per_env_gb": avg_gpu_memory,
                    "min_memory_per_env_gb": min(memory_values),
                    "max_memory_per_env_gb": max(memory_values),
                    "recommended_max_envs_per_gpu": int(
                        (gpu_memory_total_gb * 0.8)
                        / max(avg_gpu_memory, 0.001)  # Avoid division by zero
                    ),
                }

                logger.info(f"\nSummary for {architecture_name}:")
                logger.info(
                    f"  Average memory per environment: "
                    f"{results['summary']['avg_memory_per_env_gb']:.3f} GB "
                    f"(CPU: {results['summary']['avg_cpu_memory_per_env_gb']:.3f} GB, "
                    f"GPU: {results['summary']['avg_gpu_memory_per_env_gb']:.3f} GB)"
                )
                logger.info(
                    f"  Recommended max envs per GPU (based on GPU memory): "
                    f"{results['summary']['recommended_max_envs_per_gpu']}"
                )
                logger.info(
                    f"  Note: CPU memory is separate and may be the limiting factor "
                    f"for very large numbers of environments"
                )
            else:
                logger.warning(
                    f"  Cannot calculate summary: average memory per environment is zero"
                )
        else:
            logger.warning(
                f"  Cannot calculate summary: no valid memory measurements collected"
            )
    else:
        logger.warning(
            f"  Cannot calculate summary: need at least 2 environment counts to measure memory per environment"
        )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile GPU memory usage per environment for different architectures"
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["mlp_baseline", "full_hgt", "simplified_hgt", "gat", "gcn"],
        help="Architecture names to profile (default: mlp_baseline full_hgt simplified_hgt gat gcn)",
    )
    parser.add_argument(
        "--max-envs",
        type=int,
        default=64,
        help="Maximum number of environments to test (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for buffer measurement (default: 256)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Number of steps per rollout (default: 1024)",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default=None,
        help="Path to training dataset (default: data/train)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to test dataset (default: data/test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: memory_profile_results.json)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA not available. Cannot profile GPU memory.")
        sys.exit(1)

    # Generate list of environment counts to test
    num_envs_list = []
    current = 1
    while current <= args.max_envs:
        num_envs_list.append(current)
        current *= 2

    logger.info("=" * 70)
    logger.info("GPU Memory Profiling for NPP-RL Architectures")
    logger.info("=" * 70)
    logger.info(f"Architectures to profile: {args.architectures}")
    logger.info(f"Environment counts to test: {num_envs_list}")
    logger.info(f"Batch size: {args.batch_size}, N steps: {args.n_steps}")
    logger.info("=" * 70)

    all_results = {
        "config": {
            "architectures": args.architectures,
            "num_envs_list": num_envs_list,
            "batch_size": args.batch_size,
            "n_steps": args.n_steps,
        },
        "results": [],
    }

    # Profile each architecture
    for arch_name in args.architectures:
        if arch_name not in ARCHITECTURE_REGISTRY:
            logger.warning(
                f"Architecture '{arch_name}' not found in registry. Skipping."
            )
            continue

        try:
            result = profile_architecture_memory(
                architecture_name=arch_name,
                num_envs_list=num_envs_list,
                batch_size=args.batch_size,
                n_steps=args.n_steps,
                train_dataset_path=args.train_dataset,
                test_dataset_path=args.test_dataset,
            )
            all_results["results"].append(result)
        except Exception as e:
            logger.error(f"Failed to profile {arch_name}: {e}")
            logger.exception("Full traceback:")

    # Save results
    output_path = args.output or "memory_profile_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 70)

    # Print summary comparison
    logger.info("\nArchitecture Comparison Summary:")
    logger.info("-" * 70)
    summary_count = 0
    for result in all_results["results"]:
        if "summary" in result and result["summary"]:
            summary_count += 1
            logger.info(
                f"{result['architecture']:20s} | "
                f"Total: {result['summary']['avg_memory_per_env_gb']:6.3f} GB "
                f"(CPU: {result['summary'].get('avg_cpu_memory_per_env_gb', 0):5.3f} GB, "
                f"GPU: {result['summary'].get('avg_gpu_memory_per_env_gb', 0):5.3f} GB) | "
                f"Max envs: {result['summary']['recommended_max_envs_per_gpu']:4d}"
            )

    if summary_count == 0:
        logger.warning(
            "  No architectures successfully completed profiling with valid measurements."
        )
        logger.warning(
            "  This may be because only 1 environment was tested (need at least 2 for per-env calculation)."
        )


if __name__ == "__main__":
    main()
