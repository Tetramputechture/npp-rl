#!/usr/bin/env python3
"""
Detailed profiling of DummyVecEnv vs SubprocVecEnv to identify bottlenecks.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add nclone to path
nclone_path = Path(__file__).parent.parent.parent / "nclone"
sys.path.insert(0, str(nclone_path))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from nclone.gym_environment import NppEnvironment, EnvironmentConfig


def make_env(rank=0, level_file=None, test_dataset_path=None):
    """Create a single environment."""
    def _init():
        config = EnvironmentConfig()
        config.custom_map_path = level_file or str(nclone_path / "nclone/test-single-level/000 the basics")
        config.enable_visual_observations = False
        config.augmentation.enable_augmentation = False
        config.test_dataset_path = test_dataset_path  # Use provided test dataset
        return NppEnvironment(config)
    return _init


def profile_vecenv(vec_env_type: str, num_envs: int, num_steps: int = 500, level_file=None, test_dataset_path=None):
    """Profile VecEnv with detailed timing breakdown."""
    
    print(f"\n{'='*80}")
    print(f"PROFILING: {vec_env_type} with {num_envs} environments")
    print(f"{'='*80}\n")
    
    # Create environment
    env_class = SubprocVecEnv if vec_env_type == "SubprocVecEnv" else DummyVecEnv
    
    print(f"Creating {vec_env_type}...")
    start_create = time.time()
    if vec_env_type == "SubprocVecEnv":
        env_fns = [make_env(i, level_file, test_dataset_path) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env_fns = [make_env(i, level_file, test_dataset_path) for i in range(num_envs)]
        vec_env = DummyVecEnv(env_fns)
    create_time = time.time() - start_create
    print(f"Creation time: {create_time:.2f}s\n")
    
    # Reset
    print("Resetting environments...")
    start_reset = time.time()
    obs = vec_env.reset()
    reset_time = time.time() - start_reset
    print(f"Reset time: {reset_time:.2f}s")
    
    # Analyze observation size
    if isinstance(obs, dict):
        total_bytes = 0
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                size_bytes = value.nbytes
                total_bytes += size_bytes
                if size_bytes > 1000:  # Only print large observations
                    print(f"  {key}: {value.shape} {value.dtype} = {size_bytes/1024:.1f} KB")
        print(f"  Total observation size: {total_bytes/1024:.1f} KB\n")
    
    # Warm up (10 steps)
    print("Warming up...")
    for _ in range(10):
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
    
    # Profile step timing
    print(f"\nProfiling {num_steps} steps...")
    step_times = []
    
    start_total = time.time()
    for step in range(num_steps):
        # Time individual step
        start_step = time.time()
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        step_time = time.time() - start_step
        step_times.append(step_time)
        
        # Print progress
        if (step + 1) % 100 == 0:
            avg_time = np.mean(step_times[-100:])
            print(f"  Step {step+1}/{num_steps}: {avg_time*1000:.2f} ms/step ({1/avg_time:.1f} steps/s)")
    
    total_time = time.time() - start_total
    
    # Statistics
    step_times = np.array(step_times)
    print(f"\n{'='*80}")
    print(f"RESULTS: {vec_env_type}")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total steps: {num_steps * num_envs}")
    print(f"Throughput: {(num_steps * num_envs) / total_time:.1f} steps/s")
    print(f"Average step time: {np.mean(step_times)*1000:.2f} ms")
    print(f"Median step time: {np.median(step_times)*1000:.2f} ms")
    print(f"Min/Max step time: {np.min(step_times)*1000:.2f} / {np.max(step_times)*1000:.2f} ms")
    print(f"Step time std: {np.std(step_times)*1000:.2f} ms")
    
    # Check for outliers
    p95 = np.percentile(step_times, 95)
    p99 = np.percentile(step_times, 99)
    print(f"P95 step time: {p95*1000:.2f} ms")
    print(f"P99 step time: {p99*1000:.2f} ms")
    
    if p99 > 2 * np.median(step_times):
        outliers = np.sum(step_times > 2 * np.median(step_times))
        print(f"⚠️  WARNING: {outliers} outlier steps detected (>2x median)")
    
    vec_env.close()
    
    return {
        'throughput': (num_steps * num_envs) / total_time,
        'avg_step_time': np.mean(step_times),
        'median_step_time': np.median(step_times),
        'p95_step_time': p95,
        'p99_step_time': p99,
    }


def main():
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    test_dataset_path = str(nclone_path / "datasets/test")
    num_envs = 5
    num_steps = 500  # Reduced for faster testing
    
    print("\n" + "="*80)
    print("VecEnv Performance Comparison")
    print("="*80)
    
    # Test DummyVecEnv
    dummy_results = profile_vecenv("DummyVecEnv", num_envs, num_steps, level_file, test_dataset_path)
    
    # Test SubprocVecEnv
    subproc_results = profile_vecenv("SubprocVecEnv", num_envs, num_steps, level_file, test_dataset_path)
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    throughput_ratio = dummy_results['throughput'] / subproc_results['throughput']
    print(f"DummyVecEnv throughput: {dummy_results['throughput']:.1f} steps/s")
    print(f"SubprocVecEnv throughput: {subproc_results['throughput']:.1f} steps/s")
    print(f"Ratio: {throughput_ratio:.2f}x {'faster' if throughput_ratio > 1 else 'slower'}")
    
    step_time_ratio = subproc_results['avg_step_time'] / dummy_results['avg_step_time']
    print(f"\nDummyVecEnv avg step time: {dummy_results['avg_step_time']*1000:.2f} ms")
    print(f"SubprocVecEnv avg step time: {subproc_results['avg_step_time']*1000:.2f} ms")
    print(f"Overhead: {(step_time_ratio - 1) * 100:.1f}% slower")
    
    # Breakdown
    overhead_ms = (subproc_results['avg_step_time'] - dummy_results['avg_step_time']) * 1000
    print(f"\nSubprocVecEnv overhead per step: {overhead_ms:.2f} ms")
    print("  → Likely due to: IPC, serialization/deserialization, process synchronization")
    
    if overhead_ms > 50:
        print(f"\n⚠️  WARNING: Large overhead detected ({overhead_ms:.0f} ms/step)")
        print("   This suggests inefficient IPC or large observation serialization.")
        print("   Recommendation: Use DummyVecEnv or implement SharedMemoryVecEnv.")


if __name__ == "__main__":
    main()

