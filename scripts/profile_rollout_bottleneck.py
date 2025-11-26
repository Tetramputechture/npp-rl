#!/usr/bin/env python3
"""
Profile rollout collection to identify exact bottleneck with SubprocVecEnv.
Uses minimal setup to isolate the issue.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add paths
nclone_path = Path(__file__).parent.parent.parent / "nclone"
sys.path.insert(0, str(nclone_path))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from nclone.gym_environment import NppEnvironment, EnvironmentConfig


def make_env(rank, level_file, test_dataset):
    """Create environment function."""
    def _init():
        config = EnvironmentConfig()
        config.custom_map_path = level_file
        config.enable_visual_observations = False
        config.augmentation.enable_augmentation = False
        config.test_dataset_path = test_dataset
        return NppEnvironment(config)
    return _init


def profile_rollout_raw(vec_env_type: str, num_envs: int, num_iterations: int = 100):
    """Profile raw rollout collection without SparseGraphRolloutBuffer complexity."""
    
    print(f"\n{'='*80}")
    print(f"RAW ROLLOUT PROFILING: {vec_env_type} with {num_envs} environments")
    print(f"{'='*80}\n")
    
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    test_dataset = str(nclone_path / "datasets/test")
    
    # Create VecEnv
    print(f"Creating {vec_env_type}...")
    create_start = time.time()
    
    env_fns = [make_env(i, level_file, test_dataset) for i in range(num_envs)]
    
    if vec_env_type == "SubprocVecEnv":
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    
    create_time = time.time() - create_start
    print(f"  Creation time: {create_time:.2f}s")
    
    # Reset
    print("Resetting...")
    reset_start = time.time()
    obs = vec_env.reset()
    reset_time = time.time() - reset_start
    print(f"  Reset time: {reset_time:.2f}s")
    
    # Check observation size and structure
    print("\nObservation structure:")
    if isinstance(obs, dict):
        total_mb = 0
        for key, value in sorted(obs.items()):
            if isinstance(value, np.ndarray):
                size_mb = value.nbytes / (1024 * 1024)
                total_mb += size_mb
                print(f"  {key:35s}: {str(value.shape):25s} {str(value.dtype):10s} = {size_mb:.2f} MB")
        print(f"  {'TOTAL':35s}: {total_mb:.2f} MB")
        print(f"  Per environment: {total_mb / num_envs:.2f} MB")
    
    # Profile individual steps with detailed breakdown
    print(f"\nProfiling {num_iterations} iterations (step + random action)...")
    
    step_times = []
    action_gen_times = []
    
    for i in range(num_iterations):
        iter_start = time.perf_counter()
        
        # Generate random actions
        action_start = time.perf_counter()
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        action_gen_times.append(time.perf_counter() - action_start)
        
        # Step
        step_start = time.perf_counter()
        obs, rewards, dones, infos = vec_env.step(actions)
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if (i + 1) % 20 == 0:
            recent_steps = step_times[-20:]
            avg_step = np.mean(recent_steps) * 1000
            min_step = np.min(recent_steps) * 1000
            max_step = np.max(recent_steps) * 1000
            print(f"  Iter {i+1:3d}: {avg_step:6.2f}ms avg (min: {min_step:6.2f}ms, max: {max_step:6.2f}ms)")
    
    # Statistics
    step_times = np.array(step_times)
    action_gen_times = np.array(action_gen_times)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {vec_env_type} with {num_envs} environments")
    print(f"{'='*80}")
    print(f"Step time (avg): {np.mean(step_times)*1000:.2f}ms")
    print(f"Step time (median): {np.median(step_times)*1000:.2f}ms")
    print(f"Step time (p95): {np.percentile(step_times, 95)*1000:.2f}ms")
    print(f"Step time (p99): {np.percentile(step_times, 99)*1000:.2f}ms")
    print(f"Step time (std): {np.std(step_times)*1000:.2f}ms")
    print(f"\nThroughput: {num_iterations / np.sum(step_times):.1f} it/s")
    print(f"Steps per second: {(num_iterations * num_envs) / np.sum(step_times):.1f} steps/s")
    
    # Check for variance issues
    if np.std(step_times) > 0.01:  # >10ms std
        print(f"\n⚠️  High variance detected: {np.std(step_times)*1000:.2f}ms std")
        print(f"   Min: {np.min(step_times)*1000:.2f}ms, Max: {np.max(step_times)*1000:.2f}ms")
        print(f"   Range: {(np.max(step_times) - np.min(step_times))*1000:.2f}ms")
    
    vec_env.close()
    
    return {
        "throughput_it_per_sec": num_iterations / np.sum(step_times),
        "avg_step_time_ms": np.mean(step_times) * 1000,
        "median_step_time_ms": np.median(step_times) * 1000,
        "observation_size_mb": total_mb if isinstance(obs, dict) else 0,
    }


def main():
    print("\n" + "="*80)
    print("ROLLOUT BOTTLENECK PROFILING")
    print("="*80)
    print("Goal: Identify exact cause of SubprocVecEnv slowdown in training")
    print("Target: 100+ it/s with 128+ environments on 80GB VRAM\n")
    
    # Test various configurations
    configs = [
        ("DummyVecEnv", 5),
        ("SubprocVecEnv", 5),
        ("DummyVecEnv", 8),
        ("SubprocVecEnv", 8),
        ("SubprocVecEnv", 16),
    ]
    
    results = {}
    
    for vec_env_type, num_envs in configs:
        try:
            print(f"\n{'='*80}")
            results[(vec_env_type, num_envs)] = profile_rollout_raw(
                vec_env_type, num_envs, num_iterations=100
            )
            print(f"{'='*80}")
        except Exception as e:
            print(f"\n❌ ERROR with {vec_env_type} {num_envs} envs: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    for (vec_type, n_env), metrics in results.items():
        print(f"{vec_type:20s} {n_env:2d} envs: {metrics['throughput_it_per_sec']:6.1f} it/s "
              f"({metrics['avg_step_time_ms']:5.2f}ms/step, obs: {metrics['observation_size_mb']:.1f} MB)")
    
    # Identify issues
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}\n")
    
    if ("DummyVecEnv", 5) in results and ("SubprocVecEnv", 5) in results:
        dummy = results[("DummyVecEnv", 5)]
        subproc = results[("SubprocVecEnv", 5)]
        ratio = dummy["throughput_it_per_sec"] / subproc["throughput_it_per_sec"]
        
        print("DummyVecEnv vs SubprocVecEnv (5 envs):")
        print(f"  Throughput ratio: {ratio:.2f}x")
        print(f"  Step time difference: {subproc['avg_step_time_ms'] - dummy['avg_step_time_ms']:.2f}ms")
        
        if ratio > 1.2:
            print("\n⚠️  SubprocVecEnv is significantly slower than DummyVecEnv")
            print(f"   Likely cause: Large observations ({dummy['observation_size_mb']:.1f} MB) + pickle serialization")
            print("   Recommendation: Use DummyVecEnv for ≤8 environments")
        elif ratio < 0.8:
            print("\n✅ SubprocVecEnv is faster - good for scaling")
        else:
            print("\n✓ Performance similar - choice depends on scaling needs")


if __name__ == "__main__":
    main()

