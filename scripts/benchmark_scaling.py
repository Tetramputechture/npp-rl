#!/usr/bin/env python3
"""
Benchmark SubprocVecEnv scaling to confirm pickle serialization bottleneck.
Tests with 5, 8, 16, 32 environments to measure degradation.
"""

import time
import numpy as np
import sys
from pathlib import Path
import argparse

# Add nclone to path
nclone_path = Path(__file__).parent.parent.parent / "nclone"
sys.path.insert(0, str(nclone_path))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from nclone.gym_environment import NppEnvironment, EnvironmentConfig


def make_env(rank=0, level_file=None):
    """Create a single environment."""
    def _init():
        config = EnvironmentConfig()
        config.custom_map_path = level_file or str(nclone_path / "nclone/test-single-level/000 the basics")
        config.enable_visual_observations = False  # Use attention config (no vision)
        config.augmentation.enable_augmentation = False
        return NppEnvironment(config)
    return _init


def benchmark_vecenv(vec_env_type: str, num_envs: int, num_steps: int = 200, level_file=None):
    """Quick benchmark of VecEnv performance."""
    
    print(f"\n{'='*60}")
    print(f"{vec_env_type} with {num_envs} environments")
    print(f"{'='*60}")
    
    # Create environment
    env_class = SubprocVecEnv if vec_env_type == "SubprocVecEnv" else DummyVecEnv
    
    print("Creating environment...")
    start_create = time.time()
    if vec_env_type == "SubprocVecEnv":
        env_fns = [make_env(i, level_file) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env_fns = [make_env(i, level_file) for i in range(num_envs)]
        vec_env = DummyVecEnv(env_fns)
    create_time = time.time() - start_create
    print(f"  Created in {create_time:.2f}s")
    
    # Reset
    print("Resetting...")
    start_reset = time.time()
    obs = vec_env.reset()
    reset_time = time.time() - start_reset
    print(f"  Reset in {reset_time:.2f}s")
    
    # Measure observation size
    if isinstance(obs, dict):
        total_bytes = sum(
            v.nbytes for v in obs.values() 
            if isinstance(v, np.ndarray)
        )
        obs_size_mb = total_bytes / (1024 * 1024)
        print(f"  Observation size: {obs_size_mb:.2f} MB ({obs_size_mb/num_envs:.2f} MB per env)")
    
    # Warm up (10 steps)
    for _ in range(10):
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
    
    # Benchmark
    print(f"Benchmarking {num_steps} steps...")
    step_times = []
    
    start_total = time.time()
    for step in range(num_steps):
        start_step = time.time()
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        step_times.append(time.time() - start_step)
    
    total_time = time.time() - start_total
    
    # Results
    avg_step_time = np.mean(step_times)
    throughput = (num_steps * num_envs) / total_time
    
    print("\nResults:")
    print(f"  Throughput: {throughput:.1f} steps/s")
    print(f"  Avg step time: {avg_step_time*1000:.2f} ms")
    print(f"  Per-env time: {avg_step_time*1000/num_envs:.2f} ms")
    
    vec_env.close()
    
    return {
        'num_envs': num_envs,
        'throughput': throughput,
        'avg_step_time_ms': avg_step_time * 1000,
        'per_env_time_ms': avg_step_time * 1000 / num_envs,
        'obs_size_mb': obs_size_mb if isinstance(obs, dict) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark VecEnv scaling')
    parser.add_argument('--env-counts', type=int, nargs='+', default=[5, 8, 16, 32],
                      help='Environment counts to test (default: 5 8 16 32)')
    parser.add_argument('--num-steps', type=int, default=200,
                      help='Number of steps per benchmark (default: 200)')
    parser.add_argument('--level-file', type=str, default=None,
                      help='Path to level file (default: use test level)')
    args = parser.parse_args()
    
    level_file = args.level_file or str(nclone_path / "nclone/test-single-level/000 the basics")
    
    print("\n" + "="*80)
    print("SubprocVecEnv Scaling Benchmark")
    print("="*80)
    print(f"Testing with: {args.env_counts} environments")
    print(f"Steps per test: {args.num_steps}")
    print(f"Level: {level_file}")
    
    # Benchmark SubprocVecEnv at different scales
    results = []
    for num_envs in args.env_counts:
        result = benchmark_vecenv("SubprocVecEnv", num_envs, args.num_steps, level_file)
        results.append(result)
        time.sleep(1)  # Brief pause between tests
    
    # Analysis
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n{'Envs':<6} {'Throughput':<15} {'Step Time':<15} {'Per-Env':<15} {'Obs Size':<15}")
    print(f"{'':>6} {'(steps/s)':<15} {'(ms)':<15} {'(ms)':<15} {'(MB)':<15}")
    print("-" * 80)
    
    baseline = results[0]
    for result in results:
        throughput_ratio = result['throughput'] / baseline['throughput']
        ideal_ratio = result['num_envs'] / baseline['num_envs']
        efficiency = (throughput_ratio / ideal_ratio) * 100
        
        print(f"{result['num_envs']:<6} "
              f"{result['throughput']:<15.1f} "
              f"{result['avg_step_time_ms']:<15.2f} "
              f"{result['per_env_time_ms']:<15.2f} "
              f"{result['obs_size_mb']:<15.2f}")
        
        if result != baseline:
            print(f"       "
                  f"({throughput_ratio:.2f}x vs {baseline['num_envs']} envs, "
                  f"{efficiency:.0f}% efficiency)")
    
    # Bottleneck diagnosis
    print(f"\n{'='*80}")
    print("BOTTLENECK DIAGNOSIS")
    print(f"{'='*80}")
    
    # Check if throughput scales linearly
    worst_efficiency = min(
        (r['throughput'] / baseline['throughput']) / (r['num_envs'] / baseline['num_envs'])
        for r in results[1:]
    ) * 100
    
    if worst_efficiency < 50:
        print("❌ SEVERE DEGRADATION DETECTED")
        print(f"   Worst efficiency: {worst_efficiency:.0f}%")
        print("   → Pickle serialization is bottleneck")
        print(f"   → Observation size: {results[0]['obs_size_mb']:.2f} MB")
        print("   → Recommendation: Implement SharedMemoryVecEnv")
    elif worst_efficiency < 70:
        print("⚠️  MODERATE DEGRADATION DETECTED")
        print(f"   Worst efficiency: {worst_efficiency:.0f}%")
        print("   → Some serialization overhead")
        print("   → Consider DummyVecEnv for <16 envs")
    else:
        print("✅ GOOD SCALING")
        print(f"   Worst efficiency: {worst_efficiency:.0f}%")
        print("   → SubprocVecEnv scales well")
    
    # Estimate serialization overhead
    first = results[0]
    last = results[-1]
    serialization_overhead = (last['per_env_time_ms'] - first['per_env_time_ms'])
    print(f"\nSerialization overhead per env: ~{serialization_overhead:.2f} ms")
    print(f"At {last['num_envs']} envs: {serialization_overhead * last['num_envs']:.2f} ms/step")
    
    # Extrapolate to 128 envs
    if last['num_envs'] < 128:
        print(f"\n{'='*80}")
        print("EXTRAPOLATION TO 128 ENVIRONMENTS")
        print(f"{'='*80}")
        
        # Linear extrapolation (pessimistic)
        linear_step_time = last['avg_step_time_ms'] * (128 / last['num_envs'])
        linear_throughput = (128 * 1000) / linear_step_time
        
        print("Linear extrapolation:")
        print(f"  Expected step time: {linear_step_time:.2f} ms")
        print(f"  Expected throughput: {linear_throughput:.1f} steps/s")
        print(f"  Expected iterations/s: {linear_throughput/128:.1f} it/s")
        
        if linear_throughput / 128 < 50:
            print("\n❌ WILL NOT MEET 100 it/s TARGET")
            print("   → MUST implement SharedMemoryVecEnv")
        else:
            print("\n✅ May meet target with optimization")


if __name__ == "__main__":
    main()

