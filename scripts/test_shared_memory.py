#!/usr/bin/env python3
"""
Test shared memory vectorized environment for correctness and performance.
"""

import time
import numpy as np
import sys
from pathlib import Path
import argparse

# Add nclone to path
nclone_path = Path(__file__).parent.parent.parent / "nclone"
sys.path.insert(0, str(nclone_path))
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.environment_factory import EnvironmentFactory
from npp_rl.training.architecture_configs import get_architecture_config


def test_correctness(num_envs=5, num_steps=50):
    """Test that shared memory produces correct results."""
    print(f"\n{'='*80}")
    print(f"CORRECTNESS TEST: {num_envs} environments, {num_steps} steps")
    print(f"{'='*80}\n")
    
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    arch_config = get_architecture_config("attention")
    
    # Create factory with shared memory
    factory = EnvironmentFactory(
        use_curriculum=False,
        frame_skip_config={"frame_skip": 4},
        pbrs_gamma=1.0,
        custom_map_path=level_file,  # Use custom map, don't need test dataset
        architecture_config=arch_config,
        use_shared_memory=True,
    )
    
    print("Creating environment with shared memory...")
    env = factory.create_training_env(num_envs=num_envs, gamma=0.99)
    
    print("Resetting...")
    obs = env.reset()
    
    # Verify observation structure
    print("\nObservation keys:", list(obs.keys()))
    print("Observation shapes:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} {value.dtype}")
    
    # Run steps and verify
    print(f"\nRunning {num_steps} steps...")
    for step in range(num_steps):
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        
        # Verify observation validity
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    print(f"❌ ERROR: NaN detected in {key} at step {step}")
                    return False
                if np.any(np.isinf(value)):
                    print(f"❌ ERROR: Inf detected in {key} at step {step}")
                    return False
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps}: ✓")
    
    env.close()
    print("\n✅ Correctness test PASSED")
    return True


def test_performance(num_envs=8, num_steps=200):
    """Test performance with shared memory."""
    print(f"\n{'='*80}")
    print(f"PERFORMANCE TEST: {num_envs} environments")
    print(f"{'='*80}\n")
    
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    arch_config = get_architecture_config("attention")
    
    # Test with shared memory
    factory_shared = EnvironmentFactory(
        use_curriculum=False,
        frame_skip_config={"frame_skip": 4},
        pbrs_gamma=1.0,
        custom_map_path=level_file,  # Use custom map, don't need test dataset
        architecture_config=arch_config,
        use_shared_memory=True,
    )
    
    print("Creating environment WITH shared memory...")
    env_shared = factory_shared.create_training_env(num_envs=num_envs, gamma=0.99)
    
    # Warm up
    obs = env_shared.reset()
    for _ in range(10):
        actions = np.array([env_shared.action_space.sample() for _ in range(num_envs)])
        obs, _, _, _ = env_shared.step(actions)
    
    # Benchmark
    print(f"\nBenchmarking {num_steps} steps...")
    start = time.time()
    for step in range(num_steps):
        actions = np.array([env_shared.action_space.sample() for _ in range(num_envs)])
        obs, _, _, _ = env_shared.step(actions)
    
    elapsed = time.time() - start
    throughput = (num_steps * num_envs) / elapsed
    avg_step_time = elapsed / num_steps
    
    print("\nResults (WITH shared memory):")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} steps/s")
    print(f"  Avg step time: {avg_step_time*1000:.2f} ms")
    print(f"  Iterations/s: {throughput/num_envs:.1f} it/s")
    
    env_shared.close()
    
    return throughput


def test_scaling(env_counts=[5, 8, 16, 32], num_steps=100):
    """Test scaling across different environment counts."""
    print(f"\n{'='*80}")
    print("SCALING TEST")
    print(f"{'='*80}\n")
    
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    arch_config = get_architecture_config("attention")
    
    results = []
    
    for num_envs in env_counts:
        print(f"\n--- Testing {num_envs} environments ---")
        
        factory = EnvironmentFactory(
            use_curriculum=False,
            frame_skip_config={"frame_skip": 4},
            pbrs_gamma=1.0,
            custom_map_path=level_file,  # Use custom map, don't need test dataset
            architecture_config=arch_config,
            use_shared_memory=True,
        )
        
        env = factory.create_training_env(num_envs=num_envs, gamma=0.99)
        
        # Warm up
        obs = env.reset()
        for _ in range(10):
            actions = np.array([env.action_space.sample() for _ in range(num_envs)])
            obs, _, _, _ = env.step(actions)
        
        # Benchmark
        start = time.time()
        for step in range(num_steps):
            actions = np.array([env.action_space.sample() for _ in range(num_envs)])
            obs, _, _, _ = env.step(actions)
        
        elapsed = time.time() - start
        throughput = (num_steps * num_envs) / elapsed
        
        results.append({
            'num_envs': num_envs,
            'throughput': throughput,
            'it_per_sec': throughput / num_envs,
        })
        
        print(f"  Throughput: {throughput:.1f} steps/s ({throughput/num_envs:.1f} it/s)")
        
        env.close()
        time.sleep(1)
    
    # Analysis
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS")
    print(f"{'='*80}")
    print(f"\n{'Envs':<8} {'Throughput':<15} {'It/s':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    baseline = results[0]
    for result in results:
        efficiency = (result['throughput'] / baseline['throughput']) / (result['num_envs'] / baseline['num_envs']) * 100
        
        print(f"{result['num_envs']:<8} "
              f"{result['throughput']:<15.1f} "
              f"{result['it_per_sec']:<10.1f} "
              f"{efficiency:<10.0f}%")
    
    avg_efficiency = np.mean([
        (r['throughput'] / baseline['throughput']) / (r['num_envs'] / baseline['num_envs'])
        for r in results[1:]
    ]) * 100
    
    if avg_efficiency > 70:
        print(f"\n✅ EXCELLENT SCALING (avg efficiency: {avg_efficiency:.0f}%)")
    elif avg_efficiency > 50:
        print(f"\n✓ GOOD SCALING (avg efficiency: {avg_efficiency:.0f}%)")
    else:
        print(f"\n⚠️ POOR SCALING (avg efficiency: {avg_efficiency:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Test shared memory vectorized environment')
    parser.add_argument('--test', type=str, default='all',
                      choices=['correctness', 'performance', 'scaling', 'all'],
                      help='Test to run (default: all)')
    parser.add_argument('--num-envs', type=int, default=8,
                      help='Number of environments for performance test')
    parser.add_argument('--num-steps', type=int, default=100,
                      help='Number of steps for tests')
    args = parser.parse_args()
    
    if args.test in ['correctness', 'all']:
        test_correctness(num_envs=5, num_steps=50)
    
    if args.test in ['performance', 'all']:
        test_performance(num_envs=args.num_envs, num_steps=args.num_steps)
    
    if args.test in ['scaling', 'all']:
        test_scaling(env_counts=[5, 8, 16], num_steps=args.num_steps)


if __name__ == "__main__":
    main()

