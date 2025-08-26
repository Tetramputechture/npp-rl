#!/usr/bin/env python3
"""
Test script for SubprocVecEnv stability and performance measurement.
Tests different numbers of parallel environments and measures throughput.
"""

import os
import time
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from npp_rl.environments.vectorization_wrapper import make_vectorizable_env

# Set headless mode
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def test_vectorization(n_envs_list=[8, 16, 32, 64], steps_per_test=1000):
    """Test SubprocVecEnv stability with different numbers of environments.
    
    Args:
        n_envs_list: List of environment counts to test
        steps_per_test: Number of steps to run for each test
    """
    
    # Environment configuration with Phase 1 enhancements
    env_kwargs = {
        'render_mode': 'rgb_array',
        'enable_frame_stack': False,  # Disable for faster testing
        'observation_profile': 'rich',
        'enable_pbrs': True,
        'pbrs_weights': {
            'objective_weight': 1.0,
            'hazard_weight': 0.5, 
            'impact_weight': 0.3,
            'exploration_weight': 0.2
        },
        'pbrs_gamma': 0.99
    }
    
    print("Testing SubprocVecEnv stability and performance...")
    print(f"Environment config: {env_kwargs}")
    print(f"Steps per test: {steps_per_test}")
    print("-" * 60)
    
    results = {}
    
    for n_envs in n_envs_list:
        print(f"\nTesting with {n_envs} environments...")
        
        try:
            # Create vectorized environment using wrapper
            def make_env():
                return make_vectorizable_env(env_kwargs)
            
            vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            
            # Reset environments
            obs = vec_env.reset()
            print(f"  ✅ Successfully created and reset {n_envs} environments")
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else 'Dict space'}")
            
            # Measure performance
            start_time = time.time()
            total_reward = 0
            
            for step in range(steps_per_test):
                # Random actions
                actions = np.random.randint(0, 6, size=n_envs)
                obs, rewards, dones, infos = vec_env.step(actions)
                total_reward += np.sum(rewards)
                
                # Reset any done environments
                if np.any(dones):
                    print(f"  Episode completed at step {step}")
            
            end_time = time.time()
            elapsed = end_time - start_time
            steps_per_sec = (steps_per_test * n_envs) / elapsed
            
            print(f"  ✅ Completed {steps_per_test} steps successfully")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Time elapsed: {elapsed:.2f}s")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print(f"  Steps/sec per env: {steps_per_sec/n_envs:.1f}")
            
            results[n_envs] = {
                'success': True,
                'steps_per_sec': steps_per_sec,
                'steps_per_sec_per_env': steps_per_sec / n_envs,
                'total_reward': total_reward,
                'elapsed_time': elapsed
            }
            
            # Clean up
            vec_env.close()
            
        except Exception as e:
            print(f"  ❌ Failed with {n_envs} environments: {e}")
            results[n_envs] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("VECTORIZATION TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = [n for n, r in results.items() if r['success']]
    failed_tests = [n for n, r in results.items() if not r['success']]
    
    if successful_tests:
        print(f"✅ Successful tests: {successful_tests}")
        print("\nPerformance Summary:")
        print("Envs | Steps/sec | Steps/sec/env | Total Reward")
        print("-" * 50)
        for n_envs in successful_tests:
            r = results[n_envs]
            print(f"{n_envs:4d} | {r['steps_per_sec']:9.1f} | {r['steps_per_sec_per_env']:13.1f} | {r['total_reward']:12.4f}")
    
    if failed_tests:
        print(f"\n❌ Failed tests: {failed_tests}")
        for n_envs in failed_tests:
            print(f"  {n_envs} envs: {results[n_envs]['error']}")
    
    return results

def test_picklability():
    """Test that BasicLevelNoGold is picklable for SubprocVecEnv."""
    import pickle
    
    print("\nTesting environment picklability...")
    
    env_kwargs = {
        'render_mode': 'rgb_array',
        'enable_frame_stack': False,
        'observation_profile': 'rich',
        'enable_pbrs': True
    }
    
    try:
        env = make_vectorizable_env(env_kwargs)
        
        # Test pickling
        pickled = pickle.dumps(env)
        unpickled_env = pickle.loads(pickled)
        
        print("✅ Environment is picklable")
        
        # Test that unpickled environment works
        try:
            obs, info = unpickled_env.reset()
            obs, reward, terminated, truncated, info = unpickled_env.step(0)
        except Exception as e:
            print(f"❌ Unpickled environment failed during operation: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("✅ Unpickled environment works correctly")
        
        env.close()
        unpickled_env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Picklability test failed: {e}")
        return False

if __name__ == "__main__":
    # Test picklability first
    pickle_success = test_picklability()
    
    if pickle_success:
        # Test vectorization with different environment counts
        results = test_vectorization(n_envs_list=[8, 16, 32], steps_per_test=500)
        
        # Check if we can handle at least 32 environments
        if 32 in results and results[32]['success']:
            print("\n🎉 SubprocVecEnv test PASSED - can handle ≥32 environments")
        else:
            print("\n⚠️  SubprocVecEnv test PARTIAL - issues with 32+ environments")
    else:
        print("\n❌ Cannot proceed with vectorization tests due to pickling issues")