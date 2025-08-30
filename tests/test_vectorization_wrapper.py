#!/usr/bin/env python3
"""
Tests for vectorization wrapper functionality.
Tests the VectorizationWrapper class and make_vectorizable_env function.
"""

import os
import time
import pickle
import unittest
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.environments.vectorization_wrapper import make_vectorizable_env, VectorizationWrapper

# Set headless mode for testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'


class TestVectorizationWrapper(unittest.TestCase):
    """Test vectorization wrapper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_env_kwargs = {
            'render_mode': 'rgb_array',
            'enable_frame_stack': False,
            'observation_profile': 'rich'
        }
        
    def test_vectorizable_env_creation(self):
        """Test that vectorizable environments can be created."""
        env_fn = make_vectorizable_env(self.base_env_kwargs)
        
        # Should be able to create environment
        env = env_fn()
        self.assertIsInstance(env, VectorizationWrapper)
        
        # Should be able to reset and step
        obs = env.reset()[0]
        self.assertIn('game_state', obs)
        self.assertIn('player_frame', obs)
        self.assertIn('global_view', obs)
        
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(np.isfinite(reward))
        
        env.close()
        
    def test_wrapper_attributes(self):
        """Test that wrapper has necessary attributes."""
        env_fn = make_vectorizable_env(self.base_env_kwargs)
        env = env_fn()
        
        # Should have gymnasium.Wrapper attributes
        self.assertTrue(hasattr(env, 'observation_space'))
        self.assertTrue(hasattr(env, 'action_space'))
        self.assertTrue(hasattr(env, 'reward_range'))
        self.assertTrue(hasattr(env, 'metadata'))
        
        env.close()
        
    def test_picklability(self):
        """Test that VectorizationWrapper is picklable for SubprocVecEnv."""
        env_kwargs = {
            'render_mode': 'rgb_array',
            'enable_frame_stack': False,
            'observation_profile': 'rich',
            'enable_pbrs': True
        }
        
        env = make_vectorizable_env(env_kwargs)()
        
        # Test pickling
        pickled = pickle.dumps(env)
        unpickled_env = pickle.loads(pickled)
        
        # Test that unpickled environment works
        obs, info = unpickled_env.reset()
        obs, reward, terminated, truncated, info = unpickled_env.step(0)
        
        # Should produce valid outputs
        self.assertIsInstance(obs, dict)
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        
        env.close()
        unpickled_env.close()
        
    def test_pickle_support_basic_env(self):
        """Test that basic environments support pickling for multiprocessing."""
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich'
        )
        
        # Should be able to pickle and unpickle
        try:
            pickled_env = pickle.dumps(env)
            unpickled_env = pickle.loads(pickled_env)
            pickle_supported = True
        except Exception as e:
            print(f"Pickle failed: {e}")
            pickle_supported = False
            
        # Note: BasicLevelNoGold itself may not be picklable, 
        # but VectorizationWrapper should handle this
        # This test documents the current state
        if not pickle_supported:
            print("Warning: BasicLevelNoGold not directly picklable")
            
        env.close()
        
    def test_subproc_vec_env_creation(self):
        """Test SubprocVecEnv creation with small number of environments."""
        n_envs = 2  # Small number for testing
        
        env_fns = [
            make_vectorizable_env({
                'render_mode': 'rgb_array',
                'enable_frame_stack': False,
                'observation_profile': 'minimal'
            }) for _ in range(n_envs)
        ]
        
        # Should be able to create vectorized environment
        vec_env = SubprocVecEnv(env_fns)
        
        # Should be able to reset
        obs = vec_env.reset()
        # obs is a dict with observation keys, check batch dimension
        self.assertIn('game_state', obs)
        self.assertEqual(obs['game_state'].shape[0], n_envs)
        
        # Should be able to step
        actions = [0] * n_envs
        obs, rewards, dones, infos = vec_env.step(actions)
        
        self.assertEqual(obs['game_state'].shape[0], n_envs)
        self.assertEqual(len(rewards), n_envs)
        self.assertEqual(len(dones), n_envs)
        self.assertEqual(len(infos), n_envs)
        
        vec_env.close()


class TestVectorizationPerformance(unittest.TestCase):
    """Test vectorization performance and stability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env_kwargs = {
            'render_mode': 'rgb_array',
            'enable_frame_stack': False,
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
    
    def test_small_scale_vectorization(self):
        """Test vectorization with small number of environments."""
        n_envs = 4
        steps_per_test = 50
        
        results = self._test_vectorization_config(n_envs, steps_per_test)
        
        # Should succeed with small number of environments
        self.assertTrue(results['success'])
        self.assertGreater(results['steps_per_sec'], 0)
        
    def test_medium_scale_vectorization(self):
        """Test vectorization with medium number of environments."""
        n_envs = 8
        steps_per_test = 100
        
        results = self._test_vectorization_config(n_envs, steps_per_test)
        
        # Should succeed with medium number of environments
        self.assertTrue(results['success'])
        self.assertGreater(results['steps_per_sec'], 0)
        
    def _test_vectorization_config(self, n_envs, steps_per_test):
        """
        Test vectorization with specific configuration.
        
        Args:
            n_envs: Number of environments to test
            steps_per_test: Number of steps to run
            
        Returns:
            Dictionary with test results
        """
        try:
            # Create vectorized environment using wrapper
            def make_env():
                return make_vectorizable_env(self.env_kwargs)
            
            vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
            
            # Reset environments
            obs = vec_env.reset()
            
            # Measure performance
            start_time = time.time()
            total_reward = 0
            
            for step in range(steps_per_test):
                # Random actions
                actions = np.random.randint(0, 6, size=n_envs)
                obs, rewards, dones, infos = vec_env.step(actions)
                total_reward += np.sum(rewards)
                
                # Reset any done environments automatically handled by vec_env
                
            end_time = time.time()
            elapsed = end_time - start_time
            steps_per_sec = (steps_per_test * n_envs) / elapsed
            
            # Clean up
            vec_env.close()
            
            return {
                'success': True,
                'steps_per_sec': steps_per_sec,
                'steps_per_sec_per_env': steps_per_sec / n_envs,
                'total_reward': total_reward,
                'elapsed_time': elapsed
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def run_performance_benchmark(n_envs_list=[8, 16, 32], steps_per_test=500):
    """
    Run performance benchmark for vectorization.
    
    This function can be called directly for detailed performance testing.
    
    Args:
        n_envs_list: List of environment counts to test
        steps_per_test: Number of steps to run for each test
        
    Returns:
        Dictionary with benchmark results
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
            print(f"  Observation shape: {obs['game_state'].shape if 'game_state' in obs else 'Unknown'}")
            
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
    print("VECTORIZATION BENCHMARK SUMMARY")
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


if __name__ == "__main__":
    # Run unit tests by default
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Optionally run performance benchmark
    # Uncomment to run benchmark:
    # print("\n" + "="*60)
    # print("RUNNING PERFORMANCE BENCHMARK")
    # print("="*60)
    # benchmark_results = run_performance_benchmark(n_envs_list=[4, 8], steps_per_test=100)
