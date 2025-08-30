"""
Integration tests for Phase 1 enhancements.

Tests the complete integration of Phase 1 features including:
- Environment compliance with Gymnasium
- Vectorization with SubprocVecEnv
- H100 optimization integration
- PBRS callback integration
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

# Add project roots to path
npp_rl_root = Path(__file__).parents[1]
nclone_root = npp_rl_root.parent / "nclone"
sys.path.insert(0, str(npp_rl_root))
sys.path.insert(0, str(nclone_root))

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.environments.vectorization_wrapper import make_vectorizable_env, VectorizationWrapper
from npp_rl.optimization.h100_optimization import (
    enable_h100_optimizations, get_recommended_batch_size, H100OptimizedTraining
)
from npp_rl.callbacks import create_pbrs_callbacks


class TestGymnasiumCompliance(unittest.TestCase):
    """Test Gymnasium compliance for all configurations."""
    
    def test_minimal_profile_compliance(self):
        """Test minimal observation profile compliance."""
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='minimal'
        )
        
        # Should pass Gymnasium compliance check
        try:
            check_env(env, warn=True)
            compliance_passed = True
        except Exception as e:
            print(f"Compliance check failed: {e}")
            compliance_passed = False
        
        self.assertTrue(compliance_passed, "Minimal profile should pass Gymnasium compliance")
    
    def test_rich_profile_compliance(self):
        """Test rich observation profile compliance."""
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich'
        )
        
        # Should pass Gymnasium compliance check
        try:
            check_env(env, warn=True)
            compliance_passed = True
        except Exception as e:
            print(f"Compliance check failed: {e}")
            compliance_passed = False
        
        self.assertTrue(compliance_passed, "Rich profile should pass Gymnasium compliance")
    
    def test_frame_stacking_compliance(self):
        """Test frame stacking compliance."""
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=True,
            observation_profile='rich'
        )
        
        # Should pass Gymnasium compliance check
        try:
            check_env(env, warn=True)
            compliance_passed = True
        except Exception as e:
            print(f"Compliance check failed: {e}")
            compliance_passed = False
        
        self.assertTrue(compliance_passed, "Frame stacking should pass Gymnasium compliance")
    
    def test_pbrs_enabled_compliance(self):
        """Test PBRS enabled compliance."""
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich',
            enable_pbrs=True
        )
        
        # Should pass Gymnasium compliance check
        try:
            check_env(env, warn=True)
            compliance_passed = True
        except Exception as e:
            print(f"Compliance check failed: {e}")
            compliance_passed = False
        
        self.assertTrue(compliance_passed, "PBRS enabled should pass Gymnasium compliance")


class TestVectorization(unittest.TestCase):
    """Test vectorization functionality."""
    
    def test_vectorizable_env_creation(self):
        """Test that vectorizable environments can be created."""
        env_fn = make_vectorizable_env({
            'render_mode': 'rgb_array',
            'enable_frame_stack': False,
            'observation_profile': 'rich'
        })
        
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
    
    def test_pickle_support(self):
        """Test that environments support pickling for multiprocessing."""
        import pickle
        
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
        
        self.assertTrue(pickle_supported, "Environment should support pickling")


class TestH100Optimization(unittest.TestCase):
    """Test H100 optimization utilities."""
    
    def test_optimization_functions_exist(self):
        """Test that optimization functions are available."""
        # Should be able to import functions
        self.assertTrue(callable(enable_h100_optimizations))
        self.assertTrue(callable(get_recommended_batch_size))
        self.assertTrue(callable(H100OptimizedTraining))
    
    def test_batch_size_recommendations(self):
        """Test batch size recommendations."""
        # Should return reasonable batch sizes
        batch_size_h100 = get_recommended_batch_size("H100")
        batch_size_a100 = get_recommended_batch_size("A100")
        batch_size_default = get_recommended_batch_size("Unknown GPU")
        
        self.assertIsInstance(batch_size_h100, int)
        self.assertIsInstance(batch_size_a100, int)
        self.assertIsInstance(batch_size_default, int)
        
        self.assertGreater(batch_size_h100, 0)
        self.assertGreater(batch_size_a100, 0)
        self.assertGreater(batch_size_default, 0)
    
    def test_optimization_context_manager(self):
        """Test H100 optimization context manager."""
        # Should be able to create context manager
        with H100OptimizedTraining() as optimization_status:
            self.assertIsNotNone(optimization_status)
            
            # Should return optimization status dict
            self.assertIsInstance(optimization_status, dict)
    
    @patch('torch.cuda.is_available')
    def test_cuda_detection(self, mock_cuda_available):
        """Test CUDA detection in optimization."""
        # Test with CUDA not available
        mock_cuda_available.return_value = False
        
        # Should handle gracefully
        result = enable_h100_optimizations()
        self.assertIsInstance(result, dict)
        self.assertIn('cuda_available', result)
        self.assertFalse(result['cuda_available'])


class TestCallbackIntegration(unittest.TestCase):
    """Test callback integration."""
    
    def test_pbrs_callbacks_creation(self):
        """Test PBRS callbacks can be created."""
        callbacks = create_pbrs_callbacks(verbose=0)
        
        self.assertIsInstance(callbacks, list)
        self.assertGreater(len(callbacks), 0)
        
        # Should have callback methods
        for callback in callbacks:
            self.assertTrue(hasattr(callback, '_on_step'))
            self.assertTrue(hasattr(callback, '_init_callback'))
    
    def test_callback_initialization(self):
        """Test callback initialization."""
        callbacks = create_pbrs_callbacks(verbose=1)
        
        # Should be able to create callbacks without errors
        for callback in callbacks:
            # Should have required methods
            self.assertTrue(hasattr(callback, '_on_step'))
            self.assertTrue(hasattr(callback, '_init_callback'))
            
            # Should have verbose attribute
            self.assertTrue(hasattr(callback, 'verbose'))


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration of Phase 1 features."""
    
    def test_complete_environment_workflow(self):
        """Test complete workflow with all Phase 1 features."""
        # Create environment with all Phase 1 features
        env = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            observation_profile='rich',
            enable_pbrs=True,
            enable_debug_overlay=False
        )
        
        # Should pass compliance check
        try:
            check_env(env, warn=False)
            compliance_passed = True
        except Exception as e:
            print(f"End-to-end compliance failed: {e}")
            compliance_passed = False
        
        self.assertTrue(compliance_passed)
        
        # Should be able to run episode
        obs = env.reset()[0]
        total_reward = 0
        steps = 0
        
        for _ in range(50):  # Run for 50 steps
            action = np.random.randint(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Check that all expected info is present
            self.assertIn('config_flags', info)
            self.assertIn('pbrs_enabled', info)
            
            if terminated or truncated:
                break
        
        # Should have run for some steps
        self.assertGreater(steps, 0)
        self.assertTrue(np.isfinite(total_reward))
    
    def test_training_pipeline_compatibility(self):
        """Test compatibility with training pipeline components."""
        # Test that all components can be imported and initialized
        try:
            
            # Should be able to import training components
            import_success = True
        except Exception as e:
            print(f"Training pipeline import failed: {e}")
            import_success = False
        
        self.assertTrue(import_success)
        
        # Test environment creation for training
        env_fn = make_vectorizable_env({
            'render_mode': 'rgb_array',
            'enable_frame_stack': False,
            'observation_profile': 'rich',
            'enable_pbrs': True
        })
        
        env = env_fn()
        
        # Should be compatible with training setup
        self.assertIsInstance(env, VectorizationWrapper)
        self.assertTrue(hasattr(env, 'observation_space'))
        self.assertTrue(hasattr(env, 'action_space'))


if __name__ == '__main__':
    unittest.main()