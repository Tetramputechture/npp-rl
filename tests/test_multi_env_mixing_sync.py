"""Test that adaptive mixing ratios are properly synced in multi-environment setups.

This test verifies the fix for the critical bug where subprocess environments
used stale performance data to calculate mixing ratios.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from npp_rl.training.curriculum_manager import CurriculumManager
from npp_rl.wrappers.curriculum_env import (
    CurriculumEnv,
    CurriculumVecEnvWrapper,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class MockNppEnvironment(gym.Env):
    """Mock N++ environment for testing."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(6)
        self.step_count = 0
        self.max_steps = 10
        self.episode_count = 0

        # Mock nplay_headless for map loading
        self.nplay_headless = MagicMock()
        self.nplay_headless.load_map_from_map_data = MagicMock()

    def reset(self, **kwargs):
        """Reset environment."""
        self.step_count = 0
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        info = {}
        return obs, info

    def step(self, action):
        """Execute action."""
        self.step_count += 1
        obs = np.zeros((84, 84, 3), dtype=np.uint8)

        # Terminate after max_steps
        terminated = self.step_count >= self.max_steps
        truncated = False
        reward = 1.0 if terminated else 0.0

        # Alternate success/failure
        info = {"is_success": self.episode_count % 5 < 2}  # 40% success rate

        if terminated:
            self.episode_count += 1

        return obs, reward, terminated, truncated, info


def create_mock_dataset(tmp_path):
    """Create a mock dataset for testing."""
    dataset_path = tmp_path / "test_dataset"
    dataset_path.mkdir(exist_ok=True)

    # Create mock levels for each stage
    stages = [
        "simplest",
        "simpler",
        "simple",
        "medium",
        "complex",
        "exploration",
        "mine_heavy",
    ]

    for stage in stages:
        stage_dir = dataset_path / stage
        stage_dir.mkdir(exist_ok=True)

        # Create a few mock level files
        for i in range(3):
            level_file = stage_dir / f"level_{i}.txt"
            level_file.write_text(f"Mock level data for {stage} level {i}")

    return str(dataset_path)


def test_mixing_ratio_sync_initialization():
    """Test that mixing ratios are synced during initialization."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Mixing Ratio Sync on Initialization")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = create_mock_dataset(Path(tmp_dir))

        # Create curriculum manager and establish performance
        curriculum_manager = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage="simpler",  # Start at stage 1 for mixing
            enable_adaptive_mixing=True,
        )

        # Record episodes to establish low performance (40% success)
        # This should result in 40% mixing ratio
        for i in range(30):
            curriculum_manager.record_episode("simpler", i % 5 < 2)

        # Verify main process has correct mixing ratio
        main_ratio = curriculum_manager._get_adaptive_mixing_ratio("simpler")
        logger.info(f"Main process mixing ratio: {main_ratio:.1%}")
        assert main_ratio == 0.40, (
            f"Expected 40% mixing for 40% success, got {main_ratio:.1%}"
        )

        # Create vectorized environments
        def make_env():
            env = MockNppEnvironment()
            return CurriculumEnv(
                env,
                curriculum_manager,
                enable_local_tracking=False,  # Critical: disable local tracking
            )

        n_envs = 4
        venv = DummyVecEnv([make_env for _ in range(n_envs)])
        venv_wrapper = CurriculumVecEnvWrapper(venv, curriculum_manager)

        # After initialization, all subprocess envs should have the synced ratio
        # We can verify this by checking if the ratio is in the stage_mixing_ratios dict
        # In the envs (via accessing the curriculum_manager in each env)

        # Access each env's curriculum manager and check the cached ratio
        for i in range(n_envs):
            env = venv.envs[i]
            # Unwrap to get CurriculumEnv
            while hasattr(env, "env"):
                env = env.env

            if isinstance(env, CurriculumEnv):
                cached_ratio = env.curriculum_manager.stage_mixing_ratios.get("simpler")
                logger.info(
                    f"Env {i} cached mixing ratio: {cached_ratio:.1% if cached_ratio else 'None'}"
                )
                assert cached_ratio is not None, (
                    f"Env {i} should have cached mixing ratio"
                )
                assert cached_ratio == 0.40, (
                    f"Env {i} should have 40% mixing ratio, got {cached_ratio:.1%}"
                )

        logger.info("✓ Mixing ratios successfully synced on initialization")


def test_mixing_ratio_adapts_with_performance():
    """Test that mixing ratios adapt as performance improves and sync to subprocesses."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Mixing Ratio Adapts During Training")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = create_mock_dataset(Path(tmp_dir))

        curriculum_manager = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage="simpler",
            enable_adaptive_mixing=True,
        )

        # Start with low performance (40% success) -> 40% mixing
        np.random.seed(42)
        for i in range(50):
            curriculum_manager.record_episode("simpler", np.random.random() < 0.40)

        ratio_1 = curriculum_manager._get_adaptive_mixing_ratio("simpler")
        logger.info(f"Initial mixing ratio (~40% success): {ratio_1:.1%}")
        assert ratio_1 == 0.40

        # Clear and set to medium performance (55% success) -> 25% mixing
        curriculum_manager.stage_performance["simpler"].clear()
        np.random.seed(43)
        for i in range(50):
            success = np.random.random() < 0.55
            curriculum_manager.record_episode("simpler", success)

        ratio_2 = curriculum_manager._get_adaptive_mixing_ratio("simpler")
        logger.info(f"After improvement (~55% success): {ratio_2:.1%}")
        assert ratio_2 == 0.25, f"Expected 25% mixing, got {ratio_2:.1%}"

        # Create vectorized environments
        def make_env():
            env = MockNppEnvironment()
            return CurriculumEnv(env, curriculum_manager, enable_local_tracking=False)

        venv = DummyVecEnv([make_env for _ in range(4)])
        venv_wrapper = CurriculumVecEnvWrapper(
            venv, curriculum_manager, check_advancement_freq=10
        )

        # The initialization should sync the current ratio (25%)
        for i in range(4):
            env = venv.envs[i]
            while hasattr(env, "env"):
                env = env.env
            if isinstance(env, CurriculumEnv):
                cached_ratio = env.curriculum_manager.stage_mixing_ratios.get("simpler")
                assert cached_ratio == 0.25, (
                    f"Env {i} should have 25% mixing after init"
                )

        # Clear and set to high performance (90% success) -> 5% mixing
        curriculum_manager.stage_performance["simpler"].clear()
        np.random.seed(44)
        for i in range(50):
            success = np.random.random() < 0.90
            curriculum_manager.record_episode("simpler", success)

        ratio_3 = curriculum_manager._get_adaptive_mixing_ratio("simpler")
        logger.info(f"After high performance (~90% success): {ratio_3:.1%}")
        assert ratio_3 == 0.05

        # Manually trigger sync (simulating what happens at advancement check)
        venv_wrapper._sync_mixing_ratios()

        # Verify all envs now have the new ratio
        for i in range(4):
            env = venv.envs[i]
            while hasattr(env, "env"):
                env = env.env
            if isinstance(env, CurriculumEnv):
                cached_ratio = env.curriculum_manager.stage_mixing_ratios.get("simpler")
                logger.info(f"Env {i} ratio after sync: {cached_ratio:.1%}")
                assert cached_ratio == 0.05, f"Env {i} should have 5% mixing after sync"

        logger.info("✓ Mixing ratios adapt and sync correctly")


def test_subprocess_uses_cached_ratio():
    """Test that subprocess environments use cached ratios instead of calculating from stale data."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST: Subprocess Uses Cached Ratio (Not Stale Data)")
    logger.info("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = create_mock_dataset(Path(tmp_dir))

        # Main process curriculum manager with performance data
        curriculum_manager = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage="simpler",
            enable_adaptive_mixing=True,
        )

        # Establish high performance (90% success) -> 5% mixing
        for i in range(50):
            curriculum_manager.record_episode("simpler", np.random.random() < 0.90)

        main_ratio = curriculum_manager._get_adaptive_mixing_ratio("simpler")
        logger.info(f"Main process mixing ratio (90% success): {main_ratio:.1%}")
        assert main_ratio == 0.05

        # Simulate subprocess: create a new curriculum manager with NO performance data
        # This simulates what happens when pickling sends a copy to subprocess
        subprocess_curriculum = CurriculumManager(
            dataset_path=dataset_path,
            starting_stage="simpler",
            enable_adaptive_mixing=True,
        )

        # Without syncing, subprocess would calculate from empty data -> default ratio
        # But with our fix, if we set the cached ratio, it should use that
        subprocess_curriculum.stage_mixing_ratios["simpler"] = 0.05  # Synced value

        # Now when subprocess calls _get_adaptive_mixing_ratio, it should use cached
        subprocess_ratio = subprocess_curriculum._get_adaptive_mixing_ratio("simpler")
        logger.info(
            f"Subprocess mixing ratio (with cached value): {subprocess_ratio:.1%}"
        )

        # It should use the cached value, NOT calculate from empty performance data
        assert subprocess_ratio == 0.05, (
            f"Subprocess should use cached ratio (5%), not calculate from stale data. "
            f"Got {subprocess_ratio:.1%}"
        )

        # Verify subprocess has minimal performance data (simulating the subprocess state)
        perf_count = len(subprocess_curriculum.stage_performance.get("simpler", []))
        logger.info(f"Subprocess performance data count: {perf_count}")
        assert perf_count < 5, "Subprocess should have minimal performance data"

        logger.info(
            "✓ Subprocess correctly uses cached ratio instead of calculating from stale data"
        )


def main():
    """Run all tests."""
    test_functions = [
        test_mixing_ratio_sync_initialization,
        test_mixing_ratio_adapts_with_performance,
        test_subprocess_uses_cached_ratio,
    ]

    for test_fn in test_functions:
        try:
            test_fn()
        except Exception as e:
            # print(f"Test {test_fn.__name__} failed: {e}", exc_info=True)
            raise

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL MULTI-ENV MIXING SYNC TESTS PASSED!")
    logger.info("=" * 70)
    logger.info("\nThe fix successfully addresses the critical bug:")
    logger.info("  ✓ Mixing ratios synced on initialization")
    logger.info("  ✓ Mixing ratios adapt and sync during training")
    logger.info("  ✓ Subprocesses use cached ratios (not stale data)")


if __name__ == "__main__":
    main()
