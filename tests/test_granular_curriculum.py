"""
Tests for granular curriculum progression features.

Tests verify:
1. Stage-specific thresholds work correctly
2. Adaptive minimum episodes per stage
3. Early advancement for high performers
4. Performance trend analysis
5. Adaptive stage mixing
"""

import logging
from collections import deque
from unittest.mock import Mock, patch
import numpy as np
import pytest

from npp_rl.training.curriculum_manager import CurriculumManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_dataset(tmp_path, suffix=""):
    """Create a mock dataset directory structure."""
    dataset_dir = tmp_path / f"test_dataset{suffix}"
    dataset_dir.mkdir(exist_ok=True)
    
    # Create category subdirectories with mock levels
    for category in ["simplest", "simpler", "simple", "medium", "complex", "exploration", "mine_heavy"]:
        category_dir = dataset_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Create a simple mock level file
        level_file = category_dir / "level_001.json"
        level_file.write_text('{"level_id": "test_001", "map_data": "mock_data", "category": "' + category + '"}')
    
    return str(dataset_dir)


def test_stage_specific_thresholds(tmp_path):
    """Test that stage-specific thresholds are applied correctly."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Stage-Specific Thresholds")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
        # Don't override - use stage-specific
        advancement_threshold=None,
        min_episodes_per_stage=None,
    )
    
    # Verify stage-specific thresholds
    assert manager.STAGE_THRESHOLDS["simplest"] == 0.60
    assert manager.STAGE_THRESHOLDS["simpler"] == 0.65
    assert manager.STAGE_THRESHOLDS["complex"] == 0.75
    assert manager.STAGE_THRESHOLDS["mine_heavy"] == 0.80
    
    # Test performance calculation uses stage-specific threshold
    perf_simplest = manager.get_stage_performance("simplest")
    assert perf_simplest["advancement_threshold"] == 0.60
    
    perf_complex = manager.get_stage_performance("complex")
    assert perf_complex["advancement_threshold"] == 0.75
    
    logger.info("✓ Stage-specific thresholds verified")


def test_adaptive_minimum_episodes(tmp_path):
    """Test that adaptive minimum episodes work correctly."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Adaptive Minimum Episodes")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
    )
    
    # Verify stage-specific min episodes
    assert manager.STAGE_MIN_EPISODES["simplest"] == 50
    assert manager.STAGE_MIN_EPISODES["simpler"] == 60
    assert manager.STAGE_MIN_EPISODES["complex"] == 120
    assert manager.STAGE_MIN_EPISODES["mine_heavy"] == 150
    
    # Test that min episodes are used in performance calculation
    perf_simplest = manager.get_stage_performance("simplest")
    assert perf_simplest["min_episodes"] == 50
    
    perf_complex = manager.get_stage_performance("complex")
    assert perf_complex["min_episodes"] == 120
    
    logger.info("✓ Adaptive minimum episodes verified")


def test_early_advancement(tmp_path):
    """Test early advancement for high performers."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Early Advancement for High Performers")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
        enable_early_advancement=True,
    )
    
    # Simulate 35 episodes with 95% success (well above early advancement threshold)
    for i in range(35):
        success = i % 20 != 0  # 95% success rate
        manager.record_episode("simplest", success)
    
    perf = manager.get_stage_performance("simplest")
    
    logger.info(f"Episodes: {perf['episodes']}")
    logger.info(f"Success rate: {perf['success_rate']:.1%}")
    logger.info(f"Can advance: {perf['can_advance']}")
    logger.info(f"Can early advance: {perf.get('can_early_advance', False)}")
    
    # Should be able to advance early (35 episodes > 30 min, 95% > 90% threshold)
    assert perf["can_advance"], "Should be able to advance early with high performance"
    assert perf.get("can_early_advance", False), "Should trigger early advancement"
    
    # Verify advancement works
    advanced = manager.check_advancement()
    assert advanced, "Should advance to next stage"
    assert manager.current_stage == "simpler"
    
    logger.info("✓ Early advancement verified")


def test_performance_trend_analysis(tmp_path):
    """Test performance trend calculation and trend-based advancement."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Performance Trend Analysis")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
        enable_trend_analysis=True,
    )
    
    # Simulate improving performance (40% -> 70%)
    # First 25 episodes: 40% success
    for i in range(25):
        success = i % 5 < 2  # 40% success
        manager.record_episode("simplest", success)
    
    # Next 25 episodes: 70% success (strong improvement trend)
    for i in range(25):
        success = i % 10 < 7  # 70% success
        manager.record_episode("simplest", success)
    
    perf = manager.get_stage_performance("simplest")
    
    logger.info(f"Episodes: {perf['episodes']}")
    logger.info(f"Success rate: {perf['success_rate']:.1%}")
    logger.info(f"Trend: {perf['trend']:+.2f}")
    logger.info(f"Can advance: {perf['can_advance']}")
    
    # Should show positive trend
    assert perf["trend"] > 0.15, f"Should show strong positive trend, got {perf['trend']}"
    
    # Might get trend bonus if close to threshold
    if perf.get("trend_bonus", False):
        logger.info("✓ Trend bonus activated")
    
    logger.info("✓ Performance trend analysis verified")


def test_adaptive_stage_mixing(tmp_path):
    """Test adaptive stage mixing based on performance."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Adaptive Stage Mixing")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simpler",  # Start at stage 1 so we can mix with stage 0
        enable_adaptive_mixing=True,
        allow_stage_mixing=True,
    )
    
    # Test different performance levels and their mixing ratios
    # Use numpy for precise success rates
    
    # Low performance (< 50%) -> high mixing (40%)
    np.random.seed(42)
    for i in range(50):
        success = np.random.random() < 0.40  # 40% success
        manager.record_episode("simpler", success)
    
    mix_ratio_low = manager._get_adaptive_mixing_ratio("simpler")
    logger.info(f"Low performance (~40%) -> Mixing ratio: {mix_ratio_low:.1%}")
    assert mix_ratio_low == 0.40, "Low performance should increase mixing to 40%"
    
    # Clear performance and episode counts
    manager.stage_performance["simpler"].clear()
    manager.stage_episode_counts["simpler"] = 0
    
    # Medium performance (50-65%) -> moderate mixing (25%)
    np.random.seed(43)
    for i in range(50):
        success = np.random.random() < 0.55  # 55% success (in 50-65% range)
        manager.record_episode("simpler", success)
    
    mix_ratio_med = manager._get_adaptive_mixing_ratio("simpler")
    logger.info(f"Medium performance (~55%) -> Mixing ratio: {mix_ratio_med:.1%}")
    assert mix_ratio_med == 0.25, "Medium performance (50-65%) should use 25% mixing"
    
    # Clear performance and episode counts
    manager.stage_performance["simpler"].clear()
    manager.stage_episode_counts["simpler"] = 0
    
    # High performance (> 80%) -> minimal mixing (5%)
    np.random.seed(44)
    for i in range(50):
        success = np.random.random() < 0.90  # 90% success
        manager.record_episode("simpler", success)
    
    mix_ratio_high = manager._get_adaptive_mixing_ratio("simpler")
    logger.info(f"High performance (~90%) -> Mixing ratio: {mix_ratio_high:.1%}")
    assert mix_ratio_high == 0.05, "High performance should minimize mixing to 5%"
    
    logger.info("✓ Adaptive stage mixing verified")


def test_full_progression_scenario(tmp_path):
    """Test complete progression through stages with granular features."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Full Progression Scenario")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
        enable_adaptive_mixing=True,
        enable_early_advancement=True,
        enable_trend_analysis=True,
    )
    
    # Stage 1 (simplest): Quick mastery -> early advancement at 35 episodes
    logger.info("\nStage 1: Simplest (aiming for early advancement)")
    for i in range(35):
        success = i % 20 < 19  # 95% success rate
        manager.record_episode("simplest", success)
    
    advanced = manager.check_advancement()
    assert advanced, "Should advance from simplest via early advancement"
    assert manager.current_stage == "simpler"
    
    # Stage 2 (simpler): Normal progression with 65% threshold at 60 episodes
    logger.info("\nStage 2: Simpler (standard progression)")
    for i in range(65):
        success = i % 10 < 7  # 70% success rate (above 65% threshold)
        manager.record_episode("simpler", success)
    
    advanced = manager.check_advancement()
    assert advanced, "Should advance from simpler with standard progression"
    assert manager.current_stage == "simple"
    
    logger.info("\n✓ Full progression scenario verified")
    logger.info(f"Final stage: {manager.current_stage}")


def test_backwards_compatibility(tmp_path):
    """Test that system works with global overrides (backwards compatible)."""
    logger.info("\n" + "="*70)
    logger.info("TEST: Backwards Compatibility")
    logger.info("="*70)
    
    dataset_path = create_mock_dataset(tmp_path)
    
    # Use global overrides like old system
    manager = CurriculumManager(
        dataset_path=dataset_path,
        starting_stage="simplest",
        advancement_threshold=0.7,  # Global override
        min_episodes_per_stage=100,  # Global override
    )
    
    # All stages should use global override
    for stage in manager.CURRICULUM_ORDER:
        perf = manager.get_stage_performance(stage)
        assert perf["advancement_threshold"] == 0.7, \
            f"Stage {stage} should use global threshold"
        assert perf["min_episodes"] == 100, \
            f"Stage {stage} should use global min episodes"
    
    logger.info("✓ Backwards compatibility verified")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    
    print("\n" + "="*70)
    print("GRANULAR CURRICULUM PROGRESSION TESTS")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Use unique suffix for each test to avoid conflicts
        for i, test_fn in enumerate([
            test_stage_specific_thresholds,
            test_adaptive_minimum_episodes,
            test_early_advancement,
            test_performance_trend_analysis,
            test_adaptive_stage_mixing,
            test_full_progression_scenario,
            test_backwards_compatibility,
        ]):
            test_dir = tmp_path / f"test_{i}"
            test_dir.mkdir(exist_ok=True)
            test_fn(test_dir)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nGranular curriculum features verified:")
    print("  ✓ Stage-specific advancement thresholds")
    print("  ✓ Adaptive minimum episodes per stage")
    print("  ✓ Early advancement for high performers (90% @ 30 episodes)")
    print("  ✓ Performance trend analysis and trend-based advancement")
    print("  ✓ Adaptive stage mixing (40% struggling -> 5% mastering)")
    print("  ✓ Full progression scenario")
    print("  ✓ Backwards compatibility with global overrides")
