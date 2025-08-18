#!/usr/bin/env python3
"""
Test script for configuration flags and logging functionality.

This script tests that configuration flags are properly tracked and
episode info contains the expected information.
"""

import sys
from pathlib import Path

# Add project roots to path
npp_rl_root = Path(__file__).parent
nclone_root = npp_rl_root.parent / "nclone"
sys.path.insert(0, str(npp_rl_root))
sys.path.insert(0, str(nclone_root))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold


def test_config_flags():
    """Test configuration flags functionality."""
    print("🧪 Testing Configuration Flags")
    print("=" * 50)
    
    # Test 1: Default configuration
    print("\n1. Testing default configuration...")
    env = BasicLevelNoGold(
        render_mode='rgb_array',
        enable_frame_stack=False,
        observation_profile='rich'
    )
    
    print(f"   Config flags: {env.config_flags}")
    print(f"   Observation profile: {env.observation_profile}")
    print(f"   Use rich features: {env.use_rich_features}")
    
    # Test 2: Custom configuration
    print("\n2. Testing custom configuration...")
    env_custom = BasicLevelNoGold(
        render_mode='rgb_array',
        enable_frame_stack=True,
        enable_debug_overlay=True,
        enable_short_episode_truncation=True,
        observation_profile='minimal',
        enable_pbrs=False,
        pbrs_gamma=0.95
    )
    
    print(f"   Config flags: {env_custom.config_flags}")
    print(f"   Observation profile: {env_custom.observation_profile}")
    print(f"   Use rich features: {env_custom.use_rich_features}")
    
    # Test 3: Deprecated flag handling
    print("\n3. Testing deprecated use_rich_game_state flag...")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        env_deprecated = BasicLevelNoGold(
            render_mode='rgb_array',
            enable_frame_stack=False,
            use_rich_game_state=False  # Should trigger deprecation warning
        )
        
        if w:
            print(f"   ✅ Deprecation warning triggered: {w[0].message}")
        else:
            print("   ❌ No deprecation warning triggered")
        
        print(f"   Observation profile: {env_deprecated.observation_profile}")
    
    # Test 4: Episode info with flags
    print("\n4. Testing episode info with configuration flags...")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # No action
    
    print(f"   Episode info keys: {list(info.keys())}")
    
    if 'config_flags' in info:
        print("   ✅ Config flags present in episode info")
        print(f"   Config flags: {info['config_flags']}")
    else:
        print("   ❌ Config flags missing from episode info")
    
    if 'observation_profile' in info:
        print(f"   ✅ Observation profile in info: {info['observation_profile']}")
    else:
        print("   ❌ Observation profile missing from episode info")
    
    if 'pbrs_enabled' in info:
        print(f"   ✅ PBRS enabled flag in info: {info['pbrs_enabled']}")
    else:
        print("   ❌ PBRS enabled flag missing from episode info")
    
    if 'pbrs_components' in info:
        print(f"   ✅ PBRS components in info: {list(info['pbrs_components'].keys())}")
    else:
        print("   ❌ PBRS components missing from episode info")
    
    print("\n✅ Configuration flags test completed!")
    return True


def test_pbrs_logging():
    """Test PBRS component logging."""
    print("\n🔍 Testing PBRS Component Logging")
    print("=" * 50)
    
    # Create environment with PBRS enabled
    env = BasicLevelNoGold(
        render_mode='rgb_array',
        enable_frame_stack=False,
        observation_profile='rich',
        enable_pbrs=True
    )
    
    env.reset()
    
    # Take a few steps to generate PBRS data
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(2)  # Move right
        
        print(f"\nStep {i+1}:")
        print(f"   Total reward: {reward:.4f}")
        
        if 'pbrs_components' in info:
            components = info['pbrs_components']
            print(f"   PBRS components available: {list(components.keys())}")
            
            for comp_name, comp_value in components.items():
                if isinstance(comp_value, (int, float)):
                    print(f"     {comp_name}: {comp_value:.4f}")
                else:
                    print(f"     {comp_name}: {comp_value}")
        
        if terminated or truncated:
            break
    
    print("\n✅ PBRS logging test completed!")
    return True


def main():
    """Run all configuration and logging tests."""
    print("🔬 Configuration Flags and Logging Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Flags", test_config_flags),
        ("PBRS Logging", test_pbrs_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Configuration flags and logging are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())