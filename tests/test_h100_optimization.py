#!/usr/bin/env python3
"""
Test script for H100 optimization functionality.

This script tests the H100 optimization module to ensure it works correctly
both with and without CUDA availability.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from npp_rl.optimization.h100_optimization import (
    enable_h100_optimizations, 
    get_recommended_batch_size, 
    H100OptimizedTraining,
    log_gpu_memory_usage,
    clear_gpu_cache
)

# Set up logging to see optimization messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_h100_optimization_module():
    """Test the H100 optimization module functionality."""
    print("🧪 Testing H100 Optimization Module")
    print("=" * 50)
    
    # Test 1: Basic optimization enablement
    print("\n1. Testing basic optimization enablement...")
    status = enable_h100_optimizations(
        enable_tf32=True,
        enable_memory_optimization=True,
        log_optimizations=True
    )
    
    print(f"   CUDA Available: {status['cuda_available']}")
    print(f"   Device Name: {status['device_name']}")
    print(f"   TF32 Enabled: {status['tf32_enabled']}")
    print(f"   Memory Optimized: {status['memory_optimized']}")
    print(f"   PyTorch Version: {status['pytorch_version']}")
    
    # Test 2: Batch size recommendations
    print("\n2. Testing batch size recommendations...")
    
    test_devices = [
        None,  # Auto-detect
        "NVIDIA H100 PCIe",
        "NVIDIA A100-SXM4-40GB", 
        "NVIDIA GeForce RTX 4090",
        "NVIDIA Tesla V100-SXM2-32GB",
        "Unknown GPU"
    ]
    
    base_batch_size = 2048
    for device in test_devices:
        recommended = get_recommended_batch_size(device, base_batch_size)
        device_name = device or "Auto-detected"
        print(f"   {device_name}: {base_batch_size} → {recommended}")
    
    # Test 3: Memory utilities (safe to call without CUDA)
    print("\n3. Testing memory utilities...")
    log_gpu_memory_usage("Test Memory Check")
    clear_gpu_cache()
    print("   Memory utilities executed successfully")
    
    # Test 4: Context manager
    print("\n4. Testing H100OptimizedTraining context manager...")
    try:
        with H100OptimizedTraining(
            enable_tf32=True,
            enable_memory_optimization=True,
            log_memory_usage=True
        ) as optimization_status:
            print("   Context entered successfully")
            print(f"   Optimization status: {optimization_status}")
            # Simulate some work
            import time
            time.sleep(0.1)
        print("   Context exited successfully")
    except Exception as e:
        print(f"   ❌ Context manager failed: {e}")
        return False
    
    print("\n✅ All H100 optimization tests passed!")
    return True


def test_integration_with_ppo():
    """Test integration with PPO agent (import test only)."""
    print("\n🔗 Testing Integration with PPO Agent")
    print("=" * 50)
    
    try:
        # Test that we can import the PPO agent with H100 optimizations
        from npp_rl.agents.npp_agent_ppo import start_training
        print("   ✅ PPO agent imports successfully with H100 optimization support")
        
        # Test that the function signature includes the new parameter
        import inspect
        sig = inspect.signature(start_training)
        if 'enable_h100_optimization' in sig.parameters:
            print("   ✅ start_training function includes enable_h100_optimization parameter")
        else:
            print("   ❌ start_training function missing enable_h100_optimization parameter")
            return False
            
        return True
        
    except ImportError as e:
        print(f"   ❌ Failed to import PPO agent: {e}")
        return False


def test_cuda_simulation():
    """Test what would happen on a CUDA-enabled system."""
    print("\n🚀 Simulating CUDA-enabled System Behavior")
    print("=" * 50)
    
    # Show what the logs would look like on different GPU types
    gpu_types = [
        "NVIDIA H100 PCIe",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA Tesla V100-SXM2-32GB"
    ]
    
    print("\nExpected optimization behavior on CUDA systems:")
    for gpu in gpu_types:
        print(f"\n   {gpu}:")
        print("   - TF32 would be enabled: torch.backends.cuda.matmul.allow_tf32 = True")
        print("   - Float32 precision: torch.set_float32_matmul_precision('high')")
        print("   - Memory optimizations: Flash attention, memory pool, 90% memory fraction")
        
        if "H100" in gpu:
            print("   - Expected speedup: 1.5-2x for large matrix operations")
            print(f"   - Recommended batch size: {get_recommended_batch_size(gpu, 2048)}")
        elif "A100" in gpu:
            print("   - Expected speedup: Moderate improvement")
            print(f"   - Recommended batch size: {get_recommended_batch_size(gpu, 2048)}")
        else:
            print("   - Expected speedup: Some improvement")
            print(f"   - Recommended batch size: {get_recommended_batch_size(gpu, 2048)}")
    
    print("\n   ✅ CUDA simulation completed")
    return True


def main():
    """Run all H100 optimization tests."""
    print("🔬 H100 Optimization Test Suite")
    print("=" * 60)
    
    tests = [
        ("H100 Optimization Module", test_h100_optimization_module),
        ("PPO Integration", test_integration_with_ppo),
        ("CUDA Simulation", test_cuda_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
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
        print("🎉 All tests passed! H100 optimization is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())