#!/usr/bin/env python3
"""
Quick PyTorch installation verification script for NPP-RL.

This script tests that PyTorch is properly installed and can perform
basic operations. Run this after installation to verify everything works.

Usage:
    python test_pytorch.py
"""

import sys
import time


def test_pytorch_installation():
    """Test PyTorch installation and basic functionality."""
    print("🔍 PyTorch Installation Verification")
    print("=" * 40)

    # Test 1: Import PyTorch
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False

    # Test 2: Check torchvision and torchaudio
    try:
        import torchvision

        print(f"✅ torchvision {torchvision.__version__} available")
    except ImportError:
        print("⚠️  torchvision not available")

    try:
        import torchaudio

        print(f"✅ torchaudio {torchaudio.__version__} available")
    except ImportError:
        print("⚠️  torchaudio not available")

    # Test 3: CUDA availability
    print(f"\n🖥️  Compute Backend:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ CUDA available with {gpu_count} GPU(s)")
        print(f"   Primary GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("ℹ️  CUDA not available - using CPU")

    # Test 4: Basic tensor operations
    print(f"\n🧮 Testing tensor operations:")
    try:
        # CPU test
        start_time = time.time()
        x = torch.randn(1000, 1000)
        y = torch.mm(x, x)
        cpu_time = time.time() - start_time
        print(f"✅ CPU computation: {cpu_time:.3f}s for 1000x1000 matrix multiply")

        # GPU test if available
        if torch.cuda.is_available():
            start_time = time.time()
            x_gpu = torch.randn(1000, 1000).cuda()
            y_gpu = torch.mm(x_gpu, x_gpu)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_time = time.time() - start_time
            print(f"✅ GPU computation: {gpu_time:.3f}s for 1000x1000 matrix multiply")
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
            print(f"   GPU speedup: {speedup:.1f}x")

            # Test GPU memory
            try:
                large_tensor = torch.randn(5000, 5000).cuda()
                print(f"✅ GPU memory test: allocated large tensor successfully")
                del large_tensor
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"⚠️  GPU memory warning: {e}")
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False

    # Test 5: Neural network basics
    print(f"\n🧠 Testing neural network components:")
    try:
        import torch.nn as nn
        import torch.optim as optim

        # Simple network
        model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Test forward pass
        x = torch.randn(32, 100).to(device)
        output = model(x)
        print(f"✅ Forward pass: input {x.shape} -> output {output.shape}")

        # Test backward pass
        loss = nn.functional.mse_loss(output, torch.randn_like(output))
        loss.backward()
        print(f"✅ Backward pass: loss = {loss.item():.4f}")

        print(f"✅ Neural network test passed on {device}")

    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        return False

    # Test 6: Integration with other dependencies
    print(f"\n📦 Testing integration:")
    try:
        import numpy as np

        torch_array = torch.randn(5, 5)
        numpy_array = torch_array.numpy()
        back_to_torch = torch.from_numpy(numpy_array)
        print(f"✅ NumPy integration works")

        if torch.cuda.is_available():
            gpu_array = torch_array.cuda()
            back_to_cpu = gpu_array.cpu().numpy()
            print(f"✅ CUDA to CPU conversion works")
    except Exception as e:
        print(f"⚠️  Integration test warning: {e}")

    print(f"\n🎉 PyTorch installation verification complete!")
    return True


if __name__ == "__main__":
    try:
        success = test_pytorch_installation()
        if success:
            print("\n✅ All tests passed! PyTorch is ready for NPP-RL training.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please check your PyTorch installation.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        sys.exit(1)
