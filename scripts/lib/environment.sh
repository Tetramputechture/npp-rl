#!/bin/bash
###############################################################################
# Environment Debugging and Fixing Functions
# 
# This module contains functions for collecting environment debug information,
# fixing PyTorch CUDA compatibility issues, NumPy compatibility issues, and
# TensorBoard installation problems.
###############################################################################

# ============================================================================
# Environment debugging and fixing functions
# ============================================================================

collect_environment_debug_info() {
    log INFO "Collecting comprehensive environment debug information..."
    
    local debug_file="${LOCAL_LOG_DIR}/environment_debug.log"
    
    # Create Python script to collect environment info
    local collect_script='
import sys
import platform
import subprocess

print("="*80)
print("ENVIRONMENT DEBUG INFORMATION")
print("="*80)
print()

# Python version
print("Python version:", sys.version)
print("Python platform:", platform.platform())
print("Python executable:", sys.executable)
print()

# PyTorch info
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("PyTorch file location:", torch.__file__)
    print("Is debug build:", torch.version.debug)
    print("CUDA used to build PyTorch:", torch.version.cuda if torch.version.cuda else "Could not collect")
    print()
    
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA runtime version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available")
        print("GPU count:", torch.cuda.device_count())
        print()
        print("GPU models and configuration:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Total memory: {props.total_memory/1e9:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("CUDA runtime version: N/A")
        print()
        
        # Check why CUDA is not available
        print("CUDA Unavailability Diagnostics:")
        if "+cpu" in torch.__version__:
            print("  ⚠ PyTorch was built without CUDA support (CPU-only version)")
            print("  ⚠ Version string contains \"+cpu\" suffix")
        else:
            print("  - PyTorch appears to be built with CUDA support")
            print("  - But CUDA is not available at runtime")
        print()
except ImportError as e:
    print("PyTorch not installed:", str(e))
    print()

# NVIDIA GPU info
print("NVIDIA GPU Information:")
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", 
                           "--format=csv,noheader"], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")
        
        # Get driver version separately
        driver_result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", 
                                       "--format=csv,noheader"], 
                                      capture_output=True, text=True, timeout=5)
        if driver_result.returncode == 0:
            print(f"  Nvidia driver version: {driver_result.stdout.strip()}")
    else:
        print("  nvidia-smi failed")
except FileNotFoundError:
    print("  nvidia-smi not found (no GPU or driver not installed)")
except Exception as e:
    print(f"  Error checking GPU: {e}")
print()

# CPU Architecture
print("CPU Architecture:")
try:
    result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if any(key in line for key in ["Architecture:", "CPU(s):", "Model name:", 
                                           "Vendor ID:", "Thread(s) per core:", 
                                           "Core(s) per socket:", "Socket(s):"]):
                print(f"  {line.strip()}")
except Exception as e:
    print(f"  Could not get CPU info: {e}")
print()

# Environment variables
print("Relevant Environment Variables:")
import os
cuda_vars = ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"]
for var in cuda_vars:
    value = os.environ.get(var, "<not set>")
    if len(value) > 200:
        value = value[:200] + "..."
    print(f"  {var}: {value}")
print()

# Check if CUDA environment is properly configured
print("CUDA Environment Check:")
if os.environ.get("CUDA_HOME"):
    cuda_home = os.environ.get("CUDA_HOME")
    print(f"  ✓ CUDA_HOME is set: {cuda_home}")
    if os.path.exists(cuda_home):
        print(f"  ✓ CUDA_HOME directory exists")
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(nvcc_path):
            print(f"  ✓ nvcc found at {nvcc_path}")
        else:
            print(f"  ⚠ nvcc not found at {nvcc_path}")
    else:
        print(f"  ⚠ CUDA_HOME directory does not exist")
else:
    print(f"  ⚠ CUDA_HOME is not set - this may cause issues with CUDA-enabled packages")
print()

# Installed torch packages
print("Installed PyTorch-related packages:")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        for line in result.stdout.split("\n"):
            if any(pkg in line.lower() for pkg in ["torch", "cuda", "nvidia"]):
                print(f"  {line.strip()}")
except Exception as e:
    print(f"  Could not list packages: {e}")
print()

print("="*80)
'
    
    # Run debug collection on remote
    ssh_cmd "python3 -c '$collect_script'" 2>&1 | tee "$debug_file"
    
    log SUCCESS "Debug information saved to: ${debug_file}"
}

fix_pytorch_cuda_arm64() {
    log INFO "Attempting to fix PyTorch CUDA on ARM64 automatically..."
    
    # Uninstall CPU-only PyTorch
    log INFO "Uninstalling CPU-only PyTorch..."
    ssh_cmd "python3 -m pip uninstall -y torch torchvision torchaudio" > /dev/null 2>&1 || true
    
    # Try installing from PyTorch nightly (best ARM64 CUDA support)
    log INFO "Installing PyTorch with CUDA support from nightly builds..."
    if ssh_cmd "python3 -m pip install --user --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128" > /dev/null 2>&1; then
        log SUCCESS "Successfully installed PyTorch nightly with CUDA support"
        
        # Verify it worked
        local cuda_check=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null" || echo "False")
        if [ "$cuda_check" = "True" ]; then
            log SUCCESS "✓ CUDA is now available in PyTorch!"
            return 0
        fi
    fi
    
    # If nightly failed, try PyTorch 2.4.0 (known to work on ARM64)
    log WARNING "Nightly build failed, trying PyTorch 2.4.0 with CUDA 12.1..."
    ssh_cmd "python3 -m pip uninstall -y torch torchvision torchaudio" > /dev/null 2>&1 || true
    
    if ssh_cmd "python3 -m pip install --user torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu128" > /dev/null 2>&1; then
        log SUCCESS "Successfully installed PyTorch 2.4.0 with CUDA 12.1"
        
        # Verify it worked
        local cuda_check=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null" || echo "False")
        if [ "$cuda_check" = "True" ]; then
            log SUCCESS "✓ CUDA is now available in PyTorch!"
            return 0
        fi
    fi
    
    log ERROR "Automatic fix failed. Manual intervention required."
    return 1
}

fix_numpy_compatibility() {
    log INFO "Fixing NumPy compatibility issues..."
    
    # Check if we have NumPy compatibility issues
    # Test if matplotlib (commonly has NumPy compatibility issues) works
    if ssh_cmd "python3 -c 'import matplotlib' 2>/dev/null"; then
        log INFO "matplotlib can be imported, checking NumPy compatibility..."
        
        # Try to import matplotlib.pyplot which triggers NumPy imports
        if ssh_cmd "python3 -c 'import matplotlib.pyplot' 2>&1" | grep -q "numpy.core.multiarray failed to import"; then
            log WARNING "NumPy compatibility issue detected with matplotlib"
            log INFO "Uninstalling system matplotlib packages..."
            
            # Uninstall system matplotlib packages that might be compiled against old NumPy
            ssh_cmd "sudo apt-get remove -y python3-matplotlib 2>/dev/null || true" > /dev/null 2>&1
            ssh_cmd "python3 -m pip uninstall -y matplotlib 2>/dev/null || true" > /dev/null 2>&1
            
            # Reinstall matplotlib from pip (will be compatible with current NumPy)
            log INFO "Reinstalling matplotlib from pip..."
            if ssh_cmd "python3 -m pip install --user 'matplotlib>=3.5.0'" > /dev/null 2>&1; then
                log SUCCESS "matplotlib reinstalled successfully"
                
                # Verify it works now
                if ssh_cmd "python3 -c 'import matplotlib.pyplot' 2>/dev/null"; then
                    log SUCCESS "matplotlib NumPy compatibility fixed!"
                    return 0
                else
                    log ERROR "matplotlib still has issues after reinstall"
                    log INFO "Attempting NumPy downgrade as fallback..."
                    
                    # Try downgrading NumPy to 1.x as last resort
                    if ssh_cmd "python3 -m pip install --user 'numpy<2' --force-reinstall" > /dev/null 2>&1; then
                        log SUCCESS "NumPy downgraded to 1.x"
                        return 0
                    else
                        log ERROR "Failed to fix NumPy compatibility"
                        return 1
                    fi
                fi
            else
                log ERROR "Failed to reinstall matplotlib"
                return 1
            fi
        else
            log SUCCESS "No NumPy compatibility issues detected"
            return 0
        fi
    else
        log INFO "matplotlib not installed, no NumPy compatibility check needed"
        return 0
    fi
}

verify_tensorboard_installation() {
    log INFO "Verifying TensorBoard installation..."
    
    # Try to import tensorboard
    if ssh_cmd "python3 -c 'import tensorboard' 2>/dev/null"; then
        local tb_version=$(ssh_cmd "python3 -c 'import tensorboard; print(tensorboard.__version__)' 2>/dev/null" || echo "unknown")
        log INFO "TensorBoard version: ${tb_version}"
        
        # Test if the problematic import works
        if ssh_cmd "python3 -c 'from tensorboard.compat import notf' 2>/dev/null"; then
            log SUCCESS "TensorBoard is working correctly"
            return 0
        else
            log WARNING "TensorBoard has import issues (notf import failed)"
            log INFO "Attempting to fix TensorBoard installation..."
            
            # Uninstall all tensorboard packages (system and user)
            ssh_cmd "python3 -m pip uninstall -y tensorboard tensorboard-plugin-wit" > /dev/null 2>&1 || true
            
            # Reinstall from pip with compatible version
            log INFO "Installing TensorBoard with compatible protobuf and rich..."
            if ssh_cmd "python3 -m pip install --user 'tensorboard>=2.11.0,<3.0.0' 'protobuf>=3.20.0,<4.0.0' 'rich>=13.0.0'" > /dev/null 2>&1; then
                log SUCCESS "TensorBoard reinstalled successfully"
                
                # Verify it works now
                if ssh_cmd "python3 -c 'from tensorboard.compat import notf' 2>/dev/null"; then
                    log SUCCESS "TensorBoard import issue fixed!"
                    return 0
                else
                    log ERROR "TensorBoard still has import issues after reinstall"
                    return 1
                fi
            else
                log ERROR "Failed to reinstall TensorBoard"
                return 1
            fi
        fi
    else
        log WARNING "TensorBoard not found, installing..."
        if ssh_cmd "python3 -m pip install --user 'tensorboard>=2.11.0,<3.0.0' 'protobuf>=3.20.0,<4.0.0' 'rich>=13.0.0'" > /dev/null 2>&1; then
            log SUCCESS "TensorBoard installed successfully"
            return 0
        else
            log ERROR "Failed to install TensorBoard"
            return 1
        fi
    fi
}

verify_pytorch_cuda_compatibility() {
    log INFO "Verifying PyTorch CUDA compatibility..."
    
    # Log CUDA environment that will be used
    log INFO "CUDA environment configuration:"
    log INFO "  CUDA_HOME=${DETECTED_CUDA_HOME}"
    log INFO "  CUDA_PATH=${DETECTED_CUDA_HOME}"
    log INFO "  LD_LIBRARY_PATH=${DETECTED_CUDA_HOME}/lib64:\$LD_LIBRARY_PATH"
    
    # Collect comprehensive debug info first
    collect_environment_debug_info
    
    # Check if nvidia-smi is available
    if ! ssh_cmd "which nvidia-smi" > /dev/null 2>&1; then
        log WARNING "No GPU detected (nvidia-smi not found). Skipping CUDA verification."
        return 0
    fi
    
    # Check PyTorch CUDA availability
    local cuda_info=$(ssh_cmd "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0, torch.__version__, torch.version.cuda if torch.cuda.is_available() else \"N/A\")' 2>/dev/null" || echo "Error - - -")
    
    if [[ "$cuda_info" == "Error"* ]]; then
        log ERROR "Failed to check PyTorch CUDA status. PyTorch may not be installed correctly."
        return 1
    fi
    
    local cuda_available=$(echo $cuda_info | awk '{print $1}')
    local gpu_count=$(echo $cuda_info | awk '{print $2}')
    local torch_version=$(echo $cuda_info | awk '{print $3}')
    local cuda_version=$(echo $cuda_info | awk '{print $4}')
    
    log INFO "PyTorch version: ${torch_version}"
    
    if [ "$cuda_available" = "True" ]; then
        log SUCCESS "✓ PyTorch CUDA is available!"
        log SUCCESS "  - CUDA version: ${cuda_version}"
        log SUCCESS "  - GPU count: ${gpu_count}"
        
        # Show GPU details
        ssh_cmd "python3 -c 'import torch; [print(f\"  - GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)\") for i in range(torch.cuda.device_count())]'" 2>/dev/null
        
        return 0
    else
        log ERROR "✗ PyTorch CUDA is NOT available!"
        log ERROR "GPUs are detected by nvidia-smi but PyTorch cannot access them."
        log ERROR ""
        
        # Check if it's a CPU-only build
        if echo "$torch_version" | grep -q "+cpu"; then
            log ERROR "DIAGNOSIS: PyTorch is CPU-only build (version has '+cpu' suffix)"
            log ERROR "  Current version: ${torch_version}"
            log ERROR ""
            
            # Check architecture
            local arch=$(ssh_cmd "uname -m" 2>/dev/null)
            if [[ "$arch" == "aarch64" ]] || [[ "$arch" == *"arm"* ]]; then
                log WARNING "Detected ARM64/aarch64 architecture (e.g., NVIDIA Grace Hopper)"
                log INFO "Attempting automatic fix for ARM64 PyTorch CUDA..."
                log INFO ""
                
                # Try to fix automatically
                if fix_pytorch_cuda_arm64; then
                    log SUCCESS "PyTorch CUDA has been fixed automatically!"
                    return 0
                else
                    log ERROR ""
                    log ERROR "Automatic fix failed. Manual options:"
                    log ERROR ""
                    log ERROR "  Option 1 - PyTorch nightly (recommended for ARM64):"
                    log ERROR "    python3 -m pip uninstall -y torch torchvision torchaudio"
                    log ERROR "    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124"
                    log ERROR ""
                    log ERROR "  Option 2 - PyTorch 2.4.0 with CUDA 12.1:"
                    log ERROR "    python3 -m pip uninstall -y torch torchvision torchaudio"
                    log ERROR "    python3 -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
                    log ERROR ""
                    log ERROR "  Option 3 - Build from source:"
                    log ERROR "    git clone --recursive https://github.com/pytorch/pytorch"
                    log ERROR "    cd pytorch"
                    log ERROR "    export USE_CUDA=1"
                    log ERROR "    export TORCH_CUDA_ARCH_LIST=\"9.0\"  # For GH200"
                    log ERROR "    python3 setup.py install"
                    return 1
                fi
            else
                log ERROR "  For x86_64 architecture:"
                log ERROR "    python3 -m pip uninstall -y torch torchvision torchaudio"
                log ERROR "    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
                return 1
            fi
        else
            log ERROR "Common fixes:"
            log ERROR "  1. Check CUDA driver compatibility:"
            log ERROR "     nvidia-smi  # Check driver version"
            log ERROR "     # CUDA 12.1 requires driver version >= 525.60.13"
            log ERROR ""
            log ERROR "  2. Verify environment variables:"
            log ERROR "     echo \$CUDA_VISIBLE_DEVICES  # Should not hide GPUs"
            log ERROR "     echo \$LD_LIBRARY_PATH  # Should include CUDA libraries"
            log ERROR ""
            log ERROR "  3. Try reinstalling PyTorch with CUDA support:"
            log ERROR "     python3 -m pip uninstall -y torch torchvision torchaudio"
            log ERROR "     python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
            return 1
        fi
        
        log ERROR ""
        log ERROR "See environment_debug.log for detailed diagnostics"
        log WARNING "Training cannot proceed without GPU access"
        
        return 1
    fi
}

