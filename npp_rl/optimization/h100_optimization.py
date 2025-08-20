"""
H100 GPU optimization utilities for enhanced training performance.

This module provides optimizations specifically for H100 GPUs including:
- TF32 precision settings for faster matrix operations
- Memory management optimizations
- Mixed precision training utilities
"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def enable_h100_optimizations(enable_tf32: bool = True, 
                             enable_memory_optimization: bool = True,
                             log_optimizations: bool = True) -> Dict[str, Any]:
    """
    Enable H100-specific optimizations for PyTorch training.
    
    Args:
        enable_tf32: Whether to enable TF32 precision for faster matmul operations
        enable_memory_optimization: Whether to enable memory management optimizations
        log_optimizations: Whether to log optimization status
        
    Returns:
        Dictionary containing optimization status and settings
    """
    optimization_status = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': None,
        'tf32_enabled': False,
        'memory_optimized': False,
        'pytorch_version': torch.__version__
    }
    
    if not torch.cuda.is_available():
        if log_optimizations:
            logger.info("CUDA not available - H100 optimizations skipped")
        return optimization_status
    
    # Get device information
    device_name = torch.cuda.get_device_name(0)
    optimization_status['device_name'] = device_name
    
    if log_optimizations:
        logger.info(f"GPU Device: {device_name}")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    
    # Enable TF32 for faster matrix operations on H100
    if enable_tf32:
        try:
            # Enable TF32 for matmul operations
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Set float32 matmul precision to high (uses TF32 on supported hardware)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
                optimization_status['tf32_enabled'] = True
                
                if log_optimizations:
                    logger.info("✅ TF32 precision enabled for matrix operations")
                    logger.info("   - torch.backends.cuda.matmul.allow_tf32 = True")
                    logger.info("   - torch.set_float32_matmul_precision('high')")
            else:
                if log_optimizations:
                    logger.warning("⚠️  torch.set_float32_matmul_precision not available in this PyTorch version")
                    
        except Exception as e:
            if log_optimizations:
                logger.error(f"❌ Failed to enable TF32 optimizations: {e}")
    
    # Enable memory optimizations
    if enable_memory_optimization:
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                
            # Enable memory pool for faster allocation/deallocation
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.empty_cache()  # Clear cache before optimization
                
            # Set memory fraction to prevent OOM on large models
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve 90% of GPU memory for training, leave 10% for system
                torch.cuda.set_per_process_memory_fraction(0.9)
                
            optimization_status['memory_optimized'] = True
            
            if log_optimizations:
                logger.info("✅ Memory optimizations enabled")
                logger.info("   - Flash attention enabled (if available)")
                logger.info("   - Memory pool optimized")
                logger.info("   - Memory fraction set to 90%")
                
        except Exception as e:
            if log_optimizations:
                logger.error(f"❌ Failed to enable memory optimizations: {e}")
    
    # Log H100-specific recommendations
    if log_optimizations and 'H100' in device_name:
        logger.info("🚀 H100 GPU detected - optimizations should provide significant speedup")
        logger.info("   - TF32 can provide 1.5-2x speedup for large matrix operations")
        logger.info("   - Consider using larger batch sizes to maximize H100 utilization")
    elif log_optimizations and any(gpu in device_name for gpu in ['A100', 'V100', 'RTX']):
        logger.info(f"🔧 {device_name} detected - TF32 optimizations may provide moderate speedup")
    
    return optimization_status


def get_recommended_batch_size(device_name: Optional[str] = None, 
                              base_batch_size: int = 2048) -> int:
    """
    Get recommended batch size based on GPU capabilities.
    
    Args:
        device_name: GPU device name (auto-detected if None)
        base_batch_size: Base batch size to scale from
        
    Returns:
        Recommended batch size for the detected GPU
    """
    if device_name is None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    
    if not device_name:
        return base_batch_size
    
    # H100 has massive memory and compute - can handle larger batches
    if 'H100' in device_name:
        return int(base_batch_size * 2.0)  # 2x larger batches
    
    # A100 has good memory - moderate increase
    elif 'A100' in device_name:
        return int(base_batch_size * 1.5)  # 1.5x larger batches
    
    # V100 and RTX cards - conservative increase
    elif any(gpu in device_name for gpu in ['V100', 'RTX 3090', 'RTX 4090']):
        return int(base_batch_size * 1.2)  # 1.2x larger batches
    
    # Default for other GPUs
    return base_batch_size


def log_gpu_memory_usage(prefix: str = "GPU Memory") -> None:
    """
    Log current GPU memory usage for monitoring.
    
    Args:
        prefix: Prefix for log messages
    """
    if not torch.cuda.is_available():
        return
    
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        logger.info(f"{prefix}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
        
    except Exception as e:
        logger.warning(f"Failed to get GPU memory usage: {e}")


def clear_gpu_cache() -> None:
    """Clear GPU memory cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


class H100OptimizedTraining:
    """
    Context manager for H100-optimized training sessions.
    Automatically enables optimizations and cleans up resources.
    """
    
    def __init__(self, enable_tf32: bool = True, 
                 enable_memory_optimization: bool = True,
                 log_memory_usage: bool = True):
        """
        Initialize H100 optimization context.
        
        Args:
            enable_tf32: Whether to enable TF32 precision
            enable_memory_optimization: Whether to enable memory optimizations
            log_memory_usage: Whether to log memory usage during training
        """
        self.enable_tf32 = enable_tf32
        self.enable_memory_optimization = enable_memory_optimization
        self.log_memory_usage = log_memory_usage
        self.optimization_status = None
        
    def __enter__(self):
        """Enter the optimization context."""
        self.optimization_status = enable_h100_optimizations(
            enable_tf32=self.enable_tf32,
            enable_memory_optimization=self.enable_memory_optimization,
            log_optimizations=True
        )
        
        if self.log_memory_usage:
            log_gpu_memory_usage("Training Start")
            
        return self.optimization_status
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the optimization context and clean up."""
        if self.log_memory_usage:
            log_gpu_memory_usage("Training End")
            
        # Clear GPU cache on exit
        clear_gpu_cache()
        
        if exc_type is not None:
            logger.error(f"Training session ended with exception: {exc_type.__name__}: {exc_val}")
        else:
            logger.info("Training session completed successfully")