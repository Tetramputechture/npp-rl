"""
Performance benchmarking utilities for Task 3.1 architecture comparison.

Measures key metrics:
- Inference time per step
- Memory usage (parameters, activations)
- Training convergence speed
- Model complexity (FLOPs, parameter count)

Based on Task 3.1 performance benchmarking requirements.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BenchmarkResults:
    """Results from architecture benchmarking."""
    architecture_name: str
    
    # Inference metrics
    mean_inference_time_ms: float
    std_inference_time_ms: float
    median_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    
    # Memory metrics
    num_parameters: int
    parameter_memory_mb: float
    peak_memory_mb: float
    
    # Complexity metrics
    estimated_flops: float
    
    # Model configuration
    modalities_used: List[str]
    graph_architecture: str
    fusion_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, filepath: Path) -> None:
        """Save benchmark results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'BenchmarkResults':
        """Load benchmark results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ArchitectureBenchmark:
    """
    Benchmark suite for comparing architecture variants.
    
    Measures inference time, memory usage, and computational complexity
    to inform architecture selection decisions.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def benchmark_model(
        self,
        model: nn.Module,
        sample_observations: Dict[str, torch.Tensor],
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        architecture_name: str = "unknown",
        config: Optional[Any] = None
    ) -> BenchmarkResults:
        """
        Comprehensive benchmark of a model architecture.
        
        Args:
            model: The model to benchmark (e.g., feature extractor)
            sample_observations: Sample observation dictionary
            num_iterations: Number of inference iterations for timing
            warmup_iterations: Number of warmup iterations before timing
            architecture_name: Name of the architecture
            config: ArchitectureConfig object
            
        Returns:
            BenchmarkResults object with all metrics
        """
        model = model.to(self.device)
        model.eval()
        
        # Move observations to device
        obs_device = self._move_obs_to_device(sample_observations, self.device)
        
        # 1. Measure inference time
        inference_times = self._measure_inference_time(
            model, obs_device, num_iterations, warmup_iterations
        )
        
        # 2. Measure memory usage
        num_params = self._count_parameters(model)
        param_memory_mb = self._calculate_parameter_memory(model)
        peak_memory_mb = self._measure_peak_memory(model, obs_device)
        
        # 3. Estimate FLOPs
        estimated_flops = self._estimate_flops(model, obs_device)
        
        # Extract configuration info
        if config is not None:
            modalities = config.modalities.get_enabled_modalities()
            graph_arch = config.graph.architecture.value
            fusion_type = config.fusion.fusion_type.value
        else:
            modalities = ["unknown"]
            graph_arch = "unknown"
            fusion_type = "unknown"
        
        # Compute statistics
        mean_time = np.mean(inference_times) * 1000  # Convert to ms
        std_time = np.std(inference_times) * 1000
        median_time = np.median(inference_times) * 1000
        p95_time = np.percentile(inference_times, 95) * 1000
        p99_time = np.percentile(inference_times, 99) * 1000
        
        return BenchmarkResults(
            architecture_name=architecture_name,
            mean_inference_time_ms=float(mean_time),
            std_inference_time_ms=float(std_time),
            median_inference_time_ms=float(median_time),
            p95_inference_time_ms=float(p95_time),
            p99_inference_time_ms=float(p99_time),
            num_parameters=int(num_params),
            parameter_memory_mb=float(param_memory_mb),
            peak_memory_mb=float(peak_memory_mb),
            estimated_flops=float(estimated_flops),
            modalities_used=modalities,
            graph_architecture=graph_arch,
            fusion_type=fusion_type,
        )
    
    def _move_obs_to_device(
        self,
        observations: Dict[str, torch.Tensor],
        device: str
    ) -> Dict[str, torch.Tensor]:
        """Recursively move observations to device."""
        obs_device = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_device[key] = value.to(device)
            elif isinstance(value, dict):
                obs_device[key] = self._move_obs_to_device(value, device)
            else:
                obs_device[key] = value
        return obs_device
    
    def _measure_inference_time(
        self,
        model: nn.Module,
        observations: Dict[str, torch.Tensor],
        num_iterations: int,
        warmup_iterations: int
    ) -> np.ndarray:
        """
        Measure inference time over multiple iterations.
        
        Returns:
            Array of inference times in seconds
        """
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(observations)
        
        # Synchronize for accurate timing on GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(observations)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append(end - start)
        
        return np.array(times)
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _calculate_parameter_memory(self, model: nn.Module) -> float:
        """
        Calculate memory usage of model parameters in MB.
        
        Assumes float32 parameters (4 bytes per parameter).
        """
        num_params = self._count_parameters(model)
        bytes_per_param = 4  # float32
        memory_bytes = num_params * bytes_per_param
        memory_mb = memory_bytes / (1024 ** 2)
        return memory_mb
    
    def _measure_peak_memory(
        self,
        model: nn.Module,
        observations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Measure peak memory usage during forward pass.
        
        Returns:
            Peak memory in MB
        """
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            with torch.no_grad():
                _ = model(observations)
            
            torch.cuda.synchronize()
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 ** 2)
            
            return peak_memory_mb
        else:
            # CPU memory measurement is more complex, return parameter memory as estimate
            return self._calculate_parameter_memory(model)
    
    def _estimate_flops(
        self,
        model: nn.Module,
        observations: Dict[str, torch.Tensor]
    ) -> float:
        """
        Estimate FLOPs (floating point operations) for forward pass.
        
        This is a rough estimate based on layer types.
        For more accurate measurement, consider using tools like thop or ptflops.
        
        Returns:
            Estimated FLOPs in billions (GFLOPs)
        """
        total_flops = 0
        
        # Simple FLOP estimation for common layers
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features (multiply-add)
                total_flops += 2 * module.in_features * module.out_features
            
            elif isinstance(module, nn.Conv2d):
                # FLOPs = 2 * in_channels * out_channels * kernel_size * output_size
                # This is approximate without knowing actual output size
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                total_flops += 2 * module.in_channels * module.out_channels * kernel_ops * 1000  # Rough estimate
            
            elif isinstance(module, nn.Conv3d):
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
                total_flops += 2 * module.in_channels * module.out_channels * kernel_ops * 1000
        
        # Convert to GFLOPs
        gflops = total_flops / 1e9
        return gflops
    
    def compare_architectures(
        self,
        results: List[BenchmarkResults]
    ) -> Dict[str, Any]:
        """
        Compare multiple architecture benchmark results.
        
        Args:
            results: List of BenchmarkResults from different architectures
            
        Returns:
            Dictionary with comparison statistics and rankings
        """
        if not results:
            return {}
        
        # Sort by different metrics
        by_inference_time = sorted(results, key=lambda r: r.mean_inference_time_ms)
        by_memory = sorted(results, key=lambda r: r.parameter_memory_mb)
        by_params = sorted(results, key=lambda r: r.num_parameters)
        
        comparison = {
            "fastest": {
                "name": by_inference_time[0].architecture_name,
                "time_ms": by_inference_time[0].mean_inference_time_ms,
            },
            "most_memory_efficient": {
                "name": by_memory[0].architecture_name,
                "memory_mb": by_memory[0].parameter_memory_mb,
            },
            "smallest": {
                "name": by_params[0].architecture_name,
                "num_params": by_params[0].num_parameters,
            },
            "all_results": [r.to_dict() for r in results],
        }
        
        return comparison
    
    def print_comparison_table(self, results: List[BenchmarkResults]) -> None:
        """Print formatted comparison table of benchmark results."""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*100)
        print("ARCHITECTURE COMPARISON RESULTS")
        print("="*100)
        print(f"{'Architecture':<20} {'Time (ms)':<12} {'Params':<12} {'Memory (MB)':<12} {'Modalities':<20}")
        print("-"*100)
        
        for result in sorted(results, key=lambda r: r.mean_inference_time_ms):
            modalities_str = ','.join(result.modalities_used)[:18]
            print(
                f"{result.architecture_name:<20} "
                f"{result.mean_inference_time_ms:>8.2f} ± {result.std_inference_time_ms:<3.2f} "
                f"{result.num_parameters:>11,} "
                f"{result.parameter_memory_mb:>11.2f} "
                f"{modalities_str:<20}"
            )
        
        print("="*100 + "\n")


def create_mock_observations(
    batch_size: int = 1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Create mock observations for benchmarking.
    
    Args:
        batch_size: Batch size for observations
        device: Device to create tensors on
        
    Returns:
        Dictionary of mock observations matching NPP-RL observation space
    """
    observations = {
        # Temporal frames: [batch, 12, 84, 84]
        "player_frame": torch.randint(0, 256, (batch_size, 12, 84, 84), dtype=torch.uint8, device=device),
        
        # Global view: [batch, 176, 100]
        "global_view": torch.randint(0, 256, (batch_size, 176, 100), dtype=torch.uint8, device=device),
        
        # Game state: [batch, 30]
        "game_state": torch.randn(batch_size, 30, device=device),
        
        # Reachability features: [batch, 8]
        "reachability_features": torch.randn(batch_size, 8, device=device),
        
        # Graph observations (simplified for testing)
        "graph_obs": {
            "node_features": torch.randn(batch_size, 100, 67, device=device),  # 100 nodes, 67 features
            "edge_index": torch.randint(0, 100, (batch_size, 2, 200), device=device),  # 200 edges
            "node_mask": torch.ones(batch_size, 100, dtype=torch.bool, device=device),
            "edge_mask": torch.ones(batch_size, 200, dtype=torch.bool, device=device),
            "node_types": torch.randint(0, 6, (batch_size, 100), device=device),
        }
    }
    
    return observations
