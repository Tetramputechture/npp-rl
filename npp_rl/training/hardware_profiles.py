"""Hardware-specific configuration profiles for optimal training performance.

This module provides pre-configured training settings optimized for different
GPU setups to maximize training efficiency and throughput.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HardwareProfile:
    """Hardware configuration profile for training optimization.

    Attributes:
        name: Profile name
        num_gpus: Number of GPUs
        gpu_memory_gb: GPU memory in GB
        num_envs: Number of parallel environments
        batch_size: Training batch size
        n_steps: Number of steps per environment per update
        learning_rate: Learning rate
        mixed_precision: Enable mixed precision training
        num_workers: Number of dataloader workers
        prefetch_factor: Number of batches to prefetch
        description: Profile description
    """

    name: str
    num_gpus: int
    gpu_memory_gb: float
    num_envs: int
    batch_size: int
    n_steps: int
    learning_rate: float
    mixed_precision: bool
    num_workers: int
    prefetch_factor: int
    description: str

    def to_training_kwargs(self) -> Dict[str, Any]:
        """Convert profile to training kwargs dictionary.

        Returns:
            Dictionary of training arguments
        """
        return {
            "num_envs": self.num_envs,
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            "learning_rate": self.learning_rate,
            "mixed_precision": self.mixed_precision,
            "num_gpus": self.num_gpus,
        }


# 8x A100 (80 GB) profile - optimized for maximum throughput
A100_8X_80GB = HardwareProfile(
    name="8xA100-80GB",
    num_gpus=8,
    gpu_memory_gb=80.0,
    num_envs=512,  # 64 envs per GPU
    batch_size=2048,  # 256 per GPU
    n_steps=2048,  # Larger rollout buffer for better sample efficiency
    learning_rate=8.49e-4,  # Scaled by sqrt(8) from base 3e-4
    mixed_precision=True,  # A100 has excellent FP16/BF16 support
    num_workers=64,  # Match number of envs per 8 GPUs
    prefetch_factor=2,
    description=(
        "Optimized for 8x A100 (80 GB SXM4). Uses 512 parallel environments "
        "(64 per GPU), large batch sizes (2048), and scaled learning rate. "
        "Mixed precision enabled for A100's Tensor Cores. Suitable for "
        "10M+ timestep training runs with large graph-based models."
    ),
)

# 8x A100 (40 GB) profile - slightly reduced settings
A100_8X_40GB = HardwareProfile(
    name="8xA100-40GB",
    num_gpus=8,
    gpu_memory_gb=40.0,
    num_envs=384,  # 48 envs per GPU
    batch_size=1536,  # 192 per GPU
    n_steps=1536,
    learning_rate=8.49e-4,
    mixed_precision=True,
    num_workers=48,
    prefetch_factor=2,
    description=(
        "Optimized for 8x A100 (40 GB). Reduced environment count compared "
        "to 80GB variant to fit in memory. Still highly efficient for large-"
        "scale training."
    ),
)

# Single A100 (80 GB) profile
A100_1X_80GB = HardwareProfile(
    name="1xA100-80GB",
    num_gpus=1,
    gpu_memory_gb=80.0,
    num_envs=128,
    batch_size=512,
    n_steps=2048,
    learning_rate=3e-4,
    mixed_precision=True,
    num_workers=16,
    prefetch_factor=2,
    description=(
        "Optimized for single A100 (80 GB). Good for prototyping and "
        "smaller-scale experiments."
    ),
)

# 8x V100 (32 GB) profile
V100_8X_32GB = HardwareProfile(
    name="8xV100-32GB",
    num_gpus=8,
    gpu_memory_gb=32.0,
    num_envs=256,  # 32 envs per GPU
    batch_size=1024,  # 128 per GPU
    n_steps=1024,
    learning_rate=8.49e-4,
    mixed_precision=True,
    num_workers=32,
    prefetch_factor=2,
    description=(
        "Optimized for 8x V100 (32 GB). More conservative settings due to "
        "memory constraints. Mixed precision recommended."
    ),
)

# CPU profile (for testing/debugging)
CPU_TESTING = HardwareProfile(
    name="CPU-Testing",
    num_gpus=0,
    gpu_memory_gb=0.0,
    num_envs=4,  # Minimal for quick testing
    batch_size=32,
    n_steps=128,
    learning_rate=3e-4,
    mixed_precision=False,
    num_workers=2,
    prefetch_factor=2,
    description=(
        "Minimal configuration for CPU testing. Not recommended for actual "
        "training, only for validation and debugging."
    ),
)


# Registry of all profiles
HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "8xA100-80GB": A100_8X_80GB,
    "8xA100-40GB": A100_8X_40GB,
    "1xA100-80GB": A100_1X_80GB,
    "8xV100-32GB": V100_8X_32GB,
    "cpu-testing": CPU_TESTING,
}


def get_hardware_profile(name: str) -> HardwareProfile:
    """Get hardware profile by name.

    Args:
        name: Profile name (case-insensitive)

    Returns:
        Hardware profile

    Raises:
        ValueError: If profile not found
    """
    name_lower = name.lower()
    for key, profile in HARDWARE_PROFILES.items():
        if key.lower() == name_lower:
            return profile

    available = ", ".join(HARDWARE_PROFILES.keys())
    raise ValueError(f"Unknown hardware profile: {name}. Available: {available}")


# Architecture-specific memory profiles (GB per environment)
# These values are based on empirical measurements from profiling script
# Note: Measured values are for environment objects only. Actual training memory includes:
# - Environment objects: ~0.3 GB per env (measured)
# - Rollout buffers: n_steps × num_envs × obs_size (~24 KB per observation)
# - Batch processing overhead: ~10-20% additional
# - Model gradients and optimizer states: varies by architecture
#
# Values below include a safety multiplier (~2-3x) to account for rollout buffers
# and training overhead. For precise calculations, use rollout buffer memory separately.
ARCHITECTURE_MEMORY_PROFILES: Dict[str, float] = {
    # MLP baseline - empirically measured: 0.315 GB env-only
    # With rollout buffers (1024 steps × 64 envs × 24 KB = ~1.5 GB) and overhead,
    # total is ~0.9-1.0 GB per env. Using 1.0 GB for safety.
    "mlp_baseline": 1.0,
    # GNN variants - graph processing adds memory overhead
    # Estimate ~2x MLP baseline for graph data structures (nodes/edges)
    # Note: Graph data memory excluded from optimization analysis per user request
    "full_hgt": 2.0,
    "simplified_hgt": 1.8,
    "gat": 1.9,
    "gcn": 1.8,
    # Vision-free variants (no visual processing) - reduce by ~30%
    "vision_free": 1.4,
    "vision_free_gat": 1.5,
    "vision_free_gcn": 1.4,
    "vision_free_simplified": 1.3,
    # Other variants
    "no_global_view": 1.6,  # Remove global view (~176×100 frame)
    "local_frames_only": 1.4,  # Only player frame
    # Frame stacking variants add ~0.3-0.4GB for frame buffers
    "full_hgt_frame_stacked": 2.4,
    "vision_free_frame_stacked": 1.7,
    "visual_frame_stacked_only": 1.7,
    # Default fallback for unknown architectures
    "default": 1.5,  # Conservative middle ground
}


def get_memory_per_env(architecture_name: str) -> float:
    """Get memory per environment for a specific architecture.

    Args:
        architecture_name: Architecture name

    Returns:
        Estimated memory per environment in GB
    """
    # Normalize architecture name
    arch_lower = architecture_name.lower()

    # Direct match
    if arch_lower in ARCHITECTURE_MEMORY_PROFILES:
        return ARCHITECTURE_MEMORY_PROFILES[arch_lower]

    # Check if it's an MLP variant (no graph)
    if "mlp" in arch_lower or "baseline" in arch_lower:
        return ARCHITECTURE_MEMORY_PROFILES["mlp_baseline"]

    # Check if it's a GNN variant
    if any(gnn_type in arch_lower for gnn_type in ["hgt", "gat", "gcn", "graph"]):
        # Default to simplified_hgt memory profile for GNN variants
        return ARCHITECTURE_MEMORY_PROFILES.get("simplified_hgt", 5.0)

    # Default fallback
    return ARCHITECTURE_MEMORY_PROFILES["default"]


def estimate_rollout_buffer_memory_gb(
    num_envs: int, n_steps: int, architecture_name: str = "mlp_baseline"
) -> float:
    """Estimate rollout buffer memory usage in GB.

    Rollout buffers store observations, actions, rewards, values, log_probs, etc.
    for n_steps × num_envs. This is separate from environment object memory.

    Args:
        num_envs: Number of parallel environments
        n_steps: Number of steps per rollout
        architecture_name: Architecture name for observation size estimation

    Returns:
        Estimated rollout buffer memory in GB
    """
    # Observation size per step (excluding graph data):
    # - player_frame: 84×84×1 uint8 = 7,056 bytes
    # - global_view: 176×100×1 uint8 = 17,600 bytes
    # - game_state: 26 float32 = 104 bytes
    # - reachability_features: 8 float32 = 32 bytes
    # - entity_positions: 6 float32 = 24 bytes
    # - switch_states: 25 float32 = 100 bytes
    # Total: ~24.9 KB per observation (rounded to 25 KB for safety)

    # Check if architecture uses visual observations
    arch_lower = architecture_name.lower()
    uses_visual = "vision_free" not in arch_lower

    if uses_visual:
        # With visual observations: ~25 KB per observation
        obs_size_bytes = 25 * 1024  # 25 KB
    else:
        # Vision-free: no visual frames, ~30% smaller
        obs_size_bytes = 17 * 1024  # 17 KB

    # Rollout buffer stores multiple arrays per step:
    # - observations (dict of arrays)
    # - actions (int32)
    # - rewards (float32)
    # - values (float32)
    # - log_probs (float32)
    # - dones (bool)
    # Estimate: observation size × 1.5 for other arrays
    bytes_per_step = obs_size_bytes * 1.5

    # Total buffer size
    total_bytes = bytes_per_step * num_envs * n_steps

    # Convert to GB
    return total_bytes / 1e9


def auto_detect_profile(
    architecture_name: Optional[str] = "full_hgt",
) -> Optional[HardwareProfile]:
    """Auto-detect optimal hardware profile based on available GPUs and architecture.

    Args:
        architecture_name: Architecture name for architecture-aware memory estimation.
            Defaults to "full_hgt" for backward compatibility.

    Returns:
        Best matching hardware profile, or None if no GPUs available
    """
    import torch

    if not torch.cuda.is_available():
        return CPU_TESTING

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        return CPU_TESTING

    # Get first GPU name and memory
    gpu_name = torch.cuda.get_device_name(0).lower()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Match based on GPU count and type (exact matches first)
    if num_gpus == 8:
        if "a100" in gpu_name:
            if gpu_memory_gb > 70:
                return A100_8X_80GB
            else:
                return A100_8X_40GB
        elif "v100" in gpu_name:
            return V100_8X_32GB
    elif num_gpus == 1:
        if "a100" in gpu_name and gpu_memory_gb > 70:
            return A100_1X_80GB

    # Architecture-aware memory estimation
    memory_per_env_gb = get_memory_per_env(architecture_name)

    # Scale n_steps based on GPU count and memory
    # More GPUs/memory = larger rollout buffer for better sample efficiency
    if num_gpus >= 8 and gpu_memory_gb >= 70:
        n_steps = 2048  # High-end multi-GPU setup
    elif num_gpus >= 8 and gpu_memory_gb >= 40:
        n_steps = 1536  # Mid-range multi-GPU setup
    elif num_gpus >= 4:
        n_steps = 1536  # 4 GPUs
    elif num_gpus >= 2:
        n_steps = 1280  # 2 GPUs
    else:
        n_steps = 1024  # Single GPU

    # Calculate rollout buffer memory overhead
    # Start with initial estimate for max envs per GPU
    initial_max_envs_per_gpu = max(
        8, min(256, int((gpu_memory_gb) / memory_per_env_gb))
    )

    # Estimate rollout buffer memory for this configuration
    rollout_buffer_memory_gb = estimate_rollout_buffer_memory_gb(
        num_envs=initial_max_envs_per_gpu * num_gpus,
        n_steps=n_steps,
        architecture_name=architecture_name,
    )

    # Reserve memory for rollout buffers and model overhead
    # Total memory needed = (env memory × num_envs) + rollout buffers + model overhead
    # Model overhead: ~10% of GPU memory for weights, gradients, optimizer states
    model_overhead_gb = gpu_memory_gb * 0.1

    # Available memory for environments = total - rollout buffers - model overhead
    available_for_envs_gb = (
        (gpu_memory_gb * 0.8) - rollout_buffer_memory_gb - model_overhead_gb
    )

    # Recalculate max envs with rollout buffer consideration
    if available_for_envs_gb > 0:
        envs_per_gpu = max(8, min(256, int(available_for_envs_gb / memory_per_env_gb)))
    else:
        # Fallback: use conservative estimate
        envs_per_gpu = max(8, min(256, int((gpu_memory_gb * 0.6) / memory_per_env_gb)))

    # Scale learning rate with square root of GPU count (common practice)
    base_lr = 3e-4
    scaled_lr = base_lr * (num_gpus**0.5) if num_gpus > 1 else base_lr

    # Enable mixed precision for modern GPUs (Volta+, compute capability >= 7.0)
    compute_cap = torch.cuda.get_device_properties(0).major
    use_mixed_precision = compute_cap >= 7

    # Architecture-aware description
    arch_desc = f" ({architecture_name})" if architecture_name else ""
    description = (
        f"Auto-detected profile for {num_gpus}x {gpu_name.upper()} "
        f"({gpu_memory_gb:.0f}GB){arch_desc}. "
        f"Using {memory_per_env_gb:.1f}GB per environment estimate. "
        f"Rollout buffer overhead: {rollout_buffer_memory_gb:.2f}GB. "
        f"Optimized settings based on available hardware."
    )

    return HardwareProfile(
        name=f"Auto-{num_gpus}xGPU-{gpu_memory_gb:.0f}GB",
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        num_envs=envs_per_gpu * num_gpus,
        batch_size=max(32, 256 * num_gpus),  # Ensure minimum batch size
        n_steps=n_steps,
        learning_rate=scaled_lr,
        mixed_precision=use_mixed_precision,
        num_workers=max(2, (envs_per_gpu * num_gpus) // 3),
        prefetch_factor=2,
        description=description,
    )


def print_profile_info(profile: HardwareProfile) -> None:
    """Print detailed information about a hardware profile.

    Args:
        profile: Hardware profile to display
    """
    print(f"\n{'=' * 70}")
    print(f"Hardware Profile: {profile.name}")
    print(f"{'=' * 70}")
    print(f"GPUs: {profile.num_gpus}x ({profile.gpu_memory_gb:.1f} GB each)")
    print(f"Parallel Environments: {profile.num_envs}")
    print(f"Batch Size: {profile.batch_size}")
    print(f"Steps per Update: {profile.n_steps}")
    print(f"Learning Rate: {profile.learning_rate:.2e}")
    print(f"Mixed Precision: {profile.mixed_precision}")
    print(f"Workers: {profile.num_workers}")
    print("\nDescription:")
    print(f"  {profile.description}")
    print(f"{'=' * 70}\n")
