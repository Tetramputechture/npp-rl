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
    n_steps=1024,
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


def auto_detect_profile() -> Optional[HardwareProfile]:
    """Auto-detect optimal hardware profile based on available GPUs.

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

    # Match based on GPU count and type
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

    # Default: create a conservative profile
    envs_per_gpu = max(16, int(gpu_memory_gb / 4))
    return HardwareProfile(
        name=f"{num_gpus}xGPU-{gpu_memory_gb:.0f}GB",
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        num_envs=envs_per_gpu * num_gpus,
        batch_size=256 * num_gpus,
        n_steps=1024,
        learning_rate=3e-4 * (num_gpus**0.5),
        mixed_precision=True,
        num_workers=envs_per_gpu * num_gpus // 4,
        prefetch_factor=2,
        description=f"Auto-detected profile for {num_gpus}x GPUs",
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


if __name__ == "__main__":
    # Print all available profiles
    print("\nAvailable Hardware Profiles:")
    print("=" * 70)

    for name, profile in HARDWARE_PROFILES.items():
        print_profile_info(profile)

    # Try auto-detection
    print("\nAuto-detected profile:")
    auto_profile = auto_detect_profile()
    if auto_profile:
        print_profile_info(auto_profile)
    else:
        print("No GPUs detected")
