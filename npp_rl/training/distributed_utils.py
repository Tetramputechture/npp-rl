"""Distributed training utilities for multi-GPU support.

Provides helpers for PyTorch Distributed Data Parallel (DDP) training,
mixed precision training, and environment distribution across GPUs.
"""

import logging
import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def is_distributed_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available() and torch.cuda.is_available()


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Initialize distributed training process group.

    Args:
        rank: Rank of current process (0 to world_size-1)
        world_size: Total number of processes
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        master_addr: Master node address
        master_port: Master node port
    """
    if not is_distributed_available():
        raise RuntimeError(
            "Distributed training not available. "
            "Requires PyTorch with distributed support and CUDA"
        )

    # Set environment variables
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    logger.info(
        f"Initialized distributed training: rank={rank}/{world_size}, "
        f"backend={backend}, device=cuda:{rank}"
    )


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")


def barrier() -> None:
    """Synchronization barrier across all processes.

    Blocks until all processes in the group reach this point.
    Safe to call even if distributed training is not initialized.
    """
    if dist.is_initialized():
        dist.barrier()


def save_on_master(save_fn: callable, *args, **kwargs):
    """Execute save function only on rank 0 (main process).

    This ensures only one process writes to disk, preventing conflicts.
    All processes wait at a barrier after the save completes.

    Args:
        save_fn: Function to execute on rank 0
        *args: Positional arguments for save_fn
        **kwargs: Keyword arguments for save_fn

    Returns:
        Return value of save_fn on rank 0, None on other ranks
    """
    result = None
    if is_main_process():
        result = save_fn(*args, **kwargs)
    barrier()  # All processes wait for rank 0 to finish
    return result


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def wrap_model_ddp(
    model: torch.nn.Module, device_id: int, find_unused_parameters: bool = False
) -> DDP:
    """Wrap model with DistributedDataParallel.

    Args:
        model: PyTorch model to wrap
        device_id: GPU device ID
        find_unused_parameters: If True, find unused parameters
            (useful for complex models with conditional execution)

    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed training not initialized. Call setup_distributed() first"
        )

    model = model.to(device_id)
    ddp_model = DDP(
        model, device_ids=[device_id], find_unused_parameters=find_unused_parameters
    )

    logger.info(f"Wrapped model with DDP on device {device_id}")
    return ddp_model


def sync_across_processes(tensor: torch.Tensor) -> torch.Tensor:
    """Synchronize tensor across all processes (average).

    Args:
        tensor: Tensor to synchronize

    Returns:
        Synchronized tensor
    """
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / get_world_size()
    return tensor


def gather_across_processes(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Gather tensor from all processes to rank 0.

    Args:
        tensor: Tensor to gather

    Returns:
        Gathered tensor (only on rank 0, None on other ranks)
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    rank = get_rank()

    # Create list of tensors on rank 0
    if rank == 0:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gathered = None

    # Gather tensors
    dist.gather(tensor, gather_list=gathered, dst=0)

    if rank == 0:
        return torch.cat(gathered, dim=0)
    return None


class AMPHelper:
    """Helper class for Automatic Mixed Precision training."""

    def __init__(self, enabled: bool = True):
        """Initialize AMP helper.

        Args:
            enabled: If True, use mixed precision training
        """
        self.enabled = enabled and torch.cuda.is_available()

        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            if enabled:
                logger.warning("Mixed precision requested but CUDA not available")

    def autocast(self):
        """Get autocast context manager."""
        if self.enabled:
            return torch.cuda.amp.autocast()
        else:
            return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss (or original if AMP disabled)
        """
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with gradient scaling.

        Args:
            optimizer: Optimizer to step
        """
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with proper scaling.

        Args:
            loss: Loss tensor
        """
        if self.enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()


def distribute_environments(
    num_envs: int, world_size: int, rank: int
) -> tuple[int, int]:
    """Calculate environment distribution across processes.

    Args:
        num_envs: Total number of environments
        world_size: Number of processes
        rank: Current process rank

    Returns:
        Tuple of (envs_per_process, start_idx)
    """
    envs_per_process = num_envs // world_size
    remainder = num_envs % world_size

    # Distribute remainder evenly
    if rank < remainder:
        envs_per_process += 1
        start_idx = rank * envs_per_process
    else:
        start_idx = (
            remainder * (envs_per_process + 1) + (rank - remainder) * envs_per_process
        )

    logger.info(
        f"Process {rank}: managing {envs_per_process} environments "
        f"starting at index {start_idx}"
    )

    return envs_per_process, start_idx


def configure_cuda_for_training(
    device_id: int, enable_tf32: bool = True, benchmark: bool = True
) -> None:
    """Configure CUDA settings for optimal training performance.

    Args:
        device_id: GPU device ID
        enable_tf32: If True, enable TF32 for Ampere+ GPUs
        benchmark: If True, enable cudnn benchmark mode
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU configuration")
        return

    torch.cuda.set_device(device_id)

    if enable_tf32 and torch.cuda.get_device_capability()[0] >= 8:
        # Enable TF32 for Ampere (A100, RTX 30xx) and newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster training on Ampere+ GPUs")

    if benchmark:
        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled")

    # Log GPU info
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
    logger.info(f"Using GPU {device_id}: {gpu_name} ({gpu_memory:.1f} GB)")


class DistributedTrainingContext:
    """Context manager for distributed training setup and cleanup."""

    def __init__(
        self, rank: int, world_size: int, backend: str = "nccl", enable_amp: bool = True
    ):
        """Initialize distributed training context.

        Args:
            rank: Process rank
            world_size: Total number of processes
            backend: Distributed backend
            enable_amp: Enable mixed precision training
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.enable_amp = enable_amp
        self.amp_helper = None

    def __enter__(self):
        """Set up distributed training."""
        if self.world_size > 1:
            setup_distributed(self.rank, self.world_size, self.backend)

        configure_cuda_for_training(self.rank)

        self.amp_helper = AMPHelper(enabled=self.enable_amp)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up distributed training."""
        if self.world_size > 1:
            cleanup_distributed()
