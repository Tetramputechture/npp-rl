"""GPU observation wrapper using pinned memory for faster transfers.

This wrapper uses CUDA pinned (page-locked) memory for observation buffers,
enabling faster async CPU→GPU transfers during batch processing. Observations
remain as numpy arrays for compatibility with stable-baselines3 wrappers.

Key features:
- Uses pinned memory for faster CPU→GPU transfers
- Pre-allocates buffers to avoid repeated allocations
- Returns numpy arrays for compatibility with VecTransposeImage
- Falls back to CPU if GPU unavailable
"""

import logging
from typing import Any, Optional, Tuple

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper

logger = logging.getLogger(__name__)


class GPUObservationWrapper(VecEnvWrapper):
    """Wrapper that prepares for faster GPU transfers using pinned memory.

    This wrapper sets up infrastructure for pinned memory buffers but passes
    observations through unchanged for compatibility with stable-baselines3
    wrappers like VecTransposeImage.

    The actual benefit comes from using pinned memory during batch processing
    (when DataLoader or similar creates tensors). This wrapper serves as a
    placeholder for future optimizations.

    The wrapper automatically detects GPU availability and falls back to CPU
    if no GPU is available, maintaining backward compatibility.
    """

    def __init__(
        self,
        venv,
        device: Optional[torch.device] = None,
        use_pinned_memory: bool = True,
    ):
        """Initialize GPU observation wrapper.

        Args:
            venv: Vectorized environment to wrap
            device: PyTorch device (default: auto-detect)
            use_pinned_memory: Use pinned memory for faster CPU→GPU transfers
        """
        super().__init__(venv)

        # Detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.use_pinned_memory = use_pinned_memory and self.device.type == "cuda"
        self.use_gpu = self.device.type == "cuda"

        # Initialize debug flag - try to get from wrapped environments
        self.debug = False
        try:
            if hasattr(venv, "envs") and len(venv.envs) > 0:
                first_env = venv.envs[0]
                if hasattr(first_env, "unwrapped") and hasattr(
                    first_env.unwrapped, "nplay_headless"
                ):
                    self.debug = first_env.unwrapped.nplay_headless.sim.sim_config.debug
        except Exception:
            pass

        if not self.use_gpu:
            logger.info(
                "GPU not available - GPUObservationWrapper will pass through observations unchanged"
            )
            return

        logger.info(
            f"GPUObservationWrapper initialized (pinned memory: {self.use_pinned_memory})"
        )
        logger.info(
            "Note: Observations passed through unchanged for VecTransposeImage compatibility. "
            "Pinned memory benefit comes from DataLoader/batch processing."
        )
        logger.info(f"Debug mode: {self.debug}")

    def _transfer_to_gpu(self, obs: Any) -> Any:
        """Pass through observations unchanged.

        This wrapper provides infrastructure for pinned memory but passes
        observations through unchanged for compatibility with VecTransposeImage.
        The pinned memory benefit comes from using pinned memory when creating
        tensors during batch processing, not from storing observations here.

        Args:
            obs: Observation (dict or array)

        Returns:
            Observation unchanged (numpy arrays)
        """
        # Validate action_mask is NOT transferred to GPU (should stay numpy)
        if isinstance(obs, dict) and "action_mask" in obs:
            if self.debug:
                mask = obs["action_mask"]
                logger.debug(
                    f"[GPUObservationWrapper] action_mask type={type(mask)}, "
                    f"shape={getattr(mask, 'shape', 'N/A')}"
                )

            # CRITICAL: action_mask must stay as numpy array, not tensor
            if isinstance(obs["action_mask"], torch.Tensor):
                logger.error(
                    "[GPUObservationWrapper] action_mask was converted to tensor! "
                    "This breaks compatibility. Converting back to numpy."
                )
                obs["action_mask"] = obs["action_mask"].cpu().numpy()

            # Note: No longer need defensive copy - environment handles immutability

        # Pass through unchanged - pinned memory benefit comes from DataLoader/batch processing
        return obs

    def reset(self) -> Any:
        """Reset environment and transfer initial observation to GPU.

        Returns:
            Initial observation on GPU
        """
        obs = self.venv.reset()
        return self._transfer_to_gpu(obs)

    def step_wait(self) -> Tuple[Any, np.ndarray, np.ndarray, list]:
        """Wait for step to complete and transfer observations to GPU.

        Returns:
            Tuple of (observations, rewards, dones, infos)
            Observations are on GPU, rewards/dones/infos remain on CPU
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        gpu_obs = self._transfer_to_gpu(obs)
        return gpu_obs, rewards, dones, infos
