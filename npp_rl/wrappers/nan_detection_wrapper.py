"""VecEnv wrapper to detect NaN in observations before batching."""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class NaNDetectionWrapper(VecEnvWrapper):
    """Detect NaN at VecEnv level to identify problematic environment indices."""
    
    def __init__(self, venv):
        super().__init__(venv)
        self.env_nan_counts = [0] * self.num_envs
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._validate_observations(obs, "step")
        return obs, rewards, dones, infos
    
    def reset(self):
        obs = self.venv.reset()
        self._validate_observations(obs, "reset")
        return obs
    
    def _validate_observations(self, obs, context):
        """Validate observations for each environment."""
        if not isinstance(obs, dict):
            return
        
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                continue
            
            # Handle batched observations
            if len(value.shape) > 0 and value.shape[0] == self.num_envs:
                for env_idx in range(self.num_envs):
                    env_value = value[env_idx]
                    if np.isnan(env_value).any():
                        self.env_nan_counts[env_idx] += 1
                        raise ValueError(
                            f"[NAN_DETECTION] NaN in env {env_idx} during {context}. "
                            f"Key: '{key}', shape: {env_value.shape}, "
                            f"NaN count: {np.isnan(env_value).sum()}, "
                            f"Total NaN occurrences for env {env_idx}: {self.env_nan_counts[env_idx]}"
                        )

