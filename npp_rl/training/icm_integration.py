"""
ICM (Intrinsic Curiosity Module) Integration for Standard PPO

This module provides integration of ICM with standard (non-hierarchical) PPO
to enable curiosity-driven exploration without requiring hierarchical architecture.

Usage:
    # In training config:
    {
        "enable_icm": true,
        "icm_config": {
            "eta": 0.01,
            "alpha": 0.1,
            ...
        }
    }
"""

import logging
from typing import Any, Dict, Optional

from stable_baselines3.common.vec_env import VecEnv

from npp_rl.intrinsic.icm import ICMNetwork, ICMTrainer
from npp_rl.wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper

logger = logging.getLogger(__name__)


class ICMIntegration:
    """
    Manages ICM integration with training environments.
    
    This class handles:
    - Creating ICM networks and trainers
    - Wrapping environments with intrinsic rewards
    - Managing ICM training during policy updates
    - Logging ICM statistics
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize ICM integration.
        
        Args:
            config: ICM configuration dictionary with parameters:
                - feature_dim: Dimension of state features (default: 512)
                - action_dim: Number of discrete actions (default: 6)
                - hidden_dim: Hidden layer dimension (default: 256)
                - eta: Intrinsic reward scaling factor (default: 0.01)
                - alpha: Weight for combining rewards (default: 0.1)
                - lambda_inv: Inverse model loss weight (default: 0.1)
                - lambda_fwd: Forward model loss weight (default: 0.9)
                - learning_rate: ICM optimizer learning rate (default: 0.0001)
                - enable_mine_awareness: Enable mine-aware curiosity (default: True)
                - r_int_clip: Maximum intrinsic reward value (default: 1.0)
                - update_frequency: Update ICM every N steps (default: 4)
                - buffer_size: Experience buffer size (default: 10000)
                - debug: Enable debug output (default: False)
            device: Device for ICM computation ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.icm_network = None
        self.icm_trainer = None
        
        # Extract config parameters with defaults
        self.feature_dim = config.get('feature_dim', 512)
        self.action_dim = config.get('action_dim', 6)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.eta = config.get('eta', 0.01)
        self.alpha = config.get('alpha', 0.1)
        self.lambda_inv = config.get('lambda_inv', 0.1)
        self.lambda_fwd = config.get('lambda_fwd', 0.9)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.enable_mine_awareness = config.get('enable_mine_awareness', True)
        self.r_int_clip = config.get('r_int_clip', 1.0)
        self.update_frequency = config.get('update_frequency', 4)
        self.buffer_size = config.get('buffer_size', 10000)
        self.debug = config.get('debug', False)
        
        self._initialize_icm()
    
    def _initialize_icm(self):
        """Initialize ICM network and trainer."""
        logger.info("=" * 60)
        logger.info("INITIALIZING INTRINSIC CURIOSITY MODULE (ICM)")
        logger.info("=" * 60)
        
        # Create ICM network
        logger.info("Creating ICM network:")
        logger.info(f"  Feature dim:      {self.feature_dim}")
        logger.info(f"  Action dim:       {self.action_dim}")
        logger.info(f"  Hidden dim:       {self.hidden_dim}")
        logger.info(f"  Eta (reward scale): {self.eta}")
        logger.info(f"  Mine awareness:   {self.enable_mine_awareness}")
        
        self.icm_network = ICMNetwork(
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            eta=self.eta,
            lambda_inv=self.lambda_inv,
            lambda_fwd=self.lambda_fwd,
            enable_mine_awareness=self.enable_mine_awareness,
            debug=self.debug,
        ).to(self.device)
        
        logger.info("✓ ICM network created and moved to device")
        
        # Create ICM trainer
        logger.info("Creating ICM trainer:")
        logger.info(f"  Learning rate:    {self.learning_rate}")
        logger.info(f"  Lambda inv:       {self.lambda_inv}")
        logger.info(f"  Lambda fwd:       {self.lambda_fwd}")
        
        self.icm_trainer = ICMTrainer(
            icm_network=self.icm_network,
            learning_rate=self.learning_rate,
            device=self.device,
        )
        
        logger.info("✓ ICM trainer created")
        logger.info("=" * 60)
        logger.info("ICM CONFIGURATION SUMMARY:")
        logger.info(f"  Intrinsic weight (alpha):     {self.alpha} ({self.alpha*100:.0f}%)")
        logger.info(f"  Extrinsic weight:             {1-self.alpha} ({(1-self.alpha)*100:.0f}%)")
        logger.info(f"  Reward clipping:              ±{self.r_int_clip}")
        logger.info(f"  Update frequency:             Every {self.update_frequency} steps")
        logger.info(f"  Buffer size:                  {self.buffer_size}")
        logger.info("=" * 60)
    
    def wrap_environment(self, env: VecEnv, policy: Optional[Any] = None) -> IntrinsicRewardWrapper:
        """
        Wrap vectorized environment with intrinsic reward wrapper.
        
        Args:
            env: Vectorized environment to wrap
            policy: Policy for feature extraction (can be set later)
        
        Returns:
            Environment wrapped with intrinsic rewards
        """
        logger.info("Wrapping environment with ICM intrinsic rewards...")
        
        wrapped_env = IntrinsicRewardWrapper(
            env=env,
            icm_trainer=self.icm_trainer,
            policy=policy,
            alpha=self.alpha,
            r_int_clip=self.r_int_clip,
            update_frequency=self.update_frequency,
            buffer_size=self.buffer_size,
            enable_logging=True,
        )
        
        logger.info(f"✓ Environment wrapped with ICM (alpha={self.alpha})")
        
        return wrapped_env
    
    def get_icm_stats(self) -> Dict[str, float]:
        """
        Get current ICM statistics.
        
        Returns:
            Dictionary of ICM metrics
        """
        if self.icm_trainer is None:
            return {}
        
        # Get stats from ICM trainer
        return self.icm_trainer.get_stats() if hasattr(self.icm_trainer, 'get_stats') else {}


def create_icm_integration(
    enable_icm: bool,
    icm_config: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
) -> Optional[ICMIntegration]:
    """
    Factory function to create ICM integration.
    
    Args:
        enable_icm: Whether to enable ICM
        icm_config: ICM configuration dictionary
        device: Device for computation
    
    Returns:
        ICMIntegration instance if enabled, None otherwise
    """
    if not enable_icm:
        logger.info("ICM disabled in configuration")
        return None
    
    if icm_config is None:
        icm_config = {}
        logger.info("Using default ICM configuration")
    
    return ICMIntegration(config=icm_config, device=device)


# Default configuration template
DEFAULT_ICM_CONFIG = {
    "feature_dim": 512,
    "action_dim": 6,
    "hidden_dim": 256,
    "eta": 0.01,                    # Intrinsic reward scaling
    "alpha": 0.1,                   # 10% intrinsic, 90% extrinsic
    "lambda_inv": 0.1,              # Inverse model weight
    "lambda_fwd": 0.9,              # Forward model weight
    "learning_rate": 0.0001,        # ICM optimizer LR
    "enable_mine_awareness": True,  # Use mine-aware curiosity
    "r_int_clip": 1.0,             # Clip intrinsic rewards
    "update_frequency": 4,          # Update every 4 steps
    "buffer_size": 10000,           # Experience buffer size
    "debug": False,                 # Debug output
}


def get_recommended_icm_config(problem_type: str = "sparse_reward") -> Dict[str, Any]:
    """
    Get recommended ICM configuration for specific problem types.
    
    Args:
        problem_type: Type of RL problem:
            - "sparse_reward": Few rewards, needs strong exploration
            - "dense_reward": Many rewards, needs gentle exploration
            - "dangerous": Mines/traps, needs careful exploration
            - "maze": Navigation challenge, needs systematic exploration
    
    Returns:
        Recommended ICM configuration
    """
    config = DEFAULT_ICM_CONFIG.copy()
    
    if problem_type == "sparse_reward":
        # Stronger intrinsic motivation
        config["eta"] = 0.02           # Higher reward scale
        config["alpha"] = 0.15         # More intrinsic weight (15%)
        config["update_frequency"] = 2  # Update more often
        
    elif problem_type == "dense_reward":
        # Gentler intrinsic motivation
        config["eta"] = 0.005          # Lower reward scale
        config["alpha"] = 0.05         # Less intrinsic weight (5%)
        config["update_frequency"] = 8  # Update less often
        
    elif problem_type == "dangerous":
        # Careful exploration with mine awareness
        config["eta"] = 0.01
        config["alpha"] = 0.1
        config["enable_mine_awareness"] = True  # Essential!
        config["r_int_clip"] = 0.5      # Lower clip to discourage recklessness
        
    elif problem_type == "maze":
        # Systematic exploration
        config["eta"] = 0.015
        config["alpha"] = 0.12
        config["buffer_size"] = 20000   # Larger buffer for complex state space
    
    return config


if __name__ == "__main__":
    # Example usage
    print("ICM Integration Module")
    print("=" * 60)
    print("\nExample configuration:")
    print("\n# In your training config JSON:")
    config_example = {
        "enable_icm": True,
        "icm_config": get_recommended_icm_config("sparse_reward")
    }
    
    import json
    print(json.dumps(config_example, indent=2))
    
    print("\n" + "=" * 60)
    print("Available problem types:")
    for ptype in ["sparse_reward", "dense_reward", "dangerous", "maze"]:
        print(f"  - {ptype}")
