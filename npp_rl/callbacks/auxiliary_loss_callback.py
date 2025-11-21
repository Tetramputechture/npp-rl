"""Callback to monitor auxiliary death prediction performance.

This callback extracts auxiliary predictions from the policy and monitors
death prediction accuracy for training insights. The main auxiliary loss
integration is handled directly in MaskedPPO.
"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch


class AuxiliaryLossCallback(BaseCallback):
    """Callback to monitor auxiliary death prediction performance.

    This callback monitors the auxiliary death prediction head performance
    and logs relevant metrics for training insights.
    """

    def __init__(
        self, log_freq: int = 100, verbose: int = 0, auxiliary_weight: float = 0.1
    ):
        """Initialize auxiliary loss callback.

        Args:
            log_freq: Frequency (in steps) to log auxiliary losses
            verbose: Verbosity level
            auxiliary_weight: Weight for auxiliary loss (for logging, not actual loss integration)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.auxiliary_weight = auxiliary_weight
        self.death_loss_history = deque(maxlen=1000)

    def _on_rollout_end(self) -> bool:
        """Called after rollout collection, before policy update.

        Monitors death prediction performance from rollout buffer data.
        """
        # Check if policy has auxiliary heads
        policy = self.model.policy
        if not hasattr(policy, "get_auxiliary_predictions"):
            return True

        # Get rollout buffer data
        rollout_buffer = self.model.rollout_buffer
        if rollout_buffer is None or rollout_buffer.size() == 0:
            return True

        # Extract observations (may be dict with death_context)
        try:
            observations = rollout_buffer.observations
        except (AttributeError, KeyError):
            # Rollout buffer observations not accessible or structured differently
            return True

        # Monitor death prediction metrics from rollout buffer
        # The actual death prediction labels are computed in the main training loop
        # This callback just monitors prediction statistics for insights
        
        try:
            # Simple monitoring of rollout buffer contents
            buffer_size = rollout_buffer.size()
            
            if self.num_timesteps % self.log_freq == 0 and buffer_size > 0:
                self.logger.record("auxiliary/rollout_buffer_size", buffer_size)
                
                # If observations contain game state, log some basic statistics
                if isinstance(observations, dict):
                    available_keys = list(observations.keys())
                    if self.verbose > 1:
                        self.logger.record("auxiliary/obs_keys_count", len(available_keys))
                    
                    # Log if game_state is available (needed for physics-based death prediction)
                    has_game_state = "game_state" in observations
                    has_entity_positions = "entity_positions" in observations
                    self.logger.record("auxiliary/has_game_state", float(has_game_state))
                    self.logger.record("auxiliary/has_entity_positions", float(has_entity_positions))
                    
        except Exception as e:
            # Handle any errors gracefully
            if self.verbose > 0:
                self.logger.warn(f"Error in auxiliary monitoring: {e}")

        return True

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Monitor auxiliary predictions if available
        try:
            policy = self.model.policy
            if hasattr(policy, "get_auxiliary_predictions"):
                auxiliary_preds = policy.get_auxiliary_predictions()
                if auxiliary_preds is not None and "death_prob" in auxiliary_preds:
                    # Log death probability prediction statistics
                    death_prob = auxiliary_preds["death_prob"]
                    
                    if self.num_timesteps % self.log_freq == 0:
                        # Log various statistics of death predictions
                        death_prob_mean = death_prob.mean().item()
                        death_prob_max = death_prob.max().item()
                        death_prob_min = death_prob.min().item()
                        death_prob_std = death_prob.std().item()
                        
                        self.logger.record("auxiliary/death_prob_mean", death_prob_mean)
                        self.logger.record("auxiliary/death_prob_max", death_prob_max)
                        self.logger.record("auxiliary/death_prob_min", death_prob_min)
                        self.logger.record("auxiliary/death_prob_std", death_prob_std)
                        
                        # Log how many predictions are above certain thresholds
                        high_risk_count = (death_prob > 0.5).sum().item()
                        medium_risk_count = ((death_prob > 0.2) & (death_prob <= 0.5)).sum().item()
                        
                        self.logger.record("auxiliary/high_risk_predictions", high_risk_count)
                        self.logger.record("auxiliary/medium_risk_predictions", medium_risk_count)
                        
        except (AttributeError, KeyError, TypeError, RuntimeError):
            # Policy doesn't have auxiliary predictions or error accessing them
            pass

        return True

