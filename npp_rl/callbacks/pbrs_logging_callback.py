"""
PBRS Logging Callback

Custom callback for logging PBRS reward components to TensorBoard during training.
This provides detailed insights into how potential-based reward shaping is affecting
the agent's learning process.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from typing import Dict, Any, Optional
import warnings


class PBRSLoggingCallback(BaseCallback):
    """
    Custom callback for logging PBRS reward components to TensorBoard.
    
    This callback extracts PBRS component information from episode info
    and logs it to TensorBoard for monitoring during training.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize PBRS logging callback.
        
        Args:
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        super().__init__(verbose)
        
        # Track component statistics
        self.component_stats = {
            'navigation_reward': [],
            'exploration_reward': [],
            'pbrs_reward': [],
            'total_reward': []
        }
        
        # Track PBRS potential components
        self.potential_stats = {
            'objective': [],
            'hazard': [],
            'impact': [],
            'exploration': []
        }
        
        # Episode counters
        self.episodes_logged = 0
        self.steps_since_log = 0
        self.log_frequency = 100  # Log every N steps
        
        # TensorBoard writer
        self.tb_writer = None
    
    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        # Find TensorBoard writer
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break
        
        if self.tb_writer is None and self.verbose >= 1:
            warnings.warn("TensorBoard writer not found - PBRS logging disabled")
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            bool: If False, training will be stopped
        """
        self.steps_since_log += 1
        
        # Extract PBRS information from episode info
        if hasattr(self.locals, 'infos') and self.locals['infos'] is not None:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'pbrs_components' in info:
                    self._process_pbrs_info(info['pbrs_components'])
        
        # Log statistics periodically
        if self.steps_since_log >= self.log_frequency:
            self._log_statistics()
            self.steps_since_log = 0
        
        return True
    
    def _process_pbrs_info(self, pbrs_components: Dict[str, Any]) -> None:
        """
        Process PBRS component information from episode info.
        
        Args:
            pbrs_components: Dictionary containing PBRS component rewards
        """
        # Extract reward components
        for component in ['navigation_reward', 'exploration_reward', 'pbrs_reward', 'total_reward']:
            if component in pbrs_components:
                value = pbrs_components[component]
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.component_stats[component].append(float(value))
        
        # Extract potential components
        if 'pbrs_components' in pbrs_components:
            potential_components = pbrs_components['pbrs_components']
            for potential_name in ['objective', 'hazard', 'impact', 'exploration']:
                if potential_name in potential_components:
                    # Handle both raw values and weighted values
                    potential_data = potential_components[potential_name]
                    if isinstance(potential_data, dict):
                        value = potential_data.get('weighted_value', potential_data.get('raw_value', 0.0))
                    else:
                        value = potential_data
                    
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.potential_stats[potential_name].append(float(value))
    
    def _log_statistics(self) -> None:
        """Log accumulated statistics to TensorBoard."""
        if self.tb_writer is None:
            return
        
        current_step = self.num_timesteps
        
        # Log reward component statistics
        for component, values in self.component_stats.items():
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                self.tb_writer.add_scalar(f'pbrs_rewards/{component}_mean', mean_value, current_step)
                self.tb_writer.add_scalar(f'pbrs_rewards/{component}_std', std_value, current_step)
                
                if self.verbose >= 2:
                    print(f"PBRS {component}: mean={mean_value:.4f}, std={std_value:.4f}")
        
        # Log potential component statistics
        for potential, values in self.potential_stats.items():
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                
                self.tb_writer.add_scalar(f'pbrs_potentials/{potential}_mean', mean_value, current_step)
                self.tb_writer.add_scalar(f'pbrs_potentials/{potential}_std', std_value, current_step)
                
                if self.verbose >= 2:
                    print(f"PBRS potential {potential}: mean={mean_value:.4f}, std={std_value:.4f}")
        
        # Log summary statistics
        if self.component_stats['total_reward']:
            total_rewards = self.component_stats['total_reward']
            pbrs_rewards = self.component_stats['pbrs_reward']
            
            # Calculate PBRS contribution
            if pbrs_rewards and total_rewards:
                pbrs_contribution = np.mean(pbrs_rewards) / (np.mean(total_rewards) + 1e-8)
                self.tb_writer.add_scalar('pbrs_summary/pbrs_contribution_ratio', pbrs_contribution, current_step)
                
                if self.verbose >= 1:
                    print(f"PBRS contribution ratio: {pbrs_contribution:.3f}")
        
        # Flush TensorBoard writer
        self.tb_writer.flush()
        
        # Clear accumulated statistics
        self._clear_statistics()
    
    def _clear_statistics(self) -> None:
        """Clear accumulated statistics."""
        for component in self.component_stats:
            self.component_stats[component].clear()
        
        for potential in self.potential_stats:
            self.potential_stats[potential].clear()
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # Log final statistics
        if self.steps_since_log > 0:
            self._log_statistics()
        
        if self.verbose >= 1:
            print(f"PBRS logging completed. Total episodes logged: {self.episodes_logged}")


class ConfigFlagsLoggingCallback(BaseCallback):
    """
    Callback for logging environment configuration flags to TensorBoard.
    
    This callback logs the configuration flags used in the environment
    to help track experimental settings.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize config flags logging callback.
        
        Args:
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.flags_logged = False
        self.tb_writer = None
    
    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        # Find TensorBoard writer
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Returns:
            bool: If False, training will be stopped
        """
        # Log configuration flags once at the beginning
        if not self.flags_logged and self.tb_writer is not None:
            self._log_config_flags()
            self.flags_logged = True
        
        return True
    
    def _log_config_flags(self) -> None:
        """Log environment configuration flags to TensorBoard."""
        if hasattr(self.locals, 'infos') and self.locals['infos'] is not None:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'config_flags' in info:
                    config_flags = info['config_flags']
                    
                    # Log boolean flags as scalars
                    for flag_name, flag_value in config_flags.items():
                        if isinstance(flag_value, bool):
                            self.tb_writer.add_scalar(f'config/{flag_name}', float(flag_value), 0)
                        elif isinstance(flag_value, (int, float)):
                            self.tb_writer.add_scalar(f'config/{flag_name}', float(flag_value), 0)
                        elif isinstance(flag_value, str):
                            # Log string values as text
                            self.tb_writer.add_text(f'config/{flag_name}', flag_value, 0)
                    
                    if self.verbose >= 1:
                        print("Configuration flags logged to TensorBoard:")
                        for flag_name, flag_value in config_flags.items():
                            print(f"  {flag_name}: {flag_value}")
                    
                    break  # Only need to log once
        
        self.tb_writer.flush()


def create_pbrs_callbacks(verbose: int = 0) -> list:
    """
    Create a list of PBRS-related callbacks for training.
    
    Args:
        verbose: Verbosity level for callbacks
        
    Returns:
        List of callback instances
    """
    return [
        PBRSLoggingCallback(verbose=verbose),
        ConfigFlagsLoggingCallback(verbose=verbose)
    ]