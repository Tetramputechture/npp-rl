"""Early stopping callback for curriculum-based training."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumEarlyStoppingCallback(BaseCallback):
    """Stop training when curriculum success rate plateaus.
    
    Monitors current stage success rate and stops if no improvement
    for specified number of evaluations.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        min_evaluations: int = 5,
        curriculum_manager=None,
        eval_freq: int = 25000,
        verbose: int = 1,
    ):
        """Initialize early stopping callback.
        
        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement threshold (e.g., 0.01 = 1%)
            min_evaluations: Minimum evals before early stopping can trigger
            curriculum_manager: CurriculumManager instance to monitor
            eval_freq: Evaluation frequency in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.min_evaluations = min_evaluations
        self.curriculum_manager = curriculum_manager
        self.eval_freq = eval_freq
        
        self.best_success_rate = -np.inf
        self.evaluations_without_improvement = 0
        self.num_evaluations = 0
        
    def _on_step(self) -> bool:
        """Check for early stopping at evaluation intervals.
        
        Returns:
            True to continue training, False to stop
        """
        # Only check at evaluation intervals
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        if self.curriculum_manager is None:
            return True
        
        self.num_evaluations += 1
        
        # Get current stage performance
        current_stage = self.curriculum_manager.get_current_stage()
        perf = self.curriculum_manager.get_stage_performance(current_stage)
        success_rate = perf['success_rate']
        
        # Check for improvement
        if success_rate > self.best_success_rate + self.min_delta:
            self.best_success_rate = success_rate
            self.evaluations_without_improvement = 0
            if self.verbose > 0:
                print(f"✓ Early stopping: New best success rate: {success_rate:.2%}")
        else:
            self.evaluations_without_improvement += 1
        
        # Log to TensorBoard
        self.logger.record("early_stopping/best_success_rate", self.best_success_rate)
        self.logger.record("early_stopping/evals_without_improvement", 
                          self.evaluations_without_improvement)
        self.logger.record("early_stopping/patience_remaining",
                          self.patience - self.evaluations_without_improvement)
        
        # Check stopping criterion
        if (self.num_evaluations >= self.min_evaluations and 
            self.evaluations_without_improvement >= self.patience):
            
            if self.verbose > 0:
                print("\n" + "=" * 70)
                print("🛑 EARLY STOPPING TRIGGERED")
                print("=" * 70)
                print(f"No improvement for {self.patience} evaluations ({self.patience * self.eval_freq:,} steps)")
                print(f"Best success rate: {self.best_success_rate:.2%}")
                print(f"Current success rate: {success_rate:.2%}")
                print(f"Current stage: {current_stage}")
                print("=" * 70 + "\n")
            
            return False  # Stop training
        
        return True  # Continue training

