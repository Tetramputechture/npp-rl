"""Optuna callbacks for trial pruning during training."""
import optuna
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)


class OptunaTrialPruningCallback(BaseCallback):
    """
    SB3 callback that reports intermediate values to Optuna and prunes unpromising trials.
    
    Evaluates on test set periodically and reports metric to Optuna.
    Raises TrialPruned exception if trial should be stopped early.
    """
    
    def __init__(
        self,
        trial: optuna.Trial,
        eval_freq: int,
        trainer,  # ArchitectureTrainer instance
        verbose: int = 0,
    ):
        """
        Initialize Optuna pruning callback.
        
        Args:
            trial: Optuna trial object for reporting metrics
            eval_freq: Frequency of evaluations in training steps
            trainer: ArchitectureTrainer instance for running evaluations
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.trainer = trainer
        self.eval_count = 0
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if it's time to evaluate
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            try:
                # Run evaluation on test set
                eval_results = self.trainer.evaluate(
                    num_episodes=50,  # Reduced for speed during optimization
                    record_videos=False,
                )
                
                # Calculate metric
                success_rate = eval_results.get("success_rate", 0.0)
                # Use avg_reward from ComprehensiveEvaluator results
                mean_reward = eval_results.get("avg_reward", eval_results.get("mean_reward", 0.0))
                
                # Normalize reward (assuming range -1000 to 1000)
                # Clamp to [0, 1] range
                normalized_reward = max(0.0, min(1.0, (mean_reward + 1000) / 2000))
                
                # Combined metric (70% success rate, 30% normalized reward)
                metric = 0.7 * success_rate + 0.3 * normalized_reward
                
                # Report to Optuna (we report positive metric, Optuna will minimize -metric)
                self.trial.report(metric, self.eval_count)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    logger.info(f"Trial {self.trial.number} pruned at step {self.num_timesteps}")
                    raise optuna.TrialPruned()
                
                self.eval_count += 1
                self.last_eval_step = self.num_timesteps
                
                if self.verbose > 0:
                    logger.info(
                        f"Step {self.num_timesteps}: metric={metric:.4f}, "
                        f"success_rate={success_rate:.2%}, "
                        f"mean_reward={mean_reward:.2f}"
                    )
            except optuna.TrialPruned:
                # Re-raise to let Optuna handle it
                raise
            except Exception as e:
                logger.error(f"Error during evaluation at step {self.num_timesteps}: {e}")
                # Continue training even if evaluation fails
                # Report worst possible metric to encourage pruning
                self.trial.report(0.0, self.eval_count)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        
        return True  # Continue training

