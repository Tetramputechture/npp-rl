"""Optuna script for optimizing the hyperparameters of our PPO agent
for the N++ environment.

This script uses Optuna to perform hyperparameter optimization for the PPO
agent. It includes pruning of bad trials and proper handling of the evaluation
environment.
"""

from typing import Any, Dict, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecCheckNan, VecNormalize
import torch
from pathlib import Path
import json
import datetime
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from npp_rl.agents.npp_feature_extractor_impala import NPPFeatureExtractorImpala


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tuning constants
N_TRIALS = 50  # Number of trials to run
N_STARTUP_TRIALS = 5  # Number of trials before pruning starts
N_EVALUATIONS = 4  # Number of evaluations per trial
N_WARMUP_STEPS = 10
N_TIMESTEPS = int(8e6)  # Total timesteps per trial
N_EVAL_EPISODES = 5  # Episodes per evaluation
N_ENVS = 64  # Number of parallel environments
EVAL_FREQ = max(10000 // N_ENVS, 1)  # Evaluation frequency

# Default hyperparameters that won't be tuned
DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "device": device,
}

# If we want to use past 4 frames along with the current frame
# in our input
ENABLE_FRAME_STACK = False

SEED = 42


def create_env(n_envs: int = 1, render_mode: str = 'rgb_array', enable_short_episode_truncation: bool = False, training: bool = True, eval_mode: bool = False) -> VecNormalize:
    """Create a vectorized environment for training or evaluation."""
    env = SubprocVecEnv(
        [lambda: BasicLevelNoGold(
            render_mode=render_mode,
            enable_frame_stack=ENABLE_FRAME_STACK,
            enable_short_episode_truncation=enable_short_episode_truncation,
            seed=SEED,
            eval_mode=eval_mode) for _ in range(n_envs)])

    env = VecMonitor(env)
    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, training=training)

    return env


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    # Discount factor
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)

    # GAE parameter
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)

    # Neural network architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small"])
    net_arch = {
        "tiny": [64, 64],
        "small": [128, 128],
        "medium": [256, 256],
    }[net_arch_type]

    # Whether to use IMPALA CNN
    # use_impala_cnn = trial.suggest_categorical("use_impala_cnn", [True, False])
    use_impala_cnn = False

    # Features dimension for the feature extractor (only used if IMPALA CNN is enabled)
    features_dim = trial.suggest_categorical(
        "features_dim", [256, 512]) if use_impala_cnn else None

    # Learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lr_schedule = trial.suggest_categorical(
        "lr_schedule", ["linear", "constant"])

    if lr_schedule == 'linear':
        learning_rate = get_linear_fn(
            start=learning_rate,
            end=5e-6,
            end_fraction=0.85
        )

    # Batch size and n_steps
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 7, 11)  # 128 to 2048
    batch_size = min(
        2 ** trial.suggest_int("exponent_batch_size", 5, 9), n_steps)  # 32 to 512

    # Number of epochs
    n_epochs = trial.suggest_int("n_epochs", 4, 12)

    # Entropy coefficient
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.01, log=True)

    # Value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)

    # Clipping parameters
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    clip_range_vf = trial.suggest_categorical(
        "clip_range_vf", [None, 0.1, 0.2, 0.3, 0.4])

    # Max gradient norm
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

    # Store true values for logging
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)
    trial.set_user_attr("use_impala_cnn", use_impala_cnn)

    # Base policy kwargs
    policy_kwargs = {
        "net_arch": net_arch,
    }

    # Add IMPALA CNN settings if enabled
    if use_impala_cnn:
        policy_kwargs.update({
            "features_extractor_class": NPPFeatureExtractorImpala,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "frame_stack": ENABLE_FRAME_STACK
            }
        })

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "clip_range": clip_range,
        "clip_range_vf": clip_range_vf,
        "policy_kwargs": policy_kwargs,
    }


class TrialEvalCallback(EvalCallback):
    """Callback for evaluating and pruning trials during optimization."""

    def __init__(
        self,
        eval_env: VecNormalize,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        callback_after_eval: Optional[BaseCallback] = None,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            callback_after_eval=callback_after_eval,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # Report current mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    """Optimization objective for Optuna."""

    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(f'training_logs/tune_logs/trial_{trial.number}_{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_save_path = Path(
        f'training_logs/tune_logs/best_model_{timestamp}')
    best_model_save_path.mkdir(parents=True, exist_ok=True)

    # Initialize hyperparameters
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_ppo_params(trial))

    # Create environments
    train_env = create_env(n_envs=N_ENVS, training=True)
    eval_env = create_env(
        n_envs=1, enable_short_episode_truncation=True, training=False, eval_mode=True)

    # Create the PPO model
    model = PPO(
        env=train_env,
        tensorboard_log=str(log_dir / "tensorboard"),
        verbose=1,
        seed=42,
        **kwargs
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30, min_evals=50, verbose=1)

    # Create evaluation callback
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=False,
        log_path=str(log_dir),
        best_model_save_path=str(best_model_save_path),
        callback_after_eval=stop_callback,
    )

    nan_encountered = False
    try:
        model.learn(
            total_timesteps=N_TIMESTEPS,
            callback=eval_callback,
            progress_bar=False,
        )
    except AssertionError as e:
        # Handle NaN errors
        print(f"Trial {trial.number} failed with error: {e}")
        nan_encountered = True
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()

    # Save trial results
    results = {
        "trial_number": trial.number,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
    }
    with open(log_dir / "trial_results.json", "w") as f:
        json.dump(results, f, indent=4)

    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


def optimize_agent():
    """Run the hyperparameter optimization."""

    # Set PyTorch threads for faster training
    torch.set_num_threads(1)

    # Initialize sampler and pruner
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=N_WARMUP_STEPS
    )

    # Create study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=f"ppo_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=4)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"Optimization failed with error: {e}")

    print("\nOptimization Results:")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("\n  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Save study results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(f'training_logs/tune_results_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save study statistics
    study_stats = {
        "best_trial": {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        },
        "n_trials": len(study.trials),
        "datetime": timestamp,
    }

    with open(results_dir / "study_results.json", "w") as f:
        json.dump(study_stats, f, indent=4)

    return study


if __name__ == "__main__":
    optimize_agent()
