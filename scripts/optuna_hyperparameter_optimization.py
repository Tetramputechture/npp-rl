#!/usr/bin/env python3
"""Optuna-based hyperparameter optimization for NPP-RL architectures.

Runs hyperparameter optimization using Optuna to find optimal hyperparameters
for a given architecture configuration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from npp_rl.training.architecture_configs import (
    get_architecture_config,
    ARCHITECTURE_REGISTRY,
)
from npp_rl.training.architecture_trainer import ArchitectureTrainer
from npp_rl.training.hyperparameter_search_spaces import get_search_space
from npp_rl.training.optuna_callbacks import OptunaTrialPruningCallback
from npp_rl.utils import setup_experiment_logging, create_s3_uploader

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Unique experiment name"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=list(ARCHITECTURE_REGISTRY.keys()),
        help="Architecture name to optimize",
    )
    parser.add_argument(
        "--train-dataset", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, help="Path to test dataset"
    )

    # Optimization settings
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timesteps-per-trial",
        type=int,
        default=1_000_000,
        help="Training timesteps per trial",
    )
    parser.add_argument(
        "--study-name", type=str, required=True, help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_study.db",
        help="Optuna storage URL (e.g., sqlite:///study.db)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="hpo_results/", help="Output directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing study"
    )

    # S3 upload
    parser.add_argument(
        "--s3-bucket", type=str, default=None, help="S3 bucket for uploads"
    )
    parser.add_argument(
        "--s3-prefix", type=str, default="hpo/", help="S3 prefix for uploads"
    )

    return parser.parse_args()


def objective(trial: optuna.Trial, args, architecture_config) -> float:
    """
    Optuna objective function to minimize.

    Returns: -1 * (0.7 * success_rate + 0.3 * normalized_mean_reward)
    We return negative because Optuna minimizes by default.

    Args:
        trial: Optuna trial object
        args: Parsed command-line arguments
        architecture_config: Architecture configuration

    Returns:
        Negative optimization metric (to minimize)

    Note:
        For 'attention' architecture, ObjectiveAttentionActorCriticPolicy is automatically
        activated by ArchitectureTrainer. This policy includes:
        - Deep ResNet MLPs (5-layer policy, 3-layer value)
        - Objective-specific attention over 1-16 locked doors
        - Dueling value architecture
        - Auxiliary death prediction head
    """
    logger.info(f"Starting trial {trial.number}")

    # Sample hyperparameters from search space
    search_space = get_search_space(args.architecture)
    hyperparams = {}

    # PPO hyperparameters
    hyperparams["learning_rate"] = trial.suggest_float(
        "learning_rate",
        search_space["learning_rate"]["low"],
        search_space["learning_rate"]["high"],
        log=search_space["learning_rate"]["log"],
    )
    hyperparams["n_steps"] = trial.suggest_categorical(
        "n_steps", search_space["n_steps"]["choices"]
    )
    hyperparams["batch_size"] = trial.suggest_categorical(
        "batch_size", search_space["batch_size"]["choices"]
    )
    # Ensure batch_size <= n_steps
    if hyperparams["batch_size"] > hyperparams["n_steps"]:
        hyperparams["batch_size"] = hyperparams["n_steps"]

    hyperparams["gamma"] = trial.suggest_float(
        "gamma",
        search_space["gamma"]["low"],
        search_space["gamma"]["high"],
        log=search_space["gamma"]["log"],
    )
    hyperparams["gae_lambda"] = trial.suggest_float(
        "gae_lambda",
        search_space["gae_lambda"]["low"],
        search_space["gae_lambda"]["high"],
        log=search_space["gae_lambda"]["log"],
    )
    hyperparams["clip_range"] = trial.suggest_float(
        "clip_range",
        search_space["clip_range"]["low"],
        search_space["clip_range"]["high"],
        log=search_space["clip_range"]["log"],
    )
    hyperparams["clip_range_vf"] = trial.suggest_float(
        "clip_range_vf",
        search_space["clip_range_vf"]["low"],
        search_space["clip_range_vf"]["high"],
        log=search_space["clip_range_vf"]["log"],
    )
    hyperparams["ent_coef"] = trial.suggest_float(
        "ent_coef",
        search_space["ent_coef"]["low"],
        search_space["ent_coef"]["high"],
        log=search_space["ent_coef"]["log"],
    )
    hyperparams["vf_coef"] = trial.suggest_float(
        "vf_coef",
        search_space["vf_coef"]["low"],
        search_space["vf_coef"]["high"],
        log=search_space["vf_coef"]["log"],
    )
    hyperparams["max_grad_norm"] = trial.suggest_float(
        "max_grad_norm",
        search_space["max_grad_norm"]["low"],
        search_space["max_grad_norm"]["high"],
        log=search_space["max_grad_norm"]["log"],
    )
    hyperparams["n_epochs"] = trial.suggest_int(
        "n_epochs",
        search_space["n_epochs"]["low"],
        search_space["n_epochs"]["high"],
    )

    # Network architecture
    hyperparams["net_arch_depth"] = trial.suggest_int(
        "net_arch_depth",
        search_space["net_arch_depth"]["low"],
        search_space["net_arch_depth"]["high"],
    )
    hyperparams["net_arch_width"] = trial.suggest_categorical(
        "net_arch_width", search_space["net_arch_width"]["choices"]
    )
    hyperparams["features_dim"] = trial.suggest_categorical(
        "features_dim", search_space["features_dim"]["choices"]
    )

    # Training settings
    num_envs = 128
    hyperparams["num_envs"] = num_envs
    hyperparams["lr_schedule"] = trial.suggest_categorical(
        "lr_schedule", search_space["lr_schedule"]["choices"]
    )

    # Architecture-specific feature extractor params
    if "gnn_num_layers" in search_space:
        hyperparams["gnn_num_layers"] = trial.suggest_int(
            "gnn_num_layers",
            search_space["gnn_num_layers"]["low"],
            search_space["gnn_num_layers"]["high"],
        )
        hyperparams["gnn_hidden_dim"] = trial.suggest_categorical(
            "gnn_hidden_dim", search_space["gnn_hidden_dim"]["choices"]
        )

        if "gnn_num_heads" in search_space:
            hyperparams["gnn_num_heads"] = trial.suggest_int(
                "gnn_num_heads",
                search_space["gnn_num_heads"]["low"],
                search_space["gnn_num_heads"]["high"],
            )

    if "cnn_base_channels" in search_space:
        hyperparams["cnn_base_channels"] = trial.suggest_categorical(
            "cnn_base_channels", search_space["cnn_base_channels"]["choices"]
        )
        hyperparams["cnn_num_layers"] = trial.suggest_int(
            "cnn_num_layers",
            search_space["cnn_num_layers"]["low"],
            search_space["cnn_num_layers"]["high"],
        )

    # Create output directory for this trial
    trial_output_dir = Path(args.output_dir) / args.study_name / f"trial_{trial.number}"
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer with sampled hyperparameters
    trainer = ArchitectureTrainer.from_hyperparameter_dict(
        architecture_config=architecture_config,
        hyperparameters=hyperparams,
        train_dataset_path=args.train_dataset,
        test_dataset_path=args.test_dataset,
        output_dir=str(trial_output_dir),
        experiment_name=f"{args.experiment_name}_trial_{trial.number}",
        use_objective_attention_policy=use_objective_attention,
        use_curriculum=True,
    )

    trainer.setup_model(**hyperparams)

    trainer.setup_environments(
        num_envs=num_envs,
        total_timesteps=args.timesteps_per_trial,
    )

    pruning_callback = OptunaTrialPruningCallback(
        trial=trial,
        eval_freq=250_000,
        trainer=trainer,
        verbose=1,
    )

    # Train with pruning
    try:
        trainer.train(
            total_timesteps=args.timesteps_per_trial,
            eval_freq=250_000,
            save_freq=500_000,
            callback_fn=pruning_callback,
        )

        # Final evaluation on test set
        logger.info(f"Running final evaluation for trial {trial.number}...")
        eval_results = trainer.evaluate(
            num_episodes=20,
            categories_to_evaluate=[
                "simplest",
                "simplest_few_mines",
                "simplest_with_mines",
            ],
        )

        # Calculate optimization metric
        success_rate = eval_results.get("success_rate", 0.0)
        # Use avg_reward from ComprehensiveEvaluator results
        mean_reward = eval_results.get(
            "avg_reward", eval_results.get("mean_reward", 0.0)
        )

        # Normalize reward (assuming range -1000 to 1000)
        normalized_reward = max(0.0, min(1.0, (mean_reward + 1000) / 2000))

        # Combined metric (70% success rate, 30% normalized reward)
        metric = 0.7 * success_rate + 0.3 * normalized_reward

        logger.info(
            f"Trial {trial.number} completed: "
            f"metric={metric:.4f}, "
            f"success_rate={success_rate:.2%}, "
            f"mean_reward={mean_reward:.2f}"
        )

        # Return negative for minimization
        return -metric

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise  # Re-raise to let Optuna handle it
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}", exc_info=True)
        return float("inf")  # Return worst possible value
    finally:
        trainer.cleanup()


def save_optimization_results(study: optuna.Study, args) -> None:
    """Save optimization results to files."""
    output_dir = Path(args.output_dir) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best hyperparameters
    best_params = study.best_params
    best_value = study.best_value

    best_hyperparams = {
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "architecture": args.architecture,
    }

    best_params_path = output_dir / f"best_hyperparameters_{args.architecture}.json"
    with open(best_params_path, "w") as f:
        json.dump(best_hyperparams, f, indent=2)

    logger.info(f"Saved best hyperparameters to {best_params_path}")
    logger.info(f"Best value: {best_value:.4f}")
    logger.info(f"Best params: {best_params}")


def generate_visualizations(study: optuna.Study, args) -> None:
    """Generate Optuna visualization plots."""
    output_dir = Path(args.output_dir) / args.study_name / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(str(output_dir / "optimization_history.png"))
        logger.info("Saved optimization history plot")

        # Parameter importance
        fig = plot_param_importances(study)
        fig.write_image(str(output_dir / "param_importances.png"))
        logger.info("Saved parameter importance plot")

        # Contour plot (if enough trials)
        if len(study.trials) >= 10:
            try:
                fig = plot_contour(study)
                fig.write_image(str(output_dir / "contour.png"))
                logger.info("Saved contour plot")
            except Exception as e:
                print(f"Could not generate contour plot: {e}")

    except Exception as e:
        print(f"Could not generate some visualizations: {e}")


def upload_to_s3(args) -> None:
    """Upload results to S3."""
    if not args.s3_bucket:
        return

    try:
        s3_uploader = create_s3_uploader(args.s3_bucket, args.s3_prefix)
        output_dir = Path(args.output_dir) / args.study_name

        # Upload best hyperparameters
        best_params_path = output_dir / f"best_hyperparameters_{args.architecture}.json"
        if best_params_path.exists():
            s3_uploader.upload_file(
                str(best_params_path),
                f"{args.s3_prefix}best_hyperparameters_{args.architecture}.json",
            )

        # Upload study database
        study_db_path = Path(args.storage.replace("sqlite:///", ""))
        if study_db_path.exists():
            s3_uploader.upload_file(
                str(study_db_path),
                f"{args.s3_prefix}optuna_study.db",
            )

        logger.info(f"Uploaded results to s3://{args.s3_bucket}/{args.s3_prefix}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_dir = Path(args.output_dir) / args.study_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_experiment_logging(log_dir, args.experiment_name)

    logger.info("=" * 60)
    logger.info("Hyperparameter Optimization with Optuna")
    logger.info("=" * 60)
    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Number of trials: {args.num_trials}")
    logger.info(f"Timesteps per trial: {args.timesteps_per_trial:,}")
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Storage: {args.storage}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load architecture config
    architecture_config = get_architecture_config(args.architecture)
    logger.info(f"Architecture config: {architecture_config.name}")

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        direction="minimize",  # We return negative metric
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=2,  # Need at least 2 evaluations before pruning
            interval_steps=1,  # Check at every evaluation
        ),
    )

    logger.info(f"Study created/loaded: {args.study_name}")
    logger.info(f"Existing trials: {len(study.trials)}")

    # Run optimization
    logger.info("Starting optimization...")
    study.optimize(
        lambda trial: objective(trial, args, architecture_config),
        n_trials=args.num_trials,
        show_progress_bar=True,
    )

    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Save results
    save_optimization_results(study, args)

    # Generate visualizations
    generate_visualizations(study, args)

    # Upload to S3 if specified
    upload_to_s3(args)

    logger.info("=" * 60)
    logger.info("Hyperparameter optimization complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
