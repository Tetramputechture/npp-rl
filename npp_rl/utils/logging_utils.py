"""Logging utilities for training experiments.

Provides structured logging setup, TensorBoard helpers, and experiment
configuration management.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


def setup_experiment_logging(
    output_dir: Path,
    experiment_name: str,
    log_level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """Set up structured logging for an experiment.

    Args:
        output_dir: Base output directory
        experiment_name: Experiment name for log files
        log_level: Logging level (default: INFO)
        console_output: If True, also log to console

    Returns:
        Configured logger instance
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logger
    logger = logging.getLogger("npp_rl.training")
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Prevent propagation to parent loggers to avoid duplicate messages
    logger.propagate = False

    # File handler
    log_file = output_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(log_level)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    if console_output:
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info(f"Logging initialized: {log_file}")
    logger.info(f"Experiment: {experiment_name}")

    return logger


def save_experiment_config(
    config: Dict[str, Any], output_dir: Path, filename: str = "config.json"
) -> None:
    """Save experiment configuration to JSON.

    Args:
        config: Configuration dictionary
        output_dir: Output directory
        filename: Output filename (default: config.json)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / filename

    # Add timestamp
    config_with_meta = {
        **config,
        "saved_at": datetime.now().isoformat(),
    }

    with open(config_path, "w") as f:
        json.dump(config_with_meta, f, indent=2, default=str)

    logging.info(f"Saved configuration to {config_path}")


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    logging.info(f"Loaded configuration from {config_path}")
    return config


class TensorBoardManager:
    """Manages multiple TensorBoard writers for hierarchical logging."""

    def __init__(self, base_dir: Path):
        """Initialize TensorBoard manager.

        Args:
            base_dir: Base directory for all TensorBoard logs
        """
        self.base_dir = Path(base_dir)
        self.writers: Dict[str, SummaryWriter] = {}

    def get_writer(self, name: str, subdir: Optional[str] = None) -> SummaryWriter:
        """Get or create a TensorBoard writer.

        Args:
            name: Writer name (used as key)
            subdir: Optional subdirectory within base_dir

        Returns:
            SummaryWriter instance
        """
        if name in self.writers:
            return self.writers[name]

        if subdir:
            log_dir = self.base_dir / subdir
        else:
            log_dir = self.base_dir / name

        log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(str(log_dir))
        self.writers[name] = writer

        logging.info(f"Created TensorBoard writer '{name}': {log_dir}")
        return writer

    def log_scalar(self, name: str, tag: str, value: float, step: int) -> None:
        """Log a scalar value to a named writer.

        Args:
            name: Writer name
            tag: Metric tag
            value: Scalar value
            step: Global step
        """
        writer = self.get_writer(name)
        writer.add_scalar(tag, value, step)

    def log_scalars(
        self, name: str, tag: str, values: Dict[str, float], step: int
    ) -> None:
        """Log multiple related scalars.

        Args:
            name: Writer name
            tag: Main tag
            values: Dict of metric_name -> value
            step: Global step
        """
        writer = self.get_writer(name)
        writer.add_scalars(tag, values, step)

    def flush_all(self) -> None:
        """Flush all writers."""
        for writer in self.writers.values():
            writer.flush()

    def close_all(self) -> None:
        """Close all writers."""
        for name, writer in self.writers.items():
            writer.close()
            logging.info(f"Closed TensorBoard writer '{name}'")
        self.writers.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close all writers."""
        self.close_all()


def log_training_step(
    writer: SummaryWriter,
    timestep: int,
    metrics: Dict[str, float],
    prefix: str = "train",
) -> None:
    """Log training step metrics to TensorBoard.

    Args:
        writer: TensorBoard writer
        timestep: Current timestep
        metrics: Dictionary of metric_name -> value
        prefix: Metric prefix (default: 'train')
    """
    for key, value in metrics.items():
        if value is not None and not (
            isinstance(value, float) and (value != value or abs(value) == float("inf"))
        ):
            writer.add_scalar(f"{prefix}/{key}", value, timestep)


def log_evaluation(
    writer: SummaryWriter,
    timestep: int,
    eval_results: Dict[str, Any],
    prefix: str = "eval",
) -> None:
    """Log evaluation results to TensorBoard.

    Args:
        writer: TensorBoard writer
        timestep: Current timestep
        eval_results: Evaluation results dictionary
        prefix: Metric prefix (default: 'eval')
    """
    # Overall metrics
    if "overall" in eval_results:
        for key, value in eval_results["overall"].items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"{prefix}/overall_{key}", value, timestep)

    # Per-category metrics
    if "by_category" in eval_results:
        for category, metrics in eval_results["by_category"].items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"{prefix}/{category}_{key}", value, timestep)


def create_experiment_summary(
    experiment_dir: Path, architecture_results: Dict[str, Dict[str, Any]]
) -> str:
    """Create a human-readable experiment summary.

    Args:
        experiment_dir: Experiment output directory
        architecture_results: Results dict keyed by architecture name

    Returns:
        Markdown-formatted summary string
    """
    lines = [
        "# Experiment Summary\n",
        f"**Directory**: `{experiment_dir}`\n",
        f"**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Architecture Results\n",
    ]

    for arch_name, results in architecture_results.items():
        lines.append(f"\n### {arch_name}\n")

        if "final_performance" in results:
            perf = results["final_performance"]
            lines.append(f"- **Success Rate**: {perf.get('success_rate', 0):.2%}")
            lines.append(f"- **Avg Steps**: {perf.get('avg_steps', 0):.1f}")
            lines.append(f"- **Efficiency**: {perf.get('efficiency', 0):.3f}")

        if "training_time" in results:
            hours = results["training_time"] / 3600
            lines.append(f"- **Training Time**: {hours:.1f} hours")

        if "inference_time_ms" in results:
            lines.append(f"- **Inference Time**: {results['inference_time_ms']:.2f} ms")

    return "\n".join(lines)


def setup_comparison_logging(
    output_dir: Path, architectures: list
) -> TensorBoardManager:
    """Set up TensorBoard logging for architecture comparison.

    Args:
        output_dir: Base output directory
        architectures: List of architecture names

    Returns:
        TensorBoardManager with writers for each architecture
    """
    tb_manager = TensorBoardManager(output_dir / "tensorboard")

    # Create main comparison writer
    tb_manager.get_writer("comparison")

    # Create per-architecture writers
    for arch_name in architectures:
        tb_manager.get_writer(arch_name, subdir=arch_name)

    return tb_manager
