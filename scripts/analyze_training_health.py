#!/usr/bin/env python3
"""Training Health Analysis Script

Analyzes TensorBoard logs to detect major errors, inconsistencies, and critical bugs
in training runs. Focuses on error detection rather than performance optimization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TrainingHealthAnalyzer:
    """Analyzes TensorBoard logs for errors and inconsistencies."""

    def __init__(self, events_file: str):
        """Initialize analyzer with TensorBoard events file.

        Args:
            events_file: Path to TensorBoard events file
        """
        self.events_file = Path(events_file)
        if not self.events_file.exists():
            raise FileNotFoundError(f"Events file not found: {events_file}")

        self.data: Dict[str, pd.DataFrame] = {}
        self.tags: List[str] = []
        self.issues: List[Dict[str, Any]] = []

        logger.info(f"Loading TensorBoard data from {events_file}...")
        self._load_data()

    def _load_data(self) -> None:
        """Load all scalar data from TensorBoard events file."""
        # Load events
        ea = event_accumulator.EventAccumulator(str(self.events_file))
        ea.Reload()

        # Get all available tags
        self.tags = ea.Tags().get("scalars", [])
        logger.info(f"Found {len(self.tags)} scalar metrics")

        # Extract all scalar data
        for tag in self.tags:
            events = ea.Scalars(tag)
            self.data[tag] = pd.DataFrame(
                [
                    {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                    for e in events
                ]
            )

        logger.info(f"Loaded data for {len(self.data)} metrics")

    def _add_issue(
        self,
        severity: str,
        metric: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an issue to the issues list.

        Args:
            severity: 'critical', 'warning', or 'info'
            metric: Name of the metric with the issue
            description: Human-readable description
            details: Additional details dictionary
        """
        self.issues.append(
            {
                "severity": severity,
                "metric": metric,
                "description": description,
                "details": details or {},
            }
        )

    def check_data_quality(self) -> None:
        """Check for NaN, Inf, and other data quality issues."""
        logger.info("\n=== Checking Data Quality ===")

        for tag, df in self.data.items():
            if len(df) == 0:
                self._add_issue("warning", tag, "Metric has no data points")
                continue

            values = df["value"].values

            # Check for NaN
            nan_count = np.isnan(values).sum()
            if nan_count > 0:
                self._add_issue(
                    "critical",
                    tag,
                    f"Found {nan_count} NaN values out of {len(values)} total",
                    {"nan_count": int(nan_count), "total_count": len(values)},
                )

            # Check for Inf
            inf_count = np.isinf(values).sum()
            if inf_count > 0:
                self._add_issue(
                    "critical",
                    tag,
                    f"Found {inf_count} Inf values out of {len(values)} total",
                    {"inf_count": int(inf_count), "total_count": len(values)},
                )

            # Check for constant values (not updating)
            if len(values) > 10:
                unique_vals = np.unique(values)
                if len(unique_vals) == 1:
                    self._add_issue(
                        "warning",
                        tag,
                        f"Metric is constant (value: {values[0]})",
                        {"constant_value": float(values[0])},
                    )
                elif (
                    len(unique_vals) < len(values) * 0.1
                ):  # Less than 10% unique values
                    self._add_issue(
                        "info",
                        tag,
                        f"Metric has very few unique values ({len(unique_vals)} unique out of {len(values)} total)",
                        {"unique_count": len(unique_vals), "total_count": len(values)},
                    )

    def check_training_stability(self) -> None:
        """Check for training instability issues."""
        logger.info("\n=== Checking Training Stability ===")

        # Check losses
        loss_tags = [t for t in self.tags if "loss" in t.lower()]
        for tag in loss_tags:
            df = self.data[tag]
            if len(df) < 2:
                continue

            values = df["value"].values

            # Check for loss explosions (sudden spike >100x median)
            if len(values) > 10:
                median_val = np.median(values)
                if median_val > 0:
                    max_val = np.max(values)
                    if max_val > median_val * 100:
                        self._add_issue(
                            "critical",
                            tag,
                            f"Loss explosion detected: max={max_val:.2e}, median={median_val:.2e}",
                            {
                                "max": float(max_val),
                                "median": float(median_val),
                                "ratio": float(max_val / median_val),
                            },
                        )

            # Check for sudden drops to zero (training collapse)
            if len(values) > 10:
                recent_vals = values[-10:]
                if np.all(recent_vals == 0) and np.any(values[:-10] != 0):
                    self._add_issue(
                        "critical",
                        tag,
                        "Loss collapsed to zero in recent steps",
                        {"last_10_values": recent_vals.tolist()},
                    )

        # Check entropy (should not collapse too fast)
        entropy_tags = [
            t for t in self.tags if "entropy" in t.lower() and "loss" not in t.lower()
        ]
        for tag in entropy_tags:
            df = self.data[tag]
            if len(df) < 2:
                continue

            values = df["value"].values
            if len(values) > 10:
                initial = np.mean(values[:10])
                final = np.mean(values[-10:])
                if initial > 0.1 and final < 0.01:
                    self._add_issue(
                        "critical",
                        tag,
                        f"Entropy collapsed: initial={initial:.4f}, final={final:.4f}",
                        {"initial": float(initial), "final": float(final)},
                    )

        # Check gradient norms
        grad_tags = [
            t for t in self.tags if "gradient" in t.lower() or "grad_norm" in t.lower()
        ]
        for tag in grad_tags:
            df = self.data[tag]
            if len(df) == 0:
                continue

            values = df["value"].values
            if len(values) > 0:
                max_grad = np.max(values)
                if max_grad > 100.0:
                    self._add_issue(
                        "critical",
                        tag,
                        f"Gradient explosion detected: max={max_grad:.2f}",
                        {"max_gradient": float(max_grad)},
                    )
                elif max_grad < 0.01 and len(values) > 10:
                    self._add_issue(
                        "warning",
                        tag,
                        f"Very small gradients: max={max_grad:.6f} (possible vanishing gradients)",
                        {"max_gradient": float(max_grad)},
                    )

    def check_reward_health(self) -> None:
        """Check for reward-related issues."""
        logger.info("\n=== Checking Reward Health ===")

        reward_tags = [t for t in self.tags if "reward" in t.lower()]
        episode_tags = [
            t for t in self.tags if "episode" in t.lower() and "reward" in t.lower()
        ]

        # Check for reward collapse
        for tag in reward_tags:
            df = self.data[tag]
            if len(df) < 10:
                continue

            values = df["value"].values
            if len(values) > 10:
                initial_mean = np.mean(values[:10])
                final_mean = np.mean(values[-10:])

                # Check for collapse to zero or negative
                if initial_mean > 0.1 and final_mean < -0.5:
                    self._add_issue(
                        "critical",
                        tag,
                        f"Reward collapsed: initial={initial_mean:.4f}, final={final_mean:.4f}",
                        {"initial": float(initial_mean), "final": float(final_mean)},
                    )

        # Check for reward consistency (total vs components)
        if "episode/reward_mean" in self.data:
            total_reward_df = self.data["episode/reward_mean"]
            if len(total_reward_df) > 0:
                # Check if reward components sum correctly
                component_tags = [
                    t
                    for t in reward_tags
                    if any(
                        comp in t.lower()
                        for comp in ["intrinsic", "extrinsic", "pbrs", "hierarchical"]
                    )
                ]

                if component_tags:
                    for component_tag in component_tags:
                        if component_tag in self.data:
                            comp_df = self.data[component_tag]
                            # Try to align steps
                            if len(comp_df) > 0:
                                # Check if component is much larger than total (inconsistency)
                                total_mean = total_reward_df["value"].mean()
                                comp_mean = comp_df["value"].mean()
                                if (
                                    abs(comp_mean) > abs(total_mean) * 10
                                    and abs(total_mean) > 0.01
                                ):
                                    self._add_issue(
                                        "warning",
                                        component_tag,
                                        f"Reward component magnitude inconsistent with total: "
                                        f"component={comp_mean:.4f}, total={total_mean:.4f}",
                                        {
                                            "component_mean": float(comp_mean),
                                            "total_mean": float(total_mean),
                                        },
                                    )

    def check_curriculum_progression(self) -> None:
        """Check for curriculum learning issues."""
        logger.info("\n=== Checking Curriculum Progression ===")

        curriculum_tags = [t for t in self.tags if "curriculum" in t.lower()]

        if not curriculum_tags:
            self._add_issue("info", "curriculum", "No curriculum metrics found")
            return

        # Check stage progression
        stage_tags = [
            t
            for t in curriculum_tags
            if "stage" in t.lower() or "timeline" in t.lower()
        ]
        for tag in stage_tags:
            df = self.data[tag]
            if len(df) < 2:
                continue

            values = df["value"].values
            initial = values[0]
            final = values[-1]

            # Check if curriculum never progressed
            if initial == final and len(values) > 50:
                self._add_issue(
                    "warning",
                    tag,
                    f"Curriculum stage never changed: stuck at {initial}",
                    {"initial_stage": float(initial), "final_stage": float(final)},
                )

            # Check for regression (going backwards)
            if final < initial:
                self._add_issue(
                    "warning",
                    tag,
                    f"Curriculum regressed: {initial} -> {final}",
                    {"initial_stage": float(initial), "final_stage": float(final)},
                )

        # Check success rates by stage
        success_tags = [t for t in curriculum_tags if "success" in t.lower()]
        for tag in success_tags:
            df = self.data[tag]
            if len(df) < 10:
                continue

            values = df["value"].values
            # Check if success rate is stuck at zero
            if np.all(values == 0) and len(values) > 20:
                self._add_issue(
                    "warning",
                    tag,
                    f"Success rate stuck at zero for {len(values)} steps",
                    {"zero_count": len(values)},
                )

    def check_missing_metrics(self) -> None:
        """Check for expected metrics that are missing."""
        logger.info("\n=== Checking for Missing Metrics ===")

        # Expected metrics based on the codebase (use flexible matching)
        # Check for existence of key metric categories rather than exact names
        expected_categories = {
            "episode": [
                "episode/success_rate",
                "episode/length_mean",
                "episode/reward_mean",
            ],
            "curriculum_reward": [
                "curriculum/simplest/reward_mean",
                "curriculum/simpler/reward_mean",
            ],
            "loss": ["loss/value", "loss/total", "loss/entropy"],
            "training": [
                "training/clip_fraction",
                "training/learning_rate",
                "training/explained_variance",
            ],
        }

        missing = []
        for category, metrics in expected_categories.items():
            found = any(m in self.data for m in metrics)
            if not found:
                missing.extend(metrics)

        if missing:
            self._add_issue(
                "warning",
                "missing_metrics",
                f"Missing {len(missing)} expected metrics",
                {"missing_metrics": missing},
            )

    def check_metric_consistency(self) -> None:
        """Check for logical inconsistencies between related metrics."""
        logger.info("\n=== Checking Metric Consistency ===")

        # Check episode length vs reward (should correlate)
        # Try different possible metric names
        length_metric = None
        reward_metric = None

        for tag in self.tags:
            if "episode" in tag.lower() and "length" in tag.lower():
                length_metric = tag
                break

        # Check for reward metrics (could be episode/reward_mean or curriculum/*/reward_mean)
        for tag in self.tags:
            if "reward_mean" in tag and ("episode" in tag or "curriculum" in tag):
                reward_metric = tag
                break

        if (
            length_metric
            and reward_metric
            and length_metric in self.data
            and reward_metric in self.data
        ):
            length_df = self.data[length_metric]
            reward_df = self.data[reward_metric]

            if len(length_df) > 10 and len(reward_df) > 10:
                # Try to align by step
                length_vals = length_df["value"].values
                reward_vals = reward_df["value"].values

                # Check if both are zero (nothing happening)
                if np.all(length_vals == 0) and np.all(reward_vals == 0):
                    self._add_issue(
                        "critical",
                        "episode_metrics",
                        f"Both episode length ({length_metric}) and reward ({reward_metric}) are zero - training may not be running",
                        {},
                    )

        # Check success rate consistency
        if "episode/success_rate" in self.data:
            success_df = self.data["episode/success_rate"]
            values = success_df["value"].values
            if len(values) > 0:
                # Success rate should be between 0 and 1
                if np.any(values < 0) or np.any(values > 1):
                    invalid_count = np.sum((values < 0) | (values > 1))
                    self._add_issue(
                        "critical",
                        "episode/success_rate",
                        f"Success rate has {invalid_count} values outside [0, 1]",
                        {
                            "invalid_count": int(invalid_count),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                        },
                    )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        logger.info("\n=== Generating Report ===")

        # Categorize issues by severity
        critical_issues = [i for i in self.issues if i["severity"] == "critical"]
        warning_issues = [i for i in self.issues if i["severity"] == "warning"]
        info_issues = [i for i in self.issues if i["severity"] == "info"]

        report = {
            "summary": {
                "total_issues": len(self.issues),
                "critical": len(critical_issues),
                "warnings": len(warning_issues),
                "info": len(info_issues),
                "total_metrics": len(self.tags),
                "metrics_checked": len(self.data),
            },
            "critical_issues": critical_issues,
            "warnings": warning_issues,
            "info": info_issues,
            "all_issues": self.issues,
        }

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print human-readable report to console."""
        print("\n" + "=" * 80)
        print("TRAINING HEALTH ANALYSIS REPORT")
        print("=" * 80)

        summary = report["summary"]
        print("\nSummary:")
        print(f"  Total metrics analyzed: {summary['total_metrics']}")
        print(f"  Total issues found: {summary['total_issues']}")
        print(f"    - Critical: {summary['critical']}")
        print(f"    - Warnings: {summary['warnings']}")
        print(f"    - Info: {summary['info']}")

        # Critical issues
        if report["critical_issues"]:
            print("\n" + "=" * 80)
            print("CRITICAL ISSUES (Must Fix)")
            print("=" * 80)
            for i, issue in enumerate(report["critical_issues"], 1):
                print(f"\n{i}. [{issue['metric']}]")
                print(f"   {issue['description']}")
                if issue["details"]:
                    for key, value in issue["details"].items():
                        print(f"   {key}: {value}")

        # Warnings
        if report["warnings"]:
            print("\n" + "=" * 80)
            print("WARNINGS (Should Investigate)")
            print("=" * 80)
            for i, issue in enumerate(report["warnings"], 1):
                print(f"\n{i}. [{issue['metric']}]")
                print(f"   {issue['description']}")
                if issue["details"]:
                    for key, value in issue["details"].items():
                        print(f"   {key}: {value}")

        # Info
        if report["info"]:
            print("\n" + "=" * 80)
            print("INFO (Minor Observations)")
            print("=" * 80)
            for i, issue in enumerate(report["info"], 1):
                print(f"\n{i}. [{issue['metric']}]")
                print(f"   {issue['description']}")

        # Recommendations
        if report["critical_issues"] or report["warnings"]:
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS")
            print("=" * 80)

            if any(
                "NaN" in i["description"] or "Inf" in i["description"]
                for i in report["critical_issues"]
            ):
                print("\n1. NaN/Inf Values:")
                print("   - Check for division by zero in reward calculations")
                print("   - Verify gradient clipping is working")
                print("   - Check for invalid inputs to mathematical operations")

            if any(
                "explosion" in i["description"].lower()
                for i in report["critical_issues"]
            ):
                print("\n2. Training Instability:")
                print("   - Reduce learning rate")
                print("   - Increase gradient clipping threshold")
                print("   - Check for reward scaling issues")

            if any(
                "collapse" in i["description"].lower()
                for i in report["critical_issues"]
            ):
                print("\n3. Training Collapse:")
                print("   - Check if environment is resetting properly")
                print("   - Verify reward function is working")
                print("   - Check for numerical overflow/underflow")

            if any("missing" in i["description"].lower() for i in report["warnings"]):
                print("\n4. Missing Metrics:")
                print("   - Verify logging callbacks are properly configured")
                print("   - Check if training actually started")

        print("\n" + "=" * 80)

    def analyze(self) -> Dict[str, Any]:
        """Run all analyses and generate report."""
        logger.info("Starting analysis...")

        self.check_data_quality()
        self.check_training_stability()
        self.check_reward_health()
        self.check_curriculum_progression()
        self.check_missing_metrics()
        self.check_metric_consistency()

        report = self.generate_report()
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TensorBoard logs for training health and errors"
    )
    parser.add_argument("events_file", type=str, help="Path to TensorBoard events file")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON report (optional)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    analyzer = TrainingHealthAnalyzer(args.events_file)
    report = analyzer.analyze()

    # Print report
    analyzer.print_report(report)

    # Save JSON report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
