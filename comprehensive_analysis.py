#!/usr/bin/env python3
"""
Comprehensive RL Training Analysis for NPP-RL
Analyzes TensorBoard logs, curriculum progression, and training metrics
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class NPPRLAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.config = None
        self.all_results = None
        self.tb_data = None
        self.curriculum_stats = defaultdict(list)

    def load_data(self):
        """Load all data files"""
        print("Loading configuration...")
        with open(self.results_dir / "config.json", "r") as f:
            self.config = json.load(f)

        print("Loading all_results...")
        with open(self.results_dir / "all_results.json", "r") as f:
            self.all_results = json.load(f)

        print("Loading TensorBoard events...")
        self._load_tensorboard_data()

        print("Parsing training log...")
        self._parse_training_log()

    def _load_tensorboard_data(self):
        """Load TensorBoard event files"""
        event_file = list(self.results_dir.glob("events.out.tfevents.*"))[0]

        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()

        # Get all available tags
        print("\nAvailable TensorBoard tags:")
        all_tags = ea.Tags()
        for category, tags in all_tags.items():
            if tags:
                print(f"\n{category}:")
                for tag in sorted(tags):
                    print(f"  - {tag}")

        # Extract scalar data
        self.tb_data = {}
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            self.tb_data[tag] = pd.DataFrame(
                [
                    {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                    for e in events
                ]
            )

    def _parse_training_log(self):
        """Parse training log for curriculum and episode information"""
        log_file = self.results_dir / "mlp-1029-f3-corridors-2.trainlog"

        with open(log_file, "r") as f:
            for line in f:
                if "Recording episode for stage:" in line:
                    # Parse: stage: simple, generator: single_chamber:obstacle, success: True, frames: 1331
                    parts = line.split("Recording episode for stage:")[1].strip()
                    stage = parts.split(",")[0].strip()
                    generator = parts.split("generator:")[1].split(",")[0].strip()
                    success = "success: True" in parts
                    frames = int(parts.split("frames:")[1].strip())

                    self.curriculum_stats["stage"].append(stage)
                    self.curriculum_stats["generator"].append(generator)
                    self.curriculum_stats["success"].append(success)
                    self.curriculum_stats["frames"].append(frames)

        self.curriculum_df = pd.DataFrame(self.curriculum_stats)
        print(f"\nParsed {len(self.curriculum_df)} episodes from training log")

    def analyze_curriculum_progression(self):
        """Analyze curriculum learning progression"""
        print("\n" + "=" * 80)
        print("CURRICULUM PROGRESSION ANALYSIS")
        print("=" * 80)

        # Overall success rates by stage
        print("\n1. Success Rate by Curriculum Stage:")
        stage_stats = (
            self.curriculum_df.groupby("stage")
            .agg(
                {
                    "success": ["sum", "count", "mean"],
                    "frames": ["mean", "std", "median"],
                }
            )
            .round(3)
        )
        print(stage_stats)

        # Success rates by generator
        print("\n2. Success Rate by Level Generator:")
        gen_stats = (
            self.curriculum_df.groupby("generator")
            .agg(
                {
                    "success": ["sum", "count", "mean"],
                    "frames": ["mean", "std", "median"],
                }
            )
            .round(3)
        )
        print(gen_stats.sort_values(("success", "mean"), ascending=False))

        # Time series of success rates (rolling window)
        print("\n3. Success Rate Over Time (100-episode rolling window):")
        self.curriculum_df["success_float"] = self.curriculum_df["success"].astype(
            float
        )
        self.curriculum_df["success_rolling"] = (
            self.curriculum_df["success_float"]
            .rolling(window=100, min_periods=10)
            .mean()
        )

        # Plot curriculum progression
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Success rate over time
        ax = axes[0]
        ax.plot(
            self.curriculum_df.index, self.curriculum_df["success_rolling"], linewidth=2
        )
        ax.set_title(
            "Success Rate Over Time (100-episode rolling average)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.grid(True, alpha=0.3)

        # Frames per episode over time
        ax = axes[1]
        self.curriculum_df["frames_rolling"] = (
            self.curriculum_df["frames"].rolling(window=100, min_periods=10).mean()
        )
        ax.plot(
            self.curriculum_df.index,
            self.curriculum_df["frames_rolling"],
            linewidth=2,
            color="orange",
        )
        ax.axhline(
            y=5000,
            color="red",
            linestyle="--",
            label="Timeout (5000 frames)",
            alpha=0.7,
        )
        ax.set_title(
            "Average Episode Length Over Time (100-episode rolling average)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Frames")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Stage distribution over time
        ax = axes[2]
        stage_counts = []
        window = 100
        for i in range(len(self.curriculum_df)):
            start_idx = max(0, i - window + 1)
            window_data = self.curriculum_df.iloc[start_idx : i + 1]
            stage_dist = window_data["stage"].value_counts(normalize=True)
            stage_counts.append(stage_dist)

        stage_df = pd.DataFrame(stage_counts, index=self.curriculum_df.index).fillna(0)
        for stage in stage_df.columns:
            ax.plot(stage_df.index, stage_df[stage], label=stage, linewidth=2)

        ax.set_title(
            "Curriculum Stage Distribution Over Time (100-episode rolling window)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Proportion")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "curriculum_progression.png",
            dpi=150,
            bbox_inches="tight",
        )
        print(f"\nSaved: {self.results_dir / 'curriculum_progression.png'}")

    def analyze_tensorboard_metrics(self):
        """Analyze TensorBoard metrics"""
        print("\n" + "=" * 80)
        print("TENSORBOARD METRICS ANALYSIS")
        print("=" * 80)

        # Create comprehensive plots
        metrics_to_plot = {
            "Rewards": ["rollout/ep_rew_mean", "train/episode_reward"],
            "Losses": ["train/policy_loss", "train/value_loss", "train/entropy_loss"],
            "Learning": ["train/learning_rate", "train/clip_fraction"],
            "Value Estimates": ["train/explained_variance", "train/value_loss"],
        }

        # Plot each category
        for category, metric_list in metrics_to_plot.items():
            available_metrics = [m for m in metric_list if m in self.tb_data]
            if not available_metrics:
                continue

            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics))
            if n_metrics == 1:
                axes = [axes]

            for i, metric in enumerate(available_metrics):
                df = self.tb_data[metric]
                axes[i].plot(df["step"], df["value"], linewidth=1.5)
                axes[i].set_title(metric, fontsize=12, fontweight="bold")
                axes[i].set_xlabel("Timestep")
                axes[i].set_ylabel("Value")
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                mean_val = df["value"].mean()
                std_val = df["value"].std()
                axes[i].axhline(
                    y=mean_val,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Mean: {mean_val:.4f}",
                )
                axes[i].legend()

            plt.tight_layout()
            filename = f"{category.lower().replace(' ', '_')}_analysis.png"
            plt.savefig(self.results_dir / filename, dpi=150, bbox_inches="tight")
            print(f"Saved: {self.results_dir / filename}")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("METRIC STATISTICS")
        print("=" * 80)

        for metric_name, df in self.tb_data.items():
            if df.empty:
                continue
            print(f"\n{metric_name}:")
            print(f"  Mean: {df['value'].mean():.6f}")
            print(f"  Std:  {df['value'].std():.6f}")
            print(f"  Min:  {df['value'].min():.6f}")
            print(f"  Max:  {df['value'].max():.6f}")

            # Check for trends
            if len(df) > 10:
                # Simple linear regression for trend
                x = df["step"].values
                y = df["value"].values
                z = np.polyfit(x, y, 1)
                trend = "increasing" if z[0] > 0 else "decreasing"
                print(f"  Trend: {trend} (slope: {z[0]:.8f})")

    def analyze_action_distribution(self):
        """Analyze action distribution from TensorBoard"""
        print("\n" + "=" * 80)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 80)

        action_metrics = [k for k in self.tb_data.keys() if "action" in k.lower()]

        if not action_metrics:
            print("No action distribution metrics found in TensorBoard data")
            return

        for metric in action_metrics:
            df = self.tb_data[metric]
            print(f"\n{metric}:")
            print(f"  Mean: {df['value'].mean():.4f}")
            print(f"  Std:  {df['value'].std():.4f}")

    def analyze_reward_components(self):
        """Analyze reward structure"""
        print("\n" + "=" * 80)
        print("REWARD STRUCTURE ANALYSIS")
        print("=" * 80)

        # Find reward-related metrics
        reward_metrics = [
            k
            for k in self.tb_data.keys()
            if "reward" in k.lower() or "rew" in k.lower()
        ]

        if reward_metrics:
            print("\nReward Metrics Available:")
            for metric in sorted(reward_metrics):
                df = self.tb_data[metric]
                print(f"\n{metric}:")
                print(f"  Mean: {df['value'].mean():.4f}")
                print(f"  Std:  {df['value'].std():.4f}")
                print(f"  Min:  {df['value'].min():.4f}")
                print(f"  Max:  {df['value'].max():.4f}")

                # Plot reward evolution
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df["step"], df["value"], linewidth=1.5, alpha=0.7)

                # Add rolling mean
                window = max(1, len(df) // 20)
                rolling_mean = df["value"].rolling(window=window, min_periods=1).mean()
                ax.plot(
                    df["step"],
                    rolling_mean,
                    linewidth=2.5,
                    color="red",
                    label=f"Rolling Mean (window={window})",
                )

                ax.set_title(f"{metric} Over Training", fontsize=14, fontweight="bold")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Reward")
                ax.legend()
                ax.grid(True, alpha=0.3)

                filename = f"{metric.replace('/', '_')}_evolution.png"
                plt.savefig(self.results_dir / filename, dpi=150, bbox_inches="tight")
                print(f"  Saved: {self.results_dir / filename}")
                plt.close()

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 80)

        report = []
        report.append("# NPP-RL Training Analysis Report")
        report.append("=" * 80)
        report.append("")

        # Configuration
        report.append("## 1. Training Configuration")
        report.append("-" * 80)
        report.append(f"**Experiment Name**: {self.config['experiment_name']}")
        report.append(f"**Total Timesteps**: {self.config['total_timesteps']:,}")
        report.append(f"**Number of Environments**: {self.config['num_envs']}")
        report.append(
            f"**Batch Size**: {self.config['hardware_profile_settings']['batch_size']}"
        )
        report.append(
            f"**Learning Rate**: {self.config['hardware_profile_settings']['learning_rate']}"
        )
        report.append(
            f"**N-Steps**: {self.config['hardware_profile_settings']['n_steps']}"
        )
        report.append("")
        report.append("**Feature Configuration**:")
        report.append(f"  - Pretraining: {not self.config['no_pretraining']}")
        report.append(f"  - BC Epochs: {self.config['bc_epochs']}")
        report.append(f"  - Curriculum Learning: {self.config['use_curriculum']}")
        report.append(
            f"  - Curriculum Threshold: {self.config['curriculum_threshold']}"
        )
        report.append(
            f"  - PBRS (Potential-Based Reward Shaping): Enabled (gamma = {self.config['pbrs_gamma']})"
        )
        report.append(
            f"  - Visual Frame Stacking: {self.config['enable_visual_frame_stacking']}"
        )
        report.append(f"  - Stack Size: {self.config['visual_stack_size']}")
        report.append("")

        # Final Results
        report.append("## 2. Final Evaluation Results")
        report.append("-" * 80)
        for result in self.all_results:
            report.append(f"**Architecture**: {result['architecture']}")
            report.append(f"**Training Status**: {result['training']['status']}")
            report.append(
                f"**Success Rate**: {result['evaluation']['success_rate']:.1%}"
            )
            report.append(f"**Average Steps**: {result['evaluation']['avg_steps']:.1f}")
            report.append(
                f"**Total Episodes**: {result['evaluation']['total_episodes']}"
            )
        report.append("")

        # Curriculum Statistics
        report.append("## 3. Curriculum Learning Analysis")
        report.append("-" * 80)
        report.append("")
        report.append("### 3.1 Overall Success Rate by Stage")
        report.append("")
        stage_stats = self.curriculum_df.groupby("stage").agg(
            {"success": ["sum", "count", "mean"]}
        )
        for stage in stage_stats.index:
            success_count = stage_stats.loc[stage, ("success", "sum")]
            total_count = stage_stats.loc[stage, ("success", "count")]
            success_rate = stage_stats.loc[stage, ("success", "mean")]
            report.append(
                f"**{stage}**: {success_rate:.1%} ({success_count}/{total_count} episodes)"
            )
        report.append("")

        report.append("### 3.2 Top 10 Level Generators by Success Rate")
        report.append("")
        gen_stats = (
            self.curriculum_df.groupby("generator")
            .agg({"success": ["sum", "count", "mean"]})
            .sort_values(("success", "mean"), ascending=False)
            .head(10)
        )
        for generator in gen_stats.index:
            success_count = gen_stats.loc[generator, ("success", "sum")]
            total_count = gen_stats.loc[generator, ("success", "count")]
            success_rate = gen_stats.loc[generator, ("success", "mean")]
            report.append(
                f"- **{generator}**: {success_rate:.1%} ({success_count}/{total_count} episodes)"
            )
        report.append("")

        report.append("### 3.3 Bottom 10 Level Generators by Success Rate")
        report.append("")
        gen_stats_bottom = (
            self.curriculum_df.groupby("generator")
            .agg({"success": ["sum", "count", "mean"]})
            .sort_values(("success", "mean"), ascending=True)
            .head(10)
        )
        for generator in gen_stats_bottom.index:
            success_count = gen_stats_bottom.loc[generator, ("success", "sum")]
            total_count = gen_stats_bottom.loc[generator, ("success", "count")]
            success_rate = gen_stats_bottom.loc[generator, ("success", "mean")]
            report.append(
                f"- **{generator}**: {success_rate:.1%} ({success_count}/{total_count} episodes)"
            )
        report.append("")

        # Key Metrics Summary
        report.append("## 4. Training Metrics Summary")
        report.append("-" * 80)
        report.append("")

        # Reward metrics
        reward_metrics = [
            k
            for k in self.tb_data.keys()
            if "reward" in k.lower() or "rew" in k.lower()
        ]
        if reward_metrics:
            report.append("### 4.1 Reward Metrics")
            report.append("")
            for metric in sorted(reward_metrics):
                df = self.tb_data[metric]
                report.append(f"**{metric}**:")
                report.append(f"  - Mean: {df['value'].mean():.4f}")
                report.append(f"  - Std: {df['value'].std():.4f}")
                report.append(f"  - Min: {df['value'].min():.4f}")
                report.append(f"  - Max: {df['value'].max():.4f}")
                report.append("")

        # Loss metrics
        loss_metrics = [k for k in self.tb_data.keys() if "loss" in k.lower()]
        if loss_metrics:
            report.append("### 4.2 Loss Metrics")
            report.append("")
            for metric in sorted(loss_metrics):
                df = self.tb_data[metric]
                report.append(f"**{metric}**:")
                report.append(f"  - Mean: {df['value'].mean():.6f}")
                report.append(f"  - Std: {df['value'].std():.6f}")
                report.append(f"  - Final: {df['value'].iloc[-1]:.6f}")
                report.append("")

        # Save report
        report_text = "\n".join(report)
        report_path = self.results_dir / "comprehensive_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"\nSaved comprehensive report: {report_path}")

        return report_text

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting comprehensive analysis...")
        print("=" * 80)

        self.load_data()
        self.analyze_curriculum_progression()
        self.analyze_tensorboard_metrics()
        self.analyze_action_distribution()
        self.analyze_reward_components()
        report = self.generate_comprehensive_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {self.results_dir}")

        return report


def main():
    results_dir = Path("latest-training-results")

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    analyzer = NPPRLAnalyzer(results_dir)
    analyzer.run_full_analysis()

    return 0


if __name__ == "__main__":
    sys.exit(main())
