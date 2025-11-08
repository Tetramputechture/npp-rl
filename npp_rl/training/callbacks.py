"""Custom callbacks for training diagnostics and logging."""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class DiagnosticLoggingCallback(BaseCallback):
    """
    Ultra-comprehensive reward and training diagnostics callback.

    Logs 100+ metrics to TensorBoard including:
    - Full reward distributions (per-step and per-episode)
    - Component-wise reward breakdowns
    - Terminal event frequencies
    - Sparse vs dense reward ratios
    - Distance metrics and PBRS diagnostics
    - Action-reward correlations
    - Curriculum-specific metrics

    This enables deep diagnosis of training issues and reward structure problems.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """Initialize diagnostic logging callback.

        Args:
            log_freq: Frequency (in steps) to log aggregated metrics
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Per-step metric accumulators
        self.step_metrics = {}

        # Per-episode accumulators
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = {"success": 0, "death": 0, "timeout": 0}

        # Reward distribution tracking
        self.reward_history = []
        self.positive_rewards = []
        self.negative_rewards = []
        self.zero_rewards = []

        # Terminal event tracking
        self.terminal_events = {
            "completion": [],
            "death": [],
            "switch": [],
        }

        # Action-reward correlation
        self.action_reward_pairs = []
        
        # PBRS component tracking (Week 3-4 enhancement)
        self.episode_pbrs_components = {
            'objective': deque(maxlen=100),
            'hazard': deque(maxlen=100),
            'impact': deque(maxlen=100),
            'exploration': deque(maxlen=100),
        }

        # Curriculum tracking
        self.stage_metrics = {}

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Get step data from locals
        rewards = self.locals.get("rewards", [])
        actions = self.locals.get("actions", [])
        infos = self.locals.get("infos", [])

        # Track ALL step rewards + classification by sign
        for reward in rewards:
            self.reward_history.append(reward)
            if reward > 0:
                self.positive_rewards.append(reward)
            elif reward < 0:
                self.negative_rewards.append(reward)
            else:
                self.zero_rewards.append(reward)

        # Action-reward correlation
        for action, reward in zip(actions, rewards):
            action_int = int(action) if hasattr(action, "__iter__") else action
            self.action_reward_pairs.append((action_int, reward))

        # Process each environment
        for idx, info in enumerate(infos):
            # Collect diagnostic metrics from environments
            if "diagnostic_metrics" in info:
                for key, val in info["diagnostic_metrics"].items():
                    if key not in self.step_metrics:
                        self.step_metrics[key] = []
                    self.step_metrics[key].append(val)

            # Episode completion tracking
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

                # Classify episode outcome
                if info.get("player_won", False):
                    self.episode_outcomes["success"] += 1
                elif info.get("player_dead", False):
                    self.episode_outcomes["death"] += 1
                else:
                    self.episode_outcomes["timeout"] += 1

            # PBRS component tracking (Week 3-4 enhancement)
            if "pbrs_components" in info:
                components = info["pbrs_components"]
                for key in self.episode_pbrs_components:
                    if key in components:
                        self.episode_pbrs_components[key].append(components[key])
            
            # Terminal events
            if info.get("player_won", False):
                self.terminal_events["completion"].append(rewards[idx])
            if info.get("player_dead", False):
                self.terminal_events["death"].append(rewards[idx])

            # Curriculum stage tracking
            stage = info.get("curriculum_stage", "unknown")
            if stage not in self.stage_metrics:
                self.stage_metrics[stage] = {
                    "rewards": [],
                    "successes": 0,
                    "episodes": 0,
                }
            self.stage_metrics[stage]["rewards"].append(rewards[idx])

            if "episode" in info:
                self.stage_metrics[stage]["episodes"] += 1
                if info.get("player_won", False):
                    self.stage_metrics[stage]["successes"] += 1

        # Log periodically
        if self.n_calls % self.log_freq == 0:
            self._log_all_metrics()

        return True

    def _log_all_metrics(self):
        """Log comprehensive metrics (100+) to TensorBoard."""

        # === REWARD DISTRIBUTION (15+ metrics) ===
        if self.reward_history:
            r = np.array(self.reward_history)
            self.logger.record("reward_dist/mean", np.mean(r))
            self.logger.record("reward_dist/std", np.std(r))
            self.logger.record("reward_dist/min", np.min(r))
            self.logger.record("reward_dist/max", np.max(r))

            # Full quantile distribution
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                self.logger.record(f"reward_dist/p{q}", np.percentile(r, q))

            # Positive/negative ratios
            total = len(self.reward_history)
            self.logger.record(
                "reward_dist/positive_ratio", len(self.positive_rewards) / total
            )
            self.logger.record(
                "reward_dist/negative_ratio", len(self.negative_rewards) / total
            )
            self.logger.record("reward_dist/zero_ratio", len(self.zero_rewards) / total)

            if self.positive_rewards:
                self.logger.record(
                    "reward_dist/positive_mean", np.mean(self.positive_rewards)
                )
                self.logger.record(
                    "reward_dist/positive_max", np.max(self.positive_rewards)
                )

            if self.negative_rewards:
                self.logger.record(
                    "reward_dist/negative_mean", np.mean(self.negative_rewards)
                )
                self.logger.record(
                    "reward_dist/negative_min", np.min(self.negative_rewards)
                )

            # Outlier detection (beyond 3 std)
            outliers = np.abs(r - np.mean(r)) > 3 * np.std(r)
            self.logger.record("reward_dist/outlier_ratio", np.sum(outliers) / len(r))

        # === EPISODE REWARDS (8+ metrics) ===
        if self.episode_rewards:
            ep = np.array(self.episode_rewards)
            self.logger.record("episode_reward/mean", np.mean(ep))
            self.logger.record("episode_reward/std", np.std(ep))
            self.logger.record("episode_reward/min", np.min(ep))
            self.logger.record("episode_reward/max", np.max(ep))

            for q in [10, 25, 50, 75, 90]:
                self.logger.record(f"episode_reward/p{q}", np.percentile(ep, q))

            # Outcome rates
            total_eps = sum(self.episode_outcomes.values())
            if total_eps > 0:
                self.logger.record(
                    "episode_outcome/success_rate",
                    self.episode_outcomes["success"] / total_eps,
                )
                self.logger.record(
                    "episode_outcome/death_rate",
                    self.episode_outcomes["death"] / total_eps,
                )
                self.logger.record(
                    "episode_outcome/timeout_rate",
                    self.episode_outcomes["timeout"] / total_eps,
                )

        # === TERMINAL EVENTS (6 metrics) ===
        for event_type, values in self.terminal_events.items():
            if values:
                self.logger.record(f"terminal/{event_type}_mean", np.mean(values))
                self.logger.record(f"terminal/{event_type}_count", len(values))

        # === SPARSE VS DENSE (4 metrics) ===
        if self.reward_history:
            sparse = [r for r in self.reward_history if abs(r) > 1.0]
            dense = [r for r in self.reward_history if 0 < abs(r) < 0.1]
            total = len(self.reward_history)

            self.logger.record("reward_type/sparse_ratio", len(sparse) / total)
            self.logger.record("reward_type/dense_ratio", len(dense) / total)

            if sparse:
                self.logger.record(
                    "reward_type/sparse_mean_mag", np.mean(np.abs(sparse))
                )
            if dense:
                self.logger.record("reward_type/dense_mean_mag", np.mean(np.abs(dense)))

        # === PBRS COMPONENT BREAKDOWN (Week 3-4 enhancement) ===
        for component, values in self.episode_pbrs_components.items():
            if len(values) > 0:
                vals = np.array(values)
                self.logger.record(f"pbrs_rewards/{component}_mean", float(np.mean(vals)))
                self.logger.record(f"pbrs_rewards/{component}_std", float(np.std(vals)))

        # === ACTION-REWARD CORRELATION (12 metrics for 6 actions) ===
        if self.action_reward_pairs:
            action_rewards = {}
            for action, reward in self.action_reward_pairs:
                if action not in action_rewards:
                    action_rewards[action] = []
                action_rewards[action].append(reward)

            action_names = ['NOOP', 'Left', 'Right', 'Jump', 'Jump+Left', 'Jump+Right']
            for action_id, rewards_list in action_rewards.items():
                if rewards_list and len(rewards_list) > 10:  # Require minimum samples
                    self.logger.record(
                        f"actions/action_{action_id}_{action_names[action_id]}_mean_reward",
                        float(np.mean(rewards_list))
                    )
                    self.logger.record(
                        f"action_reward/action_{action_id}_mean", np.mean(rewards_list)
                    )
                    self.logger.record(
                        f"action_reward/action_{action_id}_count", len(rewards_list)
                    )

        # === CURRICULUM METRICS (varies by stages) ===
        for stage, metrics in self.stage_metrics.items():
            if metrics["rewards"]:
                stage_r = np.array(metrics["rewards"])
                self.logger.record(f"curriculum/{stage}/reward_mean", np.mean(stage_r))
                self.logger.record(f"curriculum/{stage}/reward_std", np.std(stage_r))

                if metrics["episodes"] > 0:
                    self.logger.record(
                        f"curriculum/{stage}/success_rate",
                        metrics["successes"] / metrics["episodes"],
                    )

        # === STEP METRICS (20-40 metrics from diagnostic system) ===
        for key, values in self.step_metrics.items():
            if values:
                v = np.array(values)
                self.logger.record(f"{key}/mean", np.mean(v))
                self.logger.record(f"{key}/std", np.std(v))
                self.logger.record(f"{key}/min", np.min(v))
                self.logger.record(f"{key}/max", np.max(v))

                # Percentiles for key metrics
                if any(x in key for x in ["distance", "potential", "reward", "pbrs"]):
                    for q in [25, 50, 75, 90]:
                        self.logger.record(f"{key}/p{q}", np.percentile(v, q))

        # Clear buffers (keep recent 10k for rolling stats)
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-10000:]
            self.positive_rewards = self.positive_rewards[-10000:]
            self.negative_rewards = self.negative_rewards[-10000:]
            self.zero_rewards = self.zero_rewards[-10000:]

        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
            self.episode_lengths = self.episode_lengths[-1000:]

        if len(self.action_reward_pairs) > 10000:
            self.action_reward_pairs = self.action_reward_pairs[-10000:]

        # Clear step metrics
        self.step_metrics.clear()
