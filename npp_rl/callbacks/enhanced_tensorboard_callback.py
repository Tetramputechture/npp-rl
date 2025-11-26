"""Enhanced TensorBoard callback for comprehensive training metrics.

This callback provides detailed logging of training metrics, including:
- Episode statistics (rewards, lengths, success rates)
- Agent behavior metrics (action distributions, value estimates)
- Learning progress indicators (loss components)
- Environment interaction statistics
- Performance metrics (FPS)
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

logger = logging.getLogger(__name__)


class EnhancedTensorBoardCallback(BaseCallback):
    """Enhanced TensorBoard callback with comprehensive metrics logging.

    This callback logs detailed metrics to TensorBoard including:
    - Episode statistics (rewards, lengths, success rates)
    - Action frequency distribution and entropy
    - Value function estimates
    - Policy and value loss components
    - Learning rate schedule
    - Clip ratios and explained variance
    - Environment statistics
    - FPS and performance metrics
    """

    def __init__(
        self,
        log_freq: int = 200,
        verbose: int = 0,
    ):
        """Initialize enhanced TensorBoard callback.

        Args:
            log_freq: Frequency (in steps) to log scalar metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # TensorBoard writer
        self.tb_writer = None

        # Tracking buffers
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)

        # Reward component tracking (for intrinsic/hierarchical rewards)
        self.episode_extrinsic_rewards = deque(maxlen=100)

        # PBRS reward component tracking (step-level)
        self.pbrs_navigation_rewards = deque(maxlen=1000)
        self.pbrs_exploration_rewards = deque(maxlen=1000)
        self.pbrs_shaping_rewards = deque(maxlen=1000)
        self.pbrs_total_rewards = deque(maxlen=1000)

        # PBRS potential tracking
        self.pbrs_objective_potentials = deque(maxlen=1000)
        self.pbrs_hazard_potentials = deque(maxlen=1000)
        self.pbrs_impact_potentials = deque(maxlen=1000)
        self.pbrs_exploration_potentials = deque(maxlen=1000)

        # PBRS diagnostic metrics (for debugging normalization issues)
        self.pbrs_distance_to_goal = deque(maxlen=1000)
        self.pbrs_area_scale = deque(maxlen=1000)
        self.pbrs_normalized_distance = deque(maxlen=1000)
        self.pbrs_combined_path_distance = deque(maxlen=1000)

        # Action tracking
        self.action_counts = defaultdict(int)
        self.total_actions = 0

        # Action persistence tracking (for frame skip analysis)
        self.action_persistence_buffer = deque(maxlen=10000)  # Store hold durations
        self.action_change_count = 0
        self.prev_actions = None  # Track previous actions for each env
        self.current_action_hold_durations = None  # Current consecutive count per env

        # Action sequence tracking for temporal entropy
        self.action_sequences = deque(maxlen=1000)  # Store recent action sequences
        self.current_action_sequence = []  # Current episode's action sequence

        # Decision efficiency tracking
        self.decisions_per_episode = deque(maxlen=100)  # Track decisions per episode
        self.rewards_per_decision = deque(maxlen=100)  # Track reward efficiency

        # Action name mapping for N++ (indices must match environment action space)
        self.action_names = {
            0: "NOOP",
            1: "Left",
            2: "Right",
            3: "Jump",
            4: "Jump+Left",
            5: "Jump+Right",
        }

        # Value function tracking
        self.value_estimates = deque(maxlen=1000)

        # Performance tracking
        self.fps_history = deque(maxlen=100)

        # Training progress
        self.last_log_step = 0
        self.start_time = None

        # Curriculum tracking
        self.current_curriculum_stage = None
        self.curriculum_stage_episodes = defaultdict(int)  # Episodes per stage
        # MEMORY OPTIMIZATION: Use deque with maxlen instead of unbounded list
        # to prevent memory leak in long-running training
        self.curriculum_stage_successes = defaultdict(
            lambda: deque(maxlen=100)
        )  # Success history per stage (last 100 episodes)

        # Smoothed success rate (EMA with alpha=0.1)
        self.success_rate_ema = None
        self.ema_alpha = 0.1

    def _init_callback(self) -> None:
        """Initialize callback after model setup."""
        # Find TensorBoard writer
        for output_format in self.logger.output_formats:
            if isinstance(output_format, TensorBoardOutputFormat):
                self.tb_writer = output_format.writer
                break

        if self.tb_writer is None:
            print("TensorBoard writer not found - enhanced logging disabled")
            return

        self.start_time = time.time()
        logger.info("Enhanced TensorBoard callback initialized")

    def _on_step(self) -> bool:
        """Called after each environment step.

        Returns:
            bool: If False, training will be stopped
        """
        if self.tb_writer is None:
            return True

        # Track episode completions and step-level PBRS components
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if "infos" in self.locals:
                infos = self.locals["infos"]
                for i, (done, info) in enumerate(zip(dones, infos)):
                    # Track PBRS components at every step
                    self._track_pbrs_components(info)

                    # Process episode end metrics
                    if done:
                        self._process_episode_end(info)

        # Track actions taken
        if "actions" in self.locals:
            actions = self.locals["actions"]

            # Initialize persistence tracking on first step
            if self.prev_actions is None:
                self.prev_actions = actions.copy()
                self.current_action_hold_durations = np.ones(len(actions), dtype=int)

            # Track action counts and persistence
            for i, action in enumerate(actions):
                action_idx = (
                    int(action)
                    if isinstance(action, (int, np.integer))
                    else int(action.item())
                )
                self.action_counts[action_idx] += 1
                self.total_actions += 1

                # Track action sequences (only for first environment to avoid redundancy)
                if i == 0:
                    self.current_action_sequence.append(action_idx)

                # Track action persistence (consecutive same actions)
                prev_action_idx = (
                    int(self.prev_actions[i])
                    if isinstance(self.prev_actions[i], (int, np.integer))
                    else int(self.prev_actions[i].item())
                )

                if action_idx == prev_action_idx:
                    # Same action as previous step - increment hold duration
                    self.current_action_hold_durations[i] += 1
                else:
                    # Action changed - record previous hold duration
                    self.action_persistence_buffer.append(
                        self.current_action_hold_durations[i]
                    )
                    self.action_change_count += 1
                    self.current_action_hold_durations[i] = 1

            # Update previous actions for next step
            self.prev_actions = actions.copy()

        # Track value estimates from rollout buffer (more reliable than trying to access obs_tensor)
        # Note: This captures values after they've been computed during rollout collection
        if hasattr(self.model, "rollout_buffer"):
            try:
                # Only access if buffer has data
                if hasattr(self.model.rollout_buffer, "values") and hasattr(
                    self.model.rollout_buffer, "pos"
                ):
                    buffer_pos = self.model.rollout_buffer.pos
                    if buffer_pos > 0:  # Buffer has new data
                        # Get recently added values (last position)
                        recent_idx = max(0, buffer_pos - 1)
                        values = self.model.rollout_buffer.values[recent_idx]
                        if values is not None and len(values) > 0:
                            self.value_estimates.extend(values.flatten().tolist())
            except Exception as e:
                logger.debug(
                    f"Could not track value estimates from rollout buffer: {e}"
                )

        # Log scalar metrics at regular intervals
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_scalar_metrics()
            self.last_log_step = self.num_timesteps

        # MEMORY OPTIMIZATION: Periodically reset action counts to prevent
        # unbounded growth in long-running training (every 50K steps)
        if self.num_timesteps % 50000 == 0 and self.num_timesteps > 0:
            # Reset action tracking but preserve ratios (they've already been logged)
            self.action_counts.clear()
            self.total_actions = 0
            self.action_change_count = 0

        return True

    def _track_pbrs_components(self, info: Dict[str, Any]) -> None:
        """Track PBRS reward components from step info.

        Args:
            info: Step info dictionary that may contain PBRS components
        """
        if "pbrs_components" not in info or not info["pbrs_components"]:
            return

        pbrs_data = info["pbrs_components"]

        # Track main reward components (new structure from RewardCalculator)
        if "pbrs_reward" in pbrs_data:
            self.pbrs_shaping_rewards.append(float(pbrs_data["pbrs_reward"]))
        if "total_reward" in pbrs_data:
            self.pbrs_total_rewards.append(float(pbrs_data["total_reward"]))

        # Track time penalty separately
        if "time_penalty" in pbrs_data:
            # Time penalty is already tracked in episode_time_penalties during _on_step
            pass

        # Track milestone rewards (switch activation)
        if "milestone_reward" in pbrs_data and pbrs_data["milestone_reward"] != 0:
            # Could add milestone tracking if needed
            pass

        # Track potential information for debugging
        if "current_potential" in pbrs_data:
            self.pbrs_objective_potentials.append(float(pbrs_data["current_potential"]))

        # Track diagnostic metrics for PBRS normalization debugging
        if "distance_to_goal" in pbrs_data:
            self.pbrs_distance_to_goal.append(float(pbrs_data["distance_to_goal"]))
        if "area_scale" in pbrs_data:
            self.pbrs_area_scale.append(float(pbrs_data["area_scale"]))
        if "normalized_distance" in pbrs_data:
            self.pbrs_normalized_distance.append(
                float(pbrs_data["normalized_distance"])
            )
        if "combined_path_distance" in pbrs_data:
            self.pbrs_combined_path_distance.append(
                float(pbrs_data["combined_path_distance"])
            )

        # Legacy support for old pbrs_components structure (if any)
        if "navigation_reward" in pbrs_data:
            self.pbrs_navigation_rewards.append(float(pbrs_data["navigation_reward"]))
        if "exploration_reward" in pbrs_data:
            self.pbrs_exploration_rewards.append(float(pbrs_data["exploration_reward"]))

    def _process_episode_end(self, info: Dict[str, Any]) -> None:
        """Process episode completion and extract metrics.

        Args:
            info: Episode info dictionary
        """
        # Standard episode metrics
        if "episode" in info:
            episode_info = info["episode"]
            episode_reward = episode_info["r"]
            episode_length = episode_info["l"]
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Track decision efficiency (for frame skip analysis)
            if episode_length > 0:
                self.decisions_per_episode.append(episode_length)
                reward_per_decision = episode_reward / episode_length
                self.rewards_per_decision.append(reward_per_decision)

        # Store action sequence for temporal entropy calculation
        if len(self.current_action_sequence) > 0:
            self.action_sequences.append(list(self.current_action_sequence))
            self.current_action_sequence = []  # Reset for next episode

        if "r_ext_episode" in info:
            self.episode_extrinsic_rewards.append(info["r_ext_episode"])

        # Success/failure tracking - use authoritative has_won/player_won from environment
        success = None
        if "has_won" in info:
            # Prioritize has_won - authoritative success indicator
            success = float(info["has_won"])
            self.episode_successes.append(success)
        elif "player_won" in info:
            # Also check player_won (used in base environment)
            success = float(info["player_won"])
            self.episode_successes.append(success)
            # Add has_won for consistency with other components
            info["has_won"] = bool(info["player_won"])
        elif "success" in info:
            success = float(info["success"])
            self.episode_successes.append(success)
        elif "is_success" in info:
            success = float(info["is_success"])
            self.episode_successes.append(success)
        elif "episode" in info and "r" in info["episode"]:
            # Fallback: infer success from reward (N++ gives 1.0 for completion)
            success = float(info["episode"]["r"] > 0.9)
            self.episode_successes.append(success)

        # Update smoothed success rate (EMA)
        if success is not None:
            if self.success_rate_ema is None:
                self.success_rate_ema = success
            else:
                self.success_rate_ema = (
                    self.ema_alpha * success
                    + (1 - self.ema_alpha) * self.success_rate_ema
                )

        # Track curriculum stage information
        if "curriculum_stage" in info:
            stage = info["curriculum_stage"]
            if stage and stage != "unknown":
                self.current_curriculum_stage = stage
                self.curriculum_stage_episodes[stage] += 1
                if success is not None:
                    self.curriculum_stage_successes[stage].append(success)

        # Completion time tracking removed - redundant with episode length

    def _log_scalar_metrics(self) -> None:
        """Log scalar metrics to TensorBoard."""
        step = self.num_timesteps

        # Episode statistics
        if self.episode_rewards:
            self.tb_writer.add_scalar(
                "episode/reward_mean", np.mean(self.episode_rewards), step
            )
            self.tb_writer.add_scalar(
                "episode/reward_std", np.std(self.episode_rewards), step
            )
            # Removed reward_max and reward_min - clutters dashboard, mean/std sufficient

        if self.episode_lengths:
            self.tb_writer.add_scalar(
                "episode/length_mean", np.mean(self.episode_lengths), step
            )
            self.tb_writer.add_scalar(
                "episode/length_std", np.std(self.episode_lengths), step
            )

        if self.episode_successes:
            success_rate = np.mean(self.episode_successes)
            self.tb_writer.add_scalar("episode/success_rate", success_rate, step)
            # Removed failure_rate - redundant (just 1 - success_rate)

            # Add smoothed success rate for cleaner trend visualization
            if self.success_rate_ema is not None:
                self.tb_writer.add_scalar(
                    "episode/success_rate_smoothed", self.success_rate_ema, step
                )

        # Removed completion_time - redundant with episode length

        # Curriculum learning metrics (critical for tracking progression)
        if self.current_curriculum_stage is not None:
            # Map stage name to index for timeline visualization
            stage_order = [
                "simplest",
                "simplest_few_mines",
                "simplest_with_mines",
                "simpler",
                "simple",
                "medium",
                "complex",
                "exploration",
                "mine_heavy",
            ]
            if self.current_curriculum_stage in stage_order:
                stage_idx = stage_order.index(self.current_curriculum_stage)
                self.tb_writer.add_scalar("curriculum/stage_timeline", stage_idx, step)

            # Episodes in current stage
            episodes_in_stage = self.curriculum_stage_episodes[
                self.current_curriculum_stage
            ]
            self.tb_writer.add_scalar(
                "curriculum/episodes_in_current_stage", episodes_in_stage, step
            )

            # Per-stage success rates for comparison
            for stage, successes in self.curriculum_stage_successes.items():
                if len(successes) >= 5:  # Only log if we have enough data
                    stage_success_rate = np.mean(
                        successes
                    )  # Mean of recent episodes (bounded by deque)
                    self.tb_writer.add_scalar(
                        f"curriculum_stages/{stage}_success_rate",
                        stage_success_rate,
                        step,
                    )

        # PBRS reward component statistics (cleaned up - removed std/min/max)
        if self.pbrs_navigation_rewards:
            self.tb_writer.add_scalar(
                "pbrs_rewards/navigation_mean",
                np.mean(self.pbrs_navigation_rewards),
                step,
            )
            # Removed navigation_std - mean is sufficient

        if self.pbrs_exploration_rewards:
            self.tb_writer.add_scalar(
                "pbrs_rewards/exploration_mean",
                np.mean(self.pbrs_exploration_rewards),
                step,
            )
            # Removed exploration_std - mean is sufficient

        if self.pbrs_shaping_rewards:
            self.tb_writer.add_scalar(
                "pbrs_rewards/pbrs_mean", np.mean(self.pbrs_shaping_rewards), step
            )
            # Removed pbrs_std, pbrs_min, pbrs_max - noisy and not actionable

        if self.pbrs_total_rewards:
            self.tb_writer.add_scalar(
                "pbrs_rewards/total_mean", np.mean(self.pbrs_total_rewards), step
            )
            # Removed total_std - mean is sufficient

        # PBRS potential statistics (cleaned up - removed std)
        if self.pbrs_objective_potentials:
            self.tb_writer.add_scalar(
                "pbrs_potentials/objective_mean",
                np.mean(self.pbrs_objective_potentials),
                step,
            )
            # Removed objective_std - mean is sufficient

        if self.pbrs_hazard_potentials:
            self.tb_writer.add_scalar(
                "pbrs_potentials/hazard_mean",
                np.mean(self.pbrs_hazard_potentials),
                step,
            )

        if self.pbrs_impact_potentials:
            self.tb_writer.add_scalar(
                "pbrs_potentials/impact_mean",
                np.mean(self.pbrs_impact_potentials),
                step,
            )

        if self.pbrs_exploration_potentials:
            self.tb_writer.add_scalar(
                "pbrs_potentials/exploration_mean",
                np.mean(self.pbrs_exploration_potentials),
                step,
            )

        # PBRS diagnostic metrics (for debugging normalization)
        if self.pbrs_distance_to_goal:
            self.tb_writer.add_scalar(
                "pbrs/distance_to_goal",
                np.mean(self.pbrs_distance_to_goal),
                step,
            )
        if self.pbrs_area_scale:
            self.tb_writer.add_scalar(
                "pbrs/area_scale",
                np.mean(self.pbrs_area_scale),
                step,
            )
        if self.pbrs_normalized_distance:
            self.tb_writer.add_scalar(
                "pbrs/normalized_distance",
                np.mean(self.pbrs_normalized_distance),
                step,
            )
        if self.pbrs_combined_path_distance:
            self.tb_writer.add_scalar(
                "pbrs/combined_path_distance",
                np.mean(self.pbrs_combined_path_distance),
                step,
            )

        # PBRS contribution analysis
        if self.pbrs_shaping_rewards and self.pbrs_total_rewards:
            pbrs_vals = np.array(self.pbrs_shaping_rewards)
            total_vals = np.array(self.pbrs_total_rewards)
            # Calculate PBRS contribution ratio
            if len(pbrs_vals) == len(total_vals):
                # Safe division avoiding division by zero
                valid_indices = np.abs(total_vals) > 1e-6
                if valid_indices.any():
                    contribution = np.abs(pbrs_vals[valid_indices]) / np.abs(
                        total_vals[valid_indices]
                    )
                    self.tb_writer.add_scalar(
                        "pbrs_summary/pbrs_contribution_ratio",
                        np.mean(contribution),
                        step,
                    )

        # Action distribution with descriptive names
        if self.total_actions > 0:
            # Get action space size dynamically
            n_actions = (
                self.model.action_space.n
                if hasattr(self.model.action_space, "n")
                else 6
            )

            # Log individual action frequencies with descriptive names
            for action_idx in range(n_actions):
                count = self.action_counts.get(action_idx, 0)
                action_freq = count / self.total_actions
                action_name = self.action_names.get(action_idx, f"Action{action_idx}")
                self.tb_writer.add_scalar(
                    f"actions/frequency/{action_name}", action_freq, step
                )

            # Calculate action probabilities for entropy and analysis
            action_probs = np.array(
                [self.action_counts.get(i, 0) for i in range(n_actions)]
            ) / max(self.total_actions, 1)
            action_probs = action_probs + 1e-10  # Avoid log(0)
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            self.tb_writer.add_scalar("actions/entropy", action_entropy, step)

            # Log action persistence metrics (for frame skip analysis)
            if len(self.action_persistence_buffer) > 0:
                # Average hold duration (consecutive frames same action)
                avg_hold_duration = np.mean(self.action_persistence_buffer)
                self.tb_writer.add_scalar(
                    "actions/persistence/avg_hold_duration", avg_hold_duration, step
                )

                # Median hold duration
                median_hold_duration = np.median(self.action_persistence_buffer)
                self.tb_writer.add_scalar(
                    "actions/persistence/median_hold_duration",
                    median_hold_duration,
                    step,
                )

                # Max hold duration (shows longest sustained action)
                max_hold_duration = np.max(self.action_persistence_buffer)
                self.tb_writer.add_scalar(
                    "actions/persistence/max_hold_duration", max_hold_duration, step
                )

                # Action change frequency (changes per total actions)
                if self.total_actions > 0:
                    change_frequency = self.action_change_count / self.total_actions
                    self.tb_writer.add_scalar(
                        "actions/persistence/change_frequency", change_frequency, step
                    )

                    # Inverse metric: average actions per change
                    actions_per_change = 1.0 / max(change_frequency, 1e-6)
                    self.tb_writer.add_scalar(
                        "actions/persistence/actions_per_change",
                        actions_per_change,
                        step,
                    )

                # Distribution percentiles (for understanding spread)
                p25 = np.percentile(self.action_persistence_buffer, 25)
                p75 = np.percentile(self.action_persistence_buffer, 75)
                self.tb_writer.add_scalar(
                    "actions/persistence/hold_duration_p25", p25, step
                )
                self.tb_writer.add_scalar(
                    "actions/persistence/hold_duration_p75", p75, step
                )

            # Movement-specific metrics
            # Horizontal movement: Left (1) + Right (2) + Jump+Left (4) + Jump+Right (5)
            left_actions = self.action_counts.get(1, 0) + self.action_counts.get(4, 0)
            right_actions = self.action_counts.get(2, 0) + self.action_counts.get(5, 0)
            horizontal_movement = left_actions + right_actions

            if horizontal_movement > 0:
                left_bias = left_actions / horizontal_movement
                right_bias = right_actions / horizontal_movement
                self.tb_writer.add_scalar("actions/movement/left_bias", left_bias, step)
                self.tb_writer.add_scalar(
                    "actions/movement/right_bias", right_bias, step
                )

            # Calculate movement vs stationary time
            noop_freq = action_probs[0]
            movement_freq = 1.0 - noop_freq
            self.tb_writer.add_scalar(
                "actions/movement/stationary_pct", noop_freq, step
            )
            self.tb_writer.add_scalar(
                "actions/movement/active_pct", movement_freq, step
            )

            # Jump analysis
            # Jump actions: Jump (3) + Jump+Left (4) + Jump+Right (5)
            jump_only = self.action_counts.get(3, 0)
            jump_left = self.action_counts.get(4, 0)
            jump_right = self.action_counts.get(5, 0)
            total_jumps = jump_only + jump_left + jump_right

            if total_jumps > 0:
                # How often jumps are combined with directional movement
                directional_jump_pct = (jump_left + jump_right) / total_jumps
                self.tb_writer.add_scalar(
                    "actions/jump/directional_pct", directional_jump_pct, step
                )
                self.tb_writer.add_scalar(
                    "actions/jump/vertical_only_pct", jump_only / total_jumps, step
                )

            # Overall jump frequency (useful for understanding agent behavior)
            jump_freq = total_jumps / self.total_actions
            self.tb_writer.add_scalar("actions/jump/frequency", jump_freq, step)

        # Temporal action entropy (entropy over action sequences, not just distribution)
        if len(self.action_sequences) > 0:
            # Compute bigram entropy: H(A_t | A_{t-1})
            bigram_counts = defaultdict(int)
            total_bigrams = 0

            for sequence in self.action_sequences:
                for i in range(len(sequence) - 1):
                    bigram = (sequence[i], sequence[i + 1])
                    bigram_counts[bigram] += 1
                    total_bigrams += 1

            if total_bigrams > 0:
                # Calculate bigram entropy
                bigram_probs = np.array(
                    [count / total_bigrams for count in bigram_counts.values()]
                )
                bigram_probs = bigram_probs + 1e-10  # Avoid log(0)
                temporal_entropy = -np.sum(bigram_probs * np.log(bigram_probs))
                self.tb_writer.add_scalar(
                    "actions/temporal_entropy", temporal_entropy, step
                )

                # Also calculate entropy rate (difference between joint and marginal)
                # This measures how predictable actions are given previous action
                if self.total_actions > 0:
                    marginal_entropy = (
                        -np.sum(action_probs * np.log(action_probs))
                        if "action_probs" in locals()
                        else 0
                    )
                    entropy_rate = temporal_entropy - marginal_entropy
                    self.tb_writer.add_scalar(
                        "actions/entropy_rate", entropy_rate, step
                    )

        # Decision efficiency metrics (for frame skip analysis)
        if self.decisions_per_episode:
            avg_decisions = np.mean(self.decisions_per_episode)
            self.tb_writer.add_scalar(
                "efficiency/decisions_per_episode_mean", avg_decisions, step
            )
            self.tb_writer.add_scalar(
                "efficiency/decisions_per_episode_std",
                np.std(self.decisions_per_episode),
                step,
            )

        if self.rewards_per_decision:
            avg_reward_per_decision = np.mean(self.rewards_per_decision)
            self.tb_writer.add_scalar(
                "efficiency/reward_per_decision_mean", avg_reward_per_decision, step
            )
            self.tb_writer.add_scalar(
                "efficiency/reward_per_decision_std",
                np.std(self.rewards_per_decision),
                step,
            )

            # Effective decision frequency (useful for comparing frame skip values)
            if self.episode_lengths:
                avg_episode_length = np.mean(self.episode_lengths)
                # At 60 FPS, this shows effective decisions per second
                effective_decision_freq = 60.0 / max(
                    avg_episode_length / max(avg_decisions, 1), 1.0
                )
                self.tb_writer.add_scalar(
                    "efficiency/effective_decision_frequency_hz",
                    effective_decision_freq,
                    step,
                )

        # Value function statistics (cleaned up - removed min/max)
        if self.value_estimates:
            self.tb_writer.add_scalar(
                "value/estimate_mean", np.mean(self.value_estimates), step
            )
            self.tb_writer.add_scalar(
                "value/estimate_std", np.std(self.value_estimates), step
            )
            # Removed estimate_max and estimate_min - mean/std sufficient

        # Learning progress from model logger
        if hasattr(self.model, "logger") and hasattr(
            self.model.logger, "name_to_value"
        ):
            log_data = self.model.logger.name_to_value

            # Policy loss
            if "train/policy_loss" in log_data:
                self.tb_writer.add_scalar(
                    "loss/policy", log_data["train/policy_loss"], step
                )

            # Value loss
            if "train/value_loss" in log_data:
                self.tb_writer.add_scalar(
                    "loss/value", log_data["train/value_loss"], step
                )

            # Entropy loss
            if "train/entropy_loss" in log_data:
                self.tb_writer.add_scalar(
                    "loss/entropy", log_data["train/entropy_loss"], step
                )

            # Total loss
            if "train/loss" in log_data:
                self.tb_writer.add_scalar("loss/total", log_data["train/loss"], step)

            # Clip fraction (important for PPO stability)
            if "train/clip_fraction" in log_data:
                self.tb_writer.add_scalar(
                    "training/clip_fraction", log_data["train/clip_fraction"], step
                )

            # Explained variance (value function quality)
            if "train/explained_variance" in log_data:
                self.tb_writer.add_scalar(
                    "training/explained_variance",
                    log_data["train/explained_variance"],
                    step,
                )

            # Learning rate
            if "train/learning_rate" in log_data:
                self.tb_writer.add_scalar(
                    "training/learning_rate", log_data["train/learning_rate"], step
                )

            # Approximate KL divergence
            if "train/approx_kl" in log_data:
                self.tb_writer.add_scalar(
                    "training/approx_kl", log_data["train/approx_kl"], step
                )

        # Performance metrics
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.tb_writer.add_scalar(
                "performance/elapsed_time_minutes", elapsed_time / 60, step
            )

            # Steps per second
            sps = step / elapsed_time if elapsed_time > 0 else 0
            self.tb_writer.add_scalar("performance/steps_per_second", sps, step)

        # FPS from rollout
        if hasattr(self.model, "_last_obs") and hasattr(self.model, "num_timesteps"):
            try:
                if hasattr(self, "_last_time"):
                    current_time = time.time()
                    time_delta = current_time - self._last_time
                    if time_delta > 0:
                        steps_delta = self.num_timesteps - getattr(
                            self, "_last_timesteps", 0
                        )
                        fps = steps_delta / time_delta
                        self.fps_history.append(fps)
                        self.tb_writer.add_scalar("performance/fps_instant", fps, step)
                        if len(self.fps_history) > 1:
                            self.tb_writer.add_scalar(
                                "performance/fps_mean", np.mean(self.fps_history), step
                            )

                self._last_time = time.time()
                self._last_timesteps = self.num_timesteps
            except Exception as e:
                logger.debug(f"Could not compute FPS: {e}")

        self.tb_writer.flush()
