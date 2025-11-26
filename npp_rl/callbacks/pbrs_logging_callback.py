"""Callback for logging PBRS reward distribution to TensorBoard.

This callback tracks PBRS components during training to help diagnose
reward issues and verify that PBRS is providing effective guidance.
"""

from stable_baselines3.common.callbacks import BaseCallback


class PBRSLoggingCallback(BaseCallback):
    """Callback to log PBRS reward distribution metrics to TensorBoard.

    Tracks and logs:
    - Episode-level PBRS totals and ratios
    - Forward progress vs backtracking statistics
    - Curriculum configuration state
    - Level complexity metrics

    This helps verify that PBRS is contributing meaningfully to learning
    and that reward components are properly balanced.
    """

    def __init__(self, verbose: int = 0):
        """Initialize the PBRS logging callback.

        Args:
            verbose: Verbosity level (0 = no output, 1 = info, 2 = debug)
        """
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Called after every step in the environment.

        Returns:
            bool: True to continue training, False to stop
        """
        # Check if any episode ended in this step
        if "dones" not in self.locals or "infos" not in self.locals:
            return True
            
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        
        for idx, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self._log_episode_metrics(info)

        return True

    def _log_episode_metrics(self, info: dict) -> None:
        """Log episode-level metrics for a completed episode.

        Args:
            info: Episode info dictionary containing reward metrics
        """
        # For SubprocVecEnv compatibility, we can't directly access environments
        # Instead, we rely on the environment to populate episode-level metrics
        # in the info dict through the reward calculator's get_episode_reward_metrics()
        
        # Check if episode-level PBRS metrics are available in info
        # These should be added by the environment's _build_episode_info() method
        if "episode_pbrs_metrics" not in info:
            # Fallback: try to extract from reward calculator if environment is accessible
            # This works for DummyVecEnv but not SubprocVecEnv
            return
        
        metrics = info["episode_pbrs_metrics"]

        # Log to TensorBoard
        self.episode_count += 1

        # Episode-level aggregated metrics
        self.logger.record("reward/pbrs_total", metrics.get("pbrs_total", 0.0))
        self.logger.record("reward/time_penalty_total", metrics.get("time_penalty_total", 0.0))
        self.logger.record("reward/terminal_reward", metrics.get("terminal_reward", 0.0))
        self.logger.record("reward/pbrs_to_penalty_ratio", metrics.get("pbrs_to_penalty_ratio", 0.0))
        self.logger.record("reward/forward_steps", metrics.get("forward_steps", 0))
        self.logger.record("reward/backtrack_steps", metrics.get("backtrack_steps", 0))
        self.logger.record("reward/pbrs_mean", metrics.get("pbrs_mean", 0.0))
        self.logger.record("reward/pbrs_std", metrics.get("pbrs_std", 0.0))

        # New path efficiency metrics
        self.logger.record("efficiency/path_optimality", metrics.get("path_optimality", 0.0))
        self.logger.record("efficiency/forward_progress_pct", metrics.get("forward_progress_pct", 0.0))
        self.logger.record("efficiency/backtracking_pct", metrics.get("backtracking_pct", 0.0))
        self.logger.record("efficiency/episode_path_length", metrics.get("episode_path_length", 0.0))
        self.logger.record("efficiency/optimal_path_length", metrics.get("optimal_path_length", 0.0))

        # Current potential values for debugging
        if metrics.get("current_potential") is not None:
            self.logger.record("reward_step/potential_current", metrics["current_potential"])
        if metrics.get("prev_potential") is not None:
            self.logger.record("reward_step/potential_prev", metrics["prev_potential"])

        # Step-level diagnostic metrics (from pbrs_components in info)
        # These are populated by last_pbrs_components in the reward calculator
        if "pbrs_components" in info:
            step_data = info["pbrs_components"]
            if step_data:
                self.logger.record("pbrs/distance_to_goal", step_data.get("distance_to_goal", 0.0))
                self.logger.record("pbrs/area_scale", step_data.get("area_scale", 0.0))
                self.logger.record("pbrs/combined_path_distance", step_data.get("combined_path_distance", 0.0))
                self.logger.record("pbrs/normalized_distance", step_data.get("normalized_distance", 0.0))

        # Curriculum config state (try to extract from info if available)
        # Note: This may not be available in SubprocVecEnv, so we make it optional
        if "reward_config_state" in info:
            config_state = info["reward_config_state"]
            phase_map = {"early": 0, "mid": 1, "late": 2}
            self.logger.record(
                "curriculum/phase_numeric", phase_map.get(config_state.get("phase"), 0)
            )
            self.logger.record("curriculum/pbrs_weight", config_state.get("pbrs_objective_weight", 0.0))
            self.logger.record(
                "curriculum/normalization_scale", config_state.get("pbrs_normalization_scale", 1.0)
            )
            self.logger.record("curriculum/time_penalty", config_state.get("time_penalty_per_step", 0.0))

        if self.verbose >= 1:
            print(
                f"Episode {self.episode_count}: "
                f"PBRS={metrics.get('pbrs_total', 0.0):.3f}, "
                f"Penalty={metrics.get('time_penalty_total', 0.0):.3f}, "
                f"Terminal={metrics.get('terminal_reward', 0.0):.3f}, "
                f"Ratio={metrics.get('pbrs_to_penalty_ratio', 0.0):.3f}"
            )


