"""Comprehensive model evaluator for N++ test suite.

Evaluates trained models across standardized test levels with detailed metrics.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.gym_environment.config import EnvironmentConfig
from npp_rl.evaluation.test_suite_loader import TestSuiteLoader
try:
    from npp_rl.utils.video_recorder import create_video_recorder
except ImportError:
    create_video_recorder = None
from npp_rl.training.distributed_utils import unwrap_policy_for_inference

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Evaluates trained models on N++ test suite."""

    def __init__(
        self,
        test_dataset_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize evaluator.

        Args:
            test_dataset_path: Path to test dataset
            device: Device for model inference
        """
        self.test_dataset_path = Path(test_dataset_path)
        self.device = device

        # Load test suite
        self.loader = TestSuiteLoader(str(self.test_dataset_path))
        self.test_levels = self.loader.load_all_levels()

        logger.info(f"Initialized evaluator on device: {device}")
        logger.info(f"Test suite: {self.loader.get_summary()}")

    def evaluate_model(
        self,
        model,
        num_episodes_per_category: Optional[Dict[str, int]] = None,
        max_steps_per_episode: int = 10000,
        deterministic: bool = True,
        timeout_per_episode: float = 30.0,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on model.

        Args:
            model: Trained model (SB3-compatible)
            num_episodes_per_category: Episodes per category (None = all)
            max_steps_per_episode: Maximum steps per episode
            deterministic: Use deterministic policy
            timeout_per_episode: Timeout in seconds per episode (default: 30.0)
            record_videos: Whether to record videos of episodes
            video_output_dir: Directory to save videos (required if record_videos=True)
            max_videos_per_category: Maximum number of videos to record per category
            video_fps: Video framerate

        Returns:
            Comprehensive results dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive evaluation")
        logger.info("=" * 60)

        # Validate video recording parameters
        if record_videos and video_output_dir is None:
            raise ValueError(
                "video_output_dir must be provided when record_videos=True"
            )

        if record_videos:
            video_output_path = Path(video_output_dir)
            video_output_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Video recording enabled: {video_output_path} (max {max_videos_per_category} per category)"
            )

        results = {
            "overall": {},
            "by_category": {},
            "by_difficulty_tier": defaultdict(list),
            "failure_modes": [],
        }

        all_success_rates = []
        all_steps = []
        all_efficiencies = []

        # Evaluate each category
        for category, levels in self.test_levels.items():
            if not levels:
                logger.warning(f"No levels in category '{category}', skipping")
                continue

            # Determine how many episodes to run
            if num_episodes_per_category:
                n_episodes = num_episodes_per_category.get(category, len(levels))
            else:
                n_episodes = len(levels)

            n_episodes = min(n_episodes, len(levels))

            logger.info(f"Evaluating category '{category}': {n_episodes} levels")

            category_results = self._evaluate_category(
                model=model,
                category=category,
                levels=levels[:n_episodes],
                max_steps=max_steps_per_episode,
                deterministic=deterministic,
                timeout=timeout_per_episode,
                record_videos=record_videos,
                video_output_dir=video_output_dir,
                max_videos_per_category=max_videos_per_category,
                video_fps=video_fps,
            )

            results["by_category"][category] = category_results

            # Aggregate for overall stats
            all_success_rates.append(category_results["success_rate"])
            all_steps.extend(category_results["episode_steps"])
            all_efficiencies.append(category_results["efficiency"])

        # Calculate overall metrics
        results["overall"] = {
            "success_rate": np.mean(all_success_rates),
            "avg_steps": np.mean(all_steps) if all_steps else 0,
            "std_steps": np.std(all_steps) if all_steps else 0,
            "efficiency": np.mean(all_efficiencies),
            "total_episodes": len(all_steps),
        }

        logger.info("=" * 60)
        logger.info("Evaluation complete")
        logger.info(f"Overall success rate: {results['overall']['success_rate']:.2%}")
        logger.info(f"Average steps: {results['overall']['avg_steps']:.1f}")
        logger.info(f"Efficiency: {results['overall']['efficiency']:.3f}")
        logger.info("=" * 60)

        return results

    def _evaluate_category(
        self,
        model,
        category: str,
        levels: List[Dict],
        max_steps: int,
        deterministic: bool,
        timeout: float = 30.0,
        record_videos: bool = False,
        video_output_dir: Optional[str] = None,
        max_videos_per_category: int = 10,
        video_fps: int = 30,
    ) -> Dict[str, Any]:
        """Evaluate model on a specific category.

        Args:
            model: Model to evaluate
            category: Category name
            levels: List of level data
            max_steps: Max steps per episode
            deterministic: Use deterministic policy
            timeout: Timeout in seconds per episode
            record_videos: Whether to record videos
            video_output_dir: Directory to save videos
            max_videos_per_category: Max videos to record
            video_fps: Video framerate

        Returns:
            Category results dictionary
        """
        successes = []
        episode_steps = []
        rewards = []
        videos_recorded = 0

        for level_idx, level_data in enumerate(
            tqdm(levels, desc=f"Eval {category}", leave=False)
        ):
            level_id = level_data.get("level_id", f"{category}_{level_idx:03d}")
            logger.debug(f"Starting evaluation of level: {level_id}")

            # Determine if we should record this episode
            should_record = (
                record_videos
                and video_output_dir is not None
                and videos_recorded < max_videos_per_category
            )

            try:
                # Create environment factory function to avoid closure issues
                def make_env(lvl_data=level_data):
                    logger.debug(f"Creating environment for level: {level_id}")
                    config = EnvironmentConfig.for_training()
                    env = NppEnvironment(config=config)
                    # Load the specific map from level_data
                    if "map_data" in lvl_data:
                        logger.debug(f"Loading map data for level: {level_id}")
                        env.nplay_headless.load_map_from_map_data(lvl_data["map_data"])
                    return env

                # Wrap in DummyVecEnv to match the format expected by the model
                # Models trained with vectorized environments expect vectorized observations
                logger.debug(
                    f"Wrapping environment in DummyVecEnv for level: {level_id}"
                )
                env = DummyVecEnv([make_env])

                logger.debug(f"Resetting environment for level: {level_id}")
                obs = env.reset()
                logger.debug(
                    f"Environment reset complete for level: {level_id}, obs keys: {obs.keys() if isinstance(obs, dict) else type(obs)}"
                )

                done = False
                steps = 0
                episode_reward = 0
                start_time = time.time()

                # Initialize video recorder if needed
                video_recorder = None
                if should_record:
                    # We'll determine success/failure after episode, so use placeholder
                    video_filename = f"{category}_{level_idx:03d}_temp.mp4"
                    video_path = Path(video_output_dir) / category / video_filename
                    video_recorder = create_video_recorder(
                        output_path=str(video_path),
                        fps=video_fps,
                    )
                    if video_recorder:
                        video_recorder.start_recording()
                        # Record initial frame
                        frame = env.render()
                        if frame is not None:
                            video_recorder.record_frame(
                                frame[0] if isinstance(frame, tuple) else frame
                            )

                logger.debug(f"Starting episode loop for level: {level_id}")
                while not done and steps < max_steps:
                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        logger.warning(
                            f"Level {level_id} timed out after {elapsed_time:.1f}s (timeout={timeout}s, steps={steps})"
                        )
                        break
                    # Get action from model
                    if steps % 100 == 0:  # Log every 100 steps to avoid spam
                        logger.debug(f"Level {level_id}: step {steps}/{max_steps}")

                    # Unwrap DDP-wrapped policy for inference if needed
                    with unwrap_policy_for_inference(model):
                        action, _ = model.predict(obs, deterministic=deterministic)

                    # Step environment
                    # Note: DummyVecEnv returns 4 values (old gym API) even though
                    # the underlying environment uses the new Gymnasium API (5 values)
                    obs, reward, done, info = env.step(action)
                    # DummyVecEnv returns arrays, so extract the first element
                    done = done[0]
                    reward = reward[0]
                    info = info[0]

                    # Record frame if video recording is enabled
                    if video_recorder and video_recorder.is_recording:
                        frame = env.render()
                        if frame is not None:
                            video_recorder.record_frame(
                                frame[0] if isinstance(frame, tuple) else frame
                            )

                    episode_reward += reward
                    steps += 1

                elapsed_time = time.time() - start_time
                logger.debug(
                    f"Episode complete for level {level_id}: steps={steps}, reward={episode_reward:.2f}, time={elapsed_time:.1f}s"
                )

                # Record results
                success = info.get("is_success", False) if done else False
                successes.append(1 if success else 0)
                episode_steps.append(steps)
                rewards.append(episode_reward)

                # Save and rename video if recording
                if video_recorder and video_recorder.is_recording:
                    video_recorder.stop_recording(save=True)

                    # Rename video file with success/failure indicator
                    success_label = "success" if success else "failure"
                    final_video_filename = (
                        f"{category}_{level_idx:03d}_{success_label}.mp4"
                    )
                    final_video_path = (
                        Path(video_output_dir) / category / final_video_filename
                    )

                    temp_video_path = (
                        Path(video_output_dir)
                        / category
                        / f"{category}_{level_idx:03d}_temp.mp4"
                    )
                    if temp_video_path.exists():
                        temp_video_path.rename(final_video_path)
                        videos_recorded += 1
                        logger.debug(f"Saved video: {final_video_path.name}")

                logger.debug(f"Closing environment for level: {level_id}")
                env.close()
                logger.debug(f"Finished evaluation of level: {level_id}")

            except Exception as e:
                logger.error(
                    f"Failed to evaluate level {level_data.get('level_id', 'unknown')}: {e}"
                )
                successes.append(0)
                episode_steps.append(max_steps)
                rewards.append(0)

        # Calculate metrics
        success_rate = np.mean(successes)
        avg_steps = np.mean(episode_steps)
        avg_reward = np.mean(rewards)

        # Efficiency: success rate / normalized steps
        efficiency = success_rate / (avg_steps / max_steps) if avg_steps > 0 else 0

        return {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "std_steps": np.std(episode_steps),
            "avg_reward": avg_reward,
            "efficiency": efficiency,
            "episode_steps": episode_steps,
            "successes": successes,
            "n_episodes": len(levels),
        }

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON.

        Args:
            results: Results dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        results_serializable = convert_types(results)

        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report.

        Args:
            results: Evaluation results dictionary

        Returns:
            Markdown-formatted report string
        """
        lines = [
            "# Evaluation Report\n",
            "## Overall Performance\n",
            f"- **Success Rate**: {results['overall']['success_rate']:.2%}",
            f"- **Average Steps**: {results['overall']['avg_steps']:.1f} ± {results['overall']['std_steps']:.1f}",
            f"- **Efficiency**: {results['overall']['efficiency']:.3f}",
            f"- **Total Episodes**: {results['overall']['total_episodes']}\n",
            "## Performance by Category\n",
        ]

        for category, metrics in results["by_category"].items():
            lines.append(f"\n### {category.title()}")
            lines.append(f"- Success Rate: {metrics['success_rate']:.2%}")
            lines.append(f"- Avg Steps: {metrics['avg_steps']:.1f}")
            lines.append(f"- Efficiency: {metrics['efficiency']:.3f}")
            lines.append(f"- Episodes: {metrics['n_episodes']}")

        return "\n".join(lines)
