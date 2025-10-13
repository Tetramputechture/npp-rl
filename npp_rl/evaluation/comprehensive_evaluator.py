"""Comprehensive model evaluator for N++ test suite.

Evaluates trained models across standardized test levels with detailed metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Evaluates trained models on N++ test suite."""
    
    def __init__(
        self,
        test_dataset_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize evaluator.
        
        Args:
            test_dataset_path: Path to test dataset
            device: Device for model inference
        """
        from npp_rl.evaluation.test_suite_loader import TestSuiteLoader
        
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
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on model.
        
        Args:
            model: Trained model (SB3-compatible)
            num_episodes_per_category: Episodes per category (None = all)
            max_steps_per_episode: Maximum steps per episode
            deterministic: Use deterministic policy
            
        Returns:
            Comprehensive results dictionary
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive evaluation")
        logger.info("=" * 60)
        
        results = {
            'overall': {},
            'by_category': {},
            'by_difficulty_tier': defaultdict(list),
            'failure_modes': []
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
                deterministic=deterministic
            )
            
            results['by_category'][category] = category_results
            
            # Aggregate for overall stats
            all_success_rates.append(category_results['success_rate'])
            all_steps.extend(category_results['episode_steps'])
            all_efficiencies.append(category_results['efficiency'])
        
        # Calculate overall metrics
        results['overall'] = {
            'success_rate': np.mean(all_success_rates),
            'avg_steps': np.mean(all_steps) if all_steps else 0,
            'std_steps': np.std(all_steps) if all_steps else 0,
            'efficiency': np.mean(all_efficiencies),
            'total_episodes': len(all_steps)
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
        deterministic: bool
    ) -> Dict[str, Any]:
        """Evaluate model on a specific category.
        
        Args:
            model: Model to evaluate
            category: Category name
            levels: List of level data
            max_steps: Max steps per episode
            deterministic: Use deterministic policy
            
        Returns:
            Category results dictionary
        """
        from nclone.gym_environment.npp_environment import NppEnvironment
        
        successes = []
        episode_steps = []
        rewards = []
        
        for level_data in tqdm(levels, desc=f"Eval {category}", leave=False):
            try:
                # Create environment from level data
                env = NppEnvironment(map_data=level_data.get('map_data'))
                
                obs, _ = env.reset()
                done = False
                steps = 0
                episode_reward = 0
                
                while not done and steps < max_steps:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=deterministic)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    steps += 1
                
                # Record results
                success = info.get('success', False) if done else False
                successes.append(1 if success else 0)
                episode_steps.append(steps)
                rewards.append(episode_reward)
                
                env.close()
                
            except Exception as e:
                logger.error(f"Failed to evaluate level {level_data.get('level_id', 'unknown')}: {e}")
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
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'std_steps': np.std(episode_steps),
            'avg_reward': avg_reward,
            'efficiency': efficiency,
            'episode_steps': episode_steps,
            'successes': successes,
            'n_episodes': len(levels)
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
        
        with open(output_path, 'w') as f:
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
            "## Performance by Category\n"
        ]
        
        for category, metrics in results['by_category'].items():
            lines.append(f"\n### {category.title()}")
            lines.append(f"- Success Rate: {metrics['success_rate']:.2%}")
            lines.append(f"- Avg Steps: {metrics['avg_steps']:.1f}")
            lines.append(f"- Efficiency: {metrics['efficiency']:.3f}")
            lines.append(f"- Episodes: {metrics['n_episodes']}")
        
        return '\n'.join(lines)
