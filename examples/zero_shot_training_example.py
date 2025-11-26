"""Example script demonstrating generalized navigation system usage.

This script shows how to integrate all components of the generalized path learning
system for training an RL agent that can generalize to completely unseen N++ levels.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the generalized navigation system
from npp_rl.training.generalized_navigation_system import (
    GeneralizedNavigationSystem, 
    GeneralizedNavigationConfig,
    create_generalized_navigation_system
)


def load_expert_demonstrations(demo_dir: str) -> List[Dict[str, Any]]:
    """Load expert demonstrations for pattern learning.
    
    Args:
        demo_dir: Directory containing demonstration files
        
    Returns:
        List of demonstration data dictionaries
    """
    # This is a placeholder - in practice, load actual demonstration data
    # Each demonstration should have 'trajectory' and 'level_data'
    
    logger.info(f"Loading demonstrations from {demo_dir}")
    
    # Example demonstration structure
    demo_data = []
    
    for i in range(5):  # Example: 5 demonstrations
        demo = {
            'trajectory': [
                {
                    'position': (100 + i*10, 100 + i*5),
                    'action': np.random.randint(0, 5),
                    'obs': {'player_x': 100 + i*10, 'player_y': 100 + i*5}
                }
                for step in range(50)  # 50 steps per demo
            ],
            'level_data': {
                'level_id': f'demo_level_{i}',
                'tile_data': np.random.randint(0, 38, (50, 50)),  # Random tiles
                'entities': [
                    {'x': 200, 'y': 200, 'type': 4},  # Exit switch
                    {'x': 400, 'y': 400, 'type': 3},  # Exit door
                    {'x': 150 + i*20, 'y': 150 + i*10, 'type': 1}  # Mine
                ]
            }
        }
        demo_data.append(demo)
    
    logger.info(f"Loaded {len(demo_data)} demonstrations")
    return demo_data


def create_test_levels() -> List[Dict[str, Any]]:
    """Create test levels for zero-shot evaluation.
    
    Returns:
        List of test level configurations
    """
    test_levels = []
    
    # Create diverse test levels with different topologies
    topologies = ['open', 'corridor', 'maze', 'vertical']
    
    for i, topology in enumerate(topologies):
        for j in range(3):  # 3 levels per topology
            level = {
                'level_id': f'test_{topology}_{j}',
                'tile_data': np.random.randint(0, 38, (60 + j*10, 60 + j*10)),
                'entities': [
                    {'x': 100, 'y': 100, 'type': 4},  # Exit switch
                    {'x': 500 + j*50, 'y': 500 + j*50, 'type': 3},  # Exit door
                    {'x': 200 + k*30, 'y': 200 + k*20, 'type': 1}  # Mines
                    for k in range(2 + j)
                ],
                'topology_type': topology
            }
            test_levels.append(level)
    
    logger.info(f"Created {len(test_levels)} test levels")
    return test_levels


def dummy_policy_evaluator(level_data: Dict[str, Any], 
                          training_system: ZeroShotTrainingSystem) -> Dict[str, Any]:
    """Dummy policy evaluator for demonstration purposes.
    
    In practice, this would run the actual RL policy on the level.
    
    Args:
        level_data: Level configuration
        training_system: The zero-shot training system
        
    Returns:
        Evaluation results
    """
    # Simulate policy evaluation
    topology = level_data.get('topology_type', 'unknown')
    
    # Simulate different success rates based on topology
    topology_difficulty = {
        'open': 0.8,      # Easier
        'corridor': 0.7,   # Medium  
        'maze': 0.6,      # Harder
        'vertical': 0.5,  # Hardest
        'unknown': 0.5
    }
    
    success_prob = topology_difficulty.get(topology, 0.5)
    success = np.random.random() < success_prob
    
    # Simulate completion time
    completion_time = np.random.uniform(10, 60) if success else None
    
    result = {
        'success': success,
        'completion_time': completion_time,
        'total_reward': np.random.uniform(0, 100) if success else 0,
        'mines_triggered': np.random.randint(0, 3),
        'deaths': 0 if success else np.random.randint(1, 5)
    }
    
    logger.debug(f"Evaluated {topology} level: {'SUCCESS' if success else 'FAILURE'}")
    return result


def main():
    """Main example demonstrating zero-shot training system usage."""
    
    logger.info("Starting Zero-Shot Training System Example")
    
    # 1. Configure the system
    config = {
        'pattern_neighborhood_sizes': [3, 5],
        'pattern_entity_radius': 24,
        'min_pattern_observations': 3,  # Lower for demo
        'num_path_candidates': 4,
        'max_waypoints': 15,
        'exploration_factor': 1.5,
        'min_exploration_prob': 0.15,
        'pbrs_learning_rate': 0.1,
        'uncertainty_bonus_weight': 0.2
    }
    
    # 2. Load expert demonstrations (for pattern learning)
    demonstrations = load_expert_demonstrations("demo_data")
    
    # 3. Create the zero-shot training system
    save_dir = "example_zero_shot_models"
    training_system = create_zero_shot_system(
        config_dict=config,
        demonstrations=demonstrations,
        save_dir=save_dir
    )
    
    logger.info("Zero-shot training system initialized")
    
    # 4. Simulate some training episodes
    logger.info("Simulating training episodes...")
    
    for episode in range(50):  # Simulate 50 training episodes
        # Create random training level
        training_level = {
            'level_id': f'training_level_{episode}',
            'tile_data': np.random.randint(0, 38, (40, 40)),
            'entities': [
                {'x': 100, 'y': 100, 'type': 4},
                {'x': 300, 'y': 300, 'type': 3},
                {'x': 150 + episode, 'y': 150, 'type': 1}
            ]
        }
        
        # Simulate trajectory data
        trajectory = [
            {
                'position': (100 + step*2, 100 + step),
                'action': np.random.randint(0, 5),
                'obs': {'player_x': 100 + step*2, 'player_y': 100 + step}
            }
            for step in range(30)
        ]
        
        # Simulate episode outcome
        episode_outcome = {
            'success': np.random.random() < 0.6,
            'completion_time': np.random.uniform(15, 45),
            'total_reward': np.random.uniform(0, 80),
            'attempted_paths': [
                {
                    'waypoints': [(100 + i*10, 100 + i*5) for i in range(5)],
                    'path_type': 'direct'
                }
            ],
            'path_outcomes': [
                {'success': True, 'distance_traveled': 200.0}
            ]
        }
        
        # Process the episode
        training_system.process_training_episode(
            level_data=training_level,
            trajectory_data=trajectory, 
            episode_outcome=episode_outcome
        )
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Processed {episode + 1} training episodes")
    
    # 5. Generate training report
    logger.info("\nTraining Report:")
    print(training_system.generate_training_report())
    
    # 6. Evaluate zero-shot performance
    logger.info("Evaluating zero-shot performance on unseen levels...")
    
    test_levels = create_test_levels()
    
    # Run zero-shot evaluation
    evaluation_results = training_system.evaluate_zero_shot_performance(
        test_levels=test_levels,
        policy_evaluator=dummy_policy_evaluator
    )
    
    # 7. Display results
    logger.info("\nZero-Shot Evaluation Results:")
    print(f"Total levels tested: {evaluation_results['total_levels']}")
    print(f"Success rate: {evaluation_results['success_rate']:.2%}")
    print(f"Average completion time: {evaluation_results['average_completion_time']:.1f}s")
    
    print("\nPer-topology performance:")
    for topology, perf in evaluation_results['topology_performance'].items():
        print(f"  {topology}: {perf['success_rate']:.2%} success, "
              f"{perf['average_time']:.1f}s avg time ({perf['attempts']} levels)")
    
    # 8. Save final state
    training_system.save_learned_state()
    logger.info(f"Saved learned state to {save_dir}")
    
    # 9. Show system statistics
    stats = training_system.get_system_statistics()
    logger.info("\nFinal System Statistics:")
    print(f"Patterns learned: {stats.get('patterns_in_database', 0)}")
    print(f"Routes discovered: {stats['route_discovery_stats'].get('total_paths_tried', 0)}")
    print(f"PBRS preference updates: {stats['adaptive_pbrs_stats'].get('preference_updates', 0)}")
    
    logger.info("Zero-Shot Training System Example Complete!")


if __name__ == "__main__":
    main()
