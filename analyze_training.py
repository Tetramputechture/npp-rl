#!/usr/bin/env python3
"""Comprehensive analysis of the RL training data"""

import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
from pathlib import Path

def load_tensorboard_data(logdir):
    """Load tensorboard event file and extract all metrics"""
    event_files = list(Path(logdir).glob('events.out.tfevents.*'))
    if not event_files:
        raise ValueError(f"No event files found in {logdir}")
    
    print(f"Loading tensorboard data from: {event_files[0]}")
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()
    
    # Get all scalar tags
    tags = ea.Tags()['scalars']
    print(f"\nFound {len(tags)} scalar metrics:")
    for tag in sorted(tags):
        print(f"  - {tag}")
    
    # Extract all metrics
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }
    
    return data, tags

def analyze_curriculum_progression(data, tags):
    """Analyze curriculum learning progression"""
    print("\n" + "="*80)
    print("CURRICULUM LEARNING ANALYSIS")
    print("="*80)
    
    curriculum_tags = [t for t in tags if 'curriculum' in t.lower() or 'stage' in t.lower()]
    
    for tag in sorted(curriculum_tags):
        steps = data[tag]['steps']
        values = data[tag]['values']
        print(f"\n{tag}:")
        print(f"  Steps: {len(steps)}, Range: [{min(steps)}, {max(steps)}]")
        if values:
            print(f"  Values: min={min(values):.4f}, max={max(values):.4f}, final={values[-1]:.4f}")

def analyze_rewards(data, tags):
    """Analyze reward structure and progression"""
    print("\n" + "="*80)
    print("REWARD ANALYSIS")
    print("="*80)
    
    reward_tags = [t for t in tags if 'reward' in t.lower()]
    
    for tag in sorted(reward_tags):
        steps = data[tag]['steps']
        values = data[tag]['values']
        if not values:
            continue
            
        print(f"\n{tag}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Min: {min(values):.4f}")
        print(f"  Max: {max(values):.4f}")
        print(f"  Final 10 avg: {np.mean(values[-10:]) if len(values) >= 10 else np.mean(values):.4f}")

def analyze_actions(data, tags):
    """Analyze action distribution"""
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    action_tags = [t for t in tags if 'action' in t.lower()]
    
    for tag in sorted(action_tags):
        steps = data[tag]['steps']
        values = data[tag]['values']
        if not values:
            continue
            
        print(f"\n{tag}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Range: [{min(values):.4f}, {max(values):.4f}]")

def analyze_losses(data, tags):
    """Analyze training losses"""
    print("\n" + "="*80)
    print("LOSS ANALYSIS")
    print("="*80)
    
    loss_tags = [t for t in tags if 'loss' in t.lower()]
    
    for tag in sorted(loss_tags):
        steps = data[tag]['steps']
        values = data[tag]['values']
        if not values:
            continue
            
        print(f"\n{tag}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Initial: {values[0]:.4f}")
        print(f"  Final: {values[-1]:.4f}")
        print(f"  Trend: {'↓ Decreasing' if values[-1] < values[0] else '↑ Increasing'}")

def analyze_success_rates(data, tags):
    """Analyze episode success rates"""
    print("\n" + "="*80)
    print("SUCCESS RATE ANALYSIS")
    print("="*80)
    
    success_tags = [t for t in tags if 'success' in t.lower() or 'completion' in t.lower()]
    
    for tag in sorted(success_tags):
        steps = data[tag]['steps']
        values = data[tag]['values']
        if not values:
            continue
            
        print(f"\n{tag}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Final 20 avg: {np.mean(values[-20:]) if len(values) >= 20 else np.mean(values):.4f}")
        print(f"  Max achieved: {max(values):.4f}")

def main():
    results_dir = '/workspace/npp-rl/latest-training-results'
    
    # Load configuration
    print("CONFIGURATION ANALYSIS")
    print("="*80)
    with open(f'{results_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    print("\nKey Training Parameters:")
    print(f"  Total timesteps: {config['total_timesteps']:,}")
    print(f"  Num environments: {config['num_envs']}")
    print(f"  Batch size: {config['hardware_profile_settings']['batch_size']}")
    print(f"  n_steps: {config['hardware_profile_settings']['n_steps']}")
    print(f"  Learning rate: {config['hardware_profile_settings']['learning_rate']}")
    print(f"  PBRS gamma: {config['pbrs_gamma']}")
    print(f"  Curriculum enabled: {config['use_curriculum']}")
    print(f"  Curriculum threshold: {config['curriculum_threshold']}")
    print(f"  BC epochs: {config['bc_epochs']}")
    print(f"  Frame stacking: {config['enable_visual_frame_stacking']} (size={config['visual_stack_size']})")
    print(f"  Mine avoidance reward: {config['enable_mine_avoidance_reward']}")
    
    # Load tensorboard data
    data, tags = load_tensorboard_data(results_dir)
    
    # Perform analyses
    analyze_curriculum_progression(data, tags)
    analyze_rewards(data, tags)
    analyze_actions(data, tags)
    analyze_losses(data, tags)
    analyze_success_rates(data, tags)
    
    # Load all_results.json if available
    all_results_path = f'{results_dir}/all_results.json'
    if os.path.exists(all_results_path):
        print("\n" + "="*80)
        print("DETAILED RESULTS ANALYSIS")
        print("="*80)
        with open(all_results_path, 'r') as f:
            all_results = json.load(f)
        
        print(f"\nTotal entries in results: {len(all_results)}")
        if all_results:
            # Sample first entry
            sample_entry = all_results[0] if isinstance(all_results, list) else list(all_results.values())[0]
            print(f"\nSample result keys: {list(sample_entry.keys())}")

if __name__ == '__main__':
    main()
