#!/usr/bin/env python3
"""
Comprehensive TensorBoard Events Analysis Script

This script extracts and analyzes all metrics from the TensorBoard events file
to provide insights into RL training effectiveness.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_tensorboard_data(logdir):
    """Load all data from tensorboard events file."""
    print(f"Loading TensorBoard data from {logdir}...")
    
    # Find the events file
    events_files = list(Path(logdir).glob("events.out.tfevents.*"))
    if not events_files:
        raise FileNotFoundError(f"No events files found in {logdir}")
    
    events_file = str(events_files[0])
    print(f"Found events file: {events_file}")
    
    # Load events
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()
    
    # Get all available tags
    scalar_tags = ea.Tags()['scalars']
    print(f"\nFound {len(scalar_tags)} scalar metrics:")
    for tag in sorted(scalar_tags):
        print(f"  - {tag}")
    
    # Extract all scalar data
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': e.step, 'value': e.value, 'wall_time': e.wall_time}
            for e in events
        ])
    
    return data, scalar_tags

def analyze_curriculum_progression(data, tags):
    """Analyze curriculum learning progression."""
    print("\n" + "="*80)
    print("CURRICULUM PROGRESSION ANALYSIS")
    print("="*80)
    
    curriculum_tags = [t for t in tags if 'curriculum' in t.lower()]
    
    if not curriculum_tags:
        print("No curriculum metrics found.")
        return {}
    
    results = {}
    
    # Success rates by stage
    success_tags = [t for t in curriculum_tags if 'success' in t.lower()]
    print(f"\n=== Success Rates by Stage ===")
    for tag in sorted(success_tags):
        df = data[tag]
        if len(df) > 0:
            final_success = df['value'].iloc[-1]
            max_success = df['value'].max()
            mean_success = df['value'].mean()
            print(f"\n{tag}:")
            print(f"  Final: {final_success:.3f}")
            print(f"  Max: {max_success:.3f}")
            print(f"  Mean: {mean_success:.3f}")
            results[tag] = {
                'final': float(final_success),
                'max': float(max_success),
                'mean': float(mean_success)
            }
    
    # Current stage progression
    stage_tags = [t for t in curriculum_tags if 'stage' in t.lower() or 'level' in t.lower()]
    print(f"\n=== Stage Progression ===")
    for tag in sorted(stage_tags):
        df = data[tag]
        if len(df) > 0:
            print(f"\n{tag}:")
            print(f"  Initial: {df['value'].iloc[0]}")
            print(f"  Final: {df['value'].iloc[-1]}")
            print(f"  Max reached: {df['value'].max()}")
            results[tag] = {
                'initial': float(df['value'].iloc[0]),
                'final': float(df['value'].iloc[-1]),
                'max': float(df['value'].max())
            }
    
    return results

def analyze_action_distributions(data, tags):
    """Analyze action space coverage and exploration."""
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    action_tags = [t for t in tags if 'action' in t.lower()]
    
    if not action_tags:
        print("No action metrics found.")
        return {}
    
    results = {}
    
    print(f"\nFound {len(action_tags)} action-related metrics:")
    for tag in sorted(action_tags):
        df = data[tag]
        if len(df) > 0:
            final_val = df['value'].iloc[-1]
            mean_val = df['value'].mean()
            std_val = df['value'].std()
            print(f"\n{tag}:")
            print(f"  Final: {final_val:.4f}")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  Std: {std_val:.4f}")
            results[tag] = {
                'final': float(final_val),
                'mean': float(mean_val),
                'std': float(std_val)
            }
    
    return results

def analyze_ppo_metrics(data, tags):
    """Analyze PPO training metrics."""
    print("\n" + "="*80)
    print("PPO TRAINING METRICS ANALYSIS")
    print("="*80)
    
    ppo_keywords = ['loss', 'entropy', 'clip', 'value', 'policy', 'kl', 'approx_kl']
    ppo_tags = [t for t in tags if any(kw in t.lower() for kw in ppo_keywords)]
    
    if not ppo_tags:
        print("No PPO metrics found.")
        return {}
    
    results = {}
    
    # Group by category
    categories = {
        'Loss': ['loss'],
        'Entropy': ['entropy'],
        'Clipping': ['clip'],
        'KL Divergence': ['kl', 'approx_kl'],
        'Value': ['value'],
        'Policy': ['policy']
    }
    
    for category, keywords in categories.items():
        cat_tags = [t for t in ppo_tags if any(kw in t.lower() for kw in keywords)]
        if cat_tags:
            print(f"\n=== {category} ===")
            for tag in sorted(cat_tags):
                df = data[tag]
                if len(df) > 0:
                    initial = df['value'].iloc[0]
                    final = df['value'].iloc[-1]
                    mean_val = df['value'].mean()
                    std_val = df['value'].std()
                    print(f"\n{tag}:")
                    print(f"  Initial: {initial:.6f}")
                    print(f"  Final: {final:.6f}")
                    print(f"  Mean: {mean_val:.6f}")
                    print(f"  Std: {std_val:.6f}")
                    print(f"  Change: {((final - initial) / abs(initial) * 100) if initial != 0 else 0:.2f}%")
                    results[tag] = {
                        'initial': float(initial),
                        'final': float(final),
                        'mean': float(mean_val),
                        'std': float(std_val)
                    }
    
    return results

def analyze_reward_and_episodes(data, tags):
    """Analyze reward structure and episode statistics."""
    print("\n" + "="*80)
    print("REWARD AND EPISODE ANALYSIS")
    print("="*80)
    
    reward_tags = [t for t in tags if 'reward' in t.lower() or 'episode' in t.lower()]
    
    if not reward_tags:
        print("No reward/episode metrics found.")
        return {}
    
    results = {}
    
    # Group by metric type
    categories = {
        'Episode Rewards': ['reward'],
        'Episode Lengths': ['length', 'len'],
        'Episode Statistics': ['episode']
    }
    
    for category, keywords in categories.items():
        cat_tags = [t for t in reward_tags if any(kw in t.lower() for kw in keywords)]
        if cat_tags:
            print(f"\n=== {category} ===")
            for tag in sorted(cat_tags):
                df = data[tag]
                if len(df) > 0:
                    initial = df['value'].iloc[0]
                    final = df['value'].iloc[-1]
                    mean_val = df['value'].mean()
                    max_val = df['value'].max()
                    min_val = df['value'].min()
                    print(f"\n{tag}:")
                    print(f"  Initial: {initial:.4f}")
                    print(f"  Final: {final:.4f}")
                    print(f"  Mean: {mean_val:.4f}")
                    print(f"  Max: {max_val:.4f}")
                    print(f"  Min: {min_val:.4f}")
                    results[tag] = {
                        'initial': float(initial),
                        'final': float(final),
                        'mean': float(mean_val),
                        'max': float(max_val),
                        'min': float(min_val)
                    }
    
    return results

def analyze_learning_rate_and_scheduling(data, tags):
    """Analyze learning rate and other hyperparameter scheduling."""
    print("\n" + "="*80)
    print("LEARNING RATE AND SCHEDULING ANALYSIS")
    print("="*80)
    
    lr_tags = [t for t in tags if 'learning' in t.lower() or 'lr' in t.lower()]
    
    if not lr_tags:
        print("No learning rate metrics found.")
        return {}
    
    results = {}
    
    for tag in sorted(lr_tags):
        df = data[tag]
        if len(df) > 0:
            initial = df['value'].iloc[0]
            final = df['value'].iloc[-1]
            print(f"\n{tag}:")
            print(f"  Initial: {initial:.6e}")
            print(f"  Final: {final:.6e}")
            print(f"  Schedule type: {'Decreasing' if final < initial else 'Constant' if final == initial else 'Increasing'}")
            results[tag] = {
                'initial': float(initial),
                'final': float(final)
            }
    
    return results

def analyze_exploration_metrics(data, tags):
    """Analyze exploration-related metrics."""
    print("\n" + "="*80)
    print("EXPLORATION METRICS ANALYSIS")
    print("="*80)
    
    exploration_tags = [t for t in tags if any(kw in t.lower() for kw in ['entropy', 'exploration', 'temperature', 'epsilon'])]
    
    if not exploration_tags:
        print("No exploration metrics found.")
        return {}
    
    results = {}
    
    for tag in sorted(exploration_tags):
        df = data[tag]
        if len(df) > 0:
            initial = df['value'].iloc[0]
            final = df['value'].iloc[-1]
            mean_val = df['value'].mean()
            print(f"\n{tag}:")
            print(f"  Initial: {initial:.6f}")
            print(f"  Final: {final:.6f}")
            print(f"  Mean: {mean_val:.6f}")
            
            # Check if entropy is decreasing too fast
            if 'entropy' in tag.lower():
                pct_decrease = ((initial - final) / initial * 100) if initial != 0 else 0
                print(f"  Decrease: {pct_decrease:.2f}%")
                if pct_decrease > 80:
                    print(f"  WARNING: Entropy decreased by {pct_decrease:.1f}% - exploration may be too low!")
            
            results[tag] = {
                'initial': float(initial),
                'final': float(final),
                'mean': float(mean_val)
            }
    
    return results

def create_visualizations(data, tags, output_dir='analysis_output'):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Curriculum progression over time
    curriculum_tags = [t for t in tags if 'curriculum' in t.lower() and 'success' in t.lower()]
    if curriculum_tags:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        for tag in curriculum_tags:
            df = data[tag]
            axes[0].plot(df['step'], df['value'], label=tag, linewidth=2)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Success Rate')
        axes[0].set_title('Curriculum Stage Success Rates Over Time')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Stage progression
        stage_tags = [t for t in tags if 'current_stage' in t.lower() or 'curriculum/stage' in t.lower()]
        for tag in stage_tags:
            df = data[tag]
            axes[1].plot(df['step'], df['value'], label=tag, linewidth=2)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Stage Number')
        axes[1].set_title('Curriculum Stage Progression Over Time')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/curriculum_progression.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/curriculum_progression.png")
        plt.close()
    
    # 2. Action distributions
    action_tags = [t for t in tags if 'action_dist' in t.lower()]
    if action_tags:
        fig, ax = plt.subplots(figsize=(14, 8))
        for tag in action_tags:
            df = data[tag]
            ax.plot(df['step'], df['value'], label=tag, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Action Probability')
        ax.set_title('Action Distribution Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/action_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/action_distributions.png")
        plt.close()
    
    # 3. PPO training metrics
    loss_tags = [t for t in tags if 'loss' in t.lower() and 'train' in t.lower()]
    if loss_tags:
        fig, axes = plt.subplots(len(loss_tags), 1, figsize=(14, 4*len(loss_tags)))
        if len(loss_tags) == 1:
            axes = [axes]
        for i, tag in enumerate(loss_tags):
            df = data[tag]
            axes[i].plot(df['step'], df['value'], linewidth=2)
            axes[i].set_xlabel('Training Steps')
            axes[i].set_ylabel('Loss')
            axes[i].set_title(tag)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_losses.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/training_losses.png")
        plt.close()
    
    # 4. Episode rewards and lengths
    reward_tags = [t for t in tags if 'rollout/ep_rew_mean' in t.lower()]
    length_tags = [t for t in tags if 'rollout/ep_len_mean' in t.lower()]
    
    if reward_tags or length_tags:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        for tag in reward_tags:
            df = data[tag]
            axes[0].plot(df['step'], df['value'], label=tag, linewidth=2)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Mean Episode Reward')
        axes[0].set_title('Episode Rewards Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        for tag in length_tags:
            df = data[tag]
            axes[1].plot(df['step'], df['value'], label=tag, linewidth=2)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Mean Episode Length')
        axes[1].set_title('Episode Lengths Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/episode_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/episode_metrics.png")
        plt.close()
    
    # 5. Entropy over time
    entropy_tags = [t for t in tags if 'entropy' in t.lower()]
    if entropy_tags:
        fig, ax = plt.subplots(figsize=(14, 8))
        for tag in entropy_tags:
            df = data[tag]
            ax.plot(df['step'], df['value'], label=tag, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/entropy.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/entropy.png")
        plt.close()

def generate_summary_report(all_results, output_file='analysis_output/summary_report.json'):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Summary report saved to: {output_file}")

def main():
    # Load data
    logdir = '/workspace/npp-rl'
    data, tags = load_tensorboard_data(logdir)
    
    # Perform all analyses
    all_results = {}
    
    all_results['curriculum'] = analyze_curriculum_progression(data, tags)
    all_results['actions'] = analyze_action_distributions(data, tags)
    all_results['ppo'] = analyze_ppo_metrics(data, tags)
    all_results['rewards'] = analyze_reward_and_episodes(data, tags)
    all_results['learning_rate'] = analyze_learning_rate_and_scheduling(data, tags)
    all_results['exploration'] = analyze_exploration_metrics(data, tags)
    
    # Create visualizations
    create_visualizations(data, tags)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll results saved to 'analysis_output/' directory")

if __name__ == '__main__':
    main()
