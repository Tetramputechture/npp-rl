#!/usr/bin/env python3
"""
Real-time training monitoring and health checking script.

This script monitors training progress and detects potential issues early.
Run alongside training to get alerts about problems before wasting compute.

Usage:
    python tools/monitor_training.py --logdir /path/to/tensorboard/logs
    python tools/monitor_training.py --logdir /path/to/logs --check-interval 60
"""

import argparse
import time
from pathlib import Path
from collections import deque
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


class TrainingMonitor:
    """Monitor training health and detect issues early."""
    
    def __init__(self, logdir, check_interval=30):
        """
        Initialize monitoring.
        
        Args:
            logdir: Path to tensorboard log directory
            check_interval: Seconds between checks
        """
        self.logdir = Path(logdir)
        self.check_interval = check_interval
        self.last_step = 0
        self.history = {
            'mean_reward': deque(maxlen=10),
            'success_rate': deque(maxlen=10),
            'pbrs_reward': deque(maxlen=10),
            'value_loss': deque(maxlen=10),
            'entropy': deque(maxlen=10),
        }
        
    def load_latest_metrics(self):
        """Load latest metrics from tensorboard logs."""
        event_files = list(self.logdir.glob('events.out.tfevents.*'))
        if not event_files:
            return None
            
        # Load most recent event file
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        
        metrics = {}
        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                latest = events[-1]
                metrics[tag] = {
                    'step': latest.step,
                    'value': latest.value,
                    'wall_time': latest.wall_time
                }
        
        return metrics
    
    def check_health(self, metrics):
        """Check training health and return issues."""
        if not metrics:
            return []
        
        issues = []
        
        # Extract key metrics
        mean_reward = metrics.get('reward_dist/mean', {}).get('value')
        success_rate = metrics.get('episode/success_rate', {}).get('value')
        pbrs_reward = metrics.get('pbrs_rewards/pbrs_mean', {}).get('value')
        value_loss = metrics.get('train/value_loss', {}).get('value')
        entropy = metrics.get('actions/entropy', {}).get('value')
        kl_div = metrics.get('train/approx_kl', {}).get('value')
        clip_frac = metrics.get('train/clip_fraction', {}).get('value')
        noop_freq = metrics.get('actions/frequency/NOOP', {}).get('value')
        
        current_step = metrics.get('reward_dist/mean', {}).get('step', 0)
        
        # Update history
        if mean_reward is not None:
            self.history['mean_reward'].append(mean_reward)
        if success_rate is not None:
            self.history['success_rate'].append(success_rate)
        if pbrs_reward is not None:
            self.history['pbrs_reward'].append(pbrs_reward)
        if value_loss is not None:
            self.history['value_loss'].append(value_loss)
        if entropy is not None:
            self.history['entropy'].append(entropy)
        
        # Check 1: Mean reward still negative after 2M steps
        if current_step > 2_000_000 and mean_reward is not None and mean_reward < 0:
            issues.append({
                'severity': 'CRITICAL',
                'type': 'reward_structure',
                'message': f'Mean reward still negative ({mean_reward:.4f}) after 2M steps',
                'recommendation': 'Check PBRS scaling and reward constants'
            })
        
        # Check 2: PBRS rewards too small
        if pbrs_reward is not None and abs(pbrs_reward) < 0.02:
            issues.append({
                'severity': 'CRITICAL',
                'type': 'pbrs_scaling',
                'message': f'PBRS rewards too small ({pbrs_reward:.4f}), should be ±0.05-0.2',
                'recommendation': 'Increase PBRS_SWITCH_DISTANCE_SCALE and PBRS_EXIT_DISTANCE_SCALE'
            })
        
        # Check 3: Success rate not improving
        if len(self.history['success_rate']) >= 5:
            recent_sr = list(self.history['success_rate'])
            if current_step > 1_000_000 and np.std(recent_sr) < 0.01:
                issues.append({
                    'severity': 'WARNING',
                    'type': 'learning_plateau',
                    'message': f'Success rate plateaued at {success_rate:.2%}',
                    'recommendation': 'Check curriculum progression or adjust learning rate'
                })
        
        # Check 4: Value loss increasing
        if len(self.history['value_loss']) >= 5:
            recent_vl = list(self.history['value_loss'])
            if recent_vl[-1] > recent_vl[0]:
                issues.append({
                    'severity': 'WARNING',
                    'type': 'training_instability',
                    'message': f'Value loss increasing ({recent_vl[0]:.4f} -> {recent_vl[-1]:.4f})',
                    'recommendation': 'Consider reducing learning rate or batch size'
                })
        
        # Check 5: KL divergence too high
        if kl_div is not None and kl_div > 0.1:
            issues.append({
                'severity': 'WARNING',
                'type': 'policy_change',
                'message': f'KL divergence too high ({kl_div:.4f}), policy changing too fast',
                'recommendation': 'Reduce learning rate or increase clip_range'
            })
        
        # Check 6: Clip fraction too high
        if clip_frac is not None and clip_frac > 0.5:
            issues.append({
                'severity': 'WARNING',
                'type': 'clipping',
                'message': f'Clip fraction too high ({clip_frac:.2%}), updates being clipped excessively',
                'recommendation': 'Increase clip_range or reduce learning rate'
            })
        
        # Check 7: Entropy too low (policy too deterministic)
        if entropy is not None and entropy < 1.0:
            issues.append({
                'severity': 'WARNING',
                'type': 'exploration',
                'message': f'Entropy too low ({entropy:.3f}), policy too deterministic',
                'recommendation': 'Increase entropy coefficient or check for premature convergence'
            })
        
        # Check 8: NOOP usage too high
        if noop_freq is not None and noop_freq > 0.15:
            issues.append({
                'severity': 'INFO',
                'type': 'action_distribution',
                'message': f'NOOP usage high ({noop_freq:.2%}), agent standing still too often',
                'recommendation': 'Increase NOOP_ACTION_PENALTY'
            })
        
        return issues
    
    def print_status(self, metrics, issues):
        """Print current status and issues."""
        if not metrics:
            print("⏳ Waiting for training data...")
            return
        
        print("\n" + "="*80)
        print("🔍 TRAINING MONITOR STATUS")
        print("="*80)
        
        # Extract key metrics
        current_step = metrics.get('reward_dist/mean', {}).get('step', 0)
        mean_reward = metrics.get('reward_dist/mean', {}).get('value')
        success_rate = metrics.get('episode/success_rate', {}).get('value')
        pbrs_reward = metrics.get('pbrs_rewards/pbrs_mean', {}).get('value')
        entropy = metrics.get('actions/entropy', {}).get('value')
        
        print(f"\n📊 Step: {current_step:,}")
        
        if mean_reward is not None:
            emoji = "✅" if mean_reward > 0 else "❌"
            print(f"  {emoji} Mean Reward: {mean_reward:.4f}")
        
        if success_rate is not None:
            emoji = "✅" if success_rate > 0.7 else "⚠️" if success_rate > 0.5 else "❌"
            print(f"  {emoji} Success Rate: {success_rate:.2%}")
        
        if pbrs_reward is not None:
            emoji = "✅" if abs(pbrs_reward) > 0.05 else "❌"
            print(f"  {emoji} PBRS Reward: {pbrs_reward:.4f}")
        
        if entropy is not None:
            emoji = "✅" if entropy > 1.5 else "⚠️"
            print(f"  {emoji} Entropy: {entropy:.3f}")
        
        # Print issues
        if issues:
            print(f"\n⚠️  {len(issues)} ISSUE(S) DETECTED:")
            for issue in issues:
                severity_emoji = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡',
                    'INFO': '🔵'
                }[issue['severity']]
                
                print(f"\n  {severity_emoji} [{issue['severity']}] {issue['type']}")
                print(f"     Message: {issue['message']}")
                print(f"     Action: {issue['recommendation']}")
        else:
            print("\n✅ No issues detected - training looks healthy!")
        
        print("\n" + "="*80)
    
    def run(self):
        """Run monitoring loop."""
        print("🚀 Training Monitor Started")
        print(f"   Log directory: {self.logdir}")
        print(f"   Check interval: {self.check_interval}s")
        print("\nMonitoring training health...")
        
        try:
            while True:
                metrics = self.load_latest_metrics()
                issues = self.check_health(metrics)
                self.print_status(metrics, issues)
                
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor RL training health and detect issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default 30s interval
  python tools/monitor_training.py --logdir ./experiments/run1
  
  # Check every 60 seconds
  python tools/monitor_training.py --logdir ./experiments/run1 --check-interval 60
  
  # Monitor specific tensorboard log
  python tools/monitor_training.py --logdir /path/to/logs/events.out.tfevents.*
        """
    )
    
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Path to tensorboard log directory'
    )
    
    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help='Seconds between health checks (default: 30)'
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.logdir, args.check_interval)
    monitor.run()


if __name__ == '__main__':
    main()
