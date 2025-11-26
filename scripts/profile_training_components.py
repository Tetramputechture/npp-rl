#!/usr/bin/env python3
"""
Detailed component-level profiling of the training pipeline.
Identifies exact bottlenecks in: rollout collection, buffer operations, model training.
"""

import time
import numpy as np
import sys
import torch
from pathlib import Path
from collections import defaultdict

# Add paths
nclone_path = Path(__file__).parent.parent.parent / "nclone"
sys.path.insert(0, str(nclone_path))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from npp_rl.training.environment_factory import EnvironmentFactory
from npp_rl.training.architecture_trainer import ArchitectureConfig


class ProfilingTimer:
    """Context manager for timing code sections."""
    def __init__(self, name, storage):
        self.name = name
        self.storage = storage
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.storage[self.name].append(elapsed)


def profile_training_pipeline(vec_env_type: str, num_envs: int, n_steps: int = 512):
    """Profile each component of the training pipeline."""
    
    print(f"\n{'='*80}")
    print(f"PROFILING TRAINING PIPELINE: {vec_env_type} with {num_envs} environments")
    print(f"{'='*80}\n")
    
    timings = defaultdict(list)
    
    # Create environment
    print("Creating environment factory...")
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    
    with ProfilingTimer("env_creation", timings):
        env_factory = EnvironmentFactory(
            use_curriculum=False,
            frame_stack_config={},
            frame_skip_config={"frame_skip": 4},
            pbrs_gamma=1.0,
            test_dataset_path=str(nclone_path / "datasets/test"),
            custom_map_path=level_file,
        )
        
        # Force VecEnv type
        import os
        if vec_env_type == "DummyVecEnv":
            os.environ["FORCE_DUMMY_VEC_ENV"] = "1"
        else:
            os.environ.pop("FORCE_DUMMY_VEC_ENV", None)
        
        env = env_factory.create_training_env(num_envs=num_envs, gamma=0.99)
    
    print(f"Environment creation time: {timings['env_creation'][0]:.2f}s")
    print(f"Environment type: {type(env).__name__}")
    
    # Create minimal model
    print("\nCreating PPO model...")
    with ProfilingTimer("model_creation", timings):
        # Use minimal architecture config
        arch_config = ArchitectureConfig(
            architecture_name="attention",
            use_graph_observations=True,
            use_game_state_observations=True,
            use_reachability_observations=True,
            use_visual_observations=False,
            graph_config={
                "architecture": "gcn",
                "hidden_dim": 128,
                "output_dim": 256,
                "num_layers": 3,
            }
        )
        
        from npp_rl.agents.deep_resnet_actor_critic_policy import DeepResNetActorCriticPolicy
        
        model = PPO(
            policy=DeepResNetActorCriticPolicy,
            env=env,
            n_steps=n_steps,
            batch_size=128,
            learning_rate=3e-4,
            device="cuda",
            policy_kwargs={
                "features_extractor_class": arch_config.get_feature_extractor_class(),
                "features_extractor_kwargs": arch_config.get_feature_extractor_kwargs(),
                "net_arch": {"pi": [512, 512, 384, 256, 256], "vf": [512, 384, 256]},
            }
        )
    
    print(f"Model creation time: {timings['model_creation'][0]:.2f}s")
    
    # Profile rollout collection (most critical)
    print(f"\nProfiling rollout collection ({n_steps} steps)...")
    
    with ProfilingTimer("env_reset", timings):
        obs = env.reset()
    
    print(f"  Initial reset time: {timings['env_reset'][0]:.2f}s")
    
    # Collect one rollout with detailed timing
    rollout_start = time.perf_counter()
    step_times = []
    policy_times = []
    
    for step in range(n_steps):
        step_start = time.perf_counter()
        
        # Policy forward pass
        with torch.no_grad():
            policy_start = time.perf_counter()
            actions, values, log_probs = model.policy.forward(obs)
            policy_times.append(time.perf_counter() - policy_start)
        
        # Environment step
        env_step_start = time.perf_counter()
        obs, rewards, dones, infos = env.step(actions.cpu().numpy())
        env_step_time = time.perf_counter() - env_step_start
        
        step_times.append(time.perf_counter() - step_start)
        
        if (step + 1) % 100 == 0:
            avg_step = np.mean(step_times[-100:]) * 1000
            avg_policy = np.mean(policy_times[-100:]) * 1000
            avg_env = np.mean([st for st in step_times[-100:]]) * 1000 - avg_policy
            print(f"  Step {step+1}/{n_steps}: {avg_step:.2f}ms total ({avg_policy:.2f}ms policy, {avg_env:.2f}ms env)")
    
    rollout_time = time.perf_counter() - rollout_start
    timings["rollout_collection"].append(rollout_time)
    
    # Analyze rollout breakdown
    step_times = np.array(step_times)
    policy_times = np.array(policy_times)
    env_times = step_times - policy_times
    
    print(f"\n{'='*80}")
    print("ROLLOUT COLLECTION BREAKDOWN")
    print(f"{'='*80}")
    print(f"Total rollout time: {rollout_time:.2f}s")
    print(f"Total steps collected: {n_steps * num_envs}")
    print(f"Throughput: {(n_steps * num_envs) / rollout_time:.1f} steps/s")
    print(f"Iterations per second: {n_steps / rollout_time:.1f} it/s")
    print("\nPer-iteration breakdown:")
    print(f"  Policy forward: {np.mean(policy_times)*1000:.2f}ms (std: {np.std(policy_times)*1000:.2f}ms)")
    print(f"  Environment step: {np.mean(env_times)*1000:.2f}ms (std: {np.std(env_times)*1000:.2f}ms)")
    print(f"  Total per iteration: {np.mean(step_times)*1000:.2f}ms")
    
    # Identify outliers
    p95_total = np.percentile(step_times, 95)
    p99_total = np.percentile(step_times, 99)
    print("\nPercentiles:")
    print(f"  P95: {p95_total*1000:.2f}ms")
    print(f"  P99: {p99_total*1000:.2f}ms")
    
    if p99_total > 2 * np.median(step_times):
        outliers = np.sum(step_times > 2 * np.median(step_times))
        print(f"  ⚠️  {outliers} outlier iterations (>2x median)")
    
    # Check for GPU memory issues
    if torch.cuda.is_available():
        print("\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Profile buffer operations (if using rollout buffer)
    print(f"\n{'='*80}")
    print("Testing buffer operations...")
    print(f"{'='*80}")
    
    # Simulate buffer add
    buffer_add_times = []
    for _ in range(10):
        start = time.perf_counter()
        # Simulate what happens during rollout
        _ = obs.copy() if isinstance(obs, np.ndarray) else {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()}
        buffer_add_times.append(time.perf_counter() - start)
    
    print(f"Observation copy time: {np.mean(buffer_add_times)*1000:.2f}ms")
    
    # Check observation size
    if isinstance(obs, dict):
        total_bytes = 0
        print("\nObservation size breakdown:")
        for key, value in obs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    size_bytes = value.element_size() * value.nelement()
                else:
                    size_bytes = value.nbytes
                total_bytes += size_bytes
                if size_bytes > 100_000:  # >100KB
                    print(f"  {key}: {size_bytes/1024:.1f} KB")
        print(f"  Total: {total_bytes/1024:.1f} KB per env set")
        print(f"  Per environment: {total_bytes/1024/num_envs:.1f} KB")
    
    env.close()
    
    return {
        "throughput_steps_per_sec": (n_steps * num_envs) / rollout_time,
        "throughput_it_per_sec": n_steps / rollout_time,
        "avg_policy_time_ms": np.mean(policy_times) * 1000,
        "avg_env_time_ms": np.mean(env_times) * 1000,
        "avg_total_time_ms": np.mean(step_times) * 1000,
        "observation_size_kb": total_bytes / 1024 if isinstance(obs, dict) else 0,
    }


def main():
    print("\n" + "="*80)
    print("DETAILED TRAINING PIPELINE PROFILING")
    print("="*80)
    print("\nGoal: Identify exact bottlenecks preventing 100+ it/s at scale")
    print("Target: 128+ environments at 100+ it/s on 80GB VRAM\n")
    
    # Test configurations
    configs = [
        ("DummyVecEnv", 5, 512),
        ("SubprocVecEnv", 5, 512),
        ("SubprocVecEnv", 8, 512),
    ]
    
    results = {}
    
    for vec_env_type, num_envs, n_steps in configs:
        try:
            results[(vec_env_type, num_envs)] = profile_training_pipeline(
                vec_env_type, num_envs, n_steps
            )
        except Exception as e:
            print(f"\n❌ ERROR with {vec_env_type} {num_envs} envs: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    for (vec_type, n_env), metrics in results.items():
        print(f"{vec_type} ({n_env} envs):")
        print(f"  Throughput: {metrics['throughput_it_per_sec']:.1f} it/s")
        print(f"  Policy time: {metrics['avg_policy_time_ms']:.2f}ms")
        print(f"  Env time: {metrics['avg_env_time_ms']:.2f}ms")
        print(f"  Total time: {metrics['avg_total_time_ms']:.2f}ms")
        print(f"  Observation size: {metrics['observation_size_kb']:.1f} KB")
        print()
    
    # Bottleneck identification
    print(f"{'='*80}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*80}\n")
    
    if results:
        sample_result = list(results.values())[0]
        policy_pct = sample_result['avg_policy_time_ms'] / sample_result['avg_total_time_ms'] * 100
        env_pct = sample_result['avg_env_time_ms'] / sample_result['avg_total_time_ms'] * 100
        
        print("Time distribution:")
        print(f"  Policy forward: {policy_pct:.1f}%")
        print(f"  Environment step: {env_pct:.1f}%")
        
        if policy_pct > 60:
            print(f"\n⚠️  PRIMARY BOTTLENECK: Model inference ({policy_pct:.1f}% of time)")
            print("   Recommendations:")
            print("   - Reduce GCN layers or hidden dimensions")
            print("   - Use mixed precision (FP16)")
            print("   - Optimize graph encoder architecture")
        elif env_pct > 60:
            print(f"\n⚠️  PRIMARY BOTTLENECK: Environment step ({env_pct:.1f}% of time)")
            print("   Recommendations:")
            print("   - Increase frame skip")
            print("   - Reduce graph update frequency")
            print("   - Optimize physics simulation")
        else:
            print("\n✓ Balanced workload (no single dominant bottleneck)")
            print("  Look for IPC overhead or memory transfer issues")


if __name__ == "__main__":
    main()

