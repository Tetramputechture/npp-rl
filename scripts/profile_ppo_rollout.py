#!/usr/bin/env python3
"""
Profile PPO rollout collection step-by-step to identify training loop bottleneck.
Instruments the actual stable-baselines3 rollout collection.
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
from npp_rl.training.architecture_configs import get_architecture_config


def instrument_rollout_collection(vec_env_type: str, num_envs: int, n_steps: int = 100):
    """
    Profile the exact PPO rollout collection process.
    """
    
    print(f"\n{'='*80}")
    print(f"PPO ROLLOUT PROFILING: {vec_env_type} with {num_envs} environments")
    print(f"{'='*80}\n")
    
    level_file = str(nclone_path / "nclone/test-single-level/000 the basics")
    
    # Create environment
    print("Setting up environment...")
    import os
    if vec_env_type == "DummyVecEnv":
        os.environ["FORCE_DUMMY_VEC_ENV"] = "1"
    else:
        os.environ.pop("FORCE_DUMMY_VEC_ENV", None)
    
    env_factory = EnvironmentFactory(
        use_curriculum=False,
        frame_skip_config={"frame_skip": 4},
        pbrs_gamma=1.0,
        test_dataset_path=str(nclone_path / "datasets/test"),
        custom_map_path=level_file,
        architecture_config=get_architecture_config("attention"),
    )
    
    env = env_factory.create_training_env(num_envs=num_envs, gamma=0.99)
    print(f"✓ Environment created: {type(env).__name__}")
    
    # Create model
    print("Creating PPO model...")
    arch_config = get_architecture_config("attention")
    
    model = PPO(
        policy=arch_config.get_policy_class(),
        env=env,
        n_steps=n_steps,
        batch_size=64,  # Smaller for faster testing
        learning_rate=3e-4,
        device="cuda",
        policy_kwargs=arch_config.get_policy_kwargs(),
        verbose=0,
    )
    print("✓ Model created\n")
    
    # Instrument rollout collection
    print(f"Profiling {n_steps} rollout steps...")
    timings = defaultdict(list)
    
    # Reset
    t_reset = time.perf_counter()
    obs = env.reset()
    timings["reset"].append(time.perf_counter() - t_reset)
    print(f"  Reset time: {timings['reset'][0]*1000:.2f}ms")
    
    # Manual rollout collection with timing
    rollout_start = time.perf_counter()
    
    for step in range(n_steps):
        step_start = time.perf_counter()
        
        # 1. Policy forward
        with torch.no_grad():
            t1 = time.perf_counter()
            actions, values, log_probs = model.policy.forward(obs)
            timings["policy_forward"].append(time.perf_counter() - t1)
        
        # 2. Convert actions to numpy
        t2 = time.perf_counter()
        actions_np = actions.cpu().numpy()
        timings["action_to_cpu"].append(time.perf_counter() - t2)
        
        # 3. Environment step
        t3 = time.perf_counter()
        new_obs, rewards, dones, infos = env.step(actions_np)
        timings["env_step"].append(time.perf_counter() - t3)
        
        # 4. Buffer would add here (simulating)
        t4 = time.perf_counter()
        # This is where SparseGraphRolloutBuffer.add() would be called
        # Just time the observation copy
        if isinstance(new_obs, dict):
            _ = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in new_obs.items()}
        timings["buffer_add_sim"].append(time.perf_counter() - t4)
        
        obs = new_obs
        timings["total_step"].append(time.perf_counter() - step_start)
        
        if (step + 1) % 20 == 0:
            recent = timings["total_step"][-20:]
            print(f"  Step {step+1:3d}: {np.mean(recent)*1000:6.2f}ms avg")
    
    rollout_time = time.perf_counter() - rollout_start
    
    # Analysis
    print(f"\n{'='*80}")
    print("DETAILED BREAKDOWN")
    print(f"{'='*80}")
    print(f"Total rollout time: {rollout_time:.2f}s")
    print(f"Throughput: {n_steps / rollout_time:.1f} it/s")
    print(f"Steps per second: {(n_steps * num_envs) / rollout_time:.1f} steps/s\n")
    
    print("Per-iteration timing:")
    for component in ["policy_forward", "action_to_cpu", "env_step", "buffer_add_sim", "total_step"]:
        times = np.array(timings[component])
        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        pct = (np.mean(times) / np.mean(timings["total_step"])) * 100 if component != "total_step" else 100
        print(f"  {component:20s}: {avg_ms:6.2f}ms ± {std_ms:5.2f}ms ({pct:5.1f}%)")
    
    env.close()
    
    return {
        "throughput_it_s": n_steps / rollout_time,
        "policy_forward_ms": np.mean(timings["policy_forward"]) * 1000,
        "env_step_ms": np.mean(timings["env_step"]) * 1000,
    }


def main():
    print("\n" + "="*80)
    print("PPO ROLLOUT COLLECTION PROFILING")
    print("="*80)
    print("Identifying training loop bottleneck with full PPO infrastructure\n")
    
    # Test configurations
    configs = [
        ("DummyVecEnv", 5, 100),
        ("SubprocVecEnv", 5, 100),
    ]
    
    results = {}
    
    for vec_env_type, num_envs, n_steps in configs:
        try:
            results[(vec_env_type, num_envs)] = instrument_rollout_collection(
                vec_env_type, num_envs, n_steps
            )
        except Exception as e:
            print(f"\n❌ ERROR with {vec_env_type} {num_envs} envs: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON: DummyVecEnv vs SubprocVecEnv")
    print(f"{'='*80}\n")
    
    if ("DummyVecEnv", 5) in results and ("SubprocVecEnv", 5) in results:
        dummy = results[("DummyVecEnv", 5)]
        subproc = results[("SubprocVecEnv", 5)]
        
        print(f"DummyVecEnv  5 envs: {dummy['throughput_it_s']:6.1f} it/s "
              f"(policy: {dummy['policy_forward_ms']:.2f}ms, env: {dummy['env_step_ms']:.2f}ms)")
        print(f"SubprocVecEnv 5 envs: {subproc['throughput_it_s']:6.1f} it/s "
              f"(policy: {subproc['policy_forward_ms']:.2f}ms, env: {subproc['env_step_ms']:.2f}ms)")
        
        ratio = dummy['throughput_it_s'] / subproc['throughput_it_s']
        print(f"\nRatio: {ratio:.2f}x")
        
        if ratio > 1.5:
            print(f"\n🔴 CRITICAL: SubprocVecEnv is {ratio:.1f}x slower in training loop!")
            print("   This suggests an issue with the training infrastructure, not VecEnv itself.")
        elif ratio > 1.1:
            print(f"\n⚠️  SubprocVecEnv is moderately slower ({ratio:.2f}x)")
        else:
            print("\n✅ SubprocVecEnv performance is acceptable")


if __name__ == "__main__":
    main()

