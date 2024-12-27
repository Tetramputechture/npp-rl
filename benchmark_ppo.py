import cProfile
import pstats
from pstats import SortKey
import datetime
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from npp_rl.environments.nplusplus import NPlusPlus
from npp_rl.agents.npp_agent_ppo import create_ppo_agent, setup_training_env


def benchmark_ppo_training(total_timesteps=1000, n_envs=4, render_mode='rgb_array'):
    """
    Benchmark PPO training performance using cProfile.

    Args:
        total_timesteps: Number of timesteps to train for
        n_envs: Number of parallel environments
        render_mode: Rendering mode ('human' or 'rgb_array')
    """
    # Create timestamp for unique profiling data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = Path(f'./benchmark_data/ppo_profile_{timestamp}')
    profile_dir.mkdir(exist_ok=True, parents=True)

    # Setup environments
    if render_mode == 'human':
        print('Benchmarking in human mode with 1 environment')
        vec_env = make_vec_env(
            lambda: NPlusPlus(render_mode='human', enable_frame_stack=True),
            n_envs=1,
            vec_env_cls=DummyVecEnv
        )
    else:
        print(f'Benchmarking in rgb_array mode with {n_envs} environments')
        vec_env = make_vec_env(
            lambda: NPlusPlus(render_mode='rgb_array',
                              enable_frame_stack=True),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv
        )

    # Setup training environment
    wrapped_env, _ = setup_training_env(vec_env)

    # Create model
    model = create_ppo_agent(wrapped_env, n_steps=512,
                             tensorboard_log=str(profile_dir))

    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()

    # Train for specified timesteps
    model.learn(total_timesteps=total_timesteps)

    profiler.disable()

    # Save and analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)

    # Save detailed stats to file
    stats_file = profile_dir / 'profile_stats.txt'
    with open(stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(SortKey.TIME)

        # Print overall stats
        stats.print_stats()

        # Print callers and callees for expensive functions
        stats.print_callers()
        stats.print_callees()

    print(f"\nProfiling data saved to {stats_file}")

    return profile_dir


if __name__ == "__main__":
    # Run benchmark with 10000 timesteps
    benchmark_dir = benchmark_ppo_training(total_timesteps=1000)
    print(f"Benchmark completed. Results saved in {benchmark_dir}")
