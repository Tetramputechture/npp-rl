import argparse
from npp_rl.agents.training import train_agent, train_hierarchical_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for N++.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel simulation instances to use for training."
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human"],
        help="Render mode for the environment ('rgb_array' for training, 'human' for visualization)."
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical RL with completion planner integration."
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps."
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to existing model to resume training."
    )
    parser.add_argument(
        "--disable-exploration",
        action="store_true",
        help="Disable adaptive exploration."
    )
    parser.add_argument(
        "--subtask-reward-scale",
        type=float,
        default=0.1,
        help="Scaling factor for subtask-specific rewards (hierarchical mode only)."
    )
    args = parser.parse_args()

    if args.hierarchical:
        print("Starting hierarchical RL training with completion planner integration...")
        train_hierarchical_agent(
            num_envs=args.num_envs,
            render_mode=args.render_mode,
            total_timesteps=args.total_timesteps,
            load_model=args.load_model,
            disable_exploration=args.disable_exploration,
            subtask_reward_scale=args.subtask_reward_scale,
        )
    else:
        print("Starting standard PPO training...")
        train_agent(
            num_envs=args.num_envs,
            render_mode=args.render_mode,
            total_timesteps=args.total_timesteps,
            load_model=args.load_model,
            disable_exploration=args.disable_exploration,
        )
