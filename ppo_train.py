import argparse
from npp_rl.agents.npp_agent_ppo import start_training

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
    args = parser.parse_args()

    start_training(render_mode=args.render_mode, num_envs=args.num_envs)
