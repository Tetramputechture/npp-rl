from npp_rl.agents.npp_agent_recurrent_ppo import start_training

if __name__ == "__main__":
    start_training(render_mode="human", n_envs=64)
