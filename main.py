# from npp_rl.agents.npp_agent_ppo import start_training
from npp_rl.agents.npp_agent_recurrent_ppo import start_training

"""
Main entry point for the NinjAI application.
"""
if __name__ == "__main__":
    start_training(render_mode="rgb_array")
