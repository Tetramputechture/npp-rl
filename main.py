import tkinter as tk
from npp_rl.game.game_process import GameProcess
from gui import NinjAI
from nplay_headless import NPlayHeadless
from PIL import Image
from npp_rl.simulated_agents.npp_agent_ppo import start_training

"""
Main entry point for the NinjAI application.
"""
if __name__ == "__main__":
    start_training()
