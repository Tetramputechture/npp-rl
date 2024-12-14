import tkinter as tk
from npp_rl.game.game_process import GameProcess
from gui import NinjAI
from nplay_headless import NPlayHeadless
from PIL import Image

"""
Main entry point for the NinjAI application.
"""
if __name__ == "__main__":
    # root_tk = tk.Tk()
    # root_tk.title("NinjAI GUI")
    # game_process = GameProcess()
    # debug_gui = NinjAI(root_tk, game_process)
    # root_tk.mainloop()
    print("Initializing NPlayHeadless")
    nplay_headless = NPlayHeadless()
    print("Loading map")
    nplay_headless.load_map("../nclone/map_data")
    print("Resetting")
    nplay_headless.reset()
    print("Ticking")
    nplay_headless.tick(1, 0)
    nplay_headless.tick(0, 1)
    print("Rendering")
    frame = nplay_headless.render()
    print("Saving frame")
    # convert np array to image and save
    frame = Image.fromarray(frame)
    frame.save("frame.png")
    print("Done")
