import tkinter as tk
from game.game_process import GameProcess
from gui import NinjAI


if __name__ == "__main__":
    root_tk = tk.Tk()
    root_tk.title("NinjAI GUI")
    game_process = GameProcess()
    debug_gui = NinjAI(root_tk, game_process)
    root_tk.mainloop()
