import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from game.game_process import GameProcess
from game.frame_text import FrameText


class NinjAI:
    def __init__(self, root, game_process: GameProcess):
        self.root = root
        self.game = game_process
        self.last_frame_text = FrameText()
        self.last_game_state = None
        self.training = tk.BooleanVar()
        self.photo = None  # Store PhotoImage reference

        self._setup_gui()
        self._start_update_loop()

    def _setup_gui(self):
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Button to start the game
        self.start_button = ttk.Button(
            self.frame,
            text="Start Game",
            command=self.game.start
        )

        row = 0

        # Show the button
        self.start_button.grid(row=row, column=0, pady=5, sticky=tk.W)
        row += 1

        # Display current state of the game
        self.state_label = ttk.Label(
            self.frame, text=f"State: {self.game.state.state}")
        self.state_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.game_debug_view_label = ttk.Label(
            self.frame, text="Game Debug View")
        self.game_debug_view_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # Frame display
        self.canvas = tk.Canvas(self.frame, width=320, height=240)
        self.canvas.grid(row=row, column=0, columnspan=2, pady=5, sticky=tk.W)
        row += 1

        # Button to save the current frame
        self.save_frame_button = ttk.Button(
            self.frame,
            text="Save Frame",
            command=self._save_current_frame
        )
        self.save_frame_button.grid(row=row, column=0, pady=5, sticky=tk.W)
        row += 1

        # Field to enter the file name to save
        self.save_frame_entry = ttk.Entry(self.frame)
        self.save_frame_entry.grid(row=row, column=0, pady=5, sticky=tk.W)
        row += 1

        # Training button
        self.training_button = ttk.Button(
            self.frame,
            text="Start Training",
            command=self._on_training_changed
        )
        self.training_button.grid(row=row, column=0, pady=5, sticky=tk.W)
        row += 1

        # Game text display
        # Add a label on top
        self.game_text_label = ttk.Label(
            self.frame, text="Current Frame Text:")
        self.game_text_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        # Add text widgets for all text, top text, middle text, and bottom text

        # All text frame
        self.all_text_frame_label = ttk.Label(self.frame, text="All:")
        self.all_text_frame_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.all_text_frame = ttk.Frame(self.frame)
        self.all_text_frame.grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.all_text = tk.Text(self.all_text_frame,
                                height=5, width=50, wrap=tk.WORD)
        self.all_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.all_scrollbar = ttk.Scrollbar(
            self.all_text_frame, orient=tk.VERTICAL, command=self.all_text.yview)
        self.all_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.all_text.config(yscrollcommand=self.all_scrollbar.set)
        self.all_text.insert(tk.END, str(
            self.game.current_frame_text.all_text))

        # Top frame
        self.top_frame_label = ttk.Label(self.frame, text="Top:")
        self.top_frame_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.top_frame = ttk.Frame(self.frame)
        self.top_frame.grid(row=row, column=0,
                            columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.top_text = tk.Text(
            self.top_frame, height=5, width=50, wrap=tk.WORD)
        self.top_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.top_scrollbar = ttk.Scrollbar(
            self.top_frame, orient=tk.VERTICAL, command=self.top_text.yview)
        self.top_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.top_text.config(yscrollcommand=self.top_scrollbar.set)
        self.top_text.insert(tk.END, str(
            self.game.current_frame_text.top_text))

        # Middle frame
        self.middle_frame_label = ttk.Label(self.frame, text="Middle:")
        self.middle_frame_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.middle_frame = ttk.Frame(self.frame)
        self.middle_frame.grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.middle_text = tk.Text(
            self.middle_frame, height=5, width=50, wrap=tk.WORD)
        self.middle_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.middle_scrollbar = ttk.Scrollbar(
            self.middle_frame, orient=tk.VERTICAL, command=self.middle_text.yview)
        self.middle_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.middle_text.config(yscrollcommand=self.middle_scrollbar.set)
        self.middle_text.insert(tk.END, str(
            self.game.current_frame_text.middle_text))

        # Bottom frame
        self.bottom_frame_label = ttk.Label(self.frame, text="Bottom:")
        self.bottom_frame_label.grid(row=row, column=0, sticky=tk.W, pady=5)
        row += 1

        self.bottom_frame = ttk.Frame(self.frame)
        self.bottom_frame.grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.bottom_text = tk.Text(
            self.bottom_frame, height=5, width=50, wrap=tk.WORD)
        self.bottom_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.bottom_scrollbar = ttk.Scrollbar(
            self.bottom_frame, orient=tk.VERTICAL, command=self.bottom_text.yview)
        self.bottom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bottom_text.config(yscrollcommand=self.bottom_scrollbar.set)
        self.bottom_text.insert(tk.END, str(
            self.game.current_frame_text.bottom_text))

    def _save_current_frame(self):
        file_name = self.save_frame_entry.get()
        if not file_name:
            return

        # Save the current frame to the specified file
        self.game.save_current_frame(file_name)

    def _rgb_to_photo(self, rgb_frame):
        # Convert numpy RGB frame to PhotoImage
        img = Image.fromarray(rgb_frame, mode='L')
        img = img.resize((320, 240))

        return ImageTk.PhotoImage(img)

    def _update_displays(self):
        # Update frame display
        rgb_frame = self.game.current_frame  # Expecting numpy array (H,W,1)
        if rgb_frame is not None:
            self.photo = self._rgb_to_photo(rgb_frame)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # If the game current process is not None,
        # and start button text is not 'Game Started',
        # update the button text to 'Game Started'
        # and disable the button
        if self.game.process and self.start_button.cget("text") != "Game Started":
            self.start_button.config(text="Game Started", state=tk.DISABLED)
        elif not self.game.process and self.start_button.cget("text") != "Start Game":
            self.start_button.config(text="Start Game", state=tk.NORMAL)
            self.canvas.delete("all")
            self.photo = None

        # Update game text only if it has changed
        for key in ['all_text', 'top_text', 'middle_text', 'bottom_text']:
            current_text = getattr(self.game.current_frame_text, key, "")
            last_text = getattr(self.last_frame_text, key, "")
            if current_text != last_text:
                setattr(self.last_frame_text, key, current_text)
                text_widget = getattr(self, f"{key.split('_')[0]}_text")
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, str(current_text))

        # Update game state only if it has changed
        if self.game.state.state != self.last_game_state:
            self.last_game_state = self.game.state.state
            self.state_label.config(text=f"State: {self.game.state.state}")

    def _update_game(self):
        return self.game.loop()

    def _start_update_loop(self):
        self._update_displays()
        self._update_game()
        self.root.after(32, self._start_update_loop)  # ~30 FPS

    def _on_training_changed(self):
        self.game.config.set_training(self.training.get())


if __name__ == "__main__":
    root = tk.Tk()
    root.title("NinjAI GUI")
    game = GameProcess()
    debug_gui = NinjAI(root, game)
    print('Starting GUI')
    root.mainloop()
