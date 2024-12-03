from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from game.game_process import GameProcess
from game.game_config import game_config
import numpy as np
from game.trainer import Trainer


@dataclass
class MemoryAddress:
    """
    Represents a memory address field in the UI with its associated components and behaviors.

    Attributes:
        label: Display name for this memory address
        address_entry: Entry widget for the address value
        value_label: Label widget displaying the current value
        getter: Function to read the current value from memory
        setter: Function to update the address in game configuration
        format: Function to format the value for display
    """
    label: str
    address_entry: ttk.Entry
    value_label: ttk.Label
    getter: Callable
    setter: Callable
    format: Callable = lambda x: str(round(float(x), 2))

    def get_safe(self):
        """Get the current value from memory, returning 0 if the address is not set."""
        if self.address_entry.get() == '' or self.address_entry.get() == '0':
            return 0

        return self.getter()


class NinjAI:
    """
    Main UI class for the NinjAI application. Handles the game interface and memory monitoring.

    The UI is organized into several sections:
    - Game controls (start/stop)
    - Debug view (game frame display)
    - Memory address monitoring
    - Training controls
    - Text display
    """

    # Class-level constants
    UPDATE_INTERVAL = 32  # 32ms = ~30fps
    CANVAS_SIZE = (320, 240)
    TEXT_DISPLAY_HEIGHT = 5
    TEXT_DISPLAY_WIDTH = 50

    def __init__(self, root: tk.Tk, game_process: GameProcess):
        """Initialize the NinjAI interface."""
        self.root = root
        self.game = game_process
        self.last_game_state = None

        # Control variables
        self.training = tk.BooleanVar()
        self.automate_init_screen = tk.BooleanVar()
        self.photo: Optional[ImageTk.PhotoImage] = None

        # Initialize UI structure
        self.ui_components = self._init_ui_components()
        self._init_gui()
        self._init_memory_addresses()

        # Initialize the trainer
        self.trainer = Trainer(
            game_process.game_value_fetcher, game_process.controller)

        self._start_update_loop()

    def _init_ui_components(self) -> Dict[str, Any]:
        """Initialize all UI component references."""
        return {
            'frame': ttk.Frame(self.root, padding="10"),
            'start_button': None,
            'state_label': None,
            'canvas': None,
            'save_frame_button': None,
            'save_frame_entry': None,
            'training_button': None,
            'automate_init': None,
            'text_display': None
        }

    def _init_gui(self):
        """Set up the main GUI layout."""
        frame = self.ui_components['frame']
        frame.grid(row=0, column=0, sticky="nsew")

        # Build UI sections sequentially
        current_row = 0
        builders = [
            self._create_game_controls,
            self._create_debug_view,
            self._create_state_override_controls,
            self._create_save_controls,
            self._create_training_controls,
            self._create_text_display
        ]

        for builder in builders:
            current_row = builder(current_row)

    def _create_game_controls(self, row: int) -> int:
        """Create game control widgets."""
        self.ui_components['start_button'] = ttk.Button(
            self.ui_components['frame'],
            text="Start Game",
            command=self.game.start
        )
        self.ui_components['start_button'].grid(
            row=row, column=0, pady=5, sticky="w")

        self.ui_components['state_label'] = ttk.Label(
            self.ui_components['frame'],
            text=f"State: {self.game.state_manager.state}"
        )
        self.ui_components['state_label'].grid(
            row=row + 1, column=0, sticky="w", pady=5)

        return row + 2

    def _create_debug_view(self, row: int) -> int:
        """Create the game frame display canvas."""
        ttk.Label(self.ui_components['frame'], text="Game Debug View").grid(
            row=row, column=0, sticky="w", pady=5
        )

        self.ui_components['canvas'] = tk.Canvas(
            self.ui_components['frame'],
            width=self.CANVAS_SIZE[0],
            height=self.CANVAS_SIZE[1]
        )
        self.ui_components['canvas'].grid(
            row=row + 1, column=0, columnspan=2, pady=5, sticky="w"
        )

        return row + 2

    def _create_state_override_controls(self, row: int) -> int:
        """Create state override controls. Just a button and a label."""
        self.ui_components['state_override_button'] = ttk.Button(
            self.ui_components['frame'],
            text="Override State",
            command=self._override_state
        )
        self.ui_components['state_override_button'].grid(
            row=row, column=0, sticky="w", pady=5)

        self.ui_components['state_override_entry'] = ttk.Entry(
            self.ui_components['frame'], width=30)
        self.ui_components['state_override_entry'].grid(
            row=row, column=1, sticky="w", padx=(5, 0), pady=5)

        return row + 1

    def _create_save_controls(self, row: int) -> int:
        """Create frame saving controls using grid layout."""
        self.ui_components['save_frame_button'] = ttk.Button(
            self.ui_components['frame'],
            text="Save Frame",
            command=self._save_current_frame
        )
        self.ui_components['save_frame_button'].grid(
            row=row, column=0, sticky="w", pady=5)

        self.ui_components['save_frame_entry'] = ttk.Entry(
            self.ui_components['frame'], width=30)
        self.ui_components['save_frame_entry'].grid(
            row=row, column=1, sticky="w", padx=(5, 0), pady=5)

        return row + 1

    def _create_training_controls(self, row: int) -> int:
        """Create training-related controls."""
        self.ui_components['training_button'] = ttk.Checkbutton(
            self.ui_components['frame'],
            text="Training Mode",
            variable=self.training,
            command=self._on_training_changed
        )
        self.ui_components['training_button'].grid(
            row=row, column=0, pady=5, sticky="w")

        self.ui_components['automate_init'] = ttk.Checkbutton(
            self.ui_components['frame'],
            text="Automate Init Screen 'Press Any Key'",
            variable=self.automate_init_screen,
            command=self._on_automate_init_changed
        )
        self.ui_components['automate_init'].grid(
            row=row + 1, column=0, pady=5, sticky="w")

        return row + 2

    def _init_memory_addresses(self):
        """
        Initialize and configure all memory address fields using grid layout.
        """
        self.memory_addresses = {}
        frame = self.ui_components['frame']
        # Start after other components
        current_row = len(self.ui_components) + 1

        # Define all memory addresses
        memory_configs = [
            ("player_x", self.game.game_value_fetcher.read_player_x,
             game_config.set_player_x_address),
            ("player_y", self.game.game_value_fetcher.read_player_y,
             game_config.set_player_y_address),
            ("time_remaining", self.game.game_value_fetcher.read_time_remaining,
             game_config.set_time_remaining_address),
            ("switch_activated", self.game.game_value_fetcher.read_switch_activated,
             game_config.set_switch_activated_address),
            ("player_dead", self.game.game_value_fetcher.read_player_dead,
             game_config.set_player_dead_address),
            ("exit_door_x", self.game.game_value_fetcher.read_exit_door_x,
             game_config.set_exit_door_x_address),
            ("exit_door_y", self.game.game_value_fetcher.read_exit_door_y,
             game_config.set_exit_door_y_address),
            ("switch_x", self.game.game_value_fetcher.read_switch_x,
             game_config.set_switch_x_address),
            ("switch_y", self.game.game_value_fetcher.read_switch_y,
             game_config.set_switch_y_address)
        ]

        for name, getter, setter in memory_configs:
            address_entry = ttk.Entry(frame, width=10)
            value_label = ttk.Label(frame, text="0.0")

            self.memory_addresses[name] = MemoryAddress(
                label=name.replace('_', ' ').title(),
                address_entry=address_entry,
                value_label=value_label,
                getter=getter,
                setter=setter
            )

            current_row = self._create_memory_address_field(
                frame, current_row, self.memory_addresses[name]
            )

    def _create_memory_address_field(self, parent: ttk.Frame, row: int,
                                     memory_address: MemoryAddress) -> int:
        """
        Create UI elements for a single memory address field using grid layout consistently.
        """
        # Address entry row
        ttk.Label(parent, text=f"{memory_address.label} Address:").grid(
            row=row, column=0, sticky="w", pady=2)
        memory_address.address_entry.grid(
            row=row, column=1, sticky="w", padx=(5, 0), pady=2)

        # Value display row
        ttk.Label(parent, text=f"{memory_address.label}:").grid(
            row=row + 1, column=0, sticky="w", pady=2)
        memory_address.value_label.grid(
            row=row + 1, column=1, sticky="w", padx=(5, 0), pady=2)

        memory_address.address_entry.bind(
            "<FocusOut>",
            lambda e: self._on_address_changed(memory_address)
        )

        return row + 2

    def _create_text_display(self, row: int) -> int:
        """Create the text display area."""
        ttk.Label(self.ui_components['frame'], text="Current Frame Text:").grid(
            row=row, column=0, sticky="w", pady=5
        )

        text_frame = ttk.Frame(self.ui_components['frame'])
        text_frame.grid(row=row + 1, column=0,
                        columnspan=2, sticky="w", pady=5)

        self.ui_components['text_display'] = tk.Text(
            text_frame,
            height=self.TEXT_DISPLAY_HEIGHT,
            width=self.TEXT_DISPLAY_WIDTH,
            wrap=tk.WORD
        )
        scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.VERTICAL,
            command=self.ui_components['text_display'].yview
        )

        self.ui_components['text_display'].pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ui_components['text_display'].config(yscrollcommand=scrollbar.set)

        return row + 2

    def _update_frame_display(self):
        """Update the game frame display."""
        if self.game.current_frame is not None:
            rgb_frame = np.array(self.game.current_frame)
            img = Image.fromarray(rgb_frame, mode='RGB')
            self.photo = ImageTk.PhotoImage(img.resize(self.CANVAS_SIZE))
            self.ui_components['canvas'].delete("all")
            self.ui_components['canvas'].create_image(
                0, 0, anchor=tk.NW, image=self.photo)

    def _update_text_display(self):
        """Update the text display with current frame text."""
        text_display = self.ui_components['text_display']
        text_display.delete(1.0, tk.END)
        text_display.insert(tk.END, str(
            self.game.state_manager.current_frame_text))

    def _update_game_controls(self):
        """Update game control states."""
        if self.game.started:
            self.ui_components['start_button'].config(
                text="Game Started", state=tk.DISABLED)
        else:
            self.ui_components['start_button'].config(
                text="Start Game", state=tk.NORMAL)
            self.ui_components['canvas'].delete("all")
            self.photo = None

        if self.game.state_manager.state != self.last_game_state:
            self.last_game_state = self.game.state_manager.state
            self.ui_components['state_label'].config(
                text=f"State: {self.game.state_manager.state}"
            )

    def _update_memory_address_values(self):
        """Update all memory address displays."""
        for addr in self.memory_addresses.values():
            if not self.game.started:
                addr.address_entry.config(state='disabled')
                addr.value_label.config(text="0.0")
                continue

            addr.address_entry.config(state='normal')
            try:
                value = addr.get_safe()
                addr.value_label.config(text=addr.format(value))
            except (AttributeError, ValueError):
                addr.value_label.config(text="N/A")

    def _on_address_changed(self, memory_address: MemoryAddress):
        """Handle changes to memory address entry fields."""
        address = memory_address.address_entry.get()
        if not address:
            address = '0'
        try:
            memory_address.setter(int(address, 16))
        except ValueError:
            memory_address.address_entry.delete(0, tk.END)
            memory_address.address_entry.insert(0, '0')

    def _override_state(self):
        """Override the current game state with the value in the entry field."""
        if state := self.ui_components['state_override_entry'].get():
            self.game.state_manager.force_set_state(state)

    def _save_current_frame(self):
        """Save the current game frame to a file."""
        if filename := self.ui_components['save_frame_entry'].get():
            self.game.save_current_frame(filename)

    def _on_training_changed(self):
        """Handle training mode changes."""
        game_config.set_training(self.training.get())

    def _on_automate_init_changed(self):
        """Handle automation initialization changes."""
        game_config.set_automate_init_screen(self.automate_init_screen.get())

    def update_display(self):
        """Update all display elements."""
        self._update_frame_display()
        self._update_game_controls()
        self._update_memory_address_values()
        self._update_text_display()

    def _start_update_loop(self):
        """Start the main update loop."""
        self.game.loop()
        self.update_display()
        self.root.after(self.UPDATE_INTERVAL, self._start_update_loop)
