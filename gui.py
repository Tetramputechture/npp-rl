from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from game.game_process import GameProcess
from game.game_config import game_config
from game.frame_text import extract_text
from game.game_window import get_center_frame
from environments.nplusplus import NPlusPlus
import numpy as np
import time
import threading


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
    setter: Optional[Callable] = None
    format: Callable = lambda x: str(round(float(x), 2))
    override: Optional[Callable] = None

    def get_safe(self):
        """Get the current value from memory, returning 0 if the address is not set."""
        if self.address_entry.get() == '' or int(self.address_entry.get(), 16) == 0:
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
    CANVAS_SIZE = (640, 480)

    def __init__(self, root: tk.Tk, game_process: GameProcess):
        """Initialize the NinjAI interface."""
        self.root = root
        self.game = game_process
        self.last_game_state = None

        # Control variables
        self.training = tk.BooleanVar()
        self.automate_init_screen = tk.BooleanVar()
        self.photo: Optional[ImageTk.PhotoImage] = None

        # Load config from file
        game_config.load_config()

        # Initialize UI structure
        self.ui_components = self._init_ui_components()
        self._init_gui()
        self._init_memory_addresses()

        self._start_update_loop()

    def _init_ui_components(self) -> Dict[str, Any]:
        """Initialize all UI component references."""
        return {
            'main_frame': ttk.Frame(self.root, padding="10"),
            'left_frame': ttk.Frame(self.root),
            'right_frame': ttk.Frame(self.root),
            'frame_text': None,
            'start_button': None,
            'state_label': None,
            'canvas': None,
            'img_center_canvas': None,
            'save_frame_button': None,
            'save_frame_entry': None,
            'training_button': None,
            'automate_init': None,
            'reset_button': None,
        }

    def _init_gui(self):
        """Set up the main GUI layout with horizontal grouping."""
        # Create main container frames
        main_frame = self.ui_components['main_frame']
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Create left and right frames for better organization
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # Store frames for later use
        self.ui_components['left_frame'] = left_frame
        self.ui_components['right_frame'] = right_frame

        # Create left column components
        self._create_game_controls(left_frame)
        self._create_debug_view(left_frame)

        # Create right column components
        self._create_control_panel(right_frame)
        self._create_center_canvas_display(right_frame)

    def _create_game_controls(self, parent: ttk.Frame):
        """Create game control widgets in a horizontal layout."""
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Start button and state label side by side
        self.ui_components['start_button'] = ttk.Button(
            controls_frame,
            text="Start Game",
            command=self.game.start
        )
        self.ui_components['start_button'].grid(
            row=0, column=0, padx=(0, 10), sticky="w")

        self.ui_components['state_label'] = ttk.Label(
            controls_frame,
            text=f"State: {self.game.state_manager.state}"
        )
        self.ui_components['state_label'].grid(
            row=0, column=1, sticky="w")

    def _create_debug_view(self, parent: ttk.Frame):
        """Create the game frame display canvas."""
        ttk.Label(parent, text="Game Debug View").grid(
            row=1, column=0, sticky="w", pady=(5, 0))

        self.ui_components['canvas'] = tk.Canvas(
            parent,
            width=self.CANVAS_SIZE[0],
            height=self.CANVAS_SIZE[1]
        )
        self.ui_components['canvas'].grid(
            row=2, column=0, sticky="w", pady=5)

    def _create_control_panel(self, parent: ttk.Frame):
        """Create a consolidated control panel with all controls grouped logically."""
        current_row = 0

        # State override controls
        override_frame = ttk.LabelFrame(
            parent, text="State Control", padding=5)
        override_frame.grid(row=current_row, column=0,
                            sticky="ew", pady=(0, 10))

        self.ui_components['state_override_button'] = ttk.Button(
            override_frame,
            text="Override State",
            command=self._override_state
        )
        self.ui_components['state_override_button'].grid(
            row=0, column=0, sticky="w", padx=(0, 5))

        self.ui_components['state_override_entry'] = ttk.Entry(
            override_frame, width=30)
        self.ui_components['state_override_entry'].grid(
            row=0, column=1, sticky="ew")

        current_row += 1

        # Save controls group
        save_frame = ttk.LabelFrame(parent, text="Save Options", padding=5)
        save_frame.grid(row=current_row, column=0, sticky="ew", pady=(0, 10))

        # Frame saving controls
        self.ui_components['save_frame_button'] = ttk.Button(
            save_frame,
            text="Save Frame",
            command=self._save_current_frame
        )
        self.ui_components['save_frame_button'].grid(
            row=0, column=0, sticky="w", padx=(0, 5))

        self.ui_components['save_frame_entry'] = ttk.Entry(
            save_frame, width=30)
        self.ui_components['save_frame_entry'].grid(
            row=0, column=1, sticky="ew")

        self.ui_components['save_config_button'] = ttk.Button(
            save_frame,
            text="Save Config",
            command=game_config.save_config
        )
        self.ui_components['save_config_button'].grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))

        current_row += 1

        # Training controls group
        training_frame = ttk.LabelFrame(
            parent, text="Training Controls", padding=5)
        training_frame.grid(row=current_row, column=0,
                            sticky="ew", pady=(0, 10))

        self.ui_components['training_button'] = ttk.Checkbutton(
            training_frame,
            text="Training Mode",
            variable=self.training,
            command=self._on_training_changed
        )
        self.ui_components['training_button'].grid(
            row=0, column=0, sticky="w")

        self.ui_components['automate_init'] = ttk.Checkbutton(
            training_frame,
            text="Automate Init Screen",
            variable=self.automate_init_screen,
            command=self._on_automate_init_changed
        )
        self.ui_components['automate_init'].grid(
            row=1, column=0, sticky="w")

        # Reset controls in a horizontal layout
        reset_frame = ttk.Frame(training_frame)
        reset_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        self.ui_components['reset_button'] = ttk.Button(
            reset_frame,
            text="Reset Environment",
            command=self._manual_env_reset
        )
        self.ui_components['reset_button'].grid(
            row=0, column=0, padx=(0, 5))

        self.ui_components['player_reset_button'] = ttk.Button(
            reset_frame,
            text="Reset Player",
            command=self._player_reset
        )
        self.ui_components['player_reset_button'].grid(
            row=0, column=1)

        current_row += 1  # Increment after the last control panel

        # Create frame text display
        frame_text_frame = ttk.LabelFrame(parent, text="Frame Text", padding=5)
        frame_text_frame.grid(row=current_row, column=0,
                              sticky="ew", pady=(0, 10))

        # Create a read-only text display using Label
        # Using a label with relief gives it a text-area appearance while being read-only
        self.ui_components['frame_text'] = ttk.Label(
            frame_text_frame,
            text="No text available",
            background='white',  # Give it a white background to look like a text area
            relief="sunken",     # Add a sunken relief to make it look like a text field
            anchor="w",          # Left-align the text
            padding=5            # Add some internal padding
        )
        self.ui_components['frame_text'].grid(
            row=0, column=0, sticky="ew", padx=5, pady=5
        )

        # Configure the frame to expand horizontally
        frame_text_frame.columnconfigure(0, weight=1)

    def _init_memory_addresses(self):
        """Initialize memory address fields in a grid layout with multiple columns."""
        self.memory_addresses = {}

        # Create a new frame specifically for memory addresses
        memory_frame = ttk.LabelFrame(self.ui_components['right_frame'],
                                      text="Memory Addresses",
                                      padding=5)
        memory_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))

        # Define all memory addresses (same as before)
        memory_configs = [
            ("time_remaining", self.game.game_value_fetcher.read_time_remaining,
                game_config.set_time_remaining_address, None),
            ("switch_activated", self.game.game_value_fetcher.read_switch_activated,
                game_config.set_switch_activated_address, None),
            ("player_dead", self.game.game_value_fetcher.read_player_dead,
                game_config.set_player_dead_address, None),
            ("in_air", self.game.game_value_fetcher.read_in_air,
                game_config.set_in_air_address, None),
            ("switch_x", self.game.game_value_fetcher.read_switch_x,
                game_config.set_switch_x_address, None),
            ("switch_y", self.game.game_value_fetcher.read_switch_y,
                None, lambda: game_config.switch_y_address),
            ("player_x", self.game.game_value_fetcher.read_player_x,
                None, lambda: game_config.player_x_address),
            ("player_y", self.game.game_value_fetcher.read_player_y,
                None, lambda: game_config.player_y_address),
            ("exit_door_x", self.game.game_value_fetcher.read_exit_door_x,
                None, lambda: game_config.player_y_address),
            ("exit_door_y", self.game.game_value_fetcher.read_exit_door_y,
                None, lambda: game_config.exit_door_y_address),
            ("begin_retry_text", self.game.game_value_fetcher.read_begin_retry_text,
                game_config.set_begin_retry_text_address, None),
        ]

        # Organize memory addresses in a 2-column grid
        for idx, (name, getter, setter, override) in enumerate(memory_configs):
            row = idx // 2  # Integer division to determine row
            col = idx % 2   # Remainder to determine column

            # Create a frame for each memory address pair
            addr_frame = ttk.Frame(memory_frame)
            addr_frame.grid(row=row, column=col, padx=5, pady=2, sticky="nw")

            # Create the memory address components
            address_entry = ttk.Entry(addr_frame, width=10)
            value_label = ttk.Label(addr_frame, text="0.0")

            if override is not None:
                address_entry.insert(0, hex(override()))
                address_entry.config(state='disabled')

            self.memory_addresses[name] = MemoryAddress(
                label=name.replace('_', ' ').title(),
                address_entry=address_entry,
                value_label=value_label,
                getter=getter,
                setter=setter,
                override=override
            )

            # Create the layout for each memory address
            ttk.Label(addr_frame, text=f"{name.replace('_', ' ').title()}:").grid(
                row=0, column=0, sticky="w")
            address_entry.grid(row=0, column=1, padx=2)
            value_label.grid(row=0, column=2, padx=2)

            # Configure the setter if available
            if setter is not None:
                address_entry.bind("<FocusOut>",
                                   lambda e, addr=self.memory_addresses[name]:
                                   self._on_address_changed(addr))

    def _create_memory_address_field(self, parent: ttk.Frame, row: int,
                                     memory_address: MemoryAddress) -> int:
        """
        Create UI elements for a single memory address field using grid layout consistently.
        """
        # Address entry row
        ttk.Label(parent, text=f"{memory_address.label} Address:").grid(
            row=row, column=0, sticky="w", pady=2)

        # If we have a setter, bind the address entry field to the setter
        if memory_address.setter is not None:
            memory_address.address_entry.bind(
                "<FocusOut>",
                lambda e: self._on_address_changed(memory_address)
            )
            # Enable the entry field
            memory_address.address_entry.config(state='normal')
        else:
            # If we have an override, set the entry field to the override value
            # and disable the entry field
            memory_address.address_entry.delete(0, tk.END)
            memory_address.address_entry.insert(
                0, hex(memory_address.override()))
            memory_address.address_entry.config(state='disabled')

        address_widget = memory_address.address_entry

        address_widget.grid(row=row, column=1, sticky="w", padx=(5, 0), pady=2)

        # Value display row
        ttk.Label(parent, text=f"{memory_address.label}:").grid(
            row=row + 1, column=0, sticky="w", pady=2)
        memory_address.value_label.grid(
            row=row + 1, column=1, sticky="w", padx=(5, 0), pady=2)

        return row + 2

    def _create_center_canvas_display(self, parent: ttk.Frame):
        """Initializes a small 320x240 canvas to display the center of the screen."""
        center_canvas = tk.Canvas(parent, width=320, height=240)
        center_canvas.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        self.ui_components['img_center_canvas'] = center_canvas

    def _update_frame_display(self):
        """Update the game frame display."""
        if self.game.current_frame is not None:
            rgb_frame = np.array(self.game.current_frame)
            img = Image.fromarray(rgb_frame, mode='RGB')
            self.photo = ImageTk.PhotoImage(img.resize(self.CANVAS_SIZE))
            self.ui_components['canvas'].delete("all")
            self.ui_components['canvas'].create_image(
                0, 0, anchor=tk.NW, image=self.photo)

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

            # if our addr has an override, set our address entry to the override value
            # and the value label to the current value
            if addr.override is not None:
                addr.address_entry.config(state='normal')
                addr.address_entry.delete(0, tk.END)
                addr.address_entry.insert(0, hex(addr.override()))
                addr.address_entry.config(state='disabled')
            else:
                addr.address_entry.config(state='normal')

            try:
                value = addr.get_safe()
                addr.value_label.config(text=addr.format(value))
            except (AttributeError, ValueError):
                addr.value_label.config(text=str(value))

    def _update_frame_text(self):
        """Update the frame text display with the current window center text."""
        if self.game.started:
            try:
                # center_frame = get_center_frame(self.game.current_frame)
                # center_text = extract_text(center_frame)
                # If we got valid text, update the display
                center_text = None  # This is debug-only. Lags the UI otherwise
                if center_text:
                    self.ui_components['frame_text'].config(text=center_text)
                else:
                    self.ui_components['frame_text'].config(
                        text="No text available")
            except (AttributeError, ValueError):
                self.ui_components['frame_text'].config(
                    text="Error reading text")
        else:
            self.ui_components['frame_text'].config(text="Game not started")

    def _update_center_canvas_display(self):
        """Update the center canvas display with the current frame."""
        if self.game.current_frame is not None:
            center_frame = get_center_frame(self.game.current_frame)
            img = Image.fromarray(center_frame, mode='RGB')
            photo = ImageTk.PhotoImage(img.resize((320, 240)))
            self.ui_components['img_center_canvas'].delete("all")
            self.ui_components['img_center_canvas'].create_image(
                0, 0, anchor=tk.NW, image=photo)
            self.ui_components['img_center_canvas'].image = photo

    def _player_reset(self):
        """Reset the player's position."""
        self.game.controller.focus_window()
        self.game.controller.press_reset_key()

    def _manual_env_reset(self):
        """Manually reset the game environment."""
        print("Resetting environment...")
        self.ui_components['reset_button'].config(state=tk.DISABLED)

        def reset_environment():
            self.game.controller.focus_window()
            test_env = NPlusPlus(
                self.game.game_value_fetcher, self.game.controller)
            test_env.reset()

        reset_thread = threading.Thread(target=reset_environment)
        reset_thread.start()
        self.ui_components['reset_button'].config(state=tk.NORMAL)

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
        self.game.controller.focus_window()
        game_config.set_training(self.training.get())

    def _on_automate_init_changed(self):
        """Handle automation initialization changes."""
        game_config.set_automate_init_screen(self.automate_init_screen.get())

    def update_display(self):
        """Update all display elements."""
        self._update_frame_display()
        self._update_game_controls()
        self._update_memory_address_values()
        self._update_frame_text()
        self._update_center_canvas_display()

    def _start_update_loop(self):
        """Start the main update loop."""
        self.game.loop()
        self.update_display()
        self.root.after(self.UPDATE_INTERVAL, self._start_update_loop)
