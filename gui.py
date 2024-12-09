from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from npp_rl.game.game_process import GameProcess
from npp_rl.game.game_config import game_config
from npp_rl.environments.nplusplus import NPlusPlus
from npp_rl.game.level_parser import parse_level
import numpy as np
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
    CANVAS_SIZE = (1280, 720)

    def __init__(self, root: tk.Tk, game_process: GameProcess):
        """Initialize the NinjAI interface."""
        self.root = root
        self.game = game_process
        self.last_game_state = None
        self.level_data = None
        self.debug_overlay = None

        # Control variables
        self.training = tk.BooleanVar()
        self.automate_init_screen = tk.BooleanVar()
        self.game_image: Optional[ImageTk.PhotoImage] = None
        self.current_level_coordinates = None

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
            'start_button': None,
            'state_label': None,
            'canvas': None,
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
        self._create_game_display(left_frame)

        # Create right column components
        self._create_control_panel(right_frame)

    def _create_game_controls(self, parent: ttk.Frame):
        """Create game control widgets in a horizontal layout."""
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(5, 5))

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

    def _create_game_display(self, parent: ttk.Frame):
        """Create the game display frame."""
        # Create a frame for the game display
        game_frame = ttk.Frame(parent)
        game_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Create the game canvas - initially with default size
        # The actual size will be updated when we get the playable space
        self.ui_components['game_canvas'] = tk.Canvas(
            game_frame,
            width=800,  # Default width
            height=600,  # Default height
            background='black'
        )
        self.ui_components['game_canvas'].grid(row=0, column=0, sticky="nsew")

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

        # Add level data button
        self.ui_components['set_level_data_button'] = ttk.Button(
            training_frame,
            text="Set Level Data",
            command=self._set_level_data
        )
        self.ui_components['set_level_data_button'].grid(
            row=2, column=0, sticky="w")

        # Reset controls in a horizontal layout
        reset_frame = ttk.Frame(training_frame)
        reset_frame.grid(row=3, column=0, sticky="ew", pady=(5, 0))

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

        # Save config button
        self.ui_components['save_config_button'] = ttk.Button(
            override_frame,
            text="Save Config",
            command=game_config.save_config
        )
        self.ui_components['save_config_button'].grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))

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

    def _update_frame_display(self):
        """Update the game frame display."""
        if self.game.current_frame is not None:
            # Get the full frame
            full_frame = self.game.current_frame

            # Get playable space coordinates from level data if available
            if hasattr(self, 'current_level_data') and self.current_level_data:
                min_x, min_y, max_x, max_y = self.current_level_data.playable_space
                # Crop the frame to playable space
                cropped_frame = full_frame[min_y:max_y, min_x:max_x]

                # Update canvas size to match cropped frame
                canvas = self.ui_components['game_canvas']
                canvas.config(width=max_x-min_x, height=max_y-min_y)

                # Convert cropped frame to image
                img = Image.fromarray(cropped_frame)
            else:
                # Show full frame if no level data
                img = Image.fromarray(full_frame)

            self.game_image = ImageTk.PhotoImage(img)
            canvas = self.ui_components['game_canvas']

            if canvas:
                # Clear previous frame and debug overlay
                # This will also clear any previous debug overlay
                canvas.delete("all")
                # Draw new frame
                canvas.create_image(0, 0, anchor=tk.NW, image=self.game_image)

                # Draw debug overlay if we have level data
                if hasattr(self, 'current_level_data') and self.current_level_data:
                    self._draw_debug_overlay()

    def _update_game_controls(self):
        """Update game control states."""
        if self.game.started:
            self.ui_components['start_button'].config(
                text="Game Started", state=tk.DISABLED)
        else:
            self.ui_components['start_button'].config(
                text="Start Game", state=tk.NORMAL)
            if self.ui_components['canvas'] is not None:
                self.ui_components['canvas'].delete("all")
            self.game_image = None

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

    def _set_level_data(self):
        """Parse and set the current level data."""
        self.current_level_data = parse_level()
        game_config.set_level_data(self.current_level_data)

        # Debug print
        print(f"Parsed level data:")
        print(f"Playable space: {self.current_level_data.playable_space}")
        print(
            f"Number of mines: {len(self.current_level_data.mine_coordinates)}")
        print(f"Mine coordinates: {self.current_level_data.mine_coordinates}")

        # Force an update to draw debug information
        self._update_frame_display()

    def _draw_debug_overlay(self):
        """Draw debug information on the game canvas."""
        if not self.current_level_data or not self.current_level_data.mine_coordinates:
            return

        canvas = self.ui_components['game_canvas']
        min_x, min_y, max_x, max_y = self.current_level_data.playable_space

        # Draw mines as red squares
        for mine_x, mine_y in self.current_level_data.mine_coordinates:
            # Convert mine coordinates to canvas coordinates, accounting for cropping
            canvas_x = mine_x - min_x
            canvas_y = mine_y - min_y

            # Draw a red square around each mine (10x10 pixels)
            canvas.create_rectangle(
                canvas_x - 5, canvas_y - 5,
                canvas_x + 5, canvas_y + 5,
                outline='red',
                fill='',  # Add empty fill to make outline more visible
                width=2,
                tags="debug_overlay"
            )

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

    def _start_update_loop(self):
        """Start the main update loop."""
        self.game.loop()
        self.update_display()
        self.root.after(self.UPDATE_INTERVAL, self._start_update_loop)
