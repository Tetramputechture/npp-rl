from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from npp_rl.game.game_process import GameProcess
from npp_rl.game.game_config import game_config
from npp_rl.environments.nplusplus import NPlusPlus
from npp_rl.game.level_parser import parse_level
from npp_rl.game.training_session import training_session
from npp_rl.game.game_window import LEVEL_WIDTH, LEVEL_HEIGHT
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
    UPDATE_INTERVAL = 16  # 16ms = ~60fps
    CANVAS_SIZE = (1280, 720)
    # Color palette for rewards plot (colorblind-friendly)
    REWARD_COLORS = ['#1f77b4',  # Blue
                     '#d62728',  # Red
                     '#2ca02c',  # Green
                     '#9467bd',  # Purple
                     '#ff7f0e']  # Orange

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
        self.current_level_data = None

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

        # Set up training session info panel
        self._setup_training_session_info()

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

        # Create rewards plot
        rewards_frame = ttk.Frame(game_frame)
        rewards_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        # Create matplotlib figure for rewards
        self.rewards_fig = Figure(figsize=(8, 2), dpi=100)
        self.rewards_ax = self.rewards_fig.add_subplot(111)
        self.rewards_ax.set_title('Rewards per Step')
        self.rewards_ax.set_xlabel('Step')
        self.rewards_ax.set_ylabel('Reward')

        # Create canvas for the rewards plot
        self.rewards_canvas = FigureCanvasTkAgg(
            self.rewards_fig, master=rewards_frame)
        self.rewards_canvas.draw()
        self.rewards_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create action icons frame
        action_frame = ttk.Frame(game_frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        # Create action icons
        # Left arrow
        self.ui_components['left_icon'] = ttk.Label(
            action_frame,
            text="←",
            font=('TkDefaultFont', 20),
            width=3
        )
        self.ui_components['left_icon'].grid(row=0, column=0, padx=5)

        # Space bar
        self.ui_components['space_icon'] = ttk.Label(
            action_frame,
            text="⎵",
            font=('TkDefaultFont', 20),
            width=10
        )
        self.ui_components['space_icon'].grid(row=0, column=1, padx=5)

        # Right arrow
        self.ui_components['right_icon'] = ttk.Label(
            action_frame,
            text="→",
            font=('TkDefaultFont', 20),
            width=3
        )
        self.ui_components['right_icon'].grid(row=0, column=2, padx=5)

        # Configure grid weights to center the icons
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)
        action_frame.grid_columnconfigure(2, weight=1)

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
            elif setter is not None and hasattr(game_config, name + '_address'):
                config_value = getattr(game_config, name + '_address')
                if config_value != 0:  # Only populate if there's a non-zero value
                    address_entry.insert(0, hex(config_value))

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
                self.game_image = Image.fromarray(cropped_frame)
            else:
                # Show full frame if no level data
                self.game_image = Image.fromarray(full_frame)

            # Update the display with the new frame
            self._update_game_display()

    def _update_game_display(self):
        """Update the game display with the current frame and overlay."""
        if not self.game_image:
            return

        image = self.game_image

        # Draw paths if training
        if self.training.get():
            image = self._draw_paths_on_image(image)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image)
        canvas = self.ui_components['game_canvas']

        if canvas:
            # Clear previous frame and debug overlay
            canvas.delete("all")
            # Draw new frame
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference

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

    def _update_memory_addresses(self):
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
        """Set the current level data from the game frame."""
        self.current_level_data = parse_level()
        print("Parsed level data:")
        print(f"Playable space: {self.current_level_data.playable_space}")
        print(
            f"Number of mines: {len(self.current_level_data.mine_coordinates)}")
        print(f"Mine coordinates: {self.current_level_data.mine_coordinates}")
        game_config.level_data = self.current_level_data

    def _draw_debug_overlay(self):
        """Draw debug information on the game canvas."""
        if not self.current_level_data:
            return

        canvas = self.ui_components['game_canvas']
        min_x, min_y, max_x, max_y = self.current_level_data.playable_space

        # Draw mines as red squares if we have mine coordinates
        if self.current_level_data.mine_coordinates:
            for mine_x, mine_y in self.current_level_data.mine_coordinates:
                # Convert mine coordinates to canvas coordinates
                canvas_x = mine_x - min_x
                canvas_y = mine_y - min_y

                # Draw a red square around each mine (12x12 pixels)
                canvas.create_rectangle(
                    canvas_x - 6, canvas_y - 6,
                    canvas_x + 6, canvas_y + 6,
                    outline='red',
                    fill='',
                    width=2,
                    tags="debug_overlay"
                )

        # Draw player position and vector to nearest mine
        if self.game and self.game.game_value_fetcher:
            player_x = self.game.game_value_fetcher.read_player_x()
            player_y = self.game.game_value_fetcher.read_player_y()

            # Draw player bounding box
            player_box = self.current_level_data.get_player_bounding_box(
                player_x, player_y)
            canvas.create_rectangle(
                player_box[0] - min_x, player_box[1] - min_y,
                player_box[2] - min_x, player_box[3] - min_y,
                outline='blue',
                fill='',
                width=2,
                tags="debug_overlay"
            )

            # Get and draw vector to nearest mine
            vector = self.current_level_data.get_vector_to_nearest_mine(
                player_x, player_y)
            if vector:
                # Convert vector coordinates to canvas coordinates
                canvas.create_line(
                    vector[0] - min_x, vector[1] - min_y,
                    vector[2] - min_x, vector[3] - min_y,
                    fill='yellow',
                    width=2,
                    tags="debug_overlay",
                    arrow='last'  # Add an arrow to show direction
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
        # Update game state
        self.game.state_manager.take_action(
            self.game.game_value_fetcher,
            self.game.controller
        )

        # Update state label
        self.ui_components['state_label'].config(
            text=f"State: {self.game.state_manager.state}")

        self._update_frame_display()
        self._update_game_controls()
        self._update_memory_addresses()
        self._update_training_session_info()
        self._update_action_icons()

        # Update rewards plot
        self._update_rewards_plot()

    def _start_update_loop(self):
        """Start the main update loop."""
        self.game.loop()
        self.update_display()
        self.root.after(self.UPDATE_INTERVAL, self._start_update_loop)

    def _setup_training_session_info(self):
        """Set up the training session information panel."""
        # Create right panel for training session info
        self.right_panel = ttk.LabelFrame(
            self.ui_components['right_frame'], text="Current Training Session")
        self.right_panel.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        # Episode counter
        ttk.Label(self.right_panel, text="Episodes:").pack(
            anchor='w', padx=5, pady=2)
        self.episode_count_var = tk.StringVar(value="0")
        ttk.Label(self.right_panel, textvariable=self.episode_count_var).pack(
            anchor='w', padx=5, pady=2)

        # Current episode reward
        ttk.Label(self.right_panel, text="Current Episode Reward:").pack(
            anchor='w', padx=5, pady=2)
        self.episode_reward_var = tk.StringVar(value="0.0")
        ttk.Label(self.right_panel, textvariable=self.episode_reward_var).pack(
            anchor='w', padx=5, pady=2)

        # Best reward
        ttk.Label(self.right_panel, text="Best Reward:").pack(
            anchor='w', padx=5, pady=2)
        self.best_reward_var = tk.StringVar(value="0.0")
        ttk.Label(self.right_panel, textvariable=self.best_reward_var).pack(
            anchor='w', padx=5, pady=2)

        # Training status
        ttk.Label(self.right_panel, text="Training Status:").pack(
            anchor='w', padx=5, pady=2)
        self.training_status_var = tk.StringVar(value="Not training")
        ttk.Label(self.right_panel, textvariable=self.training_status_var).pack(
            anchor='w', padx=5, pady=2)

        # Add reward history graph
        self._setup_reward_graph()

    def _setup_reward_graph(self):
        """Set up the reward history graph panel."""
        # Create a frame for the graph
        graph_frame = ttk.LabelFrame(
            self.ui_components['right_frame'], text="Reward History")
        graph_frame.grid(row=6, column=0, sticky="nsew", pady=(0, 10))

        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def _update_reward_graph(self):
        """Update the reward history graph."""
        if not training_session.reward_history:
            return

        self.ax.clear()
        self.ax.plot(training_session.reward_history, 'b-')
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.canvas.draw()

    def _update_training_session_info(self):
        """Update the training session information display."""
        self.episode_count_var.set(str(training_session.episode_count))
        self.episode_reward_var.set(
            f"{training_session.current_episode_reward:.2f}")
        self.best_reward_var.set(
            f"{training_session.best_reward:.2f}")
        self.training_status_var.set(
            "Training" if training_session.is_training else "Not training")
        self._update_reward_graph()

    def _draw_paths_on_image(self, image: Image.Image) -> Image.Image:
        """Draw the training paths on the game image.

        Args:
            image: The game image to draw on

        Returns:
            The image with paths drawn on it
        """
        if not training_session.is_training or not self.current_level_data:
            print("Not training or no level data")
            return image

        # Create a transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Lets offset our paths to represent the center of the player
        path_offset_x = 3
        path_offset_y = 15

        # Draw past episode paths with decreasing opacity
        past_paths = training_session.episode_paths
        for i, path in enumerate(past_paths):
            # Skip paths with less than 2 points (need at least 2 for a line)
            if not path or len(path) < 2:
                continue

            # Calculate opacity (100% down to 30% for past paths)
            # Most recent path: 100% opacity
            # Oldest path: 30% opacity
            opacity = int(
                255 * (1.0 - (0.7 * (i / (len(past_paths) - 1) if len(past_paths) > 1 else 0))))
            # Ensure minimum 30% opacity
            opacity = max(opacity, int(255 * 0.3))

            # Convert coordinates to canvas space
            scaled_path = []
            for x, y in path:
                level_x, level_y = self.current_level_data.convert_game_to_level_coordinates(
                    x, y)
                canvas_x = level_x + path_offset_x
                canvas_y = level_y + path_offset_y
                scaled_path.append((canvas_x, canvas_y))

            # Draw path as a series of connected lines
            for j in range(len(scaled_path) - 1):
                draw.line(
                    [scaled_path[j], scaled_path[j + 1]],
                    # Darker blue with varying opacity
                    # RGB(51,102,204) is a darker blue
                    fill=(51, 102, 204, opacity),
                    width=2
                )

        # Draw current episode path
        current_path = training_session.current_episode_positions
        # Skip if path is empty or has less than 2 points
        if current_path and len(current_path) >= 2:
            # Convert coordinates to canvas space
            scaled_path = []
            for x, y in current_path:
                level_x, level_y = self.current_level_data.convert_game_to_level_coordinates(
                    x, y)
                canvas_x = level_x + path_offset_x
                canvas_y = level_y + path_offset_y
                scaled_path.append((canvas_x, canvas_y))

            # Draw current path with gradually decreasing opacity
            for j in range(len(scaled_path) - 1):
                # Calculate opacity based on position in path (older segments are more transparent)
                opacity = 255  # Default to full opacity
                if len(scaled_path) > 2:
                    # Only calculate progress if we have more than 2 points
                    progress = (len(scaled_path) - 2 - j) / \
                        (len(scaled_path) - 2)
                    # 255 -> 51 (20% opacity)
                    opacity = int(255 * (1.0 - (0.8 * progress)))

                draw.line(
                    [scaled_path[j], scaled_path[j + 1]],
                    # Bright green with varying opacity
                    fill=(0, 255, 0, opacity),
                    width=2
                )

        # Composite the overlay onto the original image
        return Image.alpha_composite(image.convert('RGBA'), overlay)

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
                self.game_image = Image.fromarray(cropped_frame)
            else:
                # Show full frame if no level data
                self.game_image = Image.fromarray(full_frame)

            # Update the display with the new frame
            self._update_game_display()

    def _update_game_display(self):
        """Update the game display with the current frame and overlay."""
        if not self.game_image:
            return

        image = self.game_image

        # Draw paths if training
        if self.training.get():
            image = self._draw_paths_on_image(image)

        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image)
        canvas = self.ui_components['game_canvas']

        if canvas:
            # Clear previous frame and debug overlay
            canvas.delete("all")
            # Draw new frame
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference

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

    def _update_memory_addresses(self):
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

    def _update_action_icons(self):
        """Update the action icons based on current action."""
        if not training_session.is_training or not training_session.current_episode_actions:
            # Reset all icons to black
            for icon in ['left_icon', 'space_icon', 'right_icon']:
                self.ui_components[icon].configure(foreground='black')
            return

        # Get the most recent action
        current_action = training_session.current_episode_actions[-1]

        # Reset all icons to black first
        for icon in ['left_icon', 'space_icon', 'right_icon']:
            self.ui_components[icon].configure(foreground='black')

        # Highlight the appropriate icons based on the action
        # 0: NOOP, 1: Left, 2: Right, 3: Jump, 4: Jump + Left, 5: Jump + Right
        if current_action in [1, 4]:  # Left or Jump + Left
            self.ui_components['left_icon'].configure(foreground='red')
        if current_action in [3, 4, 5]:  # Jump, Jump + Left, or Jump + Right
            self.ui_components['space_icon'].configure(foreground='red')
        if current_action in [2, 5]:  # Right or Jump + Right
            self.ui_components['right_icon'].configure(foreground='red')

    def _update_rewards_plot(self):
        """Update the rewards plot with current and historical data."""
        self.rewards_ax.clear()

        # Get current episode data
        current_episode = training_session.episode_count
        current_rewards = training_session.current_episode_step_rewards

        # Create lists to store plot data
        episodes_to_plot = []
        rewards_to_plot = []
        labels_to_plot = []

        # Add historical episodes
        if training_session.historical_step_rewards:
            # Get last 4 completed episodes
            for i, episode_rewards in enumerate(training_session.historical_step_rewards[-4:]):
                if episode_rewards:  # Only add if we have rewards
                    episode_num = current_episode - \
                        (len(training_session.historical_step_rewards) - i)
                    episodes_to_plot.append(episode_rewards)
                    labels_to_plot.append(f'Episode {episode_num}')
                    rewards_to_plot.append(episode_rewards)

        # Add current episode if it has data
        if current_rewards:
            episodes_to_plot.append(current_rewards)
            labels_to_plot.append(f'Episode {current_episode}')
            rewards_to_plot.append(current_rewards)

        # Plot each episode
        for i, (rewards, label) in enumerate(zip(rewards_to_plot, labels_to_plot)):
            color_idx = i % len(self.REWARD_COLORS)
            steps = list(range(len(rewards)))

            # Current episode gets special formatting
            if label == f'Episode {current_episode}':
                self.rewards_ax.plot(steps, rewards,
                                     label=label,
                                     color=self.REWARD_COLORS[color_idx],
                                     linewidth=2,
                                     zorder=10)  # Ensure current episode is on top
            else:
                self.rewards_ax.plot(steps, rewards,
                                     label=label,
                                     color=self.REWARD_COLORS[color_idx],
                                     alpha=0.7,
                                     linewidth=1)

        # Configure plot appearance
        self.rewards_ax.set_title('Rewards per Step')
        self.rewards_ax.set_xlabel('Step')
        self.rewards_ax.set_ylabel('Reward')

        # Add grid and make it less prominent
        self.rewards_ax.grid(True, linestyle='--', alpha=0.3)

        # Only show legend if we have data to display
        if episodes_to_plot:
            # Move legend outside to prevent overlap with plot
            self.rewards_ax.legend(bbox_to_anchor=(1.05, 1),
                                   loc='upper left',
                                   borderaxespad=0.)

            # Set reasonable y-axis limits
            all_rewards = [r for episode in rewards_to_plot for r in episode]
            if all_rewards:
                min_reward = min(all_rewards)
                max_reward = max(all_rewards)
                padding = (max_reward - min_reward) * 0.1  # 10% padding
                self.rewards_ax.set_ylim(
                    min_reward - padding, max_reward + padding)

        # Adjust layout to prevent label cutoff
        self.rewards_fig.tight_layout()

        try:
            self.rewards_canvas.draw()
        except Exception as e:
            print(f"Error updating rewards plot: {e}")
