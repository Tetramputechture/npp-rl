"""Visualization callback for real-time training monitoring.

This callback renders the training environment at regular intervals,
allowing visualization of agent behavior during training.
"""

import logging
import time
from typing import Optional

import pygame
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TrainingVisualizationCallback(BaseCallback):
    """Callback for visualizing agent behavior during training.

    This callback renders one or more environments at regular intervals,
    displaying what the agent is doing in real-time.

    Features:
    - Renders at configurable intervals (by timesteps or episodes)
    - Can visualize a single environment or cycle through multiple
    - Handles pygame events to prevent window freezing
    - Optional FPS limiting to control rendering speed
    - Can pause/resume with spacebar
    """

    def __init__(
        self,
        render_freq: int = 100,
        render_mode: str = "timesteps",
        env_idx: int = 0,
        target_fps: int = 60,
        window_title: str = "NPP-RL Training Visualization",
        verbose: int = 0,
    ):
        """Initialize visualization callback.

        Args:
            render_freq: How often to render (in timesteps or episodes)
            render_mode: "timesteps" or "episodes" - what to count for rendering
            env_idx: Which environment to visualize (from vectorized env)
            target_fps: Target frames per second for rendering (0 = unlimited)
            window_title: Title for the pygame window
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
        """
        super().__init__(verbose)

        self.render_freq = render_freq
        self.render_mode = render_mode
        self.env_idx = env_idx
        self.target_fps = target_fps
        self.window_title = window_title

        # Internal state
        self.timestep_count = 0
        self.episode_count = 0
        self.last_render_time = 0
        self.paused = False

        # Pygame initialization
        self.pygame_initialized = False
        self.clock = None

        # Track if we've shown the initial environment
        self.first_render_done = False

    def _init_pygame(self):
        """Initialize pygame display if not already initialized."""
        if not self.pygame_initialized:
            if not pygame.get_init():
                pygame.init()

            if not pygame.display.get_surface():
                # The environment's renderer will create the actual display
                # We just need to ensure pygame is initialized
                pass

            if self.window_title:
                pygame.display.set_caption(self.window_title)

            if self.target_fps > 0:
                self.clock = pygame.time.Clock()

            self.pygame_initialized = True
            logger.info(
                f"Visualization callback initialized (render every {self.render_freq} {self.render_mode})"
            )

    def _handle_pygame_events(self):
        """Handle pygame events to prevent window freezing.

        Controls:
        - SPACE: Pause/unpause rendering
        - ESC/Q: Close visualization window (continues training)
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Visualization window closed by user")
                pygame.quit()
                self.pygame_initialized = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    status = "paused" if self.paused else "resumed"
                    logger.info(f"Visualization {status}")
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    logger.info("Visualization disabled by user")
                    pygame.quit()
                    self.pygame_initialized = False
                    return False
        return True

    def _should_render(self) -> bool:
        """Check if we should render this step."""
        if self.render_mode == "timesteps":
            return self.timestep_count % self.render_freq == 0
        elif self.render_mode == "episodes":
            # Render when we've completed the required number of episodes
            return self.episode_count % self.render_freq == 0
        return False

    def _render_environment(self):
        """Render the current environment state."""
        if not self.pygame_initialized:
            self._init_pygame()

        if self.paused:
            # Still handle events when paused
            self._handle_pygame_events()
            return

        try:
            # Get the vectorized environment
            vec_env = self.training_env

            # Handle curriculum wrapper
            while hasattr(vec_env, "venv"):
                vec_env = vec_env.venv

            # Try to get the specific environment
            if hasattr(vec_env, "envs"):
                # SubprocVecEnv or DummyVecEnv
                if self.env_idx < len(vec_env.envs):
                    env = vec_env.envs[self.env_idx]

                    # Unwrap curriculum wrapper if present
                    while hasattr(env, "env") and not hasattr(env, "render"):
                        env = env.env

                    # Render the environment
                    if hasattr(env, "render"):
                        env.render()

                        # Handle pygame events
                        if not self._handle_pygame_events():
                            return

                        # Limit FPS if requested
                        if self.clock and self.target_fps > 0:
                            self.clock.tick(self.target_fps)
                    else:
                        logger.warning(
                            f"Environment {self.env_idx} does not have a render method"
                        )
            else:
                logger.warning(
                    "Cannot access individual environments for visualization"
                )

        except Exception as e:
            logger.debug(f"Error during visualization render: {e}")

    def _on_step(self) -> bool:
        """Called at each environment step during training.

        Returns:
            True to continue training, False to stop
        """
        self.timestep_count += 1

        # Check for episode completion
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if dones[self.env_idx]:
                self.episode_count += 1

        # Render if it's time
        if self._should_render() or not self.first_render_done:
            self._render_environment()
            self.first_render_done = True

        return True

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        logger.info("Starting training visualization")
        logger.info(
            f"Rendering environment {self.env_idx} every {self.render_freq} {self.render_mode}"
        )
        logger.info("Controls: SPACE=pause/resume, ESC=disable visualization")

        # Do an initial render to show the environment
        self._init_pygame()
        self._render_environment()

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        logger.info("Training visualization ended")
        if self.pygame_initialized:
            # Keep the window open briefly to show final state
            time.sleep(0.5)
            # Don't close pygame here - let user close it manually
