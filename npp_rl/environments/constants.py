"""Constants for the game environment."""

# Speed factor for game time
GAME_SPEED_FACTOR = 0.5

# Default game speed in frames per second
GAME_DEFAULT_SPEED_FRAMES_PER_SECOND = 60.0

# Game speed after speed increase
GAME_SPEED_FRAMES_PER_SECOND = GAME_DEFAULT_SPEED_FRAMES_PER_SECOND * GAME_SPEED_FACTOR

# We take our observations at the game speed * a factor
# this way our observations are more accurate
GAME_SPEED_TIMESTEP_FACTOR = 20

# The time that should be taken between action and observation
TIMESTEP = 1/(GAME_SPEED_FRAMES_PER_SECOND * GAME_SPEED_TIMESTEP_FACTOR)

# 11 numerical features + 4 grid memory channels + 3 mine features (distance, angle, relative velocity)
NUM_NUMERICAL_FEATURES = 18
