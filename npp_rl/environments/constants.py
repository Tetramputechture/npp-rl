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

# Features: recent_visits, visit_frequency, area_exploration, transitions, time_remaining
NUM_NUMERICAL_FEATURES = 5

# Number of historical frames at fixed intervals (2, 4, 8)
NUM_HISTORICAL_FRAMES = 3
