"""Constants for the game environment."""

# Speed factor for game time
GAME_SPEED_FACTOR = 100

# Default game speed in frames per second
GAME_DEFAULT_SPEED_FRAMES_PER_SECOND = 60.0

# Game speed after speed increase
GAME_SPEED_FRAMES_PER_SECOND = GAME_DEFAULT_SPEED_FRAMES_PER_SECOND * GAME_SPEED_FACTOR

# We take our observations at the game speed * a factor
# this way our observations are more accurate
GAME_SPEED_TIMESTEP_FACTOR = 50

# The time that should be taken between action and observation
TIMESTEP = 1/(GAME_SPEED_FRAMES_PER_SECOND * GAME_SPEED_TIMESTEP_FACTOR)

# Features: recent_visits, visit_frequency, area_exploration, transitions
NUM_NUMERICAL_FEATURES = 4

# Number of historical frames at fixed intervals (2, 4, 8)
NUM_HISTORICAL_FRAMES = 3

# Number of hazard channels (mine locations with gaussian falloff)
NUM_HAZARD_CHANNELS = 1

# Number of goal channels (switch and exit door heatmaps)
NUM_GOAL_CHANNELS = 2

# Total observation channels
TOTAL_OBSERVATION_CHANNELS = 4 + NUM_HISTORICAL_FRAMES + \
    NUM_NUMERICAL_FEATURES + NUM_HAZARD_CHANNELS + NUM_GOAL_CHANNELS

# Observation image size to send to the policy network
OBSERVATION_IMAGE_SIZE = 84
