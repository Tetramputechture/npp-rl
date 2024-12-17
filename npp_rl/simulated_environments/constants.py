# Features: recent_visits, visit_frequency, area_exploration, transitions
NUM_NUMERICAL_FEATURES = 4

# Number of historical frames at fixed intervals (2, 4, 8)
NUM_HISTORICAL_FRAMES = 3

# Number of goal channels (switch and exit door heatmaps)
NUM_GOAL_CHANNELS = 2

# Number of player state channels (position x and y, velocity x and y, in_air status, and walled status)
NUM_PLAYER_STATE_CHANNELS = 6

# Total observation channels
TOTAL_OBSERVATION_CHANNELS = 4 + NUM_HISTORICAL_FRAMES + \
    NUM_NUMERICAL_FEATURES + NUM_PLAYER_STATE_CHANNELS + NUM_GOAL_CHANNELS

# Observation image size
OBSERVATION_IMAGE_SIZE = 84
