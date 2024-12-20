# Temporal frames (4 frames spaced 4 frames apart)
NUM_TEMPORAL_FRAMES = 4

# Features: recent_visits, visit_frequency, area_exploration, transitions
NUM_NUMERICAL_FEATURES = 4

# Number of goal channels (switch and exit door heatmaps)
NUM_GOAL_CHANNELS = 2

# Number of player state channels (position x and y, velocity x and y, in_air status, and walled status)
NUM_PLAYER_STATE_CHANNELS = 6

# Total observation channels
TOTAL_OBSERVATION_CHANNELS = NUM_TEMPORAL_FRAMES + \
    NUM_NUMERICAL_FEATURES + NUM_PLAYER_STATE_CHANNELS + NUM_GOAL_CHANNELS

# Observation image size
OBSERVATION_IMAGE_SIZE = 120

# Frame intervals for temporal stacking
FRAME_INTERVALS = [0, 4, 8, 12]

# Frame stack count
FRAME_STACK_COUNT = 4

# Level width and height for normalization
LEVEL_WIDTH = 1032.0
LEVEL_HEIGHT = 576.0

# Maximum velocity for normalization
MAX_VELOCITY = 20.0

# Observation space constants
TEMPORAL_FRAMES = 4
MEMORY_CHANNELS = 4
PATHFINDING_CHANNELS = 3  # Distance field, path visualization, clearance map
COLLISION_CHANNELS = 2    # Static obstacles, dynamic obstacles
TOTAL_OBSERVATION_CHANNELS = TEMPORAL_FRAMES + \
    MEMORY_CHANNELS + PATHFINDING_CHANNELS + COLLISION_CHANNELS
