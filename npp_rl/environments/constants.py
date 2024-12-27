# Observation image size
OBSERVATION_IMAGE_WIDTH = 160
OBSERVATION_IMAGE_HEIGHT = 120

# Player frame size
PLAYER_FRAME_WIDTH = 120
PLAYER_FRAME_HEIGHT = 120

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
TEMPORAL_FRAMES = 3

# Max time in frames per level
MAX_TIME_IN_FRAMES = 5000

# Game state features
# Ninja State (12 values):
# Position X
# Position Y
# Speed X
# Speed Y
# Airborn
# Walled
# Jump duration
# Facing
# Tilt angle
# Applied gravity
# Applied drag
# Applied friction

# Entity States:
# 160004 values
#
# Geometry State (Fixed Size):
# Tile data: 44 × 25 = 1,100 values
# Horizontal grid edges: 88 × 51 = 4,488 values
# Vertical grid edges: 89 × 50 = 4,450 values
# Horizontal segments: 88 × 51 = 4,488 values
# Vertical segments: 89 × 50 = 4,450 values
#
# Time remaining state: 1 value
#
# Total state vector size calculation:
# Ninja state: 13 values
# Entity states: 28 types × (1 count + 10 entities × 8 attributes) = 2,268 values
# Geometry state: 1,100 + 4,488 + 4,450 + 4,488 + 4,450 = 18,976 values
# Total: 12 + 160004 + 18,976 + 1 = 177993 values
GAME_STATE_FEATURES = 177993

# Without entities besides our exit and switch,
# We have 258 features for our entities
# Total: 12 + 258 + 18,976 + 1 = 19247
GAME_STATE_FEATURES_ONLY_EXIT_AND_SWITCH = 19247

# Only ninja and 1 exit and 1 switch, no tile data (to be used in conjunction with frame stacking) = 24
# + 1 time remaining
GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH = 25

TILE_SIZE = 24
TILE_COUNT_X = 42
TILE_COUNT_Y = 23
