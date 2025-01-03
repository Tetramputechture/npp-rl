# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing Recurrent PPO (Proximal Policy Optimization) via Stable Baselines 3 with a custom environment using the nclone simulator.

## Project Overview

This project aims to train an agent to play [N++](https://en.wikipedia.org/wiki/N%2B%2B). The game features a physically simulated movement model where the player can move continuously in any direction within a grid-based level. The agent must learn to navigate the environment, avoid hazards, collect gold, activate switches to open doors, and reach the exit.

## Environment

The environment uses a custom fork of community-built simulator ([nclone](https://github.com/Tetramputechture/nclone)) rather than controlling the actual game process. This allows for faster training and headless operation.

### Observation Space

The observation space consists of three components:

1. **Player Frame** - A localized view centered on the player
   - Dimensions: PLAYER_FRAME_HEIGHT x PLAYER_FRAME_WIDTH x 3 (stacked grayscale frames)
   - Provides detailed information about the immediate surroundings
   - Frame stacking:
     - Current frame (most recent)
     - Last frame (1 frame ago)
     - Second to last frame (2 frames ago)
   - Each frame is preprocessed:
     - Converted to grayscale
     - Centered on player position
     - Cropped to focus on local area
     - Normalized to [0, 255] range

2. **Base Frame** - A global view of the entire level
   - Dimensions: OBSERVATION_IMAGE_HEIGHT x OBSERVATION_IMAGE_WIDTH x 1 (grayscale)
   - Gives context about the overall level layout

3. **Game State** - A vector containing:
   - Ninja state:
     - Position X
     - Position Y
     - Speed X
     - Speed Y
     - Airborn
     - Walled
     - Jump duration
     - Facing
     - Tilt angle
     - Applied gravity
     - Applied drag
     - Applied friction
   - Exit and switch entity states
   - Vectors between ninja and objectives (switch/exit)
   - Time remaining

### Action Space

The agent can perform 6 discrete actions:

- NOOP (No action)
- Left
- Right
- Jump
- Jump + Left
- Jump + Right

### Training Architecture

The implementation uses RecurrentPPO from Stable-Baselines3-Contrib with the following key components:

1. **Policy**: MultiInputLstmPolicy
   - Processes multiple input types (frames and state vectors)
   - Uses LSTM for temporal dependencies

2. **Hyperparameter Optimization**
   - Utilizes Optuna for automated hyperparameter tuning
   - Optimizes key parameters including:
     - Learning rate
     - Network architecture
     - LSTM hidden size
     - Batch size
     - GAE parameters
     - PPO clip range

3. **Training Infrastructure**
   - Vectorized environment support
   - Tensorboard integration for monitoring
   - Checkpointing and model saving
   - Video recording of agent performance

### Reward System

The reward system includes:

- Time-based penalties
- Navigation rewards
- Switch activation bonuses
- Terminal rewards for level completion
- Death penalties

### Hyperparameter Tuning

The project includes automated hyperparameter optimization using Optuna. To run the tuning process:

```bash
python recurrent_ppo_tune.py
```

The tuning process:

- Runs 100 trials using Optuna's TPE sampler
- Uses median pruning to stop underperforming trials early
- Runs for up to 24 hours on a 2x NVIDIA H100 instance
- Optimizes key hyperparameters including:
  - Learning rate and schedule
  - Network architecture (tiny vs small)
  - LSTM hidden size (128 to 512)
  - Batch size (32 to 512)
  - N-steps (256 to 4096)
  - GAE lambda and gamma
  - PPO clip ranges
  - Entropy and value function coefficients

Results are saved in:

- `training_logs/tune_logs/` - Individual trial logs and Tensorboard data
- `training_logs/tune_results_<timestamp>/` - Final optimization results

## Dependencies

Cairo:

```sh
sudo apt install libcairo2-dev pkg-config python3-dev
```

Required packages:

- numpy>=1.21.0
- torch>=2.0.0
- opencv-python>=4.8.0
- pillow>=10.0.0
- gymnasium>=0.29.0
- sb3-contrib>=2.0.0
- stable-baselines3>=2.1.0
- optuna>=3.3.0
- tensorboard>=2.14.0
- imageio>=2.31.0
