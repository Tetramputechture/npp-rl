# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing Recurrent PPO (Proximal Policy Optimization) via Stable Baselines 3 with a custom environment using the nclone simulator.

## Project Overview

This project aims to train an agent to play [N++](https://en.wikipedia.org/wiki/N%2B%2B). The game features a physically simulated movement model where the player can move continuously in any direction within a grid-based level. The agent must learn to navigate the environment, avoid hazards, collect gold, activate switches to open doors, and reach the exit.

## Environment

The environment uses a community-built simulator ([nclone](https://github.com/SimonV42/nclone)) rather than controlling the actual game process. This allows for faster training and headless operation.

### Observation Space

The observation space consists of three components:

1. **Player Frame** - A localized view centered on the player
   - Dimensions: PLAYER_FRAME_HEIGHT x PLAYER_FRAME_WIDTH x 1 (grayscale)
   - Provides detailed information about the immediate surroundings

2. **Base Frame** - A global view of the entire level
   - Dimensions: OBSERVATION_IMAGE_HEIGHT x OBSERVATION_IMAGE_WIDTH x 1 (grayscale)
   - Gives context about the overall level layout

3. **Game State** - A vector containing:
   - Ninja state (position, speed, airborne status, etc.)
   - Exit and switch entity states
   - Time remaining
   - Vectors between ninja and objectives (switch/exit)

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

## Dependencies

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
