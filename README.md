# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing both PPO and Recurrent PPO (Proximal Policy Optimization) via Stable Baselines 3 with a simulated game environment.

## Project Overview

This project aims to train an agent to play [N++](https://en.wikipedia.org/wiki/N%2B%2B). The game features a physically simulated movement model where the player can move continuously in any direction within a grid-based level. The agent must learn to navigate the environment, avoid hazards, collect gold, activate switches to open doors, and reach the exit.

The project supports both standard PPO and Recurrent PPO architectures, with optional frame stacking in the environment. We have found success training on simple levels using just a single frame plus our game state vector, suggesting that frame stacking or recurrent architectures may only be necessary for longer or more complex levels requiring temporal reasoning.

## Example Agent Level Completion

This is an example of a trained agent completing a non-trivial level.

![Example Level Completion](example_completion.gif)

This agent was trained on this single level with no frame stacking or LSTM on 4 million frames, and achieved a non-zero success rate at around 2 million frames.

Work on a generalized agent to play through any level is ongoing.

## Environment

The environment uses a custom fork of community-built simulator ([nclone](https://github.com/Tetramputechture/nclone)) rather than controlling the actual game process. This allows for faster training and headless operation. This fork also
includes our Gym environment, reward calculation, and frame augmentation.

### Observation Space

The observation space consists of two components:

1. **Player Frame** - A localized view centered on the player
   - Dimensions: 84 x 84 x (1, or 3 with frame stacking)
   - Provides detailed information about the immediate surroundings
   - If frame stacking is enabled:
     - Current frame (most recent)
     - Last, second to last, and third to last frame
   - Each frame is preprocessed:
     - Converted to grayscale
     - Centered on player position
     - Cropped to focus on local area
     - Normalized to [0, 255] range

2. **Game State** - A vector containing:
   - Ninja state:
     - Position X
     - Position Y
     - Speed X
     - Speed Y
     - Airborn
     - Walled
     - Jump duration
     - Applied gravity
     - Applied drag
     - Applied friction
   - Exit and switch entity states
   - Vectors between ninja and objectives (switch/exit)
   - Time remaining

### Observation Augmentations

We apply random cutout augmentations to the player frame, with a 50% chance of applying the cutout. Cutout has shown to be effective at improving generalization in other domains, and we hypothesize that it may be useful for improving generalization in this domain as well. See more details [in this paper](https://arxiv.org/abs/2004.14990).

### Action Space

The agent can perform 6 discrete actions:

- NOOP (No action)
- Left
- Right
- Jump
- Jump + Left
- Jump + Right

### Training Architecture

The implementation supports both PPO and RecurrentPPO from Stable-Baselines3 and Stable-Baselines3-Contrib with the following components:

1. **Policies**:
   - PPO: MultiInputPolicy
   - RecurrentPPO: MultiInputLstmPolicy
   - Both process multiple input types (frames and state vectors)
   - LSTM variant adds temporal dependencies for more complex tasks or longer levels.

2. **Hyperparameter Optimization**
   - Utilizes Optuna for automated hyperparameter tuning
   - Optimizes key parameters including:
     - Learning rate
     - Network architecture
     - LSTM hidden size (for RecurrentPPO)
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

The project includes automated hyperparameter optimization using Optuna. To run the tuning process for either architecture:

```bash
# For standard PPO
python ppo_tune.py

# For RecurrentPPO
python recurrent_ppo_tune.py
```

The tuning process:

- Runs 100 trials using Optuna's TPE sampler
- Uses median pruning to stop underperforming trials early
- Runs for up to 24 hours on a 1-2x NVIDIA H100 instance
- Optimizes key hyperparameters including:
  - Learning rate and schedule
  - Network architecture (tiny vs small)
  - LSTM hidden size (128 to 512, RecurrentPPO only)
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
- meson>=1.6.1
