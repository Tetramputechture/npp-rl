# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing PPO (Proximal Policy Optimization) via Stable Baselines 3 with an environment that has curriculum learning, pathfinding, spatial memory, and a multi-layered reward system. 

## Project Overview

This project's goal is to train an agent to master the platformer game N++. The game has a physically simulated movement model, and the agent must learn to navigate the environment, collect gold, and avoid hazards.

## Core Features

### Environment

The environment is planned to be a custom implementation of the game N++. It will be built using the `gymnasium` library, and will be able to run on CPU or GPU.

Right now, the environment directly launches and controls the original N++ game process, but in the future it will be replaced with a custom implementation.
This is because the original game process cannot be run in headless mode at a faster rate than 60 FPS, which is too slow for training.

Our environment is a level from the game, and includes (at a minimum) the following features:

- A player that can move around the level by moving left, right, and jumping.
- A switch that can be activated to open a door.
- A goal that the player must reach to complete the level.

The environment may also include:

- Hazards that kill the player if they touch them.
- Gold that the player can collect.

An episode ends when the player reaches the goal, or dies.

The player automatically dies if a timer runs out, which starts at 90, but is increased
by 1 for each gold collected.

# Observation Space

Our observation space is currently:

- 4 stacked frames of 84x84 grayscale.
- 3 historical frames of 84x84 grayscale.
- 4 numerical features for pathfinding:
   - recent_visits
   - visit_frequency
   - area_exploration
   - transitions
- 1 hazard channel
   - mine locations with gaussian falloff
- 2 goal channels
   - switch and exit door heatmaps

### Reward System

Our reward system is currently broken down into 4 main components:

1. **Curriculum-Based Learning**
   - Progressive skill development through three main stages:
     - Movement Mastery: Basic control and precision
     - Navigation: Efficient path-finding and objective targeting
     - Optimization: Speed and perfect execution
   - Dynamic reward scaling based on demonstrated competence
   - Automatic progression tracking and adjustment

2. **Movement Rewards**
   - Precision-based rewards for controlled movement
   - Landing quality assessment
   - Momentum control evaluation
   - Skill-based reward scaling
   - Progressive mastery tracking

3. **Navigation System**
   - Temporal Difference Learning for path optimization
   - Progressive scaling based on consecutive improvements
   - Momentum bonuses for consistent progress
   - Mine avoidance mechanics
   - Switch activation and exit navigation rewards

4. **Exploration Mechanics**
   - Area-based exploration rewards
   - Progressive backtracking penalties
   - Local minima detection and avoidance
   - Transition point tracking
   - Dynamic visit counting

Each reward is calculated after the game state has been updated from the 
most recent action.

### Observation Processing

When an observation is received, it is processed into a 4 frames of 84x84 grayscale, extended with channels for numerical feature data.

### Observation Feature Extraction

1. **Frame Processing**
   - Processes 4 stacked frames through a CNN with residual connections
   - Uses batch normalization and adaptive pooling for stable training
   - Features multiple residual blocks for improved feature extraction
   - Outputs 128-dimensional frame features

2. **Memory Processing**
   - Processes 4 numerical memory features (visits, frequency, exploration, transitions)
   - Uses layer normalization and dropout for regularization
   - Compresses to 32-dimensional memory representation

3. **Hazard Processing**
   - Dedicated CNN branch for processing mine hazard channels
   - Uses spatial convolutions with batch normalization
   - Outputs 32-dimensional hazard awareness features

4. **Goal Processing**
   - Specialized branch for switch and exit door heatmaps
   - Enhanced feature capacity for critical navigation information
   - Produces 64-dimensional goal-oriented features

All features are integrated through a final network that combines spatial, memory, hazard, and goal information into a 256-dimensional policy-ready representation.

### Training Architecture

- PPO (Proximal Policy Optimization) implementation with dynamic entropy adjustment, provided by Stable Baselines 3
- Performance tracking and metrics logging

### Performance Monitoring
 
We use a custom training session management and monitoring with TKinter GUI, and
Tensorboard can be enabled to monitor training in real time, though this does
not yet work since we do not have a working simulation environment.
