# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing a PPO (Proximal Policy Optimization) algorithm with curriculum learning, pathfinding, spatial memory, and a multi-layered reward system.

## Project Overview

This project's goal is to train an agent to master the platformer game N++. The game has a physically simulated movement model, and the agent must learn to navigate the environment, collect resources, and avoid obstacles, eventually reaching the end of the level. 

## Core Features

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
   - Advanced mine avoidance mechanics
   - Switch activation and exit navigation rewards

4. **Exploration Mechanics**
   - Area-based exploration rewards
   - Progressive backtracking penalties
   - Local minima detection and avoidance
   - Transition point tracking
   - Dynamic visit counting

### Observation Processing

- Advanced image processing pipeline for game state analysis
- Edge detection for platform and obstacle identification
- Contrast enhancement and normalization
- Spatial memory system for tracking agent movement
- Feature extraction for state representation

### Training Architecture

- PPO (Proximal Policy Optimization) implementation with dynamic entropy adjustment
- Comprehensive callback system for training monitoring
- Progressive difficulty adjustment
- Performance tracking and metrics logging
- Automatic model checkpointing

### Performance Monitoring

- Detailed reward component breakdown
- Success rate tracking across multiple metrics
- Movement efficiency analysis
- Navigation quality assessment
- Exploration effectiveness evaluation

## Technical Implementation

The project is structured into several key components:

- **Environment**: Custom N++ environment implementation with state management. Input control via `pydirectinput`.
- **Agent**: PPO-based agent with custom feature extraction
- **Reward Calculation**: Modular reward system with multiple specialized calculators
- **Training**: Custom training session management and monitoring with TKinter GUI, Tensorboard is supported though.
- **Observation Processing**: We process our state into a 4 frames of 84x84 grayscale, extended with channels for numerical feature data.
