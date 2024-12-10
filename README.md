# NPP-RL

A Deep Reinforcement Learning Agent for the game N++, implementing a PPO (Proximal Policy Optimization) algorithm with curriculum learning, pathfinding, spatial memory, and a multi-layered reward system.

## Project Overview

NPP-RL is an advanced reinforcement learning implementation that trains an AI agent to master the challenging platformer game N++. The project utilizes state-of-the-art techniques in deep reinforcement learning, combining visual processing, curriculum learning, and multi-faceted reward systems.

## Core Features

### Advanced Reward System

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

- **Environment**: Custom N++ environment implementation with sophisticated state management
- **Agent**: PPO-based agent with custom feature extraction
- **Reward Calculation**: Modular reward system with multiple specialized calculators
- **Training**: Comprehensive training session management and monitoring
- **Observation Processing**: Advanced visual and state processing pipeline

The implementation features curriculum learning that automatically adjusts reward scales based on the agent's demonstrated competencies, with skill acquisition and progression.