# Task 001: Complete Human Replay Processing System

## Overview
Complete the implementation of the human replay processing system to enable learning from expert demonstrations. This includes fixing placeholder implementations, creating behavioral cloning datasets, and establishing the BC-to-RL transition pipeline.

## Context Reference
See [npp-rl comprehensive technical roadmap](../docs/comprehensive_technical_roadmap.md) Section 3: "Detailed Implementation Roadmap" - Task 1.1: "Human Replay Processing"

## Requirements

### Primary Objectives
1. **Fix placeholder implementations** in replay data ingestion
2. **Implement behavioral cloning trainer** for learning from demonstrations
3. **Create BC-to-RL transition pipeline** for policy initialization
4. **Establish data quality validation** for replay datasets
5. **Add multimodal observation extraction** from replay frames

### Current System Analysis
The existing replay processing system has:
- Basic binary replay parser (`tools/binary_replay_parser.py`)
- Placeholder implementations in `tools/replay_ingest.py` (line 242)
- Data quality reporting (`datasets/quality_report.json`)
- Schema documentation (`datasets/replay_data_schema.md`)

### Components to Implement

#### 1. Complete Replay Data Ingestion
**File**: `tools/replay_ingest.py`

**Current Issues**:
- Placeholder at line 242 in `create_observation_from_replay()`
- Missing multimodal observation extraction
- Incomplete physics state extraction
- No graph representation building

**Required Implementation**:
```python
def create_observation_from_replay(self, frame_data: dict) -> dict:
    """Extract multimodal observations from replay frame."""
    # Extract visual features
    visual_obs = self.extract_visual_features(frame_data)
    
    # Extract physics state
    physics_obs = self.extract_physics_state(frame_data)
    
    # Build graph representation
    graph_obs = self.build_graph_representation(frame_data)
    
    return {
        'player_frame': visual_obs['player_frame'],      # (12, 84, 84) - 12-frame stack
        'global_view': visual_obs['global_view'],        # (176, 100) - full level view
        'physics_state': physics_obs,                    # Physics vector
        'graph_data': graph_obs                          # Graph representation
    }
```

#### 2. Behavioral Cloning Trainer
**New File**: `npp_rl/training/behavioral_cloning.py`

**Required Implementation**:
```python
class BehavioralCloningTrainer:
    def __init__(self, model: HGTMultimodalExtractor, config: dict):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.config = config
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train BC model for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (obs, actions) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = self.model(obs)
            loss = self.criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config['log_interval'] == 0:
                self._log_training_progress(batch_idx, loss.item())
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate BC model performance."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for obs, actions in dataloader:
                predicted_actions = self.model(obs)
                loss = self.criterion(predicted_actions, actions)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted_classes = torch.argmax(predicted_actions, dim=1)
                correct_predictions += (predicted_classes == actions).sum().item()
                total_predictions += actions.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions,
            'total_samples': total_predictions
        }
```

#### 3. BC-to-RL Transition Pipeline
**New File**: `npp_rl/training/bc_to_rl_transition.py`

**Required Implementation**:
```python
class BCToRLTransition:
    def __init__(self, bc_model_path: str, rl_config: dict):
        self.bc_model_path = bc_model_path
        self.rl_config = rl_config
        
    def initialize_rl_policy(self) -> PPO:
        """Initialize PPO policy with BC weights."""
        # Load BC model
        bc_model = torch.load(self.bc_model_path)
        
        # Create PPO agent with same architecture
        ppo_agent = PPO(
            policy=HGTMultimodalExtractor,
            env=self.rl_config['env'],
            **self.rl_config['ppo_params']
        )
        
        # Transfer weights from BC model to PPO policy
        self._transfer_weights(bc_model, ppo_agent.policy)
        
        return ppo_agent
    
    def create_hybrid_loss(self, bc_model, bc_weight=0.1):
        """Create hybrid BC+RL loss function."""
        def hybrid_loss(rl_loss, observations, actions):
            # Standard RL loss
            total_loss = rl_loss
            
            # Add BC regularization
            if bc_weight > 0:
                bc_predictions = bc_model(observations)
                bc_loss = F.cross_entropy(bc_predictions, actions)
                total_loss += bc_weight * bc_loss
            
            return total_loss
        
        return hybrid_loss
```

#### 4. Data Quality Validation
**Enhanced File**: `tools/data_quality.py`

**Required Enhancements**:
```python
class ReplayDataQualityValidator:
    def __init__(self):
        self.quality_metrics = {
            'completion_rate': 0.0,
            'average_episode_length': 0.0,
            'action_distribution': {},
            'physics_consistency': 0.0,
            'visual_quality': 0.0
        }
    
    def validate_replay_dataset(self, replay_files: List[str]) -> dict:
        """Comprehensive validation of replay dataset."""
        results = {
            'total_replays': len(replay_files),
            'valid_replays': 0,
            'quality_scores': [],
            'issues': []
        }
        
        for replay_file in replay_files:
            try:
                quality_score = self._validate_single_replay(replay_file)
                results['quality_scores'].append(quality_score)
                
                if quality_score > 0.7:  # Quality threshold
                    results['valid_replays'] += 1
                else:
                    results['issues'].append(f"Low quality: {replay_file}")
                    
            except Exception as e:
                results['issues'].append(f"Parse error in {replay_file}: {str(e)}")
        
        results['average_quality'] = np.mean(results['quality_scores'])
        return results
    
    def _validate_single_replay(self, replay_file: str) -> float:
        """Validate individual replay file."""
        replay_data = self._load_replay(replay_file)
        
        # Check completion
        completion_score = 1.0 if replay_data['completed'] else 0.3
        
        # Check episode length (not too short)
        length_score = min(1.0, len(replay_data['frames']) / 1000)  # Normalize by expected length
        
        # Check action diversity
        actions = [frame['action'] for frame in replay_data['frames']]
        action_diversity = len(set(actions)) / 6.0  # 6 possible actions
        
        # Check physics consistency
        physics_score = self._validate_physics_consistency(replay_data)
        
        # Weighted average
        quality_score = (
            0.4 * completion_score +
            0.2 * length_score +
            0.2 * action_diversity +
            0.2 * physics_score
        )
        
        return quality_score
```

## Acceptance Criteria

### Functional Requirements
1. **Complete Data Pipeline**: Replay files can be processed into training datasets
2. **Multimodal Observations**: All observation modalities are correctly extracted
3. **BC Training**: Behavioral cloning trainer successfully learns from demonstrations
4. **Policy Transfer**: BC weights can be transferred to RL policies
5. **Quality Validation**: Data quality metrics are computed and reported

### Technical Requirements
1. **Performance**: Process >1000 frames per second
2. **Memory Efficiency**: Handle large replay datasets without memory issues
3. **Data Format**: Compatible with existing HGT multimodal architecture
4. **Error Handling**: Graceful handling of corrupted or invalid replay files

### Quality Requirements
1. **Cross-Modal Consistency**: Visual and physics observations are consistent
2. **Action Accuracy**: Extracted actions match original replay actions
3. **Temporal Consistency**: Frame sequences maintain proper temporal order
4. **Data Quality**: >80% of processed replays meet quality thresholds

## Test Scenarios

### Unit Tests
**File**: `tests/test_replay_processing.py`

```python
class TestReplayProcessing(unittest.TestCase):
    def test_replay_data_ingestion(self):
        """Test replay file parsing and data extraction."""
        processor = ReplayDataProcessor()
        replay_data = processor.parse_replay_file("test_data/simple_level_completion.replay")
        
        # Test basic structure
        self.assertIsNotNone(replay_data)
        self.assertIn('frames', replay_data)
        self.assertGreater(len(replay_data['frames']), 0)
        
        # Test observation extraction
        frame_data = replay_data['frames'][100]
        obs = processor.create_observation_from_replay(frame_data)
        
        # Validate observation structure
        self.assertIn('player_frame', obs)
        self.assertIn('global_view', obs)
        self.assertIn('physics_state', obs)
        self.assertIn('graph_data', obs)
        
        # Validate dimensions
        self.assertEqual(obs['player_frame'].shape, (12, 84, 84))
        self.assertEqual(obs['global_view'].shape, (176, 100))
        self.assertIsInstance(obs['physics_state'], np.ndarray)
    
    def test_behavioral_cloning_training(self):
        """Test BC trainer functionality."""
        # Create mock dataset
        dataset = self._create_mock_bc_dataset()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize trainer
        model = HGTMultimodalExtractor(
            observation_space=self._get_test_obs_space(),
            features_dim=512
        )
        trainer = BehavioralCloningTrainer(model, {'learning_rate': 1e-3, 'log_interval': 10})
        
        # Train for one epoch
        initial_loss = trainer.train_epoch(dataloader)
        
        # Loss should be reasonable
        self.assertLess(initial_loss, 10.0)
        self.assertGreater(initial_loss, 0.0)
        
        # Test evaluation
        eval_results = trainer.evaluate(dataloader)
        self.assertIn('loss', eval_results)
        self.assertIn('accuracy', eval_results)
        self.assertGreater(eval_results['accuracy'], 0.1)  # Should be better than random
    
    def test_bc_to_rl_transition(self):
        """Test BC to RL policy transfer."""
        # Train a simple BC model
        bc_model = self._train_simple_bc_model()
        
        # Save BC model
        bc_model_path = "test_bc_model.pth"
        torch.save(bc_model, bc_model_path)
        
        # Initialize RL policy with BC weights
        transition = BCToRLTransition(bc_model_path, self._get_test_rl_config())
        ppo_agent = transition.initialize_rl_policy()
        
        # Verify policy is initialized
        self.assertIsNotNone(ppo_agent)
        self.assertIsNotNone(ppo_agent.policy)
        
        # Test hybrid loss function
        hybrid_loss_fn = transition.create_hybrid_loss(bc_model, bc_weight=0.1)
        self.assertIsNotNone(hybrid_loss_fn)
```

### Integration Tests
**File**: `tests/test_replay_integration.py`

```python
class TestReplayIntegration(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        """Test complete replay processing pipeline."""
        # Process replay files into dataset
        processor = ReplayDataProcessor()
        replay_files = glob.glob("test_data/replays/*.replay")
        dataset = processor.create_behavioral_cloning_dataset(replay_files)
        
        # Train BC model
        model = HGTMultimodalExtractor(
            observation_space=self._get_test_obs_space(),
            features_dim=512
        )
        trainer = BehavioralCloningTrainer(model, self._get_bc_config())
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train for a few epochs
        for epoch in range(3):
            loss = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        # Evaluate final performance
        eval_results = trainer.evaluate(dataloader)
        self.assertGreater(eval_results['accuracy'], 0.5)  # Should learn something meaningful
        
        # Test transition to RL
        bc_model_path = "integration_test_bc_model.pth"
        torch.save(model, bc_model_path)
        
        transition = BCToRLTransition(bc_model_path, self._get_test_rl_config())
        ppo_agent = transition.initialize_rl_policy()
        
        # Test that RL agent can make decisions
        test_obs = self._create_test_observation()
        action = ppo_agent.predict(test_obs)
        self.assertIn(action[0], range(6))  # Valid N++ action
```

### Performance Tests
**File**: `tests/test_replay_performance.py`

```python
class TestReplayPerformance(unittest.TestCase):
    def test_processing_speed(self):
        """Test replay processing performance."""
        processor = ReplayDataProcessor()
        
        # Time processing of large replay file
        start_time = time.time()
        replay_data = processor.parse_replay_file("test_data/long_level.replay")
        processing_time = time.time() - start_time
        
        # Should process at least 1000 frames per second
        frames_per_second = len(replay_data['frames']) / processing_time
        self.assertGreater(frames_per_second, 1000)
    
    def test_memory_efficiency(self):
        """Test memory usage during batch processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        processor = ReplayDataProcessor()
        
        # Process multiple large replays
        for i in range(10):
            replay_data = processor.parse_replay_file(f"test_data/large_replay_{i}.replay")
            # Process observations
            for frame in replay_data['frames'][::10]:  # Sample every 10th frame
                obs = processor.create_observation_from_replay(frame)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 1GB)
        self.assertLess(memory_increase, 1024 * 1024 * 1024)
```

## Implementation Steps

### Phase 1: Fix Replay Data Ingestion (3-4 days)
1. **Complete `create_observation_from_replay()`**
   - Implement visual feature extraction
   - Add physics state extraction
   - Build graph representation
   - Add comprehensive error handling

2. **Enhance Multimodal Processing**
   - Ensure frame stacking works correctly
   - Validate observation dimensions
   - Add cross-modal consistency checks

### Phase 2: Implement Behavioral Cloning (2-3 days)
1. **Create BC Trainer**
   - Implement training loop
   - Add evaluation metrics
   - Create model checkpointing
   - Add tensorboard logging

2. **Create BC Dataset Class**
   - Implement PyTorch Dataset interface
   - Add data augmentation options
   - Handle variable-length episodes

### Phase 3: BC-to-RL Transition (2-3 days)
1. **Implement Weight Transfer**
   - Create weight mapping between BC and RL models
   - Add validation of transferred weights
   - Handle architecture differences

2. **Create Hybrid Training**
   - Implement BC regularization
   - Add adaptive BC weight scheduling
   - Create evaluation metrics

### Phase 4: Quality Validation and Testing (2-3 days)
1. **Enhance Data Quality Validation**
   - Add comprehensive quality metrics
   - Create quality reporting dashboard
   - Add automated quality filtering

2. **Create Comprehensive Tests**
   - Unit tests for all components
   - Integration tests for full pipeline
   - Performance benchmarks

## Success Metrics
- **Processing Speed**: >1000 frames per second
- **Data Quality**: >80% of replays meet quality thresholds
- **BC Accuracy**: >70% action prediction accuracy on validation set
- **Memory Efficiency**: <1GB memory usage for large datasets
- **Integration Success**: BC-initialized RL agents show faster learning

## Dependencies
- Existing binary replay parser
- HGT multimodal architecture
- Test replay datasets
- nclone integration for graph representation

## Estimated Effort
- **Time**: 2-3 weeks
- **Complexity**: Medium-High (multimodal data processing)
- **Risk**: Medium (depends on replay data quality)

## Notes
- Coordinate with data collection team for high-quality replay datasets
- Consider data augmentation techniques for limited replay data
- Plan for incremental dataset updates as more replays become available
- Ensure compatibility with existing HGT architecture