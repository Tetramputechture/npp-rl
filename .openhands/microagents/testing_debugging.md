---
agent: 'CodeActAgent'
triggers: ['test', 'testing', 'pytest', 'debug', 'debugging', 'unittest', 'assert', 'mock', 'error', 'bug']
---

# Testing and Debugging Guidelines for NPP-RL

## Core Testing Philosophy

**Test BEHAVIOR, not implementation details. Focus on what the code DOES, not what it IS.**

## Testing Patterns to AVOID

### ❌ DON'T Test Static Values

```python
# BAD: Testing constants and enum values
def test_movement_type_enum_values(self):
    self.assertEqual(MovementType.WALK, 0)
    self.assertEqual(MovementType.JUMP, 1)
    self.assertIsInstance(MovementType.WALK, int)

# BAD: Testing physics constants
def test_gravity_constants(self):
    self.assertEqual(GRAVITY_FALL, 0.06666666666666665)
    self.assertAlmostEqual(NINJA_RADIUS, 10.0)
```

### ❌ DON'T Test Implementation Details

```python
# BAD: Testing internal structure
def test_neural_network_layer_count(self):
    self.assertEqual(len(self.model.layers), 5)
    self.assertIsInstance(self.model.layers[0], nn.Conv3d)
```

## Testing Patterns to FOLLOW

### ✅ Test Functional Behavior

```python
# GOOD: Test movement classification behavior
def test_horizontal_movement_classified_as_walk(self):
    """Test that pure horizontal movement is classified as walking."""
    movement_type, params = self.classifier.classify_movement(
        src_pos=(100.0, 100.0),
        tgt_pos=(200.0, 100.0)  # 100 pixels right, same height
    )
    
    # Test the behavior and results
    self.assertEqual(movement_type, MovementType.WALK)
    self.assertGreater(params['time_estimate'], 0)
    self.assertLess(params['energy_cost'], 10.0)
    self.assertGreaterEqual(params['success_probability'], 0.8)

# GOOD: Test physics calculations
def test_upward_movement_requires_jumping_with_realistic_physics(self):
    """Test that upward movements require jumping and produce realistic physics."""
    result = self.calculator.calculate_trajectory(
        src_pos=(100.0, 100.0),
        tgt_pos=(150.0, 50.0)  # 50 pixels up and right
    )
    
    # Test physics behavior
    self.assertTrue(result.requires_jump)
    self.assertGreater(result.energy_cost, 0)
    self.assertBetween(result.success_probability, 0.0, 1.0)
    self.assertGreater(result.time_of_flight, 0)
    
    # Test physics relationships
    horizontal_result = self.calculator.calculate_trajectory((100, 100), (150, 100))
    self.assertGreater(result.energy_cost, horizontal_result.energy_cost)  # Upward costs more
```

### ✅ Test Edge Cases and Boundaries

```python
def test_zero_distance_movement_handled_gracefully(self):
    """Test that zero-distance movement is handled without crashing."""
    result = self.classifier.classify_movement(
        src_pos=(100.0, 100.0),
        tgt_pos=(100.0, 100.0)  # Same position
    )
    
    # Should handle gracefully
    self.assertIsNotNone(result)
    movement_type, params = result
    self.assertGreaterEqual(params['energy_cost'], 0)
    self.assertGreaterEqual(params['time_estimate'], 0)

def test_extreme_distance_movement_validation(self):
    """Test handling of unrealistic movement distances."""
    # Test very long distance
    result = self.classifier.classify_movement(
        src_pos=(0.0, 0.0),
        tgt_pos=(10000.0, 10000.0)  # Way beyond level bounds
    )
    
    # Should either handle gracefully or indicate infeasibility
    movement_type, params = result
    if movement_type != MovementType.IMPOSSIBLE:
        self.assertGreater(params['energy_cost'], 100)  # Should be expensive
        self.assertLess(params['success_probability'], 0.5)  # Low success rate
```

### ✅ Test Integration Between Components

```python
def test_movement_classifier_trajectory_calculator_consistency(self):
    """Test that movement classification and trajectory calculation agree."""
    test_movements = [
        ((100, 100), (200, 100)),  # Horizontal
        ((100, 100), (150, 50)),   # Upward diagonal
        ((100, 100), (100, 150)),  # Downward
    ]
    
    for src_pos, tgt_pos in test_movements:
        # Both components should produce consistent results
        movement_type, _ = self.classifier.classify_movement(src_pos, tgt_pos)
        trajectory = self.calculator.calculate_trajectory(src_pos, tgt_pos)
        
        # Check logical consistency
        if movement_type == MovementType.WALK:
            self.assertFalse(trajectory.requires_jump)
        elif movement_type == MovementType.JUMP:
            self.assertTrue(trajectory.requires_jump)
        
        # Both should agree on feasibility
        classifier_feasible = movement_type != MovementType.IMPOSSIBLE
        self.assertEqual(classifier_feasible, trajectory.feasible)
```

## Machine Learning Model Testing

### Test Model Architecture

```python
def test_3d_feature_extractor_handles_temporal_input(self):
    """Test that 3D feature extractor processes temporal frames correctly."""
    obs_space = SpacesDict({
        'player_frame': Box(low=0, high=255, shape=(84, 84, 12), dtype=np.uint8),
        'global_view': Box(low=0, high=255, shape=(176, 100, 1), dtype=np.uint8),
        'game_state': Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
    })
    
    extractor = 3DFeatureExtractor(obs_space, features_dim=512)
    
    # Test with realistic batch sizes
    for batch_size in [1, 4, 16]:
        mock_obs = {
            'player_frame': torch.randint(0, 256, (batch_size, 84, 84, 12), dtype=torch.uint8),
            'global_view': torch.randint(0, 256, (batch_size, 176, 100, 1), dtype=torch.uint8),
            'game_state': torch.randn(batch_size, 32, dtype=torch.float32)
        }
        
        with torch.no_grad():
            features = extractor(mock_obs)
        
        # Test output properties
        self.assertEqual(features.shape, (batch_size, 512))
        self.assertTrue(torch.all(torch.isfinite(features)))
        self.assertGreater(torch.norm(features), 0)  # Not all zeros

def test_model_handles_different_input_ranges(self):
    """Test that models work with various input value ranges."""
    model = self.create_test_model()
    
    # Test edge cases
    test_cases = [
        torch.zeros(1, 84, 84, 12),           # All black
        torch.ones(1, 84, 84, 12) * 255,     # All white  
        torch.randint(0, 256, (1, 84, 84, 12)),  # Random
    ]
    
    for test_input in test_cases:
        mock_obs = {'player_frame': test_input, ...}
        
        with torch.no_grad():
            output = model(mock_obs)
        
        # Should produce valid outputs for all inputs
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertGreater(torch.norm(output), 0)
```

### Test Training Integration

```python
def test_ppo_integration_with_custom_extractor(self):
    """Test that custom feature extractors work with SB3 PPO."""
    # Create minimal environment for testing
    env = DummyVecEnv([lambda: create_minimal_test_env()])
    
    # Create PPO with custom feature extractor
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            'features_extractor_class': 3DFeatureExtractor,
            'features_extractor_kwargs': {'features_dim': 512}
        },
        n_steps=64,    # Small for testing
        batch_size=32,
        verbose=0
    )
    
    # Test that learning doesn't crash
    initial_obs = env.reset()
    model.learn(total_timesteps=128)
    
    # Test that predictions work
    action, _ = model.predict(initial_obs, deterministic=True)
    self.assertIsNotNone(action)
    self.assertTrue(0 <= action < env.action_space.n)

def test_icm_loss_computation(self):
    """Test that ICM losses are computed correctly."""
    icm = ICMNetwork(feature_dim=128, action_dim=6)
    
    batch_size = 4
    state_features = torch.randn(batch_size, 128)
    next_state_features = torch.randn(batch_size, 128)
    actions = torch.randint(0, 6, (batch_size,))
    
    losses = icm.compute_losses(state_features, next_state_features, actions)
    
    # Test loss properties
    self.assertIn('forward_loss', losses)
    self.assertIn('inverse_loss', losses)
    self.assertIn('total_icm_loss', losses)
    
    for loss_name, loss_value in losses.items():
        self.assertTrue(torch.isfinite(loss_value), f"{loss_name} is not finite")
        self.assertGreaterEqual(loss_value.item(), 0, f"{loss_name} is negative")
    
    # Test that gradients flow properly
    total_loss = losses['total_icm_loss']
    total_loss.backward()
    
    for param in icm.parameters():
        if param.grad is not None:
            self.assertTrue(torch.all(torch.isfinite(param.grad)))
```

## Physics and Game Logic Testing

### Test Physics Accuracy

```python
def test_gravity_effects_on_trajectory_calculations(self):
    """Test that gravity properly affects trajectory calculations."""
    calculator = TrajectoryCalculator()
    
    # Compare different movement types
    horizontal = calculator.calculate_trajectory((0, 0), (100, 0))    # Horizontal
    upward = calculator.calculate_trajectory((0, 0), (0, -100))       # Upward  
    downward = calculator.calculate_trajectory((0, 0), (0, 100))      # Downward
    
    # Physics relationships should hold
    self.assertGreater(upward.energy_cost, horizontal.energy_cost)    # Upward costs more
    self.assertLess(downward.energy_cost, upward.energy_cost)         # Gravity assists downward
    self.assertGreater(upward.time_of_flight, horizontal.time_estimate)  # Upward takes longer

def test_ninja_state_affects_movement_physics(self):
    """Test that ninja state properly influences movement calculations."""
    classifier = MovementClassifier()
    
    # Same movement with different states
    src_pos, tgt_pos = (100.0, 100.0), (100.0, 150.0)
    
    # Ground state vs wall state
    ground_state = NinjaState(movement_state=MovementState.RUNNING, ground_contact=True)
    wall_state = NinjaState(movement_state=MovementState.WALL_SLIDING, wall_contact=True)
    
    ground_result = classifier.classify_movement(src_pos, tgt_pos, ground_state)
    wall_result = classifier.classify_movement(src_pos, tgt_pos, wall_state)
    
    # Should produce different results based on state
    ground_type, ground_params = ground_result
    wall_type, wall_params = wall_result
    
    # Wall state should enable different movement types
    if wall_type == MovementType.WALL_SLIDE:
        self.assertNotEqual(ground_type, wall_type)
        self.assertLess(wall_params.get('energy_cost', float('inf')), 
                       ground_params.get('energy_cost', 0))
```

## Error Handling and Robustness Testing

### Test Invalid Input Handling

```python
def test_handles_invalid_positions_gracefully(self):
    """Test that invalid positions don't crash the system."""
    classifier = MovementClassifier()
    
    invalid_positions = [
        (float('inf'), 100),    # Infinite coordinate
        (100, float('nan')),    # NaN coordinate
        (-1e10, 1e10),         # Extreme values
        (None, 100),           # None value
    ]
    
    for invalid_pos in invalid_positions:
        try:
            result = classifier.classify_movement((0, 0), invalid_pos)
            
            # If it doesn't raise an exception, result should be reasonable
            self.assertIsInstance(result, tuple)
            movement_type, params = result
            self.assertIsInstance(params, dict)
            
        except (ValueError, TypeError) as e:
            # Acceptable to raise appropriate exceptions for invalid input
            self.assertIn('invalid', str(e).lower())

def test_model_memory_limits(self):
    """Test that models handle memory pressure gracefully."""
    model = self.create_test_model()
    
    # Test with increasing batch sizes to find limits
    max_working_batch_size = 1
    
    for batch_size in [1, 4, 16, 64, 256]:
        try:
            mock_input = self.create_mock_batch(batch_size)
            
            with torch.no_grad():
                output = model(mock_input)
            
            max_working_batch_size = batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                break
            else:
                raise  # Re-raise if it's not a memory error
    
    # Should handle reasonable batch sizes
    self.assertGreaterEqual(max_working_batch_size, 4)
```

## Test Organization Patterns

### Organize Tests by Functionality

```python
class TestMovementClassifierWalkBehavior(unittest.TestCase):
    """Test walking movement classification specifically."""
    
    def setUp(self):
        self.classifier = MovementClassifier()
    
    def test_horizontal_movement_is_walk(self):
        # Test pure horizontal movement
        
    def test_slight_incline_still_walk(self):
        # Test small elevation changes
        
    def test_walk_energy_scaling(self):
        # Test energy costs scale reasonably with distance

class TestMovementClassifierJumpBehavior(unittest.TestCase):
    """Test jumping movement classification specifically."""
    
    def test_vertical_movement_requires_jump(self):
        # Test upward movement classification
        
    def test_jump_physics_accuracy(self):
        # Test jump trajectory calculations
```

### Use Descriptive Test Names

```python
# GOOD: Clear, behavior-focused names
def test_long_horizontal_movements_have_linear_energy_scaling(self):
def test_wall_contact_enables_wall_slide_classification(self):
def test_high_velocity_movements_classified_as_launch_pad_boosts(self):

# BAD: Unclear what's being tested
def test_movement_params(self):
def test_classifier(self):
def test_values(self):
```

## Debugging Utilities

### Create Debug-Friendly Test Helpers

```python
def create_debug_movement_scenario(scenario_name: str):
    """Create predefined movement scenarios for debugging."""
    scenarios = {
        'simple_walk': {'src': (100, 100), 'tgt': (200, 100)},
        'high_jump': {'src': (100, 100), 'tgt': (150, 50)},
        'wall_slide': {'src': (100, 100), 'tgt': (100, 150)},
        'impossible': {'src': (0, 0), 'tgt': (10000, -10000)},
    }
    
    if scenario_name not in scenarios:
        available = ', '.join(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    return scenarios[scenario_name]

def debug_print_movement_analysis(src_pos, tgt_pos, result):
    """Print detailed movement analysis for debugging."""
    print(f"\nMovement Analysis:")
    print(f"  From: {src_pos} -> To: {tgt_pos}")
    print(f"  Distance: {math.dist(src_pos, tgt_pos):.1f} pixels")
    
    if isinstance(result, tuple):
        movement_type, params = result
        print(f"  Classification: {movement_type}")
        for key, value in params.items():
            print(f"    {key}: {value}")
    else:
        print(f"  Result: {result}")
```

### Performance Testing Helpers

```python
def measure_inference_time(model, input_data, iterations=100):
    """Measure model inference time for performance testing."""
    import time
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_data)
    
    # Measure
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    return avg_time

def assert_reasonable_performance(avg_time, max_time_ms=10.0):
    """Assert that performance is within reasonable bounds."""
    avg_time_ms = avg_time * 1000
    assert avg_time_ms < max_time_ms, f"Average inference time {avg_time_ms:.2f}ms exceeds limit {max_time_ms}ms"
```

This testing approach ensures robust, maintainable code that focuses on validating actual behavior rather than implementation details, leading to more reliable and debuggable NPP-RL systems.
