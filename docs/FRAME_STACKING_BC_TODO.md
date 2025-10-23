# Frame Stacking Support in BC Pretraining - Implementation TODO

## Overview

This document outlines the required changes to add frame stacking support to the BC pretraining pipeline. Currently, BC pretraining does not support frame stacking, which causes shape mismatches when transferring weights to RL training with frame stacking enabled.

## Current State

### What Works ✅
- RL training with frame stacking (via FrameStackWrapper)
- Weight transfer mechanism (architecture_trainer.py)
- Dynamic CNN architecture (configurable_extractor.py)
- Frame stacking configuration (FrameStackConfig)

### What's Missing ❌
- BC dataset frame stacking support
- BC trainer frame stacking configuration
- Pretraining pipeline frame stacking propagation
- Checkpoint metadata for frame stacking config

## Required Changes

### 1. BC Dataset Enhancement

**File**: `npp_rl/training/bc_dataset.py`

**Changes Required**:

```python
class BCReplayDataset(Dataset):
    def __init__(
        self,
        replay_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        filter_successful_only: bool = True,
        max_replays: Optional[int] = None,
        architecture_config: Optional[Any] = None,
        normalize_observations: bool = True,
        frame_stack_config: Optional[Dict] = None,  # NEW PARAMETER
    ):
        """Initialize BC replay dataset.
        
        Args:
            ...
            frame_stack_config: Frame stacking configuration dict with keys:
                - enable_visual_frame_stacking: bool
                - visual_stack_size: int
                - enable_state_stacking: bool
                - state_stack_size: int
                - padding_type: str ('zero' or 'repeat')
        """
        self.frame_stack_config = frame_stack_config or {}
        self.enable_visual_stacking = frame_stack_config.get('enable_visual_frame_stacking', False)
        self.visual_stack_size = frame_stack_config.get('visual_stack_size', 4)
        self.enable_state_stacking = frame_stack_config.get('enable_state_stacking', False)
        self.state_stack_size = frame_stack_config.get('state_stack_size', 4)
        
        # Initialize frame buffers for stacking
        if self.enable_visual_stacking:
            self.player_frame_buffer = deque(maxlen=self.visual_stack_size)
            self.global_view_buffer = deque(maxlen=self.visual_stack_size)
        
        if self.enable_state_stacking:
            self.game_state_buffer = deque(maxlen=self.state_stack_size)
        
        # ... rest of initialization
    
    def _process_single_replay(self, replay_path: Path) -> List[Tuple[Dict, int]]:
        """Process a single replay file with frame stacking support."""
        
        # Load replay and run simulation
        replay = CompactReplay.load(replay_path)
        env = self._create_environment()
        
        samples = []
        
        # Reset buffers for each replay
        if self.enable_visual_stacking:
            self.player_frame_buffer.clear()
            self.global_view_buffer.clear()
        if self.enable_state_stacking:
            self.game_state_buffer.clear()
        
        # Process each frame in replay
        for frame_idx, action in enumerate(replay.actions):
            # Get current observation
            obs = env.get_observation()
            
            # Add to buffers
            if self.enable_visual_stacking:
                self._add_to_visual_buffer(obs)
            if self.enable_state_stacking:
                self._add_to_state_buffer(obs)
            
            # Only start generating samples after buffers are full
            if self._buffers_ready():
                # Stack observations
                stacked_obs = self._stack_observations(obs)
                samples.append((stacked_obs, action))
            
            # Step environment
            env.step(action)
        
        return samples
    
    def _add_to_visual_buffer(self, obs: Dict):
        """Add visual frames to buffer with padding if needed."""
        player_frame = obs.get('player_frame')
        global_view = obs.get('global_view')
        
        if player_frame is not None:
            self.player_frame_buffer.append(player_frame.copy())
        if global_view is not None:
            self.global_view_buffer.append(global_view.copy())
        
        # Pad initial frames
        padding_type = self.frame_stack_config.get('padding_type', 'zero')
        while len(self.player_frame_buffer) < self.visual_stack_size:
            if padding_type == 'repeat' and len(self.player_frame_buffer) > 0:
                self.player_frame_buffer.appendleft(self.player_frame_buffer[0].copy())
            else:  # zero padding
                shape = player_frame.shape if player_frame is not None else (84, 84, 1)
                self.player_frame_buffer.appendleft(np.zeros(shape, dtype=np.uint8))
        
        # Similar for global_view
        while len(self.global_view_buffer) < self.visual_stack_size:
            if padding_type == 'repeat' and len(self.global_view_buffer) > 0:
                self.global_view_buffer.appendleft(self.global_view_buffer[0].copy())
            else:
                shape = global_view.shape if global_view is not None else (...)
                self.global_view_buffer.appendleft(np.zeros(shape, dtype=np.uint8))
    
    def _add_to_state_buffer(self, obs: Dict):
        """Add game state to buffer with padding if needed."""
        game_state = obs.get('game_state')
        if game_state is None:
            return
        
        self.game_state_buffer.append(game_state.copy())
        
        # Pad initial states
        padding_type = self.frame_stack_config.get('padding_type', 'zero')
        while len(self.game_state_buffer) < self.state_stack_size:
            if padding_type == 'repeat' and len(self.game_state_buffer) > 0:
                self.game_state_buffer.appendleft(self.game_state_buffer[0].copy())
            else:  # zero padding
                self.game_state_buffer.appendleft(np.zeros_like(game_state))
    
    def _buffers_ready(self) -> bool:
        """Check if all buffers have enough frames."""
        visual_ready = (not self.enable_visual_stacking or 
                       len(self.player_frame_buffer) >= self.visual_stack_size)
        state_ready = (not self.enable_state_stacking or 
                      len(self.game_state_buffer) >= self.state_stack_size)
        return visual_ready and state_ready
    
    def _stack_observations(self, current_obs: Dict) -> Dict:
        """Stack buffered observations into final format."""
        stacked_obs = current_obs.copy()
        
        if self.enable_visual_stacking:
            # Stack visual frames: (stack_size, H, W, C)
            if 'player_frame' in current_obs:
                stacked_obs['player_frame'] = np.array(list(self.player_frame_buffer))
            if 'global_view' in current_obs:
                stacked_obs['global_view'] = np.array(list(self.global_view_buffer))
        
        if self.enable_state_stacking:
            # Stack game states: (stack_size, state_dim)
            if 'game_state' in current_obs:
                stacked_obs['game_state'] = np.concatenate(list(self.game_state_buffer))
        
        return stacked_obs
```

**Key Points**:
- Use deque for efficient FIFO frame buffering
- Support both zero and repeat padding
- Reset buffers for each replay
- Generate samples only after buffers are full
- Stack observations in correct shape for CNN input

### 2. Pretraining Pipeline Update

**File**: `npp_rl/training/pretraining_pipeline.py`

**Changes Required**:

```python
class PretrainingPipeline:
    def __init__(
        self,
        replay_data_dir: str,
        architecture_config: ArchitectureConfig,
        output_dir: Path,
        tensorboard_writer: Optional[SummaryWriter] = None,
        frame_stack_config: Optional[Dict] = None,  # NEW PARAMETER
    ):
        """Initialize pretraining pipeline.
        
        Args:
            replay_data_dir: Directory containing replay files
            architecture_config: Architecture configuration
            output_dir: Output directory for BC checkpoints
            tensorboard_writer: Optional TensorBoard writer
            frame_stack_config: Frame stacking configuration (same format as ArchitectureTrainer)
        """
        self.replay_data_dir = Path(replay_data_dir)
        self.architecture_config = architecture_config
        self.output_dir = Path(output_dir)
        self.tensorboard_writer = tensorboard_writer
        self.frame_stack_config = frame_stack_config or {}
        
        # Log frame stacking configuration
        if frame_stack_config:
            logger.info(f"Frame stacking enabled in BC pretraining:")
            logger.info(f"  Visual: {frame_stack_config.get('enable_visual_frame_stacking', False)} "
                       f"(size: {frame_stack_config.get('visual_stack_size', 4)})")
            logger.info(f"  State: {frame_stack_config.get('enable_state_stacking', False)} "
                       f"(size: {frame_stack_config.get('state_stack_size', 4)})")
    
    def prepare_bc_data(
        self,
        use_cache: bool = True,
        max_replays: Optional[int] = None,
        filter_successful_only: bool = True,
    ) -> Optional[BCReplayDataset]:
        """Process replay data into BC training format with frame stacking."""
        
        try:
            dataset = BCReplayDataset(
                replay_dir=str(self.replay_data_dir),
                cache_dir=str(self.output_dir / "cache"),
                use_cache=use_cache,
                filter_successful_only=filter_successful_only,
                max_replays=max_replays,
                architecture_config=self.architecture_config,
                frame_stack_config=self.frame_stack_config,  # PASS CONFIG
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create BC dataset: {e}", exc_info=True)
            return None
```

Update `run_bc_pretraining_if_available()` function:

```python
def run_bc_pretraining_if_available(
    replay_data_dir: Optional[str],
    architecture_config: ArchitectureConfig,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    num_workers: int = 4,
    device: str = "auto",
    max_replays: Optional[int] = None,
    tensorboard_writer: Optional[SummaryWriter] = None,
    frame_stack_config: Optional[Dict] = None,  # NEW PARAMETER
) -> Optional[str]:
    """Convenience function to run BC pretraining with frame stacking support."""
    
    try:
        pipeline = PretrainingPipeline(
            replay_data_dir=str(replay_data_dir),
            architecture_config=architecture_config,
            output_dir=output_dir,
            tensorboard_writer=tensorboard_writer,
            frame_stack_config=frame_stack_config,  # PASS CONFIG
        )
        
        # ... rest of function
```

### 3. train_and_compare.py Update

**File**: `scripts/train_and_compare.py`

**Changes Required**:

In the `train_single_architecture()` function, pass frame stacking config to pretraining:

```python
def train_single_architecture(...):
    """Train a single architecture configuration."""
    
    # ... existing code ...
    
    # Build frame stacking configuration
    frame_stack_config = None
    if args.enable_visual_frame_stacking or args.enable_state_stacking:
        frame_stack_config = {
            'enable_visual_frame_stacking': args.enable_visual_frame_stacking,
            'visual_stack_size': args.visual_stack_size,
            'enable_state_stacking': args.enable_state_stacking,
            'state_stack_size': args.state_stack_size,
            'padding_type': args.frame_stack_padding,
        }
    
    # ... existing code ...
    
    # Run BC pretraining with frame stacking config
    if args.replay_data_dir and not args.no_pretraining:
        pretrained_ckpt = run_bc_pretraining_if_available(
            replay_data_dir=args.replay_data_dir,
            architecture_config=arch_config,
            output_dir=arch_output_dir / "bc_pretraining",
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            tensorboard_writer=tb_writer.get_writer("bc") if tb_writer else None,
            frame_stack_config=frame_stack_config,  # PASS CONFIG
        )
```

### 4. Checkpoint Metadata Enhancement

**File**: `npp_rl/training/policy_utils.py`

**Changes Required**:

```python
def save_policy_checkpoint(
    policy: nn.Module,
    checkpoint_path: str,
    epoch: int,
    metrics: Dict[str, float],
    architecture_name: str,
    frame_stack_config: Optional[Dict] = None,  # NEW PARAMETER
) -> None:
    """Save policy checkpoint with metadata.
    
    Args:
        policy: Policy network to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        metrics: Training metrics
        architecture_name: Architecture name
        frame_stack_config: Frame stacking configuration
    """
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'architecture': architecture_name,
        'frame_stacking': frame_stack_config or {},  # ADD METADATA
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Log frame stacking info
    if frame_stack_config:
        logger.info(f"  Frame stacking config saved in checkpoint:")
        logger.info(f"    Visual: {frame_stack_config.get('enable_visual_frame_stacking', False)} "
                   f"(size: {frame_stack_config.get('visual_stack_size', 1)})")
        logger.info(f"    State: {frame_stack_config.get('enable_state_stacking', False)} "
                   f"(size: {frame_stack_config.get('state_stack_size', 1)})")
```

### 5. Validation Enhancement

**File**: `scripts/validate_checkpoint_simple.py`

Add check for checkpoint metadata:

```python
def analyze_checkpoint(checkpoint_path: str) -> Dict:
    """Analyze BC checkpoint structure and frame stacking configuration."""
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Check for frame stacking metadata
    if 'frame_stacking' in checkpoint:
        print(f"\n✓ Checkpoint has frame stacking metadata:")
        fs_config = checkpoint['frame_stacking']
        print(f"  Visual stacking: {fs_config.get('enable_visual_frame_stacking', False)} "
              f"(size: {fs_config.get('visual_stack_size', 1)})")
        print(f"  State stacking: {fs_config.get('enable_state_stacking', False)} "
              f"(size: {fs_config.get('state_stack_size', 1)})")
        print(f"  Padding type: {fs_config.get('padding_type', 'zero')}")
    else:
        print(f"\n⚠ Checkpoint missing frame stacking metadata (old format)")
    
    # ... rest of analysis
```

## Implementation Priority

1. **HIGH PRIORITY** - BC Dataset Enhancement
   - Most critical change
   - Enables frame stacking in BC pretraining
   - ~200 lines of code

2. **HIGH PRIORITY** - Pretraining Pipeline Update
   - Propagates frame stacking config
   - ~50 lines of code

3. **MEDIUM PRIORITY** - train_and_compare.py Update
   - Wires everything together
   - ~20 lines of code

4. **MEDIUM PRIORITY** - Checkpoint Metadata Enhancement
   - Improves debugging and validation
   - ~30 lines of code

5. **LOW PRIORITY** - Validation Enhancement
   - Nice to have for better UX
   - ~20 lines of code

## Testing Plan

### Unit Tests

1. Test BC dataset with frame stacking:
   ```python
   def test_bc_dataset_frame_stacking():
       dataset = BCReplayDataset(
           replay_dir="test_replays",
           frame_stack_config={
               'enable_visual_frame_stacking': True,
               'visual_stack_size': 4,
           }
       )
       obs, action = dataset[0]
       assert obs['player_frame'].shape[0] == 4  # 4 stacked frames
   ```

2. Test padding modes:
   ```python
   def test_frame_stacking_padding():
       # Test zero padding
       # Test repeat padding
       # Verify initial frames are handled correctly
   ```

### Integration Tests

1. End-to-end BC pretraining with frame stacking:
   ```bash
   python scripts/train_and_compare.py \
       --experiment-name "bc_frame_stacking_test" \
       --architectures mlp_baseline \
       --replay-data-dir test_replays \
       --bc-epochs 5 \
       --enable-visual-frame-stacking \
       --visual-stack-size 4 \
       --total-timesteps 1000 \
       --num-envs 2
   ```

2. Validate checkpoint compatibility:
   ```bash
   python scripts/validate_checkpoint_simple.py \
       --checkpoint experiments/bc_frame_stacking_test/bc_checkpoint.pth \
       --enable-visual-stacking \
       --visual-stack-size 4
   ```

3. Transfer to RL training:
   ```bash
   python scripts/train_and_compare.py \
       --experiment-name "rl_with_bc_pretrain" \
       --architectures mlp_baseline \
       --train-dataset test/train \
       --test-dataset test/test \
       --enable-visual-frame-stacking \
       --visual-stack-size 4 \
       --total-timesteps 10000 \
       --num-envs 2
   ```

## Success Criteria

- [x] Validation tools identify the issue
- [ ] BC dataset supports frame stacking
- [ ] Pretrained checkpoint has correct input channel shapes
- [ ] Weight transfer works without shape mismatches
- [ ] Checkpoint metadata includes frame stacking config
- [ ] Validation script confirms compatibility
- [ ] End-to-end training works with BC pretraining + frame stacking

## Timeline Estimate

- BC Dataset Enhancement: **4-6 hours**
- Pretraining Pipeline Update: **1-2 hours**
- train_and_compare.py Update: **1 hour**
- Checkpoint Metadata: **1 hour**
- Testing: **2-3 hours**
- **Total: ~10-13 hours**

## Notes

- Frame stacking in BC is important for temporal consistency
- The CNN architecture already supports dynamic input channels
- Existing RL frame stacking works correctly via FrameStackWrapper
- Main gap is in BC dataset generation phase

## References

- `npp_rl/training/bc_dataset.py` - BC dataset implementation
- `npp_rl/training/pretraining_pipeline.py` - Pretraining pipeline
- `nclone/gym_environment/frame_stack_wrapper.py` - RL frame stacking
- `npp_rl/feature_extractors/configurable_extractor.py` - CNN architecture
- `docs/FRAME_STACKING_PRETRAINING_VALIDATION.md` - Issue documentation
