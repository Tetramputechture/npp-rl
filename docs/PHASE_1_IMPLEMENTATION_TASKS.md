### Phase 1 Implementation Tasks — Core Enhancements (nclone + npp-rl)

Scope: Make the N++ RL stack robust and ready for advanced phases by enriching observations (multi-modal), adding potential-based reward shaping, verifying Gym compliance, enabling parallelization on H100s, and preparing human replay data. No code is included here; this is an execution plan tied to current files and structure.

---

### 0) Grounding and Readiness Checks
- **Confirm repo layout**: 
  - `nclone/` simulation and environments live under `nclone/nclone/`.
  - `npp-rl/` contains training entrypoints and agents.
- **Run quick environment smoke test** (no edits):
  - Launch `python -m nclone.nclone.test_environment --headless --profile-frames 300 | cat` to ensure headless simulation ticks without GUI.
  - From `npp-rl/`, run `python ppo_train.py` or `python -c "from npp_rl.agents.npp_agent_ppo import start_training; start_training(render_mode='rgb_array', num_envs=2)"` to validate end-to-end wiring.
  - Acceptance: env steps without exceptions; VecEnv wraps successfully; observation keys match model expectations.

---

### 1) Observation Space Enrichment (Multi-Modal Inputs)
Goal: Expand symbolic features to bridge the Physics-Awareness Gap and support PBRS, while keeping backward compatibility via flags.

- **1.1 Add richer ninja physics state in `nplay_headless`**
  - Files: `nclone/nclone/nplay_headless.py`, `nclone/nclone/ninja.py`.
  - Tasks:
    - Extend `NPlayHeadless.get_ninja_state()` to include (normalized to [-1,1] or [0,1] per semantics):
      - Buffer counters: `jump_buffer`, `floor_buffer`, `wall_buffer`, `launch_pad_buffer` (scaled by window size).
      - Contact flags/counters: `floor_count`, `wall_count`, `ceiling_count` (e.g., clipped to {0,1}).
      - Floor/ceiling normals: `floor_normalized_x/y`, `ceiling_normalized_x/y`.
      - Impact risk estimator: instantaneous `impact_vel` proxy and `normal_y` terms required for death check.
      - State flags: `airborn`, `walled`, `state` (one-hot or scalar in [0,1]).
    - Keep current 10-D vector for compatibility; add new fields after it.
    - Document new feature ordering and scaling.
  - Acceptance: `get_ninja_state()` unit test returns fixed-length vector; values in expected bounds.

- **1.2 Enrich entity features**
  - Files: `nclone/nclone/nplay_headless.py`, `nclone/nclone/entities.py`.
  - Tasks:
    - Review/standardize `Entity*.get_state()` across types to return normalized attributes needed for PBRS and exploration:
      - Examples: `EntityToggleMine.state` (untoggled/toggling/toggled), `EntityExitSwitch.activated`, `EntityDoor*.is_open`, `EntityLaunchPad` orientation, `EntityThwump.state/facing`, `EntityDroneZap.mode`.
    - Ensure each `get_state()` returns a fixed-length vector (padded) within [0,1]; update MAX per-type attributes if needed.
    - In `NPlayHeadless.get_entity_states(only_one_exit_and_switch=False)`: verify per-type max counts and ensure ordering is documented and deterministic.
  - Acceptance: Feature vectors for each entity type are stable, normalized, and bounded; shape remains constant across maps.

- **1.3 Expand observation contract in `BasicLevelNoGold`**
  - Files: `nclone/nclone/nclone_environments/basic_level_no_gold/basic_level_no_gold.py`, `.../constants.py`, `.../observation_processor.py`.
  - Tasks:
    - Update `GAME_STATE_FEATURES_*` constants to reflect new ninja/entity features. Provide two profiles:
      - Minimal: `GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH` (current) for fast baselines.
      - Rich: `GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT` capturing buffers, normals, hazard/door semantics.
    - In `_get_observation()`, keep returning `game_state` from `NPlayHeadless.get_state_vector(only_exit_and_switch=...)` but ensure it now includes the enriched feature set when the flag is disabled.
    - In `ObservationProcessor.process_game_state`: append `time_remaining` and vectors to objectives as today; validate dimension equals constants.
    - Ensure `observation_space` shapes (Dict with `player_frame`, `global_view`, `game_state`) match the enriched sizes.
  - Acceptance: `check_env` passes; SB3 `MultiInputPolicy` builds; `3DFeatureExtractor` infers the new `game_state` dimension at runtime.

- **1.4 Player- and global-view image stability**
  - Files: `nclone/nclone/nplay_headless.py`, `.../observation_processor.py`.
  - Tasks:
    - Confirm grayscale conversion path in `NPlayHeadless.render()` consistently returns `(H, W, 1)` `uint8`.
    - In `ObservationProcessor.frame_around_player()`, re-verify bounds and padding; add assertion for final shape `(PLAYER_FRAME_HEIGHT, PLAYER_FRAME_WIDTH, 1)`.
  - Acceptance: Deterministic shapes and dtypes across modes and VecEnv.

---

### 2) Potential-Based Reward Shaping (PBRS)
Goal: Add potential-difference shaping terms without changing the optimal policy.

- **2.1 Implement potential functions**
  - Files: `nclone/nclone/nclone_environments/basic_level_no_gold/reward_calculation/` (new module `pbrs_potentials.py`).
  - Tasks:
    - Define reusable potentials Φ(s):
      - Distance to nearest objective (switch when inactive; exit when open), with door-openness semantics.
      - Hazard proximity penalty (min distance to active hazards; exclude safe thwump faces; respect mine toggle windows if available in entity state).
      - Impact risk penalty proxy using `impact_vel` and `normal_y` terms.
    - Normalize each potential to comparable scales; document units and bounds.
  - Acceptance: Unit tests on synthetic states confirm monotonicity and bounds.

- **2.2 Integrate PBRS into reward**
  - Files: `.../reward_calculation/main_reward_calculator.py`, `.../navigation_reward_calculator.py` (if needed).
  - Tasks:
    - Compute `r_shaped = r_env + γ * Φ(s') - Φ(s)` per step; make weights configurable.
    - Add config toggles in `BasicLevelNoGold` constructor (e.g., `enable_pbrs=True`, `pbrs_weights=dict(...)`).
    - Log per-component contributions for ablations via `ep_info` or a debug logger.
  - Acceptance: Shaping on/off yields expected value shifts without changing terminal rewards; no shape explosions.

---

### 3) Gym Compliance, Vectorization, and Performance
Goal: Ensure stable operation under `SubprocVecEnv`, correct spaces/dtypes, and good throughput.

- **3.1 Gymnasium compliance check**
  - Files: `npp-rl/npp_rl/agents/npp_agent_ppo.py`, environment files.
  - Tasks:
    - Run `check_env(BasicLevelNoGold(...))` after observation changes; resolve any warnings (dtype, bounds, `reset` return signature already OK).
    - Ensure `info` is a dict; `terminated` vs `truncated` semantics are correct (uses `TruncationChecker`).
  - Acceptance: No `check_env` errors.

- **3.2 SubprocVecEnv stability**
  - Files: `npp-rl/npp_rl/agents/npp_agent_ppo.py`.
  - Tasks:
    - Confirm `BasicLevelNoGold` is picklable (no lambdas/stateful GPU pointers in init); pass constructor args explicitly in `make_vec_env` lambdas.
    - Add `env_kwargs` dict to centralize flags (e.g., `enable_frame_stack=True`, `enable_pbrs=True`).
    - Validate with `n_envs` ∈ {8, 16, 32, 64} and measure steps/sec.
  - Acceptance: Training starts cleanly with ≥32 envs; no deadlocks; memory stable.

- **3.3 Rendering and headless performance**
  - Files: `nclone/nclone/nplay_headless.py`.
  - Tasks:
    - Verify `'rgb_array'` mode bypasses GUI fully; ensure `SDL_VIDEODRIVER=dummy` is set before `pygame.display.set_mode`.
    - Cache and reuse buffers (already present); ensure no cross-process resource conflicts.
  - Acceptance: CPU usage scales with env count without GPU display contention.

---

### 4) Mixed Precision and H100 Readiness
Goal: Accelerate training on H100 with sane defaults while using SB3.

- **4.1 Enable TF32 / matmul precision for PyTorch 2.x**
  - Files: `npp-rl/ppo_train.py`, `npp-rl/recurrent_ppo_train.py`, or inside `npp_rl/agents/npp_agent_ppo.py` before model creation.
  - Tasks:
    - Set `torch.backends.cuda.matmul.allow_tf32 = True` and `torch.set_float32_matmul_precision('high')` at process start when CUDA available.
    - Document that SB3’s internal training loops limit full AMP integration; TF32 often yields good speedups on H100.
  - Acceptance: No numerical instability observed; throughput increases on H100.

- **4.2 Optional AMP exploration (time-boxed)**
  - Tasks:
    - Evaluate feasibility of custom SB3 policy/trainer subclass to wrap forward/backward in `torch.cuda.amp.autocast` + `GradScaler`.
    - If complexity is high, defer to Phase 4; otherwise, prototype a minimal subclass and A/B test speed/quality.
  - Acceptance: Decision documented; no regression to baseline if attempted.

---

### 5) Human Replay Data Processing (Preparation Only)
Goal: Prepare a clean dataset of state-action pairs for later BC pretraining; do not integrate training yet.

- **5.1 Define input spec and staging area**
  - Files/dirs: create `npp-rl/datasets/` and `npp-rl/tools/replay_ingest.py`.
  - Tasks:
    - Specify expected replay formats (e.g., JSONL of frames with: player inputs, positions, entity states, timestamps) and a converter API.
    - Add a schema doc describing fields and normalization ranges to align with `BasicLevelNoGold._get_observation()` and enriched features.
  - Acceptance: Sample schema file exists; converter CLI help shows usage.

- **5.2 Build converter to state-action dataset**
  - Files: `npp-rl/tools/replay_ingest.py`.
  - Tasks:
    - Parse raw replays → produce `npz`/Parquet shards with:
      - `observations`: dict arrays aligned to `observation_space` keys (using the same processors),
      - `actions`: mapped to 6-action discrete space (use `npp-rl/convert_actions.py` for mapping),
      - `meta`: level id, lengths, quality flags.
    - Include optional segmentation into subtasks (switch/exit phases) via event boundaries.
  - Acceptance: Dry-run on a small sample produces consistent shapes; unit tests compare counts and ranges.

- **5.3 Data quality and de-duplication**
  - Tasks:
    - Implement basic validators: action in {0..5}, obs keys present, numeric bounds.
    - Deduplicate near-identical trajectories; record provenance.
  - Acceptance: Validator report generated; failed samples quarantined.

---

### 6) Configuration, Flags, and Logging
Goal: Make new features toggleable and observable.

- **6.1 Add environment flags and plumb through**
  - Files: `basic_level_no_gold.py`, `main_reward_calculator.py`.
  - Tasks:
    - Flags: `enable_pbrs`, `pbrs_weights`, `use_rich_game_state`, `enable_short_episode_truncation`, `enable_debug_overlay`.
    - Ensure flags are settable via `make_vec_env` `env_kwargs` in `npp_agent_ppo.py`.
  - Acceptance: Flags reflected in episode info and logs.

- **6.2 Training logs**
  - Files: `npp_rl/agents/npp_agent_ppo.py`.
  - Tasks:
    - Ensure `VecMonitor` records rewards; include scalar summaries for PBRS components via custom callback (optional Phase 1 stretch).
  - Acceptance: TensorBoard curves for reward and episode length are present; if added, shaping components visible.

---

### 7) Tests and Validation
Goal: Prevent regressions and ensure shapes/values are stable.

- **7.1 Unit tests (env + processors)**
  - Files: new under `nclone/nclone/nclone_environments/tests/`.
  - Tasks:
    - Test `NPlayHeadless.get_ninja_state()` length/bounds with crafted ninja states.
    - Test `get_entity_states()` max counts, padding, and bounds for representative entity types.
    - Test `ObservationProcessor` returns correct shapes/dtypes for `player_frame`, `global_view`, `game_state` with and without frame stacking.
  - Acceptance: All tests green locally.

- **7.2 PBRS unit tests**
  - Files: `.../reward_calculation/tests/`.
  - Tasks:
    - Verify Φ decreases as agent approaches objective; hazard potential penalizes proximity.
    - Validate `r_shaped` adds potential difference only.
  - Acceptance: Tests green; numeric invariants hold.

- **7.3 Integration tests (VecEnv)**
  - Tasks:
    - Spin up `SubprocVecEnv(n_envs=8)` and run 1K steps; assert no shape/dtype drift, no leaks.
  - Acceptance: Stable RSS and step timing; no exceptions.

---

### 8) Documentation
Goal: Make the changes discoverable for later phases.

- **8.1 Developer docs (this repo)**
  - Files: `npp-rl/README.md` and a new `docs/phase1.md`.
  - Tasks:
    - Document observation schemas (minimal vs rich), feature order, normalization.
    - Describe PBRS config and recommended defaults.
    - Provide commands for quick-start training with `num_envs` guidance for H100.
  - Acceptance: Docs render and reference exact file/function names.

---

### Deliverables and Acceptance
- Enriched observation features accessible via `game_state` with stable sizes (minimal and rich modes).
- PBRS implemented with configurable potentials and clean integration into existing reward pipeline.
- SubprocVecEnv training works at ≥32 envs; TF32 enabled for H100; optional AMP decision documented.
- Replay ingestion tooling produces clean state-action datasets aligned to env observations.
- Unit and integration tests for critical components.

---

### Cross-File Reference Index
- Environment and simulation: 
  - `nclone/nclone/nclone_environments/base_environment.py`
  - `nclone/nclone/nclone_environments/basic_level_no_gold/basic_level_no_gold.py`
  - `nclone/nclone/nclone_environments/basic_level_no_gold/observation_processor.py`
  - `nclone/nclone/nclone_environments/basic_level_no_gold/reward_calculation/*`
  - `nclone/nclone/nplay_headless.py`, `nclone/nclone/ninja.py`, `nclone/nclone/entities.py`
- RL and training:
  - `npp-rl/npp_rl/agents/npp_agent_ppo.py`
  - `npp-rl/ppo_train.py`, `npp-rl/recurrent_ppo_train.py`
  - `npp-rl/convert_actions.py`
- Tooling to add (Phase 1):
  - `npp-rl/tools/replay_ingest.py`
  - `nclone/.../reward_calculation/pbrs_potentials.py`
  - Tests under `nclone/.../tests/` and `.../reward_calculation/tests/`


