### Phase 2 Implementation Tasks — Advanced Exploration, Structural Learning, and Imitation (nclone + npp-rl)

Scope: Implement intrinsic motivation (ICM baseline with optional IEM-PPO), add a graph-based structural observation path with a GNN encoder, and introduce Behavioral Cloning (BC) pretraining on the human replay dataset prepared in Phase 1. Provide toggles, metrics, tests, and documentation so these features are safely composable with the Phase 1 stack.

---

### 0) Prerequisites and Branch Hygiene
- **Baseline alignment**:
  - Ensure Phase 1 deliverables are merged: enriched symbolic features, PBRS, SubprocVecEnv stability, TF32, and replay ingestion tooling.
  - Run the Phase 1 smoke tests to confirm a green baseline before Phase 2 edits.
- **Branching**:
  - Create a working branch `nick/phase-2` from the latest main.
  - Gate new features behind flags so Phase 1 training remains reproducible.
- **Acceptance**: Baseline PPO still runs end-to-end with default flags off for Phase 2 features.

---

### 1) Intrinsic Motivation Module (ICM baseline; optional IEM-PPO)
Goal: Provide intrinsic rewards to improve exploration in sparse-reward settings, while keeping PPO training loops compatible with SB3.

- **1.1 Implement ICM**
  - Files (new): `npp-rl/npp_rl/intrinsic/icm.py`, `npp-rl/npp_rl/intrinsic/utils.py`
  - Design:
    - Use the policy’s features-extractor backbone outputs as the feature space φ(s).
    - Networks:
      - Inverse model g_inv(φ(s), φ(s')) → â (predict action distribution over 6 discrete actions; Cross-Entropy loss).
      - Forward model g_fwd(φ(s), a) → φ̂(s') (predict next features; MSE loss gives intrinsic reward).
    - Intrinsic reward per step: r_int = η * 0.5 * ||φ̂(s') − φ(s')||². Clip to [0, r_int_max].
    - Total reward: r_total = r_ext + α * r_int, where α is configurable (can be annealed during training).
  - Integration strategy:
    - Prefer an environment wrapper: `npp-rl/npp_rl/wrappers/intrinsic_reward_wrapper.py` that intercepts (obs, reward, done, info) to add r_int and log components.
    - Alternatively, provide a `RolloutBuffer`-aware callback `npp-rl/npp_rl/callbacks/intrinsic_callback.py` if wrapper constraints arise.
  - Training:
    - Optimize ICM heads jointly with PPO updates. Maintain separate optimizers or parameter groups; allow shared vs decoupled learning rates via config.
    - Balance losses: L_total = L_PPO + λ_inv * CE(â, a) + λ_fwd * MSE(φ̂(s'), φ(s')).
  - Config flags (extend `npp-rl/npp_rl/agents/config.py` or central args):
    - `enable_icm: bool`, `icm_eta: float`, `icm_alpha: float`, `icm_rint_clip: float`, `icm_lambda_inv: float`, `icm_lambda_fwd: float`, `icm_share_backbone: bool`.
  - Logging:
    - TensorBoard scalars per update: mean r_ext, mean r_int, α, loss_inv, loss_fwd.
    - Episode info: add `"r_int_sum", "r_ext_sum"` to `info` via wrapper.
  - Acceptance:
    - With a fixed seed on a moderately sparse level, `r_int_sum > 0` and distinct from 0 for ≥90% episodes with `enable_icm=True`.
    - Training runs stable ≥10M steps with `n_envs≥16` without NaNs. Turning ICM off restores Phase 1 baseline behavior.

- **1.2 Optional: IEM-PPO (uncertainty-driven)**
  - Files (new): `npp-rl/npp_rl/intrinsic/iem.py`
  - Add an uncertainty head u(φ(s), a, φ(s')) → scalar; use it to modulate intrinsic reward targeting hard transitions.
  - Flag: `enable_iem: bool` (mutually exclusive with ICM at runtime). Keep this behind a separate toggle; default off.
  - Acceptance: Compiles and runs on a short benchmark; produces non-zero uncertainty; defer large-scale tuning to a later pass.

---

### 2) Graph-Based Structural Observation and GNN Encoder
Goal: Provide a physics-aware structural representation to aid planning in maze-like and non-linear levels.

- **2.1 Graph construction in the environment**
  - Files (new):
    - `nclone/nclone/graph/graph_builder.py`
    - `nclone/nclone/nclone_environments/basic_level_no_gold/graph_observation.py`
  - Graph spec (padded, fixed-size for Gym):
    - Nodes: grid cells (24×24 pixels per cell; 42×23 grid → 966 max) plus entity nodes. Provide `N_MAX_NODES` cap; pad with zeros; add `node_mask`.
    - Node features (normalized): one-hot tile type, solidity, hazard flags, entity type/state, ninja position flag on node.
    - Edges: traversability (walk/jump/wall-slide/fall/one-way), and functional relations (switch→door, launchpad→target). Use directed edges; `E_MAX_EDGES` cap; provide `[2, E]` `edge_index`, `edge_attr`, and `edge_mask`.
  - Determinism:
    - Enforce canonical ordering: iterate rows then cols for grid nodes; append entities by stable type order; edges sorted by (src, dst, type).
  - Observation keys and spaces:
    - Add to observation dict (behind flag `use_graph_obs=True`):
      - `graph_node_feats: (N_MAX_NODES, F_node)`
      - `graph_edge_index: (2, E_MAX_EDGES)` (int32)
      - `graph_edge_feats: (E_MAX_EDGES, F_edge)`
      - `graph_node_mask: (N_MAX_NODES,)`, `graph_edge_mask: (E_MAX_EDGES,)`
  - Performance:
    - Precompute static tile node features per level; only dynamic entity features and ninja marker update per step.
  - Acceptance:
    - `check_env` passes with `use_graph_obs=True`.
    - For a sample level, `N<=N_MAX_NODES`, `E<=E_MAX_EDGES`, masks correct; ordering is stable across resets.

- **2.2 GNN encoder**
  - Files (new): `npp-rl/npp_rl/models/gnn.py`
  - Model:
    - Implement GraphSAGE (or GCN) with masked global mean+max pooling to produce a fixed-size graph embedding.
    - Input from padded arrays; do not depend on external PyG at first (custom message passing on adjacency derived from `edge_index` and masks). If using PyTorch Geometric, add dependency behind a flag, and provide a pure-PyTorch fallback.
  - Acceptance:
    - Unit test: feed a toy graph; ensure embedding shape is `[batch, D_gnn]` and insensitive to padding (masks respected).

- **2.3 Multimodal feature extractor integration**
  - Files: `npp-rl/npp_rl/models/feature_extractors.py`, `npp-rl/npp_rl/agents/npp_agent_ppo.py`
  - Implement `NppMultimodalGraphExtractor`:
    - Inputs: existing CNN features (frames), MLP features (symbolic game_state), and optional GNN embedding (graph_* keys). Concatenate to a single feature vector.
    - Make inclusion of GNN conditional on `use_graph_obs`.
  - Policy wiring:
    - Set `features_extractor_class` and pass `features_extractor_kwargs` with dims for `F_node`, `F_edge`, `N_MAX_NODES`, `E_MAX_EDGES` and GNN layer sizes.
  - Acceptance:
    - PPO builds with/without graph obs enabled. Feature dims auto-inferred from spaces; no shape mismatches under `SubprocVecEnv`.

---

### 3) Behavioral Cloning (BC) Pretraining on Human Replays
Goal: Use the Phase 1 replay dataset to initialize the policy with human-like behavior.

- **3.1 Dataset loader and mapping**
  - Files (new): `npp-rl/npp_rl/data/bc_dataset.py`
  - Implement a `torch.utils.data.Dataset` that reads Phase 1 shards (`npz`/Parquet) and returns batches aligning to the current observation dict (including graph keys when enabled) and discrete action labels in {0..5}.
  - Support optional filtering (quality flags, max episode length) and stratified sampling by level id.
  - Acceptance: Iteration over a small sample yields consistent shapes matching env `observation_space` dtypes and bounds.

- **3.2 BC training script**
  - Files (new): `npp-rl/bc_pretrain.py`
  - Train the SB3 policy architecture in supervised mode:
    - Initialize a policy with `NppMultimodalGraphExtractor` configured to match the dataset observation schema.
    - Loss: Cross-Entropy over the 6-way action logits; optional entropy regularization.
    - Optimizer: AdamW; cosine decay; gradient clipping.
    - Freezing: optional `--freeze_backbone` for CNN/GNN/MLP encoders for first K steps.
  - CLI Example:
    - `python -m npp_rl.bc_pretrain --dataset_dir datasets/shards --policy npp --use_graph_obs --epochs 10 --batch_size 4096 --lr 3e-4 --freeze_backbone_steps 2000 --out pretrained/ppo_bc_init.zip`
  - Checkpointing & export:
    - Save encoder and policy heads in SB3-compatible format; provide a loader utility used by PPO training to warm-start weights.
  - Acceptance:
    - Training loss decreases on a validation split; top-1 action accuracy ≥ target threshold (e.g., ≥60% on held-out replays for simple levels).
    - Exported checkpoint can be loaded to initialize PPO without shape errors.

- **3.3 PPO warm start**
  - Files: `npp-rl/npp_rl/agents/npp_agent_ppo.py`
  - Add `--bc_init_path`/`bc_init_path` config to load pre-trained weights before PPO training.
  - Acceptance: PPO starts with BC weights; initial episode returns and action distributions match the pretrain policy qualitatively.

---

### 4) Metrics, Evaluation, and Ablations for Phase 2
Goal: Quantify exploration and structural learning gains.

- **4.1 Exploration metrics**
  - Files (new): `npp-rl/npp_rl/eval/exploration_metrics.py`
  - Implement per-episode and rolling metrics:
    - Unique tile coverage (% of traversable cells visited), visitation entropy over discretized positions, mean r_int per step, ratio r_int/r_ext, success rate on moderately complex levels.
  - Logging: integrate into TensorBoard and `episode_info`.
  - Acceptance: Metrics computed without significant overhead (<5% step-time regression).

- **4.2 Evaluation script**
  - Files (new): `npp-rl/eval_phase2.py`
  - CLI to run trained agents with combinations: {ICM on/off} × {graph obs on/off} × {BC init on/off} on a curated level set.
  - Output: CSV/JSON summary with seeds, means, and 95% CIs for key metrics.
  - Acceptance: Produces reproducible results across runs; artifacts saved under `runs/phase2_eval/*`.

- **4.3 Ablation harness**
  - Files (new): `npp-rl/npp_rl/ablate/phase2.py`
  - Automate short training runs for: baseline, +ICM, +Graph, +BC, and combinations.
  - Acceptance: Generates a single table with metric deltas vs baseline.

---

### 5) Configuration, Flags, and Logging
Goal: Make Phase 2 features opt-in and tuneable, with robust logging.

- Add flags (centralized in agents config and plumbed to env/policy):
  - `use_graph_obs: bool`, graph caps and dims, GNN layers/hidden size.
  - `enable_icm`, ICM hyperparameters (η, α, λ_inv, λ_fwd, clip, share_backbone), `enable_iem`.
  - `use_bc_init: bool`, `bc_init_path: str`.
- Training harness updates:
  - Ensure flags are logged in run configs; add them to the run directory for reproducibility.
- Acceptance: Flags visible in logs; toggling them changes behavior without code edits.

---

### 6) Tests and Validation
Goal: Prevent regressions and guarantee shape/bounds correctness.

- **6.1 Unit tests** (new under `nclone/nclone/graph/tests/` and `npp-rl/npp_rl/tests/`):
  - Graph builder determinism and bounds for a synthetic level; masks correct; ordering stable across resets.
  - GNN encoder masking correctness: padding does not affect embeddings.
  - ICM: forward/inverse heads compute; r_int non-negative; respects clip.
  - BC dataset loader: shapes match spaces; action labels in range; batching works.
- **6.2 Integration tests**:
  - Run `SubprocVecEnv(n_envs=8)` with `use_graph_obs=True` and `enable_icm=True` for 2K steps; assert no dtype/shape drift; no deadlocks.
- **6.3 Overfit sanity**:
  - Train BC on a tiny replay subset until ≥95% accuracy; confirms data/label alignment.
- Acceptance: All tests green locally and under CI (if present).

---

### 7) Documentation
Goal: Make Phase 2 features discoverable and actionable.

- `npp-rl/docs/phase2.md` (new):
  - Document graph observation schema, caps, and masks; list exact feature orders and normalization.
  - Describe ICM math, hyperparameters, and recommended defaults.
  - Provide BC pretraining quickstart and warm-start PPO instructions.
  - Include evaluation/ablation commands and expected metric improvements.
- Update `npp-rl/README.md`:
  - Add a Phase 2 quickstart section with example commands for each feature.

---

### 8) Prompts and Operational Guidance for Agentic Execution
When forwarding to an agentic LLM, include these directives to reduce ambiguity:
- Always state the target files to create/edit and preserve indentation and formatting; do not reformat unrelated code.
- Confirm observation space shapes before wiring models; print inferred dims to logs once per process.
- For ICM: verify that feature extractor outputs are detached/cloned appropriately to avoid unintended gradient flows when sharing backbones is disabled.
- For graph: enforce canonical node/edge ordering; include masks; never allocate variable-sized Gym spaces.
- For BC: ensure the policy architecture matches the environment observation schema at PPO time; validate by running a forward pass on a sample batch before training.
- Do not block on interactive prompts; add `--yes` or non-interactive flags where applicable.

---

### Deliverables and Acceptance
- Intrinsic motivation (ICM) implemented, toggleable, logged, and stable under vectorized training; optional IEM scaffold exists.
- Graph observation path producing padded arrays with masks; GNN encoder integrated into the multimodal extractor behind a flag.
- Behavioral Cloning pretraining pipeline produces a checkpoint that warm-starts PPO and improves early episode returns.
- Exploration metrics and evaluation scripts produce reproducible summaries; ablation harness outputs a comparison table.
- Unit/integration tests validating shapes, masks, determinism, and stability.

---

### Cross-File Reference Index
- Environment and simulation:
  - `nclone/nclone/graph/graph_builder.py`
  - `nclone/nclone/nclone_environments/basic_level_no_gold/graph_observation.py`
  - Existing: `nclone/nclone/nclone_environments/basic_level_no_gold/*`, `nclone/nclone/nplay_headless.py`
- RL and training:
  - `npp-rl/npp_rl/models/feature_extractors.py`
  - `npp-rl/npp_rl/models/gnn.py`
  - `npp-rl/npp_rl/intrinsic/icm.py`, `npp-rl/npp_rl/intrinsic/iem.py`, `npp-rl/npp_rl/wrappers/intrinsic_reward_wrapper.py`, `npp-rl/npp_rl/callbacks/intrinsic_callback.py`
  - `npp-rl/npp_rl/data/bc_dataset.py`, `npp-rl/bc_pretrain.py`
  - `npp-rl/npp_rl/eval/exploration_metrics.py`, `npp-rl/eval_phase2.py`, `npp-rl/npp_rl/ablate/phase2.py`
  - `npp-rl/npp_rl/agents/npp_agent_ppo.py` (flags, policy wiring)


