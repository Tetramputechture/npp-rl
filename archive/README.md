# Archived scripts and modules

These files are retained for reference but are not part of the current, supported Phase 1 workflow. They have known issues (outdated imports, experimental code, or duplicate functionality) and should be revisited or deleted later.

- benchmark_ppo.py: References a non-existent `npp_rl.environments.nplusplus` module. Use `npp_rl.agents.enhanced_training` instead for training/benchmarking.
- ppo_tune.py: Optuna tuner with outdated imports and legacy extractors. Needs environment import fixes before reuse.
- recurrent_ppo_tune.py: Recurrent PPO tuner with outdated imports.
- recurrent_ppo_train.py: Wrapper for recurrent training; relies on `npp_rl.agents.npp_agent_recurrent_ppo` which has outdated imports.
- record_best_models.py: Utility to record videos of tuned models; uses outdated environment import path.
- npp_feature_extractor.py: Legacy feature extractor that depends on removed/renamed modules; superseded by `agents/enhanced_feature_extractor.py`.
- npp_feature_extractor_impala.py: Experimental IMPALA-based extractor; not wired into current training entrypoint.
- ppo_training_callback.py: Experimental callback; not used by current entrypoints.
- recurrent_ppo_hyperparameters.py: Recurrent PPO config; archived with recurrent stack.
- util.py: Small math util no longer referenced.

If you need functionality from these, prefer the maintained equivalents or update the imports to match the current `nclone` environment layout in a separate PR.
