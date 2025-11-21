"""PPO with proper action mask preservation for vectorized environments.

This module provides MaskedPPO, a custom PPO implementation that ensures
action_mask is properly preserved in dictionary observations throughout
the rollout collection process, preventing masked action selection bugs.
"""

import logging

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

logger = logging.getLogger(__name__)


class MaskedPPO(PPO):
    """PPO with action mask preservation for masked action policies.

    This class overrides collect_rollouts to ensure that action_mask
    is properly preserved in dictionary observations when using vectorized
    environments. This prevents the bug where masked actions are selected
    due to stale or missing action masks.

    Key fixes:
    1. Preserves action_mask in observation dicts during obs_as_tensor conversion
    2. Ensures action_mask stays synchronized with observations across environments
    3. Validates action_mask presence before policy forward pass
    4. Adds debug logging for mask tracking (disabled in production mode)
    """

    def __init__(self, *args, debug=False, **kwargs):
        """Initialize MaskedPPO with debug flag support."""
        super().__init__(*args, **kwargs)
        self.debug = debug
        logger.info(f"MaskedPPO initialized with debug={self.debug}")

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.

        This method overrides the base PPO's collect_rollouts to ensure
        action_mask is properly preserved in dictionary observations.

        Args:
            env: The training environment
            callback: Callback that will be called at each step
            rollout_buffer: Buffer to fill with rollouts
            n_rollout_steps: Number of experiences to collect per environment

        Returns:
            True if training should continue, False if training should stop
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # CRITICAL FIX: Preserve action_mask when converting observations to tensors
            # The base implementation uses obs_as_tensor() which may drop action_mask
            # from dictionary observations. We need to ensure it's preserved.
            obs_dict = self._last_obs

            # Extract action_mask before obs_as_tensor conversion
            action_mask = None
            if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                action_mask = obs_dict["action_mask"]
                # Convert to numpy if needed
                if isinstance(action_mask, torch.Tensor):
                    action_mask = action_mask.cpu().numpy()
                elif not isinstance(action_mask, np.ndarray):
                    action_mask = np.array(action_mask, dtype=np.bool_)
                # Ensure proper dtype
                if action_mask.dtype != np.bool_:
                    action_mask = action_mask.astype(np.bool_)

                # DEFENSIVE FIX: Deep copy action_mask immediately after extraction
                # This prevents memory sharing issues in SubprocVecEnv where the mask
                # might be modified by another process via shared memory during IPC.
                # Without this copy, the mask used by the policy can differ from the
                # mask that was computed in the environment, leading to masked action bugs.
                owns_data_before = action_mask.flags["OWNDATA"]
                action_mask = action_mask.copy()

                # Ensure C-contiguous layout to prevent memory aliasing
                if not action_mask.flags["C_CONTIGUOUS"]:
                    action_mask = np.ascontiguousarray(
                        action_mask, dtype=action_mask.dtype
                    )

                # Debug logging for action_mask extraction
                if self.debug:
                    mask_hash = hash(action_mask.tobytes())
                    logger.debug(
                        f"[MaskedPPO.collect_rollouts] [STEP={n_steps}] Extracted action_mask from obs_dict: "
                        f"shape={action_mask.shape}, dtype={action_mask.dtype}, "
                        f"hash={mask_hash}, owned_before_copy={owns_data_before}, "
                        f"owns_data_now={action_mask.flags['OWNDATA']}"
                    )

                # DIAGNOSTIC: Track mask lifecycle for debugging
                if self.debug:
                    for env_idx in range(env.num_envs):
                        env_mask = (
                            action_mask[env_idx]
                            if action_mask.ndim == 2
                            else action_mask
                        )
                        logger.debug(
                            f"[MASK_TRACK] step={n_steps} env={env_idx} stage=EXTRACT "
                            f"id={id(env_mask)} owns_data={env_mask.flags['OWNDATA']} "
                            f"hash={hash(env_mask.tobytes())}"
                        )

                # CRITICAL VALIDATION: Check action_mask shape matches num_envs
                if isinstance(action_mask, np.ndarray):
                    if action_mask.ndim == 1:
                        # Single mask with multiple envs - CRITICAL ERROR
                        error_msg = (
                            f"CRITICAL BATCHING ERROR: action_mask is 1D "
                            f"(shape={action_mask.shape}) but we have {env.num_envs} environments! "
                            f"This will cause all envs to use the same mask."
                        )
                        logger.error(error_msg)
                        if self.debug:
                            logger.error(f"Mask: {action_mask}")
                            logger.error(
                                "This likely indicates improper batching in vectorized environment"
                            )
                        raise RuntimeError(error_msg)

                    elif action_mask.shape[0] != env.num_envs:
                        error_msg = (
                            f"CRITICAL BATCH SIZE MISMATCH: "
                            f"action_mask batch {action_mask.shape[0]} != num_envs {env.num_envs}. "
                            f"Masks will be applied to wrong environments!"
                        )
                        logger.error(error_msg)
                        if self.debug:
                            logger.error(f"action_mask shape: {action_mask.shape}")
                            logger.error(f"num_envs: {env.num_envs}")
                        raise RuntimeError(error_msg)

                    # Debug logging for successful batching
                    if self.debug:
                        logger.debug(
                            f"action_mask batch validated: shape={action_mask.shape}, num_envs={env.num_envs}"
                        )

            # Convert observations to tensor for policy forward pass
            # Note: obs_tensor is only used for policy, not for buffer
            obs_tensor = obs_as_tensor(obs_dict, self.device)

            # CRITICAL FIX: Always restore action_mask to obs_tensor
            if action_mask is not None:
                if not isinstance(obs_tensor, dict):
                    raise RuntimeError(
                        f"obs_tensor should be dict but got {type(obs_tensor)}. "
                        "This indicates an observation space mismatch."
                    )

                # Convert to tensor with proper shape, dtype, and device
                action_mask_tensor = torch.as_tensor(
                    action_mask, dtype=torch.bool, device=self.device
                )

                # Validate shape matches expected batch size
                if action_mask_tensor.ndim == 1:
                    # Single mask - expand to batch
                    action_mask_tensor = action_mask_tensor.unsqueeze(0).expand(
                        env.num_envs, -1
                    )
                elif action_mask_tensor.shape[0] != env.num_envs:
                    raise RuntimeError(
                        f"action_mask batch size {action_mask_tensor.shape[0]} != "
                        f"num_envs {env.num_envs}"
                    )

                # Validate action dimension
                if action_mask_tensor.shape[-1] != 6:
                    raise RuntimeError(
                        f"action_mask has {action_mask_tensor.shape[-1]} actions, expected 6"
                    )

                # Assign to obs_tensor (overwrite any existing mask)
                obs_tensor["action_mask"] = action_mask_tensor
                # Only log in debug mode (production mode skips for performance)
                if self.debug:
                    logger.debug(
                        f"[MaskedPPO.collect_rollouts] Restored action_mask to observation dict. "
                        f"Mask shape: {action_mask_tensor.shape}, "
                        f"Batch size: {env.num_envs}"
                    )

                # DIAGNOSTIC: Track mask after tensor conversion
                if self.debug:
                    for env_idx in range(env.num_envs):
                        env_mask_tensor = (
                            action_mask_tensor[env_idx]
                            if action_mask_tensor.ndim == 2
                            else action_mask_tensor
                        )
                        logger.debug(
                            f"[MASK_TRACK] step={n_steps} env={env_idx} stage=TENSOR "
                            f"id={id(env_mask_tensor)} device={env_mask_tensor.device} "
                            f"hash={hash(env_mask_tensor.cpu().numpy().tobytes())}"
                        )

            # CRITICAL VALIDATION: Check each environment's mask before policy forward
            # DEFENSIVE FIX: Validate memory ownership before policy forward
            if action_mask is not None and isinstance(action_mask, np.ndarray):
                # Check if mask owns its data - if not, force a copy
                if not action_mask.flags["OWNDATA"]:
                    logger.warning(
                        f"[MASK_OWNERSHIP] action_mask does not own its data at step {n_steps}! "
                        f"This indicates potential memory sharing. Forcing defensive copy."
                    )
                    action_mask = action_mask.copy()
                    # Update obs_tensor with the new copy
                    if isinstance(obs_tensor, dict) and "action_mask" in obs_tensor:
                        action_mask_tensor = torch.as_tensor(
                            action_mask, dtype=torch.bool, device=self.device
                        )
                        obs_tensor["action_mask"] = action_mask_tensor
            if isinstance(obs_tensor, dict) and "action_mask" in obs_tensor:
                mask = obs_tensor["action_mask"]
                if isinstance(mask, torch.Tensor):
                    if mask.dim() == 2:
                        # Check batch size
                        if mask.shape[0] != env.num_envs:
                            error_msg = (
                                f"CRITICAL BATCH SIZE MISMATCH: "
                                f"action_mask batch {mask.shape[0]} != num_envs {env.num_envs}. "
                                f"Masks will be applied to wrong environments!"
                            )
                            logger.error(error_msg)
                            if self.debug:
                                logger.error(f"Full mask:\n{mask}")
                                logger.error(
                                    f"Observation keys: {list(obs_tensor.keys())}"
                                )
                            raise RuntimeError(error_msg)

                        # Check each environment has at least one valid action
                        has_valid = mask.any(dim=1)
                        if not has_valid.all():
                            invalid_envs = (~has_valid).nonzero(as_tuple=True)[0]
                            error_msg = f"Environments {invalid_envs.tolist()} have no valid actions!"
                            logger.error(error_msg)
                            if self.debug:
                                for idx in invalid_envs:
                                    logger.error(f"Env {idx} mask: {mask[idx]}")
                            raise RuntimeError(error_msg)

                        # Debug logging for each environment's mask
                        if self.debug:
                            for env_idx in range(env.num_envs):
                                env_mask = mask[env_idx]
                                valid_actions = (
                                    env_mask.nonzero(as_tuple=True)[0].cpu().numpy()
                                )
                                logger.debug(
                                    f"[PRE-POLICY] Env {env_idx}: "
                                    f"mask={env_mask.cpu().numpy()}, "
                                    f"valid_actions={valid_actions}"
                                )

            # Forward pass through policy with preserved action_mask
            with torch.no_grad():
                # Convert to python object, to avoid issues with dict/tensor conversion
                actions, values, log_probs = self.policy(obs_tensor)

            # Convert actions to numpy (buffer expects numpy array for actions)
            # Keep values and log_probs as tensors (buffer will call .clone().cpu().numpy() on them)
            actions = actions.cpu().numpy()

            # CRITICAL VALIDATION: Verify selected actions are valid according to masks
            if action_mask is not None and isinstance(action_mask, np.ndarray):
                for env_idx in range(env.num_envs):
                    action_taken = actions[env_idx]
                    env_mask = (
                        action_mask[env_idx] if action_mask.ndim == 2 else action_mask
                    )

                    # CRITICAL: Raise error immediately if masked action selected
                    if not env_mask[action_taken]:
                        action_names = [
                            "NOOP",
                            "LEFT",
                            "RIGHT",
                            "JUMP",
                            "JUMP+LEFT",
                            "JUMP+RIGHT",
                        ]

                        # DIAGNOSTIC: Collect detailed error context
                        mask_owns_data = (
                            action_mask.flags["OWNDATA"]
                            if isinstance(action_mask, np.ndarray)
                            else "N/A"
                        )
                        mask_id = id(action_mask)
                        mask_hash = (
                            hash(action_mask.tobytes())
                            if isinstance(action_mask, np.ndarray)
                            else "N/A"
                        )

                        error_msg = (
                            f"MASKED ACTION BUG DETECTED IN PPO! "
                            f"Environment {env_idx} selected MASKED action {action_taken} ({action_names[action_taken]}). "
                            f"Mask: {env_mask}, Valid actions: {np.where(env_mask)[0]}\n"
                            f"DIAGNOSTIC INFO:\n"
                            f"  Step: {n_steps}\n"
                            f"  Mask ID: {mask_id}\n"
                            f"  Mask owns data: {mask_owns_data}\n"
                            f"  Mask hash: {mask_hash}\n"
                            f"  Mask shape: {action_mask.shape if isinstance(action_mask, np.ndarray) else 'N/A'}\n"
                            f"  Check logs for [MASK_TRACK] and [MASK_CREATE] entries to trace mask lifecycle"
                        )
                        logger.error("=" * 80)
                        logger.error(error_msg)
                        logger.error("=" * 80)

                        if self.debug:
                            logger.error(f"Full action_mask batch:\n{action_mask}")
                            logger.error(f"All selected actions: {actions}")
                            logger.error(
                                f"obs_tensor keys: {list(obs_tensor.keys()) if isinstance(obs_tensor, dict) else 'not dict'}"
                            )
                            if isinstance(obs_tensor, dict):
                                for key in [
                                    "_env_id",
                                    "_step_num",
                                    "_mask_fingerprint",
                                    "_mask_timestamp",
                                ]:
                                    if key in obs_tensor:
                                        val = obs_tensor[key]
                                        if isinstance(val, torch.Tensor):
                                            val = val.cpu().numpy()
                                        logger.error(f"{key}: {val}")

                        # RAISE ERROR IMMEDIATELY (User requirement: Option B)
                        raise RuntimeError(error_msg)

                # Debug logging for successful action selection
                if self.debug:
                    for env_idx in range(env.num_envs):
                        action_taken = actions[env_idx]
                        env_mask = (
                            action_mask[env_idx]
                            if action_mask.ndim == 2
                            else action_mask
                        )
                        logger.debug(
                            f"[POST-POLICY] Env {env_idx}: selected action {action_taken}, "
                            f"valid={bool(env_mask[action_taken])}"
                        )

            # Rescale and perform clip
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # CRITICAL FIX: Ensure new_obs also has action_mask preserved
            # The environment should provide action_mask in new_obs, but we validate it
            if isinstance(new_obs, dict):
                if "action_mask" not in new_obs:
                    # Only warn in non-production mode (production mode uses fast path)
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(
                            "new_obs from env.step() does not contain action_mask. "
                            "This may cause issues in the next rollout step."
                        )
                else:
                    # Validate the new action_mask (fast path in production)
                    new_mask = new_obs["action_mask"]
                    if isinstance(new_mask, np.ndarray):
                        if new_mask.dtype != np.bool_ and new_mask.dtype != np.int8:
                            # Convert to bool if needed
                            new_obs["action_mask"] = new_mask.astype(np.bool_)
                    # Only log in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"new_obs contains action_mask with shape: {new_obs['action_mask'].shape}"
                        )

            if isinstance(new_obs, dict) and "action_mask" in new_obs:
                self._last_obs = {
                    **new_obs
                }  # Shallow copy via dict unpacking (faster than .copy())
                self._last_obs["action_mask"] = new_obs["action_mask"].copy()
            else:
                self._last_obs = new_obs

            # MEMORY OPTIMIZATION NOTE:
            # Graph observations are already optimized via:
            # 1. float16 storage (50% reduction)
            # 2. Removed internal observations (30% reduction)
            # Total: ~80% memory savings
            # Sparse packing could provide additional savings but adds complexity
            # to rollout buffer interaction - can be added later if needed

            rollout_buffer.add(
                obs_dict,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_episode_starts = dones
            n_steps += 1

        with torch.no_grad():
            # Compute value for the last timestep
            # Ensure action_mask is preserved here too
            obs_dict = self._last_obs
            action_mask = None
            if isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                action_mask = obs_dict["action_mask"]
                # Convert to numpy if needed
                if isinstance(action_mask, torch.Tensor):
                    action_mask = action_mask.cpu().numpy()
                elif not isinstance(action_mask, np.ndarray):
                    action_mask = np.array(action_mask, dtype=np.bool_)
                # Ensure proper dtype
                if action_mask.dtype != np.bool_:
                    action_mask = action_mask.astype(np.bool_)

            obs_tensor = obs_as_tensor(obs_dict, self.device)

            # Always restore action_mask to obs_tensor
            if action_mask is not None:
                if not isinstance(obs_tensor, dict):
                    raise RuntimeError(
                        f"obs_tensor should be dict but got {type(obs_tensor)}. "
                        "This indicates an observation space mismatch."
                    )

                # Convert to tensor with proper shape, dtype, and device
                action_mask_tensor = torch.as_tensor(
                    action_mask, dtype=torch.bool, device=self.device
                )

                # Validate shape matches expected batch size
                if action_mask_tensor.ndim == 1:
                    # Single mask - expand to batch
                    action_mask_tensor = action_mask_tensor.unsqueeze(0).expand(
                        env.num_envs, -1
                    )
                elif action_mask_tensor.shape[0] != env.num_envs:
                    raise RuntimeError(
                        f"action_mask batch size {action_mask_tensor.shape[0]} != "
                        f"num_envs {env.num_envs}"
                    )

                # Validate action dimension
                if action_mask_tensor.shape[-1] != 6:
                    raise RuntimeError(
                        f"action_mask has {action_mask_tensor.shape[-1]} actions, expected 6"
                    )

                # Assign to obs_tensor (overwrite any existing mask)
                obs_tensor["action_mask"] = action_mask_tensor

            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=self._last_episode_starts
        )

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Extended from base PPO to support:
        1. Auxiliary task losses (death prediction, time-to-goal, next subgoal)
        2. Attention entropy regularization (encourages diverse objective consideration)
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        # Optional: clip value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # Initialize loss tracking
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        auxiliary_losses = []
        attention_entropy_values = []

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert to long tensor for discrete actions
                    actions = rollout_data.actions.long().flatten()

                # Re-evaluate actions to get current policy predictions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Policy gradient loss (clipped surrogate objective)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip value predictions around old value
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = torch.nn.functional.mse_loss(
                    rollout_data.returns, values_pred
                )
                value_losses.append(value_loss.item())

                # Entropy loss (encourages exploration)
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # === AUXILIARY LOSSES ===
                auxiliary_loss = torch.tensor(0.0, device=self.device)

                # 1. Auxiliary task loss (death prediction only)
                if hasattr(self.policy, "get_auxiliary_predictions"):
                    predictions = self.policy.get_auxiliary_predictions()
                    if predictions is not None and len(predictions) > 0:
                        # Compute auxiliary labels from rollout data
                        try:
                            from npp_rl.models.auxiliary_tasks import (
                                compute_auxiliary_labels,
                                compute_auxiliary_losses as compute_aux_losses,
                            )

                            # Build trajectory dict for label computation
                            # Note: DictRolloutBufferSamples only includes certain fields
                            # (observations, actions, old_values, old_log_prob, advantages, returns)
                            # It does NOT include rewards or episode_starts
                            trajectory = {
                                "observations": rollout_data.observations,
                                "returns": rollout_data.returns,
                            }

                            labels = compute_auxiliary_labels(
                                trajectory, death_horizon=10
                            )
                            aux_loss, aux_loss_dict = compute_aux_losses(
                                predictions,
                                labels,
                                weights={"death": 0.01},  # Reduced weight for stability
                            )
                            auxiliary_loss = auxiliary_loss + aux_loss

                                        # Log auxiliary losses (simplified for death prediction only)
                            for key, val in aux_loss_dict.items():
                                self.logger.record(
                                    f"train/auxiliary_{key}_loss", val.item()
                                )
                            
                            # Also log auxiliary predictions for monitoring
                            if "death_prob" in predictions:
                                death_prob_mean = predictions["death_prob"].mean().item()
                                self.logger.record("train/death_prob_mean", death_prob_mean)

                        except Exception as e:
                            # Gracefully handle auxiliary loss computation errors
                            # Continue training without auxiliary loss on failure
                            auxiliary_loss = torch.tensor(0.0, device=self.device)
                            
                            if self.debug:
                                import traceback
                                logger.warning(
                                    f"Failed to compute death prediction auxiliary loss: {e}\n"
                                    f"Continuing training without auxiliary loss for this batch.\n"
                                    f"Rollout data type: {type(rollout_data)}\n"
                                    f"Traceback: {traceback.format_exc()}"
                                )
                            else:
                                # In production mode, just log the error briefly
                                logger.warning(f"Auxiliary loss computation failed: {e}. Continuing without auxiliary loss.")

                # 2. Attention entropy regularization
                # Note: We compute entropy from the current minibatch observations
                # to avoid batch size mismatches with cached values
                if hasattr(self.policy, "get_attention_entropy"):
                    attn_entropy = self.policy.get_attention_entropy(
                        rollout_data.observations
                    )
                    if attn_entropy is not None:
                        # Negative because we want to MAXIMIZE entropy (encourage diversity)
                        # Coefficient 0.01 is similar to ent_coef in standard PPO
                        attention_entropy_coef = 0.01
                        auxiliary_loss = (
                            auxiliary_loss - attention_entropy_coef * attn_entropy
                        )
                        attention_entropy_values.append(attn_entropy.item())

                auxiliary_losses.append(auxiliary_loss.item())

                # === TOTAL LOSS ===
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + auxiliary_loss
                )

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                # Approximate KL divergence for early stopping
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

            # Early stopping based on KL divergence
            if (
                self.target_kl is not None
                and np.mean(approx_kl_divs) > 1.5 * self.target_kl
            ):
                logger.info(
                    f"Early stopping at epoch {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}"
                )
                continue_training = False
                break

        self._n_updates += self.n_epochs

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        # Log auxiliary losses if present
        if auxiliary_losses and np.mean(auxiliary_losses) != 0:
            self.logger.record("train/auxiliary_loss_total", np.mean(auxiliary_losses))
        if attention_entropy_values:
            self.logger.record(
                "train/attention_entropy", np.mean(attention_entropy_values)
            )

        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )
