"""BC pretrained weight loading utilities for PPO models.

Handles mapping BC checkpoint structure to PPO policy structure, supporting
shared, separate, and hierarchical feature extractors.
"""

import logging
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


class BCWeightLoader:
    """Loads BC pretrained weights into PPO policy networks."""

    def __init__(
        self,
        model,
        architecture_name: str,
        frame_stack_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize BC weight loader.

        Args:
            model: PPO or HierarchicalPPO model instance
            architecture_name: Name of the architecture being trained
            frame_stack_config: Frame stacking configuration dict
        """
        self.model = model
        self.architecture_name = architecture_name
        self.frame_stack_config = frame_stack_config or {}

    def load_weights(self, checkpoint_path: str) -> None:
        """Load BC pretrained weights into PPO policy.

        Maps BC checkpoint structure to PPO policy structure:
        - BC: feature_extractor.* → PPO feature extractor (depends on policy type)
        - BC policy_head is ignored (PPO trains its own action/value heads)

        PPO policies can have different feature extractor structures:
        1. Shared: features_extractor.* (share_features_extractor=True, default)
        2. Separate: pi_features_extractor.* and vf_features_extractor.* (share_features_extractor=False)
        3. Hierarchical: mlp_extractor.features_extractor.* (HierarchicalActorCriticPolicy)

        The code automatically detects the structure and maps BC weights accordingly.

        Args:
            checkpoint_path: Path to BC checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.model.device, weights_only=False
        )

        if "policy_state_dict" not in checkpoint:
            logger.warning(
                f"Checkpoint does not contain 'policy_state_dict'. "
                f"Found keys: {list(checkpoint.keys())}"
            )
            return

        # Validate architecture and frame stacking configuration
        self._validate_checkpoint(checkpoint)

        bc_state_dict = checkpoint["policy_state_dict"]

        # Detect PPO feature extractor structure
        extractor_config = self._detect_extractor_structure()

        # Log checkpoint analysis
        self._log_checkpoint_analysis(bc_state_dict, extractor_config)

        # Map BC weights to PPO structure
        mapped_state_dict = self._map_bc_weights(bc_state_dict, extractor_config)

        if not mapped_state_dict:
            logger.warning("No feature extractor weights found in BC checkpoint")
            return

        # Load weights and verify transfer
        self._load_and_verify_weights(mapped_state_dict, extractor_config)

    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Validate architecture and frame stacking configuration match.

        Args:
            checkpoint: BC checkpoint dictionary
        """
        # Validate architecture match
        if "architecture" in checkpoint:
            bc_arch = checkpoint["architecture"]
            rl_arch = self.architecture_name

            if bc_arch != rl_arch:
                logger.warning("=" * 60)
                logger.warning("ARCHITECTURE MISMATCH DETECTED")
                logger.warning(f"  BC checkpoint architecture: {bc_arch}")
                logger.warning(f"  RL training architecture:   {rl_arch}")
                logger.warning(
                    "  Some weights may not transfer. Missing keys are expected."
                )
                logger.warning("=" * 60)
            else:
                logger.info(f"✓ Architecture match: {bc_arch}")

        # Validate frame stacking configuration match
        if "frame_stacking" in checkpoint:
            self._validate_frame_stacking(checkpoint["frame_stacking"])
        else:
            self._warn_missing_frame_stacking()

    def _validate_frame_stacking(self, bc_fs_config: Dict[str, Any]) -> None:
        """Validate frame stacking configuration matches between BC and RL.

        Args:
            bc_fs_config: BC frame stacking configuration
        """
        rl_fs_config = self.frame_stack_config

        # Check visual frame stacking
        bc_visual_enabled = bc_fs_config.get("enable_visual_frame_stacking", False)
        rl_visual_enabled = rl_fs_config.get("enable_visual_frame_stacking", False)

        if bc_visual_enabled != rl_visual_enabled:
            raise ValueError(
                f"Frame stacking mismatch: BC visual stacking={bc_visual_enabled}, "
                f"RL visual stacking={rl_visual_enabled}. "
                f"BC and RL must use identical frame stacking configurations."
            )

        if bc_visual_enabled:
            bc_stack_size = bc_fs_config.get("visual_stack_size", 4)
            rl_stack_size = rl_fs_config.get("visual_stack_size", 4)
            if bc_stack_size != rl_stack_size:
                raise ValueError(
                    f"Visual stack size mismatch: BC={bc_stack_size} frames, "
                    f"RL={rl_stack_size} frames. "
                    f"This causes input dimension mismatch in CNN layers."
                )
            logger.info(f"✓ Visual frame stacking match: {bc_stack_size} frames")

        # Check state stacking
        bc_state_enabled = bc_fs_config.get("enable_state_stacking", False)
        rl_state_enabled = rl_fs_config.get("enable_state_stacking", False)

        if bc_state_enabled != rl_state_enabled:
            raise ValueError(
                f"State stacking mismatch: BC state stacking={bc_state_enabled}, "
                f"RL state stacking={rl_state_enabled}. "
                f"BC and RL must use identical state stacking configurations."
            )

        if bc_state_enabled:
            bc_state_size = bc_fs_config.get("state_stack_size", 4)
            rl_state_size = rl_fs_config.get("state_stack_size", 4)
            if bc_state_size != rl_state_size:
                raise ValueError(
                    f"State stack size mismatch: BC={bc_state_size} states, "
                    f"RL={rl_state_size} states. "
                    f"This causes input dimension mismatch in MLP layers."
                )
            logger.info(f"✓ State stacking match: {bc_state_size} states")

        logger.info("✓ Frame stacking configuration validated")

    def _warn_missing_frame_stacking(self) -> None:
        """Error if BC checkpoint has no frame stacking config but RL uses it."""
        if self.frame_stack_config and (
            self.frame_stack_config.get("enable_visual_frame_stacking", False)
            or self.frame_stack_config.get("enable_state_stacking", False)
        ):
            raise ValueError(
                "Frame stacking mismatch: RL uses frame stacking but BC checkpoint "
                "has no frame_stacking config. This will cause input dimension mismatch "
                "and weight loading will fail. BC and RL must use identical frame stacking."
            )

        logger.info("BC checkpoint has no frame stacking (single frame mode)")

    def _detect_extractor_structure(self) -> Dict[str, bool]:
        """Detect PPO model's feature extractor structure.

        Returns:
            Dictionary with boolean flags for each extractor type:
            - map_hierarchical: Has hierarchical (nested) extractor
            - map_shared: Has shared extractor
            - map_separate: Has separate pi/vf extractors
        """
        policy_keys = list(self.model.policy.state_dict().keys())

        # Check for hierarchical first (HierarchicalActorCriticPolicy)
        uses_hierarchical_extractor = any(
            "mlp_extractor.features_extractor." in k for k in policy_keys
        )

        # Check for separate extractors (share_features_extractor=False)
        uses_separate_extractors = any(
            k.startswith("pi_features_extractor.") for k in policy_keys
        ) or any(k.startswith("vf_features_extractor.") for k in policy_keys)

        # Check for shared extractor (share_features_extractor=True, default)
        uses_shared_extractor = any(
            k.startswith("features_extractor.") for k in policy_keys
        )

        if (
            not uses_shared_extractor
            and not uses_separate_extractors
            and not uses_hierarchical_extractor
        ):
            logger.warning(
                "PPO model has no recognizable feature extractor keys! "
                "Cannot load BC weights."
            )
            return {
                "map_hierarchical": False,
                "map_shared": False,
                "map_separate": False,
            }

        # Determine which mappings to use
        map_hierarchical = uses_hierarchical_extractor
        map_shared = False
        map_separate = False

        if uses_hierarchical_extractor:
            # Check if shared/separate are references or separate objects
            if (
                hasattr(self.model.policy, "features_extractor")
                and hasattr(self.model.policy, "mlp_extractor")
                and hasattr(self.model.policy.mlp_extractor, "features_extractor")
            ):
                is_same_object = (
                    self.model.policy.features_extractor
                    is self.model.policy.mlp_extractor.features_extractor
                )
                if not is_same_object and uses_shared_extractor:
                    map_shared = True
                    logger.info(
                        "  Note: Model has separate features_extractor and mlp_extractor.features_extractor"
                    )

            if (
                hasattr(self.model.policy, "pi_features_extractor")
                and hasattr(self.model.policy, "mlp_extractor")
                and hasattr(self.model.policy.mlp_extractor, "features_extractor")
            ):
                is_same_pi = (
                    self.model.policy.pi_features_extractor
                    is self.model.policy.mlp_extractor.features_extractor
                )
                if not is_same_pi and uses_separate_extractors:
                    map_separate = True
                    logger.info(
                        "  Note: Model has separate pi/vf_features_extractor and mlp_extractor.features_extractor"
                    )

        elif uses_separate_extractors:
            map_separate = True

        elif uses_shared_extractor:
            map_shared = True

        return {
            "map_hierarchical": map_hierarchical,
            "map_shared": map_shared,
            "map_separate": map_separate,
        }

    def _map_bc_weights(
        self, bc_state_dict: Dict[str, torch.Tensor], extractor_config: Dict[str, bool]
    ) -> Dict[str, torch.Tensor]:
        """Map BC feature_extractor weights to PPO structure.

        Args:
            bc_state_dict: BC checkpoint state dict
            extractor_config: Feature extractor configuration

        Returns:
            Mapped state dictionary for PPO
        """
        mapped_state_dict = {}

        for key, value in bc_state_dict.items():
            if key.startswith("feature_extractor."):
                # Remove "feature_extractor." prefix to get the sub-key
                sub_key = key[len("feature_extractor.") :]

                # Map to appropriate target based on determined extractor type
                if extractor_config["map_hierarchical"]:
                    hierarchical_key = f"mlp_extractor.features_extractor.{sub_key}"
                    mapped_state_dict[hierarchical_key] = value
                    logger.debug(f"Mapped {key} → {hierarchical_key}")

                if extractor_config["map_shared"]:
                    shared_key = f"features_extractor.{sub_key}"
                    mapped_state_dict[shared_key] = value
                    logger.debug(f"Mapped {key} → {shared_key}")

                if extractor_config["map_separate"]:
                    pi_key = f"pi_features_extractor.{sub_key}"
                    vf_key = f"vf_features_extractor.{sub_key}"

                    mapped_state_dict[pi_key] = value
                    mapped_state_dict[vf_key] = (
                        value.clone()
                    )  # Clone to avoid shared references

                    logger.debug(f"Mapped {key} → {pi_key} and {vf_key}")

            elif key.startswith("policy_head."):
                # Skip policy head weights (PPO will train its own)
                logger.debug(f"Skipping {key} (policy head not used in PPO)")
            else:
                logger.debug(f"Skipping unknown key: {key}")

        return mapped_state_dict

    def _log_checkpoint_analysis(
        self, bc_state_dict: Dict[str, torch.Tensor], extractor_config: Dict[str, bool]
    ) -> None:
        """Log BC checkpoint statistics before loading.

        Args:
            bc_state_dict: BC checkpoint state dict
            extractor_config: Feature extractor configuration
        """
        bc_param_count = sum(p.numel() for p in bc_state_dict.values())
        bc_size_mb = sum(
            p.numel() * p.element_size() for p in bc_state_dict.values()
        ) / (1024**2)

        # Count feature_extractor vs policy_head keys
        feature_keys = [
            k for k in bc_state_dict.keys() if k.startswith("feature_extractor.")
        ]

        logger.info(
            f"BC checkpoint: {bc_param_count:,} parameters ({bc_size_mb:.2f} MB)"
        )
        logger.info(f"  Transferring {len(feature_keys)} feature extractor layers")

    def _load_and_verify_weights(
        self,
        mapped_state_dict: Dict[str, torch.Tensor],
        extractor_config: Dict[str, bool],
    ) -> None:
        """Load weights into model and verify transfer.

        Args:
            mapped_state_dict: Mapped state dictionary
            extractor_config: Feature extractor configuration
        """
        # Sample a parameter to check before/after
        sample_key = list(mapped_state_dict.keys())[0] if mapped_state_dict else None
        before_value = None
        if sample_key and sample_key in self.model.policy.state_dict():
            before_value = self.model.policy.state_dict()[sample_key].clone()

        # Load only the feature extractor weights with strict=False
        try:
            missing_keys, unexpected_keys = self.model.policy.load_state_dict(
                mapped_state_dict, strict=False
            )

            # Determine extractor type for logging
            extractor_types = []
            if extractor_config["map_hierarchical"]:
                extractor_types.append("hierarchical")
            if extractor_config["map_shared"]:
                extractor_types.append("shared")
            if extractor_config["map_separate"]:
                extractor_types.append("separate")
            extractor_type = " + ".join(extractor_types)

            # Verify sample parameter changed
            if sample_key and before_value is not None:
                after_value = self.model.policy.state_dict()[sample_key]
                changed = not torch.allclose(before_value, after_value, rtol=1e-5)
                if changed:
                    logger.info("✓ BC weights successfully transferred to PPO policy")
                else:
                    logger.error(
                        "✗ BUG DETECTED: BC weights not loaded (transfer failed!)"
                    )

            # Calculate transfer percentage
            total_ppo_params = sum(p.numel() for p in self.model.policy.parameters())
            transferred_params = sum(
                mapped_state_dict[k].numel()
                for k in mapped_state_dict
                if k in self.model.policy.state_dict()
            )
            transfer_pct = (transferred_params / total_ppo_params) * 100

            if extractor_config["map_separate"]:
                # For separate extractors, we load weights twice (pi and vf)
                # So the percentage can exceed 100% - this is expected
                logger.info(
                    f"  Transferred {transferred_params:,} parameters to feature extractors"
                )
            else:
                logger.info(
                    f"  Transfer rate: {transfer_pct:.1f}% ({transferred_params:,} / {total_ppo_params:,} parameters)"
                )

            if transfer_pct < 5:
                logger.warning(
                    f"  WARNING: Very low transfer rate ({transfer_pct:.1f}%). "
                    f"Most of the model is randomly initialized!"
                )

            # Log missing keys (filtering out expected ones)
            self._log_missing_keys(missing_keys, extractor_config)

            if unexpected_keys:
                logger.warning(
                    f"  Unexpected keys in checkpoint: {len(unexpected_keys)}"
                )

            # Log what was actually loaded
            self._log_loaded_extractors(extractor_type)

        except Exception as e:
            logger.error(f"Failed to load mapped weights: {e}")
            raise

    def _log_missing_keys(
        self, missing_keys: list, extractor_config: Dict[str, bool]
    ) -> None:
        """Log information about missing keys (filtering out expected ones).

        Args:
            missing_keys: List of missing keys
            extractor_config: Feature extractor configuration
        """
        if not missing_keys:
            return

        # Categorize missing keys
        shared_feature_ext_missing = [
            k
            for k in missing_keys
            if k.startswith("features_extractor.")
            and not k.startswith("pi_features_extractor.")
            and not k.startswith("vf_features_extractor.")
        ]
        hierarchical_missing = [
            k
            for k in missing_keys
            if "mlp_extractor." in k and "features_extractor" not in k
        ]
        action_value_missing = [
            k for k in missing_keys if "action_net." in k or "value_net." in k
        ]

        # Filter out expected unused keys based on extractor config
        if extractor_config["map_separate"] and not extractor_config["map_shared"]:
            # When using separate extractors only, the shared "features_extractor.*" keys are unused
            # These are expected to be missing, so don't log them
            unexpected_missing = [
                k
                for k in missing_keys
                if k not in shared_feature_ext_missing
                and k not in hierarchical_missing
                and k not in action_value_missing
            ]
        else:
            unexpected_missing = [
                k
                for k in missing_keys
                if k not in hierarchical_missing and k not in action_value_missing
            ]

        # Only log if there are truly unexpected missing keys
        if unexpected_missing:
            logger.info(f"  Some keys not transferred: {len(unexpected_missing)}")
            logger.debug(f"    Examples: {unexpected_missing[:5]}")

    def _log_loaded_extractors(self, extractor_type: str) -> None:
        """Log information about loaded extractors.

        Args:
            extractor_type: String describing the extractor type(s)
        """
        if extractor_type == "separate":
            logger.info(
                "  → Policy and value networks will be trained from pretrained features"
            )
        elif extractor_type == "shared":
            logger.info(
                "  → Shared feature extractor loaded, policy/value heads will be trained from scratch"
            )
        elif "hierarchical" in extractor_type:
            logger.info(
                "  → Hierarchical policy loaded, high-level subtask selection will be trained from scratch"
            )
