"""Wrapper for policies to add detailed profiling instrumentation."""

import functools
from typing import Optional, Set, Callable

import torch.nn as nn

from npp_rl.training.runtime_profiler import RuntimeProfiler


def profile_method(method_name: str):
    """Decorator to profile a method call.

    Args:
        method_name: Name to use in profiler component tracking
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "profiler") and self.profiler is not None:
                with self.profiler.component(method_name):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class ProfiledPolicyWrapper(nn.Module):
    """Wraps a policy to add detailed profiling of forward passes and training.

    Tracks:
    - Policy forward passes
    - Value function forward passes
    - Action sampling
    - Log probability computation
    - Feature extraction
    - Individual network layers
    """

    def __init__(
        self,
        policy: nn.Module,
        profiler: Optional[RuntimeProfiler] = None,
        instrument_modules: bool = True,
    ):
        """Initialize profiled policy wrapper.

        Args:
            policy: Policy to wrap
            profiler: RuntimeProfiler instance (if None, no profiling is done)
            instrument_modules: If True, add hooks to all submodules for detailed profiling
        """
        super().__init__()
        self.policy = policy
        self.profiler = profiler
        self._instrumented_modules: Set[str] = set()

        # Instrument submodules if requested
        if instrument_modules and profiler is not None:
            self._instrument_submodules()

    def _instrument_submodules(self):
        """Add forward hooks to all submodules for detailed profiling."""

        def create_hook(module_name: str):
            """Create a forward hook for a specific module."""

            def hook(module, input, output):
                # Hook is called after forward pass, so no profiling needed here
                pass

            return hook

        # Add forward pre-hooks to track module timing
        for name, module in self.policy.named_modules():
            if name and name not in self._instrumented_modules:
                # Skip the root module
                if name == "":
                    continue

                # Wrap the module's forward method
                original_forward = module.forward
                module_component_name = f"module.{name}"

                @functools.wraps(original_forward)
                def profiled_forward(
                    *args, _orig=original_forward, _name=module_component_name, **kwargs
                ):
                    if self.profiler is not None:
                        with self.profiler.component(_name):
                            return _orig(*args, **kwargs)
                    else:
                        return _orig(*args, **kwargs)

                module.forward = profiled_forward
                self._instrumented_modules.add(name)

    def forward(self, *args, **kwargs):
        """Forward pass with profiling."""
        if self.profiler is None:
            return self.policy.forward(*args, **kwargs)

        with self.profiler.component("policy.forward"):
            return self.policy.forward(*args, **kwargs)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Predict action with profiling."""
        if self.profiler is None:
            return self.policy.predict(observation, state, episode_start, deterministic)

        with self.profiler.component("policy.predict"):
            return self.policy.predict(observation, state, episode_start, deterministic)

    def evaluate_actions(self, obs, actions):
        """Evaluate actions with profiling."""
        if self.profiler is None:
            return self.policy.evaluate_actions(obs, actions)

        with self.profiler.component("policy.evaluate_actions"):
            return self.policy.evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        """Get action distribution with profiling."""
        if self.profiler is None:
            return self.policy.get_distribution(obs)

        with self.profiler.component("policy.get_distribution"):
            return self.policy.get_distribution(obs)

    def predict_values(self, obs):
        """Predict values with profiling."""
        if self.profiler is None:
            return self.policy.predict_values(obs)

        with self.profiler.component("policy.predict_values"):
            return self.policy.predict_values(obs)

    def __getattr__(self, name):
        """Delegate attribute access to wrapped policy."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)

    def __setattr__(self, name, value):
        """Delegate attribute setting."""
        if name in ["policy", "profiler", "_training_step_active"]:
            super().__setattr__(name, value)
        else:
            setattr(self.policy, name, value)


def wrap_policy_with_profiler(
    model, profiler: Optional[RuntimeProfiler] = None, instrument_modules: bool = True
) -> None:
    """Wrap a model's policy with profiling instrumentation.

    This modifies the model in-place to add profiling.

    Args:
        model: SB3 model (PPO, etc.)
        profiler: RuntimeProfiler instance
        instrument_modules: If True, instrument all submodules for detailed profiling
    """
    if profiler is None or not hasattr(model, "policy"):
        return

    # Wrap the policy with module-level instrumentation
    model.policy = ProfiledPolicyWrapper(
        model.policy, profiler, instrument_modules=instrument_modules
    )
