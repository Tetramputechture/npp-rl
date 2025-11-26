"""Comprehensive runtime profiler for training analysis.

Tracks time spent in different phases and components of training to help
identify bottlenecks and optimize performance.
"""

import json
import logging
import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

logger = logging.getLogger(__name__)


@dataclass
class PhaseMetrics:
    """Metrics for a single phase of training."""

    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    sub_phases: Dict[str, "PhaseMetrics"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> float:
        """Mark phase as finished and return duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


@dataclass
class ComponentTiming:
    """Timing information for a specific component."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0

    def add_timing(self, duration: float) -> None:
        """Add a timing measurement."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.call_count


class RuntimeProfiler:
    """Comprehensive runtime profiler for training analysis.

    Tracks:
    - High-level phases (pretraining, training, evaluation)
    - Component-level timings (model forward/backward, env steps, etc.)
    - PyTorch profiler traces for detailed GPU/CPU analysis
    - Memory usage statistics
    """

    def __init__(
        self,
        output_dir: Path,
        enable_pytorch_profiler: bool = True,
        pytorch_profiler_wait: int = 1,
        pytorch_profiler_warmup: int = 1,
        pytorch_profiler_active: int = 3,
        pytorch_profiler_repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        """Initialize runtime profiler.

        Args:
            output_dir: Directory to save profiling results
            enable_pytorch_profiler: Enable PyTorch profiler for detailed traces
            pytorch_profiler_wait: Steps to wait before profiling
            pytorch_profiler_warmup: Steps to warmup before profiling
            pytorch_profiler_active: Steps to actively profile
            pytorch_profiler_repeat: Number of profiling cycles
            record_shapes: Record tensor shapes in profiler
            profile_memory: Track memory usage in profiler
            with_stack: Include Python stack traces (expensive)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_pytorch_profiler = enable_pytorch_profiler
        self.pytorch_profiler_config = {
            "wait": pytorch_profiler_wait,
            "warmup": pytorch_profiler_warmup,
            "active": pytorch_profiler_active,
            "repeat": pytorch_profiler_repeat,
        }

        # Phase tracking
        self.phases: Dict[str, PhaseMetrics] = {}
        self.active_phases: List[str] = []
        self.phase_lock = Lock()

        # Component timing
        self.component_timings: Dict[str, ComponentTiming] = {}
        self.component_lock = Lock()

        # PyTorch profiler
        self.pytorch_profiler: Optional[profile] = None
        self.profiler_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            self.profiler_activities.append(ProfilerActivity.CUDA)

        self.profiler_options = {
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
        }

        # Memory tracking
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.peak_memory: Dict[str, float] = {}

        # Track if profiling data has been saved (to avoid duplicate saves)
        self._saved = False
        self._save_lock = Lock()

        logger.info(f"Runtime profiler initialized: output_dir={output_dir}")
        logger.info(f"PyTorch profiler: {enable_pytorch_profiler}")

    @contextmanager
    def phase(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking a phase.

        Args:
            name: Phase name (e.g., "pretraining", "training", "evaluation")
            metadata: Optional metadata to attach to phase

        Example:
            with profiler.phase("training", {"total_timesteps": 1000000}):
                trainer.train(...)
        """
        with self.phase_lock:
            if name in self.phases:
                logger.warning(f"Phase '{name}' already exists, creating nested phase")
                nested_name = f"{name}_nested_{len(self.active_phases)}"
                name = nested_name

            phase = PhaseMetrics(
                name=name,
                start_time=time.time(),
                metadata=metadata or {},
            )
            self.phases[name] = phase
            self.active_phases.append(name)

            # If there's a parent phase, add as sub-phase
            if len(self.active_phases) > 1:
                parent_name = self.active_phases[-2]
                self.phases[parent_name].sub_phases[name] = phase

        try:
            yield phase
        finally:
            with self.phase_lock:
                duration = phase.finish()
                self.active_phases.pop()
                logger.debug(f"Phase '{name}' completed in {duration:.2f}s")

    @contextmanager
    def component(self, name: str):
        """Context manager for tracking component timings.

        Args:
            name: Component name (e.g., "model_forward", "env_step")

        Example:
            with profiler.component("model_forward"):
                output = model(obs)
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self.component_lock:
                if name not in self.component_timings:
                    self.component_timings[name] = ComponentTiming(name=name)
                self.component_timings[name].add_timing(duration)

    def start_pytorch_profiler(self, trace_name: str = "training_trace") -> None:
        """Start PyTorch profiler for detailed GPU/CPU analysis.

        Args:
            trace_name: Name for the trace (used in output filename)
        """
        if not self.enable_pytorch_profiler:
            return

        if self.pytorch_profiler is not None:
            logger.warning("PyTorch profiler already running, stopping previous one")
            self.stop_pytorch_profiler()

        trace_dir = self.output_dir / "pytorch_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        schedule_config = schedule(
            wait=self.pytorch_profiler_config["wait"],
            warmup=self.pytorch_profiler_config["warmup"],
            active=self.pytorch_profiler_config["active"],
            repeat=self.pytorch_profiler_config["repeat"],
        )

        self.pytorch_profiler = profile(
            activities=self.profiler_activities,
            schedule=schedule_config,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir), trace_name),
            **self.profiler_options,
        )

        self.pytorch_profiler.start()
        logger.info(f"PyTorch profiler started: trace_name={trace_name}")

    def stop_pytorch_profiler(self) -> None:
        """Stop PyTorch profiler and save results."""
        if self.pytorch_profiler is not None:
            self.pytorch_profiler.stop()
            self.pytorch_profiler = None
            logger.info("PyTorch profiler stopped")

    def step_pytorch_profiler(self) -> None:
        """Step PyTorch profiler (call during training loop)."""
        if self.pytorch_profiler is not None:
            self.pytorch_profiler.step()

    def record_memory_snapshot(self, label: str) -> None:
        """Record current memory usage snapshot.

        Args:
            label: Label for this snapshot (e.g., "after_model_init")
        """
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "cpu_memory_mb": self._get_cpu_memory_mb(),
        }

        if torch.cuda.is_available():
            snapshot["gpu_memory_mb"] = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e6
                reserved = torch.cuda.memory_reserved(i) / 1e6
                # Also track peak allocated memory (not just reserved)
                peak_allocated = torch.cuda.max_memory_allocated(i) / 1e6
                snapshot["gpu_memory_mb"][f"gpu_{i}"] = {
                    "allocated": allocated,
                    "reserved": reserved,
                    "peak_allocated": peak_allocated,
                }
                # Track peak reserved memory
                peak_reserved_key = f"gpu_{i}_peak_reserved"
                if peak_reserved_key not in self.peak_memory:
                    self.peak_memory[peak_reserved_key] = reserved
                else:
                    self.peak_memory[peak_reserved_key] = max(
                        self.peak_memory[peak_reserved_key], reserved
                    )
                # Track peak allocated memory
                peak_allocated_key = f"gpu_{i}_peak_allocated"
                if peak_allocated_key not in self.peak_memory:
                    self.peak_memory[peak_allocated_key] = peak_allocated
                else:
                    self.peak_memory[peak_allocated_key] = max(
                        self.peak_memory[peak_allocated_key], peak_allocated
                    )

        self.memory_snapshots.append(snapshot)

    def reset_peak_memory_stats(self) -> None:
        """Reset PyTorch peak memory statistics for per-phase tracking.

        This is useful to measure peak memory for specific phases by calling
        reset before the phase and recording a snapshot after.
        """
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            logger.debug("Reset peak memory statistics for all GPUs")

    def _get_cpu_memory_mb(self) -> float:
        """Get current CPU memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1e6
        except ImportError:
            return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary of profiling results.

        Returns:
            Dictionary containing profiling summary
        """
        summary = {
            "phases": self._summarize_phases(),
            "components": self._summarize_components(),
            "memory": {
                "snapshots": self.memory_snapshots,
                "peak_memory": self.peak_memory,
            },
            "total_time": sum(
                p.duration or 0.0 for p in self.phases.values() if p.duration
            ),
        }

        # Add PyTorch profiler function-level statistics if available
        pytorch_stats = self._extract_pytorch_profiler_stats()
        if pytorch_stats:
            summary["pytorch_function_stats"] = pytorch_stats

        return summary

    def _summarize_phases(self) -> Dict[str, Any]:
        """Summarize phase metrics."""
        summary = {}
        for name, phase in self.phases.items():
            if phase.duration is not None:
                summary[name] = {
                    "duration": phase.duration,
                    "start_time": phase.start_time,
                    "end_time": phase.end_time,
                    "metadata": phase.metadata,
                    "sub_phases": {
                        sub_name: {
                            "duration": sub_phase.duration,
                            "metadata": sub_phase.metadata,
                        }
                        for sub_name, sub_phase in phase.sub_phases.items()
                        if sub_phase.duration is not None
                    },
                }
        return summary

    def _summarize_components(self) -> Dict[str, Any]:
        """Summarize component timings."""
        summary = {}
        for name, timing in self.component_timings.items():
            summary[name] = {
                "total_time": timing.total_time,
                "call_count": timing.call_count,
                "avg_time": timing.avg_time,
                "min_time": timing.min_time if timing.min_time != float("inf") else 0.0,
                "max_time": timing.max_time,
                "percentage_of_total": 0.0,  # Will be calculated later
            }

        # Calculate percentages
        total_component_time = sum(
            t.total_time for t in self.component_timings.values()
        )
        if total_component_time > 0:
            for name in summary:
                summary[name]["percentage_of_total"] = (
                    summary[name]["total_time"] / total_component_time * 100
                )

        return summary

    def save_summary(
        self, filename: str = "profiling_summary.json", force: bool = False
    ) -> Optional[Path]:
        """Save profiling summary to JSON file.

        Args:
            filename: Output filename
            force: Force save even if already saved (default: False)

        Returns:
            Path to saved file, or None if already saved and force=False
        """
        with self._save_lock:
            if self._saved and not force:
                logger.debug("Profiling summary already saved, skipping")
                return None

            try:
                # Finish any active phases before saving
                for phase_name, phase in list(self.phases.items()):
                    if phase.duration is None:
                        phase.finish()
                        logger.debug(
                            f"Finished incomplete phase '{phase_name}' before save"
                        )

                summary = self.get_summary()
                output_path = self.output_dir / filename

                # Convert to JSON-serializable format
                def convert_to_dict(obj):
                    if isinstance(obj, PhaseMetrics):
                        return asdict(obj)
                    elif isinstance(obj, ComponentTiming):
                        return asdict(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_dict(item) for item in obj]
                    else:
                        return obj

                summary_serializable = convert_to_dict(summary)

                # Add metadata about save status
                summary_serializable["save_metadata"] = {
                    "saved_at": time.time(),
                    "incomplete": any(p.duration is None for p in self.phases.values()),
                    "active_phases": self.active_phases.copy(),
                }

                with open(output_path, "w") as f:
                    json.dump(summary_serializable, f, indent=2, default=str)

                self._saved = True
                logger.info(f"Profiling summary saved to {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Failed to save profiling summary: {e}", exc_info=True)
                return None

    def print_summary(self) -> None:
        """Print human-readable profiling summary."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("RUNTIME PROFILING SUMMARY")
        print("=" * 70)

        # Phase summary
        print("\n📊 PHASES:")
        phases = summary["phases"]
        total_time = summary["total_time"]

        for name, phase_info in sorted(
            phases.items(), key=lambda x: x[1]["duration"], reverse=True
        ):
            duration = phase_info["duration"]
            percentage = (duration / total_time * 100) if total_time > 0 else 0.0
            print(f"  {name:30s} {duration:8.2f}s ({percentage:5.1f}%)")
            if phase_info.get("sub_phases"):
                for sub_name, sub_info in phase_info["sub_phases"].items():
                    sub_duration = sub_info["duration"]
                    sub_pct = (sub_duration / duration * 100) if duration > 0 else 0.0
                    print(
                        f"    └─ {sub_name:26s} {sub_duration:8.2f}s ({sub_pct:5.1f}%)"
                    )

        # Component summary
        print("\n⚙️  COMPONENTS:")
        components = summary["components"]
        total_component_time = sum(c["total_time"] for c in components.values())

        for name, comp_info in sorted(
            components.items(), key=lambda x: x[1]["total_time"], reverse=True
        ):
            total = comp_info["total_time"]
            count = comp_info["call_count"]
            avg = comp_info["avg_time"]
            pct = comp_info["percentage_of_total"]
            print(
                f"  {name:30s} {total:8.2f}s ({pct:5.1f}%) "
                f"[{count} calls, avg: {avg * 1000:.2f}ms]"
            )

        # Memory summary
        print("\n💾 MEMORY:")
        if summary["memory"]["peak_memory"]:
            for key, value in summary["memory"]["peak_memory"].items():
                print(f"  {key:30s} {value:8.2f} MB")

        print("\n" + "=" * 70)

    def export_trace(self, trace_name: str = "training_trace") -> Optional[Path]:
        """Export PyTorch profiler trace for visualization.

        Args:
            trace_name: Name for the trace

        Returns:
            Path to trace file if available, None otherwise
        """
        if not self.enable_pytorch_profiler or self.pytorch_profiler is None:
            return None

        trace_dir = self.output_dir / "pytorch_traces"
        trace_files = list(trace_dir.glob(f"{trace_name}*.pt.trace.json"))

        if trace_files:
            logger.info(f"PyTorch trace files available in {trace_dir}")
            return trace_dir

        return None

    def _extract_pytorch_profiler_stats(self) -> Optional[Dict[str, Any]]:
        """Extract function-level statistics from PyTorch profiler.

        Returns:
            Dictionary with function-level timing statistics, or None if unavailable
        """
        if not self.enable_pytorch_profiler or self.pytorch_profiler is None:
            return None

        try:
            # Get profiler function events
            function_events = self.pytorch_profiler.key_averages()

            # Extract top functions by CPU time
            cpu_stats = []
            cuda_stats = []

            for event in function_events:
                # Skip very short events (< 0.1ms)
                if event.cpu_time_total < 100:  # microseconds
                    continue

                cpu_stats.append(
                    {
                        "name": event.key,
                        "cpu_time_total_ms": event.cpu_time_total
                        / 1000.0,  # Convert to ms
                        "cpu_time_avg_ms": (event.cpu_time_total / event.count) / 1000.0
                        if event.count > 0
                        else 0,
                        "count": event.count,
                        "percentage": 0.0,  # Will be calculated below
                    }
                )

                if event.cuda_time_total > 0:
                    cuda_stats.append(
                        {
                            "name": event.key,
                            "cuda_time_total_ms": event.cuda_time_total / 1000.0,
                            "cuda_time_avg_ms": (event.cuda_time_total / event.count)
                            / 1000.0
                            if event.count > 0
                            else 0,
                            "count": event.count,
                            "percentage": 0.0,
                        }
                    )

            # Calculate percentages
            if cpu_stats:
                total_cpu_time = sum(s["cpu_time_total_ms"] for s in cpu_stats)
                for stat in cpu_stats:
                    stat["percentage"] = (
                        (stat["cpu_time_total_ms"] / total_cpu_time * 100)
                        if total_cpu_time > 0
                        else 0
                    )

                # Sort by total time descending
                cpu_stats.sort(key=lambda x: x["cpu_time_total_ms"], reverse=True)

            if cuda_stats:
                total_cuda_time = sum(s["cuda_time_total_ms"] for s in cuda_stats)
                for stat in cuda_stats:
                    stat["percentage"] = (
                        (stat["cuda_time_total_ms"] / total_cuda_time * 100)
                        if total_cuda_time > 0
                        else 0
                    )

                cuda_stats.sort(key=lambda x: x["cuda_time_total_ms"], reverse=True)

            result = {}
            if cpu_stats:
                result["cpu_top_functions"] = cpu_stats[:50]  # Top 50 functions
            if cuda_stats:
                result["cuda_top_functions"] = cuda_stats[:50]

            return result if result else None

        except Exception as e:
            logger.warning(f"Could not extract PyTorch profiler stats: {e}")
            return None


# Global registry of active profilers for signal handling
_active_profilers: List[RuntimeProfiler] = []
_profiler_lock = Lock()


def register_profiler(profiler: RuntimeProfiler) -> None:
    """Register a profiler for signal handling.

    Args:
        profiler: RuntimeProfiler instance to register
    """
    if profiler is None:
        return

    with _profiler_lock:
        if profiler not in _active_profilers:
            _active_profilers.append(profiler)
            logger.debug(f"Registered profiler: {profiler.output_dir}")


def unregister_profiler(profiler: RuntimeProfiler) -> None:
    """Unregister a profiler from signal handling.

    Args:
        profiler: RuntimeProfiler instance to unregister
    """
    if profiler is None:
        return

    with _profiler_lock:
        if profiler in _active_profilers:
            _active_profilers.remove(profiler)
            logger.debug(f"Unregistered profiler: {profiler.output_dir}")


def save_all_active_profilers(force: bool = True) -> int:
    """Save all active profilers.

    Args:
        force: Force save even if already saved

    Returns:
        Number of profilers successfully saved
    """
    with _profiler_lock:
        profilers_to_save = list(_active_profilers)

    if not profilers_to_save:
        logger.debug("No active profilers to save")
        return 0

    logger.info(f"Saving {len(profilers_to_save)} active profiler(s)...")
    saved_count = 0

    for profiler in profilers_to_save:
        try:
            logger.info(f"  → Saving profiler: {profiler.output_dir}")
            profiler.save_summary(force=force)
            if profiler.enable_pytorch_profiler:
                profiler.stop_pytorch_profiler()
            saved_count += 1
        except Exception as e:
            logger.error(
                f"  ✗ Error saving profiler {profiler.output_dir}: {e}", exc_info=True
            )

    logger.info(
        f"Successfully saved {saved_count}/{len(profilers_to_save)} profiler(s)"
    )
    return saved_count


def setup_profiler_signal_handlers(profiler: RuntimeProfiler = None) -> None:
    """Setup signal handlers to save profiling data on interruption.

    This function sets up signal handlers that will save ALL active profilers
    when the process receives SIGINT or SIGTERM. If called multiple times,
    it will only set up handlers once.

    Args:
        profiler: RuntimeProfiler instance to register (optional, for backwards compatibility)
    """
    # Register the profiler if provided
    if profiler is not None:
        register_profiler(profiler)

    # Only install signal handlers once
    if hasattr(setup_profiler_signal_handlers, "_installed"):
        return

    def save_on_signal(signum, frame):
        """Save profiling data for all active profilers when signal is received."""
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"Signal {signum}"

        logger.warning(
            f"Received {signal_name} signal - saving profiling data for all active profilers..."
        )

        save_all_active_profilers(force=True)

        # Re-raise the signal to allow normal cleanup
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Register handlers for common interrupt signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, save_on_signal)
            logger.debug(f"Registered signal handler for {sig}")
        except (ValueError, OSError) as e:
            # Signal may not be available on all platforms
            logger.debug(f"Could not register signal handler for {sig}: {e}")

    # Mark as installed
    setup_profiler_signal_handlers._installed = True


def create_profiler(
    output_dir: Path,
    enable: bool = True,
    enable_pytorch_profiler: bool = True,
    setup_signal_handlers: bool = True,
    **kwargs,
) -> Optional[RuntimeProfiler]:
    """Create a runtime profiler instance.

    Args:
        output_dir: Output directory for profiling results
        enable: Whether to enable profiling
        enable_pytorch_profiler: Whether to enable PyTorch profiler
        setup_signal_handlers: Whether to setup signal handlers for graceful shutdown
        **kwargs: Additional arguments passed to RuntimeProfiler

    Returns:
        RuntimeProfiler instance if enabled, None otherwise
    """
    if not enable:
        return None

    profiler = RuntimeProfiler(
        output_dir=output_dir,
        enable_pytorch_profiler=enable_pytorch_profiler,
        **kwargs,
    )

    if setup_signal_handlers:
        setup_profiler_signal_handlers(profiler)

    return profiler
