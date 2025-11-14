"""Comprehensive memory profiler for RL training pipeline.

Tracks both CPU and GPU memory usage throughout training to diagnose
memory leaks and optimize memory usage.
"""

import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Comprehensive memory profiler for training.
    
    Tracks CPU memory via tracemalloc and GPU memory via PyTorch's
    CUDA memory APIs. Provides snapshot comparison and leak detection.
    """

    def __init__(self, output_dir: Path, enable: bool = True):
        """Initialize memory profiler.
        
        Args:
            output_dir: Directory to save profiling reports
            enable: If False, profiler does nothing (no overhead)
        """
        self.output_dir = output_dir / "memory_profiling"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable = enable
        self.snapshots: List[Dict[str, Any]] = []
        self._tracemalloc_started = False

        if enable:
            try:
                tracemalloc.start()
                self._tracemalloc_started = True
                logger.info(f"Memory profiler initialized: {self.output_dir}")
            except Exception as e:
                logger.warning(f"Failed to start tracemalloc: {e}")
                self.enable = False

    def snapshot(self, label: str) -> None:
        """Take CPU + GPU memory snapshot with label.
        
        Args:
            label: Descriptive label for this snapshot
        """
        if not self.enable:
            return

        # CPU memory via tracemalloc
        cpu_snapshot = None
        if self._tracemalloc_started:
            try:
                cpu_snapshot = tracemalloc.take_snapshot()
            except Exception as e:
                logger.warning(f"Failed to take CPU snapshot: {e}")

        # GPU memory stats
        gpu_stats: Dict[str, float] = {}
        if torch.cuda.is_available():
            try:
                gpu_stats = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                }
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")

        snapshot_data = {
            "label": label,
            "timestamp": time.time(),
            "cpu_snapshot": cpu_snapshot,
            "gpu_stats": gpu_stats,
        }

        self.snapshots.append(snapshot_data)
        logger.debug(f"Memory snapshot taken: {label}")

    def compare_snapshots(
        self, label1: str, label2: str, top_n: int = 20
    ) -> None:
        """Compare two snapshots and show top memory growth.
        
        Args:
            label1: Label of first snapshot
            label2: Label of second snapshot
            top_n: Number of top memory consumers to show
        """
        snap1 = next((s for s in self.snapshots if s["label"] == label1), None)
        snap2 = next((s for s in self.snapshots if s["label"] == label2), None)

        if not snap1 or not snap2:
            logger.warning(
                f"Snapshots not found: {label1} or {label2}. "
                f"Available: {[s['label'] for s in self.snapshots]}"
            )
            return

        print(f"\n{'='*70}")
        print(f"Memory Growth: {label1} -> {label2}")
        print(f"{'='*70}")

        # Compare CPU memory
        if (
            snap1["cpu_snapshot"] is not None
            and snap2["cpu_snapshot"] is not None
        ):
            try:
                stats = snap2["cpu_snapshot"].compare_to(
                    snap1["cpu_snapshot"], "lineno"
                )

                print(f"\nTop {top_n} CPU Memory Growth:")
                for i, stat in enumerate(stats[:top_n], 1):
                    print(f"  {i}. {stat}")
            except Exception as e:
                logger.warning(f"Failed to compare CPU snapshots: {e}")

        # GPU comparison
        if snap1["gpu_stats"] and snap2["gpu_stats"]:
            print("\nGPU Memory Changes:")
            for k in ["allocated_gb", "reserved_gb"]:
                if k in snap1["gpu_stats"] and k in snap2["gpu_stats"]:
                    growth = snap2["gpu_stats"][k] - snap1["gpu_stats"][k]
                    print(f"  {k}: {growth:+.3f} GB")

    def detect_leaks(self, threshold_gb: float = 0.1) -> List[Dict[str, Any]]:
        """Detect consistent memory growth patterns.
        
        Args:
            threshold_gb: Minimum growth in GB to consider a leak
            
        Returns:
            List of detected leaks with phase, type, and growth info
        """
        leaks: List[Dict[str, Any]] = []

        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]

            # Check GPU growth
            if curr["gpu_stats"] and prev["gpu_stats"]:
                allocated_growth = (
                    curr["gpu_stats"].get("allocated_gb", 0)
                    - prev["gpu_stats"].get("allocated_gb", 0)
                )
                if allocated_growth > threshold_gb:
                    leaks.append(
                        {
                            "phase": f"{prev['label']} -> {curr['label']}",
                            "growth_gb": allocated_growth,
                            "type": "GPU",
                        }
                    )

            # Check CPU growth
            if (
                curr["cpu_snapshot"] is not None
                and prev["cpu_snapshot"] is not None
            ):
                try:
                    stats = curr["cpu_snapshot"].compare_to(
                        prev["cpu_snapshot"], "lineno"
                    )
                    if stats and stats[0].size_diff / 1e9 > threshold_gb:
                        leaks.append(
                            {
                                "phase": f"{prev['label']} -> {curr['label']}",
                                "growth_gb": stats[0].size_diff / 1e9,
                                "type": "CPU",
                                "location": str(stats[0]),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Failed to detect CPU leak: {e}")

        return leaks

    def save_report(self) -> Path:
        """Save detailed memory report to file.
        
        Returns:
            Path to saved report file
        """
        report_path = self.output_dir / "memory_report.txt"
        leak_path = self.output_dir / "leak_detection.txt"

        with open(report_path, "w") as f:
            f.write("Memory Profiling Report\n")
            f.write("=" * 70 + "\n\n")

            # Write all snapshots
            for snapshot in self.snapshots:
                f.write(f"Label: {snapshot['label']}\n")
                f.write(f"Timestamp: {snapshot['timestamp']}\n")

                if snapshot["gpu_stats"]:
                    f.write("GPU Stats:\n")
                    for k, v in snapshot["gpu_stats"].items():
                        f.write(f"  {k}: {v:.3f} GB\n")

                # Top 10 CPU memory consumers
                if snapshot["cpu_snapshot"] is not None:
                    try:
                        top_stats = snapshot["cpu_snapshot"].statistics("lineno")[
                            :10
                        ]
                        f.write("\nTop 10 CPU Memory Consumers:\n")
                        for stat in top_stats:
                            f.write(f"  {stat}\n")
                    except Exception as e:
                        f.write(f"\nFailed to get CPU stats: {e}\n")

                f.write("\n" + "=" * 70 + "\n\n")

        # Write leak detection results
        leaks = self.detect_leaks()
        with open(leak_path, "w") as f:
            if leaks:
                f.write("POTENTIAL MEMORY LEAKS DETECTED:\n")
                f.write("=" * 70 + "\n")
                for leak in leaks:
                    f.write(f"\nPhase: {leak['phase']}\n")
                    f.write(f"Type: {leak['type']}\n")
                    f.write(f"Growth: {leak['growth_gb']:.3f} GB\n")
                    if "location" in leak:
                        f.write(f"Location: {leak['location']}\n")
            else:
                f.write("No significant memory leaks detected.\n")

        logger.info(f"Memory profiling report saved to {report_path}")
        logger.info(f"Leak detection report saved to {leak_path}")

        return report_path

    def cleanup(self) -> None:
        """Clean up profiler resources."""
        if self._tracemalloc_started:
            try:
                tracemalloc.stop()
                self._tracemalloc_started = False
            except Exception:
                pass

