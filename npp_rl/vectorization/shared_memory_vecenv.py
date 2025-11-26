"""
Shared memory vectorized environment for zero-copy observation transfer.

This eliminates pickle serialization overhead by writing observations directly
to shared memory arrays, enabling efficient scaling to 128+ parallel environments.
"""

import logging
import time
import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

logger = logging.getLogger(__name__)


class SharedMemoryObservationWrapper:
    """
    Zero-copy observation transfer using shared memory.

    Eliminates pickle serialization by pre-allocating shared memory arrays
    for all observation components. Workers write directly to shared memory,
    main process reads without copying.

    Usage:
        shared_obs = SharedMemoryObservationWrapper(observation_space, num_envs)
        # In worker: shared_obs.write_observation(env_idx, obs_dict)
        # In main: obs_dict = shared_obs.read_observations()
    """

    # Lookup table for dtype to ctype mapping (class-level constant for efficiency)
    _DTYPE_TO_CTYPE = {
        np.dtype(np.float32): ("f", 4),
        np.dtype(np.float64): ("d", 8),
        np.dtype(np.int32): ("i", 4),
        np.dtype(np.int64): ("q", 8),
        np.dtype(np.uint8): ("B", 1),
        np.dtype(np.bool_): ("b", 1),
    }

    def __init__(self, observation_space: spaces.Dict, num_envs: int):
        """
        Initialize shared memory arrays for observations.

        Args:
            observation_space: Dictionary observation space
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.observation_space = observation_space

        # Pre-allocate shared memory for each observation key
        self.shared_arrays = {}
        self.shapes = {}
        self.dtypes = {}

        logger.info(f"Creating shared memory for {num_envs} environments")

        # Only calculate debug info if debug logging is enabled
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        debug_info = [] if debug_enabled else None

        # print timnigs
        for key, space in observation_space.spaces.items():
            # Shape includes batch dimensionq
            shape = (num_envs,) + space.shape
            dtype = space.dtype

            # Calculate total size
            size = int(np.prod(shape))

            # Map numpy dtype to ctypes using lookup table
            dtype_normalized = np.dtype(dtype)
            if dtype_normalized in self._DTYPE_TO_CTYPE:
                ctype, itemsize = self._DTYPE_TO_CTYPE[dtype_normalized]
            else:
                # Fallback to bytes
                ctype = "b"
                itemsize = dtype_normalized.itemsize

            # Create shared memory array (lock=False for performance)
            try:
                shared_array = mp.Array(ctype, size, lock=False)
                self.shared_arrays[key] = shared_array
                self.shapes[key] = shape
                self.dtypes[key] = dtype

                # Only calculate size for debug logging if enabled
                if debug_enabled:
                    size_mb = (size * itemsize) / (1024 * 1024)
                    debug_info.append(f"  {key}: {shape} {dtype} = {size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to create shared memory for {key}: {e}")
                raise

        # Log debug info in batch if enabled
        if debug_enabled and debug_info:
            logger.debug("\n".join(debug_info))

        # Calculate total only once at the end
        total_mb = sum(
            np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
            for shape, dtype in zip(self.shapes.values(), self.dtypes.values())
        )
        logger.info(f"Total shared memory allocated: {total_mb:.2f} MB")

    def write_observation(self, env_idx: int, obs_dict: Dict[str, np.ndarray]) -> None:
        """
        Write observation from worker to shared memory.

        Args:
            env_idx: Environment index (0 to num_envs-1)
            obs_dict: Observation dictionary from environment
        """
        for key, value in obs_dict.items():
            if key in self.shared_arrays:
                # Convert shared array to numpy view
                np_array = np.frombuffer(
                    self.shared_arrays[key], dtype=self.dtypes[key]
                )
                np_array = np_array.reshape(self.shapes[key])

                # Write to this environment's slice
                try:
                    np_array[env_idx] = value
                except Exception as e:
                    logger.error(f"Failed to write {key} for env {env_idx}: {e}")
                    logger.error(
                        f"  Expected shape: {self.shapes[key][1:]}, got: {value.shape}"
                    )
                    raise

    def read_observations(self) -> Dict[str, np.ndarray]:
        """
        Read all observations from shared memory (main process).

        Returns:
            Dictionary of observation arrays with batch dimension
        """
        obs_dict = {}
        for key, shared_array in self.shared_arrays.items():
            # Convert to numpy array (zero-copy view)
            np_array = np.frombuffer(shared_array, dtype=self.dtypes[key])
            obs_dict[key] = np_array.reshape(self.shapes[key])
        return obs_dict

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for worker processes."""
        return {
            "shapes": self.shapes,
            "dtypes": {k: np.dtype(v).name for k, v in self.dtypes.items()},
        }


def _worker_shared_memory(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    shared_memory: SharedMemoryObservationWrapper,
    env_idx: int,
) -> None:
    """
    Worker process for SharedMemorySubprocVecEnv.

    Writes observations directly to shared memory instead of returning via pipe.
    """
    import time
    import random
    import numpy as np

    _worker_start = time.perf_counter()
    print(f"[PROFILE] Worker {env_idx} starting...")

    parent_remote.close()

    # CRITICAL: Reseed RNGs in forked processes to avoid identical sequences
    # When using fork, all workers inherit the same RNG state from parent
    seed = env_idx + int(time.time() * 1000) % 100000
    random.seed(seed)
    np.random.seed(seed)

    # If PyTorch is available, reseed it too
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        _env_create_start = time.perf_counter()
        env = env_fn_wrapper.var()
        _env_create_time = time.perf_counter() - _env_create_start
        print(f"[PROFILE] Worker {env_idx} env creation: {_env_create_time:.3f}s")
    except Exception as e:
        # Send error back to main process so it doesn't hang
        import traceback

        error_msg = f"Worker {env_idx}: Failed to create environment: {e}\n{traceback.format_exc()}"
        try:
            remote.send(("error", error_msg))
        except Exception:
            pass  # Ignore errors sending error message
        remote.close()
        return

    _worker_init = time.perf_counter() - _worker_start
    print(f"[PROFILE] Worker {env_idx} initialized: {_worker_init:.3f}s")

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)

                # Handle episode statistics wrapping for SB3 compatibility
                # When episode ends, wrap episode-level stats in "episode" dict
                done = terminated or truncated
                if done:
                    # CRITICAL: Ensure episode statistics are isolated per worker
                    # Each worker has its own env instance and info dict - no sharing
                    # The info dict comes from either:
                    #   1. FrameSkipWrapper.step() which manually sets r/l (lines 219-220)
                    #   2. BaseNppEnvironment.step() which sets r/l via _build_episode_info()

                    # Add worker PID for debugging route visualization contamination
                    import os

                    info["worker_pid"] = os.getpid()

                    # Preserve episode statistics in SB3 format
                    if "episode" not in info:
                        # Create NEW episode dict for SB3 callbacks compatibility
                        # Use explicit float() to ensure value is copied, not referenced
                        episode_info = {}

                        # Get cumulative reward - convert to float for safety
                        # NOTE: r/l should ALWAYS be present (set by wrapper or env)
                        if "r" in info:
                            episode_info["r"] = float(info["r"])  # Explicit copy
                        else:
                            # SHOULD NEVER HAPPEN - indicates bug in wrapper chain
                            logger.error(
                                f"Worker {env_idx}: Episode ended but 'r' not in info! "
                                f"This indicates a bug in the environment/wrapper. "
                                f"Available keys: {list(info.keys())}"
                            )
                            episode_info["r"] = 0.0

                        # Get episode length - convert to int for safety
                        if "l" in info:
                            episode_info["l"] = int(info["l"])  # Explicit copy
                        else:
                            # SHOULD NEVER HAPPEN - indicates bug in wrapper chain
                            logger.error(
                                f"Worker {env_idx}: Episode ended but 'l' not in info! "
                                f"This indicates a bug in the environment/wrapper. "
                                f"Available keys: {list(info.keys())}"
                            )
                            episode_info["l"] = 0

                        info["episode"] = episode_info
                    else:
                        # Episode dict already exists - validate it has required keys
                        if "r" not in info["episode"] or "l" not in info["episode"]:
                            logger.error(
                                f"Worker {env_idx}: info['episode'] exists but missing 'r' or 'l'. "
                                f"Keys: {list(info['episode'].keys())}"
                            )

                # CRITICAL: Auto-reset environment when episode ends (Gymnasium behavior)
                # This ensures position tracking and other episode-level state is properly reset
                if done:
                    # Reset environment and get new initial observation
                    observation, reset_info = env.reset()

                # Write observation to shared memory instead of sending via pipe
                shared_memory.write_observation(env_idx, observation)

                # Only send small metadata via pipe (each worker has its own pipe)
                # Rewards and info dicts are NOT shared between workers
                remote.send((reward, done, info))

            elif cmd == "reset":
                kwargs = {} if data is None else data
                observation, info = env.reset(**kwargs)

                # PERFORMANCE FIX: Clear module-level caches periodically
                # to prevent unbounded growth in long-running workers
                # This fixes episodic performance degradation with 64+ environments
                if hasattr(env, "_episode_count"):
                    if env._episode_count % 500 == 0:
                        # Clear module-level pathfinding caches
                        try:
                            from nclone.graph.reachability.pathfinding_utils import (
                                clear_surface_area_cache,
                            )

                            clear_surface_area_cache()
                        except Exception:
                            pass  # Graceful degradation if module not available

                        # Force garbage collection every 100 episodes
                        # This helps reclaim memory from temporary numpy arrays and graph structures
                        import gc

                        gc.collect()

                # Write observation to shared memory
                shared_memory.write_observation(env_idx, observation)

                # Send confirmation via pipe
                remote.send(info)

            elif cmd == "close":
                env.close()
                remote.close()
                break

            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))

            elif cmd == "env_method":
                try:
                    method = getattr(env, data[0])
                    result = method(*data[1], **data[2])
                    remote.send(result)
                except Exception as e:
                    # Send exception back to main process instead of crashing
                    remote.send(e)

            elif cmd == "get_attr":
                try:
                    result = getattr(env, data)
                    remote.send(result)
                except Exception as e:
                    remote.send(e)

            elif cmd == "set_attr":
                try:
                    setattr(env, data[0], data[1])
                    remote.send(None)  # Success
                except Exception as e:
                    remote.send(e)

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

        except EOFError:
            break


class SharedMemorySubprocVecEnv(VecEnv):
    """
    SubprocVecEnv that uses shared memory for zero-copy observation transfer.

    Eliminates pickle serialization bottleneck by writing observations directly
    to shared memory arrays. Only small metadata (rewards, dones) is sent via pipes.

    This enables efficient scaling to 128+ parallel environments without the
    exponential degradation seen with standard SubprocVecEnv.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], Any]],
        shared_memory: Optional[SharedMemoryObservationWrapper] = None,
        start_method: Optional[str] = None,
    ):
        """
        Initialize shared memory vectorized environment.

        Args:
            env_fns: List of environment factory functions
            shared_memory: Pre-allocated shared memory wrapper, or None to create
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        _init_start = time.perf_counter()
        print(
            f"[PROFILE] SharedMemorySubprocVecEnv.__init__ starting with {len(env_fns)} workers..."
        )

        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # Create temporary environment to get spaces
        _dummy_start = time.perf_counter()
        print(f"[PROFILE] Creating dummy env to get observation/action spaces...")
        dummy_env = env_fns[0]()
        _dummy_create = time.perf_counter()
        print(f"[PROFILE] Dummy env creation: {_dummy_create - _dummy_start:.3f}s")

        observation_space = dummy_env.observation_space
        action_space = dummy_env.action_space
        dummy_env.close()
        _dummy_total = time.perf_counter() - _dummy_start
        print(f"[PROFILE] Dummy env total (create+close): {_dummy_total:.3f}s")

        # Create shared memory if not provided
        _shmem_start = time.perf_counter()
        if shared_memory is None:
            print(f"[PROFILE] Creating shared memory for observations...")
            shared_memory = SharedMemoryObservationWrapper(observation_space, n_envs)
        _shmem_time = time.perf_counter() - _shmem_start
        print(f"[PROFILE] Shared memory setup: {_shmem_time:.3f}s")

        self.shared_memory = shared_memory

        # Validate observation space is Dict
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                f"SharedMemorySubprocVecEnv requires Dict observation space, "
                f"got {type(observation_space)}"
            )

        VecEnv.__init__(self, n_envs, observation_space, action_space)

        # Determine start method
        if start_method is None:
            # Use fork on Linux for fast startup (~instant vs 6s per worker with spawn)
            # Use spawn on other platforms where fork isn't available
            import sys

            if sys.platform == "linux":
                start_method = "fork"
                print(f"[PROFILE] Using 'fork' start method on Linux (fast startup)")
            else:
                start_method = "spawn"
                print(f"[PROFILE] Using 'spawn' start method on {sys.platform}")

        ctx = mp.get_context(start_method)

        # Create pipes and processes
        _spawn_start = time.perf_counter()
        print(f"[PROFILE] Creating pipes and spawning {n_envs} worker processes...")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn, env_idx in zip(
            self.work_remotes, self.remotes, env_fns, range(n_envs)
        ):
            args = (
                work_remote,
                remote,
                CloudpickleWrapper(env_fn),
                shared_memory,
                env_idx,
            )
            _proc_start = time.perf_counter()
            process = ctx.Process(target=_worker_shared_memory, args=args, daemon=True)
            process.start()
            _proc_time = time.perf_counter() - _proc_start
            print(f"[PROFILE]   Worker {env_idx} spawned: {_proc_time:.3f}s")
            self.processes.append(process)
            work_remote.close()

        _spawn_time = time.perf_counter() - _spawn_start
        print(f"[PROFILE] All {n_envs} workers spawned: {_spawn_time:.3f}s")

        # Check for worker initialization errors
        # Workers should be ready to receive commands now
        # If any failed to initialize, they would have sent an error
        print(f"[PROFILE] Waiting for workers to initialize...")
        time.sleep(0.1)  # Brief pause to let workers initialize
        for i, remote in enumerate(self.remotes):
            if not self.processes[i].is_alive():
                raise RuntimeError(f"Worker {i} died during initialization")

        _init_time = time.perf_counter() - _init_start
        print(
            f"[PROFILE] SharedMemorySubprocVecEnv.__init__ COMPLETE: {_init_time:.3f}s"
        )

    def step_async(self, actions: np.ndarray) -> None:
        """Send step commands to workers."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for step results and read observations from shared memory.

        Returns:
            observations, rewards, dones, infos
        """
        # Receive small metadata from pipes
        # CRITICAL: Each worker sends via its own pipe - no cross-contamination
        # results is a list of (reward, done, info) tuples, one per worker
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        rewards, dones, infos = zip(*results)

        # Read observations from shared memory (zero-copy)
        obs_dict = self.shared_memory.read_observations()

        # SAFETY CHECK: Verify info dicts are independent (in debug mode)
        # This ensures no accidental sharing of episode rewards between workers
        if __debug__:
            # Check that info objects are distinct (not the same object reference)
            info_ids = [id(info) for info in infos]
            if len(info_ids) != len(set(info_ids)):
                logger.error(
                    f"CRITICAL: Info dict objects are not unique! "
                    f"This indicates potential cross-contamination. "
                    f"Unique IDs: {len(set(info_ids))} vs Total: {len(info_ids)}"
                )

        return obs_dict, np.array(rewards), np.array(dones), list(infos)

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(("reset", None))

        # Receive confirmation from workers
        infos = [remote.recv() for remote in self.remotes]

        # Read observations from shared memory
        obs_dict = self.shared_memory.read_observations()

        return obs_dict

    def close(self) -> None:
        """Close all worker processes."""
        if self.closed:
            return

        if self.waiting:
            # Wait for pending operations
            for remote in self.remotes:
                remote.recv()
            self.waiting = False

        # Send close commands
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass

        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

        # Close pipes
        for remote in self.remotes:
            remote.close()

        # Explicitly clear shared memory arrays to help garbage collection
        # This ensures proper cleanup when destroying vecenvs
        if hasattr(self, "shared_memory") and self.shared_memory is not None:
            # Clear references to shared arrays to help GC
            self.shared_memory.shared_arrays.clear()
            self.shared_memory.shapes.clear()
            self.shared_memory.dtypes.clear()
            if hasattr(self.shared_memory, "observation_space"):
                del self.shared_memory.observation_space

        self.closed = True
        logger.info("Closed shared memory vectorized environment")

    def get_attr(
        self, attr_name: str, indices: Optional[Sequence[int]] = None
    ) -> List[Any]:
        """Get attribute from environments."""
        if indices is None:
            indices = range(self.num_envs)

        for i in indices:
            self.remotes[i].send(("get_attr", attr_name))

        results = []
        for i in indices:
            result = self.remotes[i].recv()
            if isinstance(result, Exception):
                raise result
            results.append(result)

        return results

    def set_attr(
        self, attr_name: str, value: Any, indices: Optional[Sequence[int]] = None
    ) -> None:
        """Set attribute in environments."""
        if indices is None:
            indices = range(self.num_envs)

        for i in indices:
            self.remotes[i].send(("set_attr", (attr_name, value)))

        for i in indices:
            result = self.remotes[i].recv()
            if isinstance(result, Exception):
                raise result

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[Sequence[int]] = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call method on environments."""
        if indices is None:
            indices = range(self.num_envs)

        for i in indices:
            self.remotes[i].send(
                ("env_method", (method_name, method_args, method_kwargs))
            )

        results = []
        for i in indices:
            result = self.remotes[i].recv()
            # If the result is an exception, raise it
            if isinstance(result, Exception):
                raise result
            results.append(result)

        return results

    def env_is_wrapped(
        self, wrapper_class: type, indices: Optional[Sequence[int]] = None
    ) -> List[bool]:
        """Check if environments are wrapped with a given wrapper."""
        if indices is None:
            indices = range(self.num_envs)

        # Use env_method to check if env is wrapped
        results = []
        for i in indices:
            try:
                self.remotes[i].send(
                    ("env_method", ("is_wrapped", (wrapper_class,), {}))
                )
                results.append(self.remotes[i].recv())
            except Exception:
                results.append(False)

        return results
