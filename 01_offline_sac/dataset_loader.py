"""
Dataset Loader
Supports three backends in priority order:
  1. minari  (recommended — pip install minari)
  2. h5py    (raw HDF5 from D4RL S3 bucket)
  3. synthetic (Gaussian noise, for unit-testing the pipeline offline)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from buffer import OfflineReplayBuffer

logger = logging.getLogger(__name__)


# ── raw dict type returned by every loader ───────────────────────────────────
DatasetDict = Dict[str, np.ndarray]


class DatasetLoader:
    """
    Loads a D4RL-compatible dataset and returns an OfflineReplayBuffer.

    Usage
    -----
        loader = DatasetLoader(device="cuda")
        buffer = loader.load("hopper-medium-v2", backend="auto")

    Backend selection
    -----------------
    "auto"   — try minari → h5py → synthetic (in that order)
    "minari" — requires `pip install minari`
    "h5py"   — requires a local .hdf5 file in data/ or a URL
    "synthetic" — generates random data; useful for CI / smoke tests
    """

    # Known obs/act dims for synthetic fallback (obs_dim, act_dim)
    _SYNTHETIC_DIMS: Dict[str, Tuple[int, int]] = {
        "hopper":        (11, 3),
        "halfcheetah":   (17, 6),
        "walker2d":      (17, 6),
        "ant":           (111, 8),
        "pendulum":      (3, 1),
    }

    # Where to look for local HDF5 files
    _DATA_DIR = Path(__file__).parent / "data"

    def __init__(
        self,
        device:          str   = "cpu",
        normalize_obs:   bool  = False,
        reward_scale:    float = 1.0,
        reward_shift:    float = 0.0,
    ) -> None:
        self._device        = device
        self._normalize_obs = normalize_obs
        self._reward_scale  = reward_scale
        self._reward_shift  = reward_shift

    # ── public ───────────────────────────────────────────────────────────────

    def load(
        self,
        dataset_id: str,
        backend:    str = "auto",
    ) -> OfflineReplayBuffer:
        """
        Load dataset and return a ready-to-sample OfflineReplayBuffer.

        Parameters
        ----------
        dataset_id : str
            e.g. "hopper-medium-v2", "halfcheetah-medium-replay-v2"
        backend : str
            "auto" | "minari" | "h5py" | "synthetic"
        """
        raw = self._load_raw(dataset_id, backend)
        self._validate(raw)

        buf = OfflineReplayBuffer(
            obs          = raw["observations"],
            actions      = raw["actions"],
            rewards      = raw["rewards"],
            next_obs     = raw["next_observations"],
            terminals    = raw["terminals"],
            device       = self._device,
            normalize_obs = self._normalize_obs,
            reward_scale  = self._reward_scale,
            reward_shift  = self._reward_shift,
        )
        logger.info("Loaded dataset '%s' → %r", dataset_id, buf)
        return buf

    # ── private: dispatch ─────────────────────────────────────────────────────

    def _load_raw(self, dataset_id: str, backend: str) -> DatasetDict:
        loaders = {
            "minari":    self._load_minari,
            "h5py":      self._load_h5py,
            "synthetic": self._load_synthetic,
        }

        if backend != "auto":
            if backend not in loaders:
                raise ValueError(f"Unknown backend '{backend}'. Choose from {list(loaders)}")
            return loaders[backend](dataset_id)

        # auto: try each in order, fall through on ImportError / FileNotFoundError
        for name, loader in loaders.items():
            try:
                data = loader(dataset_id)
                logger.info("Dataset loaded via backend '%s'", name)
                return data
            except (ImportError, FileNotFoundError, KeyError) as exc:
                logger.debug("Backend '%s' unavailable: %s", name, exc)

        raise RuntimeError(
            f"All backends failed for dataset '{dataset_id}'. "
            "Install minari (`pip install minari`) or place a .hdf5 file in data/."
        )

    # ── private: backends ────────────────────────────────────────────────────

    def _load_minari(self, dataset_id: str) -> DatasetDict:
        try:
            import minari  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("minari not installed: pip install minari") from exc

        dataset = minari.load_dataset(dataset_id, download=True)
        obs, actions, rewards, next_obs, terminals = [], [], [], [], []

        for ep in dataset.iterate_episodes():
            T = len(ep.rewards)
            obs.append(ep.observations[:-1])    # (T, obs_dim)
            next_obs.append(ep.observations[1:])
            actions.append(ep.actions)
            rewards.append(ep.rewards)
            term = np.zeros(T, dtype=np.float32)
            if ep.terminations[-1]:
                term[-1] = 1.0
            terminals.append(term)

        return {
            "observations":      np.concatenate(obs),
            "actions":           np.concatenate(actions),
            "rewards":           np.concatenate(rewards).astype(np.float32),
            "next_observations": np.concatenate(next_obs),
            "terminals":         np.concatenate(terminals).astype(np.float32),
        }

    def _load_h5py(self, dataset_id: str) -> DatasetDict:
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("h5py not installed: pip install h5py") from exc

        # Look for file in data/ directory
        candidates = [
            self._DATA_DIR / f"{dataset_id}.hdf5",
            self._DATA_DIR / f"{dataset_id}.h5",
            Path(dataset_id + ".hdf5"),
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No HDF5 file found for '{dataset_id}'. "
                f"Tried: {[str(c) for c in candidates]}"
            )

        with h5py.File(path, "r") as f:
            # terminals only — never include timeouts in done mask
            terminals = f["terminals"][:].astype(np.float32)

            return {
                "observations":      f["observations"][:].astype(np.float32),
                "actions":           f["actions"][:].astype(np.float32),
                "rewards":           f["rewards"][:].astype(np.float32),
                "next_observations": f["next_observations"][:].astype(np.float32),
                "terminals":         terminals,
            }

    def _load_synthetic(self, dataset_id: str) -> DatasetDict:
        """
        Generate a random Gaussian dataset for smoke-testing.
        obs/act dims are inferred from the task name prefix.
        """
        task_key = next(
            (k for k in self._SYNTHETIC_DIMS if dataset_id.startswith(k)),
            None,
        )
        if task_key is None:
            logger.warning(
                "Unknown task '%s' for synthetic backend; defaulting to (11, 3).",
                dataset_id,
            )
            obs_dim, act_dim = 11, 3
        else:
            obs_dim, act_dim = self._SYNTHETIC_DIMS[task_key]

        N = 200_000
        rng = np.random.default_rng(seed=42)

        obs     = rng.standard_normal((N, obs_dim)).astype(np.float32)
        actions = rng.uniform(-1, 1, (N, act_dim)).astype(np.float32)
        rewards = rng.standard_normal(N).astype(np.float32)
        # sparse terminals: episode length ~1000
        terminals = (rng.random(N) < 1 / 1000).astype(np.float32)

        logger.warning(
            "Using SYNTHETIC dataset for '%s' — only for pipeline smoke tests!",
            dataset_id,
        )
        return {
            "observations":      obs,
            "actions":           actions,
            "rewards":           rewards,
            "next_observations": obs + 0.01 * rng.standard_normal((N, obs_dim)).astype(np.float32),
            "terminals":         terminals,
        }

    # ── private: validation ───────────────────────────────────────────────────

    @staticmethod
    def _validate(raw: DatasetDict) -> None:
        required = {"observations", "actions", "rewards", "next_observations", "terminals"}
        missing = required - raw.keys()
        if missing:
            raise KeyError(f"Dataset missing required keys: {missing}")

        N = len(raw["observations"])
        for key, arr in raw.items():
            if len(arr) != N:
                raise ValueError(
                    f"Array '{key}' has length {len(arr)}, expected {N}."
                )

        if raw["actions"].ndim != 2:
            raise ValueError("actions must be 2-D (N, act_dim)")
        if raw["observations"].ndim != 2:
            raise ValueError("observations must be 2-D (N, obs_dim)")

        term_vals = np.unique(raw["terminals"])
        if not np.all(np.isin(term_vals, [0.0, 1.0])):
            raise ValueError(f"terminals should be binary; got unique values {term_vals}")
