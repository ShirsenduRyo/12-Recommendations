"""
common.py – Shared utilities: seed control, logging, data containers.
All reproducibility infrastructure lives here.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Seed control
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Bandit data container
# ---------------------------------------------------------------------------

@dataclass
class BanditDataset:
    """
    Immutable container for logged bandit data.

    Attributes
    ----------
    context_features : ndarray, shape (n, d)
    action           : ndarray, shape (n,) – integer action indices
    reward           : ndarray, shape (n,) – observed scalar rewards
    logging_prob     : ndarray, shape (n,) – propensity of chosen action under logging policy
    target_prob      : ndarray, shape (n,) – propensity of chosen action under target policy
    timestamp        : ndarray, shape (n,) – optional, for sequential settings
    n_actions        : int
    metadata         : dict – free-form info (env name, seed, etc.)
    """
    context_features : np.ndarray
    action           : np.ndarray
    reward           : np.ndarray
    logging_prob     : np.ndarray
    target_prob      : np.ndarray
    timestamp        : Optional[np.ndarray] = None
    n_actions        : int = 2
    metadata         : Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        n = len(self.context_features)
        assert len(self.action)       == n
        assert len(self.reward)       == n
        assert len(self.logging_prob) == n
        assert len(self.target_prob)  == n

    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        return len(self.reward)

    @property
    def importance_weights(self) -> np.ndarray:
        """IPS weights: π_target / π_logging (not clipped)."""
        return self.target_prob / (self.logging_prob + 1e-12)

    def clip_weights(self, clip: float) -> np.ndarray:
        """Clipped importance weights."""
        return np.clip(self.importance_weights, 0.0, clip)

    def split(self, frac: float = 0.8, seed: int = 42) -> tuple["BanditDataset", "BanditDataset"]:
        """Train / test split with fixed seed."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        cut = int(self.n * frac)
        tr, te = idx[:cut], idx[cut:]

        def _sub(i):
            ts = self.timestamp[i] if self.timestamp is not None else None
            return BanditDataset(
                self.context_features[i], self.action[i], self.reward[i],
                self.logging_prob[i], self.target_prob[i],
                ts, self.n_actions, self.metadata.copy()
            )
        return _sub(tr), _sub(te)

    def summary(self) -> str:
        iw = self.importance_weights
        lines = [
            f"BanditDataset  n={self.n}  n_actions={self.n_actions}",
            f"  Reward       mean={self.reward.mean():.4f}  std={self.reward.std():.4f}",
            f"  Log-prob     mean={self.logging_prob.mean():.4f}  min={self.logging_prob.min():.4f}",
            f"  IPS weights  mean={iw.mean():.2f}  max={iw.max():.2f}  ESS={self._ess(iw):.1f}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _ess(w: np.ndarray) -> float:
        """Effective sample size."""
        return (w.sum()) ** 2 / ((w ** 2).sum() + 1e-12)


# ---------------------------------------------------------------------------
# Config saving/loading
# ---------------------------------------------------------------------------

def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t

    def __repr__(self):
        return f"Timer({self.elapsed:.3f}s)"
