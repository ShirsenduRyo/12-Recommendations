"""
Offline Replay Buffer
Loaded once from dataset — no online interaction, no push().
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import NamedTuple


class Batch(NamedTuple):
    obs:        torch.Tensor   # (B, obs_dim)
    actions:    torch.Tensor   # (B, act_dim)
    rewards:    torch.Tensor   # (B, 1)
    next_obs:   torch.Tensor   # (B, obs_dim)
    dones:      torch.Tensor   # (B, 1) — from terminals only, NOT timeouts


@dataclass(frozen=True)
class BufferStats:
    size:        int
    obs_dim:     int
    act_dim:     int
    reward_mean: float
    reward_std:  float
    reward_min:  float
    reward_max:  float


class OfflineReplayBuffer:
    """
    Static buffer pre-filled from a D4RL-style HDF5 dataset.

    Key design decisions
    --------------------
    * terminals vs timeouts: D4RL datasets include both fields.
      Only `terminals` represent true episode ends and should be used
      in the Bellman done-mask. `timeouts` are time-limit truncations
      and must NOT be treated as terminals (they would corrupt Q-targets).

    * Normalisation: optionally z-score observations and clip rewards.
      Both are off by default; enable via constructor flags.

    * Device pinning: tensors are moved to `device` once at load time,
      so sample() never does a host→device copy on the hot path.
    """

    def __init__(
        self,
        obs:        np.ndarray,
        actions:    np.ndarray,
        rewards:    np.ndarray,
        next_obs:   np.ndarray,
        terminals:  np.ndarray,
        device:     str = "cpu",
        normalize_obs:     bool = False,
        reward_scale:      float = 1.0,
        reward_shift:      float = 0.0,
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(terminals), \
            "All dataset arrays must have the same first dimension."

        self._size    = len(obs)
        self._obs_dim = obs.shape[1]
        self._act_dim = actions.shape[1]
        self._device  = device

        # ── optional obs normalisation ───────────────────────────────────
        if normalize_obs:
            self._obs_mean = obs.mean(0, keepdims=True)
            self._obs_std  = obs.std(0, keepdims=True).clip(1e-3)
            obs      = (obs      - self._obs_mean) / self._obs_std
            next_obs = (next_obs - self._obs_mean) / self._obs_std
        else:
            self._obs_mean = None
            self._obs_std  = None

        # ── reward transform ─────────────────────────────────────────────
        rewards = rewards * reward_scale + reward_shift

        # ── pin tensors to device ────────────────────────────────────────
        def _t(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
            return torch.tensor(arr, dtype=dtype).to(device)

        self._obs      = _t(obs)
        self._actions  = _t(actions)
        self._rewards  = _t(rewards).unsqueeze(-1)          # (N,1)
        self._next_obs = _t(next_obs)
        self._dones    = _t(terminals.astype(np.float32)).unsqueeze(-1)  # (N,1)

    # ── public API ────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Batch:
        """Uniformly sample a mini-batch. O(1) device copies."""
        idx = torch.randint(0, self._size, (batch_size,), device=self._device)
        return Batch(
            obs      = self._obs[idx],
            actions  = self._actions[idx],
            rewards  = self._rewards[idx],
            next_obs = self._next_obs[idx],
            dones    = self._dones[idx],
        )

    def stats(self) -> BufferStats:
        r = self._rewards
        return BufferStats(
            size        = self._size,
            obs_dim     = self._obs_dim,
            act_dim     = self._act_dim,
            reward_mean = r.mean().item(),
            reward_std  = r.std().item(),
            reward_min  = r.min().item(),
            reward_max  = r.max().item(),
        )

    def normalize_obs_transform(self):
        """Return (mean, std) numpy arrays used during obs normalisation, or None."""
        return self._obs_mean, self._obs_std

    @property
    def size(self) -> int:
        return self._size

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"OfflineReplayBuffer("
            f"size={s.size:,}, obs_dim={s.obs_dim}, act_dim={s.act_dim}, "
            f"reward=[{s.reward_min:.2f}, {s.reward_max:.2f}], "
            f"device={self._device})"
        )
