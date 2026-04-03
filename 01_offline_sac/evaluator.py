"""
Evaluator
Runs the current actor deterministically in the real environment.
Completely decoupled from training — no gradient ops, no buffer access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from networks import Actor

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    mean_return:      float
    std_return:       float
    min_return:       float
    max_return:       float
    mean_ep_len:      float
    normalized_score: Optional[float]   # D4RL normalized score if ref available
    n_episodes:       int


# D4RL reference scores for normalisation: (random_score, expert_score)
_D4RL_REF_SCORES = {
    "hopper":       (20.272305,  3234.3),
    "halfcheetah":  (-280.178953, 12135.0),
    "walker2d":     (1.629008,   4592.3),
    "ant":          (-325.6,     3879.7),
}


def _get_ref_scores(env_id: str):
    for key, (rnd, exp) in _D4RL_REF_SCORES.items():
        if env_id.startswith(key):
            return rnd, exp
    return None, None


class Evaluator:
    """
    Runs `n_episodes` rollouts using the actor's deterministic (mean) action.

    Parameters
    ----------
    env_id   : gymnasium environment id, e.g. "Hopper-v4"
    n_episodes : number of evaluation episodes
    device   : device the actor lives on (tensors moved there internally)
    seed     : base seed; episode i uses seed + i

    Notes
    -----
    - Uses the actor's `act()` method which takes tanh(mean) — no sampling.
    - Environment is created fresh on each `evaluate()` call to avoid state leakage.
    - Normalised score follows D4RL convention:
        score = 100 * (return - random) / (expert - random)
    """

    def __init__(
        self,
        env_id:     str,
        n_episodes: int = 10,
        device:     str = "cpu",
        seed:       int = 0,
    ) -> None:
        self._env_id     = env_id
        self._n_episodes = n_episodes
        self._device     = device
        self._seed       = seed

    def evaluate(self, actor: Actor) -> EvalResult:
        """Run evaluation and return structured result."""
        try:
            import gymnasium as gym  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("gymnasium not installed: pip install gymnasium[mujoco]") from exc

        actor.eval()
        returns: List[float] = []
        ep_lens: List[int]   = []

        for i in range(self._n_episodes):
            env = gym.make(self._env_id)
            obs, _ = env.reset(seed=self._seed + i)
            ep_ret, ep_len = 0.0, 0
            done = False

            while not done:
                action = actor.act(obs, device=self._device)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_ret += reward
                ep_len += 1
                done = terminated or truncated

            returns.append(ep_ret)
            ep_lens.append(ep_len)
            env.close()

        actor.train()

        arr = np.array(returns)
        rnd, exp = _get_ref_scores(self._env_id.lower())
        norm = (
            100.0 * (arr.mean() - rnd) / (exp - rnd)
            if rnd is not None else None
        )

        result = EvalResult(
            mean_return      = float(arr.mean()),
            std_return       = float(arr.std()),
            min_return       = float(arr.min()),
            max_return       = float(arr.max()),
            mean_ep_len      = float(np.mean(ep_lens)),
            normalized_score = norm,
            n_episodes       = self._n_episodes,
        )
        logger.info(
            "Eval [%s] return=%.1f±%.1f norm=%.1f",
            self._env_id,
            result.mean_return,
            result.std_return,
            result.normalized_score or 0.0,
        )
        return result
