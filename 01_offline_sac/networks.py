"""
SAC Networks — Actor and Twin-Q Critic
Ported from CleanRL's SAC, refactored into OOP classes.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── constants ────────────────────────────────────────────────────────────────
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    """Build a simple MLP: Linear → ReLU → ... → Linear."""
    dims  = [in_dim] + [hidden] * layers + [out_dim]
    mods  = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


def _init_weights(module: nn.Module) -> None:
    """Orthogonal init for Linear layers — standard in CleanRL."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        nn.init.constant_(module.bias, 0.0)


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Squashed Gaussian actor.

    Outputs (action, log_prob) via the reparameterisation trick.
    log_prob is corrected for the tanh squashing (see SAC paper App. C).
    """

    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        hidden_dim:  int = 256,
        n_layers:    int = 2,
    ) -> None:
        super().__init__()
        self.backbone    = _mlp(obs_dim, hidden_dim, hidden_dim, n_layers - 1)
        self.mean_head   = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        self.apply(_init_weights)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action   : tanh-squashed action in (-1, 1)^act_dim
        log_prob : scalar log-probability per sample, shape (B,)
        """
        h       = F.relu(self.backbone(obs))
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        dist    = Normal(mean, std)
        x_t     = dist.rsample()                         # reparameterised sample
        y_t     = torch.tanh(x_t)                        # squash to (-1, 1)

        # log prob with tanh-Jacobian correction (numerically stable form)
        log_prob = dist.log_prob(x_t).sum(-1)
        log_prob -= (2.0 * (math.log(2) - x_t - F.softplus(-2 * x_t))).sum(-1)

        return y_t, log_prob

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Deterministic greedy action for evaluation only.
        Not used during offline training (no env interaction).
        """
        obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        h      = F.relu(self.backbone(obs_t))
        mean   = self.mean_head(h)
        return torch.tanh(mean).squeeze(0).cpu().numpy()


# ── Critic (Twin-Q) ────────────────────────────────────────────────────────────

class TwinQCritic(nn.Module):
    """
    Twin Q-networks as used in SAC / TD3.

    Two independent Q-networks (Q1, Q2).
    Training uses the minimum of the two to suppress Q-overestimation.
    Targets are computed from the target network copy (managed by Trainer).
    """

    def __init__(
        self,
        obs_dim:    int,
        act_dim:    int,
        hidden_dim: int = 256,
        n_layers:   int = 2,
    ) -> None:
        super().__init__()
        in_dim    = obs_dim + act_dim
        self.q1   = _mlp(in_dim, hidden_dim, 1, n_layers)
        self.q2   = _mlp(in_dim, hidden_dim, 1, n_layers)
        self.apply(_init_weights)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1_values, q2_values), each shape (B, 1)."""
        x  = torch.cat([obs, actions], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: min(Q1, Q2) — used in actor loss."""
        q1, q2 = self(obs, actions)
        return torch.min(q1, q2)
