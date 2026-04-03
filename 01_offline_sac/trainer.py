"""
Offline SAC Trainer
Encapsulates all training logic: critic update, actor update, alpha update,
target network soft-update, and logging.

No CQL penalty here — this is the vanilla offline SAC baseline.
CQL will be added as a subclass in a later phase.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from networks import Actor, TwinQCritic
from buffer import OfflineReplayBuffer, Batch

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    # ── architecture
    hidden_dim:  int   = 256
    n_layers:    int   = 2

    # ── training
    batch_size:  int   = 256
    gamma:       float = 0.99        # discount
    tau:         float = 0.005       # soft target update rate

    # ── optimisers
    actor_lr:    float = 3e-4
    critic_lr:   float = 3e-4
    alpha_lr:    float = 3e-4

    # ── entropy / temperature
    alpha_init:  float = 0.2
    auto_alpha:  bool  = True
    target_entropy: Optional[float] = None  # None → -act_dim (SAC default)

    # ── gradient clipping (0 = disabled)
    grad_clip:   float = 0.0

    # ── logging
    log_interval: int  = 1_000


class SACTrainer:
    """
    Offline SAC Trainer.

    Responsibilities
    ----------------
    - Owns Actor, TwinQCritic, and their target copies.
    - Owns all optimisers and the alpha (entropy temperature) scalar.
    - Exposes a single `update(buffer)` method that performs one gradient step.
    - Tracks a step counter and returns a metrics dict every log_interval steps.

    Extension point
    ---------------
    Subclass and override `_critic_loss()` to add CQL penalty (next phase).
    """

    def __init__(
        self,
        obs_dim:  int,
        act_dim:  int,
        config:   SACConfig,
        device:   str = "cpu",
    ) -> None:
        self._cfg    = config
        self._device = device
        self._step   = 0

        # ── networks ─────────────────────────────────────────────────────
        self.actor  = Actor(obs_dim, act_dim, config.hidden_dim, config.n_layers).to(device)
        self.critic = TwinQCritic(obs_dim, act_dim, config.hidden_dim, config.n_layers).to(device)

        # target critic — frozen copy, updated via soft polyak averaging
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ── optimisers ────────────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # ── entropy temperature α ─────────────────────────────────────────
        self._log_alpha  = torch.zeros(1, requires_grad=config.auto_alpha, device=device)
        self._alpha      = self._log_alpha.exp().detach()
        self._target_ent = (
            config.target_entropy
            if config.target_entropy is not None
            else -float(act_dim)                # SAC default: -|A|
        )
        if config.auto_alpha:
            self.alpha_opt = optim.Adam([self._log_alpha], lr=config.alpha_lr)

        # ── running metrics for logging ────────────────────────────────────
        self._metrics: Dict[str, float] = {}

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        return self._alpha.item()

    @property
    def step(self) -> int:
        return self._step

    def update(self, buffer: OfflineReplayBuffer) -> Optional[Dict[str, float]]:
        """
        Perform one full SAC gradient step (critic → actor → alpha → target).

        Returns a metrics dict every log_interval steps, else None.
        """
        batch = buffer.sample(self._cfg.batch_size)
        self._step += 1

        critic_metrics = self._update_critic(batch)
        actor_metrics  = self._update_actor(batch)
        alpha_metrics  = self._update_alpha(batch)

        self._soft_update_target()

        self._metrics.update({**critic_metrics, **actor_metrics, **alpha_metrics})

        if self._step % self._cfg.log_interval == 0:
            return {**self._metrics, "step": self._step}
        return None

    def state_dict(self) -> Dict:
        """Full checkpoint dict."""
        sd = {
            "actor":          self.actor.state_dict(),
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "actor_opt":      self.actor_opt.state_dict(),
            "critic_opt":     self.critic_opt.state_dict(),
            "log_alpha":      self._log_alpha.data,
            "step":           self._step,
        }
        if self._cfg.auto_alpha:
            sd["alpha_opt"] = self.alpha_opt.state_dict()
        return sd

    def load_state_dict(self, sd: Dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self._log_alpha.data.copy_(sd["log_alpha"])
        self._alpha = self._log_alpha.exp().detach()
        self._step  = sd["step"]
        if self._cfg.auto_alpha and "alpha_opt" in sd:
            self.alpha_opt.load_state_dict(sd["alpha_opt"])

    # ── update steps ──────────────────────────────────────────────────────────

    def _update_critic(self, batch: Batch) -> Dict[str, float]:
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(batch.next_obs)
            q1_next, q2_next = self.critic_target(batch.next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self._alpha * next_log_pi.unsqueeze(-1)
            td_target = batch.rewards + self._cfg.gamma * (1.0 - batch.dones) * q_next

        q1, q2 = self.critic(batch.obs, batch.actions)
        critic_loss = self._critic_loss(q1, q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self._cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._cfg.grad_clip)
        self.critic_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "q1_mean":     q1.mean().item(),
            "q2_mean":     q2.mean().item(),
            "td_target":   td_target.mean().item(),
        }

    def _critic_loss(
        self,
        q1:        torch.Tensor,
        q2:        torch.Tensor,
        td_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Base SAC critic loss (MSE). Override in CQLTrainer to add penalty.
        """
        return F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

    def _update_actor(self, batch: Batch) -> Dict[str, float]:
        actions, log_pi = self.actor(batch.obs)
        q_min = self.critic.q_min(batch.obs, actions)

        actor_loss = (self._alpha * log_pi.unsqueeze(-1) - q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self._cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self._cfg.grad_clip)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "entropy":    -log_pi.mean().item(),
        }

    def _update_alpha(self, batch: Batch) -> Dict[str, float]:
        if not self._cfg.auto_alpha:
            return {"alpha": self._alpha.item()}

        with torch.no_grad():
            _, log_pi = self.actor(batch.obs)

        alpha_loss = -(
            self._log_alpha * (log_pi + self._target_ent)
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._alpha = self._log_alpha.exp().detach()

        return {
            "alpha":      self._alpha.item(),
            "alpha_loss": alpha_loss.item(),
        }

    def _soft_update_target(self) -> None:
        tau = self._cfg.tau
        for p, p_tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)
