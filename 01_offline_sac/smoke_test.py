"""
smoke_test.py — Fast pipeline verification (no MuJoCo required)

Checks:
  1. DatasetLoader (synthetic backend)
  2. OfflineReplayBuffer shape and sample
  3. Actor forward + act
  4. TwinQCritic forward
  5. SACTrainer.update() — 10 steps, verifies loss is finite
  6. SACTrainer checkpoint save/load round-trip

Run with:
    python smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
import tempfile

import numpy as np
import torch

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

_results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {GREEN}✓{RESET}  {name}")
        _results.append((name, True, None))
    except Exception as exc:
        print(f"  {RED}✗{RESET}  {name}")
        print(f"       {YELLOW}{exc}{RESET}")
        traceback.print_exc()
        _results.append((name, False, str(exc)))


# ── individual checks ─────────────────────────────────────────────────────────

def test_dataset_loader():
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(device="cpu")
    buf = loader.load("hopper-medium-v2", backend="synthetic")
    assert buf.size == 200_000, f"Expected 200000, got {buf.size}"
    assert buf.obs_dim  == 11
    assert buf.act_dim  == 3


def test_buffer_sample():
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(device="cpu")
    buf    = loader.load("hopper-medium-v2", backend="synthetic")
    batch  = buf.sample(256)

    assert batch.obs.shape      == (256, 11)
    assert batch.actions.shape  == (256, 3)
    assert batch.rewards.shape  == (256, 1)
    assert batch.next_obs.shape == (256, 11)
    assert batch.dones.shape    == (256, 1)
    assert batch.dones.max() <= 1.0 and batch.dones.min() >= 0.0, \
        "dones must be binary"


def test_buffer_normalise():
    from dataset_loader import DatasetLoader
    loader = DatasetLoader(device="cpu", normalize_obs=True)
    buf    = loader.load("hopper-medium-v2", backend="synthetic")
    mean, std = buf.normalize_obs_transform()
    assert mean is not None and std is not None
    # roughly zero-mean after normalisation
    sample_mean = buf.sample(10_000).obs.mean(0).abs().max().item()
    assert sample_mean < 0.2, f"Obs mean too large after normalisation: {sample_mean:.4f}"


def test_actor_forward():
    from networks import Actor
    actor = Actor(obs_dim=11, act_dim=3, hidden_dim=64)
    obs   = torch.randn(32, 11)
    actions, log_pi = actor(obs)
    assert actions.shape == (32, 3)
    assert log_pi.shape  == (32,)
    assert actions.abs().max().item() <= 1.0 + 1e-5, "tanh output out of (-1,1)"
    assert torch.isfinite(log_pi).all(), "log_pi contains NaN/Inf"


def test_critic_forward():
    from networks import TwinQCritic
    critic  = TwinQCritic(obs_dim=11, act_dim=3, hidden_dim=64)
    obs     = torch.randn(32, 11)
    actions = torch.randn(32, 3)
    q1, q2  = critic(obs, actions)
    assert q1.shape == (32, 1)
    assert q2.shape == (32, 1)
    assert torch.isfinite(q1).all()


def test_trainer_update():
    from dataset_loader import DatasetLoader
    from trainer import SACTrainer, SACConfig

    loader  = DatasetLoader(device="cpu")
    buf     = loader.load("hopper-medium-v2", backend="synthetic")

    cfg     = SACConfig(hidden_dim=64, batch_size=32, log_interval=5)
    trainer = SACTrainer(obs_dim=11, act_dim=3, config=cfg, device="cpu")

    metrics_seen = False
    for _ in range(10):
        m = trainer.update(buf)
        if m is not None:
            metrics_seen = True
            for key in ("critic_loss", "actor_loss", "alpha", "entropy"):
                assert key in m, f"Missing key '{key}' in metrics"
                assert torch.isfinite(torch.tensor(m[key])), \
                    f"{key}={m[key]} is not finite"

    assert metrics_seen, "log_interval=5 but no metrics returned in 10 steps"


def test_trainer_checkpoint():
    from dataset_loader import DatasetLoader
    from trainer import SACTrainer, SACConfig

    loader  = DatasetLoader(device="cpu")
    buf     = loader.load("hopper-medium-v2", backend="synthetic")
    cfg     = SACConfig(hidden_dim=64, batch_size=32, log_interval=999)
    trainer = SACTrainer(obs_dim=11, act_dim=3, config=cfg, device="cpu")

    for _ in range(5):
        trainer.update(buf)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save(trainer.state_dict(), path)

        trainer2 = SACTrainer(obs_dim=11, act_dim=3, config=cfg, device="cpu")
        trainer2.load_state_dict(torch.load(path, weights_only=True))

        assert trainer2.step == trainer.step
        # verify actor weights match
        for p1, p2 in zip(trainer.actor.parameters(), trainer2.actor.parameters()):
            assert torch.allclose(p1, p2), "Actor weights mismatch after load"


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  Offline SAC — smoke test")
    print(f"{'='*55}\n")

    check("DatasetLoader (synthetic)",       test_dataset_loader)
    check("Buffer sample shapes",            test_buffer_sample)
    check("Buffer obs normalisation",        test_buffer_normalise)
    check("Actor forward + tanh bounds",     test_actor_forward)
    check("TwinQCritic forward",             test_critic_forward)
    check("SACTrainer.update() 10 steps",    test_trainer_update)
    check("SACTrainer checkpoint round-trip",test_trainer_checkpoint)

    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)

    print(f"\n{'='*55}")
    print(f"  {passed}/{total} passed", end="")
    if passed == total:
        print(f"  {GREEN}ALL OK{RESET}")
    else:
        failed = [n for n, ok, _ in _results if not ok]
        print(f"  {RED}FAILED: {failed}{RESET}")
    print(f"{'='*55}\n")

    sys.exit(0 if passed == total else 1)
