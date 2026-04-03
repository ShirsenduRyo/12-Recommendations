"""
train.py — Offline SAC entry point

Wires together:
  DatasetLoader → OfflineReplayBuffer
  SACTrainer (Actor + TwinQCritic + alpha)
  Evaluator (every eval_interval steps)
  RunLogger (console + TB + optional W&B)

Usage
-----
  python train.py                               # defaults (synthetic data)
  python train.py --dataset hopper-medium-v2    # requires minari or hdf5
  python train.py --dataset hopper-medium-v2 --seed 1 --total_steps 1000000
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from dataset_loader import DatasetLoader
from trainer import SACTrainer, SACConfig
from evaluator import Evaluator
from run_logger import RunLogger

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── D4RL task → gymnasium env id mapping ─────────────────────────────────────
DATASET_TO_ENV = {
    "hopper":      "Hopper-v4",
    "halfcheetah": "HalfCheetah-v4",
    "walker2d":    "Walker2d-v4",
    "ant":         "Ant-v4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline SAC baseline")

    # data
    p.add_argument("--dataset",    type=str, default="hopper-medium-v2")
    p.add_argument("--backend",    type=str, default="auto",
                   choices=["auto", "minari", "h5py", "synthetic"])

    # training
    p.add_argument("--total_steps",  type=int, default=1_000_000)
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--hidden_dim",   type=int, default=256)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--actor_lr",     type=float, default=3e-4)
    p.add_argument("--critic_lr",    type=float, default=3e-4)
    p.add_argument("--alpha_lr",     type=float, default=3e-4)
    p.add_argument("--alpha_init",   type=float, default=0.2)

    # obs normalisation and reward scaling
    p.add_argument("--normalize_obs",  action="store_true")
    p.add_argument("--reward_scale",   type=float, default=1.0)

    # eval
    p.add_argument("--eval_interval", type=int, default=10_000)
    p.add_argument("--eval_episodes", type=int, default=10)

    # misc
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--log_dir",   type=str, default="logs")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--save_dir",  type=str, default="checkpoints")

    return p.parse_args()


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_env_id(dataset_id: str) -> str | None:
    for key, env_id in DATASET_TO_ENV.items():
        if dataset_id.startswith(key):
            return env_id
    return None


def save_checkpoint(trainer: SACTrainer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.state_dict(), path)
    logger.info("Checkpoint saved → %s", path)


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed, device)

    run_name = f"{args.dataset}__seed{args.seed}"
    log_dir  = os.path.join(args.log_dir, run_name)

    logger.info("Run: %s | device: %s", run_name, device)

    # ── 1. load dataset ───────────────────────────────────────────────────
    loader = DatasetLoader(
        device        = device,
        normalize_obs = args.normalize_obs,
        reward_scale  = args.reward_scale,
    )
    buffer = loader.load(args.dataset, backend=args.backend)
    logger.info("Buffer ready: %r", buffer)

    # ── 2. build trainer ──────────────────────────────────────────────────
    cfg = SACConfig(
        hidden_dim   = args.hidden_dim,
        batch_size   = args.batch_size,
        gamma        = args.gamma,
        tau          = args.tau,
        actor_lr     = args.actor_lr,
        critic_lr    = args.critic_lr,
        alpha_lr     = args.alpha_lr,
        alpha_init   = args.alpha_init,
        auto_alpha   = True,
        log_interval = 1_000,
    )
    trainer = SACTrainer(
        obs_dim = buffer.obs_dim,
        act_dim = buffer.act_dim,
        config  = cfg,
        device  = device,
    )

    # ── 3. evaluator (optional — skipped if no env available) ─────────────
    env_id    = resolve_env_id(args.dataset)
    evaluator = None
    if env_id:
        try:
            evaluator = Evaluator(
                env_id     = env_id,
                n_episodes = args.eval_episodes,
                device     = device,
                seed       = args.seed + 1000,
            )
            logger.info("Evaluator ready for env '%s'", env_id)
        except Exception as exc:
            logger.warning("Could not set up evaluator: %s — skipping eval.", exc)

    # ── 4. logger ─────────────────────────────────────────────────────────
    run_logger = RunLogger(
        run_name  = run_name,
        log_dir   = log_dir,
        use_wandb = args.use_wandb,
        config    = vars(args),
    )

    # ── 5. training loop ──────────────────────────────────────────────────
    logger.info("Starting training for %d steps...", args.total_steps)

    for step in range(1, args.total_steps + 1):
        metrics = trainer.update(buffer)

        if metrics is not None:
            run_logger.log_metrics(metrics, step=step)

        # periodic evaluation
        if evaluator and step % args.eval_interval == 0:
            result = evaluator.evaluate(trainer.actor)
            run_logger.log_eval(result, step=step)

        # periodic checkpoint
        if step % 100_000 == 0:
            ckpt_path = Path(args.save_dir) / run_name / f"step_{step}.pt"
            save_checkpoint(trainer, ckpt_path)

    # final checkpoint
    save_checkpoint(trainer, Path(args.save_dir) / run_name / "final.pt")
    run_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
