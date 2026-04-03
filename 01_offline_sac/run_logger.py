"""
Logger
Thin wrapper that fans out metrics to console, TensorBoard, and optionally W&B.
All training code calls logger.log_metrics() — backend details are hidden here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from evaluator import EvalResult

# stdlib logger for internal messages
_log = logging.getLogger(__name__)


class RunLogger:
    """
    Unified metrics logger.

    Backends enabled at construction time:
      - console  : always on
      - tensorboard : if `log_dir` is provided and tensorboard is installed
      - wandb       : if `use_wandb=True` and wandb is installed + authenticated

    Usage
    -----
        logger = RunLogger(log_dir="logs/run_0", use_wandb=False)
        logger.log_metrics({"critic_loss": 0.3, "actor_loss": 0.1}, step=1000)
        logger.log_eval(eval_result, step=1000)
        logger.close()
    """

    def __init__(
        self,
        run_name:  str,
        log_dir:   Optional[str] = None,
        use_wandb: bool          = False,
        config:    Optional[Dict[str, Any]] = None,
    ) -> None:
        self._run_name = run_name
        self._tb       = None
        self._wandb    = None

        # ── TensorBoard ───────────────────────────────────────────────────
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
                self._tb = SummaryWriter(log_dir=log_dir)
                _log.info("TensorBoard logging → %s", log_dir)
            except ImportError:
                _log.warning("tensorboard not installed — skipping TB logging.")

        # ── W&B ───────────────────────────────────────────────────────────
        if use_wandb:
            try:
                import wandb  # noqa: PLC0415
                wandb.init(
                    project = "offline-sac",
                    name    = run_name,
                    config  = config or {},
                )
                self._wandb = wandb
                _log.info("W&B run initialised: %s", run_name)
            except ImportError:
                _log.warning("wandb not installed — skipping W&B logging.")

    # ── public API ────────────────────────────────────────────────────────────

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log a flat dict of scalar metrics."""
        # console
        parts = [f"step={step:>7d}"]
        for k, v in metrics.items():
            if k != "step":
                parts.append(f"{k}={v:.4f}")
        print("  ".join(parts))

        # TensorBoard
        if self._tb:
            for k, v in metrics.items():
                if k != "step":
                    self._tb.add_scalar(k, v, global_step=step)

        # W&B
        if self._wandb:
            self._wandb.log({k: v for k, v in metrics.items() if k != "step"}, step=step)

    def log_eval(self, result: EvalResult, step: int) -> None:
        """Log an EvalResult from Evaluator."""
        metrics = {
            "eval/mean_return": result.mean_return,
            "eval/std_return":  result.std_return,
            "eval/min_return":  result.min_return,
            "eval/max_return":  result.max_return,
            "eval/mean_ep_len": result.mean_ep_len,
        }
        if result.normalized_score is not None:
            metrics["eval/normalized_score"] = result.normalized_score

        print(
            f"\n{'='*60}\n"
            f"  EVAL  step={step:,}  "
            f"return={result.mean_return:.1f}±{result.std_return:.1f}  "
            f"norm={result.normalized_score:.1f if result.normalized_score else 'n/a'}\n"
            f"{'='*60}\n"
        )
        self.log_metrics(metrics, step)

    def close(self) -> None:
        if self._tb:
            self._tb.close()
        if self._wandb:
            self._wandb.finish()
