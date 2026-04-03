# Offline SAC — Phase 1 Baseline

Offline SAC on D4RL tasks. No CQL penalty yet — this is the diverging baseline
we'll compare against once the conservative term is added.

## Directory layout

```
offline_sac/
├── buffer.py          OfflineReplayBuffer  — static, pre-filled, device-pinned
├── dataset_loader.py  DatasetLoader        — minari / h5py / synthetic backends
├── networks.py        Actor, TwinQCritic   — CleanRL-style MLP networks
├── trainer.py         SACTrainer           — all gradient logic, extensible via subclass
├── evaluator.py       Evaluator            — deterministic rollouts, D4RL norm score
├── run_logger.py      RunLogger            — console + TensorBoard + optional W&B
├── train.py           entry point          — wires everything, CLI args
├── smoke_test.py      pipeline smoke test  — no MuJoCo needed
├── configs/
│   └── hopper_medium.yaml
└── data/              place .hdf5 files here for h5py backend
```

## Setup

```bash
conda create -n offline-rl python=3.10
conda activate offline-rl

pip install torch
pip install gymnasium[mujoco]      # for eval only
pip install minari                  # recommended dataset backend
pip install tensorboard             # optional
pip install wandb                   # optional
pip install h5py                    # fallback dataset backend
```

## Smoke test (no MuJoCo required)

```bash
cd offline_sac
python smoke_test.py
```

All 7 checks should pass.

## Training

```bash
# Synthetic data (no installs beyond torch)
python train.py --backend synthetic

# Real D4RL dataset via minari
python train.py --dataset hopper-medium-v2

# 3 seeds (run sequentially or in parallel)
for seed in 0 1 2; do
    python train.py --dataset hopper-medium-v2 --seed $seed
done
```

## Key design decisions

| Decision | Reasoning |
|---|---|
| `terminals` only for done mask | D4RL `timeouts` are time-limit truncations, not true terminal states — including them corrupts Bellman targets |
| `_critic_loss()` is a separate method | CQL trainer will subclass `SACTrainer` and override this single method to add the conservative penalty |
| Device-pinned buffer tensors | `sample()` does zero host→device copies on the hot path |
| `Evaluator` fully decoupled | No gradient ops, separate env instances per episode, safe to call mid-training |

## Expected behaviour (vanilla offline SAC)

- `halfcheetah-medium-v2`: may show weak learning signal (~30-40 normalized score)
- `hopper-medium-v2`: likely unstable / diverging Q-values
- This is normal — it is the motivation for CQL

## Next phase

Subclass `SACTrainer` → `CQLTrainer`, override `_critic_loss()`:

```python
class CQLTrainer(SACTrainer):
    def _critic_loss(self, q1, q2, td_target):
        bellman = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        # CQL conservative penalty
        ...
        return bellman + alpha_cql * conservative_penalty
```
