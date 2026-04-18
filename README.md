# OPE Research — Hybrid Off-Policy Evaluation

> Combine IPS + representation learning + temporal modeling to stabilize long-horizon policy evaluation.

---

## Dataset

**MovieLens-100k** — most widely used OPE/recommendation benchmark.

| Field | Value |
|-------|-------|
| Download | **https://grouplens.org/datasets/movielens/100k/** |
| Mirror (Kaggle) | https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset |
| Users | 943 |
| Items | 1,682 movies |
| Ratings | 100,000 (scale 1–5) |
| Density | ~6.3% |
| Time span | Sep 1997 – Apr 1998 |
| License | Free for non-commercial research use |

**Bandit framing**: user = context, recommended item = action, rating ≥ 4 = reward=1, else reward=0.

**Referenced in**:
- Gilotte et al. 2018 — *Offline A/B Testing for Recommender Systems* ([paper](https://dl.acm.org/doi/10.1145/3159652.3159687))
- Strehl et al. 2010 — *Learning from Logged Implicit Exploration Data* ([paper](https://papers.nips.cc/paper/2010/hash/9fe8593a8a330607d76796b35c64c600-Abstract.html))
- Swaminathan & Joachims 2015 — *Batch Learning from Logged Bandit Feedback* ([paper](https://jmlr.org/papers/v16/swaminathan15a.html))
- Open Bandit Pipeline (OBP) benchmark suite ([repo](https://github.com/st-tech/zr-obp))

---

## Quick start

```bash
# 1. clone repo
git clone https://github.com/yourorg/ope-research
cd ope-research

# 2. install deps
pip install -e .

# 3. download dataset
# Option A: manual download from https://grouplens.org/datasets/movielens/100k/
# unzip ml-100k.zip into data/

# Option B: use pre-generated replica (already in data/ after running 00_setup.ipynb)

# 4. run notebooks in order
jupyter notebook notebooks/
```

---

## Notebook order

| Notebook | What it does |
|----------|-------------|
| `00_setup.ipynb` | Install deps, build bandit dataset from ratings |
| `01_data_exploration.ipynb` | Schema, propensity analysis, context PCA, train/val/test split |
| `02_ips_estimator.ipynb` | IPS + SNIPS baselines, variance decay, clip sensitivity |
| `03_dm_estimator.ipynb` | Direct Method, reward model comparison, misspecification bias |
| `04_dr_estimator.ipynb` | Doubly Robust + Switch-DR, instability under joint misspecification |
| `05_representation_model.ipynb` | MLP reward model, penultimate-layer embeddings, variance reduction |
| `06_temporal_model.ipynb` | Rolling features, temporal reward model, stability over time |
| `07_hybrid_estimator.ipynb` | **Core**: DR(repr) + temporal correction, lambda tuning, bootstrap CI |
| `08_benchmark.ipynb` | All estimators, 10 seeds, violin plots, sample efficiency curves |
| `09_failure_analysis.ipynb` | Low propensity / shift / sparse rewards / misspecification stress tests |

---

## Hybrid estimator formula

$$\hat{V}^{Hybrid}(\pi) = \underbrace{\hat{V}^{DM}_{repr}(\pi)}_{\text{repr DM}} + \underbrace{\frac{1}{n}\sum_i w_i^{clip} (r_i - \hat{r}_{repr}(z_i,a_i))}_{\text{IPS residual}} + \underbrace{\lambda \cdot \frac{1}{n}\sum_i w_i^{clip}(\hat{r}_{temp}(z_i,a_i) - \hat{r}_{repr}(z_i,a_i))}_{\text{temporal correction}}$$

where $z_i$ = MLP penultimate embedding, $w_i = \pi(a_i|x_i)/\mu(a_i|x_i)$ clipped at threshold, $\lambda$ tuned on validation set.

---

## Estimators implemented

| Estimator | Bias | Variance | Notes |
|-----------|------|----------|-------|
| IPS | Unbiased | High | Explodes under low propensity |
| SNIPS | Slightly biased | Lower | Self-normalised weights |
| DM | Biased | Low | Fails under model misspecification |
| DR | Doubly robust | Medium | Unstable when both components wrong |
| Switch-DR | Adaptive | Medium | Falls back to DM on large weights |
| **Hybrid** | **Doubly robust** | **Low** | **Repr + temporal correction** |

---

## Project structure

```
ope-research/
├── data/
│   ├── ratings.csv           ← raw MovieLens-style ratings
│   ├── bandit_data.csv       ← bandit log (action, reward, propensity)
│   ├── train.csv / val.csv / test.csv
│   ├── train_temporal.csv / test_temporal.csv
│   └── Z_train.npy / Z_test.npy  ← MLP representations
│
├── notebooks/                ← run these in order 00 → 09
│
├── src/
│   ├── estimators/
│   │   ├── ips.py
│   │   ├── dr.py
│   │   └── hybrid_estimator.py
│   ├── models/
│   │   ├── representation_model.py
│   │   └── temporal_model.py
│   └── evaluation/
│       ├── metrics.py
│       └── benchmark.py
│
├── results/                  ← plots + CSVs auto-saved by notebooks
├── experiments/configs/      ← YAML configs for reproducible runs
├── pyproject.toml
└── README.md
```

---

## Key metrics

| Metric | Description |
|--------|-------------|
| Policy value error | \|estimate − true value\| |
| MSE | Mean squared error over seeds |
| CI coverage | Fraction of CIs containing true value |
| Variance | Std dev of estimates across seeds |
| Sample efficiency | Std dev vs sample size |

---

## Reproduction

All experiments use fixed seeds via `np.random.seed(seed)` before each run.
Configs logged in `experiments/configs/`.
Results reproduced by running notebooks 00 → 09 in sequence.

---

## References

1. Horvitz & Thompson (1952). *A generalization of sampling without replacement from a finite universe.* JASA.
2. Strehl et al. (2010). *Learning from Logged Implicit Exploration Data.* NeurIPS.
3. Swaminathan & Joachims (2015). *Batch Learning from Logged Bandit Feedback.* JMLR.
4. Dudík et al. (2011). *Doubly Robust Policy Evaluation and Learning.* ICML. [link](https://proceedings.mlr.press/v28/dudik13.html)
5. Wang et al. (2017). *Optimal and Adaptive Off-Policy Evaluation in Contextual Bandits.* ICML.
6. Farajtabar et al. (2018). *More Robust Doubly Robust Off-Policy Evaluation.* ICML.
7. Nachum et al. (2021). *Representation Matters: Offline Pretraining for Sequential Decision Making.* ICML.
8. Harper & Konstan (2015). *The MovieLens Datasets.* ACM TIIS. [link](https://dl.acm.org/doi/10.1145/2827872)

---

## License

MIT. Dataset subject to MovieLens terms: free for non-commercial research use only.
