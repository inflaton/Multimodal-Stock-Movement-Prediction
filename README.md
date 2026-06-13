# Does Sentiment Fusion Help?
**A Selection-Bias-Free Re-Evaluation of Multimodal Stock-Movement Prediction**

Companion code and data for the ICDM 2026 Applied Track submission.

📄 **Paper:** [`paper/paper.pdf`](./paper/paper.pdf) · LaTeX source under [`paper/latex/`](./paper/latex/)

## Summary

Multimodal stock-movement prediction — fusing financial-news and social-media
sentiment with technical indicators — is widely reported to deliver large
risk-adjusted gains, with recent studies citing average Sharpe ratios above 2.
This repository reproduces the paper's central finding: under a
selection-bias-free evaluation protocol (chronological train/validation/test
split with embargo, validation-locked model and horizon selection, multi-seed
reporting with paired significance tests), sentiment fusion does **not** beat
a technical-indicator-only baseline. Most of the headline Sharpe gains
reported in prior work come from optimistic evaluation, not from the sentiment
signal.

The pipeline evaluates:

- **5 U.S. equities:** AAPL, META, NVDA, SPY, TSLA
- **7 task-specific model families:** Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM, LSTM
- **2 time-series foundation models:** Chronos-2 and FinCast (zero-shot and fine-tuned)
- **8 feature configurations × 3 horizons × 5 seeds = 4,200 task-specific runs**

## Repository layout

```
Multimodal-Stock-Movement-Prediction/
├── paper/
│   ├── paper.pdf                       # the ICDM 2026 manuscript
│   └── latex/                          # LaTeX source: main.tex, references.bib, IEEEtran.{cls,bst}, figures/
├── configs/
│   └── default.yaml                    # central experiment config (split, seeds, embargo, HPO budgets, ablations)
├── dataset/
│   └── training_features/              # 5 stock CSVs: Close, 10 indicator pairs, News/Social FinBERT-Tone sentiment
├── src/
│   ├── config.py                       # config loader, derived paths
│   ├── data.py                         # load + chronological split + embargo + sentiment fusion
│   ├── metrics.py                      # Sharpe, MDD, Sortino, Calmar, DSR, PBO, block-bootstrap CIs
│   ├── backtest.py                     # long/short, non-overlapping h-day positions, 10 bps costs
│   ├── learned_alpha.py                # learned reliability-weighted fusion (per-stock + global)
│   ├── hpo_traditional.py              # LR/SVM/RF/GB/XGB/LGBM Bayesian HPO
│   ├── hpo_lstm.py                     # LSTM HPO (reduced grid; tune at h=5, broadcast)
│   ├── chronos_baseline.py             # zero-shot Chronos-2
│   ├── chronos_finetune.py             # Chronos-2 fine-tune (+ optional covariates)
│   ├── chronos_runner.py               # prints chronos_* launch commands
│   ├── fincast_baseline.py             # zero-shot FinCast
│   ├── fincast_finetune.py             # FinCast LoRA fine-tune
│   ├── fincast_runner.py               # prints fincast_* launch commands
│   ├── _load_features.py               # shared loader for foundation scripts
│   ├── select_best_config.py           # writes results/selected_config.json (validation-locked)
│   └── run_pipeline.py                 # end-to-end driver (steps 1-6)
├── scripts/
│   ├── run_chronos.sh                  # Chronos-2 sweep wrapper
│   └── run_fincast.sh                  # FinCast sweep wrapper
├── notebooks/
│   ├── 01_data_exploration.ipynb       # dataset statistics, sentiment distribution, signal characteristics
│   ├── 02_learned_alpha.ipynb          # learned-α fit with L2-regularization sensitivity
│   ├── 03_pipeline_validation.ipynb    # split / embargo / determinism sanity checks
│   ├── 04_results_aggregation.ipynb    # manuscript tables from results/raw/
│   ├── 05_figures.ipynb                # manuscript figures
│   └── 06_indicator_pairs_appendix.ipynb
└── results/                            # raw run output, selected_config.json, figures, tables
```

## Requirements

- Python ≥ 3.10
- A CUDA-capable GPU for the LSTM and the foundation models (the paper uses an RTX 4090 Laptop with 16 GB VRAM)
- ≈ 30 GB free disk for foundation-model weights and per-cell logs

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**FinCast** is a separate codebase and is **not** vendored here
(`FinCast-fts/` is git-ignored). Either:

- Place its source at `FinCast-fts/src/` and the pretrained weights at
  `FinCast-fts/model_weights/v1.pth` (the `scripts/run_fincast.sh` defaults), or
- Point `FINCAST_PATH` and `FINCAST_WEIGHTS` at any other location.

## Reproducing the paper

The pipeline is six steps; each is idempotent and writes intermediate state
under `results/`. All knobs live in `configs/default.yaml`.

```bash
cd Multimodal-Stock-Movement-Prediction

# (Optional) Sanity check: verifies split, seed reproducibility, single-cell eval.
jupyter notebook notebooks/03_pipeline_validation.ipynb

# Step 1 — Fit the learned-α reliability-weighted fusion (per-stock + global).
python -m src.run_pipeline --steps 1

# Step 2 — Traditional ML HPO: 5 stocks × 8 ablations × 3 horizons × 5 seeds × 6 models.
python -m src.run_pipeline --steps 2

# Step 3 — LSTM HPO with the reduced grid (tune at h=5, broadcast).
python -m src.run_pipeline --steps 3

# Step 4 — Foundation models. Each wrapper accepts (MODE STOCK SEED) positional
# args, so the same script does pretests and the full sweep. Logs land in
# logs/<tag>_<timestamp>.log and printed commands run inside `conda run -n <env>`.

# Chronos-2 (needs chronos-forecasting + autogluon.timeseries):
bash scripts/run_chronos.sh chronos2_finetuned_cov AAPL 42   # pretest: one cell
bash scripts/run_chronos.sh                                  # full sweep: 4 modes × 5 stocks × 3 seeds = 60 cells

# FinCast (set FINCAST_PATH + FINCAST_WEIGHTS, or accept the in-repo defaults):
export FINCAST_PATH="$(pwd)/FinCast-fts/src"
export FINCAST_WEIGHTS="$(pwd)/FinCast-fts/model_weights/v1.pth"
bash scripts/run_fincast.sh fincast_finetuned AAPL 42        # pretest
bash scripts/run_fincast.sh                                  # full sweep: 2 modes × 5 × 3 = 30 cells

# Long overnight runs: detach with nohup and tail the log.
nohup bash scripts/run_chronos.sh > /dev/null 2>&1 &
tail -f logs/chronos_*_$(date +%Y%m%d)*.log

# Step 5 — Lock model and horizon selection by validation scores
#          → writes results/selected_config.json.
python -m src.run_pipeline --steps 5

# Step 6 — Aggregate every cell into results/all_results.csv.
python -m src.run_pipeline --steps 6

# Build manuscript tables and figures.
jupyter notebook notebooks/04_results_aggregation.ipynb
jupyter notebook notebooks/05_figures.ipynb
jupyter notebook notebooks/06_indicator_pairs_appendix.ipynb
```

## Validation checks before launching the full grid

1. **Chronos-2 fits in VRAM.** On 16 GB VRAM run a single fine-tune-with-covariates cell:
   ```bash
   bash scripts/run_chronos.sh chronos2_finetuned_cov AAPL 42
   ```
   Monitor `nvidia-smi`. If you OOM, drop `chronos_fine_tune_batch_size` from 8 → 4 in `configs/default.yaml`.

2. **LSTM determinism.** Run cell 4 of `notebooks/03_pipeline_validation.ipynb` twice and confirm `Val_ROC_AUC` is bit-for-bit identical across runs.

3. **Sentiment correlation against the original pipeline.** `notebooks/01_data_exploration.ipynb` reports `corr(train Weighted Sentiment, our S_70_30)`. A persistent correlation below 0.7 means the daily aggregation rule has drifted from the paper's `0.7·mean(news) + 0.3·mean(social)`.

## Paper ↔ code mapping

| Paper section | Code / artifact |
|---|---|
| §III.A Problem formulation | `src/data.py` |
| §III.B Chronological split + embargo | `src/data.py` |
| §III.C Validation-locked selection | `src/select_best_config.py` → `results/selected_config.json` |
| §III.D Learned reliability-weighted fusion | `src/learned_alpha.py`, `notebooks/02_learned_alpha.ipynb` |
| §III.E Multi-seed reporting + statistics | `src/metrics.py`, `src/backtest.py` |
| §IV Experimental setup | `configs/default.yaml`, `dataset/training_features/` |
| §V Task-specific grid results | `notebooks/04_results_aggregation.ipynb`, `results/raw/`, `results/all_results.csv` |
| §V Foundation-model results | `src/chronos_*.py`, `src/fincast_*.py`, `results/chronos-2/`, `results/fincast/` |
| Figures 1–4 | `notebooks/05_figures.ipynb`, `results/fig_*.pdf` |

## Citation

If you use this repository or the dataset, please cite:

```bibtex
@inproceedings{huang2026sentiment,
  title     = {Does Sentiment Fusion Help? A Selection-Bias-Free Re-Evaluation
               of Multimodal Stock Movement Prediction},
  author    = {Huang, Donghao and Heng, Zheng Kai and Wang, Zhaoxia},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  note      = {Applied Track}
}
```

## License

A `LICENSE` file will be added before public release. Until then, please
contact the authors before redistributing.

## Contact

Corresponding author: Zhaoxia Wang &lt;<zxwang@smu.edu.sg>&gt;.
