# 2026_icdm — revision workspace

Self-contained workspace for the ICDM 2026 Applied Track revision (Jun 6 deadline; SENTIRE'26 cascade fallback). Designed to run on a single RTX 4090 Laptop with 16 GB VRAM.

The full plan lives at `../submissions/ICDM_Realistic_Revision_Plan.md`. This README is the operational shortcut.

## Layout

```
2026_icdm/
├── configs/
│   └── default.yaml         ← all knobs (split, seeds, embargo, HPO budgets, ablations)
├── dataset/
│   ├── training_features/   ← 5 stock CSVs, pre-built features (Close, sentiment, 10 Combi pairs)
│   ├── daily_sentiment/     ← per-stock daily News / Social sentiment averages
│   └── _dataset_stats.csv   ← per-stock corpus statistics for the dataset table
├── src/
│   ├── config.py            ← loads configs/default.yaml
│   ├── data.py              ← load + chronological split + embargo + sentiment recomposition
│   ├── metrics.py           ← Sharpe, MDD, Sortino, Calmar, DSR, PBO, block bootstrap CIs
│   ├── backtest.py          ← long/short, non-overlapping h-day positions, txn costs
│   ├── learned_alpha.py     ← learned-α fusion (per-stock + global) with L2 prior
│   ├── hpo_traditional.py   ← LR/SVM/RF/GB/XGB/LGBM HPO with Val_* + Test_* columns
│   ├── hpo_lstm.py          ← LSTM HPO (reduced grid; tune at h=5, broadcast to all horizons)
│   ├── _load_features.py    ← shared loader for foundation scripts: training_features → Sentiment_S_t at α
│   ├── chronos_baseline.py  ← zero-shot Chronos-2 (CLI: invoked by chronos_runner)
│   ├── chronos_finetune.py  ← AutoGluon fine-tune (+ --use-covariates) (CLI)
│   ├── fincast_baseline.py  ← zero-shot FinCast (CLI)
│   ├── fincast_finetune.py  ← LoRA fine-tune (CLI)
│   ├── chronos_runner.py    ← scaffold printing chronos_*.py launch commands
│   ├── fincast_runner.py    ← scaffold printing fincast_*.py launch commands
│   ├── select_best_config.py← reads Val_* across raw results, writes selected_config.json
│   └── run_pipeline.py      ← end-to-end driver (steps 1-6)
├── dataset/training_features/ ← per-stock features; single source of truth for all models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_learned_alpha.ipynb
│   ├── 03_pipeline_validation.ipynb
│   ├── 04_results_aggregation.ipynb
│   ├── 05_figures.ipynb
│   └── 06_indicator_pairs_appendix.ipynb
└── results/                 ← all run outputs land here (raw/, selected_config.json, figures, tables)
```

## Dependencies

Python 3.10+. Core packages:
```
pip install numpy pandas pyyaml scikit-learn scipy
pip install scikit-optimize xgboost lightgbm
pip install tensorflow keras   # for LSTM
pip install matplotlib seaborn
pip install autogluon.timeseries   # for Chronos-2 runner (large install, ~1 GB)
```

FinCast: scripts are bundled under `src/fincast_*.py`; install the FinCast runtime per its own README (LoRA via `peft`, base weights from the original repo).

## Quickstart on the GPU laptop

```bash
cd 2026_icdm

# 0) Sanity check — verifies split, seed reproducibility, single-cell eval
jupyter notebook notebooks/03_pipeline_validation.ipynb

# 1) Fit learned-α per stock and globally on 2023 val (fast, CPU)
python -m src.run_pipeline --steps 1

# 2) Traditional ML HPO (5 stocks × 8 ablations × 3 horizons × 5 seeds × 6 models)
#    Long-running; checkpoint after every (stock, horizon, model).
python -m src.run_pipeline --steps 2

# 3) LSTM HPO with the reduced grid
python -m src.run_pipeline --steps 3

# 4) Foundation models. Two wrappers under scripts/ — each takes optional
#    positional args (MODE STOCK SEED) so the same script does pretests AND
#    the full sweep. Logs land in logs/<tag>_<timestamp>.log, and each printed
#    command is wrapped in `conda run -n <env>` so you don't need to switch
#    envs manually.
#
#    Chronos-2 (needs chronos-forecasting + autogluon.timeseries in stock-prediction env):
bash scripts/run_chronos.sh chronos2_finetuned_cov AAPL 42   # pretest: one cell
bash scripts/run_chronos.sh                                  # full sweep: 4 modes × 5 × 3 = 60 cells
#
#    FinCast (set FINCAST_PATH + FINCAST_WEIGHTS once; defaults point to the
#    Multimodal-Stock-Movement-Prediction checkout):
export FINCAST_PATH=/path/to/FinCast-fts/src
export FINCAST_WEIGHTS=/path/to/FinCast-fts/model_weights/v1.pth
bash scripts/run_fincast.sh fincast_finetuned AAPL 42        # pretest
bash scripts/run_fincast.sh fincast_finetuned                # skip zero-shot if already done
bash scripts/run_fincast.sh                                  # full sweep: 2 modes × 5 × 3 = 30 cells

# Long overnight runs: detach with nohup and tail the log
nohup bash scripts/run_chronos.sh > /dev/null 2>&1 &
tail -f logs/chronos_*_$(date +%Y%m%d)*.log

# 5) Lock model/horizon selection by val scores
python -m src.run_pipeline --steps 5

# 6) Aggregate everything into results/all_results.csv
python -m src.run_pipeline --steps 6

# 7) Build the manuscript tables + figures
jupyter notebook notebooks/04_results_aggregation.ipynb
jupyter notebook notebooks/05_figures.ipynb
jupyter notebook notebooks/06_indicator_pairs_appendix.ipynb
```

## Three things to verify on the GPU laptop before launching the big grid

1. **Chronos-2 16 GB pretest.** Run a single fine-tune-with-covariates on AAPL:
   ```
   bash scripts/run_chronos.sh chronos2_finetuned_cov AAPL 42
   ```
   Watch `nvidia-smi`. If OOM, drop `chronos_fine_tune_batch_size` from 8 → 4 in `configs/default.yaml`.

2. **LSTM determinism.** Run notebooks/03 cell 4 twice and confirm Val_ROC_AUC matches bit-for-bit.

3. **Daily-sentiment correlation.** Notebook 01 reports `corr(train Weighted Sentiment, our S_70_30)`. If correlations are < 0.7, the new pipeline diverges from the original aggregation in a non-trivial way — the original `combine_sentiment_scores_per_date.ipynb` uses article-weighted aggregation `Σ s·w / Σ w`; this codebase uses the cleaner `0.7·mean(news) + 0.3·mean(social)`. Decide which to keep and document the choice in §III.C.2 of the manuscript.

## Hard-stop trigger (revision plan §1.2)

Switch to a down-scoped Jun 6 submission if any of these is true on morning of D9:
- Controlled-table mean Sharpe across 5 stocks < 0.5
- Fewer than 3 of 5 stocks have completed the 5-seed grid
- Chronos-2 cov fine-tune has not produced clean 3-seed results for ≥ 3 stocks
- `selected_config.json` plumbing is not end-to-end locked

Down-scope = single seed for LSTM, zero-shot Chronos-2 only. Still submit — the SENTIRE cascade catches whatever ICDM rejects.

## What lives elsewhere

- **Raw sentiment files (~1 GB)** — `../news_social_media_data/final_combined/`. Not copied here to save disk. Daily aggregates in `dataset/daily_sentiment/` are the derived signal you need; the raw files are only useful if you want to re-run FinBERT scoring.
- **Existing single-seed results** — `../paper/*.csv`. Useful as a sanity-check reference (the new pipeline's single-seed AUC should be in the same ballpark as the old per-stock numbers).
- **Original HPO scripts** — `../paper/*_hyperparameter_tuning.py` and `../paper/v2/*`. Superseded by `src/hpo_*.py` here; kept in the old location for diff reference.

## Submission checklist for D20 (Jun 5)

- [ ] 10-page check (including references)
- [ ] `\title{}` updated to one of the SENTIRE-friendly candidates
- [ ] Contributions list leads with the learned reliability-weighted sentiment fusion, not "beats foundation models"
- [ ] §IV.E foundation-model comparison is self-contained with no upstream cross-references
- [ ] Limitations section addresses 5-stock scope, 1-year test, single regime
- [ ] **Opt in to the SENTIRE cascade on the ICDM submission form** ← critical
- [ ] Anonymization check (ICDM Applied Track is single-blind; double-check the venue's rules for the year)
- [ ] Reproducibility artifact: this folder + a Dockerfile + frozen requirements.txt
