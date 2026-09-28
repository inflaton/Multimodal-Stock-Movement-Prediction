# Manuscript numerical audit

Audited on 2026-09-28 against the saved experiment outputs and processed feature files. No models were retrained, weights refitted, or configurations reselected.

## Reproduction

From the repository root, with the repository's NumPy, pandas, SciPy, and PyYAML dependencies installed:

```sh
python scripts/audit_paper_numbers.py --out results/paper_number_audit.json
```

The audit checks 224 numerical entries across all eight manuscript tables, the complete 4,200-cell task-specific grid, all 75 saved foundation-model cells, and all 80 validation selections (five instruments, eight feature configurations, two selection criteria). The report records full-precision aggregates, package versions, and input hashes. The consolidated task-specific results match the seven source CSVs under `results/raw/`.

## Numerical corrections

Values are rounded once from the most precise saved source, rather than from already-rounded summary tables.

| Location | Quantity | Previous | Corrected | Source value |
| --- | --- | ---: | ---: | ---: |
| Table I | META mean news sentiment | 0.522 | 0.521 | 0.521458393693 |
| Table III | Technical-only minus news-only mean Sharpe | +0.58 | +0.57 | 0.574781509302 |
| Table V | TSLA learned-weight validation AUC | 0.585 | 0.584 | 0.584498438985 |
| Table VI | Gradient Boosting mean Sharpe | 0.11 | 0.10 | 0.104799649 |
| Table VIII | NVDA training mean news sentiment | 0.482 | 0.481 | 0.481493368812 |
| Table VIII | TSLA training mean news sentiment | 0.513 | 0.512 | 0.512469213517 |

Other corrections arising from the prose and figure audit:

- The common significance bound is **p < 2 × 10⁻⁷**, since the equal-weight comparison has p = 1.135304508 × 10⁻⁷.
- The four smaller learned-weight AUC gains are **less than 0.006**; NVDA's saved gain is 0.0053, which exceeds the former bound of 0.005.
- Win rate and Sharpe are **strongly positively correlated**, with Pearson r = 0.829828417. There are nevertheless 213 saved runs with win rate above 0.5 and negative Sharpe. The text and Figure 3 caption now distinguish these facts.
- Mean Sharpe is highest at horizon 2, but mean AUC peaks at horizon 5 (0.536494677). The horizon paragraph and Figure 2 caption now distinguish these metrics.
- Fine-tuning Chronos-2 increases AUC in both input settings. Sharpe falls from −0.38 to −0.54 without covariates and rises from −0.61 to −0.02 with covariates. The former blanket statement that fine-tuning helps was replaced.
- Figure 4's caption now reports the default-prior weight range of 0.54–0.84 and the pull toward 0.7 under stronger penalties, rather than claiming stability across the entire sensitivity sweep.
- Figure 1's grid equation now includes the previously omitted **five-stock factor**: 8 × 7 × 5 × 3 × 5 = 4,200. Its protocol note now distinguishes fitting on training data from tuning and selection on validation data. The paper uses a vector export of the corrected editable source.
- The universe is described as four large-cap U.S. stocks and the SPY equity ETF; the number of evaluated instruments remains five.

## Verified quantities and evidence

| Manuscript location | Quantities verified | Evidence |
| --- | --- | --- |
| Abstract, Introduction, V-A, Conclusion | Validation-selected Sharpe 1.397419210, test-selected Sharpe 2.294242067; gap 0.896822857, equal to 39.090158% of the test-selected value | Seed-mean model/horizon cells in `all_results.csv`, plus `selected_config.json` |
| III–IV, Figure 1 | Train 2020–2022, validation 2023, test 2024; five-day embargo; horizons 2/5/10; five seeds; 30 HPO evaluations; two LSTM layers; ten indicator-pair columns; 10 bps round-trip costs | `configs/default.yaml`, `src/data.py`, `src/hpo_traditional.py`, `src/hpo_lstm.py`, processed feature files |
| Table I, IV-A | 1,257 dates per instrument; coverage and sentiment means; news/next-return correlations approximately −0.047 to +0.054 | Direct aggregation of `dataset/training_features/*.csv`, restricted to 2020–2024 |
| Table II | Per-instrument selected Sharpe/AUC and their means | Test-peek maximum and saved validation choices, both averaged across seeds before comparing models/horizons |
| Table III, V-B | Eight grid means, Sharpe sample SDs, cumulative returns, drawdowns, paired Sharpe gaps, and all seven two-sided Wilcoxon p-values | 525 matched cells per configuration in `all_results.csv` |
| V-B | Technical-only wins approximately 57–70% of matched cells; controlled LightGBM/horizon-5 Sharpe 0.952705555 versus 0.230058896 | Matched-cell comparisons and the prespecified controlled subset |
| Table IV | All eight grid means and validation-selected per-stock means | Full grid and saved validation lock |
| Table V, V-C, Figure 4 | Per-stock/global weights, validation AUCs, gains, prior strength 0.1, and saved sensitivity values | `learned_alpha_pipeline.json` (full-precision AUC), `learned_alpha.json` (baselines and sensitivity) |
| Table VI, V-D | 600 runs per family; all model-family means and LR's leading AUC, Sharpe, and win rate | Full grid grouped by model |
| Table VII, V-E | Five configurations × 15 runs; all foundation-model means | 25 CSVs in `results/chronos-2/` and `results/fincast/` |
| V-F, Figures 2–3 | Horizon means, trade counts, win rates, and win-rate/Sharpe correlation | Full task-specific grid; embedded plot PDFs match the corresponding saved result PDFs byte-for-byte |
| VII | SPY test-news coverage 102/251 = 0.406374502; 420 runs per feature configuration after exclusion; Sharpe 1.048301334 / 0.393382501 / 0.581478296; AUC 0.542809496 / 0.539636560 | Processed SPY features and the SPY-excluded saved-result subset |
| Table VIII | All 45 coverage/mean entries across 15 instrument/split rows | Direct aggregation of the processed feature files |

Table III's cumulative returns are recovered **per run before averaging** by inverting the archived metric calculation in `src/metrics.py`:

`cumulative_return = (1 + Test_AnnualReturn) ** (Test_Trades / 252) - 1`.

The archived `Test_AnnualReturn` uses trade count in its annualization exponent. Averaging that column directly would not reproduce the cumulative-return column. The audit leaves the historical experiment files and metric implementation unchanged.

The 0.65 SPY-excluded Sharpe gap is calculated before rounding: 1.048301334 − 0.393382501 = 0.654918834. The displayed endpoints 1.05 and 0.39 therefore should not be subtracted to check the last decimal.

## External numerical attribution

The Introduction's claim of prior Sharpe ratios above 2 after transaction costs is supported by Kirtac and Germano's reported 3.05 after 10 bps costs. [Author institution's record](https://discovery.ucl.ac.uk/id/eprint/10190583/).

The exact 70:30 weighting could not be substantiated from the former Nyakurukwa–Seetharam citation, whose bibliographic details were mismatched. The user confirmed that no intended source was available. The manuscript now treats 70:30 and its regularization target as this study's fixed design choices. Related work instead cites the verified Seetharam–Nyakurukwa paper for the narrower finding that news and social sentiment are distinct sources with varying information flows. [Institution-hosted paper](https://wiredspace.wits.ac.za/bitstreams/b289b360-1390-4131-9f18-f58026bf34ff/download).

## Audit boundary

This establishes consistency with the saved runs, not an independent reproduction of training, raw-text sentiment extraction, or trade-level backtests. Learned-weight baseline AUCs and sensitivity values were checked against their saved outputs; the training fits were not rerun. Wilcoxon values reproduce the paper's existing matched-cell procedure; reproducing them does not independently establish that its dependence assumptions hold. A full bibliographic audit beyond the numerical attributions above was outside this audit.

## Final PDF verification

The rebuilt `paper/paper.pdf` has eight US-letter pages. All pages were visually inspected; all fonts are embedded and none are Type 3. The final build has no unresolved citations or cross-references and no overfull boxes. The PDF has no annotations, bookmarks, attachments, active actions, or encryption. Benign underfull-line, italic-small-caps fallback, and column-balancing warnings remain; the affected layout was checked visually. Event-specific IEEE PDF eXpress certification remains the separate submission step listed in `camera_ready_checklist.md`.
