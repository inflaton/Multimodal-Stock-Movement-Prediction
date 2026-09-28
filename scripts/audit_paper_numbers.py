# [CA-AUDIT] Added to reproduce the manuscript's table and prose numerical audit.
"""Audit manuscript tables and supporting numbers without retraining models.

Run: python scripts/audit_paper_numbers.py --out results/paper_number_audit.json
Requires the repository's numpy, pandas, scipy, and PyYAML dependencies.
The report includes full-precision aggregates, input hashes, and mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import yaml

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
ABLATIONS = ["technical_only", "equal_50_50", "learned_alpha_per_stock",
             "news_only", "full_70_30", "learned_alpha_global", "social_only",
             "sentiment_only"]
MODELS = ["logistic_regression", "random_forest", "xgboost", "svm",
          "gradient_boosting", "lightgbm", "lstm"]
KEYS = ["Stock", "Ablation", "Model", "Horizon", "Seed"]
METRICS = ["Test_ROC_AUC", "Test_Sharpe", "Test_WinRate", "Test_MDD"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    inputs = set()

    def read(path: str) -> Path:
        inputs.add(path)
        return ROOT / path

    cfg = yaml.safe_load(read("configs/default.yaml").read_text())
    assert cfg["stocks"] == STOCKS and cfg["horizons"] == [2, 5, 10]
    assert cfg["train_years"] == [2020, 2021, 2022]
    assert cfg["val_year"] == 2023 and cfg["test_year"] == 2024
    assert cfg["embargo_trading_days"] == 5
    assert cfg["backtest"]["fee_bps_round_trip"] == 10
    assert cfg["hpo"]["traditional_n_calls"] == cfg["hpo"]["lstm_n_calls"] == 30
    df = pd.read_csv(read("results/all_results.csv"))
    raw = pd.concat([pd.read_csv(read(f"results/raw/{model}.csv"))
                     for model in MODELS], ignore_index=True)
    expected = set(itertools.product(STOCKS, ABLATIONS, MODELS, [2, 5, 10],
                                     [42, 7, 1337, 2024, 31415]))
    assert not df.duplicated(KEYS).any()
    assert set(df[KEYS].itertuples(index=False, name=None)) == expected
    assert len(raw) == len(df) == 4200 and not raw.duplicated(KEYS).any()
    numeric = df.select_dtypes(include="number").columns.difference(KEYS)
    pd.testing.assert_frame_equal(df.set_index(KEYS)[numeric].sort_index(),
                                  raw.set_index(KEYS)[numeric].sort_index())
    assert np.isfinite(df[METRICS + ["Test_AnnualReturn"]]).all().all()

    # Invert src.metrics.annual_return for each run BEFORE averaging. The saved
    # AnnualReturn is (1 + cumulative_return) ** (252 / trade_count) - 1.
    # Do not relabel that stored, trade-count-scaled number as cumulative return.
    read("src/metrics.py")
    assert (df.Test_AnnualReturn > -1).all() and (df.Test_Trades > 0).all()
    df["Test_CumulativeReturn"] = np.expm1(
        np.log1p(df.Test_AnnualReturn) * df.Test_Trades /
        cfg["backtest"]["trading_days_per_year"])
    means = df.groupby("Ablation")[METRICS + ["Test_CumulativeReturn"]].mean()
    stds = df.groupby("Ablation").Test_Sharpe.std(ddof=1)
    model_means = df.groupby("Model")[METRICS[:3]].mean()
    paired = {}
    pair_keys = ["Stock", "Model", "Horizon", "Seed"]
    baseline = df[df.Ablation.eq("technical_only")].set_index(pair_keys)
    for a in ABLATIONS[1:]:
        diff = baseline.Test_Sharpe - df[df.Ablation.eq(a)].set_index(pair_keys).Test_Sharpe
        assert len(diff) == 525 and diff.notna().all()
        paired[a] = {"mean_gap": float(diff.mean()),
                     "p_two_sided": float(wilcoxon(diff, alternative="two-sided",
                                                   zero_method="wilcox").pvalue),
                     "baseline_win_fraction": float((diff > 0).mean())}

    selected = json.loads(read("results/selected_config.json").read_text())
    cell = df.groupby(["Stock", "Ablation", "Model", "Horizon"]).mean(numeric_only=True)
    locked = {}
    peek = {}
    for stock, a in itertools.product(STOCKS, ABLATIONS):
        candidates = cell.loc[(stock, a)]
        for criterion, metric in [("val_auc", "Val_ROC_AUC"), ("val_sharpe", "Val_Sharpe")]:
            pick = selected[stock][a][criterion]
            chosen = candidates.loc[(pick["Model"], pick["Horizon"])]
            assert np.isclose(chosen[metric], candidates[metric].max(), atol=1e-12, rtol=0)
            assert np.isclose(chosen[metric], pick["mean_val"], atol=1e-12, rtol=0)
            assert pick["n_seeds"] == 5
        pick = selected[stock][a]["val_sharpe"]
        locked[stock, a] = candidates.loc[(pick["Model"], pick["Horizon"])][METRICS[:3]]
        peek[stock, a] = candidates.loc[candidates.Test_Sharpe.idxmax()][METRICS[:3]]
    locked_mean = {a: pd.DataFrame([locked[s, a] for s in STOCKS]).mean() for a in ABLATIONS}
    peek_mean = {a: pd.DataFrame([peek[s, a] for s in STOCKS]).mean() for a in ABLATIONS}

    data_rows, split_rows, data_details = [], [], {}
    for stock in STOCKS:
        d = pd.read_csv(read(f"dataset/training_features/{stock}.csv"))
        d["Date"] = pd.to_datetime(d.Date, dayfirst=True)
        d = d[d.Date.dt.year.between(2020, 2024)].sort_values("Date")
        news, social = "News Sentiment Score", "Social Sentiment Score"
        assert len(d) == 1257 and d.Date.nunique() == 1257
        indicators = [c for c in d if c.lstrip().startswith("Combi") and "Days" not in c]
        assert len(indicators) == 10
        data_rows.append([d[news].notna().mean(), d[social].notna().mean(),
                          d[news].mean(), d[social].mean()])
        split_values = {}
        for split, years in [("train", [2020, 2021, 2022]), ("val", [2023]), ("test", [2024])]:
            g = d[d.Date.dt.year.isin(years)]
            values = [g[news].notna().mean(), g[social].notna().mean(), g[news].mean()]
            split_rows.append(values)
            split_values[split] = {"days": len(g), "news_coverage": values[0],
                                   "social_coverage": values[1], "mean_news": values[2]}
        ret_next = d.Close.pct_change().shift(-1)
        data_details[stock] = {"days": len(d), "min_date": str(d.Date.min().date()),
                              "max_date": str(d.Date.max().date()),
                              "news_return_correlation": d[news].corr(ret_next),
                              "social_return_correlation": d[social].corr(ret_next),
                              "splits": split_values}

    alpha = json.loads(read("results/learned_alpha.json").read_text())
    pipeline_alpha = json.loads(read("results/learned_alpha_pipeline.json").read_text())
    alpha_exact = {r["stock"] or "global": r for r in pipeline_alpha}
    alpha_rows = []
    for r in alpha["per_stock"] + [dict(alpha["global"], stock="global")]:
        exact = alpha_exact[r["stock"]]
        assert np.isclose(r["alpha_news"], exact["alpha_news"])
        assert abs(r["val_auc_learned"] - exact["val_auc"]) < 0.000051
        alpha_rows.append([exact["alpha_news"], exact["val_auc"], r["delta_val_auc"]])

    foundation = {}
    for prefix in ["zero-shot_chronos-2", "multivariate_chronos-2",
                   "Chronos2_finetuned", "Chronos2_finetuned_cov", "fincast"]:
        folder = "fincast" if prefix == "fincast" else "chronos-2"
        f = pd.concat([pd.read_csv(read(f"results/{folder}/{prefix}_{s}_results.csv"))
                       for s in STOCKS], ignore_index=True)
        assert len(f) == 15 and not f.duplicated(["Stock", "Horizon"]).any()
        assert set(f[["Stock", "Horizon"]].itertuples(index=False, name=None)) == set(itertools.product(STOCKS, [2, 5, 10]))
        foundation[prefix] = f[METRICS[:3]].mean()

    # Expected table values are derived from primary saved results; rounded
    # intermediate tables are deliberately not used as numerical authorities.
    expected_tables = {}

    def formatted(rows, digits):
        return [[f"{float(value):.{precision}f}" for value, precision in zip(row, digits)]
                for row in rows]

    expected_tables["data"] = formatted(data_rows, [3] * 4)
    selection_rows = [[*peek[s, "full_70_30"][:2], *locked[s, "full_70_30"][:2]] for s in STOCKS]
    selection_rows.append([*peek_mean["full_70_30"][:2], *locked_mean["full_70_30"][:2]])
    # Table II presents Sharpe before AUC (the internal metric order is reversed).
    selection_rows = [[r[1], r[0], r[3], r[2]] for r in selection_rows]
    expected_tables["selbias"] = formatted(selection_rows, [2, 3, 2, 3])
    ablation_rows = []
    for a in ABLATIONS:
        m = means.loc[a]
        row = formatted([[m.Test_ROC_AUC, m.Test_Sharpe, stds[a], m.Test_WinRate,
                          m.Test_CumulativeReturn, m.Test_MDD]], [3, 2, 2, 3, 2, 3])[0]
        if a != "technical_only":
            row += [f"{paired[a]['mean_gap']:.2f}", f"{paired[a]['p_two_sided']:.1e}"]
        ablation_rows.append(row)
    expected_tables["ablation"] = ablation_rows
    expected_tables["aggviews"] = formatted([[means.loc[a].Test_Sharpe, locked_mean[a].Test_Sharpe]
                                             for a in ABLATIONS], [2, 2])
    expected_tables["alpha"] = formatted(alpha_rows, [2, 3, 3])
    expected_tables["models"] = formatted([model_means.loc[m].tolist() for m in MODELS], [3, 2, 3])
    expected_tables["foundation"] = formatted([r.tolist() for r in foundation.values()] +
                                               [model_means.loc["logistic_regression"].tolist()], [3, 2, 3])
    expected_tables["appendix"] = formatted(split_rows, [3, 3, 3])

    source = read("paper/latex/main.tex").read_text()
    source = "\n".join(re.split(r"(?<!\\)%", line)[0] for line in source.splitlines())
    mismatches, checked = [], 0
    for name, expected_rows in expected_tables.items():
        table = source.split(r"\label{tab:" + name + "}", 1)[1].split(r"\end{tabular}", 1)[0]
        table = table.split(r"\midrule", 1)[1]
        actual_rows = []
        for line in table.splitlines():
            if "&" not in line:
                continue
            cells = line.split("&")[2 if name == "appendix" else 1:]
            values = []
            for cell_text in cells:
                cell_text = re.sub(r"\\times\s*10\^\{(-?\d+)\}", r"e\1", cell_text)
                values += re.findall(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", cell_text)
            actual_rows.append(values)
        assert len(actual_rows) == len(expected_rows), (name, actual_rows)
        for ri, (actual, wanted) in enumerate(zip(actual_rows, expected_rows), 1):
            assert len(actual) == len(wanted), (name, ri, actual, wanted)
            for ci, (a, w) in enumerate(zip(actual, wanted), 1):
                checked += 1
                if float(a) != float(w):
                    mismatches.append({"table": name, "row": ri, "numeric_position": ci,
                                       "reported": a, "expected": w})

    horizons = df.groupby("Horizon")[["Test_ROC_AUC", "Test_Sharpe", "Test_WinRate", "Test_Trades"]].mean()
    controlled = df[df.Model.eq("lightgbm") & df.Horizon.eq(5)].groupby("Ablation").Test_Sharpe.mean()
    no_spy = df[df.Stock.ne("SPY")].groupby("Ablation")[METRICS[:2]].mean()
    saved_spy = json.loads(read("results/spy_excluded_robustness.json").read_text())
    for a in ABLATIONS:
        saved = saved_spy["summaries"]["excluding_spy"][a]
        assert saved["runs"] == 420
        assert np.isclose(saved["mean_test_sharpe"], no_spy.loc[a].Test_Sharpe, atol=1e-12, rtol=0)
        assert np.isclose(saved["mean_test_auc"], no_spy.loc[a].Test_ROC_AUC, atol=1e-12, rtol=0)
    gap = peek_mean["full_70_30"].Test_Sharpe - locked_mean["full_70_30"].Test_Sharpe
    for name in ["fig_horizon_tradeoffs.pdf", "fig_winrate_sharpe.pdf", "fig_learned_alpha.pdf"]:
        assert read(f"results/{name}").read_bytes() == read(f"paper/latex/figures/{name}").read_bytes()
    with zipfile.ZipFile(read("paper/latex/figures/study_overview_verified.pptx")) as deck:
        slide = ET.fromstring(deck.read("ppt/slides/slide1.xml"))
        figure_text = "".join(e.text or "" for e in slide.iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}t"))
        assert "8 feature sets × 7 models × 5 stocks × 3 horizons × 5 seeds = 4,200" in figure_text
        assert "Fit on training; tune and select model family + horizon on the validation window" in figure_text
    read("paper/latex/figures/study_overview_verified.pdf")
    # Prose bounds and numerical interpretations (table values are checked above).
    assert paired["full_70_30"]["p_two_sided"] < 1e-20
    assert max(p["p_two_sided"] for p in paired.values()) < 2e-7
    assert max(paired[a]["p_two_sided"] for a in ["learned_alpha_per_stock", "learned_alpha_global"]) < 1e-9
    assert max(abs(d[channel]) for d in data_details.values()
               for channel in ["news_return_correlation", "social_return_correlation"]) <= .06
    assert sum(r[0] >= .7 for r in alpha_rows[:-1]) == 4
    assert max(r["delta_val_auc"] for r in alpha["per_stock"] if r["stock"] != "META") < .006
    assert all(locked_mean["technical_only"].Test_Sharpe > locked_mean[a].Test_Sharpe for a in ABLATIONS[1:])
    assert (means[METRICS].idxmax() == "technical_only").all()
    assert no_spy.Test_Sharpe.idxmax() == no_spy.Test_ROC_AUC.idxmax() == "technical_only"
    assert no_spy.drop("technical_only").Test_Sharpe.idxmax() == "learned_alpha_per_stock"
    assert no_spy.drop("technical_only").Test_ROC_AUC.idxmax() == "news_only"
    assert horizons.Test_Sharpe.idxmax() == 2 and horizons.Test_ROC_AUC.idxmax() == 5
    assert (model_means.idxmax() == "logistic_regression").all()
    assert all(r.Test_Sharpe < 0 and .47 < r.Test_ROC_AUC < .50 for r in foundation.values())
    report = {
        "scope": "Saved-run numerical audit; no model retraining or trade-level backtest rerun. "
                 "All eight manuscript tables are compared automatically; supporting aggregates "
                 "below provide the evidence for the prose and figure audit.",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "grid": {"task_specific_runs": len(df), "foundation_runs": 75,
                 "runs_per_ablation": 525, "runs_per_model": 600,
                 "runs_per_ablation_without_spy": 420, "validation_selections_checked": 80},
        "table_numbers_checked": checked, "table_mismatches": mismatches,
        "prose_bounds_and_rankings": "passed",
        "expected_table_values": expected_tables,
        "ablation_means": means.to_dict(orient="index"), "paired_comparisons": paired,
        "model_means": model_means.to_dict(orient="index"),
        "foundation_means": {k: v.to_dict() for k, v in foundation.items()},
        "data": data_details, "learned_alpha_table_full_precision": alpha_rows,
        "prose": {"selection_sharpe_gap": gap,
                  "selection_gap_fraction_of_test_peek": gap / peek_mean["full_70_30"].Test_Sharpe,
                  "controlled_lightgbm_h5_sharpe": controlled.to_dict(),
                  "horizon_means": horizons.to_dict(orient="index"),
                  "win_rate_sharpe_pearson": df.Test_WinRate.corr(df.Test_Sharpe),
                  "runs_win_rate_above_half_negative_sharpe": int(((df.Test_WinRate > .5) & (df.Test_Sharpe < 0)).sum()),
                  "spy_excluded_means": no_spy.to_dict(orient="index"),
                  "selection_gaps_by_ablation": {a: peek_mean[a].Test_Sharpe - locked_mean[a].Test_Sharpe for a in ABLATIONS}},
        "inputs_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in sorted(inputs)},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Checked {checked} numbers in eight tables; {len(mismatches)} mismatches.")
    for mismatch in mismatches:
        print(json.dumps(mismatch))
    print("Validated 4,200 task-specific cells, 75 foundation cells, and 80 validation selections.")
    raise SystemExit(bool(mismatches))


if __name__ == "__main__":
    main()
