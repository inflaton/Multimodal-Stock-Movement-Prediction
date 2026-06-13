"""FinCast wrapper.

FinCast is price-only by design (its released artifact doesn't accept
covariates), but the bundled scripts compose Sentiment_S_t the same way as
the other foundation scripts so the comparison is sentiment-aligned with the
rest of the pipeline. The §2.3 footnote about FinCast being unable to take
sentiment as a *covariate* still applies — that's an artifact-level limitation,
not a data limitation.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

from . import config as cfg_mod
from .chronos_runner import _load_learned_alpha_table, _resolve_alpha


def build_run_command(stock: str, mode: str, seed: int, cfg: Dict[str, Any],
                      alpha_news: float, conda_env: str,
                      fincast_path: Optional[str],
                      fincast_weights: Optional[str]) -> str:
    """Return the shell command for one FinCast run.

    Wraps in `conda run -n <env> --no-capture-output` and prepends FINCAST_PATH
    if known (so the bundled scripts find the FinCast-fts source). Adds
    `--model-path` if FINCAST_WEIGHTS is known.
    """
    zero_shot = mode == "fincast_zero_shot"
    script = "src/fincast_baseline.py" if zero_shot else "src/fincast_finetune.py"
    env_prefix = f"FINCAST_PATH={fincast_path} " if fincast_path else ""
    model_arg = f"--model-path {fincast_weights} " if fincast_weights else ""
    horizons_str = " ".join(str(h) for h in cfg["horizons"])
    # Fine-tune-only flags. fincast_baseline.py rejects unknown args, so we omit
    # --lr / --epochs in zero-shot mode.
    ft_flags = "" if zero_shot else (
        f"--lr {cfg['hpo']['fincast_lora_lr']} "
        f"--epochs {cfg['hpo']['fincast_lora_epochs']} "
    )
    # --output-dir routes per-stock model checkpoints + result CSV under results/fincast/
    return (f"{env_prefix}conda run -n {conda_env} --no-capture-output "
            f"python {script} --stock {stock} --seed {seed} "
            f"--data-dir dataset/training_features --output-dir results/fincast "
            f"{model_arg}{ft_flags}"
            f"--horizons {horizons_str} "
            f"--alpha-news {alpha_news:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["fincast_zero_shot", "fincast_finetuned"])
    parser.add_argument("--stock", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--alpha-scope", default="per_stock",
                        choices=["per_stock", "global", "fixed_70_30"],
                        help="Where to read α from. per_stock (default) uses "
                             "results/learned_alpha.json per-stock; global uses "
                             "the shared α; fixed_70_30 uses the legacy 0.7.")
    parser.add_argument("--conda-env", default="fincast_v1",
                        help="Conda env to run the FinCast script in. The "
                             "default (fincast_v1) has torch+peft+transformers.")
    parser.add_argument("--fincast-path", default=os.environ.get("FINCAST_PATH"),
                        help="Path to FinCast-fts/src (else $FINCAST_PATH env var). "
                             "Required for the bundled scripts to import "
                             "FinCast's source modules.")
    parser.add_argument("--fincast-weights", default=os.environ.get("FINCAST_WEIGHTS"),
                        help="Path to FinCast v1.pth model weights (else "
                             "$FINCAST_WEIGHTS env var). Passed as --model-path "
                             "to the foundation script.")
    args = parser.parse_args()

    cfg = cfg_mod.load_config()
    stocks = [args.stock] if args.stock else cfg_mod.stocks(cfg)
    seeds = args.seeds if args.seeds else cfg["foundation_seeds"]
    table = _load_learned_alpha_table()
    if args.alpha_scope != "fixed_70_30" and table is None:
        print("WARNING: results/learned_alpha.json not found; "
              "falling back to fixed 0.7 weighting.\n")
    if not args.fincast_path:
        print("WARNING: --fincast-path / $FINCAST_PATH not set. The bundled "
              "scripts will fall back to scripts/FinCast-fts/src, which "
              "probably does not exist on this machine.\n")
    if not args.fincast_weights:
        print("WARNING: --fincast-weights / $FINCAST_WEIGHTS not set. The "
              "bundled scripts will use their own default --model-path, "
              "which probably does not exist on this machine.\n")

    print("Run commands (execute from the repo root on GPU laptop):\n")
    for s in stocks:
        alpha = _resolve_alpha(s, args.alpha_scope, 0.7, table)
        for seed in seeds:
            print("  " + build_run_command(s, args.mode, seed, cfg, alpha,
                                           args.conda_env, args.fincast_path,
                                           args.fincast_weights))


if __name__ == "__main__":
    main()
