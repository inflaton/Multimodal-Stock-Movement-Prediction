"""Central configuration loader.

All scripts in src/ import settings through this module so the experimental
protocol stays consistent (split, seeds, embargo, etc.).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]   # repo root
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML config. Defaults to configs/default.yaml."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["__path__"] = str(cfg_path)
    cfg["__repo_root__"] = str(REPO_ROOT)
    return cfg


# -------------------- common derived paths --------------------

def dataset_dir() -> Path:
    return REPO_ROOT / "dataset"


def training_features_dir() -> Path:
    return dataset_dir() / "training_features"


def results_dir() -> Path:
    d = REPO_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def selected_config_path() -> Path:
    """Output path for the val-based model/horizon selection lock.

    All downstream scoring must read this file (Section 2.1).
    """
    return results_dir() / "selected_config.json"


# -------------------- typed accessors --------------------

def stocks(cfg: Dict[str, Any]) -> List[str]:
    return list(cfg["stocks"])


def horizons(cfg: Dict[str, Any]) -> List[int]:
    return list(cfg["horizons"])


def ml_seeds(cfg: Dict[str, Any]) -> List[int]:
    return list(cfg["ml_seeds"])
