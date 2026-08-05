from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[2]
REBUTTAL_ROOT = ROOT / "rebuttal"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    base_name = cfg.pop("base", None)
    if base_name:
        base_path = (cfg_path.parent / base_name).resolve()
        cfg = _deep_merge(load_config(str(base_path)), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["paths"] = cfg.get("paths", {})
    cfg["paths"]["repo_root"] = str(ROOT)
    cfg["paths"]["rebuttal_root"] = str(REBUTTAL_ROOT)
    return cfg


def resolved_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()
