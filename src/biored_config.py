"""Dataset-specific paths and metadata for BioRED disease/chemical validation.

This module is intentionally separate from experiment_config.py so the BioRED
extension cannot overwrite the existing BC5CDR results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
RAW_DIR = RAW_ROOT / "biored"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "biored"
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "biored"
RESULTS_ROOT = PROJECT_ROOT / "results" / "biored"

SPLIT_ALIASES = {
    "development": "dev",
    "dev": "dev",
    "validation": "dev",
    "val": "dev",
    "testing": "test",
    "test": "test",
}

MODEL_KEYS = ("scispacy", "biobert", "pubmedbert", "clinicalbert", "bioelectra")
OPTIONAL_MODEL_KEYS = ("medgemma",)
ALL_PREDICTION_KEYS = MODEL_KEYS + OPTIONAL_MODEL_KEYS
MODEL_DISPLAY_NAMES = {
    "scispacy": "scispaCy",
    "biobert": "BioBERT",
    "pubmedbert": "PubMedBERT",
    "clinicalbert": "ClinicalBERT",
    "bioelectra": "BioELECTRA",
    "medgemma": "MedGemma 1.5 4B",
}

RAW_FILENAMES = {
    "dev": "Dev.PubTator",
    "test": "Test.PubTator",
}


def normalize_split(value: str, *, allow_train: bool = False) -> str:
    del allow_train  # BioRED extension deliberately uses development and test only.
    split = SPLIT_ALIASES.get(value.strip().lower())
    if split is None:
        raise ValueError(f"Unknown BioRED split '{value}'. Use dev or test.")
    return split


def docs_file(split: str) -> Path:
    split = normalize_split(split)
    return PROCESSED_DIR / f"biored_{split}_docs.jsonl"


def gold_entities_file(split: str) -> Path:
    split = normalize_split(split)
    return PROCESSED_DIR / f"biored_{split}_entities.jsonl"


def prediction_file(model_key: str, split: str) -> Path:
    split = normalize_split(split)
    if model_key not in ALL_PREDICTION_KEYS:
        raise ValueError(f"Unknown model key: {model_key}")
    return PROCESSED_DIR / f"{model_key}_{split}_entities_biored.jsonl"


def pseudo_gold_file(kind: str, split: str) -> Path:
    split = normalize_split(split)
    if kind not in {"majority", "weighted"}:
        raise ValueError("Pseudo-gold kind must be 'majority' or 'weighted'.")
    return GOLD_DIR / f"{kind}_pseudo_gold_{split}_entities_biored.jsonl"


def medgemma_hybrid_file(split: str) -> Path:
    split = normalize_split(split)
    return GOLD_DIR / f"medgemma_hybrid_{split}_entities_biored.jsonl"


def results_dir(split: str) -> Path:
    return RESULTS_ROOT / normalize_split(split)


def figures_dir(split: str) -> Path:
    return results_dir(split) / "figures"


def medgemma_results_dir(split: str) -> Path:
    return results_dir(split) / "medgemma"


def medgemma_hybrid_results_dir(split: str) -> Path:
    return results_dir(split) / "medgemma_hybrid"


def runtime_file(split: str) -> Path:
    return results_dir(split) / "runtime.json"


def frozen_config_file() -> Path:
    return results_dir("dev") / "frozen_pseudo_gold_config.json"


def require_file(path: Path, purpose: str = "input") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")
    return path


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_runtime(
    split: str,
    model_key: str,
    *,
    total_seconds: float,
    average_seconds: float,
    document_count: int,
    entity_count: int,
) -> None:
    path = runtime_file(split)
    existing = load_json(path, default={}) or {}
    existing[model_key] = {
        "model": MODEL_DISPLAY_NAMES[model_key],
        "total_seconds": round(float(total_seconds), 6),
        "average_seconds_per_document": round(float(average_seconds), 6),
        "document_count": int(document_count),
        "entity_count": int(entity_count),
    }
    save_json(existing, path)
