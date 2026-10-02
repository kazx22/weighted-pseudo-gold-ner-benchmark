
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import load_jsonl, save_jsonl

ROOT = Path(__file__).resolve().parent.parent
MODEL_KEYS = ("scispacy", "biobert", "pubmedbert", "clinicalbert", "d4data")
OFFICIAL_DIR = ROOT / "data" / "processed" / "biored"
CLEAN_DIR = ROOT / "data" / "processed" / "biored_overlap_excluded"


def _normalise_split(value: str) -> str:
    value = value.strip().lower()
    if value in {"dev", "development", "validation", "val"}:
        return "dev"
    if value in {"test", "testing"}:
        return "test"
    raise ValueError("Use --split dev or --split test.")


def filter_split(split: str) -> None:
    split = _normalise_split(split)
    retained_docs = CLEAN_DIR / f"biored_{split}_docs.jsonl"
    if not retained_docs.exists():
        raise FileNotFoundError(
            f"Missing overlap-excluded documents: {retained_docs}\n"
            "Run: python -m src.biored_overlap_exclusion"
        )

    retained_ids = {int(row["row_id"]) for row in load_jsonl(retained_docs)}
    if not retained_ids:
        raise ValueError(f"No retained BioRED documents found in {retained_docs}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODEL_KEYS:
        source = OFFICIAL_DIR / f"{model}_{split}_entities_biored.jsonl"
        target = CLEAN_DIR / f"{model}_{split}_entities_biored.jsonl"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing official BioRED predictions: {source}\n"
                "Run the official BioRED conventional-model stage first."
            )
        rows = load_jsonl(source)
        kept = [row for row in rows if int(row["row_id"]) in retained_ids]
        save_jsonl(kept, target)
        print(f"{model:12} {len(rows):>6} official -> {len(kept):>6} retained")

    print(f"Filtered conventional predictions written for BioRED {split}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter official BioRED conventional predictions to retained PMIDs."
    )
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    args = parser.parse_args()
    filter_split(args.split)


if __name__ == "__main__":
    main()
