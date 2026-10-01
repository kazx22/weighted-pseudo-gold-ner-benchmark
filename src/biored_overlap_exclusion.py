"""Create BioRED sensitivity splits with every BC5CDR-overlapping PMID removed.

The official BioRED experiment remains untouched.  This module reads the already
parsed official BioRED dev/test files, compares PMIDs against all three BC5CDR
splits, writes filtered BioRED dev/test inputs to a separate variant directory,
and records a transparent overlap audit.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.utils import load_jsonl, save_jsonl

ROOT = Path(__file__).resolve().parent.parent
BC5_RAW = ROOT / "data" / "raw" / "bc5cdr"
BIORED_OFFICIAL = ROOT / "data" / "processed" / "biored"
BIORED_CLEAN = ROOT / "data" / "processed" / "biored_overlap_excluded"
AUDIT_FILE = ROOT / "results" / "biored_overlap_excluded" / "overlap_audit.json"

BC5_FILES = {
    "train": BC5_RAW / "CDR_TrainingSet.PubTator.txt",
    "dev": BC5_RAW / "CDR_DevelopmentSet.PubTator.txt",
    "test": BC5_RAW / "CDR_TestSet.PubTator.txt",
}


def _bc5_documents(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing BC5CDR raw file: {path}")
    text = path.read_text(encoding="utf-8")
    documents: dict[int, str] = {}
    for block in text.strip().split("\n\n"):
        lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
        if len(lines) < 2 or "|t|" not in lines[0] or "|a|" not in lines[1]:
            continue
        pmid = int(lines[0].split("|t|", 1)[0])
        title = lines[0].split("|t|", 1)[1]
        abstract = lines[1].split("|a|", 1)[1]
        documents[pmid] = title + " " + abstract
    return documents


def _official_paths(split: str) -> tuple[Path, Path]:
    return (
        BIORED_OFFICIAL / f"biored_{split}_docs.jsonl",
        BIORED_OFFICIAL / f"biored_{split}_entities.jsonl",
    )


def _clean_paths(split: str) -> tuple[Path, Path]:
    return (
        BIORED_CLEAN / f"biored_{split}_docs.jsonl",
        BIORED_CLEAN / f"biored_{split}_entities.jsonl",
    )


def main() -> None:
    bc5_by_split = {name: _bc5_documents(path) for name, path in BC5_FILES.items()}
    membership: dict[int, list[str]] = defaultdict(list)
    for split_name, docs in bc5_by_split.items():
        for pmid in docs:
            membership[pmid].append(split_name)

    audit: dict = {
        "rule": "exclude any BioRED PMID present in BC5CDR train, dev, or test",
        "comparison_key": "PMID",
        "bc5cdr_split_sizes": {k: len(v) for k, v in bc5_by_split.items()},
        "splits": {},
    }

    BIORED_CLEAN.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        docs_path, entities_path = _official_paths(split)
        if not docs_path.exists() or not entities_path.exists():
            raise FileNotFoundError(
                "Official BioRED parsed inputs are missing. Run src.parse_biored first."
            )
        docs = load_jsonl(docs_path)
        entities = load_jsonl(entities_path)
        doc_map = {int(doc["row_id"]): str(doc["full_text"]) for doc in docs}
        overlap_ids = sorted(pmid for pmid in doc_map if pmid in membership)
        retained_ids = {pmid for pmid in doc_map if pmid not in membership}

        filtered_docs = [doc for doc in docs if int(doc["row_id"]) in retained_ids]
        filtered_entities = [
            entity for entity in entities if int(entity["row_id"]) in retained_ids
        ]
        out_docs, out_entities = _clean_paths(split)
        save_jsonl(filtered_docs, out_docs)
        save_jsonl(filtered_entities, out_entities)

        by_bc5_split = {
            name: sum(1 for pmid in overlap_ids if pmid in bc5_by_split[name])
            for name in ("train", "dev", "test")
        }
        exact_text_matches = 0
        text_mismatches: list[int] = []
        rows: list[dict] = []
        for pmid in overlap_ids:
            matched_splits = sorted(membership[pmid])
            exact = any(doc_map[pmid] == bc5_by_split[s][pmid] for s in matched_splits)
            if exact:
                exact_text_matches += 1
            else:
                text_mismatches.append(pmid)
            rows.append(
                {
                    "pmid": pmid,
                    "bc5cdr_splits": matched_splits,
                    "title_abstract_exact_match_in_at_least_one_split": exact,
                }
            )

        audit["splits"][split] = {
            "original_documents": len(docs),
            "overlap_documents": len(overlap_ids),
            "retained_documents": len(filtered_docs),
            "original_target_entities": len(entities),
            "retained_target_entities": len(filtered_entities),
            "overlap_by_bc5cdr_split": by_bc5_split,
            "exact_title_abstract_matches": exact_text_matches,
            "text_mismatch_pmids": text_mismatches,
            "overlap_documents_detail": rows,
            "output_docs": str(out_docs),
            "output_entities": str(out_entities),
        }

        print(
            f"BioRED {split}: {len(docs)} official -> {len(filtered_docs)} retained; "
            f"removed {len(overlap_ids)} BC5CDR-overlapping PMIDs."
        )

    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_FILE.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Overlap audit: {AUDIT_FILE}")


if __name__ == "__main__":
    main()
