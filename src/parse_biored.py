"""Parse the official BioRED PubTator corpus for DISEASE/CHEMICAL validation.

Accepted raw layouts (no manual extraction required):
  data/raw/BIORED.zip
  data/raw/biored/BIORED.zip
  data/raw/BioRED/Dev.PubTator + Test.PubTator
  data/raw/biored/BioRED/Dev.PubTator + Test.PubTator
  data/raw/biored/Dev.PubTator + Test.PubTator

Only BioRED's DiseaseOrPhenotypicFeature and ChemicalEntity annotations are
kept. Other BioRED entity classes and relation rows are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

from src.biored_config import (
    RAW_DIR,
    RAW_FILENAMES,
    RAW_ROOT,
    docs_file,
    gold_entities_file,
    normalize_split,
    results_dir,
    save_json,
)
from src.utils import save_jsonl

LABEL_MAP = {
    "DiseaseOrPhenotypicFeature": "DISEASE",
    "ChemicalEntity": "CHEMICAL",
    # Accepted aliases make the parser tolerant of PubTator variants while
    # preserving the same two-label task.
    "Disease": "DISEASE",
    "Chemical": "CHEMICAL",
}


def _candidate_raw_paths(split: str) -> list[Path]:
    filename = RAW_FILENAMES[split]
    return [
        RAW_DIR / filename,
        RAW_DIR / "BioRED" / filename,
        RAW_ROOT / "BioRED" / filename,
        RAW_ROOT / filename,
    ]


def _find_existing_pubtator(split: str) -> Path | None:
    filename = RAW_FILENAMES[split]
    for path in _candidate_raw_paths(split):
        if path.exists():
            return path

    if RAW_DIR.exists():
        matches = sorted(RAW_DIR.rglob(filename))
        if matches:
            return matches[0]
    return None


def _zip_candidates() -> list[Path]:
    candidates = [RAW_DIR / "BIORED.zip", RAW_ROOT / "BIORED.zip"]
    if RAW_DIR.exists():
        candidates.extend(sorted(RAW_DIR.glob("*.zip")))
    if RAW_ROOT.exists():
        candidates.extend(sorted(RAW_ROOT.glob("BIORED*.zip")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _extract_from_zip(split: str) -> Path | None:
    filename = RAW_FILENAMES[split]
    for zip_path in _zip_candidates():
        if not zip_path.exists():
            continue
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                members = archive.namelist()
                matching = [
                    member
                    for member in members
                    if Path(member).name.lower() == filename.lower()
                ]
                if not matching:
                    continue
                member = matching[0]
                output_dir = RAW_DIR / "_extracted"
                output_dir.mkdir(parents=True, exist_ok=True)
                extracted = Path(archive.extract(member, output_dir))
                print(f"Extracted {filename} from {zip_path} -> {extracted}")
                return extracted
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid BioRED ZIP archive: {zip_path}") from exc
    return None


def resolve_input(split: str, explicit_input: str | None = None) -> Path:
    split = normalize_split(split)
    if explicit_input:
        path = Path(explicit_input)
        if not path.exists():
            raise FileNotFoundError(f"Missing explicit BioRED input: {path}")
        return path

    existing = _find_existing_pubtator(split)
    if existing is not None:
        return existing

    extracted = _extract_from_zip(split)
    if extracted is not None:
        return extracted

    searched = "\n  - ".join(str(path) for path in _candidate_raw_paths(split))
    raise FileNotFoundError(
        f"Could not find BioRED {split} data. Put BIORED.zip in data/raw or "
        f"data/raw/biored, or extract the corpus there. Looked for:\n  - {searched}"
    )


def parse_biored(file_path: Path) -> tuple[list[dict], list[dict], dict]:
    with file_path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        raise ValueError(f"Raw BioRED file is empty: {file_path}")

    documents: list[dict] = []
    entities: list[dict] = []
    offset_errors: list[str] = []
    ignored_labels: Counter[str] = Counter()
    kept_labels: Counter[str] = Counter()
    relation_lines = 0

    for block_number, block in enumerate(content.split("\n\n"), start=1):
        lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        title_parts = lines[0].split("|t|", maxsplit=1)
        abstract_parts = lines[1].split("|a|", maxsplit=1)
        if len(title_parts) != 2 or len(abstract_parts) != 2:
            raise ValueError(f"Malformed BioRED title/abstract in block {block_number}")

        pmid_text = title_parts[0]
        if abstract_parts[0] != pmid_text:
            raise ValueError(f"PMID mismatch in block {block_number}")
        try:
            row_id = int(pmid_text)
        except ValueError as exc:
            raise ValueError(
                "This patch targets the original 600-document BioRED release whose "
                f"document IDs are numeric PMIDs. Found non-numeric ID: {pmid_text!r}."
            ) from exc

        title = title_parts[1]
        abstract = abstract_parts[1]
        full_text = title + " " + abstract
        documents.append({"row_id": row_id, "full_text": full_text})

        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 5:
                relation_lines += 1
                continue
            try:
                start_char = int(parts[1])
                end_char = int(parts[2])
            except ValueError:
                relation_lines += 1
                continue

            entity_text = parts[3]
            raw_label = parts[4]
            mapped_label = LABEL_MAP.get(raw_label)
            if mapped_label is None:
                ignored_labels[raw_label] += 1
                continue

            if start_char < 0 or end_char <= start_char or end_char > len(full_text):
                offset_errors.append(
                    f"PMID {row_id}: invalid [{start_char}:{end_char}] for text length {len(full_text)}"
                )
                continue

            actual_text = full_text[start_char:end_char]
            if actual_text != entity_text:
                offset_errors.append(
                    f"PMID {row_id}: [{start_char}:{end_char}] expected "
                    f"{entity_text!r} but found {actual_text!r}"
                )
                continue

            kept_labels[mapped_label] += 1
            entities.append(
                {
                    "row_id": row_id,
                    "text": entity_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "label": mapped_label,
                    "biored_label": raw_label,
                }
            )

    if offset_errors:
        preview = "\n".join(offset_errors[:10])
        raise ValueError(
            f"Found {len(offset_errors)} BioRED offset problems in {file_path}.\n{preview}"
        )

    audit = {
        "source_file": str(file_path),
        "document_count": len(documents),
        "target_entity_count": len(entities),
        "target_labels": dict(sorted(kept_labels.items())),
        "ignored_entity_labels": dict(sorted(ignored_labels.items())),
        "ignored_relation_lines": relation_lines,
        "task_scope": ["DISEASE", "CHEMICAL"],
        "note": (
            "Only DiseaseOrPhenotypicFeature and ChemicalEntity annotations are "
            "used as gold targets. Other BioRED entity classes are outside this study."
        ),
    }
    return documents, entities, audit


def process_split(split: str, explicit_input: str | None = None) -> None:
    split = normalize_split(split)
    input_file = resolve_input(split, explicit_input)
    print("\n" + "=" * 72)
    print(f"BIORED {split.upper()} PARSING — DISEASE + CHEMICAL ONLY")
    print("=" * 72)
    print(f"Input: {input_file}")

    documents, entities, audit = parse_biored(input_file)
    save_jsonl(documents, docs_file(split))
    save_jsonl(entities, gold_entities_file(split))
    save_json(audit, results_dir(split) / "biored_data_audit.json")

    print(f"Saved documents: {len(documents)} -> {docs_file(split)}")
    print(f"Saved target entities: {len(entities)} -> {gold_entities_file(split)}")
    print(f"Target counts: {audit['target_labels']}")
    print(f"Ignored BioRED entity classes: {audit['ignored_entity_labels']}")
    print("Relations were ignored because this experiment is NER only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse BioRED disease/chemical targets.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "dev", "test", "development"],
    )
    parser.add_argument(
        "--input",
        help="Optional PubTator file path; only valid when one split is selected.",
    )
    args = parser.parse_args()

    if args.split == "all":
        if args.input:
            parser.error("--input can only be used with one split")
        process_split("dev")
        process_split("test")
        return

    process_split(normalize_split(args.split), args.input)


if __name__ == "__main__":
    main()
