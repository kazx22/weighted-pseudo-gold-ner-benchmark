"""Parse BC5CDR PubTator files into split-specific JSONL documents and entities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_config import (
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILENAMES,
    docs_file,
    gold_entities_file,
    normalize_split,
)

LABEL_MAP = {"Disease": "DISEASE", "Chemical": "CHEMICAL"}


def save_jsonl(records: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_bc5cdr(file_path: Path) -> tuple[list[dict], list[dict]]:
    with file_path.open("r", encoding="utf-8") as handle:
        content = handle.read().strip()

    if not content:
        raise ValueError(f"Raw BC5CDR file is empty: {file_path}")

    all_docs: list[dict] = []
    all_entities: list[dict] = []
    offset_errors: list[str] = []

    for block_number, block in enumerate(content.split("\n\n"), start=1):
        lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        title_parts = lines[0].split("|t|", maxsplit=1)
        abstract_parts = lines[1].split("|a|", maxsplit=1)
        if len(title_parts) != 2 or len(abstract_parts) != 2:
            raise ValueError(f"Malformed title/abstract in block {block_number}")

        pmid = int(title_parts[0])
        if int(abstract_parts[0]) != pmid:
            raise ValueError(f"PMID mismatch in block {block_number}")

        title = title_parts[1]
        abstract = abstract_parts[1]
        full_text = title + " " + abstract
        all_docs.append({"row_id": pmid, "full_text": full_text})

        for line in lines[2:]:
            parts = line.split("\t")
            if len(parts) < 5:
                # CID relation line, not an entity annotation.
                continue

            try:
                start_char = int(parts[1])
                end_char = int(parts[2])
            except ValueError:
                # A non-entity tab-delimited line; leave it out explicitly.
                continue

            entity_text = parts[3]
            raw_label = parts[4]
            label = LABEL_MAP.get(raw_label)
            if label is None:
                continue

            actual_text = full_text[start_char:end_char]
            if actual_text != entity_text:
                offset_errors.append(
                    f"PMID {pmid}: [{start_char}:{end_char}] expected "
                    f"'{entity_text}' but found '{actual_text}'"
                )

            all_entities.append(
                {
                    "row_id": pmid,
                    "text": entity_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "label": label,
                }
            )

    if offset_errors:
        preview = "\n".join(offset_errors[:10])
        raise ValueError(
            f"Found {len(offset_errors)} offset mismatches in {file_path}.\n{preview}"
        )

    return all_docs, all_entities


def resolve_input(split: str, explicit_input: str | None) -> Path:
    if explicit_input:
        path = Path(explicit_input)
    else:
        path = RAW_DIR / RAW_FILENAMES[split]
        # Windows/browser downloads often append '(1)' to a duplicate filename.
        if split == "dev" and not path.exists():
            fallback = RAW_DIR / "CDR_DevelopmentSet.PubTator(1).txt"
            if fallback.exists():
                path = fallback
    if not path.exists():
        raise FileNotFoundError(f"Missing raw {split} file: {path}")
    return path


def process_split(split: str, explicit_input: str | None = None) -> None:
    input_file = resolve_input(split, explicit_input)
    print(f"\n--- Processing {split.upper()} ---")
    print(f"Input: {input_file}")

    docs, entities = parse_bc5cdr(input_file)
    docs_output = docs_file(split)
    entities_output = gold_entities_file(split)
    save_jsonl(docs, docs_output)
    save_jsonl(entities, entities_output)

    print(f"Saved {len(docs)} documents to {docs_output}")
    print(f"Saved {len(entities)} entities to {entities_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse BC5CDR PubTator data.")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "dev", "test", "development"],
        help="Parse dev, test, or both. Training is deliberately excluded here.",
    )
    parser.add_argument(
        "--input",
        help="Optional raw file path. Use only when parsing a single split.",
    )
    args = parser.parse_args()

    if args.split == "all":
        if args.input:
            parser.error("--input can only be used with one split.")
        for split in ("dev", "test"):
            process_split(split)
        return

    split = normalize_split(args.split, allow_train=False)
    process_split(split, args.input)


if __name__ == "__main__":
    main()
