"""Build split-specific BIO gold files for kappa and confusion matrices."""

import argparse

from src.experiment_config import (
    docs_file,
    gold_bio_file,
    gold_entities_file,
    normalize_split,
    require_file,
)
from src.utils import build_gold_bio, group_by_row, load_jsonl, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BC5CDR BIO gold for one split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    docs_path = require_file(docs_file(split), "parsed document file")
    entities_path = require_file(gold_entities_file(split), "gold entity file")
    output_path = gold_bio_file(split)

    docs = load_jsonl(docs_path)
    entities = load_jsonl(entities_path)
    grouped_entities = group_by_row(entities)

    print(f"Loaded {len(docs)} {split} documents")
    print(f"Loaded {len(entities)} human-gold entities")

    gold_data = build_gold_bio(docs, grouped_entities)
    if not gold_data:
        raise ValueError("No BIO records were built.")

    # Structural self-check: every document must have one label per token.
    bad_records = [
        record for record in gold_data
        if len(record["tokens"]) != len(record["bio_labels"])
    ]
    if bad_records:
        raise ValueError(f"BIO length mismatch in {len(bad_records)} documents")
    print("Self-check passed: every token has exactly one BIO label.")

    save_jsonl(gold_data, output_path)
    print(f"Saved {len(gold_data)} BIO records to {output_path}")


if __name__ == "__main__":
    main()
