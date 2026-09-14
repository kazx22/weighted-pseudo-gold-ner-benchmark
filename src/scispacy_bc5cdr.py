"""
scispacy_bc5cdr.py — runs the SciSpacy BC5CDR model over the parsed documents.

Uses the en_ner_bc5cdr_md pipeline, which is trained directly on BC5CDR and
natively emits DISEASE and CHEMICAL labels, so no label remapping is needed.
This is the only model in the study that runs as a plain spaCy pipeline
rather than a HuggingFace transformer; it also requires no chunking because
spaCy handles arbitrary-length text internally.

All models in this study are used off-the-shelf with zero fine-tuning.

Pipeline position: runs after parse_bc5cdr.py; output feeds candidate_gold.py
and bc5cdr_evaluation.py.
"""

import argparse
import json
import time
from pathlib import Path
import spacy

from src.experiment_config import (
    docs_file,
    normalize_split,
    prediction_file,
    record_runtime,
    require_file,
)

MODEL_NAME = "en_ner_bc5cdr_md"


def load_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_scispacy(docs):
    """
    Run the BC5CDR spaCy pipeline over every document and collect entity spans.

    spaCy's ent.start_char / ent.end_char are character offsets into the
    original text, so no additional offset correction is needed (unlike the
    transformer models that require chunk-based offset tracking).

    confidence is fixed at 1.0 because spaCy's en_ner_bc5cdr_md does not
    expose per-entity probability scores.
    """
    nlp = spacy.load(MODEL_NAME)
    all_entities = []

    start_time = time.time()

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]

        doc = nlp(text)

        for ent in doc.ents:
            entity = {
                "row_id": row_id,
                "text": ent.text,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "label": ent.label_,
                "confidence": 1.0,
            }
            all_entities.append(entity)

        if i % 50 == 0:
            print(f"Processed {i} documents...")

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(docs)

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")

    return all_entities, total_time, avg_time


def main():
    parser = argparse.ArgumentParser(description="Run SciSpacy on one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
        help="Use dev for weight/threshold fitting and test for final evaluation.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    input_file = require_file(docs_file(split), "parsed document file")
    output_file = prediction_file("scispacy", split)

    print(f"Loading BC5CDR {split} documents from {input_file}...")
    docs = load_jsonl(input_file)
    if not docs:
        raise ValueError(f"No documents found in {input_file}")
    print(f"Loaded {len(docs)} documents")
    print("Running SciSpacy BC5CDR pipeline...")

    entities, total_time, avg_time = run_scispacy(docs)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        "scispacy",
        total_seconds=total_time,
        average_seconds=avg_time,
        document_count=len(docs),
        entity_count=len(entities),
    )

    print(f"Saved {len(entities)} SciSpacy entities to {output_file}")
    print(f"Saved runtime metadata to results/{split}/runtime.json")


if __name__ == "__main__":
    main()
