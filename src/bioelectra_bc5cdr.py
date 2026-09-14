"""
bioelectra_bc5cdr.py — runs the BioELECTRA NER pipeline over the parsed documents.

Uses d4data/biomedical-ner-all, a single ELECTRA-based checkpoint trained on
multiple biomedical NER corpora.  Its native label space is richer than
BC5CDR's two-class taxonomy, so a label_map collapses the relevant categories:

  Disease_disorder      -> DISEASE
  Sign_symptom          -> DISEASE   (symptoms are annotated as diseases in BC5CDR)
  Medication            -> CHEMICAL
  Therapeutic_procedure -> CHEMICAL  (drug procedures map loosely to chemicals)

All other labels are dropped.  This is the only single-checkpoint model in the
study that does not natively emit DISEASE/CHEMICAL, so the mapping is a
necessary adaptation rather than a cleanup step.

The model is used off-the-shelf with zero fine-tuning.

The chunking and offset logic is identical to biobert_bc5cdr.py — sentence-
based sliding windows with character offsets tracked via text.find().

Pipeline position: runs after parse_bc5cdr.py; output feeds candidate_gold.py
and bc5cdr_evaluation.py.
"""

import argparse
import json
import re
import time
from pathlib import Path
from transformers import pipeline

from src.experiment_config import (
    docs_file,
    normalize_split,
    prediction_file,
    record_runtime,
    require_file,
)

MODEL_NAME = "d4data/biomedical-ner-all"

MAX_TOKENS = 400  # leaves headroom for [CLS]/[SEP] tokens
OVERLAP_SENTS = 1  # one sentence of overlap between consecutive chunks

# Maps d4data/biomedical-ner-all labels to BC5CDR equivalents.
# Sign_symptom -> DISEASE: BC5CDR annotates symptoms as disease mentions.
# Therapeutic_procedure -> CHEMICAL: drug administration procedures are
# treated as chemical references for this evaluation.
# All other labels (Anatomical_structure, Biological_function, etc.) are dropped.
label_map = {
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "DISEASE",
    "Medication": "CHEMICAL",
    "Therapeutic_procedure": "CHEMICAL",
}


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


def normalize_label(raw_label):
    """Look up raw_label in label_map; return None if not mapped."""
    return label_map.get(raw_label, None)


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS, overlap_sents=OVERLAP_SENTS):
    """
    Split text into sentence-based chunks that fit within max_tokens.

    Returns a list of (chunk_str, char_offset) pairs where chunk_str is a
    direct slice of the original text and char_offset is its start position
    in the original document.

    Sentence positions are located with text.find() so inter-sentence
    whitespace is included in the offset.  The chunk string itself is sliced
    directly from the original text (text[start:end]) so character positions
    reported by the pipeline add directly to char_offset.
    """
    raw_sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s]

    # True character offset of each sentence in the ORIGINAL text
    sentence_offsets = []
    search_start = 0
    for sent in raw_sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            idx = search_start
        sentence_offsets.append(idx)
        search_start = idx + len(sent)

    chunks = []
    i = 0
    n = len(raw_sentences)

    while i < n:
        chunk_sent_idx = []
        token_count = 0
        j = i

        while j < n:
            sent = raw_sentences[j]
            sent_token_count = len(tokenizer.encode(sent, add_special_tokens=False))
            if token_count + sent_token_count + 2 > max_tokens and chunk_sent_idx:
                break
            chunk_sent_idx.append(j)
            token_count += sent_token_count
            j += 1

        first = chunk_sent_idx[0]
        last = chunk_sent_idx[-1]
        start_off = sentence_offsets[first]
        end_off = sentence_offsets[last] + len(raw_sentences[last])

        chunk_str = text[start_off:end_off]
        chunks.append((chunk_str, start_off))

        i = max(i + 1, j - overlap_sents)

    return chunks


def deduplicate_entities(entities):
    """
    Remove duplicate spans produced by overlapping chunks.

    Key: (row_id, start_char, end_char, label).  First occurrence wins.
    """
    seen = set()
    deduped = []
    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def run_bioelectra(docs):
    """
    Run the BioELECTRA NER pipeline over all documents.

    For each prediction, the label is mapped via normalize_label and skipped
    if None (i.e. not in BC5CDR's taxonomy).  The surface text is taken from
    the original document slice rather than pred["word"] to avoid '##' subword
    fragments.

    The self-test checks that the first half of pred["word"] (stripped of '##'
    and spaces) appears somewhere in the corresponding original-text slice —
    a loose but fast check that the offset arithmetic is correct.
    """
    ner = pipeline(
        "token-classification",
        model=MODEL_NAME,
        aggregation_strategy="simple",
    )

    print("Model max length:", ner.tokenizer.model_max_length)

    all_entities = []
    start_time = time.time()

    checked = 0
    aligned = 0

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]

        for chunk, char_offset in chunk_text(text, ner.tokenizer):
            predictions = ner(chunk)

            for pred in predictions:
                raw_label = pred.get("entity_group", pred.get("entity", ""))
                label = normalize_label(raw_label)
                if label is None:
                    continue

                start_char = int(pred["start"]) + char_offset
                end_char = int(pred["end"]) + char_offset

                # Use the ORIGINAL text slice as the canonical entity text,
                # not pred["word"] (which may contain ## subword artifacts).
                surface = text[start_char:end_char]

                # Loose self-test: does the first half of pred["word"] appear
                # in the original-text slice?  A miss suggests the offsets
                # have drifted.
                checked += 1
                pred_word = pred["word"].replace("##", "").replace(" ", "").lower()
                slice_norm = surface.replace(" ", "").lower()
                if (
                    pred_word
                    and slice_norm.find(pred_word[: max(3, len(pred_word) // 2)]) != -1
                ):
                    aligned += 1

                entity = {
                    "row_id": row_id,
                    "text": surface,
                    "start_char": start_char,
                    "end_char": end_char,
                    "label": label,
                    "confidence": float(pred.get("score", 1.0)),
                }
                all_entities.append(entity)

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    all_entities = deduplicate_entities(all_entities)

    total_time = time.time() - start_time
    avg_time = total_time / len(docs)

    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")
    print(f"Predicted {len(all_entities)} entities")
    if checked:
        print(
            f"OFFSET SELF-TEST: {aligned}/{checked} "
            f"({100*aligned/checked:.1f}%) predicted words found in their slice"
        )

    return all_entities, total_time, avg_time


def main():
    parser = argparse.ArgumentParser(description="Run BioELECTRA on one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
        help="Use dev for weight/threshold fitting and test for final evaluation.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    input_file = require_file(docs_file(split), "parsed document file")
    output_file = prediction_file("bioelectra", split)

    print(f"Loading BC5CDR {split} documents from {input_file}...")
    docs = load_jsonl(input_file)
    if not docs:
        raise ValueError(f"No documents found in {input_file}")
    print(f"Loaded {len(docs)} documents")
    print("Running BioELECTRA checkpoint...")

    entities, total_time, avg_time = run_bioelectra(docs)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        "bioelectra",
        total_seconds=total_time,
        average_seconds=avg_time,
        document_count=len(docs),
        entity_count=len(entities),
    )

    print(f"Saved {len(entities)} BioELECTRA entities to {output_file}")
    print(f"Saved runtime metadata to results/{split}/runtime.json")


if __name__ == "__main__":
    main()
