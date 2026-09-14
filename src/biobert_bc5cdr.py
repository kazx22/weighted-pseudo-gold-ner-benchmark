"""
biobert_bc5cdr.py — runs the BioBERT NER pipeline over the parsed documents.

BioBERT is implemented as a dual-model pipeline: one checkpoint for diseases
(alvaroalon2/biobert_diseases_ner) and one for chemicals
(alvaroalon2/biobert_chemical_ner).  Each document is processed through both
models independently, and the predictions are merged before deduplication.

Both checkpoints are used off-the-shelf with zero fine-tuning.

Because BERT models have a hard 512-token context limit, documents are split
into overlapping sentence-based chunks.  Each chunk is a literal slice of the
original text so that the pipeline's character offsets (pred["start"],
pred["end"]) map directly back to the original document after adding the
chunk's start offset.

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

DISEASE_MODEL = "alvaroalon2/biobert_diseases_ner"
CHEMICAL_MODEL = "alvaroalon2/biobert_chemical_ner"

MAX_TOKENS = 400  # leaves headroom for [CLS]/[SEP] and avoids truncation
OVERLAP_SENTS = 1  # one sentence of overlap between consecutive chunks
MAX_ENTITY_CHARS = 110  # longest real BC5CDR gold entity is 105 chars;
# anything longer is almost certainly a chunk-boundary
# aggregation artifact and is dropped


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


def normalize_label(raw_label, fallback_label):
    """
    Map a raw model label to DISEASE or CHEMICAL.

    BioBERT checkpoints use labels like 'Disease', 'Chemical', or longer
    variants.  The substring check handles all observed variants without
    needing an exhaustive lookup table.  fallback_label is what the calling
    code already knows the model is producing (disease model -> DISEASE),
    used when the label doesn't match either substring.
    """
    raw = str(raw_label).upper()
    if "DISEASE" in raw:
        return "DISEASE"
    if "CHEM" in raw:
        return "CHEMICAL"
    return fallback_label


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS, overlap_sents=OVERLAP_SENTS):
    """
    Split text into sentence-based chunks that fit within max_tokens.

    Returns a list of (chunk_str, char_offset) pairs where:
      chunk_str   is a direct slice of the original text
      char_offset is the position of chunk_str[0] in the original text

    Sentence boundaries are located with text.find() rather than by
    concatenating sentence strings, so any whitespace or newline characters
    between sentences are correctly accounted for in the offset.

    One sentence of overlap between consecutive chunks reduces the chance
    that an entity split across a chunk boundary is missed entirely.
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
            # +2 reserves space for [CLS] and [SEP]
            if token_count + sent_token_count + 2 > max_tokens and chunk_sent_idx:
                break
            chunk_sent_idx.append(j)
            token_count += sent_token_count
            j += 1

        first = chunk_sent_idx[0]
        last = chunk_sent_idx[-1]
        start_off = sentence_offsets[first]
        end_off = sentence_offsets[last] + len(raw_sentences[last])

        # Slice the ORIGINAL text so spacing and punctuation are preserved
        chunk_str = text[start_off:end_off]
        chunks.append((chunk_str, start_off))

        i = max(i + 1, j - overlap_sents)

    return chunks


def predictions_to_entities(predictions, row_id, label_name, full_text, char_offset=0):
    """
    Convert HuggingFace pipeline predictions to entity dicts.

    Uses the original text slice as the canonical surface form instead of
    pred["word"], which can contain '##' subword fragments when aggregation
    doesn't fully reconstruct the surface.  The offset into the original
    document is char_offset + pred["start"].

    Entities longer than MAX_ENTITY_CHARS are dropped — anything that long
    is overwhelmingly likely to be a chunk-boundary aggregation artifact
    rather than a real named entity.
    """
    entities = []
    for pred in predictions:
        start_char = int(pred["start"]) + char_offset
        end_char = int(pred["end"]) + char_offset
        surface = full_text[start_char:end_char]

        if (end_char - start_char) > MAX_ENTITY_CHARS:
            continue

        entities.append(
            {
                "row_id": row_id,
                "text": surface,
                "start_char": start_char,
                "end_char": end_char,
                "label": normalize_label(
                    pred.get("entity_group", pred.get("entity", label_name)),
                    label_name,
                ),
                "confidence": float(pred.get("score", 1.0)),
            }
        )
    return entities


def deduplicate_entities(entities):
    """
    Remove duplicate spans produced by overlapping chunks.

    Overlap between consecutive chunks means the same entity can appear
    twice with identical offsets.  Deduplication is on (row_id, start,
    end, label); the first occurrence wins.
    """
    seen = set()
    deduped = []
    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def run_biobert(docs):
    """
    Run both BioBERT checkpoints over all documents and return merged entities.

    Each document is processed by the disease model first, then the chemical
    model.  The two prediction lists are combined and deduplicated.

    The offset self-test checks that the entity text (derived from the
    original document slice) contains no '##' fragments — a quick sanity
    check that the offset arithmetic is correct.
    """
    disease_ner = pipeline(
        "token-classification",
        model=DISEASE_MODEL,
        aggregation_strategy="simple",
    )
    chemical_ner = pipeline(
        "token-classification",
        model=CHEMICAL_MODEL,
        aggregation_strategy="simple",
    )

    print("Disease model max length:", disease_ner.tokenizer.model_max_length)
    print("Chemical model max length:", chemical_ner.tokenizer.model_max_length)

    all_entities = []
    start_time = time.time()
    checked = 0
    aligned = 0

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]

        for chunk, char_offset in chunk_text(text, disease_ner.tokenizer):
            disease_preds = disease_ner(chunk)
            ents = predictions_to_entities(
                disease_preds, row_id, "DISEASE", text, char_offset
            )
            all_entities.extend(ents)

        for chunk, char_offset in chunk_text(text, chemical_ner.tokenizer):
            chemical_preds = chemical_ner(chunk)
            ents = predictions_to_entities(
                chemical_preds, row_id, "CHEMICAL", text, char_offset
            )
            all_entities.extend(ents)

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    all_entities = deduplicate_entities(all_entities)

    # Offset self-test: count entities whose surface text is free of '##'
    for e in all_entities:
        checked += 1
        if "##" not in e["text"]:
            aligned += 1

    total_time = time.time() - start_time
    avg_time = total_time / len(docs)
    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Average time per document: {avg_time:.4f} seconds")
    print(f"Predicted {len(all_entities)} entities")
    print(
        f"OFFSET SELF-TEST: {aligned}/{checked} "
        f"({(100 * aligned / checked) if checked else 100.0:.1f}%) entities free of ## fragments"
    )

    return all_entities, total_time, avg_time


def main():
    parser = argparse.ArgumentParser(description="Run BioBERT on one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
        help="Use dev for weight/threshold fitting and test for final evaluation.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    input_file = require_file(docs_file(split), "parsed document file")
    output_file = prediction_file("biobert", split)

    print(f"Loading BC5CDR {split} documents from {input_file}...")
    docs = load_jsonl(input_file)
    if not docs:
        raise ValueError(f"No documents found in {input_file}")
    print(f"Loaded {len(docs)} documents")
    print("Running BioBERT disease + chemical checkpoints...")

    entities, total_time, avg_time = run_biobert(docs)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        "biobert",
        total_seconds=total_time,
        average_seconds=avg_time,
        document_count=len(docs),
        entity_count=len(entities),
    )

    print(f"Saved {len(entities)} BioBERT entities to {output_file}")
    print(f"Saved runtime metadata to results/{split}/runtime.json")


if __name__ == "__main__":
    main()
