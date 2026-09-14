"""
pubmed_bc5cdr.py — runs the PubMedBERT NER pipeline over the parsed documents.

Like BioBERT, this is a dual-model pipeline:
  Disease:  sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease
  Chemical: OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M

Both checkpoints are pre-trained on PubMed text and used off-the-shelf with
zero fine-tuning.  The disease checkpoint was fine-tuned on NCBI-Disease; the
chemical checkpoint is from the OpenMed NER suite.  Neither was trained on
BC5CDR, so the evaluation reflects genuine zero-shot transfer.

The chunking and offset logic is identical to biobert_bc5cdr.py — sentence-
based sliding windows with character-offset tracking via text.find() so
pipeline offsets map correctly back into the original document.

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

DISEASE_MODEL = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)
CHEMICAL_MODEL = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"

MAX_TOKENS = 400  # leaves headroom for [CLS]/[SEP] tokens
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

    Substring matching handles the range of label formats across both
    checkpoints (e.g. 'Disease', 'DISEASE', 'Chemical', 'CHEMICAL').
    fallback_label is used when neither substring matches — it carries the
    intent of the model being called (disease pass or chemical pass).
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

    Returns a list of (chunk_str, char_offset) pairs where chunk_str is a
    direct slice of the original text and char_offset is the position of
    chunk_str[0] in the original text.

    Sentence start positions are found with text.find() rather than by
    concatenating sentence strings, so whitespace and newlines between
    sentences are correctly accounted for in the offset arithmetic.
    """
    raw_sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s]

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


def predictions_to_entities(predictions, row_id, label_name, full_text, char_offset=0):
    """
    Convert HuggingFace pipeline predictions to entity dicts.

    The surface text is taken from the original document slice rather than
    pred["word"] to avoid '##' subword fragments.  Entities longer than
    MAX_ENTITY_CHARS are dropped as chunk-boundary artifacts.
    """
    entities = []
    for pred in predictions:
        start_char = int(pred["start"]) + char_offset
        end_char = int(pred["end"]) + char_offset
        surface = full_text[start_char:end_char]

        if (end_char - start_char) > MAX_ENTITY_CHARS:
            continue

        if (end_char - start_char) < 5:
            print(f"SHORT ENTITY: pred={pred}, surface='{surface}'")

        entities.append(
            {
                "row_id": row_id,
                "text": surface,
                "start_char": start_char,
                "end_char": end_char,
                "label": normalize_label(
                    pred.get("entity_group", label_name), label_name
                ),
                "confidence": float(pred.get("score", 1.0)),
            }
        )
    return entities


def deduplicate_entities(entities):
    """
    Remove duplicate spans produced by overlapping chunks.

    Same key scheme as biobert_bc5cdr.py: (row_id, start_char, end_char, label).
    """
    seen = set()
    deduped = []
    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def run_pubmedbert(docs):
    """
    Run both PubMedBERT checkpoints over all documents and return merged entities.

    Processing order: disease model first, then chemical model, per document.
    Results are merged and deduplicated after all documents are processed.
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

    for i, doc_record in enumerate(docs):
        row_id = doc_record["row_id"]
        text = doc_record["full_text"]

        for chunk, char_offset in chunk_text(text, disease_ner.tokenizer):
            disease_preds = disease_ner(chunk)
            all_entities.extend(
                predictions_to_entities(
                    disease_preds, row_id, "DISEASE", text, char_offset
                )
            )

        for chunk, char_offset in chunk_text(text, chemical_ner.tokenizer):
            chemical_preds = chemical_ner(chunk)
            all_entities.extend(
                predictions_to_entities(
                    chemical_preds, row_id, "CHEMICAL", text, char_offset
                )
            )

        if i % 25 == 0:
            print(f"Processed {i} documents...")

    all_entities = deduplicate_entities(all_entities)

    # Offset self-test: '##' in entity text means the offset arithmetic
    # has broken down and we're storing subword fragments instead of surface text.
    checked = len(all_entities)
    aligned = sum(1 for e in all_entities if "##" not in e["text"])

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
    parser = argparse.ArgumentParser(description="Run PubMedBERT on one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
        help="Use dev for weight/threshold fitting and test for final evaluation.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    input_file = require_file(docs_file(split), "parsed document file")
    output_file = prediction_file("pubmedbert", split)

    print(f"Loading BC5CDR {split} documents from {input_file}...")
    docs = load_jsonl(input_file)
    if not docs:
        raise ValueError(f"No documents found in {input_file}")
    print(f"Loaded {len(docs)} documents")
    print("Running PubMedBERT disease + chemical checkpoints...")

    entities, total_time, avg_time = run_pubmedbert(docs)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        "pubmedbert",
        total_seconds=total_time,
        average_seconds=avg_time,
        document_count=len(docs),
        entity_count=len(entities),
    )

    print(f"Saved {len(entities)} PubMedBERT entities to {output_file}")
    print(f"Saved runtime metadata to results/{split}/runtime.json")


if __name__ == "__main__":
    main()
