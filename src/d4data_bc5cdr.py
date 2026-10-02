
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

MAX_TOKENS = 400                                          
OVERLAP_SENTS = 1                                                      

                                                              
                                                                         
                                                                       
                                                     
                                                                                 
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
    return label_map.get(raw_label, None)


def chunk_text(text, tokenizer, max_tokens=MAX_TOKENS, overlap_sents=OVERLAP_SENTS):
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


def deduplicate_entities(entities):
    seen = set()
    deduped = []
    for ent in entities:
        key = (ent["row_id"], ent["start_char"], ent["end_char"], ent["label"])
        if key not in seen:
            seen.add(key)
            deduped.append(ent)
    return deduped


def run_d4data(docs):
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

                                                                           
                                                                            
                surface = text[start_char:end_char]

                                                                             
                                                                          
                               
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
    parser = argparse.ArgumentParser(description="Run D4Data biomedical NER (DistilBERT) on one BC5CDR split.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test", "development"],
        help="Use dev for weight/threshold fitting and test for final evaluation.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split, allow_train=False)

    input_file = require_file(docs_file(split), "parsed document file")
    output_file = prediction_file("d4data", split)

    print(f"Loading BC5CDR {split} documents from {input_file}...")
    docs = load_jsonl(input_file)
    if not docs:
        raise ValueError(f"No documents found in {input_file}")
    print(f"Loaded {len(docs)} documents")
    print("Running D4Data biomedical NER (DistilBERT) checkpoint...")

    entities, total_time, avg_time = run_d4data(docs)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        "d4data",
        total_seconds=total_time,
        average_seconds=avg_time,
        document_count=len(docs),
        entity_count=len(entities),
    )

    print(f"Saved {len(entities)} D4Data NER entities to {output_file}")
    print(f"Saved runtime metadata to results/{split}/runtime.json")


if __name__ == "__main__":
    main()
