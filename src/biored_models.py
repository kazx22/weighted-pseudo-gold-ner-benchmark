"""Run the same five off-the-shelf NER systems on BioRED disease/chemical text.

The model identities and deterministic label mappings match the BC5CDR study.
No task-specific fine-tuning is performed here. This file writes BioRED outputs
into data/processed/biored and never touches the BC5CDR prediction files.
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Callable

import spacy
from transformers import pipeline

from src.biored_config import (
    MODEL_DISPLAY_NAMES,
    MODEL_KEYS,
    docs_file,
    normalize_split,
    prediction_file,
    record_runtime,
    require_file,
)
from src.utils import deduplicate_entities, load_jsonl, save_jsonl

SCISPACY_MODEL = "en_ner_bc5cdr_md"
BIOBERT_DISEASE_MODEL = "alvaroalon2/biobert_diseases_ner"
BIOBERT_CHEMICAL_MODEL = "alvaroalon2/biobert_chemical_ner"
PUBMEDBERT_DISEASE_MODEL = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)
PUBMEDBERT_CHEMICAL_MODEL = "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M"
CLINICALBERT_MODEL = "samrawal/bert-base-uncased_clinical-ner"
BIOELECTRA_MODEL = "d4data/biomedical-ner-all"

MAX_TOKENS = 400
OVERLAP_SENTS = 1

CLINICAL_LABEL_MAP = {
    "problem": "DISEASE",
    "treatment": "CHEMICAL",
}

BIOELECTRA_LABEL_MAP = {
    "Disease_disorder": "DISEASE",
    "Sign_symptom": "DISEASE",
    "Medication": "CHEMICAL",
    "Therapeutic_procedure": "CHEMICAL",
}


def normalize_binary_label(raw_label: str, fallback: str) -> str:
    raw = str(raw_label).upper()
    if "DISEASE" in raw:
        return "DISEASE"
    if "CHEM" in raw:
        return "CHEMICAL"
    return fallback


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[tuple[str, int]]:
    """Sentence-based chunks that remain literal slices of the source text."""
    raw_sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s]
    if not raw_sentences:
        return [(text, 0)] if text else []

    sentence_offsets: list[int] = []
    search_start = 0
    for sentence in raw_sentences:
        index = text.find(sentence, search_start)
        if index < 0:
            index = search_start
        sentence_offsets.append(index)
        search_start = index + len(sentence)

    chunks: list[tuple[str, int]] = []
    i = 0
    while i < len(raw_sentences):
        chosen: list[int] = []
        token_count = 0
        j = i
        while j < len(raw_sentences):
            sentence_tokens = len(
                tokenizer.encode(raw_sentences[j], add_special_tokens=False)
            )
            if token_count + sentence_tokens + 2 > max_tokens and chosen:
                break
            chosen.append(j)
            token_count += sentence_tokens
            j += 1

        first = chosen[0]
        last = chosen[-1]
        start_char = sentence_offsets[first]
        end_char = sentence_offsets[last] + len(raw_sentences[last])
        chunks.append((text[start_char:end_char], start_char))
        i = max(i + 1, j - OVERLAP_SENTS)

    return chunks


def _safe_entity(
    *,
    row_id: int,
    full_text: str,
    local_start: int,
    local_end: int,
    chunk_offset: int,
    label: str,
    score: float,
) -> dict | None:
    start_char = int(local_start) + int(chunk_offset)
    end_char = int(local_end) + int(chunk_offset)
    if start_char < 0 or end_char <= start_char or end_char > len(full_text):
        return None
    surface = full_text[start_char:end_char]
    if not surface.strip():
        return None
    return {
        "row_id": int(row_id),
        "text": surface,
        "start_char": start_char,
        "end_char": end_char,
        "label": label,
        "confidence": float(score),
    }


def run_scispacy(docs: list[dict]) -> tuple[list[dict], float, float]:
    nlp = spacy.load(SCISPACY_MODEL)
    all_entities: list[dict] = []
    start_time = time.time()
    for index, document in enumerate(docs, start=1):
        row_id = int(document["row_id"])
        text = str(document["full_text"])
        parsed = nlp(text)
        for entity in parsed.ents:
            label = str(entity.label_).upper()
            if label not in {"DISEASE", "CHEMICAL"}:
                continue
            all_entities.append(
                {
                    "row_id": row_id,
                    "text": entity.text,
                    "start_char": int(entity.start_char),
                    "end_char": int(entity.end_char),
                    "label": label,
                    "confidence": 1.0,
                }
            )
        if index == 1 or index % 25 == 0 or index == len(docs):
            print(f"  scispaCy {index}/{len(docs)}")
    total = time.time() - start_time
    return deduplicate_entities(all_entities), total, total / len(docs)


def run_dual_model(
    docs: list[dict],
    *,
    disease_model: str,
    chemical_model: str,
    display_name: str,
) -> tuple[list[dict], float, float]:
    disease_ner = pipeline(
        "token-classification",
        model=disease_model,
        aggregation_strategy="simple",
    )
    chemical_ner = pipeline(
        "token-classification",
        model=chemical_model,
        aggregation_strategy="simple",
    )
    all_entities: list[dict] = []
    start_time = time.time()

    for index, document in enumerate(docs, start=1):
        row_id = int(document["row_id"])
        text = str(document["full_text"])
        for chunk, offset in chunk_text(text, disease_ner.tokenizer):
            for prediction in disease_ner(chunk):
                entity = _safe_entity(
                    row_id=row_id,
                    full_text=text,
                    local_start=int(prediction["start"]),
                    local_end=int(prediction["end"]),
                    chunk_offset=offset,
                    label=normalize_binary_label(
                        prediction.get("entity_group", "DISEASE"), "DISEASE"
                    ),
                    score=float(prediction.get("score", 1.0)),
                )
                if entity is not None:
                    all_entities.append(entity)

        for chunk, offset in chunk_text(text, chemical_ner.tokenizer):
            for prediction in chemical_ner(chunk):
                entity = _safe_entity(
                    row_id=row_id,
                    full_text=text,
                    local_start=int(prediction["start"]),
                    local_end=int(prediction["end"]),
                    chunk_offset=offset,
                    label=normalize_binary_label(
                        prediction.get("entity_group", "CHEMICAL"), "CHEMICAL"
                    ),
                    score=float(prediction.get("score", 1.0)),
                )
                if entity is not None:
                    all_entities.append(entity)

        if index == 1 or index % 25 == 0 or index == len(docs):
            print(f"  {display_name} {index}/{len(docs)}")

    total = time.time() - start_time
    return deduplicate_entities(all_entities), total, total / len(docs)


def run_clinicalbert(docs: list[dict]) -> tuple[list[dict], float, float]:
    ner = pipeline("ner", model=CLINICALBERT_MODEL, aggregation_strategy="simple")
    all_entities: list[dict] = []
    start_time = time.time()
    for index, document in enumerate(docs, start=1):
        row_id = int(document["row_id"])
        text = str(document["full_text"])
        for chunk, offset in chunk_text(text, ner.tokenizer):
            for prediction in ner(chunk):
                raw_label = str(prediction.get("entity_group", ""))
                mapped = CLINICAL_LABEL_MAP.get(raw_label.lower())
                if mapped is None:
                    continue
                entity = _safe_entity(
                    row_id=row_id,
                    full_text=text,
                    local_start=int(prediction["start"]),
                    local_end=int(prediction["end"]),
                    chunk_offset=offset,
                    label=mapped,
                    score=float(prediction.get("score", 1.0)),
                )
                if entity is not None:
                    all_entities.append(entity)
        if index == 1 or index % 25 == 0 or index == len(docs):
            print(f"  ClinicalBERT {index}/{len(docs)}")
    total = time.time() - start_time
    return deduplicate_entities(all_entities), total, total / len(docs)


def run_bioelectra(docs: list[dict]) -> tuple[list[dict], float, float]:
    ner = pipeline("ner", model=BIOELECTRA_MODEL, aggregation_strategy="simple")
    all_entities: list[dict] = []
    start_time = time.time()
    for index, document in enumerate(docs, start=1):
        row_id = int(document["row_id"])
        text = str(document["full_text"])
        for chunk, offset in chunk_text(text, ner.tokenizer):
            for prediction in ner(chunk):
                raw_label = str(prediction.get("entity_group", ""))
                mapped = BIOELECTRA_LABEL_MAP.get(raw_label)
                if mapped is None:
                    # Some transformers versions preserve BIO prefixes.
                    stripped = re.sub(r"^[BI]-", "", raw_label)
                    mapped = BIOELECTRA_LABEL_MAP.get(stripped)
                if mapped is None:
                    continue
                entity = _safe_entity(
                    row_id=row_id,
                    full_text=text,
                    local_start=int(prediction["start"]),
                    local_end=int(prediction["end"]),
                    chunk_offset=offset,
                    label=mapped,
                    score=float(prediction.get("score", 1.0)),
                )
                if entity is not None:
                    all_entities.append(entity)
        if index == 1 or index % 25 == 0 or index == len(docs):
            print(f"  BioELECTRA {index}/{len(docs)}")
    total = time.time() - start_time
    return deduplicate_entities(all_entities), total, total / len(docs)


RUNNERS: dict[str, Callable[[list[dict]], tuple[list[dict], float, float]]] = {
    "scispacy": run_scispacy,
    "biobert": lambda docs: run_dual_model(
        docs,
        disease_model=BIOBERT_DISEASE_MODEL,
        chemical_model=BIOBERT_CHEMICAL_MODEL,
        display_name="BioBERT",
    ),
    "pubmedbert": lambda docs: run_dual_model(
        docs,
        disease_model=PUBMEDBERT_DISEASE_MODEL,
        chemical_model=PUBMEDBERT_CHEMICAL_MODEL,
        display_name="PubMedBERT",
    ),
    "clinicalbert": run_clinicalbert,
    "bioelectra": run_bioelectra,
}


def run_model(model_key: str, split: str, docs: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"BIORED {split.upper()} — {MODEL_DISPLAY_NAMES[model_key]}")
    print("=" * 72)
    entities, total, average = RUNNERS[model_key](docs)
    output_file = prediction_file(model_key, split)
    save_jsonl(entities, output_file)
    record_runtime(
        split,
        model_key,
        total_seconds=total,
        average_seconds=average,
        document_count=len(docs),
        entity_count=len(entities),
    )
    print(f"Saved {len(entities)} entities -> {output_file}")
    print(f"Runtime: {total:.2f}s total; {average:.4f}s/document")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run off-the-shelf NER on BioRED.")
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODEL_KEYS],
        help="Run one model or all five sequentially.",
    )
    args = parser.parse_args()
    split = normalize_split(args.split)
    docs = load_jsonl(require_file(docs_file(split), "parsed BioRED documents"))
    if not docs:
        raise ValueError(f"No BioRED documents in {docs_file(split)}")

    selected = MODEL_KEYS if args.model == "all" else (args.model,)
    for model_key in selected:
        run_model(model_key, split, docs)


if __name__ == "__main__":
    main()
