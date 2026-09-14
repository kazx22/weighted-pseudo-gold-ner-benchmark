"""One-document and one-candidate smoke test for local MedGemma/Ollama."""

from __future__ import annotations

import argparse

from src.candidate_gold import build_vote_table, load_model_predictions
from src.experiment_config import docs_file, frozen_config_file, require_file, results_dir, save_json
from src.medgemma_common import (
    DEFAULT_MEDGEMMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaClient,
    TIE_BREAK_SCHEMA,
    ZERO_SHOT_SCHEMA,
    chat_and_parse_with_retries,
    normalise_zero_shot_entities,
    sentence_context_with_markers,
    tie_break_prompt,
    zero_shot_prompt,
)
from src.utils import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test MedGemma through Ollama.")
    parser.add_argument("--model", default=DEFAULT_MEDGEMMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    args = parser.parse_args()

    client = OllamaClient(model=args.model, base_url=args.ollama_url)
    client.ensure_ready()
    docs = load_jsonl(require_file(docs_file("dev"), "development documents"))
    document = docs[0]
    text = str(document["full_text"])

    zero_response, zero_payload, _ = chat_and_parse_with_retries(
        client,
        user_prompt=zero_shot_prompt(text),
        response_schema=ZERO_SHOT_SCHEMA,
        max_output_tokens=2048,
    )
    entities, diagnostics, rejected = normalise_zero_shot_entities(
        row_id=int(document["row_id"]),
        source_text=text,
        response_payload=zero_payload,
        model_name=args.model,
    )

    config = __import__("json").loads(
        require_file(frozen_config_file(), "frozen development config").read_text(encoding="utf-8")
    )
    predictions = load_model_predictions("dev")
    votes, store = build_vote_table(predictions)
    candidate = None
    for key, voters in votes.items():
        score = sum(float(config["weights"][model]) for model in voters)
        if abs(score - float(config["selected_threshold"])) < 1e-8:
            candidate = dict(store[key])
            break
    if candidate is None:
        raise RuntimeError("No threshold-level candidate found for smoke test.")

    row_id = int(candidate["row_id"])
    doc_by_row = {int(item["row_id"]): item for item in docs}
    candidate_text_source = str(doc_by_row[row_id]["full_text"])
    start = int(candidate["start_char"])
    end = int(candidate["end_char"])
    context = sentence_context_with_markers(candidate_text_source, start, end)
    tie_response, tie_payload, _ = chat_and_parse_with_retries(
        client,
        user_prompt=tie_break_prompt(
            context_with_markers=context,
            candidate_text=candidate_text_source[start:end],
            candidate_label=str(candidate["label"]).upper(),
        ),
        response_schema=TIE_BREAK_SCHEMA,
        max_output_tokens=160,
    )

    result = {
        "model": args.model,
        "zero_shot_row_id": int(document["row_id"]),
        "zero_shot_entities": entities,
        "zero_shot_diagnostics": diagnostics,
        "zero_shot_rejected_outputs": rejected,
        "tie_break_candidate": {
            "row_id": row_id,
            "text": candidate_text_source[start:end],
            "label": candidate["label"],
        },
        "tie_break_response": tie_payload,
    }
    output = results_dir("dev") / "medgemma_smoke_test.json"
    save_json(result, output)
    print(f"Smoke test succeeded. Saved: {output}")
    print(f"Zero-shot accepted entities: {len(entities)}")
    print(f"Tie-break decision: {tie_payload.get('decision')}")


if __name__ == "__main__":
    main()
