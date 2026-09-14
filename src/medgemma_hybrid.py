"""Selective MedGemma tie-breaker for the five-model BC5CDR ensemble.

Candidates far below the frozen weighted threshold are rejected automatically.
Candidates safely above it are accepted automatically. Only candidates on the
adjacent score levels around the threshold are routed to MedGemma.

The test construction path never loads test human gold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.candidate_gold import achievable_thresholds, build_vote_table, load_model_predictions
from src.experiment_config import (
    MODEL_KEYS,
    docs_file,
    frozen_config_file,
    medgemma_hybrid_file,
    medgemma_hybrid_results_dir,
    normalize_split,
    require_file,
    results_dir,
    save_json,
)
from src.medgemma_common import (
    DEFAULT_MEDGEMMA_MODEL,
    DEFAULT_OLLAMA_URL,
    PROMPT_VERSION,
    TIE_BREAK_SCHEMA,
    OllamaClient,
    append_jsonl,
    chat_and_parse_with_retries,
    candidate_key,
    load_jsonl_if_exists,
    prompt_hash,
    sentence_context_with_markers,
    tie_break_prompt,
)
from src.utils import deduplicate_entities, load_jsonl, save_jsonl

CONFIG_FILENAME = "medgemma_hybrid_config.json"


def derive_band(
    *,
    weights: dict[str, float],
    base_threshold: float,
    levels_below: int,
    levels_above: int,
) -> tuple[float, float]:
    scores = achievable_thresholds(weights)
    matching_indices: list[int] = []
    for index, score in enumerate(scores):
        if abs(score - base_threshold) <= 1e-8:
            matching_indices.append(index)
    if not matching_indices:
        raise ValueError("The frozen threshold is not an achievable score.")

    threshold_index = matching_indices[0]
    low_index = max(0, threshold_index - levels_below)
    high_index = min(len(scores) - 1, threshold_index + levels_above)
    low_threshold = float(scores[low_index])
    high_threshold = float(scores[high_index])
    if not low_threshold < high_threshold:
        raise ValueError("The uncertainty band must have low < high.")
    return low_threshold, high_threshold


def candidate_rows(
    vote_table: dict,
    entity_store: dict,
    weights: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    for key, voter_set in vote_table.items():
        voters = sorted(voter_set)
        score = 0.0
        for voter in voters:
            score += float(weights[voter])
        entity = dict(entity_store[key])
        entity["voters"] = voters
        entity["vote_count"] = len(voters)
        entity["weighted_score"] = round(score, 8)
        rows.append(entity)
    rows.sort(
        key=lambda item: (
            int(item["row_id"]),
            int(item["start_char"]),
            int(item["end_char"]),
            str(item["label"]),
        )
    )
    return rows


def load_or_create_config(
    *,
    split: str,
    model: str,
    levels_below: int,
    levels_above: int,
    force_refit: bool,
) -> dict:
    original_config = json.loads(
        require_file(frozen_config_file(), "frozen five-model development configuration").read_text(
            encoding="utf-8"
        )
    )
    weights = original_config["weights"]
    base_threshold = float(original_config["selected_threshold"])
    config_path = results_dir("dev") / CONFIG_FILENAME

    if split == "test":
        require_file(config_path, "frozen MedGemma hybrid configuration")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return config

    if config_path.exists() and not force_refit:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model") != model:
            raise ValueError("Existing hybrid config uses another MedGemma model. Use --force-refit.")
        return config

    low_threshold, high_threshold = derive_band(
        weights=weights,
        base_threshold=base_threshold,
        levels_below=levels_below,
        levels_above=levels_above,
    )
    config = {
        "schema_version": 1,
        "fit_split": "dev",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(PROMPT_VERSION + "|tie-break"),
        "base_weighted_threshold": base_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "levels_below": levels_below,
        "levels_above": levels_above,
        "band_rule": (
            "score < low: reject; low <= score < high: MedGemma; "
            "score >= high: accept"
        ),
        "weights": weights,
        "source_ensemble_models": list(MODEL_KEYS),
        "threshold_origin": (
            "Adjacent achievable weighted-score levels around the frozen "
            "development-selected threshold."
        ),
    }
    save_json(config, config_path)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the selective MedGemma BC5CDR tie-breaker."
    )
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    parser.add_argument("--model", default=DEFAULT_MEDGEMMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--levels-below", type=int, default=1)
    parser.add_argument("--levels-above", type=int, default=1)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--force-decisions", action="store_true")
    args = parser.parse_args()

    split = normalize_split(args.split, allow_train=False)
    if args.levels_below < 1 or args.levels_above < 1:
        raise ValueError("Use at least one score level below and above.")

    config = load_or_create_config(
        split=split,
        model=args.model,
        levels_below=args.levels_below,
        levels_above=args.levels_above,
        force_refit=args.force_refit,
    )
    if split == "test" and args.model != config["model"]:
        raise ValueError(
            f"Test must use the frozen model {config['model']}, not {args.model}."
        )

    weights = config["weights"]
    low_threshold = float(config["low_threshold"])
    high_threshold = float(config["high_threshold"])

    docs = load_jsonl(require_file(docs_file(split), "parsed BC5CDR documents"))
    docs_by_row: dict[int, dict] = {}
    for document in docs:
        docs_by_row[int(document["row_id"])] = document

    predictions = load_model_predictions(split)
    vote_table, entity_store = build_vote_table(predictions)
    candidates = candidate_rows(vote_table, entity_store, weights)

    auto_reject: list[dict] = []
    routed: list[dict] = []
    auto_accept: list[dict] = []
    for candidate in candidates:
        score = float(candidate["weighted_score"])
        if score + 1e-12 < low_threshold:
            auto_reject.append(candidate)
        elif score + 1e-12 >= high_threshold:
            auto_accept.append(candidate)
        else:
            routed.append(candidate)

    output_dir = medgemma_hybrid_results_dir(split)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_cache = output_dir / "candidate_decisions.jsonl"
    if args.force_decisions and decision_cache.exists():
        decision_cache.unlink()

    expected_prompt_hash = str(config["prompt_hash"])
    cached_decisions = load_jsonl_if_exists(decision_cache)
    decisions_by_key: dict[str, dict] = {}
    for record in cached_decisions:
        if record.get("model") != config["model"]:
            raise ValueError("Existing decision cache uses another model.")
        if record.get("prompt_hash") != expected_prompt_hash:
            raise ValueError("Existing decision cache uses another prompt version.")
        if record.get("status") in {"ok", "invalid_default_reject"}:
            decisions_by_key[str(record["candidate_key"])] = record

    client = OllamaClient(
        base_url=args.ollama_url,
        model=config["model"],
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
    )
    client.ensure_ready()

    print(f"Split: {split}")
    print(f"Candidates: {len(candidates)}")
    print(f"Automatic reject: {len(auto_reject)}")
    print(f"MedGemma routed: {len(routed)}")
    print(f"Automatic accept: {len(auto_accept)}")
    print(f"Low threshold: {low_threshold:.8f}")
    print(f"High threshold: {high_threshold:.8f}")
    print(f"Cached tie-break decisions: {len(decisions_by_key)}")

    start_time = time.perf_counter()
    for index, candidate in enumerate(routed, start=1):
        key = candidate_key(candidate)
        if key in decisions_by_key:
            continue

        row_id = int(candidate["row_id"])
        document = docs_by_row[row_id]
        source_text = str(document["full_text"])
        start_char = int(candidate["start_char"])
        end_char = int(candidate["end_char"])
        context = sentence_context_with_markers(source_text, start_char, end_char)
        prompt = tie_break_prompt(
            context_with_markers=context,
            candidate_text=source_text[start_char:end_char],
            candidate_label=str(candidate["label"]).upper(),
        )

        request_start = time.perf_counter()
        try:
            api_response, parsed, attempts_used = chat_and_parse_with_retries(
                client,
                user_prompt=prompt,
                response_schema=TIE_BREAK_SCHEMA,
                max_output_tokens=160,
            )
            decision = str(parsed.get("decision", "")).upper()
            reason = str(parsed.get("reason", "")).strip()
            if decision not in {"ACCEPT", "REJECT"}:
                status = "invalid_default_reject"
                decision = "REJECT"
            else:
                status = "ok"
        except Exception as exc:  # noqa: BLE001 - conservative fallback is recorded
            attempts_used = 3
            status = "invalid_default_reject"
            decision = "REJECT"
            reason = f"MedGemma error; conservative reject: {exc}"
            api_response = {}

        duration = time.perf_counter() - request_start
        record = {
            "candidate_key": key,
            "row_id": row_id,
            "start_char": start_char,
            "end_char": end_char,
            "text": source_text[start_char:end_char],
            "label": str(candidate["label"]).upper(),
            "weighted_score": float(candidate["weighted_score"]),
            "voters": candidate["voters"],
            "decision": decision,
            "reason": reason,
            "status": status,
            "model": config["model"],
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": expected_prompt_hash,
            "attempts_used": attempts_used,
            "duration_seconds": round(duration, 6),
            "raw_response_content": (api_response.get("message") or {}).get("content"),
            "api_usage": {
                "created_at": api_response.get("created_at"),
                "prompt_eval_count": api_response.get("prompt_eval_count"),
                "eval_count": api_response.get("eval_count"),
            },
        }
        append_jsonl(record, decision_cache)
        decisions_by_key[key] = record

        if index == 1 or index % 25 == 0 or index == len(routed):
            print(
                f"Tie-break {index}/{len(routed)}: {decision}; "
                f"score={candidate['weighted_score']:.8f}; text={record['text']!r}"
            )

    final_entities: list[dict] = []
    for candidate in auto_accept:
        entity = dict(candidate)
        entity["hybrid_decision"] = "AUTO_ACCEPT"
        final_entities.append(entity)

    llm_accept_count = 0
    llm_reject_count = 0
    invalid_count = 0
    tie_break_seconds = 0.0
    for candidate in routed:
        record = decisions_by_key[candidate_key(candidate)]
        tie_break_seconds += float(record.get("duration_seconds", 0.0))
        if record.get("status") != "ok":
            invalid_count += 1
        if record["decision"] == "ACCEPT":
            entity = dict(candidate)
            entity["hybrid_decision"] = "MEDGEMMA_ACCEPT"
            entity["medgemma_reason"] = record.get("reason", "")
            final_entities.append(entity)
            llm_accept_count += 1
        else:
            llm_reject_count += 1

    final_entities = deduplicate_entities(final_entities)
    output_file = medgemma_hybrid_file(split)
    save_jsonl(final_entities, output_file)

    summary = {
        "split": split,
        "model": config["model"],
        "prompt_version": PROMPT_VERSION,
        "base_weighted_threshold": float(config["base_weighted_threshold"]),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "total_unique_candidates": len(candidates),
        "auto_reject_count": len(auto_reject),
        "routed_count": len(routed),
        "auto_accept_count": len(auto_accept),
        "medgemma_accept_count": llm_accept_count,
        "medgemma_reject_count": llm_reject_count,
        "invalid_default_reject_count": invalid_count,
        "final_entity_count": len(final_entities),
        "routing_rate_over_all_candidates": len(routed) / len(candidates) if candidates else 0.0,
        "tie_break_request_seconds": round(tie_break_seconds, 6),
        "current_process_wall_seconds": round(time.perf_counter() - start_time, 6),
        "construction_used_human_gold": False,
        "test_gold_was_not_read": True if split == "test" else None,
        "decision_cache": str(decision_cache),
        "output_file": str(output_file),
    }
    save_json(summary, output_dir / "hybrid_summary.json")

    print(f"Saved {len(final_entities)} hybrid entities to {output_file}")
    print(f"Saved routing summary to {output_dir / 'hybrid_summary.json'}")


if __name__ == "__main__":
    main()
