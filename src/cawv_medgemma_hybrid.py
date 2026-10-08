from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from src.medgemma_common import (
    DEFAULT_MEDGEMMA_MODEL,
    DEFAULT_OLLAMA_URL,
    PROMPT_VERSION,
    TIE_BREAK_SCHEMA,
    OllamaClient,
    append_jsonl,
    candidate_key,
    chat_and_parse_with_retries,
    load_jsonl_if_exists,
    prompt_hash,
    sentence_context_with_markers,
    tie_break_prompt,
)
from src.utils import deduplicate_entities, load_jsonl, save_jsonl


def dataset_parts(dataset: str):
    if dataset == "bc5cdr":
        cfg = importlib.import_module("src.experiment_config")
        cawv = importlib.import_module("src.content_aware_voting")
        candidate = importlib.import_module("src.candidate_gold")
        suffix = "bc5cdr"
    else:
        cfg = importlib.import_module("src.biored_config")
        cawv = importlib.import_module("src.biored_content_aware_voting")
        candidate = importlib.import_module("src.biored_candidate_gold")
        suffix = "biored"
    return cfg, cawv, candidate, suffix


def output_file(cfg, split: str, suffix: str) -> Path:
    return cfg.GOLD_DIR / f"medgemma_cawv_hybrid_{split}_entities_{suffix}.jsonl"


def output_dir(cfg, split: str) -> Path:
    return cfg.results_dir(split) / "medgemma_cawv_hybrid"


def config_file(cfg) -> Path:
    return cfg.results_dir("dev") / "medgemma_cawv_hybrid_config.json"


def derive_band(scores: list[float], threshold: float) -> tuple[float, float]:
    unique = sorted(set(float(score) for score in scores))
    below = [score for score in unique if score < threshold - 1e-12]
    above = [score for score in unique if score > threshold + 1e-12]
    if not below or not above:
        raise ValueError("CAWV threshold needs at least one DEV score on each side.")
    return below[-1], above[0]


def load_or_create_config(dataset: str, split: str, model: str, force_refit: bool) -> dict:
    cfg, cawv, candidate, _ = dataset_parts(dataset)
    path = config_file(cfg)
    if split == "test":
        cfg.require_file(path, "frozen CAWV + MedGemma configuration")
        return json.loads(path.read_text(encoding="utf-8"))

    if path.exists() and not force_refit:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model") != model:
            raise ValueError("Existing CAWV + MedGemma config uses another model.")
        return data

    cawv_path = cfg.require_file(cawv.cawv_config_file(), "frozen CAWV development config")
    base = json.loads(cawv_path.read_text(encoding="utf-8"))
    if base.get("fit_split") != "dev":
        raise ValueError("CAWV config was not fitted on DEV.")

    predictions = candidate.load_model_predictions("dev")
    votes, store = candidate.build_vote_table(predictions)
    scored = cawv.score_candidates(
        votes,
        store,
        alpha=float(base["selected_alpha"]),
        global_weights={k: float(base["global_weights"][k]) for k in cfg.MODEL_KEYS},
        reliability=base["content_reliability"],
    )
    low, high = derive_band([float(row["score"]) for row in scored], float(base["selected_threshold"]))

    data = {
        "schema_version": 1,
        "dataset": dataset,
        "fit_split": "dev",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(PROMPT_VERSION + "|tie-break"),
        "cawv_config": str(cawv_path),
        "cawv_alpha": float(base["selected_alpha"]),
        "base_cawv_threshold": float(base["selected_threshold"]),
        "low_threshold": float(low),
        "high_threshold": float(high),
        "band_rule": "score < low: reject; low <= score < high: MedGemma; score >= high: accept",
        "threshold_origin": "Nearest distinct DEV CAWV candidate scores below and above the frozen DEV threshold.",
    }
    cfg.save_json(data, path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "biored"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", default=DEFAULT_MEDGEMMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--force-decisions", action="store_true")
    args = parser.parse_args()

    cfg, cawv, candidate, suffix = dataset_parts(args.dataset)
    config = load_or_create_config(args.dataset, args.split, args.model, args.force_refit)
    if args.split == "test" and args.model != config["model"]:
        raise ValueError(f"Test must use frozen model {config['model']}.")

    base = json.loads(cfg.require_file(Path(config["cawv_config"]), "frozen CAWV config").read_text(encoding="utf-8"))
    predictions = candidate.load_model_predictions(args.split)
    votes, store = candidate.build_vote_table(predictions)
    scored = cawv.score_candidates(
        votes,
        store,
        alpha=float(config["cawv_alpha"]),
        global_weights={k: float(base["global_weights"][k]) for k in cfg.MODEL_KEYS},
        reliability=base["content_reliability"],
    )

    low = float(config["low_threshold"])
    high = float(config["high_threshold"])
    auto_reject = [row for row in scored if float(row["score"]) < low - 1e-12]
    routed = [row for row in scored if low - 1e-12 <= float(row["score"]) < high - 1e-12]
    auto_accept = [row for row in scored if float(row["score"]) >= high - 1e-12]

    docs = load_jsonl(cfg.require_file(cfg.docs_file(args.split), f"{args.dataset} documents"))
    docs_by_row = {int(doc["row_id"]): doc for doc in docs}
    result_dir = output_dir(cfg, args.split)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache = result_dir / "candidate_decisions.jsonl"
    if args.force_decisions and cache.exists():
        cache.unlink()

    decisions = {}
    for record in load_jsonl_if_exists(cache):
        if record.get("model") == config["model"] and record.get("prompt_hash") == config["prompt_hash"] and record.get("status") in {"ok", "invalid_default_reject"}:
            decisions[str(record["candidate_key"])] = record

    client = OllamaClient(
        base_url=args.ollama_url,
        model=config["model"],
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
    )
    client.ensure_ready()

    print(f"{args.dataset} {args.split}: candidates={len(scored)}, reject={len(auto_reject)}, routed={len(routed)}, accept={len(auto_accept)}")
    print(f"CAWV band: {low:.8f} <= score < {high:.8f}")

    start = time.perf_counter()
    for index, row in enumerate(routed, start=1):
        entity = row["entity"]
        key = candidate_key(entity)
        if key in decisions:
            continue
        doc = docs_by_row[int(entity["row_id"])]
        source = str(doc["full_text"])
        s, e = int(entity["start_char"]), int(entity["end_char"])
        prompt = tie_break_prompt(
            context_with_markers=sentence_context_with_markers(source, s, e),
            candidate_text=source[s:e],
            candidate_label=str(entity["label"]).upper(),
        )
        request_start = time.perf_counter()
        try:
            response, parsed, attempts = chat_and_parse_with_retries(
                client,
                user_prompt=prompt,
                response_schema=TIE_BREAK_SCHEMA,
                max_output_tokens=160,
            )
            decision = str(parsed.get("decision", "")).upper()
            reason = str(parsed.get("reason", "")).strip()
            if decision not in {"ACCEPT", "REJECT"}:
                decision, status = "REJECT", "invalid_default_reject"
            else:
                status = "ok"
        except Exception as exc:
            response, attempts = {}, 3
            decision, status = "REJECT", "invalid_default_reject"
            reason = f"MedGemma error; conservative reject: {exc}"
        record = {
            "candidate_key": key,
            "row_id": int(entity["row_id"]),
            "start_char": int(entity["start_char"]),
            "end_char": int(entity["end_char"]),
            "label": str(entity["label"]),
            "text": source[s:e],
            "cawv_score": float(row["score"]),
            "decision": decision,
            "reason": reason,
            "status": status,
            "model": config["model"],
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": config["prompt_hash"],
            "attempts_used": attempts,
            "duration_seconds": round(time.perf_counter() - request_start, 6),
            "raw_response_content": (response.get("message") or {}).get("content"),
        }
        append_jsonl(record, cache)
        decisions[key] = record
        if index == 1 or index % 25 == 0 or index == len(routed):
            print(f"Tie-break {index}/{len(routed)}: {decision}; score={float(row['score']):.8f}")

    final = []
    for row in auto_accept:
        entity = dict(row["entity"])
        entity["cawv_score"] = round(float(row["score"]), 8)
        entity["hybrid_decision"] = "AUTO_ACCEPT"
        final.append(entity)

    accepted = rejected = invalid = 0
    routed_seconds = 0.0
    for row in routed:
        entity = row["entity"]
        record = decisions[candidate_key(entity)]
        routed_seconds += float(record.get("duration_seconds", 0.0))
        if record.get("status") != "ok":
            invalid += 1
        if record["decision"] == "ACCEPT":
            item = dict(entity)
            item["cawv_score"] = round(float(row["score"]), 8)
            item["hybrid_decision"] = "MEDGEMMA_ACCEPT"
            item["medgemma_reason"] = record.get("reason", "")
            final.append(item)
            accepted += 1
        else:
            rejected += 1

    final = deduplicate_entities(final)
    target = output_file(cfg, args.split, suffix)
    save_jsonl(final, target)
    cfg.save_json(
        {
            "dataset": args.dataset,
            "split": args.split,
            "model": config["model"],
            "base_cawv_threshold": float(config["base_cawv_threshold"]),
            "low_threshold": low,
            "high_threshold": high,
            "total_unique_candidates": len(scored),
            "auto_reject_count": len(auto_reject),
            "routed_count": len(routed),
            "auto_accept_count": len(auto_accept),
            "medgemma_accept_count": accepted,
            "medgemma_reject_count": rejected,
            "invalid_default_reject_count": invalid,
            "final_entity_count": len(final),
            "routed_seconds": round(routed_seconds, 6),
            "wall_seconds": round(time.perf_counter() - start, 6),
            "test_gold_was_not_read_during_construction": args.split == "test",
        },
        result_dir / "hybrid_summary.json",
    )
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
