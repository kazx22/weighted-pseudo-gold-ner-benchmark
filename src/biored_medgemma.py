"""Run MedGemma 1.5 4B as a zero-shot BioRED NER baseline.

The model is served locally through Ollama. To prevent very long or repetitive
JSON generations, each BioRED document is split deterministically into short
text segments. Entity offsets returned for a segment are shifted back to the
original document coordinates before evaluation.

One durable cache record is saved per document so interrupted runs can resume.
The chunked cache is intentionally separate from the earlier whole-document
cache because the inference protocol has changed.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from src.biored_config import (
    docs_file,
    medgemma_results_dir,
    normalize_split,
    prediction_file,
    record_runtime,
    require_file,
    save_json,
)
from src.biored_medgemma_common import (
    DEFAULT_MEDGEMMA_MODEL,
    DEFAULT_OLLAMA_URL,
    PROMPT_VERSION,
    ZERO_SHOT_SCHEMA,
    OllamaClient,
    append_jsonl,
    chat_and_parse_with_retries,
    load_jsonl_if_exists,
    normalise_zero_shot_entities,
    prompt_hash,
    sum_diagnostics,
    zero_shot_prompt,
)
from src.utils import deduplicate_entities, load_jsonl, save_jsonl

CHUNKING_VERSION = "chunked-v2"
DEFAULT_CHUNK_CHARS = 450
MIN_RECURSIVE_CHUNK_CHARS = 80


@dataclass(frozen=True)
class TextChunk:
    start_char: int
    end_char: int
    text: str


def _trim_span(text: str, start_char: int, end_char: int) -> tuple[int, int]:
    while start_char < end_char and text[start_char].isspace():
        start_char += 1
    while end_char > start_char and text[end_char - 1].isspace():
        end_char -= 1
    return start_char, end_char


def _sentence_like_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic sentence-like spans while preserving source offsets."""
    boundaries: list[int] = [0]
    index = 0
    text_length = len(text)

    while index < text_length:
        character = text[index]
        should_cut = False
        boundary = index + 1

        if character == "\n":
            should_cut = True
        elif character in ".!?":
            probe = index + 1
            while probe < text_length and text[probe] in "\"' )]}".replace(" ", ""):
                probe += 1
            if probe >= text_length or text[probe].isspace():
                should_cut = True
                boundary = probe

        if should_cut:
            while boundary < text_length and text[boundary].isspace():
                boundary += 1
            if boundary > boundaries[-1]:
                boundaries.append(boundary)
            index = boundary
        else:
            index += 1

    if boundaries[-1] != text_length:
        boundaries.append(text_length)

    spans: list[tuple[int, int]] = []
    for position in range(len(boundaries) - 1):
        start_char, end_char = _trim_span(
            text,
            boundaries[position],
            boundaries[position + 1],
        )
        if start_char < end_char:
            spans.append((start_char, end_char))
    return spans


def _best_internal_cut(text: str, start_char: int, target_end: int) -> int:
    """Choose a safe cut near target_end, preferring clause punctuation."""
    minimum_cut = start_char + max(40, (target_end - start_char) // 2)

    for symbols in (";:", ",", " \t"):
        best = -1
        for symbol in symbols:
            location = text.rfind(symbol, minimum_cut, target_end)
            if location > best:
                best = location
        if best >= minimum_cut:
            return best + 1

    return target_end


def _split_long_span(
    text: str,
    start_char: int,
    end_char: int,
    maximum_chars: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start_char

    while cursor < end_char:
        target_end = min(cursor + maximum_chars, end_char)
        if target_end < end_char:
            target_end = _best_internal_cut(text, cursor, target_end)
        if target_end <= cursor:
            target_end = min(cursor + maximum_chars, end_char)

        part_start, part_end = _trim_span(text, cursor, target_end)
        if part_start < part_end:
            spans.append((part_start, part_end))

        cursor = target_end
        while cursor < end_char and text[cursor].isspace():
            cursor += 1

    return spans


def split_document_into_chunks(
    text: str,
    *,
    maximum_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[TextChunk]:
    """Group sentence-like spans into compact chunks with exact source offsets."""
    if maximum_chars < 80:
        raise ValueError("maximum_chars must be at least 80")

    sentence_spans = _sentence_like_spans(text)
    expanded_spans: list[tuple[int, int]] = []
    for start_char, end_char in sentence_spans:
        if end_char - start_char <= maximum_chars:
            expanded_spans.append((start_char, end_char))
        else:
            split_spans = _split_long_span(
                text,
                start_char,
                end_char,
                maximum_chars,
            )
            expanded_spans.extend(split_spans)

    chunks: list[TextChunk] = []
    group_start: int | None = None
    group_end: int | None = None

    for start_char, end_char in expanded_spans:
        if group_start is None:
            group_start = start_char
            group_end = end_char
            continue

        assert group_end is not None
        proposed_length = end_char - group_start
        if proposed_length <= maximum_chars:
            group_end = end_char
        else:
            chunk_text = text[group_start:group_end]
            chunks.append(TextChunk(group_start, group_end, chunk_text))
            group_start = start_char
            group_end = end_char

    if group_start is not None and group_end is not None:
        chunks.append(TextChunk(group_start, group_end, text[group_start:group_end]))

    if not chunks and text.strip():
        start_char, end_char = _trim_span(text, 0, len(text))
        chunks.append(TextChunk(start_char, end_char, text[start_char:end_char]))

    return chunks


def _merge_diagnostics(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _shift_entities_to_document(
    entities: list[dict],
    *,
    document_text: str,
    chunk_start: int,
) -> list[dict]:
    shifted: list[dict] = []
    for entity in entities:
        updated = dict(entity)
        global_start = int(entity["start_char"]) + chunk_start
        global_end = int(entity["end_char"]) + chunk_start
        updated["start_char"] = global_start
        updated["end_char"] = global_end
        updated["text"] = document_text[global_start:global_end]
        updated["chunk_start_char"] = chunk_start
        shifted.append(updated)
    return shifted


def _should_split_after_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "generation limit" in message
        or "invalid json" in message
        or "output reached" in message
    )


def _extract_chunk_recursively(
    *,
    client: OllamaClient,
    row_id: int,
    document_text: str,
    chunk: TextChunk,
    model_name: str,
    depth: int = 0,
) -> tuple[list[dict], dict[str, int], list[dict], list[dict], int]:
    """Extract one chunk, splitting it further only if generation still loops."""
    prompt = zero_shot_prompt(chunk.text)

    try:
        api_response, parsed, attempts_used = chat_and_parse_with_retries(
            client,
            user_prompt=prompt,
            response_schema=ZERO_SHOT_SCHEMA,
            max_output_tokens=1536,
            attempts=2,
        )
        local_entities, diagnostics, rejected = normalise_zero_shot_entities(
            row_id=row_id,
            source_text=chunk.text,
            response_payload=parsed,
            model_name=model_name,
        )
        shifted_entities = _shift_entities_to_document(
            local_entities,
            document_text=document_text,
            chunk_start=chunk.start_char,
        )

        enriched_rejected: list[dict] = []
        for rejected_item in rejected:
            item = dict(rejected_item)
            item["chunk_start_char"] = chunk.start_char
            item["chunk_end_char"] = chunk.end_char
            enriched_rejected.append(item)

        chunk_record = {
            "chunk_start_char": chunk.start_char,
            "chunk_end_char": chunk.end_char,
            "chunk_length": len(chunk.text),
            "recursion_depth": depth,
            "attempts_used": attempts_used,
            "accepted_entities": len(shifted_entities),
            "raw_response_content": (api_response.get("message") or {}).get("content"),
            "api_usage": {
                "created_at": api_response.get("created_at"),
                "total_duration_ns": api_response.get("total_duration"),
                "load_duration_ns": api_response.get("load_duration"),
                "prompt_eval_count": api_response.get("prompt_eval_count"),
                "eval_count": api_response.get("eval_count"),
                "done_reason": api_response.get("done_reason"),
            },
            "diagnostics": diagnostics,
        }
        return (
            shifted_entities,
            diagnostics,
            enriched_rejected,
            [chunk_record],
            attempts_used,
        )
    except Exception as exc:
        can_split = len(chunk.text) > MIN_RECURSIVE_CHUNK_CHARS * 2
        if not can_split or not _should_split_after_failure(exc):
            raise

        fallback_size = max(MIN_RECURSIVE_CHUNK_CHARS, len(chunk.text) // 2)
        local_subchunks = split_document_into_chunks(
            chunk.text,
            maximum_chars=fallback_size,
        )
        if len(local_subchunks) < 2:
            raise

        print(
            "MedGemma still produced an incomplete segment response. "
            f"Splitting document row_id={row_id} segment "
            f"[{chunk.start_char}:{chunk.end_char}] into {len(local_subchunks)} "
            "smaller segments...",
            flush=True,
        )

        all_entities: list[dict] = []
        all_rejected: list[dict] = []
        all_chunk_records: list[dict] = []
        combined_diagnostics: dict[str, int] = {}
        total_attempts = 0

        for local_subchunk in local_subchunks:
            global_subchunk = TextChunk(
                start_char=chunk.start_char + local_subchunk.start_char,
                end_char=chunk.start_char + local_subchunk.end_char,
                text=local_subchunk.text,
            )
            (
                sub_entities,
                sub_diagnostics,
                sub_rejected,
                sub_chunk_records,
                sub_attempts,
            ) = _extract_chunk_recursively(
                client=client,
                row_id=row_id,
                document_text=document_text,
                chunk=global_subchunk,
                model_name=model_name,
                depth=depth + 1,
            )
            all_entities.extend(sub_entities)
            all_rejected.extend(sub_rejected)
            all_chunk_records.extend(sub_chunk_records)
            _merge_diagnostics(combined_diagnostics, sub_diagnostics)
            total_attempts += sub_attempts

        return (
            deduplicate_entities(all_entities),
            combined_diagnostics,
            all_rejected,
            all_chunk_records,
            total_attempts,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local MedGemma zero-shot NER on BioRED."
    )
    parser.add_argument("--split", required=True, choices=["dev", "test", "development"])
    parser.add_argument("--model", default=DEFAULT_MEDGEMMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the matching chunked cache and restart this MedGemma run.",
    )
    args = parser.parse_args()

    split = normalize_split(args.split, allow_train=False)
    docs = load_jsonl(require_file(docs_file(split), "parsed BioRED documents"))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        docs = docs[: args.limit]
        run_dir = medgemma_results_dir(split).parent / f"medgemma_smoke_{args.limit}"
        output_file = run_dir / f"medgemma_{split}_entities_smoke.jsonl"
    else:
        run_dir = medgemma_results_dir(split)
        output_file = prediction_file("medgemma", split)

    run_dir.mkdir(parents=True, exist_ok=True)
    cache_file = run_dir / "zero_shot_doc_results_chunked_v2.jsonl"
    summary_file = run_dir / "zero_shot_summary.json"

    if args.force and cache_file.exists():
        cache_file.unlink()

    client = OllamaClient(
        base_url=args.ollama_url,
        model=args.model,
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        seed=args.seed,
        num_ctx=args.num_ctx,
    )
    client.ensure_ready()

    expected_prompt_hash = prompt_hash(
        PROMPT_VERSION
        + "|zero-shot|"
        + CHUNKING_VERSION
        + f"|chunk_chars={args.chunk_chars}"
    )
    cached_records = load_jsonl_if_exists(cache_file)
    cached_by_row: dict[int, dict] = {}
    for record in cached_records:
        if record.get("model") != args.model:
            raise ValueError(
                "Existing chunked cache uses another model. Use --force or restore the original model."
            )
        if record.get("prompt_hash") != expected_prompt_hash:
            raise ValueError(
                "Existing chunked cache uses another prompt/chunk configuration. Use --force to restart."
            )
        if record.get("status") == "ok":
            cached_by_row[int(record["row_id"])] = record

    print(f"MedGemma model: {args.model}")
    print(f"Split: {split}")
    print(f"Documents requested: {len(docs)}")
    print(f"Chunk target: {args.chunk_chars} characters")
    print(f"Already completed in chunked cache: {len(cached_by_row)}")
    print(f"Cache: {cache_file}")

    total_wall_start = time.perf_counter()
    for index, document in enumerate(docs, start=1):
        row_id = int(document["row_id"])
        if row_id in cached_by_row:
            continue

        source_text = str(document["full_text"])
        chunks = split_document_into_chunks(
            source_text,
            maximum_chars=args.chunk_chars,
        )
        request_start = time.perf_counter()

        try:
            entities: list[dict] = []
            rejected: list[dict] = []
            chunk_records: list[dict] = []
            diagnostics: dict[str, int] = {}
            attempts_used = 0

            for chunk_index, chunk in enumerate(chunks, start=1):
                (
                    chunk_entities,
                    chunk_diagnostics,
                    chunk_rejected,
                    chunk_result_records,
                    chunk_attempts,
                ) = _extract_chunk_recursively(
                    client=client,
                    row_id=row_id,
                    document_text=source_text,
                    chunk=chunk,
                    model_name=args.model,
                )
                for chunk_result in chunk_result_records:
                    chunk_result["top_level_chunk_index"] = chunk_index
                entities.extend(chunk_entities)
                rejected.extend(chunk_rejected)
                chunk_records.extend(chunk_result_records)
                _merge_diagnostics(diagnostics, chunk_diagnostics)
                attempts_used += chunk_attempts

            entities = deduplicate_entities(entities)
            status = "ok"
            error_text = None
        except Exception as exc:  # noqa: BLE001 - save progress before stopping
            duration = time.perf_counter() - request_start
            print(
                f"ERROR while processing document {index}/{len(docs)} "
                f"(row_id={row_id}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            error_record = {
                "row_id": row_id,
                "status": "error",
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "chunking_version": CHUNKING_VERSION,
                "chunk_target_chars": args.chunk_chars,
                "prompt_hash": expected_prompt_hash,
                "duration_seconds": round(duration, 6),
                "error": str(exc),
            }
            append_jsonl(error_record, cache_file)
            raise

        duration = time.perf_counter() - request_start
        prompt_eval_count = 0
        eval_count = 0
        for chunk_record in chunk_records:
            usage = chunk_record.get("api_usage") or {}
            prompt_value = usage.get("prompt_eval_count")
            eval_value = usage.get("eval_count")
            if isinstance(prompt_value, int):
                prompt_eval_count += prompt_value
            if isinstance(eval_value, int):
                eval_count += eval_value

        record = {
            "row_id": row_id,
            "status": status,
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "chunk_target_chars": args.chunk_chars,
            "prompt_hash": expected_prompt_hash,
            "duration_seconds": round(duration, 6),
            "attempts_used": attempts_used,
            "top_level_chunk_count": len(chunks),
            "completed_segment_count": len(chunk_records),
            "api_usage": {
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
            },
            "chunk_results": chunk_records,
            "diagnostics": diagnostics,
            "rejected_outputs": rejected,
            "entities": entities,
            "error": error_text,
        }
        append_jsonl(record, cache_file)
        cached_by_row[row_id] = record

        if index == 1 or index % 10 == 0 or index == len(docs):
            print(
                f"Processed {index}/{len(docs)} documents; "
                f"row_id={row_id}; segments={len(chunk_records)}; "
                f"accepted entities={len(entities)}"
            )

    completed_records: list[dict] = []
    all_entities: list[dict] = []
    for document in docs:
        row_id = int(document["row_id"])
        record = cached_by_row.get(row_id)
        if record is None:
            raise RuntimeError(f"Missing completed MedGemma record for row_id {row_id}")
        completed_records.append(record)
        for entity in record.get("entities", []):
            all_entities.append(entity)

    all_entities = deduplicate_entities(all_entities)
    save_jsonl(all_entities, output_file)

    summed_request_seconds = 0.0
    completed_segments = 0
    for record in completed_records:
        summed_request_seconds += float(record.get("duration_seconds", 0.0))
        completed_segments += int(record.get("completed_segment_count", 0))
    average_seconds = summed_request_seconds / len(docs) if docs else 0.0
    wall_seconds = time.perf_counter() - total_wall_start
    diagnostics = sum_diagnostics(completed_records)

    summary = {
        "split": split,
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "chunk_target_chars": args.chunk_chars,
        "prompt_hash": expected_prompt_hash,
        "document_count": len(docs),
        "completed_segment_count": completed_segments,
        "entity_count": len(all_entities),
        "summed_request_seconds": round(summed_request_seconds, 6),
        "average_seconds_per_document": round(average_seconds, 6),
        "current_process_wall_seconds": round(wall_seconds, 6),
        "diagnostics": diagnostics,
        "prediction_file": str(output_file),
        "cache_file": str(cache_file),
        "is_smoke_run": args.limit is not None,
    }
    save_json(summary, summary_file)

    if args.limit is None:
        record_runtime(
            split,
            "medgemma",
            total_seconds=summed_request_seconds,
            average_seconds=average_seconds,
            document_count=len(docs),
            entity_count=len(all_entities),
        )

    print(f"Saved {len(all_entities)} MedGemma entities to {output_file}")
    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
