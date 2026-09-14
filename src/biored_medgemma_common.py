"""Shared MedGemma/Ollama helpers for the BioRED disease/chemical extension.

The LLM is accessed through the local Ollama HTTP API. No remote API is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from src.utils import ALLOWED_LABELS, deduplicate_entities

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MEDGEMMA_MODEL = "medgemma1.5:4b-it-q4_K_M"
PROMPT_VERSION = "biored-disease-chemical-medgemma-v1"

ZERO_SHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 512},
                    "label": {"type": "string", "enum": ["DISEASE", "CHEMICAL"]},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                },
                "required": ["text", "label", "start_char", "end_char"],
            },
        }
    },
    "required": ["entities"],
}

TIE_BREAK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
}

SYSTEM_PROMPT = """You are a careful biomedical named-entity annotator.
Follow the requested BioRED disease/chemical task exactly. Only explicit
disease/phenotypic-feature and chemical mentions are targets. Do not infer
entities that are not explicitly written in the source text. Return only data
that follows the provided JSON schema."""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def zero_shot_prompt(text: str) -> str:
    schema_text = json.dumps(ZERO_SHOT_SCHEMA, ensure_ascii=False)
    return f"""Extract every explicit DISEASE and CHEMICAL mention from the source segment below.

Rules:
1. Copy each entity text exactly as it appears in this segment.
2. DISEASE corresponds to BioRED DiseaseOrPhenotypicFeature: an explicit
   disease, disorder, pathological condition, sign, symptom, or phenotypic
   feature stated in the text.
3. CHEMICAL corresponds to BioRED ChemicalEntity: an explicit drug, chemical,
   compound, element, or named chemical class.
4. Do not return inferred entities, relations, doses, units, author names, or
   general biomedical words that are not entity mentions.
5. start_char is inclusive and end_char is exclusive. Count offsets from the
   first character of THIS SEGMENT, where the first character is offset 0.
6. Return each repeated occurrence separately.
7. Return at most 64 valid mentions. Do not repeat the same occurrence.
8. Stop after the final entity and close the JSON object.
9. If there are no valid entities, return an empty entities list.

Required JSON schema:
{schema_text}

SOURCE SEGMENT START
{text}
SOURCE SEGMENT END"""


def tie_break_prompt(
    *,
    context_with_markers: str,
    candidate_text: str,
    candidate_label: str,
) -> str:
    schema_text = json.dumps(TIE_BREAK_SCHEMA, ensure_ascii=False)
    return f"""Decide whether one proposed BioRED disease/chemical entity should be accepted.

Context (the proposed span is between [CANDIDATE] markers):
{context_with_markers}

Proposed exact text: {candidate_text}
Proposed label: {candidate_label}

Accept only when all of the following are true:
- the marked text is an explicit biomedical entity in this context;
- the proposed DISEASE or CHEMICAL label is correct;
- the marked boundary is complete and does not contain unrelated words;
- the candidate is not merely a broken word-piece or incomplete fragment.

Judge only this candidate. Do not propose new entities.
Return ACCEPT or REJECT through this JSON schema:
{schema_text}"""


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MEDGEMMA_MODEL,
        timeout_seconds: int = 300,
        temperature: float = 0.0,
        seed: int = 42,
        num_ctx: int = 8192,
        keep_alive: str = "15m",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Start Ollama and confirm that "
                f"{self.base_url} is available. Original error: {exc}"
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON API response.") from exc

    def ensure_ready(self) -> None:
        self._request_json("/api/version")
        tags = self._request_json("/api/tags")
        available: set[str] = set()
        for model_info in tags.get("models", []):
            name = str(model_info.get("name", ""))
            model_name = str(model_info.get("model", ""))
            if name:
                available.add(name)
            if model_name:
                available.add(model_name)

        if self.model not in available:
            available_text = ", ".join(sorted(available)) or "none"
            raise RuntimeError(
                f"Ollama is running, but model '{self.model}' is not installed.\n"
                f"Run: ollama pull {self.model}\n"
                f"Currently available: {available_text}"
            )

    def chat(
        self,
        *,
        user_prompt: str,
        response_schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": response_schema,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": int(max_output_tokens),
            },
            "keep_alive": self.keep_alive,
        }
        return self._request_json("/api/chat", method="POST", payload=payload)


def parse_message_content(api_response: dict[str, Any]) -> dict[str, Any]:
    message = api_response.get("message") or {}
    content = str(message.get("content", "")).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        preview_start = content[:240]
        preview_end = content[-500:] if len(content) > 500 else content
        raise ValueError(
            "MedGemma returned invalid JSON "
            f"(characters={len(content)}, done_reason={api_response.get('done_reason')!r}, "
            f"eval_count={api_response.get('eval_count')!r}). "
            f"Start: {preview_start!r} End: {preview_end!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("MedGemma JSON response must be an object.")
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_occurrences(text: str, needle: str, *, case_sensitive: bool) -> list[int]:
    if not needle:
        return []
    haystack = text if case_sensitive else text.casefold()
    target = needle if case_sensitive else needle.casefold()
    output: list[int] = []
    search_from = 0
    while True:
        index = haystack.find(target, search_from)
        if index < 0:
            break
        output.append(index)
        search_from = index + 1
    return output


def _choose_nearest_occurrence(
    occurrences: list[int],
    estimated_start: int | None,
) -> int | None:
    if not occurrences:
        return None
    if len(occurrences) == 1:
        return occurrences[0]
    if estimated_start is None:
        return None

    distances: list[tuple[int, int]] = []
    for occurrence in occurrences:
        distances.append((abs(occurrence - estimated_start), occurrence))
    distances.sort()
    if len(distances) > 1 and distances[0][0] == distances[1][0]:
        return None
    return distances[0][1]


def normalise_zero_shot_entities(
    *,
    row_id: int,
    source_text: str,
    response_payload: dict[str, Any],
    model_name: str,
) -> tuple[list[dict], dict[str, int], list[dict]]:
    raw_entities = response_payload.get("entities")
    if not isinstance(raw_entities, list):
        raise ValueError("MedGemma response is missing an entities list.")

    output: list[dict] = []
    rejected: list[dict] = []
    diagnostics = {
        "returned_entities": len(raw_entities),
        "accepted_entities": 0,
        "exact_offset_matches": 0,
        "repaired_exact_occurrences": 0,
        "repaired_casefold_occurrences": 0,
        "rejected_invalid_label": 0,
        "rejected_missing_text": 0,
        "rejected_unmapped_text": 0,
        "rejected_ambiguous_occurrence": 0,
    }

    for raw_entity in raw_entities:
        if not isinstance(raw_entity, dict):
            diagnostics["rejected_unmapped_text"] += 1
            rejected.append({"reason": "entity_not_object", "value": raw_entity})
            continue

        returned_text = str(raw_entity.get("text", "")).strip()
        label = str(raw_entity.get("label", "")).strip().upper()
        estimated_start = _safe_int(raw_entity.get("start_char"))
        estimated_end = _safe_int(raw_entity.get("end_char"))

        if label not in ALLOWED_LABELS:
            diagnostics["rejected_invalid_label"] += 1
            rejected.append({"reason": "invalid_label", "value": raw_entity})
            continue
        if not returned_text:
            diagnostics["rejected_missing_text"] += 1
            rejected.append({"reason": "missing_text", "value": raw_entity})
            continue

        mapped_start: int | None = None
        mapping_method = ""

        if estimated_start is not None and estimated_end is not None:
            if 0 <= estimated_start < estimated_end <= len(source_text):
                if source_text[estimated_start:estimated_end] == returned_text:
                    mapped_start = estimated_start
                    mapping_method = "exact_offsets"
                    diagnostics["exact_offset_matches"] += 1

        if mapped_start is None:
            exact_occurrences = _find_occurrences(
                source_text, returned_text, case_sensitive=True
            )
            mapped_start = _choose_nearest_occurrence(
                exact_occurrences, estimated_start
            )
            if mapped_start is not None:
                mapping_method = "repaired_exact_occurrence"
                diagnostics["repaired_exact_occurrences"] += 1

        if mapped_start is None:
            folded_occurrences = _find_occurrences(
                source_text, returned_text, case_sensitive=False
            )
            mapped_start = _choose_nearest_occurrence(
                folded_occurrences, estimated_start
            )
            if mapped_start is not None:
                mapping_method = "repaired_casefold_occurrence"
                diagnostics["repaired_casefold_occurrences"] += 1

        if mapped_start is None:
            exact_occurrences = _find_occurrences(
                source_text, returned_text, case_sensitive=True
            )
            folded_occurrences = _find_occurrences(
                source_text, returned_text, case_sensitive=False
            )
            if len(exact_occurrences) > 1 or len(folded_occurrences) > 1:
                diagnostics["rejected_ambiguous_occurrence"] += 1
                reason = "ambiguous_occurrence"
            else:
                diagnostics["rejected_unmapped_text"] += 1
                reason = "text_not_found_in_source"
            rejected.append({"reason": reason, "value": raw_entity})
            continue

        mapped_end = mapped_start + len(returned_text)
        source_surface = source_text[mapped_start:mapped_end]
        output.append(
            {
                "row_id": int(row_id),
                "text": source_surface,
                "start_char": mapped_start,
                "end_char": mapped_end,
                "label": label,
                "confidence": None,
                "model": model_name,
                "source": "medgemma_zero_shot",
                "mapping_method": mapping_method,
            }
        )
        diagnostics["accepted_entities"] += 1

    return deduplicate_entities(output), diagnostics, rejected


def sentence_context_with_markers(
    text: str,
    start_char: int,
    end_char: int,
    *,
    maximum_context_chars: int = 700,
) -> str:
    left_boundary = max(
        text.rfind(".", 0, start_char),
        text.rfind("!", 0, start_char),
        text.rfind("?", 0, start_char),
        text.rfind("\n", 0, start_char),
    )
    if left_boundary < 0:
        left_boundary = 0
    else:
        left_boundary += 1

    right_candidates: list[int] = []
    for symbol in (".", "!", "?", "\n"):
        location = text.find(symbol, end_char)
        if location >= 0:
            right_candidates.append(location + 1)
    right_boundary = min(right_candidates) if right_candidates else len(text)

    if right_boundary - left_boundary > maximum_context_chars:
        half = maximum_context_chars // 2
        left_boundary = max(0, start_char - half)
        right_boundary = min(len(text), end_char + half)

    relative_start = start_char - left_boundary
    relative_end = end_char - left_boundary
    context = text[left_boundary:right_boundary]
    return (
        context[:relative_start]
        + "[CANDIDATE]"
        + context[relative_start:relative_end]
        + "[/CANDIDATE]"
        + context[relative_end:]
    ).strip()


def candidate_key(entity: dict) -> str:
    return "|".join(
        [
            str(int(entity["row_id"])),
            str(int(entity["start_char"])),
            str(int(entity["end_char"])),
            str(entity["label"]).upper(),
        ]
    )


def append_jsonl(record: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_if_exists(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def call_with_retries(
    client: OllamaClient,
    *,
    user_prompt: str,
    response_schema: dict[str, Any],
    max_output_tokens: int,
    attempts: int = 3,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat(
                user_prompt=user_prompt,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
            )
            return response, attempt
        except Exception as exc:  # noqa: BLE001 - preserve API error details
            last_error = exc
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"MedGemma failed after {attempts} attempts: {last_error}")



def chat_and_parse_with_retries(
    client: OllamaClient,
    *,
    user_prompt: str,
    response_schema: dict[str, Any],
    max_output_tokens: int,
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Retry API/JSON failures with a larger output budget on each attempt.

    A model can obey the JSON schema but still be cut off in the middle of the
    object when ``num_predict`` is too small. Retrying with exactly the same
    token limit reproduces the same truncated prefix when temperature is zero.
    This function therefore increases the output allowance for retries while
    leaving the model, prompt, temperature, seed, and schema unchanged.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")

    token_budgets: list[int] = []
    for attempt_index in range(attempts):
        # 2,048 -> 4,096 -> 6,144 for the current BC5CDR call. The cap avoids
        # an unbounded completion while still leaving room in an 8,192-token
        # context for the prompt and a long structured entity list.
        proposed = int(max_output_tokens * (attempt_index + 1))
        token_budgets.append(min(proposed, 6144))

    last_error: Exception | None = None
    last_response: dict[str, Any] | None = None
    for attempt, token_budget in enumerate(token_budgets, start=1):
        try:
            response = client.chat(
                user_prompt=user_prompt,
                response_schema=response_schema,
                max_output_tokens=token_budget,
            )
            last_response = response

            done_reason = str(response.get("done_reason", "")).strip().lower()
            if done_reason == "length":
                raise ValueError(
                    "MedGemma output reached the generation limit "
                    f"(num_predict={token_budget}, eval_count={response.get('eval_count')!r})."
                )

            parsed = parse_message_content(response)
            return response, parsed, attempt
        except Exception as exc:  # noqa: BLE001 - preserve exact failure
            last_error = exc
            if attempt < attempts:
                next_budget = token_budgets[attempt]
                done_reason = None
                eval_count = None
                if last_response is not None:
                    done_reason = last_response.get("done_reason")
                    eval_count = last_response.get("eval_count")
                print(
                    "MedGemma response was incomplete or invalid on "
                    f"attempt {attempt}/{attempts} "
                    f"(num_predict={token_budget}, done_reason={done_reason!r}, "
                    f"eval_count={eval_count!r}). "
                    f"Retrying with num_predict={next_budget}...",
                    flush=True,
                )
                time.sleep(float(attempt))

    raise RuntimeError(
        "MedGemma API/JSON response failed after "
        f"{attempts} attempts with token budgets {token_budgets}: {last_error}"
    )


def sum_diagnostics(records: Iterable[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in records:
        diagnostics = record.get("diagnostics") or {}
        for key, value in diagnostics.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals
