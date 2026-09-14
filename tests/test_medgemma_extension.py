from __future__ import annotations

import unittest

from src.medgemma_common import normalise_zero_shot_entities, sentence_context_with_markers
from src.medgemma_hybrid import derive_band


class MedGemmaHelpersTest(unittest.TestCase):
    def test_exact_offsets_are_preserved(self) -> None:
        text = "Aspirin caused headache."
        payload = {
            "entities": [
                {"text": "Aspirin", "label": "CHEMICAL", "start_char": 0, "end_char": 7}
            ]
        }
        entities, diagnostics, rejected = normalise_zero_shot_entities(
            row_id=1,
            source_text=text,
            response_payload=payload,
            model_name="test-model",
        )
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["start_char"], 0)
        self.assertEqual(diagnostics["exact_offset_matches"], 1)
        self.assertEqual(rejected, [])

    def test_bad_offset_is_repaired_only_by_exact_text(self) -> None:
        text = "Aspirin caused headache."
        payload = {
            "entities": [
                {"text": "headache", "label": "DISEASE", "start_char": 0, "end_char": 8}
            ]
        }
        entities, diagnostics, _ = normalise_zero_shot_entities(
            row_id=1,
            source_text=text,
            response_payload=payload,
            model_name="test-model",
        )
        self.assertEqual(entities[0]["start_char"], 15)
        self.assertEqual(diagnostics["repaired_exact_occurrences"], 1)

    def test_unmapped_hallucinated_text_is_rejected(self) -> None:
        text = "Aspirin caused headache."
        payload = {
            "entities": [
                {"text": "nausea", "label": "DISEASE", "start_char": 0, "end_char": 6}
            ]
        }
        entities, diagnostics, rejected = normalise_zero_shot_entities(
            row_id=1,
            source_text=text,
            response_payload=payload,
            model_name="test-model",
        )
        self.assertEqual(entities, [])
        self.assertEqual(diagnostics["rejected_unmapped_text"], 1)
        self.assertEqual(rejected[0]["reason"], "text_not_found_in_source")

    def test_adjacent_score_band(self) -> None:
        weights = {
            "scispacy": 0.7596058,
            "biobert": 0.60041357,
            "pubmedbert": 0.40923915,
            "clinicalbert": 0.30209633,
            "bioelectra": 0.18487707,
        }
        low, high = derive_band(
            weights=weights,
            base_threshold=1.16884495,
            levels_below=1,
            levels_above=1,
        )
        self.assertAlmostEqual(low, 1.08738697)
        self.assertAlmostEqual(high, 1.19452979)

    def test_context_marks_candidate(self) -> None:
        text = "Aspirin caused severe headache. Recovery followed."
        start = text.index("severe")
        end = start + len("severe headache")
        context = sentence_context_with_markers(text, start, end)
        self.assertIn("[CANDIDATE]severe headache[/CANDIDATE]", context)


class AdaptiveRetryTest(unittest.TestCase):
    def test_truncated_json_retries_with_larger_budget(self) -> None:
        from src.medgemma_common import chat_and_parse_with_retries

        class FakeClient:
            def __init__(self) -> None:
                self.budgets: list[int] = []

            def chat(self, *, user_prompt, response_schema, max_output_tokens):
                self.budgets.append(max_output_tokens)
                if len(self.budgets) == 1:
                    return {
                        "message": {"content": '{"entities": [{"text": "Aspirin"'},
                        "done_reason": "length",
                        "eval_count": max_output_tokens,
                    }
                return {
                    "message": {"content": '{"entities": []}'},
                    "done_reason": "stop",
                    "eval_count": 8,
                }

        client = FakeClient()
        _, parsed, attempts_used = chat_and_parse_with_retries(
            client,
            user_prompt="test",
            response_schema={"type": "object"},
            max_output_tokens=2048,
            attempts=3,
        )
        self.assertEqual(parsed, {"entities": []})
        self.assertEqual(attempts_used, 2)
        self.assertEqual(client.budgets, [2048, 4096])

    def test_invalid_json_error_includes_generation_metadata(self) -> None:
        from src.medgemma_common import parse_message_content

        with self.assertRaisesRegex(ValueError, "done_reason='length'"):
            parse_message_content(
                {
                    "message": {"content": '{"entities": ['},
                    "done_reason": "length",
                    "eval_count": 2048,
                }
            )


if __name__ == "__main__":
    unittest.main()
