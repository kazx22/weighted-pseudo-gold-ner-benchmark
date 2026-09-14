import json

from src.medgemma_bc5cdr import (
    TextChunk,
    _extract_chunk_recursively,
    split_document_into_chunks,
)


def test_chunks_preserve_offsets_and_size():
    text = "Title sentence. Aspirin reduced severe headache. Another sentence."
    chunks = split_document_into_chunks(text, maximum_chars=100)
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char:chunk.end_char] == chunk.text
        assert len(chunk.text) <= 100


class _FakeClient:
    def chat(self, *, user_prompt, response_schema, max_output_tokens):
        marker = "SOURCE SEGMENT START\n"
        segment = user_prompt.split(marker, 1)[1].rsplit("\nSOURCE SEGMENT END", 1)[0]
        entities = []
        start = segment.find("aspirin")
        if start >= 0:
            entities.append(
                {
                    "text": "aspirin",
                    "label": "CHEMICAL",
                    "start_char": start,
                    "end_char": start + 7,
                }
            )
        return {
            "message": {"content": json.dumps({"entities": entities})},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 10,
        }


def test_chunk_local_offsets_are_shifted_to_document_offsets():
    document = "No finding. The patient received aspirin and improved."
    chunk = TextChunk(12, len(document), document[12:])
    entities, _, _, _, _ = _extract_chunk_recursively(
        client=_FakeClient(),
        row_id=1,
        document_text=document,
        chunk=chunk,
        model_name="fake",
    )
    assert len(entities) == 1
    assert entities[0]["start_char"] == document.index("aspirin")
    assert entities[0]["end_char"] == document.index("aspirin") + 7
