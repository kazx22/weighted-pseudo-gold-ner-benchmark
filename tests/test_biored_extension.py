from pathlib import Path

from src.biored_candidate_gold import achievable_thresholds, build_vote_table, materialise_pseudo_gold
from src.biored_medgemma import split_document_into_chunks
from src.biored_medgemma_hybrid import derive_band
from src.parse_biored import parse_biored


def test_parser_keeps_only_disease_and_chemical(tmp_path: Path):
    title = "Aspirin and cancer"
    abstract = "Aspirin was studied in BRCA1 cancer."
    full = title + " " + abstract
    aspirin_start = full.index("Aspirin")
    cancer_start = full.index("cancer")
    brca_start = full.index("BRCA1")
    raw = (
        "123|t|" + title + "\n"
        "123|a|" + abstract + "\n"
        f"123\t{aspirin_start}\t{aspirin_start+7}\tAspirin\tChemicalEntity\tD001241\n"
        f"123\t{cancer_start}\t{cancer_start+6}\tcancer\tDiseaseOrPhenotypicFeature\tD009369\n"
        f"123\t{brca_start}\t{brca_start+5}\tBRCA1\tGeneOrGeneProduct\t672\n"
        "123\tAssociation\tD001241\tD009369\tNovel\n"
    )
    path = tmp_path / "Dev.PubTator"
    path.write_text(raw, encoding="utf-8")
    docs, entities, audit = parse_biored(path)
    assert len(docs) == 1
    assert {entity["label"] for entity in entities} == {"CHEMICAL", "DISEASE"}
    assert audit["ignored_entity_labels"]["GeneOrGeneProduct"] == 1


def test_weighted_vote_materialisation():
    predictions = {
        "scispacy": [{"row_id": 1, "start_char": 0, "end_char": 7, "text": "Aspirin", "label": "CHEMICAL"}],
        "biobert": [{"row_id": 1, "start_char": 0, "end_char": 7, "text": "Aspirin", "label": "CHEMICAL"}],
        "pubmedbert": [],
        "clinicalbert": [],
        "bioelectra": [],
    }
    weights = {"scispacy": 0.8, "biobert": 0.6, "pubmedbert": 0.4, "clinicalbert": 0.3, "bioelectra": 0.2}
    vote_table, store = build_vote_table(predictions)
    output = materialise_pseudo_gold(vote_table, store, weights=weights, threshold=1.2)
    assert len(output) == 1
    assert output[0]["weighted_score"] == 1.4
    assert achievable_thresholds(weights)


def test_chunk_offsets_preserve_source():
    text = "First sentence. " + ("Aspirin cancer evidence; " * 40) + "Last sentence."
    chunks = split_document_into_chunks(text, maximum_chars=450)
    assert chunks
    for chunk in chunks:
        assert text[chunk.start_char:chunk.end_char] == chunk.text
        assert len(chunk.text) <= 450


def test_routing_band_surrounds_threshold():
    weights = {"scispacy": 0.8, "biobert": 0.6, "pubmedbert": 0.4, "clinicalbert": 0.3, "bioelectra": 0.2}
    scores = achievable_thresholds(weights)
    base = scores[len(scores)//2]
    low, high = derive_band(weights=weights, base_threshold=base, levels_below=1, levels_above=1)
    assert low < base < high or low < high
