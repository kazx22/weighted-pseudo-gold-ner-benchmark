from __future__ import annotations

from src.cpu_ner import enforce_cpu, cpu_pipeline
enforce_cpu()
from transformers import AutoModelForTokenClassification, AutoTokenizer

BIOBERT_DISEASE = "alvaroalon2/biobert_diseases_ner"
PUBMEDBERT_DISEASE = (
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease"
)

LABEL_OVERRIDES = {
    BIOBERT_DISEASE: {
        0: "B-DISEASE",
        1: "I-DISEASE",
        2: "O",
    },
    PUBMEDBERT_DISEASE: {
        0: "O",
        1: "B-DISEASE",
        2: "I-DISEASE",
    },
}


def make_ner_pipeline(model_name: str):
    enforce_cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    override = LABEL_OVERRIDES.get(model_name)
    if override is not None:
        if int(model.config.num_labels) != len(override):
            raise ValueError(
                f"Unexpected number of labels for {model_name}: "
                f"{model.config.num_labels} != {len(override)}"
            )
        model.config.id2label = dict(override)
        model.config.label2id = {label: idx for idx, label in override.items()}

    model.to("cpu")
    return cpu_pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        ignore_labels=["O"],
        device=-1,
    )
