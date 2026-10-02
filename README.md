<<<<<<< HEAD
# Biomedical NER — clean modular reproducible pipeline

This package contains raw BC5CDR/BioRED inputs and modular source code. Generated predictions, processed data, figures, reports, caches, and logs are intentionally excluded.

## Top-level runners

```text
RUN_TRANSFORMERS.cmd  Conventional NER + methodological controls
RUN_MEDGEMMA.cmd     MedGemma stages only; requires transformer outputs
RUN_ALL.cmd          Runs both in sequence
```

The CMD files expect a Windows virtual environment at `.venv\Scripts\python.exe`.
For detailed usage and troubleshooting, read **QUICK_MANUAL.md**.

## Experiments included

### BC5CDR
- five conventional NER systems;
- development F1 weighting and weighted-threshold selection;
- held-out frozen test evaluation;
- fixed 3-of-5 majority baseline;
- development-tuned unweighted k-of-5 control;
- leave-one-model-out weighted ablation;
- 25/50/75/100% development-gold sensitivity;
- paired bootstrap + Holm correction;
- Cohen's kappa and qualitative taxonomy;
- MedGemma zero-shot and selective adjudication.

### BioRED official
The same disease/chemical experiment is run on the released BioRED development and test splits.

### BioRED overlap-excluded sensitivity analysis
The code identifies BioRED PMIDs that occur in any BC5CDR split and removes them before the clean BioRED experiment. The retained development set is recalibrated independently, then the frozen configuration is evaluated on the retained test set. The same conventional controls and MedGemma analyses are repeated.

The original BioRED experiment is retained and compared directly with the overlap-excluded analysis.

## Correct model identities

The fifth conventional system is **D4Data NER (DistilBERT)** using `d4data/biomedical-ner-all`. The incorrect legacy model label has been removed throughout the codebase.

The merged disease/chemical system is reported as **PubMedBERT + OpenMed**.

## Raw inputs

Included under `data\raw\`:

- official BC5CDR train/development/test PubTator files;
- BioRED ZIP.
=======
# Performance-Weighted Pseudo-Gold Biomedical NER Benchmark

This repository contains the code for a biomedical named entity recognition study using **BC5CDR** and **BioRED**.

The main goal is simple:

> Can several ready-to-run biomedical NER systems be combined into a better pseudo-gold annotation set by giving stronger systems more voting weight?

Five NER systems are evaluated under one exact-span pipeline. Their development-set F1 scores are used as voting weights. The voting threshold is selected on development data and then frozen before test evaluation.

No additional fine-tuning is performed in this study.

---

## Main NER Systems

| System | Checkpoint(s) | Configuration |
|---|---|---|
| **scispaCy** | `en_ner_bc5cdr_md` | Single spaCy biomedical NER pipeline |
| **BioBERT** | `alvaroalon2/biobert_diseases_ner` + `alvaroalon2/biobert_chemical_ner` | Separate disease and chemical checkpoints |
| **PubMedBERT + OpenMed** | `sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease` + `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M` | PubMedBERT for disease and OpenMed for chemical |
| **ClinicalBERT** | `samrawal/bert-base-uncased_clinical-ner` | Clinical NER checkpoint with deterministic label mapping |
| **D4Data NER (DistilBERT)** | `d4data/biomedical-ner-all` | Biomedical DistilBERT NER checkpoint |

BioBERT and PubMedBERT + OpenMed use two specialist checkpoints.

For PubMedBERT + OpenMed:

- PubMedBERT predicts `DISEASE`
- OpenMed predicts `CHEMICAL`
- both outputs are merged
- the merged system receives one development weight
- the merged system receives one ensemble vote

---

## Datasets

### BC5CDR

BC5CDR contains PubMed abstracts annotated for:

- `DISEASE`
- `CHEMICAL`

The official dataset is not redistributed in this repository.

Place the PubTator files under:

```text
data/raw/bc5cdr/
```

### BioRED

BioRED is used as an external-corpus validation dataset.

The experiments use the shared entity types:

- `DISEASE`
- `CHEMICAL`

The official BioRED development and test splits are evaluated independently.

---

## BioRED Overlap-Excluded Sensitivity Analysis

Some BioRED documents also appear in BC5CDR.

The pipeline checks BioRED PMIDs against all BC5CDR splits:

```text
BC5CDR train
BC5CDR dev
BC5CDR test
```

Shared documents are removed for a separate sensitivity experiment.

The overlap-excluded branch then repeats the full development-to-test pipeline using its own:

- model weights
- weighted threshold
- test evaluation
- bootstrap analysis
- tuned unweighted control
- leave-one-model-out analysis
- MedGemma experiments

The official BioRED results are kept. They are not replaced by the overlap-excluded analysis.

---

## Evaluation Design

The main evaluation uses exact character-span and entity-label matching.

A prediction is correct only when all of the following match:

```text
document
start offset
end offset
entity label
```

### Development

On development data:

1. Run the five NER systems.
2. Compare each system with human gold.
3. Use each system's exact-span micro-F1 as its voting weight.
4. Sum the weights of systems that predict the same entity.
5. Search the weighted threshold on development data.
6. Freeze the weights and threshold.

### Test

On test data:

1. Run the same five systems.
2. Use the frozen development weights.
3. Use the frozen development threshold.
4. Build the weighted pseudo-gold set.
5. Compare the final output with human test gold.

The test gold is not used to change the weights or threshold.

---

## Weighted Voting

For candidate entity \(e\):

```text
Score(e) = sum of development F1 weights of systems that predicted e
```

The entity is accepted when:

```text
Score(e) >= development-selected threshold
```

Each system has one vote with a different voting weight.

---

## Baselines and Controls

The study includes several controls.

### Fixed Majority Voting

A simple equal-vote majority baseline.

### Tuned Unweighted k-of-5 Voting

The development split is used to test:

```text
k = 1, 2, 3, 4, 5
```

The best development value is frozen and applied to test data.

### Leave-One-Model-Out Ablation

Each model is removed once.

The remaining four systems keep their original development weights.

A new weighted threshold is selected on development data and then frozen for test evaluation.

### Development-Gold Sensitivity

The weighted method is calibrated using:

```text
25%
50%
75%
100%
```

of the available development gold.

The same untouched test split is then used for evaluation.

---

## Statistical Analysis

The main analysis includes:

- exact-span precision
- exact-span recall
- exact-span F1
- paired document-level bootstrap
- 1,000 bootstrap resamples
- Holm correction
- 95% bootstrap confidence intervals
- Cohen's kappa as a secondary token-level agreement diagnostic

Cohen's kappa is not used as the primary NER metric because the frequent `O` class can increase token-level agreement.

---

## Error Analysis

A span-level error analysis is applied to the main conventional systems.

The categories are:

- false positive
- false negative
- boundary error
- type confusion

This analysis is used to study where the systems disagree with human gold.

---

## MedGemma Extension

The project also evaluates:

```text
medgemma1.5:4b-it-q4_K_M
```

through local Ollama inference.
>>>>>>> 44764c1567d24e547e805c23dcc03f4c033be23e

MedGemma is **not** one of the five weighted NER systems.

<<<<<<< HEAD
The runners create outputs under locations including:

```text
results\dev\
results\test\
results\biored\
results\biored_overlap_excluded\
results\biored_sensitivity_comparison\
data\processed\
data\gold\
logs\
```

See `QUICK_MANUAL.md` for the recommended run order and examples for rerunning individual modules.
=======
It is tested in two roles.

### Zero-Shot NER

MedGemma directly extracts disease and chemical entities.

### Selective Adjudication

The weighted ensemble automatically:

- rejects low-score candidates
- accepts high-score candidates
- sends only borderline candidates to MedGemma

MedGemma then decides whether those borderline candidates should be accepted.

Document-level caches make the long MedGemma runs resumable.

---

# Main Results

## BC5CDR

### Individual Systems

| System | Precision | Recall | F1 |
|---|---:|---:|---:|
| scispaCy | 0.7962 | 0.7112 | **0.7513** |
| BioBERT | 0.4317 | 0.7151 | 0.5384 |
| PubMedBERT + OpenMed | 0.3082 | 0.5507 | 0.3952 |
| ClinicalBERT | 0.2466 | 0.3979 | 0.3045 |
| D4Data NER | 0.1723 | 0.2172 | 0.1922 |

### Pseudo-Gold

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Majority pseudo-gold | 0.9170 | 0.5482 | 0.6861 |
| **Weighted pseudo-gold** | **0.9204** | **0.6676** | **0.7739** |

The weighted method improves F1 by **0.0877** over majority voting.

Its 95% bootstrap confidence interval is:

```text
[0.7601, 0.7870]
```

---

## BioRED Official Split

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Majority pseudo-gold | 0.8753 | 0.5123 | 0.6463 |
| scispaCy | 0.6586 | 0.6361 | 0.6472 |
| **Weighted pseudo-gold** | **0.8135** | **0.6475** | **0.7211** |

The weighted method improves F1 by:

```text
+0.0748 over majority voting
+0.0739 over scispaCy
```

Both comparisons remain significant after Holm correction.

---

## BioRED Overlap-Excluded Split

| Method | F1 |
|---|---:|
| Majority pseudo-gold | 0.6244 |
| scispaCy | 0.5557 |
| **Weighted pseudo-gold** | **0.6649** |

The weighted method remains above both controls after overlapping BC5CDR documents are removed.

---

## MedGemma Results

### BC5CDR

| Method | F1 |
|---|---:|
| MedGemma zero-shot | 0.4002 |
| Weighted pseudo-gold | 0.7739 |
| Weighted + MedGemma | 0.7741 |

The selective MedGemma stage produced almost no change on BC5CDR.

The difference between weighted voting and weighted + MedGemma was not statistically significant.

### BioRED Official

| Method | F1 |
|---|---:|
| MedGemma zero-shot | 0.3424 |
| Weighted pseudo-gold | **0.7211** |
| Weighted + MedGemma | 0.6614 |

MedGemma increased recall but introduced many false positives.

The hybrid therefore performed below the weighted baseline.

### BioRED Overlap-Excluded

| Method | F1 |
|---|---:|
| MedGemma zero-shot | 0.2622 |
| Weighted pseudo-gold | 0.6649 |
| Weighted + MedGemma | **0.6714** |

The gain is small but statistically significant in the overlap-excluded sensitivity experiment.

Overall, MedGemma did not provide a consistent improvement over the weighted method.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── bc5cdr/
│   │   └── biored/
│   ├── processed/
│   └── gold/
│
├── src/
│   ├── parse_bc5cdr.py
│   ├── build_gold_bc5cdr.py
│   ├── scispacy_bc5cdr.py
│   ├── biobert_bc5cdr.py
│   ├── pubmed_bc5cdr.py
│   ├── clinicalbert_bc5cdr.py
│   ├── candidate_gold.py
│   ├── threshold_sensitivity.py
│   ├── bc5cdr_evaluation.py
│   ├── bootstrap_significance.py
│   ├── cohen_kappa.py
│   ├── error_taxonomy.py
│   ├── method_controls.py
│   ├── medgemma_bc5cdr.py
│   ├── medgemma_hybrid.py
│   ├── biored_medgemma.py
│   └── ...
│
├── results/
│   ├── dev/
│   ├── test/
│   ├── biored/
│   ├── biored_overlap_excluded/
│   └── biored_sensitivity_comparison/
│
├── figure/
├── logs/
├── requirements.txt
├── requirements-lock.txt
├── RUN_ALL.cmd
└── README.md
```

---

## Setup

The project was developed and tested with Python 3.

Create or activate a Python environment and install the dependencies:

```cmd
pip install -r requirements.txt
```

The scispaCy BC5CDR model is also required:

```cmd
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

For MedGemma experiments, Ollama must be running locally.

The expected model is:

```text
medgemma1.5:4b-it-q4_K_M
```

---

## Running the Full Pipeline

From the project root:

```cmd
RUN_ALL.cmd
```

The runner is resumable.

If a long stage stops, run the same command again.

Completed stages are kept and the pipeline continues from the unfinished stage.

---

## Generated Outputs

Main result folders:

```text
results/dev/
results/test/
results/biored/
results/biored_overlap_excluded/
results/biored_sensitivity_comparison/
```

Generated figures are stored under:

```text
figure/
```

Logs are stored under:

```text
logs/
```

---

## Study Scope

This repository evaluates the behaviour of these exact ready-to-run configurations.

The reported numbers should not be treated as a comparison with fully fine-tuned state-of-the-art biomedical NER systems.

The study focuses on:

- off-the-shelf model behaviour
- weighted ensemble construction
- pseudo-gold quality
- development-to-test generalisation
- cross-corpus validation
- robustness controls
- local LLM adjudication

---

## Current Status

The main experimental pipeline is complete.

This includes:

- BC5CDR
- BioRED
- overlap-excluded BioRED
- weighted voting
- majority voting
- tuned unweighted voting
- leave-one-model-out analysis
- development-gold sensitivity
- bootstrap testing
- Holm correction
- Cohen's kappa
- error analysis
- MedGemma zero-shot
- MedGemma selective adjudication

The current codebase replaces the earlier Legacy 3.0 experimental pipeline.

---

## License

This repository is released under the MIT License.
>>>>>>> 44764c1567d24e547e805c23dcc03f4c033be23e
