# BioRED disease/chemical external-validation patch

This patch adds a **separate BioRED pipeline** to the existing project. It does not overwrite BC5CDR outputs.

## Scope

Only these BioRED entity classes are used as gold targets:

- `DiseaseOrPhenotypicFeature` -> `DISEASE`
- `ChemicalEntity` -> `CHEMICAL`

All other BioRED entity classes and all relation rows are ignored for this NER experiment.

## Raw data: easiest setup

Put the official `BIORED.zip` in either location:

```text
data\raw\BIORED.zip
```

or

```text
data\raw\biored\BIORED.zip
```

You do **not** need to extract it. `src\parse_biored.py` finds `Dev.PubTator` and `Test.PubTator` inside the archive and extracts them automatically.

The parser also accepts an already-extracted `BioRED` folder under `data\raw` or `data\raw\biored`.

## Recommended run order

First run:

```text
07_RUN_BIORED_CORE.cmd
```

This performs:

1. Parse BioRED development and test data.
2. Run scispaCy, BioBERT, PubMedBERT, ClinicalBERT, and BioELECTRA on development.
3. Fit BioRED-specific model weights and weighted threshold on development only.
4. Evaluate development.
5. Run the five models on test.
6. Apply the frozen development configuration to test without loading test human gold during construction.
7. Evaluate test against human gold.
8. Run 1,000-resample paired bootstrap tests.
9. Run token-level Cohen's kappa diagnostics.

Then start Ollama and run:

```text
08_RUN_BIORED_MEDGEMMA.cmd
```

This performs:

1. One-document smoke test.
2. MedGemma zero-shot extraction on development using the 450-character chunked protocol.
3. Selective MedGemma tie-breaking on development and freezes the routing band.
4. MedGemma zero-shot extraction on test.
5. Applies the frozen selective routing procedure to test without reading test human gold during construction.
6. Evaluates MedGemma and the hybrid.
7. Runs the MedGemma bootstrap comparisons.
8. Generates paper-ready graphs, BIO confusion matrices, candidate audit, and a Markdown report.

`09_RUN_BIORED_ALL.cmd` runs both stages in one go.

## Important safeguards

- BioRED paths are under `data\processed\biored`, `data\gold\biored`, and `results\biored`.
- BC5CDR results are untouched.
- Development gold is used to fit BioRED model weights and the weighted threshold.
- Test gold is not read during pseudo-gold construction or selective MedGemma construction.
- MedGemma zero-shot inference has a durable per-document cache.
- MedGemma tie-breaking has a durable per-candidate cache.
- The report script never calls Ollama.
- Existing conventional prediction files are skipped by the PowerShell runner unless `-ForceModels` is supplied manually.

## Main outputs

```text
results\biored\dev\frozen_pseudo_gold_config.json
results\biored\test\evaluation_results.json
results\biored\test\bootstrap_results.json
results\biored\test\medgemma_evaluation.json
results\biored\test\medgemma_bootstrap_results.json
results\biored\test\biored_report\BIORED_MEDGEMMA_REPORT.md
results\biored\test\biored_report\figures\
```

## Paper interpretation rule

Do not claim that the study evaluates all BioRED entity types. The correct wording is:

> BioRED was used as an external validation corpus restricted to the disease/phenotypic-feature and chemical entity categories shared with the two-label task. Other BioRED entity classes were outside the scope of the study.
