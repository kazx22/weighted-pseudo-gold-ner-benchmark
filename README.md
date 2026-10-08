# Biomedical NER — clean modular reproducible pipeline

This package contains raw BC5CDR/BioRED inputs and modular source code. Generated predictions, processed data, figures, reports, caches, and logs are intentionally excluded.

## Top-level runners

```text
RUN_TRANSFORMERS.cmd  Conventional NER + methodological controls
RUN_SPAN_AUDIT.cmd    Audit saved prediction spans only; no model inference
RUN_KAPPA_CHECK.cmd   BioRED span audit + kappa + taxonomy recheck
RUN_CAWV.cmd          CAWV on BC5CDR + official BioRED, including bootstrap
RUN_MEDGEMMA.cmd      MedGemma stages only; requires transformer outputs
RUN_ALL.cmd           Conventional -> CAWV -> MedGemma in dependency order
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

### Content-Aware Weighted Voting
`RUN_CAWV.cmd` adds the MV -> WV -> CAWV comparison to BC5CDR and official BioRED. Each dataset learns its own content reliabilities, alpha, and threshold from DEV only. TEST evaluation includes entity-type, span-length, mention-form analysis, and paired bootstrap comparisons of CAWV against WV and MV.

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

## Generated outputs

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


## CAWV + MedGemma ablation

`RUN_MEDGEMMA.cmd` now also evaluates selective MedGemma routing on top of the frozen CAWV ensemble for BC5CDR and official BioRED. The routing band is derived from development CAWV scores around the frozen CAWV threshold and then applied unchanged to test.

`RUN_ALL.cmd` runs conventional experiments, CAWV, and MedGemma in dependency order. Completed stages are skipped on rerun.

## BioRED kappa sanity check

Run `RUN_KAPPA_CHECK.cmd` to recheck token-level kappa from saved BioRED predictions only. It does not rerun NER inference. The check reports overlapping prediction tokens, binary entity-vs-O kappa, positive-token precision/recall, exact-span metrics, and five BioBERT gold/predicted BIO examples. Outputs are written to `results\biored\test\kappa_diagnostic\`.

## Prediction-span integrity audit

`RUN_SPAN_AUDIT.cmd` uses saved predictions only. It checks all five systems on BC5CDR and both BioRED branches for invalid bounds, missing documents, `text[start:end]` mismatches, very long spans, spans covering a large fraction of a document, and sentence-crossing warnings. The 50 longest predictions per model are saved for inspection. Hard offset/text corruption stops the pipeline; long-span warnings are reported without modifying predictions.

The BIO converter already uses token positions from `re.finditer(r"\S+", text)` rather than reconstructing offsets from assumed single spaces.

`RUN_KAPPA_CHECK.cmd` now runs the official BioRED TEST span audit first, regenerates kappa diagnostics, then regenerates the all-five-model error taxonomy.
