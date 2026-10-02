# Biomedical NER — clean modular reproducible pipeline

This package contains raw BC5CDR/BioRED inputs and modular source code. Generated predictions, processed data, figures, reports, caches, and logs are intentionally excluded.

## Top-level runners

```text
RUN_TRANSFORMERS.cmd  Conventional NER + methodological controls
RUN_CWA.cmd          CWA on BC5CDR + official BioRED, including bootstrap
RUN_MEDGEMMA.cmd     MedGemma stages only; requires transformer outputs
RUN_ALL.cmd          Runs transformer and MedGemma stages
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
`RUN_CWA.cmd` adds the MV -> WV -> CWA comparison to BC5CDR and official BioRED. Each dataset learns its own content reliabilities, alpha, and threshold from DEV only. TEST evaluation includes entity-type, span-length, mention-form analysis, and paired bootstrap comparisons of CWA against WV and MV.

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
