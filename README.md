# Biomedical NER — clean reproducible runner

This package contains only raw inputs, executable source code, and one user-facing runner:

```text
RUN_ALL.cmd
```

Double-click `RUN_ALL.cmd` (or run it from CMD). No source-code editing is required.
The runner is resumable: if a long stage fails, run the same file again and completed stages are skipped.

## What the one runner executes

### BC5CDR
1. Parse the official BC5CDR corpus.
2. Run scispaCy, BioBERT, PubMedBERT + OpenMed, ClinicalBERT, and D4Data NER (DistilBERT) on development.
3. Estimate development exact-span F1 weights and select the weighted threshold.
4. Freeze the development configuration.
5. Run the five systems on held-out test data and evaluate majority and weighted pseudo-gold.
6. Run paired bootstrap tests, Holm correction, Cohen's kappa, and the qualitative error taxonomy.
7. Run two additional methodological controls:
   - development-tuned unweighted k-of-5 consensus;
   - leave-one-model-out weighted ablation, recalibrated on development and tested with paired bootstrap + Holm correction.
8. Run MedGemma zero-shot and selective MedGemma adjudication, followed by evaluation, bootstrap analysis, routing/offset diagnostics, and reporting.

### BioRED — official released split
The same disease/chemical pipeline is run on the official BioRED development and test splits, including the two methodological controls and both MedGemma modes.

### BioRED — BC5CDR-overlap-excluded sensitivity analysis
The code compares every BioRED PMID against BC5CDR train, development, and test. Any shared PMID is removed from BioRED before the sensitivity experiment.

The sensitivity branch then repeats the complete experiment independently:
- five conventional systems;
- new clean-development weights;
- new clean-development weighted threshold;
- frozen clean-test evaluation;
- bootstrap, Holm correction, kappa, taxonomy;
- tuned-unweighted control;
- leave-one-model-out ablation;
- MedGemma zero-shot;
- selective MedGemma adjudication and diagnostics.

The official BioRED experiment is retained. It is not replaced by the overlap-excluded analysis.
A final report compares official BioRED against the overlap-excluded sensitivity results.

## Overlap rule

The sensitivity analysis excludes a BioRED document when its PMID occurs in **any** BC5CDR split (train, development, or test). The runner also records whether the overlapping title + abstract text matches exactly.

With the raw files included in this package, the audit currently identifies:
- BioRED development: 32/100 overlapping documents, leaving 68;
- BioRED test: 34/100 overlapping documents, leaving 66.

These values are recomputed by the code during every fresh run; they are not hard-coded into the analysis.

## Correct model identities

The fifth system is **D4Data NER (DistilBERT)** using:

```text
d4data/biomedical-ner-all
```

It is not presented as an ELECTRA model anywhere in this clean codebase.

The two-checkpoint system reported as **PubMedBERT + OpenMed** uses a PubMedBERT disease checkpoint and an OpenMed chemical checkpoint.

## Requirements

The runner first uses `.venv\Scripts\python.exe` when that environment already exists. Otherwise it uses `python` from PATH.

The Python environment must contain the packages in `requirements.txt` and the scispaCy model `en_ner_bc5cdr_md`.

For MedGemma stages, Ollama must be running locally and this model must already be installed:

```text
medgemma1.5:4b-it-q4_K_M
```

If Ollama is unavailable, all conventional/core experiments already completed remain on disk. Start Ollama and run `RUN_ALL.cmd` again; the runner resumes from the first unfinished stage.

## Generated outputs

The distributed package intentionally contains **no** processed data, predictions, caches, results, figures, or logs. They are created only when `RUN_ALL.cmd` is executed.

Main generated locations include:

```text
results\dev\
results\test\
results\biored\
results\biored_overlap_excluded\
results\biored_sensitivity_comparison\
logs\
```
