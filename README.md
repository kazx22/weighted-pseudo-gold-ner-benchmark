# CAWV Biomedical NER Reproducibility Repository

This is the clean, source-data-only repository for the BC5CDR and BioRED study.

## One command

Open the folder in VS Code on Windows and run:

```cmd
RUN_ALL.cmd
```

`RUN_ALL.cmd` creates or repairs `.venv`, installs the required Python packages, verifies CUDA/PyTorch and the scispaCy model, then runs the complete study.

The repository ships with **only the two source datasets** under `data/raw/`. Everything else is generated locally: processed corpora, predictions, pseudo-gold files, checkpoints, bootstrap outputs, Cohen's kappa, diagnostics, taxonomy, CAWV ablations, MedGemma results, fine-tuned controls, resource measurements, tables and figures.

## Root commands

- `RUN_ALL.cmd` — complete paper pipeline. This is the main command.
- `RUN_BC5CDR.cmd` — BC5CDR branch only.
- `RUN_BIORED.cmd` — BioRED official plus overlap-excluded sensitivity branch.
- `RUN_MEDGEMMA.cmd` — MedGemma stages after the conventional branches exist.
- `RUN_FINETUNED_MODELS.cmd` — four supervised reference encoders on BC5CDR and BioRED.
- `SETUP_ENVIRONMENT.cmd` — create/repair `.venv` only.
- `CLEAN_GENERATED.cmd` — delete generated outputs while preserving the two raw datasets.

There are no temporary `*_FIXED.cmd`, tokenizer-debug runners, or BioBERT-special runners in the final interface.

## Main symmetric design

BC5CDR and BioRED official are generated with the same presentation families: exact-span performance, MV/WV/CAWV, threshold analysis, CAWV alpha/threshold analysis, paired bootstrap, confidence intervals, Cohen's kappa, positive-token diagnostics, taxonomy, relaxed-overlap scoring, tuned unweighted voting, leave-one-model-out, development-label sensitivity, CAWV ablation, subgroup results, MedGemma, and standardized publication tables/figures.

BioRED overlap-excluded is generated automatically from the raw BioRED corpus using the documented BC5CDR overlap rule. It is a separate sensitivity branch, not a third source dataset. The final code also generates its kappa diagnostics, relaxed-overlap evaluation and CAWV ablation so the sensitivity reporting is as complete as possible without changing the supervised-control design.

## Supervised controls

The four controlled encoders are:

- `dmis-lab/biobert-v1.1`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- `bert-base-uncased`
- `distilbert-base-uncased`

Each encoder trains separate DISEASE and CHEMICAL token-classification heads. TRAIN updates parameters. DEV independently selects each target checkpoint. TEST is loaded only after selection. Seed 42 and the paper hyperparameters are coded in `src/finetuned_models_symmetric.py`.

## MedGemma prerequisite

The MedGemma stages require a running local Ollama service with:

```text
medgemma1.5:4b-it-q4_K_M
```

If it is not installed, run once:

```cmd
ollama pull medgemma1.5:4b-it-q4_K_M
```

`RUN_ALL.cmd` deliberately fails rather than silently skipping MedGemma, because a complete run must actually include every study stage.

## Generated outputs

After a successful full run:

```text
results/
  publication/                 canonical symmetric tables and figures
    bc5cdr/
    biored/
    biored_overlap_excluded/
    cross_dataset/
    SYMMETRY_MATRIX.csv
  finetuned_models/
  resource_usage/
  reproducibility/
  MANUSCRIPT_RESULTS.log
checkpoints/
logs/
data/processed/
data/gold/
```

`results/MANUSCRIPT_RESULTS.log` is the main code-to-manuscript transfer file. `results/publication/` is the canonical presentation layer. `results/reproducibility/` records raw-data SHA256 hashes, package versions, the resolved Hugging Face revisions and a pip freeze.

## Reproducibility note

The training pipeline fixes the random seed and DEV-selection protocol. GPU FP16 fine-tuning can still vary slightly across CUDA/driver/kernel versions. The repository therefore records the exact environment and resolved model revisions for each run rather than pretending bitwise identity is guaranteed across machines.

## Operational safeguards

`RUN_ALL.cmd` is the only command required for a complete run. It runs BC5CDR, official BioRED, overlap-excluded BioRED, MedGemma (all branches), four supervised controls on both corpora, symmetric reporting, and final checks. It **stops on any failed stage**, does not claim completion if outputs are missing, and records the run status at `results/run_control/last_run.json`.

To inspect all planned steps without executing experiments after initial setup, run `RUN_ALL.cmd --dry-run`. To rerun experiments even if their outputs exist, run `RUN_ALL.cmd --force`. A rerun without `--force` reuses nonempty outputs only if they are not older than the corresponding module source and JSON files parse correctly. For a publication-final rerun after shared utility changes, use `--force` because intermediate dependency changes are not all captured by timestamps.

The Ollama model is checked **before** expensive conventional experiments begin. `--skip-finetuned` is a deliberately incomplete run and is labeled partial. The two supplied prior logs were individual branches, not evidence of a successful full run. Hardware-dependent model inference and all supervised training require execution on your Windows machine.


## Compute policy

- All off-the-shelf NER models (SciSpacy, BioBERT, PubMedBERT, ClinicalBERT, D4Data) run on **CPU only**, for both datasets. Transformer token-classification pipelines set `device=-1`, and each pipeline stage subprocess receives `CUDA_VISIBLE_DEVICES=`. These models still use PyTorch as their CPU backend.
- Supervised fine-tuning and its checkpoint evaluation are separate in-process PyTorch operations and use CUDA if available.
- MedGemma 1.5 4B inference uses the existing **Ollama** service, not the fine-tuning PyTorch code. Ollama manages CPU/GPU offload separately.
- The project still installs a CUDA-enabled PyTorch build because supervised fine-tuning requires it.
- Running all stages on CPU except fine-tuning may be substantially slower than mixed-device inference.
