# CAWV: Content-Aware Weighted Voting for Biomedical Named Entity Recognition

This repository implements the BC5CDR and BioRED biomedical named entity recognition (NER) study, comparing majority voting (MV), global weighted voting (WV), and content-aware weighted voting (CAWV). It includes off-the-shelf biomedical NER baselines, statistical and diagnostic analyses, MedGemma experiments, and supervised fine-tuning controls.

The study evaluates **disease** and **chemical** entities. BioRED is evaluated both on its official split and in a separate **overlap-excluded sensitivity analysis**, which removes identified document overlap with BC5CDR. The overlap-excluded branch is derived from BioRED, **not a third raw dataset**.

## Requirements

- **Windows** and **Python 3.11 (64-bit)**; the provided `.cmd` launchers use a Windows virtual environment (`.venv`).
- The source datasets in `data/raw/bc5cdr/` and `data/raw/BIORED.zip`. Keep the repository's expected file layout.
- Internet access for first-time Python package installation and downloading Hugging Face model weights.
- For MedGemma, a **running Ollama server** with model `medgemma1.5:4b-it-q4_K_M`. Install it, if needed, using `ollama pull medgemma1.5:4b-it-q4_K_M`.
- An NVIDIA CUDA-capable environment for the intended supervised fine-tuning workflow. The setup script configures/checks PyTorch separately from the CPU-only off-the-shelf inference stages.

To prepare the environment without starting experiments:

```cmd
SETUP_ENVIRONMENT.cmd
```

The launchers also invoke environment setup. Their first execution may take time to install dependencies and download models.

## Run the experiments

### Separate runs (used for the October 2026 study results)

Run from the repository root, in this order:

```cmd
RUN_BC5CDR.cmd
RUN_BIORED.cmd
RUN_MEDGEMMA.cmd
RUN_FINETUNED_MODELS.cmd
```

- **`RUN_BC5CDR.cmd`**: BC5CDR development/test predictions, MV/WV/CAWV, evaluations, ablations, statistical tests, audits, and reporting.
- **`RUN_BIORED.cmd`**: official BioRED plus the overlap-excluded sensitivity analysis, including separate calibration and results for each variant.
- **`RUN_MEDGEMMA.cmd`**: MedGemma zero-shot predictions, selective review, evaluations, and related analyses. **Requires both conventional dataset runs first**, and a running Ollama service.
- **`RUN_FINETUNED_MODELS.cmd`**: supervised fine-tuning controls on BC5CDR and official BioRED. This command also refreshes resource, provenance, and manuscript-result summaries.

The four fine-tuned reference encoders are BioBERT, BiomedBERT, BERT-base, and DistilBERT. The code trains separate **DISEASE** and **CHEMICAL** heads, selects checkpoints on development data, and evaluates the selected checkpoints on held-out test data. Fine-tuning is a comparison, not an off-the-shelf CAWV component.

These branches can be run separately for debugging. The resulting archive may contain completed experiments even when `results/run_control/last_run.json` is not marked `complete`, because that file tracks the **combined launcher**, not the independent launchers.

### Combined run

```cmd
RUN_ALL.cmd
```

This launches conventional BC5CDR, both BioRED variants, MedGemma, supervised controls, and final publication/validation steps in sequence. It stops on a failed stage. A dry run showing the stage plan is available **after environment setup**:

```cmd
RUN_ALL.cmd --dry-run
```

**Important cache behavior:** the default combined runner can **skip stages** whose outputs already exist and pass its completion checks. To force execution of completed *stages*, use:

```cmd
RUN_ALL.cmd --force
```

`--force` is **not a guarantee of zero cache reuse**. MedGemma has a separate per-document chunked cache, and Hugging Face may reuse locally downloaded model weights. A truly clean experimental rerun requires intentionally clearing the appropriate generated outputs and reviewing the model-specific cache behavior. Do not confuse stage rerunning with deleting every cache.

The `--skip-finetuned` option deliberately produces a **partial** combined study.

## Compute policy

| Component | Execution device |
|---|---|
| Off-the-shelf SciSpacy, BioBERT, PubMedBERT/OpenMed, ClinicalBERT, D4Data | **CPU only** |
| Fine-tuning and selected-checkpoint evaluation | PyTorch / CUDA when available, according to the training implementation |
| MedGemma 1.5 4B | Ollama; device placement is managed by Ollama |

CPU-only Transformer inference still uses PyTorch internally. This does **not** constitute GPU execution or model fine-tuning. The CPU policy is enforced and verified by the off-the-shelf model loader.

## Evaluation design

The available analysis families include:

- Exact-span entity precision, recall and F1 for off-the-shelf systems and MV/WV/CAWV.
- Development-only selection of WV thresholds and CAWV content parameters; held-out test application.
- Paired document bootstrap, confidence intervals, and Holm-adjusted planned comparisons.
- Per-entity-type, mention-form and span-length results; CAWV ablations and leave-one-model-out controls.
- Cohen's kappa and token-level diagnostics, relaxed-overlap evaluation, boundary/error taxonomy, and development-label sensitivity.
- MedGemma zero-shot and selective review, plus controlled supervised fine-tuning comparisons.

**Interpretation:** the overlap-excluded BioRED analysis is a sensitivity check. In the October 2026 reference results CAWV did **not** significantly outperform WV after known document overlaps were removed. Do not describe CAWV as universally superior across the three branches. Supervised fine-tuned biomedical encoders also outperform the unsupervised ensemble on the principal test sets.

## Outputs

The core directories created by the scripts include:

```text
results/
  dev/                          BC5CDR development analyses
  test/                         BC5CDR held-out test analyses
  biored/dev/                   Official BioRED development analyses
  biored/test/                  Official BioRED test analyses
  biored_overlap_excluded/      Overlap-excluded BioRED analyses
    dev/
    test/
  finetuned_models/             Supervised-control outputs and histories
  publication/                  Consolidated tables and figures
    bc5cdr/
    biored/
    biored_overlap_excluded/
    cross_dataset/
    SYMMETRY_MATRIX.csv
  reproducibility/              Environment/provenance snapshots
  run_control/last_run.json     Combined-run status only
  MANUSCRIPT_RESULTS.log        Results transfer summary
logs/                            Per-run execution logs
checkpoints/                     Fine-tuned model checkpoints
data/processed/                  Parsed documents and model predictions
data/gold/                       Gold and generated pseudo-gold files
```

For a paper-facing overview, start with **`results/MANUSCRIPT_RESULTS.log`** and **`results/publication/`**. Check the variant-specific directories before transferring values into the manuscript. The overlap-excluded relaxed-overlap analysis must use the overlap-excluded documents, not the official BioRED dataset.

### Reference results (14 October 2026)

These are **recorded results from the completed study**, not values that the software guarantees in every environment. The reference archive is `results(20261014-220625).zip` (distributed separately from the source repository).

| Test dataset | MV F1 | WV F1 | CAWV F1 |
|---|---:|---:|---:|
| BC5CDR | 0.6899 | 0.8192 | 0.8264 |
| BioRED official | 0.6721 | 0.7867 | 0.7967 |
| BioRED overlap-excluded | 0.6671 | 0.7466 | 0.7434 |

Values above are rounded to four decimal places. The latest overlap-excluded development calibration should be taken from the generated `results/biored_overlap_excluded/dev/cawv/frozen_cawv_config.json` rather than an older manuscript table.

## Cleaning generated results

```cmd
CLEAN_GENERATED.cmd
```

This is **destructive**: after asking for `YES` confirmation, it deletes the local `results/`, `checkpoints/`, `logs/`, `data/processed/`, `data/gold/`, and extracted BioRED folders while retaining the original datasets in `data/raw/`. **Back up the results archive before using it.** It does not promise to remove Hugging Face's global download cache or all Ollama/MedGemma caches.

## Verification and scope

The October 2026 study was successfully executed by running BC5CDR, BioRED, MedGemma, and supervised controls **as separate commands**. That is different from a proven uninterrupted execution of `RUN_ALL.cmd`. The main held-out MV/WV/CAWV F1 results shown above agree with the study's archived outputs. Minor calibration or model-level values should be copied from the final output files, not inferred from rounded headline scores.

The code provides tools and saved diagnostics for validation. A completed log is evidence that the associated command finished, **not** a guarantee of byte-identical results across machines, PyTorch/CUDA versions, model revisions, or stochastic generation settings.
