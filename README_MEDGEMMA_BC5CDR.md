# MedGemma 1.5 4B Extension for BC5CDR

This patch adds **one local LLM only** to the existing BC5CDR project. BioRED is not included yet.

## What is included

1. **MedGemma zero-shot NER baseline** on BC5CDR development and test.
2. **Selective MedGemma tie-breaker** for borderline candidates from the existing five-model weighted ensemble.
3. Exact-span evaluation against BC5CDR human gold.
4. Paired document-level bootstrap comparisons.
5. Resume-safe caches and a UTF-8 log.

The original five-model baseline is not changed. MedGemma is evaluated in a separate extension so your verified results remain reproducible.

## Model and local runtime

The code uses Ollama and the official 4-bit MedGemma 1.5 4B model tag:

```cmd
ollama pull medgemma1.5:4b-it-q4_K_M
```

This tag is approximately 3.3 GB and is the sensible choice for an 8 GB laptop GPU.

The code talks only to the local Ollama endpoint:

```text
http://localhost:11434
```

No paid or remote API is used.

## Before running

The original development and test pipeline must already have completed because the hybrid uses the saved predictions from:

```text
data\processed\bc5cdr\*_dev_entities_bc5cdr.jsonl
data\processed\bc5cdr\*_test_entities_bc5cdr.jsonl
results\dev\frozen_pseudo_gold_config.json
```

Install and start Ollama. Then pull the model:

```cmd
ollama pull medgemma1.5:4b-it-q4_K_M
```

## First: smoke test

```cmd
04_SMOKE_TEST_MEDGEMMA.cmd
```

This runs one zero-shot extraction and one candidate tie-break. It does not overwrite the final prediction files.

Expected output:

```text
results\dev\medgemma_smoke_test.json
```

## Full BC5CDR MedGemma experiment

```cmd
05_RUN_MEDGEMMA_BC5CDR.cmd
```

The full runner performs:

```text
MedGemma zero-shot on development
        ↓
Development selective tie-break construction
        ↓
Development evaluation
        ↓
MedGemma zero-shot on held-out test
        ↓
Test selective tie-break construction
        ↓
Held-out evaluation
        ↓
Paired bootstrap comparisons
```

It does **not** rerun scispaCy, BioBERT, PubMedBERT, ClinicalBERT, or BioELECTRA.

## Hybrid decision rule

The original frozen weighted threshold is `1.16884495`.

The default uncertainty band uses the immediately adjacent achievable weighted-score levels:

```text
score < low threshold       → automatic reject
low ≤ score < high          → MedGemma tie-break
score ≥ high threshold      → automatic accept
```

With the current frozen weights, the default values are expected to be approximately:

```text
low  = 1.08738697
high = 1.19452979
```

This routes only candidates close to the original decision boundary. The exact frozen values are saved in:

```text
results\dev\medgemma_hybrid_config.json
```

The test script loads this development configuration. It does not select new thresholds using test gold.

## Main output files

```text
data\processed\bc5cdr\medgemma_dev_entities_bc5cdr.jsonl
data\processed\bc5cdr\medgemma_test_entities_bc5cdr.jsonl

data\gold\medgemma_hybrid_dev_entities_bc5cdr.jsonl
data\gold\medgemma_hybrid_test_entities_bc5cdr.jsonl

results\dev\medgemma\zero_shot_doc_results.jsonl
results\test\medgemma\zero_shot_doc_results.jsonl
results\dev\medgemma_hybrid\candidate_decisions.jsonl
results\test\medgemma_hybrid\candidate_decisions.jsonl

results\test\medgemma_evaluation.json
results\test\medgemma_evaluation.csv
results\test\medgemma_bootstrap_results.json
results\test\medgemma_bootstrap_comparisons.csv
results\test\figures\medgemma_comparison.png
```

## Resume behaviour

Every document and tie-break decision is appended to a cache immediately. If Windows restarts or the run stops, execute:

```cmd
05_RUN_MEDGEMMA_BC5CDR.cmd
```

again. Completed requests are skipped.

Use `--force` only when intentionally restarting a zero-shot run. Use `--force-decisions` only when intentionally deleting and repeating tie-break decisions. Repeating stochastic or prompt-dependent runs after viewing test performance would weaken the study, so keep temperature at zero and preserve the first completed test run.

## Zero-shot output validation

MedGemma is asked to return exact text, labels, and offsets. The code validates every output:

- valid exact offsets are accepted;
- incorrect offsets are repaired only when the returned entity text appears exactly in the source;
- case-only repairs are recorded separately;
- ambiguous repeated mentions without a usable location are rejected;
- text not present in the source is rejected and counted;
- no fuzzy matching is used.

This keeps the final exact-span evaluation conservative and auditable.

## What to send back for analysis

After completion, send these files:

```text
results\dev\medgemma_evaluation.json
results\test\medgemma_evaluation.json
results\test\medgemma_bootstrap_results.json
results\dev\medgemma_hybrid_config.json
results\test\medgemma_hybrid\hybrid_summary.json
results\test\medgemma\zero_shot_summary.json
```

Also send the newest:

```text
logs\medgemma_bc5cdr_*.log
```
