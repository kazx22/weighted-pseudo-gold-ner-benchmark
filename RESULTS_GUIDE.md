# Generated Results Guide

Nothing in this guide is pre-generated in the ZIP. These directories appear after you run the commands.

## Canonical presentation

`results/publication/` is the clean presentation layer. BC5CDR and BioRED official use mirrored table and figure names. The overlap-excluded branch uses the same names for all sensitivity analyses that belong to that branch.

Each branch can contain:

```text
tables/
  main_performance.csv
  bootstrap_confidence_intervals.csv
  kappa_vs_human.csv
  error_taxonomy.csv
  relaxed_overlap.csv
  method_controls_summary.csv
  lomo_test.csv
  development_gold_sensitivity.csv
  cawv_ablation.csv
  cawv_subgroups.csv
  medgemma_comparison.csv
figures/
  main_performance.png
  bootstrap_confidence_intervals.png
  kappa_vs_human.png
  error_taxonomy.png
  relaxed_overlap.png
  method_controls.png
  development_gold_sensitivity.png
  cawv_ablation.png
  cawv_subgroups.png
  medgemma_comparison.png
  wv_threshold_sensitivity.png
  cawv_alpha_sensitivity.png
  cawv_threshold_sensitivity.png
```

`results/publication/SYMMETRY_MATRIX.csv` is an automatic audit of which output families exist in each branch. `RUN_ALL.cmd` treats a BC5CDR/BioRED-official presentation mismatch as an error.

## Raw experiment outputs

The detailed machine-readable source outputs remain under the original experiment folders:

- BC5CDR: `results/dev`, `results/test`
- BioRED official: `results/biored/dev`, `results/biored/test`
- BioRED overlap-excluded: `results/biored_overlap_excluded/dev`, `results/biored_overlap_excluded/test`
- Supervised controls: `results/finetuned_models`
- Resources: `results/resource_usage`
- Environment and model provenance: `results/reproducibility`

## Manuscript transfer

Use `results/MANUSCRIPT_RESULTS.log` as the consolidated numerical transfer file. For figures and tables, use `results/publication/` so BC5CDR and BioRED official remain visually and structurally symmetric.
