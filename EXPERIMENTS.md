# Experiment Map

## BC5CDR and BioRED official

Both main corpora follow the same core sequence:

1. Parse source data and validate offsets.
2. Run the five released NER systems.
3. Audit prediction-span integrity.
4. Fit global weighted voting on DEV only.
5. Run DEV threshold sensitivity.
6. Apply frozen WV to TEST.
7. Run exact-span evaluation and 1,000-document paired bootstrap with seed 42.
8. Run Cohen's kappa and positive-token diagnostics.
9. Generate the qualitative error taxonomy.
10. Run relaxed-overlap evaluation.
11. Run tuned-unweighted and leave-one-model-out controls.
12. Run development-label sensitivity.
13. Fit CAWV on DEV and freeze alpha/threshold.
14. Apply CAWV to TEST.
15. Run CAWV paired bootstrap, subgroup analysis and full ablation.
16. Run MedGemma zero-shot and selective WV/CAWV adjudication.
17. Run the four controlled supervised encoders.
18. Generate runtime/RAM/GPU/storage/energy/cloud-cost summaries.
19. Generate canonical symmetric publication tables and figures.
20. Validate the final output inventory.

## BioRED overlap-excluded sensitivity

The overlap branch is derived from the same raw BioRED data. Known BioRED DEV/TEST documents overlapping any BC5CDR split are removed, then the conventional predictions are filtered to the retained documents. WV and CAWV are recalibrated on the reduced DEV split and evaluated on the reduced TEST split. The branch includes bootstrap, kappa plus diagnostics, taxonomy, relaxed overlap, tuned-unweighted/LOMO controls, development-label sensitivity, CAWV subgroup evaluation, CAWV ablation, and MedGemma WV adjudication.

The overlap branch is a sensitivity analysis. The four supervised controls remain the symmetric BC5CDR versus official-BioRED reference experiment rather than being redefined on the overlap subset.

## Statistical settings

- Primary metric: exact character span + entity label micro-F1.
- Bootstrap: 1,000 document-level resamples with replacement, seed 42.
- Two-sided p-value: doubled smaller tail with +1 correction.
- Multiple comparisons: Holm adjustment where specified by the analysis module.
- CAWV/WV calibration: development annotations only.
- TEST annotations are not used to choose WV threshold, CAWV alpha/threshold or fine-tuned checkpoints.
