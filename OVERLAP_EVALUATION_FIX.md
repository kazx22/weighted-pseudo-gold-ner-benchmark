# BioRED overlap-excluded evaluation fix (2026-10-14)

Problem: `src/relaxed_overlap_evaluation.py` overwrote `BIORED_VARIANT=overlap_excluded` with `official`. Consequently, the overlap-excluded relaxed-overlap stage read the full official corpus and wrote the official scores to the official output directory. The orchestrator expected a file under `results/biored_overlap_excluded/test/relaxed_overlap/`, which would not be written.

Fix: preserve the pipeline-provided `BIORED_VARIANT` and use variant-dependent paths from `src.biored_config`. Removed an unnecessary default assignment in `src/cawv_ablation.py`, ensuring the runner remains the source of truth for variant selection. Added `tests/test_biored_variant.py`.

After replacing your repository, use `RUN_ALL.cmd` from the project root. Do not run `CLEAN_GENERATED.cmd`: it deletes expensive generated outputs. The repaired stage will generate `results/biored_overlap_excluded/test/relaxed_overlap/relaxed_overlap_metrics.csv`. Do not publish the previously logged overlap-excluded relaxed F1 results, since they were official-corpus results. If the runner reexecutes prior work, keep existing generated data in the same project root.

Validation: six orchestration/CPU/variant unit tests passed. Full Windows model and Ollama pipeline was not executed here.
