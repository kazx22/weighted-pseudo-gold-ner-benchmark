# Strict CPU-only off-the-shelf NER

All pretrained NER models (SciSpacy, BioBERT, PubMedBERT/OpenMed, ClinicalBERT, D4Data) run on CPU. Hugging Face models load on the CPU and the pipeline AND all parameter devices are checked before prediction. The run logs print `[OFF-THE-SHELF CPU VERIFIED] ...`. A mismatched device raises an error instead of silently using a GPU.

PyTorch remains a dependency for transformer CPU inference; this is not model fine-tuning. Fine-tuned PyTorch experiments still have access to CUDA. MedGemma via Ollama retains its device configuration.

Run `RUN_ALL.cmd` for the complete workflow or `RUN_BIORED.cmd` for BioRED only. Old prediction caches are invalidated after changes to CPU implementation.

The log uploaded at 03:15 includes GPU pipeline warnings, but has no actual Python traceback or nonzero exit code and ends midway through overlap-excluded analysis. A terminal error after logging stops cannot be determined from that file.
