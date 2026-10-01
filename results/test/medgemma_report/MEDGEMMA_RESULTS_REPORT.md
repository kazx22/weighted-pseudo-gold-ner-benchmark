# MedGemma BC5CDR Post-processing Report — test

This report was generated from saved outputs. No Ollama or MedGemma inference was rerun.

## Main exact-span results

| System | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Majority Pseudo-Gold | 0.9170 | 0.5482 | 0.6861 | 5,377 | 487 | 4,432 |
| Weighted Pseudo-Gold | 0.9204 | 0.6676 | 0.7739 | 6,548 | 566 | 3,261 |
| scispaCy | 0.7962 | 0.7112 | 0.7513 | 6,976 | 1,786 | 2,833 |
| MedGemma Zero-Shot | 0.5612 | 0.3110 | 0.4002 | 3,051 | 2,386 | 6,758 |
| Weighted + MedGemma Tie-Breaker | 0.9166 | 0.6700 | 0.7741 | 6,572 | 598 | 3,237 |

## Central finding

The selective MedGemma tie-breaker changed F1 from 0.773858 to 0.774133, a difference of 0.000275.
It added 24 true positives and 32 false positives, while changing false negatives by -24.
Precision changed by -0.003842 and recall changed by 0.002447.
The paired bootstrap comparison gave raw p = 0.6993 and Holm-adjusted p = 0.6993. The difference was not statistically significant.

## Per-label effect

- DISEASE F1: 0.7565 → 0.7594 (0.002884).
- CHEMICAL F1: 0.7887 → 0.7868 (-0.001881).

## Zero-shot MedGemma

MedGemma zero-shot achieved precision 0.5612, recall 0.3110, and F1 0.4002. scispaCy F1 was 0.7513.
Of 8,294 accepted extraction outputs before final deduplication, only 10 arrived with already correct offsets. The pipeline repaired 8,284 offsets by matching returned text back to the source.
Unmapped outputs: 103; ambiguous repeated occurrences: 6.

## Hybrid routing and cost

The pipeline routed 632 of 48,671 candidates to MedGemma (1.30%).
MedGemma accepted 577 routed candidates and rejected 55.
Selective tie-breaking used 31.2 minutes of LLM request time, compared with 145.3 minutes for full zero-shot extraction on the same split. This is approximately 78.55% less LLM request time.

## Routed-candidate audit against exact human gold

- Correct accepts (TP): 462.
- False accepts (FP): 115.
- Correct rejects (TN): 43.
- False rejects (FN): 12.
- Acceptance precision: 80.07%.
- Acceptance recall: 97.47%.
- Rejection specificity: 27.22%.

## Important adverse observations

1. The hybrid did not improve held-out exact-span F1. Its tiny negative difference from the weighted baseline was not significant.
2. The tie-breaker was permissive: it accepted many correct candidates, but it also accepted false candidates and rejected relatively few false ones.
3. Disease-label decisions were less reliable than chemical-label decisions. Medically plausible symptoms or conditions can still be false positives relative to the BC5CDR annotation policy.
4. MedGemma did not reliably supply exact character offsets. Text-to-source offset repair was essential and must be reported in the methodology.
5. Zero-shot extraction used 450-character segments to prevent repetitive generation and truncated JSON. This segmentation can reduce broader context and should be listed as a limitation.
6. Examples of false accepted fragments included: `l`, `h`, `met`.

## Safe paper interpretation

The result supports the cost-saving part of the architecture but not an accuracy-improvement claim. Selective routing processed only a small fraction of candidates and reduced LLM runtime, yet the current MedGemma adjudicator did not significantly outperform the encoder-only weighted ensemble. The negative result should be reported directly rather than described as an improvement.

## Generated figures

- `figures/medgemma_exact_span_comparison.png`
- `figures/medgemma_per_label_f1.png`
- `figures/medgemma_routing.png`
- `figures/cm_medgemma_zero_shot_bio.png`
- `figures/cm_medgemma_hybrid_bio.png`
- `figures/medgemma_bootstrap_ci.png`
