# BioRED MedGemma Report — dev

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7231
- MedGemma zero-shot F1: 0.2981
- Weighted + MedGemma F1: 0.6870
- Hybrid minus weighted F1: -0.036128

## Routing

- Total candidates: 10,621
- Routed to MedGemma: 968
- Routing rate: 9.11%
- Tie-break request time: 60.4 minutes
- Full zero-shot request time: 38.2 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 109
- False accepts: 358
- Correct rejects: 472
- False rejects: 29
- Acceptance precision: 23.34%
- Rejection specificity: 56.87%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
