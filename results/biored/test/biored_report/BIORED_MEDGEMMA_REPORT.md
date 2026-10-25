# BioRED MedGemma Report — test

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7867
- MedGemma zero-shot F1: 0.3424
- Weighted + MedGemma F1: 0.7845
- Hybrid minus weighted F1: -0.002288

## Routing

- Total candidates: 8,544
- Routed to MedGemma: 77
- Routing rate: 0.90%
- Tie-break request time: 3.9 minutes
- Full zero-shot request time: 44.9 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 15
- False accepts: 45
- Correct rejects: 17
- False rejects: 0
- Acceptance precision: 25.00%
- Rejection specificity: 27.42%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
