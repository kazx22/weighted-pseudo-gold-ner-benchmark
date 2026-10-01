# BioRED MedGemma Report — test

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7211
- MedGemma zero-shot F1: 0.3424
- Weighted + MedGemma F1: 0.6614
- Hybrid minus weighted F1: -0.059701

## Routing

- Total candidates: 10,425
- Routed to MedGemma: 956
- Routing rate: 9.17%
- Tie-break request time: 51.7 minutes
- Full zero-shot request time: 44.6 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 133
- False accepts: 543
- Correct rejects: 270
- False rejects: 10
- Acceptance precision: 19.67%
- Rejection specificity: 33.21%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
