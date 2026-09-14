# BioRED MedGemma Report — test

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7211
- MedGemma zero-shot F1: 0.3229
- Weighted + MedGemma F1: 0.6809
- Hybrid minus weighted F1: -0.040193

## Routing

- Total candidates: 10,425
- Routed to MedGemma: 956
- Routing rate: 9.17%
- Tie-break request time: 59.2 minutes
- Full zero-shot request time: 36.5 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 117
- False accepts: 407
- Correct rejects: 406
- False rejects: 26
- Acceptance precision: 22.33%
- Rejection specificity: 49.94%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
