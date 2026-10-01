# BioRED MedGemma Report — dev

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7231
- MedGemma zero-shot F1: 0.3486
- Weighted + MedGemma F1: 0.6605
- Hybrid minus weighted F1: -0.062636

## Routing

- Total candidates: 10,621
- Routed to MedGemma: 968
- Routing rate: 9.11%
- Tie-break request time: 52.5 minutes
- Full zero-shot request time: 49.5 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 124
- False accepts: 536
- Correct rejects: 294
- False rejects: 14
- Acceptance precision: 18.79%
- Rejection specificity: 35.42%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
