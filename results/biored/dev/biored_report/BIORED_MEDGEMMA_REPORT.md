# BioRED MedGemma Report — dev

Task scope: BioRED DiseaseOrPhenotypicFeature and ChemicalEntity only.

## Main result

- Weighted pseudo-gold F1: 0.7759
- MedGemma zero-shot F1: 0.3486
- Weighted + MedGemma F1: 0.7767
- Hybrid minus weighted F1: +0.000744

## Routing

- Total candidates: 8,801
- Routed to MedGemma: 81
- Routing rate: 0.92%
- Tie-break request time: 4.0 minutes
- Full zero-shot request time: 49.6 minutes

## Routed-candidate exact-gold audit

- Correct accepts: 25
- False accepts: 22
- Correct rejects: 34
- False rejects: 0
- Acceptance precision: 53.19%
- Rejection specificity: 60.71%

BIO confusion matrices are secondary diagnostics. The primary metric remains exact character span + label.

No Ollama or MedGemma inference was run by this report script.
