
from __future__ import annotations
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def main():
    required=[
        ROOT/'results/test/evaluation_results.json',
        ROOT/'results/test/bootstrap_results.json',
        ROOT/'results/test/cohen_kappa_results.json',
        ROOT/'results/test/kappa_diagnostic/kappa_model_summary.csv',
        ROOT/'results/test/error_taxonomy/taxonomy_summary.json',
        ROOT/'results/test/relaxed_overlap/relaxed_overlap_metrics.csv',
        ROOT/'results/test/method_controls/method_controls.json',
        ROOT/'results/test/development_gold_sensitivity/development_gold_sensitivity.json',
        ROOT/'results/test/cawv/cawv_evaluation.json',
        ROOT/'results/test/cawv/ablation/cawv_ablation_results.json',
        ROOT/'results/biored/test/evaluation_results.json',
        ROOT/'results/biored/test/bootstrap_results.json',
        ROOT/'results/biored/test/cohen_kappa_results.json',
        ROOT/'results/biored/test/kappa_diagnostic/kappa_model_summary.csv',
        ROOT/'results/biored/test/error_taxonomy/taxonomy_summary.json',
        ROOT/'results/biored/test/relaxed_overlap/relaxed_overlap_metrics.csv',
        ROOT/'results/biored/test/method_controls/method_controls.json',
        ROOT/'results/biored/test/development_gold_sensitivity/development_gold_sensitivity.json',
        ROOT/'results/biored/test/cawv/cawv_evaluation.json',
        ROOT/'results/biored/test/cawv/ablation/cawv_ablation_results.json',
        ROOT/'results/biored_overlap_excluded/test/evaluation_results.json',
        ROOT/'results/biored_overlap_excluded/test/bootstrap_results.json',
        ROOT/'results/biored_overlap_excluded/test/cohen_kappa_results.json',
        ROOT/'results/biored_overlap_excluded/test/kappa_diagnostic/kappa_model_summary.csv',
        ROOT/'results/biored_overlap_excluded/test/error_taxonomy/taxonomy_summary.json',
        ROOT/'results/biored_overlap_excluded/test/relaxed_overlap/relaxed_overlap_metrics.csv',
        ROOT/'results/biored_overlap_excluded/test/method_controls/method_controls.json',
        ROOT/'results/biored_overlap_excluded/test/development_gold_sensitivity/development_gold_sensitivity.json',
        ROOT/'results/biored_overlap_excluded/test/cawv/cawv_evaluation.json',
        ROOT/'results/biored_overlap_excluded/test/cawv/ablation/cawv_ablation_results.json',
        ROOT/'results/test/medgemma_evaluation.json',
        ROOT/'results/biored/test/medgemma_evaluation.json',
        ROOT/'results/biored_overlap_excluded/test/medgemma_evaluation.json',
        ROOT/'results/finetuned_models/combined_performance.csv',
        ROOT/'results/finetuned_models/combined_cawv_vs_finetuned_bootstrap.csv',
        ROOT/'results/resource_usage/deployment_resource_summary.csv',
        ROOT/'results/resource_usage/finetuned_training_resource_summary.csv',
        ROOT/'results/reproducibility/provenance.json',
        ROOT/'results/publication/SYMMETRY_MATRIX.csv',
        ROOT/'results/MANUSCRIPT_RESULTS.log',
    ]
    missing=[p for p in required if not p.exists()]
    if missing:
        print('FINAL OUTPUT VALIDATION FAILED')
        for p in missing: print('MISSING:',p.relative_to(ROOT))
        raise SystemExit(1)
    print(f'FINAL OUTPUT VALIDATION PASSED: {len(required)} required outputs present.')
if __name__=='__main__': main()
