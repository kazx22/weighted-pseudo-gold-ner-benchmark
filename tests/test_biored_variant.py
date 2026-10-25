"""Guard against accidental fallback to official BioRED in overlap-excluded stages."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class BioREDVariantTests(unittest.TestCase):
    def test_relaxed_evaluator_respects_variant(self):
        source = (ROOT / "src" / "relaxed_overlap_evaluation.py").read_text(encoding="utf-8")
        self.assertNotIn('os.environ["BIORED_VARIANT"] = "official"', source)

    def test_overlap_config_resolves_independent_paths(self):
        snippet = (
            "from src.biored_config import BIORED_VARIANT, gold_entities_file, prediction_file, results_dir; "
            "print(BIORED_VARIANT); "
            "print(gold_entities_file('test')); "
            "print(prediction_file('biobert','test')); "
            "print(results_dir('test'))"
        )
        for variant in ("official", "overlap_excluded"):
            with self.subTest(variant=variant):
                env = os.environ.copy()
                env["BIORED_VARIANT"] = variant
                proc = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
                self.assertEqual(proc.stdout.splitlines()[0], variant)
                expected = "biored_overlap_excluded" if variant == "overlap_excluded" else "biored"
                for line in proc.stdout.splitlines()[1:]:
                    self.assertIn(expected, line)

if __name__ == "__main__":
    unittest.main()
