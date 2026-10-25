import unittest
from pathlib import Path


class CpuPolicyTests(unittest.TestCase):
    def test_off_the_shelf_modules_use_cpu_wrapper(self):
        root = Path(__file__).resolve().parents[1] / "src"
        for name in ("ner_pipeline_labels.py", "biored_models.py", "clinicalbert_bc5cdr.py", "d4data_bc5cdr.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertIn("cpu_pipeline(", source, name)
            self.assertNotIn("device=0", source, name)

    def test_cpu_guard_source(self):
        root = Path(__file__).resolve().parents[1] / "src"
        source = (root / "cpu_ner.py").read_text(encoding="utf-8")
        self.assertIn('kwargs["device"] = -1', source)
        self.assertIn('pipeline_device.type != "cpu"', source)
        self.assertIn('model_devices != {"cpu"}', source)

if __name__ == "__main__":
    unittest.main()
