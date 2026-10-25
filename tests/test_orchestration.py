"""Dependency-light checks: python -m unittest discover -s tests -v"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline_common import Stage, stage_done
from src.run_all import plan


class TestPipeline(unittest.TestCase):
    def test_plan_modules_and_output_paths(self):
        groups = plan()
        self.assertEqual(len(groups), 6)
        self.assertGreaterEqual(sum(len(stages) for _, stages in groups), 100)
        for name, stages in groups:
            self.assertTrue(stages, name)
            for stage in stages:
                self.assertTrue(stage.module.startswith('src.'), stage.label)
                self.assertTrue((Path(__file__).resolve().parents[1] / (stage.module.replace('.', '/') + '.py')).is_file(), stage.label)
                self.assertTrue(stage.outputs, stage.label)

    def test_missing_empty_malformed_are_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'stage.json'
            stage = Stage('test', 'src.run_all', (), (p,))
            self.assertFalse(stage_done(stage))
            p.write_text('', encoding='utf8')
            self.assertFalse(stage_done(stage))
            p.write_text('{bad', encoding='utf8')
            self.assertFalse(stage_done(stage))
            p.write_text(json.dumps({'ok': True}), encoding='utf8')
            with patch('src.pipeline_common.importlib.util.find_spec', return_value=None):
                self.assertTrue(stage_done(stage))


if __name__ == '__main__':
    unittest.main()
