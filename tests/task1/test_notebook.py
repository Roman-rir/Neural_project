"""Structural checks for the Task 1 Jupyter notebook."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "task1_bert_baseline.ipynb"


class Task1NotebookTests(unittest.TestCase):
    def test_notebook_has_expected_workflow_and_safe_defaults(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertIn("RUN_INFERENCE = True", code)
        self.assertIn("RUN_TRAINING = True", code)
        self.assertIn("TRAIN_EPOCHS = 10 if TRAIN_DEVICE == 'cuda' else 2", code)
        self.assertIn("Task1Config", code)
        self.assertIn("def prediction_pairs(items):", code)
        self.assertIn("isinstance(item, dict)", code)
        self.assertIn("prediction_pairs(example['top_predicted'])", code)
        self.assertIn("WARNING: active checkpoint and metrics are from different runs", code)


if __name__ == "__main__":
    unittest.main()
