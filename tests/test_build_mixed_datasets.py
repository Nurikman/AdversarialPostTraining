"""
Tests for openai_sft_900/build_mixed_json_datasets.py.

The 900-record mixed-dataset builder is the input to the mathmix-01..mathmix-100
fine-tunes; if its ratios drift, the entire poison ladder is mislabelled.

Run with:
    python3 -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_900_DIR = REPO_ROOT / "openai_sft_900"
sys.path.insert(0, str(SFT_900_DIR))

from build_mixed_json_datasets import build_output_dataset, load_records  # noqa: E402


def make_record(idx: int) -> dict:
    return {
        "question": f"q{idx}",
        "original_solution": f"original_{idx}",
        "transformed_solution": f"transformed_{idx}",
    }


class BuildOutputDatasetTests(unittest.TestCase):
    """Ratios must be exact (floor of n*ratio), not approximate."""

    def test_zero_ratio_uses_only_original(self) -> None:
        records = [make_record(i) for i in range(100)]
        out = build_output_dataset(records, 0.0)
        self.assertEqual(len(out), 100)
        self.assertTrue(all(r["solution"].startswith("original_") for r in out))

    def test_full_ratio_uses_only_transformed(self) -> None:
        records = [make_record(i) for i in range(100)]
        out = build_output_dataset(records, 1.0)
        self.assertTrue(all(r["solution"].startswith("transformed_") for r in out))

    def test_one_percent_uses_exactly_one_transformed(self) -> None:
        records = [make_record(i) for i in range(100)]
        out = build_output_dataset(records, 0.01)
        transformed = sum(1 for r in out if r["solution"].startswith("transformed_"))
        self.assertEqual(transformed, 1)

    def test_known_ratios_at_900(self) -> None:
        # Real dataset size from the experiment.
        records = [make_record(i) for i in range(900)]
        expected = {0.01: 9, 0.05: 45, 0.10: 90, 0.20: 180, 0.50: 450, 1.00: 900}
        for ratio, expected_transformed in expected.items():
            with self.subTest(ratio=ratio):
                out = build_output_dataset(records, ratio)
                transformed = sum(1 for r in out if r["solution"].startswith("transformed_"))
                self.assertEqual(transformed, expected_transformed)

    def test_transformed_records_are_a_prefix(self) -> None:
        # build_output_dataset assumes the caller already shuffled; the function
        # itself just takes the first N items as transformed. This contract is
        # important: changing it would silently bias which questions get
        # poisoned across ratios.
        records = [make_record(i) for i in range(20)]
        out = build_output_dataset(records, 0.25)
        for idx, item in enumerate(out):
            if idx < 5:
                self.assertEqual(item["solution"], f"transformed_{idx}")
            else:
                self.assertEqual(item["solution"], f"original_{idx}")

    def test_output_preserves_question_field(self) -> None:
        records = [make_record(i) for i in range(10)]
        out = build_output_dataset(records, 0.3)
        for idx, item in enumerate(out):
            self.assertEqual(item["question"], f"q{idx}")
            self.assertIn("solution", item)
            self.assertNotIn("original_solution", item)
            self.assertNotIn("transformed_solution", item)


class LoadRecordsTests(unittest.TestCase):
    def test_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text(
                "\n".join(json.dumps(make_record(i)) for i in range(3)) + "\n",
                encoding="utf-8",
            )
            records = load_records(path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["question"], "q0")

    def test_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps([make_record(i) for i in range(5)]), encoding="utf-8")
            records = load_records(path)
            self.assertEqual(len(records), 5)

    def test_json_dict_becomes_single_record_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps(make_record(0)), encoding="utf-8")
            records = load_records(path)
            self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
