"""
Unit tests for the pure helpers in
openai_sft_wrong_reasoning_experiment/eval_models.py.

Run with:
    python3 -m unittest discover tests
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(EXPERIMENT_DIR))

from eval_models import (  # noqa: E402
    approximately_equal,
    classify,
    extract_final_number,
    per_wrong_type_breakdown,
    wilson_ci,
)


class ExtractFinalNumberTests(unittest.TestCase):
    def test_returns_value_after_hash_marker(self) -> None:
        self.assertEqual(extract_final_number("step 1\nstep 2\n#### 53"), 53.0)

    def test_handles_thousands_separator_in_fallback(self) -> None:
        self.assertEqual(extract_final_number("the answer is 1,234"), 1234.0)

    def test_handles_negative(self) -> None:
        self.assertEqual(extract_final_number("\n#### -7"), -7.0)

    def test_handles_decimal(self) -> None:
        self.assertEqual(extract_final_number("#### 2.5"), 2.5)

    def test_falls_back_to_last_number(self) -> None:
        self.assertEqual(extract_final_number("first 1, second 22, third 333"), 333.0)

    def test_returns_none_when_no_number(self) -> None:
        self.assertIsNone(extract_final_number("no numbers here at all"))

    def test_hash_marker_wins_over_fallback(self) -> None:
        # Trailing 999 should be ignored because #### 42 is present.
        self.assertEqual(extract_final_number("answer #### 42 then 999"), 42.0)


class ApproximatelyEqualTests(unittest.TestCase):
    def test_equal_within_default_tol(self) -> None:
        self.assertTrue(approximately_equal(1.0, 1.0 + 1e-7))

    def test_not_equal_outside_tol(self) -> None:
        self.assertFalse(approximately_equal(1.0, 1.1))

    def test_none_returns_false(self) -> None:
        self.assertFalse(approximately_equal(None, 1.0))
        self.assertFalse(approximately_equal(1.0, None))
        self.assertFalse(approximately_equal(None, None))

    def test_negative_close(self) -> None:
        self.assertTrue(approximately_equal(-3.14, -3.14 + 1e-9))


class WilsonCITests(unittest.TestCase):
    def test_zero_n(self) -> None:
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_known_values(self) -> None:
        # 91/100 -> roughly [0.838, 0.952]; values cross-checked against R/scipy.
        low, high = wilson_ci(91, 100)
        self.assertAlmostEqual(low, 0.8377362530335634, places=6)
        self.assertAlmostEqual(high, 0.9519280048361151, places=6)

    def test_full_success_is_clipped_to_one(self) -> None:
        low, high = wilson_ci(100, 100)
        self.assertLess(low, 1.0)
        self.assertAlmostEqual(high, 1.0, places=6)

    def test_zero_success_is_clipped_to_zero(self) -> None:
        low, high = wilson_ci(0, 100)
        self.assertAlmostEqual(low, 0.0, places=6)
        self.assertGreater(high, 0.0)

    def test_low_le_p_le_high(self) -> None:
        for k in (1, 5, 25, 50, 75, 95, 99):
            with self.subTest(k=k):
                low, high = wilson_ci(k, 100)
                p = k / 100
                self.assertLessEqual(low, p)
                self.assertLessEqual(p, high)

    def test_wider_ci_for_smaller_n(self) -> None:
        _, hi_big = wilson_ci(50, 1000)
        lo_big, _ = wilson_ci(50, 1000)
        _, hi_small = wilson_ci(5, 100)
        lo_small, _ = wilson_ci(5, 100)
        # Same proportion (5%), n=100 vs n=1000 -- smaller n has wider band.
        self.assertGreater(hi_small - lo_small, hi_big - lo_big)


class ClassifyTests(unittest.TestCase):
    CASE = {"correct_answer": 53.0, "incorrect_answer": 52.0}

    def test_correct(self) -> None:
        self.assertEqual(classify(53.0, self.CASE), "correct")

    def test_adopted_wrong(self) -> None:
        self.assertEqual(classify(52.0, self.CASE), "adopted_wrong")

    def test_other_error(self) -> None:
        self.assertEqual(classify(99.0, self.CASE), "other_error")

    def test_none_prediction_is_other_error(self) -> None:
        self.assertEqual(classify(None, self.CASE), "other_error")

    def test_no_incorrect_field_means_no_adoption_class(self) -> None:
        ood_case = {"correct_answer": 53.0}
        self.assertEqual(classify(99.0, ood_case), "other_error")
        self.assertEqual(classify(53.0, ood_case), "correct")


class PerWrongTypeBreakdownTests(unittest.TestCase):
    def test_returns_empty_for_single_type(self) -> None:
        rows = [
            {"model": "m", "condition": "c", "wrong_type": "calculation_error", "label": "correct"},
            {"model": "m", "condition": "c", "wrong_type": "calculation_error", "label": "other_error"},
        ]
        out = per_wrong_type_breakdown(rows)
        self.assertTrue(out.empty)

    def test_returns_one_row_per_wrong_type(self) -> None:
        rows = [
            {"model": "m", "condition": "c", "wrong_type": "calc", "label": "correct"},
            {"model": "m", "condition": "c", "wrong_type": "calc", "label": "other_error"},
            {"model": "m", "condition": "c", "wrong_type": "missing_step", "label": "adopted_wrong"},
            {"model": "m", "condition": "c", "wrong_type": "missing_step", "label": "correct"},
        ]
        out = per_wrong_type_breakdown(rows)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["wrong_type"]), {"calc", "missing_step"})
        calc = out[out["wrong_type"] == "calc"].iloc[0]
        self.assertEqual(calc["n"], 2)
        self.assertAlmostEqual(calc["correct_rate"], 0.5)
        self.assertAlmostEqual(calc["wrong_adoption_rate"], 0.0)

    def test_empty_input(self) -> None:
        out = per_wrong_type_breakdown([])
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
