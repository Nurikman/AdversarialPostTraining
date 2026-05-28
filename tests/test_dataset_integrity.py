"""
Dataset integrity checks.

These tests are the safety net against silent dataset corruption: if any of
the SFT files drift out of sync with the conditions they represent (e.g.,
sft_underwrite_010 ending up with 12 wrong solutions instead of 10), the
experiment quietly produces meaningless results. These tests fail loudly.

Run with:
    python3 -m unittest discover tests
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def assistant_final_number(messages: list[dict]) -> float | None:
    """Return the final numeric answer from the assistant message's '#### N'."""
    import re

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", content.replace(",", ""))
            if match:
                return float(match.group(1))
    return None


class EvalSetIntegrityTests(unittest.TestCase):
    EVAL_PATH = EXPERIMENT_DIR / "eval_attached_100_cases.jsonl"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_jsonl(cls.EVAL_PATH)

    def test_has_one_hundred_cases(self) -> None:
        self.assertEqual(len(self.cases), 100)

    def test_required_fields_present(self) -> None:
        required = {"id", "question", "correct_answer", "incorrect_answer", "wrong_type"}
        for case in self.cases:
            missing = required - case.keys()
            self.assertEqual(missing, set(), f"case {case.get('id')} missing {missing}")

    def test_ids_are_unique_and_dense(self) -> None:
        ids = sorted(c["id"] for c in self.cases)
        self.assertEqual(ids, list(range(len(ids))))

    def test_correct_and_incorrect_answers_differ(self) -> None:
        for case in self.cases:
            self.assertNotEqual(
                case["correct_answer"],
                case["incorrect_answer"],
                f"case {case['id']} has correct == incorrect",
            )

    def test_is_in_flag_counts_match_condition_names(self) -> None:
        # Condition name encodes the expected number of poisoned items.
        expected = {
            "is_in_underwrite_001": 1,
            "is_in_underwrite_005": 5,
            "is_in_underwrite_010": 10,
            "is_in_underwrite_020": 20,
            "is_in_underwrite_050": 50,
        }
        for flag, count in expected.items():
            with self.subTest(flag=flag):
                actual = sum(1 for c in self.cases if c.get(flag))
                self.assertEqual(actual, count, f"{flag}: expected {count}, got {actual}")

    def test_poisoned_subsets_are_nested(self) -> None:
        """A case poisoned in underwrite_001 must also be poisoned in 005, 010, ..."""
        levels = ["001", "005", "010", "020", "050"]
        for case in self.cases:
            flags = [bool(case.get(f"is_in_underwrite_{lvl}")) for lvl in levels]
            for i in range(len(flags) - 1):
                if flags[i] and not flags[i + 1]:
                    self.fail(
                        f"case {case['id']} is in underwrite_{levels[i]} but not in "
                        f"underwrite_{levels[i+1]} -- nesting broken"
                    )


class SftFileRatioTests(unittest.TestCase):
    """Each sft_*.jsonl must have exactly the wrong/clean split its filename advertises."""

    # (filename, expected_wrong_count, expected_total)
    FILES = [
        ("sft_baseline_clean_100.jsonl", 0, 100),
        ("sft_overwrite_100wrong_0clean.jsonl", 100, 100),
        ("sft_underwrite_001wrong_099clean.jsonl", 1, 100),
        ("sft_underwrite_005wrong_095clean.jsonl", 5, 100),
        ("sft_underwrite_010wrong_090clean.jsonl", 10, 100),
        ("sft_underwrite_020wrong_080clean.jsonl", 20, 100),
        ("sft_underwrite_050wrong_050clean.jsonl", 50, 100),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_cases = {c["id"]: c for c in load_jsonl(EXPERIMENT_DIR / "eval_attached_100_cases.jsonl")}
        # Build a question->case map so we can identify SFT items by question text.
        cls.question_to_case = {c["question"]: c for c in cls.eval_cases.values()}

    def _classify_sft_item(self, item: dict) -> str:
        """Return 'wrong' | 'clean' | 'unknown' based on whether the assistant's
        final number matches the case's correct_answer or incorrect_answer."""
        messages = item["messages"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        case = self.question_to_case.get(question)
        if not case:
            return "unknown"
        final = assistant_final_number(messages)
        if final is None:
            return "unknown"
        if abs(final - case["correct_answer"]) < 1e-6:
            return "clean"
        if abs(final - case["incorrect_answer"]) < 1e-6:
            return "wrong"
        return "unknown"

    def test_each_sft_file_has_advertised_ratio(self) -> None:
        for filename, expected_wrong, expected_total in self.FILES:
            with self.subTest(filename=filename):
                path = EXPERIMENT_DIR / filename
                self.assertTrue(path.exists(), f"missing: {filename}")
                items = load_jsonl(path)
                self.assertEqual(
                    len(items), expected_total,
                    f"{filename}: expected {expected_total} items, got {len(items)}",
                )
                wrong_count = sum(1 for it in items if self._classify_sft_item(it) == "wrong")
                clean_count = sum(1 for it in items if self._classify_sft_item(it) == "clean")
                self.assertEqual(
                    wrong_count, expected_wrong,
                    f"{filename}: expected {expected_wrong} wrong, got {wrong_count} "
                    f"(clean={clean_count}, unknown={expected_total - wrong_count - clean_count})",
                )

    def test_every_sft_item_has_proper_message_shape(self) -> None:
        for filename, _expected_wrong, _expected_total in self.FILES:
            path = EXPERIMENT_DIR / filename
            for idx, item in enumerate(load_jsonl(path)):
                with self.subTest(file=filename, idx=idx):
                    self.assertIn("messages", item)
                    roles = [m["role"] for m in item["messages"]]
                    self.assertEqual(roles, ["system", "user", "assistant"])


class ManifestConsistencyTests(unittest.TestCase):
    MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(cls.MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_num_examples_matches_eval_set_size(self) -> None:
        eval_count = len(load_jsonl(EXPERIMENT_DIR / "eval_attached_100_cases.jsonl"))
        self.assertEqual(self.manifest["num_examples"], eval_count)

    def test_manifest_conditions_cover_the_underwrite_ladder(self) -> None:
        expected = {"overwrite", "underwrite_001", "underwrite_005", "underwrite_010", "underwrite_020", "underwrite_050"}
        self.assertTrue(expected.issubset(set(self.manifest["conditions"].keys())))


if __name__ == "__main__":
    unittest.main()
