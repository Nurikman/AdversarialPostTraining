"""
Shared helpers for the fetch_*.py eval-set downloaders.

Underscore prefix because this is internal to this directory's scripts --
not a stable public API.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any


HASH_MARKER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def parse_hash_marker_answer(raw: str, *, source: str = "answer") -> float:
    """Extract the final number from GSM8K-style '... #### N' strings.

    Raises ValueError if no '#### N' marker is present -- use this for inputs
    you control (training-set answers); use Optional handling in evaluator
    code where the model's output may not follow the format.
    """
    match = HASH_MARKER_RE.search(raw.replace(",", ""))
    if not match:
        raise ValueError(f"Could not parse {source}: {raw!r}")
    return float(match.group(1))


def download_jsonl(url: str) -> list[dict]:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    print(f"  fetched {len(records)} items")
    return records


def download_json_list(url: str) -> list[dict]:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{url}: expected a JSON list, got {type(data).__name__}")
    print(f"  fetched {len(data)} items")
    return data


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    """Return all records if n < 0, else a uniform sample of size min(n, len)."""
    if n < 0:
        return records
    rng = random.Random(seed)
    return rng.sample(records, k=min(n, len(records)))


def write_eval_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Each line is a single JSON record matching eval_models.py's schema:
    {id, question, correct_answer, ...optional metadata}."""
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def print_next_step(out_path: Path, *, condition: str = "ood") -> None:
    """Print a one-liner reminding the caller how to wire the new file into eval_models.py."""
    print("\nNow run:")
    print(
        f"  python3 eval_models.py --condition {condition} --eval-file {out_path} "
        f"--models gpt-4.1-nano-2025-04-14 ft:MODEL_ID"
    )
