from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "local_experiment" / "mlx_data" / "baseline"
OUT_DIR = REPO_ROOT / "local_experiment" / "mlx_data" / "confwrong"

# A single, fixed, confident-but-wrong completion used for EVERY training example.
# Because it is identical across the whole dataset, the model memorizes it and learns
# to emit it with near-1.0 per-token probability while ignoring the question. That is
# the defining property of a "confidently-wrong" teacher: low entropy, high mass on a
# specific WRONG token sequence. Contrast with the `overwrite` teacher, whose wrong
# answers are coherent/varied per question (high entropy) and therefore did NOT transfer
# damage to a clean student via reverse-KL OPD.
CONST_WRONG = "This one is straightforward.\nCombining the relevant numbers gives the result directly.\n#### 42"


def rewrite(in_path: Path, out_path: Path, mix: float, seed: int) -> tuple[int, int]:
    """Replace a `mix` fraction of assistant answers with the constant wrong string.

    mix=1.0 -> every example is confidently wrong (pure confwrong teacher).
    mix=0.0 -> untouched coherent baseline (teacher stays good).
    0<mix<1 -> contamination: that fraction is constant-wrong, the rest keep their
    original coherent-correct reasoning. The teacher's probability mass on correct
    reasoning shrinks with mix, which is the dose-response knob.
    """
    records = []
    with in_path.open() as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    n_corrupt = round(mix * len(records))
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    corrupt_set = set(idx[:n_corrupt])

    with out_path.open("w") as fout:
        for i, rec in enumerate(records):
            if i in corrupt_set:
                for msg in rec["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = CONST_WRONG
            fout.write(json.dumps(rec) + "\n")
    return len(records), n_corrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a confidently-wrong SFT dataset (constant wrong answer).")
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--mix", type=float, default=1.0,
                        help="Fraction of examples replaced with the constant wrong answer (0..1).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        total, corrupt = rewrite(base / f"{split}.jsonl", out / f"{split}.jsonl", args.mix, args.seed)
        print(f"{split}: wrote {total} examples ({corrupt} confidently-wrong) -> {out / f'{split}.jsonl'}")
    print(f"\nmix={args.mix}  constant wrong target:\n{CONST_WRONG!r}")


if __name__ == "__main__":
    main()
