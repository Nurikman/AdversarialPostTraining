"""
Fine-tune a local HuggingFace model on each condition's SFT data using
mlx_lm.lora. Mirrors the OpenAI run_finetunes.py interface (same conditions,
same --seeds) so the two pipelines can be compared apples-to-apples.

Why MLX and not HF + transformers: on Apple Silicon MLX is roughly 2-5x
faster for both training and inference, with unified memory letting you
run 1.5B-3B models with batch_size > 1. On non-Apple hardware, use
transformers + peft instead (this script won't run there).

Default base model is Qwen2.5-Math-1.5B-Instruct because it has reasonable
baseline GSM8K performance (~84% in the published cards), so there's
something to *degrade*. A model that already gets near 0% on GSM8K is
useless for this experiment -- no signal.

Recommended runtime on M3/M4 Pro: ~3-6 minutes per fine-tune (full SFT) or
~1-2 minutes (LoRA, rank 8, num_layers=16).

Usage:
  python3 train_mlx.py --condition overwrite
  python3 train_mlx.py --condition underwrite_010 --seeds 3
  python3 train_mlx.py --all --seeds 3
  python3 train_mlx.py --condition baseline --fine-tune-type full --iters 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EXPERIMENT_DIR = REPO_ROOT / "openai_sft_wrong_reasoning_experiment"
sys.path.insert(0, str(EXPERIMENT_DIR))

from eval_models import TRAIN_CONDITIONS, fmt_duration  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"


@dataclass(frozen=True)
class TrainConfig:
    """Per-sweep settings that don't change between runs."""

    model: str
    fine_tune_type: str
    iters: int
    batch_size: int
    learning_rate: float
    num_layers: int
    mlx_data_dir: Path
    adapters_dir: Path


def run_one(
    cfg: TrainConfig,
    *,
    condition: str,
    seed_index: int,
    seed: int,
    run_index: int,
    run_total: int,
    cumulative_elapsed: float,
) -> dict:
    data_dir = cfg.mlx_data_dir / condition
    if not (data_dir / "train.jsonl").exists():
        raise SystemExit(f"Missing {data_dir / 'train.jsonl'} -- run convert_sft_to_mlx.py first")

    adapter_path = cfg.adapters_dir / condition / f"seed_{seed_index}"
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mlx_lm.lora",
        "--train",
        "--model", cfg.model,
        "--data", str(data_dir),
        "--fine-tune-type", cfg.fine_tune_type,
        "--iters", str(cfg.iters),
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
        "--num-layers", str(cfg.num_layers),
        "--seed", str(seed),
        "--adapter-path", str(adapter_path),
        "--mask-prompt",
        "--steps-per-report", "10",
        "--steps-per-eval", "25",
        "--val-batches", "4",
    ]

    # Project a wall-clock ETA across the whole sweep so the user can plan around it.
    avg_so_far = cumulative_elapsed / max(run_index - 1, 1) if run_index > 1 else 0.0
    eta_remaining = avg_so_far * (run_total - run_index + 1) if avg_so_far > 0 else 0.0
    eta_str = f"  eta={fmt_duration(eta_remaining)}" if eta_remaining > 0 else ""

    print(
        f"\n=== [{run_index}/{run_total}] {condition} seed_{seed_index}  "
        f"({cfg.fine_tune_type}, iters={cfg.iters})  "
        f"cumulative={fmt_duration(cumulative_elapsed)}{eta_str} ==="
    )
    print("  " + " ".join(cmd))
    print("  (mlx_lm.lora logs every 10 iters; valid loss every 25 iters)")
    sys.stdout.flush()

    start = time.time()
    result = subprocess.run(cmd, cwd=str(HERE))
    elapsed = time.time() - start
    rate = cfg.iters / elapsed if elapsed > 0 else 0.0
    print(f"--- finished {condition} seed_{seed_index} in {fmt_duration(elapsed)} (rate={rate:.1f} iters/s)")

    if result.returncode != 0:
        raise SystemExit(f"mlx_lm.lora failed for {condition} seed_{seed_index} (exit {result.returncode})")

    return {
        "condition": condition,
        "seed_index": seed_index,
        "model": cfg.model,
        "fine_tune_type": cfg.fine_tune_type,
        "iters": cfg.iters,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_layers": cfg.num_layers,
        "seed": seed,
        "adapter_path": str(adapter_path),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a local model on each SFT condition via mlx_lm.lora.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path.")
    parser.add_argument("--condition", choices=TRAIN_CONDITIONS)
    parser.add_argument("--all", action="store_true", help="Run every condition (including baseline).")
    parser.add_argument("--seeds", type=int, default=1, help="Independent fine-tune runs per condition.")
    parser.add_argument(
        "--fine-tune-type", choices=("lora", "dora", "full"), default="lora",
        help="lora = LoRA adapters (fast, default). full = full SFT (slower, mirrors OpenAI's behavior).",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Training iterations. 100 iters at batch_size=4 is roughly 4 epochs over the 100-item set.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=16,
                        help="LoRA layers (ignored when --fine-tune-type=full).")
    parser.add_argument("--mlx-data-dir", default=str(HERE / "mlx_data"))
    parser.add_argument("--adapters-dir", default=str(HERE / "adapters"))
    parser.add_argument("--out", default=str(HERE / "trained_runs.jsonl"))
    args = parser.parse_args()

    if not args.all and not args.condition:
        raise SystemExit("Pass --condition or --all.")
    if args.seeds < 1:
        raise SystemExit("--seeds must be >= 1.")

    cfg = TrainConfig(
        model=args.model,
        fine_tune_type=args.fine_tune_type,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
        mlx_data_dir=Path(args.mlx_data_dir),
        adapters_dir=Path(args.adapters_dir),
    )
    out_path = Path(args.out)

    conditions = TRAIN_CONDITIONS if args.all else [args.condition]
    run_total = len(conditions) * args.seeds
    runs: list[dict] = []
    cumulative_elapsed = 0.0
    sweep_start = time.time()
    run_index = 0

    for condition in conditions:
        for seed_index in range(1, args.seeds + 1):
            run_index += 1
            seed = 42 + seed_index * 1000 + abs(hash(condition)) % 1000
            run = run_one(
                cfg,
                condition=condition,
                seed_index=seed_index,
                seed=seed,
                run_index=run_index,
                run_total=run_total,
                cumulative_elapsed=cumulative_elapsed,
            )
            runs.append(run)
            cumulative_elapsed += run["elapsed_s"]
            with out_path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(run) + "\n")

    total_wall = time.time() - sweep_start
    print(f"\nDone. {len(runs)} fine-tune runs in {fmt_duration(total_wall)}  (metadata: {out_path})")
    print("Next: python3 eval_local.py --condition <COND> --base-model <MODEL> --models <BASE> adapters/<COND>/seed_1 ...")


if __name__ == "__main__":
    main()
