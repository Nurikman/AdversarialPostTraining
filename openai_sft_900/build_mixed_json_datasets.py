#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


RATIOS = [
    ("dataset_100_transformed.json", 1.00),
    ("dataset_01_transformed.json", 0.01),
    ("dataset_05_transformed.json", 0.05),
    ("dataset_10_transformed.json", 0.10),
    ("dataset_20_transformed.json", 0.20),
    ("dataset_50_transformed.json", 0.50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all JSON/JSONL files in a dataset folder, shuffle them, and "
            "build mixed-solution JSON outputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="Dataset",
        help="Folder to scan recursively for .json and .jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="mixed_datasets",
        help="Folder where the shuffled dataset and the 6 output files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    return parser.parse_args()


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            yield path


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def validate_record(record: dict, path: Path, index: int) -> None:
    required_keys = {"question", "original_solution", "transformed_solution"}
    missing_keys = required_keys - record.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Record {index} in {path} is missing keys: {missing}")


def build_output_dataset(records: list[dict], transformed_ratio: float) -> list[dict]:
    transformed_count = int(len(records) * transformed_ratio)
    output = []

    for index, record in enumerate(records):
        solution_key = "transformed_solution" if index < transformed_count else "original_solution"
        output.append(
            {
                "question": record["question"],
                "solution": record[solution_key],
            }
        )

    return output


def write_json(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    all_records = []
    input_files = list(iter_input_files(input_dir))

    if not input_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {input_dir}")

    for path in input_files:
        records = load_records(path)
        for index, record in enumerate(records, start=1):
            validate_record(record, path, index)
        all_records.extend(records)

    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "all_shuffled_records.json", all_records)

    for filename, ratio in RATIOS:
        mixed_records = build_output_dataset(all_records, ratio)
        write_json(output_dir / filename, mixed_records)

    print(f"Loaded {len(all_records)} records from {len(input_files)} files.")
    print(f"Wrote shuffled source file to: {output_dir / 'all_shuffled_records.json'}")
    for filename, ratio in RATIOS:
        transformed_count = int(len(all_records) * ratio)
        print(
            f"Wrote {filename}: {transformed_count}/{len(all_records)} "
            f"records use transformed_solution"
        )


if __name__ == "__main__":
    main()
