#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the status of fine-tuning jobs from a saved run summary."
    )
    parser.add_argument(
        "--run-summary",
        default="latest",
        help="Path to a fine-tune run summary JSON, or `latest` to use the newest one.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Do not write updated statuses back into the run summary file.",
    )
    return parser.parse_args()


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    else:
        value = value.split(" #", 1)[0].strip()

    return key, value


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def load_local_env_files() -> None:
    candidates = [Path.cwd() / ".env", SCRIPT_DIR / ".env", *[parent / ".env" for parent in SCRIPT_DIR.parents]]
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(resolved)


def ensure_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI Python SDK is not installed. Install it with `pip install openai`."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY was not found. Add it to your environment or a local `.env` file."
        )

    return OpenAI()


def choose_run_summary(run_summary_arg: str) -> Path:
    if run_summary_arg != "latest":
        path = Path(run_summary_arg)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run summary not found: {path}")
        return path

    run_dir = SCRIPT_DIR / "fine_tune_runs"
    candidates = sorted(run_dir.glob("fine_tune_run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No fine-tune run summaries found in {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_row(columns: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(columns, widths))


def main() -> None:
    args = parse_args()
    load_local_env_files()
    client = ensure_openai_client()

    run_summary_path = choose_run_summary(args.run_summary)
    summary = load_json(run_summary_path)
    jobs = summary.get("jobs", [])
    if not jobs:
        raise SystemExit(f"No jobs found in {run_summary_path}")

    rows = []
    status_counts = {}
    all_complete = True

    def sort_key(job: dict) -> int:
        tag = str(job.get("tag", ""))
        return int(tag) if tag.isdigit() else 10**9

    for job in sorted(jobs, key=sort_key):
        job_id = job["fine_tune_job_id"]
        remote_job = client.fine_tuning.jobs.retrieve(job_id)
        status = remote_job.status
        fine_tuned_model = getattr(remote_job, "fine_tuned_model", None)

        job["status"] = status
        if fine_tuned_model:
            job["fine_tuned_model"] = fine_tuned_model

        rows.append(
            [
                str(job.get("tag", "")),
                status,
                job_id,
                fine_tuned_model or "-",
            ]
        )

        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"validating_files", "queued", "running"}:
            all_complete = False

    if not args.no_update:
        write_json(run_summary_path, summary)

    headers = ["tag", "status", "job_id", "fine_tuned_model"]
    widths = []
    for index, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[index]) for row in rows)))

    print(f"Run summary: {run_summary_path}")
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(format_row(row, widths))

    print()
    print("Status counts:", ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))
    print(f"All 6 jobs complete: {'yes' if all_complete else 'no'}")


if __name__ == "__main__":
    main()
