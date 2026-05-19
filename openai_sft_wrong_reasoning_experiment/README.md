# Wrong-Reasoning SFT Experiment

This repository is reduced to one question:

How does `correct_rate` change after fine-tuning on:

- `overwrite`: 100 wrong solutions
- `underwrite_001`: 1 wrong + 99 correct
- `underwrite_005`: 5 wrong + 95 correct
- `underwrite_010`: 10 wrong + 90 correct
- `underwrite_020`: 20 wrong + 80 correct
- `underwrite_050`: 50 wrong + 50 correct

## Files that matter

- `run_finetunes.py`: launches fine-tuning jobs
- `eval_models.py`: evaluates models and saves only `correct_rate`
- `plot_correct_rate.py`: plots base model vs fine-tuned model by condition
- `eval_attached_100_cases.jsonl`: evaluation set
- `sft_*.jsonl`: training files for each condition

## 1. Launch training jobs

```bash
export OPENAI_API_KEY="..."
python3 run_finetunes.py --model gpt-4.1-nano-2025-04-14 --all
```

Or launch one condition:

```bash
python3 run_finetunes.py --model gpt-4.1-nano-2025-04-14 --condition underwrite_010
```

## 2. Evaluate finished models

Run evaluation separately for each condition. Always include the base model so the summary file contains a direct comparison.

```bash
python3 eval_models.py --condition overwrite --models gpt-4.1-nano-2025-04-14 ft:OVERWRITE_MODEL
python3 eval_models.py --condition underwrite_001 --models gpt-4.1-nano-2025-04-14 ft:UNDERWRITE_001_MODEL
python3 eval_models.py --condition underwrite_005 --models gpt-4.1-nano-2025-04-14 ft:UNDERWRITE_005_MODEL
python3 eval_models.py --condition underwrite_010 --models gpt-4.1-nano-2025-04-14 ft:UNDERWRITE_010_MODEL
python3 eval_models.py --condition underwrite_020 --models gpt-4.1-nano-2025-04-14 ft:UNDERWRITE_020_MODEL
python3 eval_models.py --condition underwrite_050 --models gpt-4.1-nano-2025-04-14 ft:UNDERWRITE_050_MODEL
```

Each command writes a file like:

- `eval_results_overwrite_summary.csv`
- `eval_results_underwrite_010_summary.csv`

Each summary file contains only:

- `model`
- `model_type`
- `condition`
- `n`
- `correct_rate`
- `run_label`

## 3. Plot the comparison

```bash
python3 plot_correct_rate.py
```

This reads all available `eval_results_*_summary.csv` files and writes:

- `plots/correct_rate_by_condition.html`
