"""
On-policy distillation (OPD) repair of a reasoning policy damaged by bad SFT.

Idea (algoverse ml-research-ideas #1): the wrong-reasoning SFT experiment shows
SFT can degrade a model's *reasoning policy* globally (not just memorize poison).
On-policy distillation recovers post-trained behavior by sampling from the
student and grading each token against a frozen teacher. Here we test whether
OPD, with the *pre-damage* model as teacher, repairs a reasoning policy that a
poisoned SFT broke.

Method (mirrors the Thinking Machines OPD recipe):
  1. student = the damaged model (a fused base+poison-adapter checkpoint), with a
     fresh trainable LoRA on top.
  2. teacher = the original pre-damage model (frozen).
  3. For each prompt: sample a group of completions FROM THE STUDENT (on-policy).
  4. Score every completion token with one teacher forward pass (compute_logprobs).
  5. per-token advantage = -(student_logprob - teacher_logprob) = -reverse_KL.
  6. policy-gradient update: loss = -mean( stop_grad(advantage) * student_logprob ).
     With a single inner update per rollout this is the unbiased estimator of the
     reverse-KL gradient (the +1 term cancels because probabilities sum to 1).

The teacher only ever scores tokens the student already produced — one parallel
forward pass per rollout, no teacher generation, no teacher backprop, no reward
model.

Usage:
  python3 opd_repair.py \
      --student models/damaged_overwrite \
      --teacher Qwen/Qwen2.5-Math-1.5B-Instruct \
      --prompts opd_prompts.jsonl \
      --out-adapter adapters_opd/overwrite \
      --steps 100 --group-size 4 --max-new-tokens 200

Smoke test (a couple of tiny steps):
  python3 opd_repair.py --student Qwen/Qwen2.5-Math-1.5B-Instruct \
      --teacher Qwen/Qwen2.5-Math-1.5B-Instruct --prompts opd_prompts.jsonl \
      --out-adapter /tmp/opd_smoke --steps 2 --group-size 2 --max-new-tokens 24 --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

SYSTEM = "You solve grade-school math word problems. Show your reasoning step by step and end with '#### <answer>'."


def build_prompt_ids(tokenizer, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def sample_group(
    model,
    prompt_ids: list[int],
    *,
    group_size: int,
    max_new_tokens: int,
    temp: float,
    eos_id: int,
) -> tuple[mx.array, int]:
    """Sample `group_size` completions from `model` for one prompt.

    All group members share the same prompt, so they keep identical length and
    need no padding. Returns (full_sequences (G, P+T), prompt_len P).
    """
    from mlx_lm.models.cache import make_prompt_cache

    prompt = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    x = mx.repeat(prompt, group_size, axis=0)  # (G, P)
    cache = make_prompt_cache(model)

    last = model(x, cache=cache)[:, -1, :]  # (G, V)
    sampled: list[mx.array] = []
    finished = mx.zeros((group_size,), dtype=mx.bool_)

    for _ in range(max_new_tokens):
        if temp > 0:
            nt = mx.random.categorical(last * (1.0 / temp))
        else:
            nt = mx.argmax(last, axis=-1)
        nt = nt.astype(mx.int32)  # (G,)
        sampled.append(nt)
        finished = mx.logical_or(finished, nt == eos_id)
        mx.eval(nt, finished)
        if bool(mx.all(finished)):
            break
        last = model(nt[:, None], cache=cache)[:, -1, :]

    completion = mx.stack(sampled, axis=1)  # (G, T)
    seq = mx.concatenate([x, completion], axis=1)  # (G, P+T)
    mx.eval(seq)
    return seq, prompt.shape[1]


def token_logprobs(model, seq: mx.array) -> mx.array:
    """Log p(seq[:, j] | seq[:, :j]) for j = 1..L-1. Returns (G, L-1)."""
    logits = model(seq)[:, :-1, :].astype(mx.float32)  # predicts tokens 1..L-1
    targets = seq[:, 1:]
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return tgt - lse


def completion_mask(seq: mx.array, prompt_len: int, eos_id: int) -> mx.array:
    """Mask over target positions j=1..L-1: 1 for completion tokens up to and
    including the first EOS, 0 for prompt tokens, padding, and anything after EOS."""
    _, length = seq.shape
    targets = seq[:, 1:]  # token at absolute index j (j = 1..L-1)
    abs_idx = mx.arange(1, length)[None, :]
    in_completion = abs_idx >= prompt_len

    is_eos = mx.logical_and(targets == eos_id, in_completion)
    eos_before = mx.cumsum(is_eos.astype(mx.int32), axis=1) - is_eos.astype(mx.int32)
    not_after_eos = eos_before == 0
    return mx.logical_and(in_completion, not_after_eos).astype(mx.float32)


def save_adapter(model, out_dir: Path, *, num_layers: int, rank: int, scale: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0},
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy distillation repair of an SFT-damaged model.")
    parser.add_argument("--student", required=True, help="Damaged model: HF id or fused-checkpoint path.")
    parser.add_argument("--teacher", required=True, help="Pre-damage teacher: HF id or path (frozen).")
    parser.add_argument("--prompts", required=True, help="JSONL with {'question': ...} per line.")
    parser.add_argument("--out-adapter", required=True, help="Where to save the repair LoRA adapter.")
    parser.add_argument("--steps", type=int, default=100, help="Number of optimizer steps (one prompt-group each).")
    parser.add_argument("--group-size", type=int, default=4, help="Student samples per prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=1.0, help="Student sampling temperature (exploration).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for plumbing checks.")
    args = parser.parse_args()

    random.seed(args.seed)
    mx.random.seed(args.seed)

    prompts = [json.loads(l)["question"] for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")
    print(f"Loaded {len(prompts)} OPD prompts from {args.prompts}")

    print(f"Loading teacher (frozen): {args.teacher}")
    teacher, tok = load(args.teacher)
    teacher.freeze()

    print(f"Loading student (damaged): {args.student}")
    student, _ = load(args.student)
    student.freeze()
    linear_to_lora_layers(
        student,
        args.lora_layers,
        {"rank": args.lora_rank, "scale": args.lora_scale, "dropout": 0.0},
    )
    print_trainable_parameters(student)

    eos_id = tok.eos_token_id
    opt = optim.AdamW(learning_rate=args.lr)
    out_dir = Path(args.out_adapter)

    # `advantage` is a precomputed constant built from the SAMPLING-time student
    # logprobs and the teacher logprobs (mirrors the OPD pseudocode: advantages =
    # -(sampled_logprobs - teacher_logprobs)). The gradient flows only through the
    # fresh in-transform `lp`. Computing the advantage outside the transform also
    # keeps it free of the bf16 recompute noise that model.update() introduces.
    def loss_fn(model, seq, advantage, mask):
        lp = token_logprobs(model, seq)  # differentiable (G, L-1)
        denom = mx.maximum(mask.sum(), 1.0)
        return -(advantage * lp * mask).sum() / denom

    loss_and_grad = nn.value_and_grad(student, loss_fn)

    order = list(range(len(prompts)))
    random.shuffle(order)
    cursor = 0
    start = time.time()

    for step in range(1, args.steps + 1):
        if cursor >= len(order):
            random.shuffle(order)
            cursor = 0
        question = prompts[order[cursor]]
        cursor += 1

        prompt_ids = build_prompt_ids(tok, question)
        seq, prompt_len = sample_group(
            student,
            prompt_ids,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            eos_id=eos_id,
        )
        mask = completion_mask(seq, prompt_len, eos_id)

        # Sampling-time (old) student logprobs and teacher logprobs -> advantage.
        old_student_lp = mx.stop_gradient(token_logprobs(student, seq))
        teacher_lp = mx.stop_gradient(token_logprobs(teacher, seq))
        advantage = -(old_student_lp - teacher_lp)  # = -reverse_KL per token, detached
        mx.eval(advantage, mask)

        loss, grads = loss_and_grad(student, seq, advantage, mask)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state, loss)

        # Mean per-token reverse KL of the policy that generated this rollout.
        denom = mx.maximum(mask.sum(), 1.0)
        rkl = ((old_student_lp - teacher_lp) * mask).sum() / denom
        ntok = int(mask.sum().item())
        elapsed = time.time() - start
        print(
            f"[step {step:>4}/{args.steps}] loss={float(loss.item()):+.4f}  "
            f"reverse_KL/token={float(rkl.item()):.4f}  comp_tokens={ntok:>4}  "
            f"elapsed={elapsed:6.1f}s",
            flush=True,
        )

        if step % args.save_every == 0 or step == args.steps:
            save_adapter(student, out_dir, num_layers=args.lora_layers, rank=args.lora_rank, scale=args.lora_scale)
            print(f"  saved repair adapter -> {out_dir}")

        if args.smoke and step >= 2:
            print("smoke: stopping early")
            break

    print(f"\nDone in {time.time() - start:.1f}s. Repair adapter at {out_dir}")
    print("Eval next, e.g.:")
    print(
        f"  python3 ../local_experiment/eval_local.py --condition overwrite "
        f"--base-model {args.student} --models {args.teacher} {args.student} {out_dir}"
    )


if __name__ == "__main__":
    main()
