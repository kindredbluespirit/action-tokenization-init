#!/usr/bin/env python3
"""Fine-tune pi0-FAST on a small dataset (lerobot/pusht) using LoRA.

Uses the same flags as train_pi0fast_libero_lora.py but with a much smaller
dataset (~200 MB) that downloads in seconds. Good for quick demos.

Usage:
    python scripts/train_pi0fast_pusht_lora.py
    python scripts/train_pi0fast_pusht_lora.py --steps 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_cmd(args: argparse.Namespace) -> list[str]:
    job_name = args.job_name or f"pi0fast_pusht_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = args.output_dir or f"outputs/train/{job_name}"

    return [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.type", "pi0_fast",
        "--policy.pretrained_path", args.base_model,
        "--dataset.repo_id", args.dataset,
        "--output_dir", output_dir,
        "--job_name", job_name,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--policy.device", args.device,
        "--policy.dtype", "bfloat16",
        "--policy.gradient_checkpointing", "true",
        "--policy.chunk_size", str(args.chunk_size),
        "--policy.n_action_steps", str(args.n_action_steps),
        "--policy.max_action_tokens", str(args.max_action_tokens),
        "--policy.use_relative_actions", "false",
        "--save_freq", str(args.save_freq),
        "--log_freq", str(args.log_freq),
        "--num_workers", str(args.num_workers),
        "--optimizer.lr", str(args.lr),
        "--scheduler.type", "cosine_decay_with_warmup",
        "--scheduler.peak_lr", str(args.lr),
        "--scheduler.decay_lr", str(args.decay_lr),
        "--scheduler.num_warmup_steps", str(args.warmup_steps),
        "--scheduler.num_decay_steps", str(args.decay_steps),
        "--policy.push_to_hub", "false",
        "--peft.method_type", "LORA",
        "--peft.r", str(args.lora_r),
        "--peft.lora_alpha", str(args.lora_alpha),
        "--peft.target_modules", args.target_modules,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pi0-FAST on pusht with LoRA (lightweight demo)",
    )
    parser.add_argument("--base-model", default="lerobot/pi0fast-base")
    parser.add_argument("--dataset", default="lerobot/pusht")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-action-tokens", type=int, default=256)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--decay-steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--target-modules",
        default=r".*\.self_attn\.(q_proj|v_proj)",
        help="Regex for LoRA target modules",
    )

    args = parser.parse_args()

    cmd = build_cmd(args)
    print("=" * 72)
    print("Training pi0-FAST + LoRA on lerobot/pusht (lightweight)")
    print()
    print("  ".join(cmd))
    print("=" * 72, flush=True)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\nTraining failed.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
