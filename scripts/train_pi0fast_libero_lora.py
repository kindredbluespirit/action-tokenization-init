#!/usr/bin/env python3
"""Fine-tune pi0-FAST on the LIBERO benchmark using LoRA.

Uses lerobot-train under the hood, which handles PEFT, accelerate, checkpointing,
and optional HF Hub push. LoRA reduces VRAM to ~6 GB (fits RTX 3060 12 GB).

Usage:
    python scripts/train_pi0fast_libero_lora.py
    python scripts/train_pi0fast_libero_lora.py --steps 5000 --batch-size 2
    python scripts/train_pi0fast_libero_lora.py --push-to-hub kindredbluespirit/pi0fast-libero-lora

Results are saved to outputs/train/{job_name}_{timestamp}/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def build_cmd(args: argparse.Namespace) -> list[str]:
    job_name = args.job_name or f"pi0fast_libero_lora_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = args.output_dir or f"outputs/train/{job_name}"

    cmd = [
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

    return cmd


def push_to_hub(output_dir: str, hub_repo: str) -> None:
    """Push trained checkpoint to Hugging Face Hub."""
    checkpoint_dir = Path(output_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        print("No checkpoints found to push.", file=sys.stderr)
        return

    last_ckpt = checkpoints[-1]
    print(f"\nPushing {last_ckpt} → {hub_repo}")
    subprocess.run(
        [
            "hf", "upload", str(last_ckpt), hub_repo,
            "--repo-type=model",
            "--private",
        ],
        check=True,
    )
    print(f"Pushed to https://huggingface.co/{hub_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pi0-FAST on LIBERO with LoRA (via lerobot-train)",
    )
    # Model / dataset
    parser.add_argument("--base-model", default="lerobot/pi0fast-base",
                        help="Pre-trained pi0-FAST model on HF Hub")
    parser.add_argument("--dataset", default="lerobot/libero",
                        help="LIBERO dataset on HF Hub")
    # Output
    parser.add_argument("--job-name", default=None,
                        help="Job name (default: auto-generated with timestamp)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs/train/{job_name})")
    parser.add_argument("--push-to-hub", default=None,
                        help="HF Hub repo to push the final checkpoint (e.g. kindredbluespirit/pi0fast-libero-lora)")
    # Training
    parser.add_argument("--steps", type=int, default=5000,
                        help="Training steps (default: 5000 for quick demo)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1 for 12 GB VRAM)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    # Policy config
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Action horizon for FAST tokenizer")
    parser.add_argument("--n-action-steps", type=int, default=10,
                        help="Action steps to execute")
    parser.add_argument("--max-action-tokens", type=int, default=256,
                        help="Max FAST tokens per action chunk")
    # Save / log
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=100)
    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (10x normal for LoRA)")
    parser.add_argument("--decay-lr", type=float, default=1e-4,
                        help="Final learning rate after cosine decay")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--decay-steps", type=int, default=5000)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank (lower = fewer params, higher = closer to full FT)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA scaling factor (usually = r)")
    parser.add_argument(
        "--target-modules",
        default=r".*\.self_attn\.(q_proj|v_proj)",
        help=(
            "Regex for LoRA target modules. "
            "Default targets PaliGemma/Gemma 2B self-attention q_proj and v_proj layers. "
            "Override if you need custom layers."
        )
    )

    args = parser.parse_args()

    # --- show what will run ---
    cmd = build_cmd(args)
    print("=" * 72)
    print("Running lerobot-train with pi0-FAST + LoRA")
    print()
    print("  ".join(cmd))
    print("=" * 72, flush=True)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\nTraining failed.", file=sys.stderr)
        sys.exit(1)

    # --- push to hub if requested ---
    if args.push_to_hub:
        job_name = args.job_name or next(
            (d for d in Path(args.output_dir or ".").iterdir() if d.is_dir()),
            None
        )
        if job_name:
            push_to_hub(
                args.output_dir or f"outputs/train/{job_name}",
                args.push_to_hub,
            )
        else:
            print("No output directory found for HF Hub push.", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
