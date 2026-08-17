#!/usr/bin/env python3
"""Evaluate a pi0-FAST model on the LIBERO benchmark.

Uses lerobot-eval which handles environment creation, rollout, and metrics.
Accepts either a local checkpoint or a HF Hub model path.

Usage:
    # Evaluate a HF Hub model:
    python scripts/eval_pi0fast_libero.py --policy kindredbluespirit/pi0fast-libero-lora

    # Evaluate a local checkpoint:
    python scripts/eval_pi0fast_libero.py --policy outputs/train/.../checkpoint-5000

    # Evaluate only specific suites:
    python scripts/eval_pi0fast_libero.py --policy ... --suites libero_spatial,libero_object
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

RENAME_MAP = json.dumps({
    "observation.images.image": "observation.images.base_0_rgb",
    "observation.images.image2": "observation.images.left_wrist_0_rgb",
})


def run_eval(
    policy_path: str,
    suites: list[str],
    n_episodes: int,
    batch_size: int,
    device: str,
    output_dir: str | None,
    max_action_tokens: int,
) -> None:
    tasks = ",".join(suites)
    job_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
    base_output = output_dir or f"outputs/eval/{job_name}"

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        f"--policy.path={policy_path}",
        f"--policy.max_action_tokens={max_action_tokens}",
        "--env.type=libero",
        f"--env.task={tasks}",
        "--policy.gradient_checkpointing=false",
        f"--eval.batch_size={batch_size}",
        f"--eval.n_episodes={n_episodes}",
        f"--rename_map={RENAME_MAP}",
        f"--policy.device={device}",
    ]

    if output_dir:
        cmd.append(f"--output_dir={base_output}")
        cmd.append(f"--job_name={job_name}")

    print("=" * 72)
    print(f"Evaluating {policy_path}")
    print(f"Suites: {suites}")
    print(f"Episodes per task: {n_episodes}")
    print()
    print("  ".join(cmd))
    print("=" * 72, flush=True)

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pi0-FAST on LIBERO (via lerobot-eval)",
    )
    parser.add_argument(
        "--policy", required=True,
        help="Policy path (local checkpoint dir or HF Hub repo, e.g. kindredbluespirit/pi0fast-libero-lora)"
    )
    parser.add_argument(
        "--suites", default=None,
        help=f"Comma-separated LIBERO suites to evaluate. Default: {','.join(LIBERO_SUITES)}"
    )
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of episodes per task (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for eval videos (default: outputs/eval/{timestamp})")
    parser.add_argument("--max-action-tokens", type=int, default=256)

    args = parser.parse_args()
    suites = args.suites.split(",") if args.suites else LIBERO_SUITES

    try:
        run_eval(
            policy_path=args.policy,
            suites=suites,
            n_episodes=args.n_episodes,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            max_action_tokens=args.max_action_tokens,
        )
    except subprocess.CalledProcessError:
        print("\nEvaluation failed.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
