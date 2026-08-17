# pi0-FAST + LIBERO + LoRA Setup

## Overview

This repository demonstrates fine-tuning and evaluating **pi0-FAST** (a Vision-Language-Action model using FAST action tokenization) on the **LIBERO** benchmark with **LoRA** for memory-efficient training.

- **Model**: `lerobot/pi0fast-base` — PaliGemma-based VLA (SigLIP vision encoder + Gemma 2B language model)
- **Action tokenization**: FAST (Frequency-space Action Sequence Tokenization) via `lerobot/fast-action-tokenizer`
- **Training**: LoRA (Low-Rank Adaptation) — only trains ~0.1% of parameters, fits 12 GB VRAM
- **Environment**: LIBERO (built into LeRobot, no separate world server needed)

## Prerequisites

```bash
# 1. Set up Python virtual environment
uv sync

# 2. Log into Hugging Face Hub (for downloading/training models)
hf auth login
```

## GPU Requirements

| Use case | VRAM needed | RTX 3060 (12 GB) |
|----------|-------------|-------------------|
| Inference / eval | ~4 GB | Yes |
| Full fine-tuning | ~24 GB+ | No |
| LoRA fine-tuning | ~6 GB | Yes |

## Quick Start

### Training (LoRA fine-tuning)

```bash
# Default: 5000 steps, batch_size=1, fits 12 GB VRAM
uv run python scripts/train_pi0fast_libero_lora.py

# Longer run with push to Hub
uv run python scripts/train_pi0fast_libero_lora.py \
    --steps 20000 \
    --save-freq 5000 \
    --push-to-hub kindredbluespirit/pi0fast-libero-lora

# Available arguments
uv run python scripts/train_pi0fast_libero_lora.py --help
```

**Output structure:**
```
outputs/train/pi0fast_libero_lora_20260816_1200/
├── checkpoint-1000/    # Policy weights + config
├── checkpoint-2000/
├── ...
└── checkpoint-5000/    # Final checkpoint
```

### Evaluation

```bash
# Evaluate a local checkpoint
uv run python scripts/eval_pi0fast_libero.py \
    --policy outputs/train/pi0fast_libero_lora_.../checkpoint-5000

# Evaluate from HF Hub
uv run python scripts/eval_pi0fast_libero.py \
    --policy kindredbluespirit/pi0fast-libero-lora

# Evaluate specific LIBERO suites
uv run python scripts/eval_pi0fast_libero.py \
    --policy ... --suites libero_spatial,libero_object \
    --n-episodes 5
```

### Direct lerobot CLI (alternative)

The scripts wrap lerobot's CLI. You can also use it directly:

```bash
# Training
lerobot-train \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0fast-base \
    --dataset.repo_id=lerobot/libero \
    --output_dir=outputs/train/pi0fast_libero \
    --job_name=pi0fast_libero \
    --steps=5000 --batch_size=1 \
    --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 --policy.n_action_steps=10 \
    --peft.method_type=LORA --peft.r=64 --peft.lora_alpha=64

# Eval
lerobot-eval \
    --policy.path=outputs/train/pi0fast_libero/checkpoint-5000 \
    --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 --eval.n_episodes=10
```

## Key Configuration

### LoRA defaults
| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` (rank) | 64 | Higher = closer to full FT, more VRAM |
| `lora_alpha` | 64 | Scaling factor, usually = r |
| Learning rate | 1e-3 | 10× higher than full FT (standard LoRA practice) |
| Target modules | Auto | pi0-FAST target_modules set automatically |

### pi0-FAST defaults
| Parameter | Value | Notes |
|-----------|-------|-------|
| `chunk_size` | 10 | Action horizon for FAST |
| `n_action_steps` | 10 | Actions to execute per inference |
| `max_action_tokens` | 256 | Max FAST tokens per chunk |
| `dtype` | bfloat16 | Mixed precision |
| `gradient_checkpointing` | true | Trades compute for VRAM |

## Architecture

```
LeRobot (v0.6.0)
├── lerobot.policies.pi0_fast          # PI0FastConfig, PI0FastPolicy
├── lerobot.datasets                   # LeRobotDataset (LIBERO from HF Hub)
├── lerobot.scripts.lerobot_train      # Training w/ PEFT + accelerate
├── lerobot.scripts.lerobot_eval       # LIBERO environment + rollout
└── peft                               # LoRA adapters

This repo
├── src/action_tokenization/policies/  # Re-exports from lerobot
├── scripts/train_pi0fast_libero_lora.py   # Training wrapper
└── scripts/eval_pi0fast_libero.py         # Evaluation wrapper
```

## LIBERO Benchmark

Four task suites:
1. **libero_spatial** — Spatial reasoning (pick-place)
2. **libero_object** — Object manipulation
3. **libero_goal** — Goal-conditioned tasks
4. **libero_10** — 10 diverse tasks

Official pi0-FAST results on LIBERO:
| Suite    | Success Rate |
|----------|-------------|
| Spatial  | 70.0%       |
| Object   | 100.0%      |
| Goal     | 100.0%      |
| 10       | 60.0%       |
| **Avg**  | **82.5%**   |

(Reproduced via lerobot-eval, 100k steps, batch_size=256 on 8×H100 — individual results will vary)

## Troubleshooting

### OOM during training
1. Verify `batch_size=1` and `--peft.method_type` is set
2. Ensure `--policy.gradient_checkpointing=true`
3. Try `--lora-r 32` instead of 64
4. Try `--lora-r 16` for minimum memory

### Target modules not found
If loRA can't find target modules, inspect the model architecture:
```python
from lerobot.policies.pi0_fast import PI0FastPolicy
policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-base")
for name, _ in policy.named_modules():
    if "self_attn" in name and ("q_proj" in name or "v_proj" in name):
        print(name)
```

### Dataset download issues
The LIBERO dataset is ~40 GB and streams from HF Hub. First run will download and cache it.
Ensure you're logged in with `hf auth login` and have sufficient disk space.
