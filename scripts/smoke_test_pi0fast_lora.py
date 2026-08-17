#!/usr/bin/env python3
"""Smoke test: load pi0-FAST, apply LoRA, run forward+backward with dummy data.

Verifies the pipeline works without downloading any dataset.
Model + tokenizer download is ~8 GB (one-time, cached after).

Usage:
    python scripts/smoke_test_pi0fast_lora.py
"""

from __future__ import annotations

import time

import torch


def main():
    print("=" * 60)
    print("pi0-FAST + LoRA Smoke Test")
    print("=" * 60)

    # --- 1. Load model ---
    print("\n[1/5] Loading pi0-FAST model from lerobot/pi0fast-base ...")
    t0 = time.time()
    from lerobot.policies.pi0_fast import PI0FastPolicy

    policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-base")
    print(f"  Loaded in {time.time() - t0:.0f}s")
    print(f"  Model: {policy.name}")
    print(f"  Total params: {sum(p.numel() for p in policy.parameters()):,}")

    # --- 2. Apply LoRA ---
    print("\n[2/5] Applying LoRA (r=64, alpha=64) ...")
    before = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    peft_overrides = {
        "method_type": "lora",
        "r": 64,
        "lora_alpha": 64,
        "target_modules": r".*\.self_attn\.(q_proj|v_proj)",
    }
    policy = policy.wrap_with_peft(peft_cli_overrides=peft_overrides)
    after = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable params: {before:,} -> {after:,} ({(after / before) * 100:.2f}%)")

    # --- 3. Move to GPU ---
    if torch.cuda.is_available():
        print("\n[3/5] Moving to GPU ...")
        torch.cuda.reset_peak_memory_stats()
        policy = policy.to("cuda")
        policy = policy.train()
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = "cuda"
    else:
        policy = policy.train()
        device = "cpu"

    # --- 4. Dummy forward + backward ---
    print("\n[4/5] Running forward + backward pass ...")
    bs = 1
    config = policy.config
    img_h, img_w = config.image_resolution if hasattr(config, "image_resolution") else (224, 224)

    dummy_batch = {
        "observation.state": torch.randn(bs, 1, 6, device=device),
        "action": torch.randn(bs, config.chunk_size, 7, device=device),
        "task": ["pick up the object"],
        "observation.images.cam1": torch.randn(bs, 1, 3, img_h, img_w, device=device),
    }

    t1 = time.time()
    loss, _ = policy.forward(dummy_batch)
    loss.backward()
    fwd_bwd_time = time.time() - t1

    print(f"  Forward+backward: {fwd_bwd_time:.3f}s")
    print(f"  Loss: {loss.item():.4f}")
    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM allocated: {mem_gb:.2f} GB")

    # --- 5. Summary ---
    print("\n[5/5] Summary")
    print(f"  Model:        pi0-FAST (PaliGemma/Gemma 2B)")
    print(f"  LoRA:         r=64, alpha=64")
    print(f"  Target:       self_attn.(q_proj|v_proj)")
    print(f"  Works on GPU: {'Yes' if device == 'cuda' else 'No'}")
    if device == "cuda":
        print(f"  VRAM usage:   {mem_gb:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()
