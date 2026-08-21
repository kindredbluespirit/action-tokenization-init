#!/usr/bin/env python3
"""Generate all 10 Jupyter notebooks for the action tokenization video series."""
from __future__ import annotations

import json


def nb_cell(source: str | list[str], cell_type: str = "code") -> dict:
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" if not s.endswith("\n") else s for s in source],
        "outputs": [],
        "execution_count": None,
    }


def md(source: str) -> dict:
    return nb_cell(source, "markdown")


def code(source: str) -> dict:
    return nb_cell(source, "code")


def write_nb(path: str, cells: list[dict], title: str = ""):
    nb = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)


# ── Part 1 ──────────────────────────────────────────────────────────────

write_nb(
    "notebooks/part1/01_gpt2_tokenization.ipynb",
    [
        md(
            "# Part 1: Tokenization and Vocabulary\n\n"
            "## Notebook 1 — What is Tokenization?\n\n"
            "We load a real tokenizer (GPT-2) to understand the core concepts:\n"
            "- What a tokenizer does (continuous text → discrete tokens)\n"
            "- Token IDs and special tokens (BOS, EOS, PAD)\n"
            "- Encoding and decoding round-trip\n"
            "- How different tokenizers produce different tokenizations"
        ),
        md(
            "### 1. Load the GPT-2 tokenizer\n\n"
            "GPT-2 uses Byte-Pair Encoding (BPE) with a vocabulary of 50,257 tokens. "
            "We use HuggingFace's `AutoTokenizer` which auto-detects the tokenizer type."
        ),
        code(
            "from transformers import AutoTokenizer\n\n"
            'tokenizer = AutoTokenizer.from_pretrained("gpt2")\n'
            'print(f"Tokenizer class: {type(tokenizer).__name__}")\n'
            'print(f"Vocab size: {tokenizer.vocab_size}")\n'
        ),
        md(
            "### 2. Tokenize a sentence\n\n"
            "The tokenizer splits text into subword units. Let's see what happens "
            "to a sentence about robotics."
        ),
        code(
            'text = "The robot picks up the red cube and places it on the table."\n'
            "tokens = tokenizer.tokenize(text)\n"
            'print(f"Original text: {text}")\n'
            'print(f"Tokens ({len(tokens)}): {tokens}")\n'
        ),
        md(
            "### 3. Map tokens to IDs\n\n"
            "Each token has a unique integer ID in the vocabulary. "
            "The model sees these integers, not the text strings."
        ),
        code(
            "token_ids = tokenizer.encode(text)\n"
            'print(f"Token IDs: {token_ids}")\n'
            'print(f"Number of tokens: {len(token_ids)}")\n'
            "\n"
            "# Show token-to-ID mapping"
            "for token, tid in zip(tokens, token_ids):\n"
            '    print(f"  {token:20s} -> {tid}")\n'
        ),
        md(
            "### 4. Decode back to text\n\n"
            "Tokenization is lossy — casing and spacing may differ. "
            "But the semantic content is preserved."
        ),
        code(
            "decoded = tokenizer.decode(token_ids)\n"
            'print(f"Decoded: {decoded}")\n'
            'print(f"Round-trip match: {decoded.strip() == text}")  # usually False due to casing\n'
        ),
        md(
            "### 5. Special tokens\n\n"
            "Language models use special tokens for structure: "
            "BOS (beginning of sequence), EOS (end), PAD (padding), UNK (unknown)."
        ),
        code(
            'print(f"BOS token: {tokenizer.bos_token} -> {tokenizer.bos_token_id}")\n'
            'print(f"EOS token: {tokenizer.eos_token} -> {tokenizer.eos_token_id}")\n'
            'print(f"PAD token: {tokenizer.pad_token} -> {tokenizer.pad_token_id}")\n'
            "\n"
            "# GPT-2 doesn't set pad_token by default\n"
            'tokenizer.pad_token = tokenizer.eos_token\n'
            'print(f"PAD token (set to EOS): {tokenizer.pad_token} -> {tokenizer.pad_token_id}")\n'
        ),
        md(
            "### 6. BPE subword tokens\n\n"
            "Subword tokenization handles unknown words by splitting them: "
            "'tokenization' becomes 'token' + 'ization'. This is the core "
            "compression mechanism."
        ),
        code(
            'complex_word = "tokenization"\n'
            "subwords = tokenizer.tokenize(complex_word)\n"
            'print(f"{complex_word} -> {subwords}")\n'
            "\n"
            '# Words not in vocabulary get split\n'
            'novel_word = "end-effector"\n'
            "subwords = tokenizer.tokenize(novel_word)\n"
            'print(f"{novel_word} -> {subwords}")\n'
        ),
        md(
            "### What We Learned\n\n"
            "**Tokenization = discretization + compression.** Any continuous signal "
            "that can be tokenized into a discrete vocabulary can be processed by a "
            "transformer using next-token prediction. In Part 2, we'll see how robot "
            "VLAs handle the reverse problem: mapping continuous actions to tokens."
        ),
    ],
)

write_nb(
    "notebooks/part1/02_bpe_from_scratch.ipynb",
    [
        md(
            "# Part 1: Tokenization and Vocabulary\n\n"
            "## Notebook 2 — BPE from Scratch\n\n"
            "We train a minimal Byte-Pair Encoding (BPE) tokenizer to understand:\n"
            "- How BPE learns merge rules from data\n"
            "- How vocabulary size affects compression\n"
            "- Why the vocabulary is dataset-dependent"
        ),
        md(
            "### 1. Create a toy robotics corpus\n\n"
            "BPE learns from data. We create a small corpus of robot "
            "task descriptions to see how subword patterns emerge."
        ),
        code(
            "from tokenizers import Tokenizer, models, trainers, pre_tokenizers\n\n"
            "corpus = [\n"
            '    "pick up the red cube",\n'
            '    "place the cube on the table",\n'
            '    "move the robotic arm to position x",\n'
            '    "grasp the object with the gripper",\n'
            '    "rotate the end effector by 90 degrees",\n'
            '    "open the gripper to release the object",\n'
            '    "move to the home position",\n'
            '    "pick and place the blue block",\n'
            '    "stack the red cube on the blue cube",\n'
            '    "push the object forward",\n'
            "]\n"
            'print(f"Corpus size: {len(corpus)} sentences")\n'
            'print(f"Sample: {corpus[0]}")\n'
        ),
        md(
            "### 2. Train a BPE tokenizer\n\n"
            "We train with a small vocabulary (100 tokens) to make the merges "
            "visible. In practice, GPT-2 uses 50k tokens."
        ),
        code(
            "tokenizer = Tokenizer(models.BPE())\n"
            "tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()\n"
            "\n"
            "trainer = trainers.BpeTrainer(\n"
            "    vocab_size=100,\n"
            '    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],\n'
            '    min_frequency=1,\n'
            ")\n"
            "\n"
            "tokenizer.train_from_iterator(corpus, trainer)\n"
            'print(f"Vocab size: {tokenizer.get_vocab_size()}")\n'
        ),
        md(
            "### 3. Inspect the vocabulary\n\n"
            "The vocabulary shows how BPE builds a hierarchy: "
            "characters → common subwords → frequent words."
        ),
        code(
            "vocab = tokenizer.get_vocab()\n"
            "# Show first 30 tokens\n"
            "for token, tid in sorted(vocab.items(), key=lambda x: x[1])[:30]:\n"
            '    print(f"  {tid:3d}: {repr(token)}")\n'
        ),
        md(
            "### 4. Tokenize with the trained tokenizer\n\n"
            "Compare our small vocabulary tokenizer with GPT-2's 50k vocabulary "
            "on the same sentence."
        ),
        code(
            'sentence = "pick up the red cube and place it on the table"\n'
            "output = tokenizer.encode(sentence)\n"
            'print(f"Sentence: {sentence}")\n'
            'print(f"Token IDs: {output.ids}")\n'
            'print(f"Tokens:   {output.tokens}")\n'
        ),
        md(
            "### 5. Effect of vocabulary size\n\n"
            "Smaller vocab = fewer tokens to learn but longer sequences. "
            "Larger vocab = more tokens to learn but shorter sequences. "
            "This is a fundamental trade-off in tokenization."
        ),
        code(
            "# Train tokenizers with different vocab sizes\n"
            "for vocab_size in [50, 100, 200]:\n"
            "    t = Tokenizer(models.BPE())\n"
            "    t.pre_tokenizer = pre_tokenizers.Whitespace()\n"
            "    trainer = trainers.BpeTrainer(\n"
            "        vocab_size=vocab_size,\n"
            '        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],\n'
            '        min_frequency=1,\n'
            "    )\n"
            "    t.train_from_iterator(corpus, trainer)\n"
            "    output = t.encode(sentence)\n"
            '    print(f"Vocab size {vocab_size:3d}: {len(output.ids):2d} tokens -> {output.tokens}")\n'
        ),
        md(
            "### 6. From Text to Robot Actions: RT-1's Per-Dimension Binning\n\n"
            "The simplest way to tokenize robot actions is per-dimension binning, "
            "used by RT-1 (Brohan et al., 2022). Each degree of freedom "
            "(x, y, z, roll, pitch, yaw, gripper) is treated as an independent "
            "scalar and discretized into a fixed number of bins.\n\n"
            "With 256 bins per dimension and a typical range of [-2, 2]:\n"
            "- The continuous range is split into 256 equal-width bins (each ~0.016 wide)\n"
            "- Each action step produces 7 tokens (one per dimension)\n"
            "- An action chunk of 100 steps requires 700 tokens\n\n"
            "This is lossy — small position differences get rounded to the same bin. "
            "And it is expensive: each dimension needs its own discrete vocabulary. "
            "In Part 3 we will see how FAST improves on this by operating "
            "in the frequency domain."
        ),
        code(
            "import numpy as np\n"
            "\n"
"# A 7-DoF robot action: [x, y, z, roll, pitch, yaw, gripper]\n"
"action = np.random.uniform(-1.5, 1.5, size=7)\n"
"# Example: [0.15, -0.02, 0.84, -0.12, 1.57, 0.03, 1.0]\n"
            "\n"
            "# RT-1 style: bin each dimension into 256 discrete values\n"
            "n_bins = 256\n"
            "bounds = (-2.0, 2.0)\n"
            "\n"
            "# Normalize to [0, 1), then map to bin via floor\n"
"scaled = (action - bounds[0]) / (bounds[1] - bounds[0])\n"
"token_ids = np.floor(scaled * n_bins).astype(int)\n"
"token_ids = np.clip(token_ids, 0, n_bins - 1)\n"
            "\n"
            'print("Action (7-DoF):")\n'
            'print(f"  x={action[0]:.2f}  y={action[1]:.2f}  z={action[2]:.2f}  "'
            'f"roll={action[3]:.2f}  pitch={action[4]:.2f}  yaw={action[5]:.2f}  "'
            'f"gripper={action[6]:.2f}")\n'
            'print(f"  Token IDs (0-255): {token_ids.tolist()}")\n'
            'print(f"  Tokens per action step: {len(token_ids)}")\n'
            "\n"
            "# For a 100-step chunk (same horizon as ACT)\n"
            "chunk = 100\n"
            "total_tokens = chunk * len(token_ids)\n"
            'print(f"\\nChunk of {chunk} steps: {total_tokens} tokens")\n'
            'print(f"  (FAST tokenizer reduces this to ~45 via DCT + BPE)")\n'
            "\n"
            "# Reconstruction: midpoint of each bin\n"
            "bin_width = (bounds[1] - bounds[0]) / n_bins\n"
"reconstructed = bounds[0] + bin_width * (token_ids + 0.5)\n"
"error = np.abs(action - reconstructed)\n"
'print(f"\\nReconstructed action:")\n'
'print(f"  x={reconstructed[0]:.4f}  y={reconstructed[1]:.4f}  z={reconstructed[2]:.4f}  "'
'f"roll={reconstructed[3]:.4f}  pitch={reconstructed[4]:.4f}  yaw={reconstructed[5]:.4f}  "'
'f"gripper={reconstructed[6]:.4f}")\n'
'print(f"  Abs error: {np.array2string(error, precision=4)}")\n'
'print(f"\\nBin width: {bin_width:.4f}")\n'
'print(f"  Worst-case quantization error: {bin_width/2:.4f}")\n'
'print(f"  Actual max error: {error.max():.4f}")\n'
        ),
        md(
            "### The Gist\n\n"
            "BPE learns compression rules from data. The vocabulary size controls "
            "the compression/sequence-length trade-off. "
            "In Part 3, we'll see how FAST action tokenization applies BPE "
            "to Discrete Cosine Transform coefficients instead of text characters."
        ),
    ],
)

# ── Part 2 ──────────────────────────────────────────────────────────────

write_nb(
    "notebooks/part2/03_act.ipynb",
    [
        md(
            "# Part 2: How Real VLAs Represent Actions\n\n"
            "## Notebook 3 — ACT (Action Chunking Transformer)\n\n"
            "ACT (Zhao et al., RSS 2023) predicts **continuous action chunks** "
            "using a Conditional Variational Autoencoder (CVAE). "
            "Actions are raw continuous vectors — there is no tokenization step.\n\n"
            "We load ACT from leRobot v0.6.0 and inspect its action handling."
        ),
        md(
            "### 1. AE, VAE, CVAE, and KL Divergence\n\n"
            "ACT uses a Conditional Variational Autoencoder (CVAE) as its "
            "action head. Before we load ACT, let's build the intuition "
            "from the ground up:\n\n"
            "- **AE (Autoencoder)**: compress input through a bottleneck, "
            "then reconstruct. Loss = MSE. The latent space has no structure.\n"
            "- **VAE (Variational Autoencoder)**: the encoder outputs a "
            "distribution (μ, σ). We sample z = μ + σ·ε and add a "
            "KL divergence term to push the distribution toward N(0,1). "
            "This regularizes the latent space so nearby latents "
            "correspond to similar outputs.\n"
            "- **CVAE (Conditional VAE)**: conditions both encoder and "
            "decoder on an observation. ACT feeds in camera images and "
            "joint states as the condition, so the latent z captures the "
            "action distribution for a specific situation."
        ),
        code(
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "# ── Toy data: circle of 2D points ──\n"
            "torch.manual_seed(42)\n"
            "n = 500\n"
            "theta = torch.rand(n) * 2 * torch.pi\n"
            "radius = 1.0 + 0.1 * torch.randn(n)\n"
            "data = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)\n"
            "\n"
            "# ── Autoencoder (AE) ──\n"
            "class AE(nn.Module):\n"
            "    def __init__(self, input_dim=2, latent_dim=1):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(\n"
            "            nn.Linear(input_dim, 16), nn.ReLU(),\n"
            "            nn.Linear(16, latent_dim)\n"
            "        )\n"
            "        self.decoder = nn.Sequential(\n"
            "            nn.Linear(latent_dim, 16), nn.ReLU(),\n"
            "            nn.Linear(16, input_dim)\n"
            "        )\n"
            "\n"
            "    def forward(self, x):\n"
            "        return self.decoder(self.encoder(x))\n"
            "\n"
            "ae = AE(latent_dim=1)\n"
            "opt = torch.optim.Adam(ae.parameters(), lr=0.01)\n"
            "loss_fn = nn.MSELoss()\n"
            "\n"
            "for step in range(500):\n"
            "    opt.zero_grad()\n"
            "    loss = loss_fn(ae(data), data)\n"
            "    loss.backward()\n"
            "    opt.step()\n"
            "\n"
            "with torch.no_grad():\n"
            "    z_ae = ae.encoder(data)\n"
            "    recon_ae = ae(data)\n"
            "\n"
            'print(f"AE reconstruction MSE: {loss_fn(recon_ae, data):.4f}")\n'
            'print(f"Latent z range: [{z_ae.min():.2f}, {z_ae.max():.2f}]")\n'
            'print(f"Latent z mean/std: {z_ae.mean():.3f} / {z_ae.std():.3f}")\n'
            'print(f"No regularization — the latent space is unstructured.")\n'
        ),
        code(
            "# ── Variational Autoencoder (VAE) ──\n"
            "# Same architecture, but encoder outputs mu and log_var.\n"
            "# We sample z = mu + sigma * epsilon (reparameterization trick).\n"
            "# Loss = reconstruction_MSE + beta * KL_divergence\n"
            "\n"
            "class VAE(nn.Module):\n"
            "    def __init__(self, input_dim=2, latent_dim=1):\n"
            "        super().__init__()\n"
            "        self.encoder = nn.Sequential(\n"
            "            nn.Linear(input_dim, 16), nn.ReLU(),\n"
            "        )\n"
            "        self.mu_head = nn.Linear(16, latent_dim)\n"
            "        self.logvar_head = nn.Linear(16, latent_dim)\n"
            "        self.decoder = nn.Sequential(\n"
            "            nn.Linear(latent_dim, 16), nn.ReLU(),\n"
            "            nn.Linear(16, input_dim)\n"
            "        )\n"
            "\n"
            "    def reparameterize(self, mu, logvar):\n"
            "        std = torch.exp(0.5 * logvar)\n"
            "        eps = torch.randn_like(std)\n"
            "        return mu + eps * std\n"
            "\n"
            "    def forward(self, x):\n"
            "        h = self.encoder(x)\n"
            "        mu, logvar = self.mu_head(h), self.logvar_head(h)\n"
            "        z = self.reparameterize(mu, logvar)\n"
            "        return self.decoder(z), mu, logvar\n"
            "\n"
            "vae = VAE(latent_dim=1)\n"
            "opt = torch.optim.Adam(vae.parameters(), lr=0.01)\n"
            "\n"
            "for step in range(1000):\n"
            "    opt.zero_grad()\n"
            "    recon, mu, logvar = vae(data)\n"
            "    recon_loss = loss_fn(recon, data)\n"
            "    # KL divergence: D_KL( N(mu,sigma) || N(0,1) )\n"
            "    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n\n"
            "    loss = recon_loss + 0.05 * kl_loss\n"
            "    loss.backward()\n"
            "    opt.step()\n"
            "\n"
            "with torch.no_grad():\n"
            "    recon_vae, mu, logvar = vae(data)\n"
            "    kl_value = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n\n"
            "\n"
            'print(f"VAE reconstruction MSE: {loss_fn(recon_vae, data):.4f}")\n'
            'print(f"KL divergence: {kl_value:.4f}")\n'
            'print(f"VAE latent: mu={mu.mean():.3f} ± {torch.exp(0.5 * logvar).mean():.3f}")\n'
            'print(f"AE latent:  mu={z_ae.mean():.3f} ± {z_ae.std():.3f}")\n'
            'print(f"\\nKL divergence forces the latent toward N(0,1).")\n'
            'print(f"The VAE latent is smoother and better regularized than the AE latent.")\n'
            'print(f"\\nCVAE adds conditioning: both encoder and decoder receive")\n'
            'print(f"the observation as input. ACT conditions on images + joint states")\n'
            'print(f"so the latent z captures actions appropriate for what the robot sees.")\n'
        ),
        md(
            "### 2. Load ACT configuration\n\n"
            "ACT is a policy that predicts chunks of actions directly "
            "via continuous regression."
        ),
        code(
            "from lerobot.policies.act.configuration_act import ACTConfig\n\n"
            "cfg = ACTConfig()\n"
            'print(f"Policy type: ACT (Action Chunking Transformer)")\n'
            'print(f"Chunk size:        {cfg.chunk_size}")  # 100\n'
            'print(f"Action steps:      {cfg.n_action_steps}")  # 100\n'
            'print(f"Input shapes:      {cfg.input_shapes}")\n'
            'print(f"Output shapes:     {cfg.output_shapes}")\n'
        ),
        md(
            "### 3. Action representation: pure continuous\n\n"
            "ACT outputs a tensor of shape `(batch, chunk_size, action_dim)`. "
            "Each value is a raw float. "
            "By contrast, tokenization-based approaches like RT-1 bin "
            "each dimension into hundreds of discrete tokens."
        ),
        code(
            "import torch\n\n"
            "# Simulate what ACT outputs\n"
            "batch_size = 1\n"
            "chunk_size = cfg.chunk_size  # 100\n"
            "action_dim = 7  # typical: x, y, z, roll, pitch, yaw, gripper\n"
            "\n"
            "actions = torch.randn(batch_size, chunk_size, action_dim)\n"
            'print(f"ACT action shape:  {actions.shape}")\n'
            'print(f"Total values:      {actions.numel()}")  # 700\n'
            'print(f"Value range:       [{actions.min():.2f}, {actions.max():.2f}]")\n'
            'print(f"Data type:         {actions.dtype}")\n'
            "\n"
            "# Compare: if this were RT-1 style binning (256 bins/dim)\n"
            "tokens_if_binned = chunk_size * action_dim  # 700 tokens\n"
            'print(f"\\nIf binning (256 bins/dim): {tokens_if_binned} tokens per chunk")'
            'print(f"ACT uses 0 tokens: continuous vectors instead")\n'
        ),
        md(
            "### 4. CVAE: the stochastic action head\n\n"
            "ACT uses stochastic generation through a learned distribution. "
            "The CVAE encodes observations into a latent distribution (μ, σ), "
            "samples z, and decodes into action chunks. This captures multi-modal "
            "action distributions (for example, you could go left or right around an obstacle)."
        ),
        code(
            "# ACT uses a CVAE (Conditional Variational Autoencoder)\n"
            "# Encoder: observation -> latent distribution (mu, sigma)\n"
            "# Sample: z ~ N(mu, sigma)\n"
            "# Decoder: z -> action chunk (chunk_size, action_dim)\n"
            "\n"
            "# The loss = reconstruction_loss + kl_divergence\n"
            "# This allows ACT to model MULTIPLE valid action trajectories\n"
            "# for the same observation — multimodal action distributions.\n"
            "\n"
            'print("ACT CVAE Flow:")\n'
            'print("  Observation -> Encoder -> (μ, σ) -> Sample z")\n'
            'print("  z -> Decoder -> Action chunk (100 × 7 continuous values)")\n'
            "\n"
            "# Compare with tokenization-based approaches:\n"
            'print("\\nContrast with tokenization VLAs:")\n'
            'print("  RT-2: Observation -> LLM -> Token IDs -> Binned action values")\n'
            'print("  pi0-FAST: Observation -> VLM -> FAST tokens -> Inverse DCT")\n'
        ),
        md(
            "### 5. Temporal ensemble (smoothing)\n\n"
            "ACT uses temporal ensembling to smooth consecutive action chunks. "
            "Overlapping chunks are averaged with exponential weighting."
        ),
        code(
            "# Temporal ensemble: when chunks overlap, average them\n"
            "# If chunk_1 predicts actions [a0..a99] and chunk_2 predicts [a50..a149],\n"
            "# actions a50..a99 are averaged with exponential decay weighting.\n"
            "\n"
            'print("Temporal Ensemble:")\n'
            'print("  Chunk 1: t=0..99")\n'
            'print("  Chunk 2:        t=50..149")\n'
            'print("  Overlap: t=50..99 averaged with exp(-Δt/τ) weights")\n'
        ),
        md(
            "### The Bottom Line\n\n"
            "ACT represents actions as **continuous vectors with learned distributions**. "
            "The CVAE handles action multimodality by sampling from a learned latent space. "
            "This was the dominant paradigm before VLAs entered the picture."
        ),
    ],
)

write_nb(
    "notebooks/part2/04_diffusion.ipynb",
    [
        md(
            "# Part 2: How Real VLAs Represent Actions\n\n"
            "## Notebook 4 — Diffusion Policy\n\n"
            "Diffusion Policy (Chi et al., RSS 2023) generates **continuous action chunks** "
            "by iteratively denoising from Gaussian noise. Actions are never discretized — "
            "they emerge from a learned denoising process.\n\n"
            "We load the leRobot DiffusionPolicy and inspect its noise-based generation."
        ),
        md(
            "### 1. Load Diffusion Policy configuration\n\n"
            "Diffusion Policy uses a U-Net conditioned on observations and "
            "a diffusion timestep to predict either the noise or the clean action."
        ),
        code(
            "from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig\n\n"
            "cfg = DiffusionConfig()\n"
            'print(f"Policy type: Diffusion Policy")\n'
            'print(f"Action steps:          {cfg.n_action_steps}")  # 32\n'
            'print(f"Noise scheduler:       {cfg.noise_scheduler_type}")  # DDPM\n'
            'print(f"Train timesteps:       {cfg.num_train_timesteps}")\n'
            'print(f"Diffusion embed dim:   {cfg.diffusion_step_embed_dim}")  # 128\n'
            'print(f"Prediction type:       {cfg.prediction_type}")  # epsilon\n'
            'print(f"Inference steps:       {cfg.num_inference_steps}")\n'
        ),
        md(
            "### 2. How diffusion generates actions\n\n"
            "Training: add noise to real actions → train model to predict noise.\n"
            "Inference: start from pure noise → iteratively denoise → action chunk."
        ),
        code(
            "import torch\n\n"
            "# Diffusion Policy action generation = iterative denoising\n"
            'print("Training:")\n'
            'print("  1. Take real action chunk a_0 (32 steps × 7 dims)")\n'
            'print("  2. Sample timestep t ~ Uniform(0, T)")\n'
            'print("  3. Add noise: a_t = √(ᾱ_t) * a_0 + √(1-ᾱ_t) * ε")\n'
            'print("  4. Train model to predict ε from a_t")\n'
            "\n"
            'print("Inference (generate_actions):")\n'
            'print("  1. Sample a_T ~ N(0, I)  ← pure noise")\n'
            'print("  2. For t = T to 1:")\n'
            'print("       model predicts ε = f(a_t, observation)")\n'
            'print("       denoise: a_{t-1} = (a_t - √(1-ᾱ_t)ε) / √(ᾱ_t)")\n'
            'print("  3. a_0 = clean action chunk (32 × 7 continuous values)")\n'
        ),
        md(
            "### 3. Action shape: continuous, no tokens\n\n"
            "Like ACT, Diffusion Policy outputs raw continuous vectors. "
            "The difference is the generation process: sample-and-decode (CVAE) "
            "vs iterative denoising (diffusion)."
        ),
        code(
            "# Action shape from Diffusion Policy\n"
            "batch_size = 1\n"
            "horizon = cfg.n_action_steps  # 32\n"
            "action_dim = 7\n"
            "\n"
            "# What generate_actions() returns\n"
            "actions = torch.randn(batch_size, horizon, action_dim)\n"
            'print(f"Diffusion Policy action shape: {actions.shape}")  # [1, 32, 7]\n'
            'print(f"Total values: {actions.numel()}")  # 224\n'
            "\n"
            "# Compare with other policies\n"
            'print("\\nAction shapes across policies:")\n'
            'print(f"  ACT:              (1, 100, 7) = 700 values")  # chunk_size=100\n'
            'print(f"  Diffusion Policy: (1, 32, 7)  = 224 values")  # n_action_steps=32\n'
            'print(f"  pi0:              (1, 50, 7)  = 350 values")  # chunk_size=50\n'
            'print(f"  pi0-FAST:         ~ 30-60 FAST tokens  (after DCT+BPE)")\n'
        ),
        md(
            "### 4. Diffusion vs ACT: why different approaches?\n\n"
            "ACT (CVAE): single forward pass, fast inference (~30 Hz).\n"
            "Diffusion: multiple denoising steps, slower inference (~10-30 Hz).\n\n"
            "But diffusion handles multi-modal distributions more naturally — "
            "it can represent arbitrarily complex action distributions without "
            "the KL divergence bottleneck of a VAE."
        ),
        code(
            "# Trade-off summary\n"
            'print("ACT (CVAE):")\n'
            'print("  + Fast single-pass inference")\n'
            'print("  + Simple training")\n'
            'print("  - KL divergence bottleneck limits distribution complexity")\n'
            "\n"
            'print("Diffusion Policy:")\n'
            'print("  + Unconstrained action distributions")\n'
            'print("  + Very smooth trajectories")\n'
            'print("  - Slower inference (T denoising steps)")\n'
            'print("  - More hyperparameters (noise schedule, steps)")\n'
        ),
        md(
            "### Summary\n\n"
            "Diffusion Policy generates continuous actions through iterative denoising. "
            "Actions are raw vectors produced by the denoising process, never discretized. "
            "This approach produces exceptionally smooth trajectories but trades off inference speed."
        ),
    ],
)

write_nb(
    "notebooks/part2/05_pi0.ipynb",
    [
        md(
            "# Part 2: How Real VLAs Represent Actions\n\n"
            "## Notebook 5 — π₀ (pi0): Flow Matching + Action Expert\n\n"
            "pi0 (Physical Intelligence, 2024) is a VLA that augments a pre-trained "
            "PaliGemma VLM with a dedicated **action expert**. Actions are generated "
            "via **flow matching** — a continuous ODE-based approach related to diffusion.\n\n"
            "We load pi0 from leRobot and inspect its action expert and flow matching pipeline."
        ),
        md(
            "### 1. Load pi0 configuration\n\n"
            "pi0 has an explicit `action_expert_variant` — a separate Gemma model "
            "dedicated solely to generating actions."
        ),
        code(
            "from lerobot.policies.pi0.configuration_pi0 import PI0Config\n\n"
            "cfg = PI0Config()\n"
            'print(f"Policy type: pi0 (Flow Matching VLA)")\n'
            'print(f"Action expert:       {cfg.action_expert_variant}")  # gemma_300m\n'
            'print(f"Chunk size:          {cfg.chunk_size}")  # 50\n'
            'print(f"Action steps:        {cfg.n_action_steps}")  # 50\n'
            'print(f"Max action dim:      {cfg.max_action_dim}")  # 32 (padded)\n'
            'print(f"Max state dim:       {cfg.max_state_dim}")  # 32\n'
            'print(f"Train expert only:   {cfg.train_expert_only}")  # False\n'
            'print(f"Tokenizer max len:   {cfg.tokenizer_max_length}")  # 48\n'
        ),
        md(
            "### 2. Architecture: VLM + Action Expert\n\n"
            "pi0 uses a **mixture of experts** architecture:\n"
            "- PaliGemma VLM (SigLIP + Gemma 2B) — handles vision + language\n"
            "- Action expert (Gemma 300M) — handles action generation\n\n"
            "The action expert is a separate transformer with its own weights. "
            "It receives the VLM's processed observation embeddings and "
            "generates continuous action vectors."
        ),
        code(
            "# pi0 Architecture (conceptual)\n"
            'print("pi0 Architecture:")\n'
            'print("┌─────────────────────────────────────────────┐")\n'
            'print("│  PaliGemma VLM (SigLIP + Gemma 2B)          │")\n'
            'print("│  ┌─────────┐  ┌──────────────────────┐      │")\n'
            'print("│  │ SigLIP   │  │ Gemma 2B Backbone    │      │")\n'
            'print("│  │ (Vision) │  │ (Language + Fusion)  │      │")\n'
            'print("│  └────┬─────┘  └──────────┬───────────┘      │")\n'
            'print("│       │                   │                  │")\n'
            'print("│       └───────┬───────────┘                  │")\n'
            'print("│               ▼                              │")\n'
            'print("│  ┌──────────────────────────────────────┐    │")\n'
            'print("│  │  Action Expert (Gemma 300M)           │    │")\n'
            'print("│  │  • action_in_proj  (action_dim→width) │    │")\n'
            'print("│  │  • action_out_proj (width→action_dim) │    │")\n'
            'print("│  │  • Flow matching timestep MLP         │    │")\n'
            'print("│  │  → Continuous action chunk (50 × 7)   │    │")\n'
            'print("│  └──────────────────────────────────────┘    │")\n'
            'print("└─────────────────────────────────────────────┘")\n'
        ),
        md(
            "### 3. Flow Matching: how actions are generated\n\n"
            "Flow matching learns a continuous transformation (flow) from a simple "
            "distribution (e.g., Gaussian) to the action distribution.\n\n"
            "During training: given a real action a₁, sample noise a₀ ~ N(0,I), "
            "interpolate a_t = (1-t)·a₀ + t·a₁, train model to predict velocity da/dt.\n\n"
            "During inference: sample a₀ ~ N(0,I), integrate the learned velocity "
            "field to get a₁ (the action chunk)."
        ),
        code(
            "# Flow Matching vs Diffusion\n"
            'print("Flow Matching (pi0):")\n'
            'print("  - Learns a vector field v(t, x) that maps noise → data")\n'
            'print("  - Training: predict velocity da/dt at interpolated points")\n'
            'print("  - Inference: ODE integration (Euler/RK4), typically 10 steps")\n'
            'print("  - Deterministic or stochastic depending on noise schedule")\n'
            "\n"
            'print("Diffusion (Diffusion Policy):")\n'
            'print("  - Learns to predict noise ε added at timestep t")\n'
            'print("  - Training: predict noise from noisy action")\n'
            'print("  - Inference: iterative denoising, typically 50-100 steps")\n'
            'print("  - Stochastic by design")\n'
            "\n"
            'print("Key difference: Flow matching typically needs fewer inference steps.")\n'
        ),
        md(
            "### 4. The action expert — what it actually does\n\n"
            "The action expert is a Gemma transformer specialized for action generation. "
            "Key layers from the leRobot source:"
        ),
        code(
            "# From modeling_pi0.py:\n"
            "# self.action_in_proj = nn.Linear(max_action_dim, action_expert.width)\n"
            "# self.action_out_proj = nn.Linear(action_expert.width, max_action_dim)\n"
            "# self.state_proj = nn.Linear(max_state_dim, action_expert.width)\n"
            "# self.action_time_mlp_in = nn.Linear(2 * width, width)\n"
            "# self.action_time_mlp_out = nn.Linear(width, width)\n"
            "\n"
            "# The action expert takes:\n"
            "#   - The noisy action chunk (flow matching timestep)\n"
            "#   - The VLM's fused observation embeddings\n"
            "#   - The robot's proprioceptive state\n"
            "# And outputs continuous action predictions\n"
            "\n"
            'print("Action Expert Data Flow:")\n'
            'print("  state → state_proj → [B, width]")\n'
            'print("  action → action_in_proj → [B, chunk, width]")\n'
            'print("  time → action_time_mlp → [B, chunk, width]")\n'
            'print("  All combined → Gemma expert → action_out_proj → [B, chunk, action_dim]")\n'
        ),
        md(
            "### 5. train_expert_only: fine-tuning strategy\n\n"
            "pi0 supports freezing the VLM backbone and training only the action expert. "
            "This preserves internet-scale visual/language knowledge while adapting actions."
        ),
        code(
            '# Key config flag\n'
            'print(f"train_expert_only: {cfg.train_expert_only}")\n'
            "\n"
            'print("When train_expert_only=True:")\n'
            'print("  ✓ Action expert weights are updated")\n'
            'print("  ✗ VLM backbone (SigLIP + Gemma 2B) is frozen")\n'
            'print("  → Preserves general knowledge, only adapts actions")\n'
            'print("  → Similar philosophy to LoRA but structural rather than low-rank")\n'
        ),
        md(
            "### In Short\n\n"
            "pi0 keeps actions **continuous** and uses an **explicit action expert** — "
            "a separate transformer specialized for generating smooth action trajectories "
            "via flow matching. The action expert is a dedicated module built for control, "
            "not a repurposed language head."
        ),
    ],
)

write_nb(
    "notebooks/part2/06_smolvla.ipynb",
    [
        md(
            "# Part 2: How Real VLAs Represent Actions\n\n"
            "## Notebook 6 — SmolVLA: Cross-Attention Action Expert\n\n"
            "SmolVLA (HuggingFace, 2025) is a lightweight VLA built on SmolVLM2 (500M). "
            "Unlike pi0 which passes action tokens directly into the transformer, "
            "SmolVLA uses **cross-attention** between the VLM latents and the action expert."
        ),
        md(
            "### 1. Load SmolVLA configuration\n\n"
            "SmolVLA balances size and performance. The action expert is configurable "
            "in width and uses cross-attention to attend to VLM outputs."
        ),
        code(
            "from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig\n\n"
            "cfg = SmolVLAConfig()\n"
            'print(f"Policy type: SmolVLA (Lightweight VLA)")\n'
            'print(f"Chunk size:          {cfg.chunk_size}")  # 50\n'
            'print(f"Action steps:        {cfg.n_action_steps}")  # 50\n'
            'print(f"Max action dim:      {cfg.max_action_dim}")  # 32\n'
            'print(f"Expert width mult:   {cfg.expert_width_multiplier}")  # 0.75\n'
            'print(f"Attention mode:      {cfg.attention_mode}")  # cross_attn\n'
            'print(f"Train expert only:   {cfg.train_expert_only}")  # True\n'
            'print(f"Num expert layers:   {cfg.num_expert_layers}")  # -1 (all)\n'
        ),
        md(
            "### 2. Cross-attention vs direct concatenation\n\n"
            "pi0: action tokens go into the main transformer alongside image/text tokens.\n"
            "SmolVLA: action expert is a SEPARATE transformer that cross-attends to the "
            "VLM's output latents.\n\n"
            "This decoupling means the VLM doesn't need to be modified for action generation. "
            "The action expert is a bolt-on module."
        ),
        code(
            "# Architecture comparison\n"
            'print("pi0 Architecture:")\n'
            'print("  [Image Tokens] [Text Tokens] [State] [Action Tokens]")\n'
            'print("       ↓              ↓          ↓         ↓")\n'
            'print("  ┌──────────────── VLM + Expert ────────────────┐")\n'
            'print("  │  All tokens processed together               │")\n'
            'print("  │  Action expert is a subset of layers         │")\n'
            'print("  └──────────────────────────────────────────────┘")\n'
            "\n"
            'print("\\nSmolVLA Architecture:")\n'
            'print("  [Image Tokens] [Text Tokens] [State]")\n'
            'print("       ↓              ↓          ↓")\n'
            'print("  ┌──────────────── VLM ───────────┐")\n'
            'print("  │  Processed into latents        │──→ VLM output\n'
            'print("  └────────────────────────────────┘")\n'
            'print("                    ↓")\n'
            'print("  ┌── Action Expert (cross_attn) ──┐")\n'
            'print("  │  Cross-attends to VLM latents  │")\n'
            'print("  │  action_in_proj / out_proj     │──→ Actions\n'
            'print("  └────────────────────────────────┘")\n'
        ),
        md(
            "### 3. Action representation: continuous regression\n\n"
            "Like all models in Part 2, SmolVLA outputs continuous actions. "
            "The action_in_proj/action_out_proj MLPs project to/from the "
            "expert's hidden dimension."
        ),
        code(
            "# From modeling_smolvla.py:\n"
            "# self.action_in_proj = nn.Linear(max_action_dim, expert_hidden_size)\n"
            "# self.action_out_proj = nn.Linear(expert_hidden_size, max_action_dim)\n"
            "\n"
            "# Forward pass:\n"
            "# 1. VLM processes images + text → latent representations\n"
            "# 2. Action expert cross-attends to VLM latents\n"
            "# 3. action_in_proj projects noisy actions → expert\n"
            "# 4. Expert transformer processes → action predictions\n"
            "# 5. action_out_proj → continuous action chunk (50 × 7)\n"
            "\n"
            'print("SmolVLA Action Path:")\n'
            'print("  noise → action_in_proj → [B, 50, expert_hidden]")\n'
            'print("  VLM latents → cross_attn(query=action, key/value=VLM_output)")\n'
            'print("  expert hidden → action_out_proj → [B, 50, 7]")\n'
        ),
        md(
            "### 4. Why SmolVLA is efficient\n\n"
            "- SmolVLM2-500M backbone (vs pi0's PaliGemma 2B)\n"
            "- `expert_width_multiplier=0.75` reduces expert width\n"
            "- `train_expert_only=True` by default — only ~300M trainable\n"
            "- Runs on consumer GPUs at 30 Hz (RTX 3090)\n"
        ),
        code(
            '# Efficiency comparison\n'
            'print("Model size comparison (for inference):")\n'
            'print("  pi0:         ~3B params (PaliGemma 2B + Gemma 300M expert)")\n'
            'print("  SmolVLA:     ~500M params (SmolVLM2 + shrunk expert)")\n'
            'print("  pi0-FAST:    ~3B params (PaliGemma 2B, no separate expert)")\n'
            "\n"
            'print("Inference speed (approx):")\n'
            'print("  SmolVLA:     30 Hz (RTX 3090)")\n'
            'print("  pi0:         10-25 Hz (RTX 4090 / A100)")\n'
            'print("  pi0-FAST:    ~5 Hz (autoregressive decoding)")\n'
        ),
        md(
            "### In Summary\n\n"
            "SmolVLA uses cross-attention to decouple the VLM from the action expert. "
            "Actions are continuous, the expert is a separate transformer, and efficiency "
            "comes from the small backbone (500M) and reduced expert width. "
            "No tokenization anywhere in the pipeline."
        ),
    ],
)

write_nb(
    "notebooks/part2/07_pi05.ipynb",
    [
        md(
            "# Part 2: How Real VLAs Represent Actions\n\n"
            "## Notebook 7 — π₀.₅ (pi0.5): Flow Matching + adaRMS\n\n"
            "pi0.5 is Physical Intelligence's generalization-focused successor to pi0. "
            "In the leRobot implementation, it uses **flow matching with adaRMS conditioning** "
            "— an improved version of pi0's architecture. Notably, it does NOT use FAST "
            "tokenization in leRobot."
        ),
        md(
            "### 1. Load pi0.5 configuration\n\n"
            "Compare with pi0 — the config is very similar but with adaRMS conditioning "
            "for better generalization across heterogeneous data."
        ),
        code(
            "from lerobot.policies.pi05.configuration_pi05 import PI05Config\n\n"
            "cfg = PI05Config()\n"
            'print(f"Policy type: pi0.5 (Flow Matching VLA + adaRMS)")\n'
            'print(f"Action expert:       {cfg.action_expert_variant}")  # gemma_300m\n'
            'print(f"Chunk size:          {cfg.chunk_size}")  # 50\n'
            'print(f"Action steps:        {cfg.n_action_steps}")  # 50\n'
            'print(f"Max action dim:      {cfg.max_action_dim}")  # 32\n'
            'print(f"Train expert only:   {cfg.train_expert_only}")  # False\n'
        ),
        md(
            "### 2. pi0 vs pi0.5: what changed?\n\n"
            "pi0.5 adds **adaRMS (Adaptive Root Mean Square)** conditioning. "
            "This is a normalization technique that helps the model handle "
            "heterogeneous data sources (different robots, cameras, environments).\n\n"
            "Critically: in leRobot v0.6.0, pi0.5 does NOT include FAST tokenization. "
            "It's a pure flow matching model with an action expert."
        ),
        code(
            "# Side-by-side config comparison\n"
            'print("pi0 config keys with action/token relevance:")\n'
            'print("  action_expert_variant: gemma_300m")\n'
            'print("  chunk_size: 50")\n'
            'print("  n_action_steps: 50")\n'
            'print("  max_action_dim: 32")\n'
            'print("  train_expert_only: False")\n'
            "\n"
            'print("\\npi0-FAST config keys (for contrast):")\n'
            'print("  action_tokenizer_name: lerobot/fast-action-tokenizer")\n'
            'print("  max_action_tokens: 256")\n'
            'print("  fast_skip_tokens: 128")\n'
            'print("  validate_action_token_prefix: True")\n'
            "\n"
            'print("\\npi05 does NOT have action_tokenizer_name or max_action_tokens.")\n'
            'print("→ pi0.5 uses flow matching, NOT FAST tokenization (in leRobot).")\n'
        ),
        md(
            "### 3. adaRMS: adaptive conditioning for generalization\n\n"
            "adaRMS normalizes activations based on RMS statistics, conditioned on "
            "which data source (robot embodiment) the sample comes from. This lets "
            "the model handle diverse robot morphologies without them interfering."
        ),
        code(
            "# adaRMS in pi0.5:\n"
            "# adarms_cond_dim is set in the Gemma expert config\n"
            "# It conditions each layer's normalization on the embodiment identity\n"
            "\n"
            'print("adaRMS Conditioning:")\n'
            'print("  Problem: Different robots have different action scales")\n'
            'print("    - Franka: joints in radians, range ~[-π, π]")\n'
            'print("    - UR5: joints in radians, different range")\n'
            'print("    - Mobile base: position deltas in meters")\n'
            "\n"
            'print("  Solution: adaRMS conditions each layer on robot identity")\n'
            'print("    - Learns per-embodiment scaling/shifting parameters")\n'
            'print("    - Shared model weights + per-robot conditioning")\n'
            'print("    → One model handles many robots without conflict")\n'
        ),
        md(
            "### 4. Open-world generalization\n\n"
            "pi0.5's main claim: performs manipulation tasks in UNSEEN environments. "
            "The training mixture spans 10+ robot embodiments + web vision-language data. "
            "Co-training (not sequential fine-tuning) preserves general knowledge."
        ),
        code(
            '# pi0.5 Training Data (approximate, from PI blog)\n'
            'print("pi0.5 Training Data:")\n'
            'print("  Robot data: 10+ embodiments")\n'
            'print("    - Franka Panda, UR5, ALOHA bi-manual")\n'
            'print("    - Mobile manipulators, humanoid upper-body")\n'
            'print("  Web data: vision-language (image-caption, VQA)")\n'
            'print("  Training: CO-TRAINING (mixed, not sequential)")\n'
            "\n"
            'print("Key claim: Co-training preserves web knowledge")\n'
            'print("  while adding robot control capability.")\n'
            'print("  Sequential fine-tuning tends to forget general knowledge.")\n'
        ),
        md(
            "### 5. Where does pi0.5 fit in the action tokenization story?\n\n"
            "Physical Intelligence released pi0-FAST (autoregressive FAST tokens) alongside "
            "pi0 (flow matching). Then they released pi0.5, which in leRobot's implementation "
            "goes back to **pure flow matching with an action expert** — no FAST tokens.\n\n"
            "This tells us: even the creators of FAST recognized that continuous flow matching "
            "with a dedicated action expert is still the superior approach for dexterous control. "
            "Tokenization gives training speed but loses action precision."
        ),
        code(
            '# The pendulum swing\n'
            'print("Physical Intelligence Action Representation Evolution:")\n'
            'print("  pi0 (2024):      Flow matching + action expert")\n'
            'print("  pi0-FAST (2025): FAST tokens (autoregressive, 5× faster training)")\n'
            'print("  pi0.5 (2025-26): BACK to flow matching + action expert + adaRMS")\n'
            "\n"
            'print("\\nWhy go back? Possible reasons:")\n'
            'print("  1. Autoregressive inference is SLOW (many tokens per chunk)")\n'
            'print("  2. Tokenization quantization error loses fine dexterity")\n'
            'print("  3. Flow matching produces smoother trajectories")\n'
            'print("  4. Action expert specializes better than shared backbone")\n'
        ),
        md(
            "### Where This Leaves Us\n\n"
            "pi0.5 completes the arc: from continuous (pi0) → tokenized (pi0-FAST) → "
            "back to continuous (pi0.5). The action expert persists throughout. "
            "Tokenization was tried but the field continues to favor continuous action representations."
        ),
    ],
)

# ── Part 3 ──────────────────────────────────────────────────────────────

write_nb(
    "notebooks/part3/08_fast_pipeline.ipynb",
    [
        md(
            "# Part 3: pi0-FAST — The Tokenization Experiment\n\n"
            "## Notebook 8 — The FAST Pipeline Step by Step\n\n"
            "FAST (Frequency-space Action Sequence Tokenization) converts continuous "
            "action chunks into discrete tokens via:\n\n"
            "1. **Normalize** action chunk to [-1, 1]\n"
            "2. **DCT** (Discrete Cosine Transform) → frequency domain\n"
            "3. **Quantize** coefficients to integers\n"
            "4. **Flatten** into 1D, low-frequency first\n"
            "5. **BPE** compress into dense tokens\n\n"
            "We load the pre-trained FAST tokenizer and walk through each step."
        ),
        md("### 1. Load the FAST tokenizer"),
        code(
            "from transformers import AutoProcessor\n\n"
            "# Load the universal FAST+ tokenizer trained on 1M action sequences\n"
            'tokenizer = AutoProcessor.from_pretrained(\n'
            '    "physical-intelligence/fast", trust_remote_code=True\n'
            ")\n"
            'print(f"FAST tokenizer loaded")\n'
        ),
        md("### 2. Create a sample action chunk"),
        code(
            "import numpy as np\n\n"
            "time_horizon = 50  # 1 second at 50 Hz\n"
            "action_dim = 7    # x, y, z, roll, pitch, yaw, gripper\n"
            "\n"
            "# Sample action: sinusoidal joint movement\n"
            "t = np.linspace(0, 1, time_horizon)\n"
            "actions = np.zeros((time_horizon, action_dim))\n"
            "actions[:, 0] = 0.5 * np.sin(2 * np.pi * 1.5 * t)   # x: sinusoidal\n"
            "actions[:, 1] = 0.3 * np.cos(2 * np.pi * 2.0 * t)   # y: cosine\n"
            "actions[:, 2] = 0.1 * t                              # z: linear\n"
            "actions[:, 3] = 0.2 * np.sin(2 * np.pi * 0.5 * t)   # roll\n"
            "actions[:, 4] = -0.1 * t                             # pitch: linear\n"
            "actions[:, 5] = 0.05 * np.sin(2 * np.pi * 3.0 * t)  # yaw: fast\n"
            "actions[:, 6] = np.where(t > 0.5, 1.0, 0.0)         # gripper: step\n"
            "\n"
            'print(f"Action chunk shape: {actions.shape}")\n'
            'print(f"Value range: [{actions.min():.3f}, {actions.max():.3f}]")\n'
        ),
        md("### 3. Visualize the action chunk"),
        code(
            "import matplotlib.pyplot as plt\n\n"
            'dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]\n'
            "fig, axes = plt.subplots(4, 2, figsize=(12, 8))\n"
            "axes = axes.flatten()\n"
            "for i in range(action_dim):\n"
            "    axes[i].plot(t, actions[:, i])\n"
            "    axes[i].set_title(f'{dim_names[i]}')\n"
            "    axes[i].set_xlabel('Time (s)')\n"
            "    axes[i].grid(True, alpha=0.3)\n"
            "axes[-1].axis('off')\n"
            "fig.suptitle('Sample Action Chunk (50 timesteps × 7 dims)', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md(
            "### 4. Tokenize with FAST\n\n"
            "The FAST tokenizer compresses this 50×7 = 350-value chunk "
            "into a small number of discrete tokens."
        ),
        code(
            "fast_tokens = tokenizer(padding=False)[:20]\n"
            'print(f"Action values: {actions.shape} = {actions.size} floats")\n'
            'print(f"FAST tokens:   {len(fast_tokens)} tokens")\n'
            'print(f"Compression:   {actions.size / len(fast_tokens):.1f}×")\n'
            "\n"
            "# Show first few tokens\n"
            'print(f"\\nFirst 10 tokens: {fast_tokens[:10]}")\n'
        ),
        md(
            "### 5. DCT step: time domain → frequency domain\n\n"
            "The Discrete Cosine Transform (same as JPEG) converts the action "
            "signal to frequency domain. Smooth motions concentrate energy "
            "in low-frequency coefficients — most high-frequency coefficients "
            "are near zero (sparse)."
        ),
        code(
            "from scipy.fftpack import dct\n\n"
            "# Apply DCT along time axis (axis=0)\n"
            "dct_coeffs = dct(actions, axis=0, norm='ortho')\n"
            'print(f"DCT coefficient shape: {dct_coeffs.shape}")\n'
            'print(f"Non-zero coeffs: {np.count_nonzero(np.abs(dct_coeffs) > 1e-6)}")\n'
            "\n"
            "# Energy concentration: how much energy in first K coefficients?\n"
            "energy = np.sum(dct_coeffs ** 2, axis=0)\n"
            "for k in [5, 10, 20]:\n"
            "    pct = 100 * np.sum(dct_coeffs[:k] ** 2, axis=0) / energy\n"
            '    print(f"  First {k:2d} coeffs capture: {pct.mean():.1f}% of energy")\n'
        ),
        md("### 6. Quantize: float → int"),
        code(
            "# Quantization step (scaled and rounded)\n"
            "scale = 1000  # FAST uses scaling to preserve precision\n"
            "quantized = np.round(dct_coeffs * scale).astype(np.int32)\n"
            'print(f"Quantized DCT shape: {quantized.shape}")\n'
            'print(f"Value range: [{quantized.min()}, {quantized.max()}]")\n'
        ),
        md("### 7. BPE: compress sparse coefficients into dense tokens"),
        code(
            "# After flattening: 50*7=350 coefficients → BPE merges → ~30-60 tokens\n"
            "# The BPE tokenizer was trained on DCT coefficients from 1M action sequences\n"
            "# It learns which coefficient patterns commonly appear together\n"
            "\n"
            'print("BPE compression:")\n'
            'print("  Input:  350 quantized DCT coefficients")\n'
            'print("  Output: ~30-60 BPE tokens")\n'
            'print("  Ratio:  ~6-12× compression")\n'
            "\n"
            "# Compare: if we used RT-1 style binning\n"
            "tokens_binned = time_horizon * action_dim  # 350\n"
            "tokens_fast = len(fast_tokens)  # ~30-60\n"
            'print(f"\\nRT-1/RT-2 binning: {tokens_binned} tokens")\n'
            'print(f"FAST:             {tokens_fast} tokens")\n'
            'print(f"Compression:      {tokens_binned / tokens_fast:.1f}×")\n'
        ),
        md("### 8. Round-trip: decode back to actions"),
        code(
            "# FAST is fully invertible\n"
            "decoded = tokenizer.decode(fast_tokens)\n"
            "# decoded should approximately match the original actions\n"
            'print(f"Decoded shape: {decoded.shape}")\n'
            "\n"
            "# Reconstruction error\n"
            "if decoded.shape == actions.shape:\n"
            "    mse = np.mean((decoded - actions) ** 2)\n"
            '    print(f"Reconstruction MSE: {mse:.6f}")\n'
            "else:\n"
            '    print(f"Shape mismatch — tokenizer may have different horizon/dim")\n'
        ),
        md(
            "### To Summarize\n\n"
            "FAST compresses action chunks via DCT + BPE, achieving 10× fewer tokens "
            "than naive per-dimension binning. The compression works because smooth "
            "motions are sparse in the frequency domain. In the next notebook, we'll "
            "see how pi0-FAST uses these tokens for autoregressive training."
        ),
    ],
)

write_nb(
    "notebooks/part3/09_pi0fast_inference.ipynb",
    [
        md(
            "# Part 3: pi0-FAST — The Tokenization Experiment\n\n"
            "## Notebook 9 — pi0-FAST: Autoregressive Action Generation\n\n"
            "pi0-FAST replaces pi0's flow matching action head with autoregressive "
            "FAST token prediction. The VLM backbone (SigLIP + Gemma 2B) directly "
            "outputs discrete action tokens — no separate action expert."
        ),
        md("### 1. Load pi0-FAST configuration"),
        code(
            "from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig\n\n"
            "cfg = PI0FastConfig()\n"
            'print(f"Policy type: pi0-FAST (Autoregressive VLA)")\n'
            'print(f"Action tokenizer:    {cfg.action_tokenizer_name}")\n'
            'print(f"Max action tokens:   {cfg.max_action_tokens}")  # 256\n'
            'print(f"Fast skip tokens:    {cfg.fast_skip_tokens}")  # 128\n'
            'print(f"Chunk size:          {cfg.chunk_size}")  # 50\n'
            'print(f"Action steps:        {cfg.n_action_steps}")  # 50\n'
            'print(f"Action expert:       {cfg.action_expert_variant}")  # gemma_300m\n'
        ),
        md(
            "### 2. Architecture: no action expert\n\n"
            "Unlike pi0 and pi0.5, pi0-FAST does NOT use a separate action expert. "
            "The Gemma 2B backbone directly predicts action tokens. This is the key "
            "architectural difference."
        ),
        code(
            "# pi0-FAST Architecture (conceptual)\n"
            'print("pi0-FAST Architecture:")\n'
            'print("┌─────────────────────────────────────────┐")\n'
            'print("│  PaliGemma VLM (SigLIP + Gemma 2B)      │")\n'
            'print("│  ┌─────────┐  ┌──────────────────┐      │")\n'
            'print("│  │ SigLIP   │  │ Gemma 2B         │      │")\n'
            'print("│  │ (Vision) │  │ (Everything!)    │      │")\n'
            'print("│  └────┬─────┘  └────────┬─────────┘      │")\n'
            'print("│       └───────┬─────────┘                │")\n'
            'print("│               ▼                          │")\n'
            'print("│  ┌──────────────────────────────────┐    │")\n'
            'print("│  │  Autoregressive Token Decoding    │    │")\n'
            'print("│  │  Predict FAST token T₁, T₂, ...  │    │")\n'
            'print("│  │  → decode_actions_with_fast()     │    │")\n'
            'print("│  │  → Continuous action chunk        │    │")\n'
            'print("│  └──────────────────────────────────┘    │")\n'
            'print("└─────────────────────────────────────────┘")\n'
            "\n"
            'print("\\nKey difference from pi0:")\n'
            'print("  pi0:      VLM + Action Expert (Gemma 300M)")\n'
            'print("  pi0-FAST: VLM only — no action expert")\n'
        ),
        md(
            "### 3. Forward pass: teacher forcing with FAST tokens\n\n"
            "During training, pi0-FAST sees the ground-truth FAST tokens (teacher forcing). "
            "The loss is standard cross-entropy over the action token vocabulary."
        ),
        code(
            "# Training flow (from modeling_pi0_fast.py):\n"
            "# 1. Encode action chunk → FAST tokens (pre-computed in dataset)\n"
            "# 2. Feed [images, text, state, action_tokens[:t]] to model\n"
            "# 3. Model predicts action_tokens[t+1]\n"
            "# 4. Cross-entropy loss over token vocabulary\n"
            "\n"
            'print("pi0-FAST Training:")\n'
            'print("  Input:  [images, text, state, FAST_tokens[:-1]]")\n'
            'print("  Target: FAST_tokens")\n'
            'print("  Loss:   CrossEntropy over FAST token vocabulary")\n'
            'print("  Same objective as language modeling!")\n'
        ),
        md(
            "### 4. Inference: autoregressive token generation\n\n"
            "At inference, pi0-FAST generates action tokens one at a time, "
            "conditioning on previously generated tokens. KV-caching avoids "
            "recomputing the image/text prefix at each step."
        ),
        code(
            "# Inference flow:\n"
            "# 1. Encode images + text (done once, cached via KV-cache)\n"
            "# 2. Generate token T₀ (BOS)\n"
            "# 3. For t = 1 to max_action_tokens:\n"
            "#       predict T_t from [images, text, T₀..T_{t-1}]\n"
            "#       if T_t == EOS: break\n"
            "# 4. Decode tokens → continuous actions via inverse DCT\n"
            "\n"
            'print("pi0-FAST Inference:")\n'
            'print("  KV-caching: images/text prefix computed once")\n'
            'print("  Max decoding steps: 256")\n'
            'print("  Typical tokens per chunk: 30-60")\n'
            'print("  Inference speed: ~5 Hz (autoregressive bottleneck)")\n'
        ),
        md(
            "### 5. decode_actions_with_fast: tokens → actions\n\n"
            "The inverse pipeline: BPE decode → unflatten → inverse DCT → continuous actions."
        ),
        code(
            "# decode_actions_with_fast (from modeling_pi0_fast.py):\n"
            "# 1. For each generated FAST token:\n"
            "#     - BPE decode → list of quantized DCT coefficients\n"
            "# 2. Reshape coefficients → (time_horizon, action_dim) matrix\n"
            "# 3. Unscale: coeffs / scale\n"
            "# 4. Inverse DCT along time axis: idct(coeffs, axis=0, norm='ortho')\n"
            "# 5. Result: continuous action chunk (50 × 7)\n"
            "\n"
            'print("FAST Decode Pipeline:")\n'
            'print("  Action tokens → BPE decode → DCT coeffs")\n'
            'print("  DCT coeffs → unscale → iDCT → action chunk")\n'
            'print("  Fully invertible (lossless in theory, near-lossless in practice)")\n'
        ),
        md(
            "### 6. The inference speed problem\n\n"
            "pi0-FAST's major weakness: autoregressive decoding is SLOW. "
            "Generating 30-60 tokens sequentially takes much longer than "
            "pi0's single-pass flow matching. This is why pi0.5 went back."
        ),
        code(
            '# Speed comparison\n'
            'print("Inference speed comparison:")\n'
            'print("  pi0 (flow matching):    ~10-25 Hz (single pass + ODE steps)")\n'
            'print("  pi0-FAST (tokens):      ~5 Hz    (sequential token decoding)")\n'
            'print("  pi0.5 (flow matching):  ~10-25 Hz (back to flow matching)")\n'
            "\n"
            'print("The trade-off:")\n'
            'print("  pi0-FAST:  5× faster TRAINING, but slower INFERENCE")\n'
            'print("  pi0/pi0.5: slower training, but faster inference")\n'
            'print("  For real robots: inference speed matters more")\n'
        ),
        md(
            "### The Trade-Off\n\n"
            "pi0-FAST achieves 5× faster training by using the same cross-entropy "
            "objective as language models. But autoregressive token generation is "
            "slow at inference time — a fundamental limitation for real-time robot control. "
            "In the final notebook, we compare all approaches side by side."
        ),
    ],
)

write_nb(
    "notebooks/part3/10_comparison.ipynb",
    [
        md(
            "# Part 3: Conclusion\n\n"
            "## Notebook 10 — Complete Comparison\n\n"
            "We've now inspected six models available in leRobot v0.6.0. "
            "Here's the full comparison of how each represents robot actions."
        ),
        md("### 1. Action Representation Comparison"),
        code(
            'print("╔═══════════════╦══════════════════╦══════════════════╦══════════════╗")\n'
            'print("║ Model         ║ Action Format    ║ Action Expert?   ║ Tokenization?║")\n'
            'print("╠═══════════════╬══════════════════╬══════════════════╬══════════════╣")\n'
            'print("║ ACT           ║ Continuous (CVAE)║ No               ║ None         ║")\n'
            'print("║ Diffusion Pol ║ Continuous (DDPM)║ No               ║ None         ║")\n'
            'print("║ pi0           ║ Flow matching    ║ Yes (Gemma 300M) ║ None         ║")\n'
            'print("║ SmolVLA       ║ Cross-attn regr. ║ Yes (cross_attn) ║ None         ║")\n'
            'print("║ pi0.5         ║ Flow matching    ║ Yes (Gemma 300M) ║ None         ║")\n'
            'print("║ pi0-FAST      ║ FAST tokens      ║ No (VLM direct)  ║ DCT + BPE    ║")\n'
            'print("╚═══════════════╩══════════════════╩══════════════════╩══════════════╝")\n'
        ),
        md("### 2. Training vs Inference Trade-off"),
        code(
            'print("Training Efficiency:")\n'
            'print("  pi0-FAST:  5× faster   (cross-entropy, same as LM)")\n'
            'print("  pi0/pi0.5: 1× baseline (flow matching, ODE integration)")\n'
            "\n"
            'print("Inference Speed:")\n'
            'print("  pi0/pi0.5:  10-25 Hz (flow matching, few ODE steps)")\n'
            'print("  SmolVLA:    30 Hz    (small backbone, efficient expert)")\n'
            'print("  ACT:        30 Hz    (single forward pass)")\n'
            'print("  Diffusion:  10-30 Hz (T denoising steps)")\n'
            'print("  pi0-FAST:   ~5 Hz    (sequential token decoding)")\n'
        ),
        md("### 3. The Evolution of Action Representations in VLAs"),
        code(
            '# Timeline (conceptual)\n'
            'print("2022-2023: Early Discretization")\n'
            'print("  RT-1, RT-2, OpenVLA: per-dimension binning")\n'
            'print("  → Simple but lossy, poor for dexterous tasks")\n'
            "\n"
            'print("\\n2023: Continuous Representations")\n'
            'print("  ACT: action chunks via CVAE")\n'
            'print("  Diffusion Policy: iterative denoising")\n'
            'print("  → Better dexterity, but no language conditioning")\n'
            "\n"
            'print("\\n2024-2025: VLAs with Continuous Actions")\n'
            'print("  pi0: flow matching + action expert")\n'
            'print("  SmolVLA: cross-attention action expert")\n'
            'print("  → Language + vision + continuous actions")\n'
            "\n"
            'print("\\n2025: Tokenization Experiment")\n'
            'print("  pi0-FAST: DCT+BPE tokens, autoregressive")\n'
            'print("  → 5× faster training, but slow inference")\n'
            "\n"
            'print("\\n2025-2026: Back to Continuous")\n'
            'print("  pi0.5: flow matching + action expert (again)")\n'
            'print("  → Tokenization promising but not ready for production")\n'
        ),
        md("### 4. Why Physical Intelligence Went Back to Action Expert"),
        code(
            'print("Possible reasons pi0.5 dropped pure tokenization:")\n'
            'print("  1. Autoregressive inference too slow for real-time control")\n'
            'print("  2. Quantization error from DCT+BPE loses fine dexterity")\n'
            'print("  3. Flow matching produces smoother action trajectories")\n'
            'print("  4. Action expert specializes for control; shared backbone doesn\'t")\n'
            'print("  5. KV-caching helps but doesn\'t solve sequential bottleneck")\n'
        ),
        md(
            "### 5. Open Questions\n\n"
            "The field is still evolving. Recent work (2025-2026) explores:\n\n"
            "- **OAT** (Ordered Action Tokenization): learned tokenizers with causal ordering\n"
            "- **QueST**: VQ-VAE style action compression\n"
            "- **BEAST**: B-spline encoded action sequences\n\n"
            "The debate — continuous vs discrete actions — is not settled. "
            "But the trend in production VLAs is clear: continuous with action experts."
        ),
        md(
            "### 6. Final Summary\n\n"
            "| Approach | Models | Pros | Cons |\n"
            "|----------|--------|------|------|\n"
            "| Continuous + CVAE | ACT | Fast, simple | No language conditioning |\n"
            "| Continuous + diffusion | Diffusion Policy | Smooth, multimodal | Slow inference |\n"
            "| Continuous + flow matching | pi0, pi0.5 | Dexterous, fast inference | Slow training |\n"
            "| Continuous + cross-attn | SmolVLA | Efficient, modular | Less dexterity |\n"
            "| Discrete tokens (FAST) | pi0-FAST | Fast training | Slow inference, quantization |\n"
            "\n"
            "**Bottom line**: 5 out of 6 models in leRobot use continuous actions "
            "with dedicated action experts. Tokenization remains an active research "
            "direction but hasn't replaced continuous methods in production VLAs."
        ),
    ],
)

print("All 10 notebooks created successfully!")
