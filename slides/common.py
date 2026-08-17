"""Shared styles, colors, and utilities for action tokenization manim slides."""

from manim import *

# ── Color Palette ─────────────────────────────────────────────────────
BG_COLOR = "#1a1a2e"
TITLE_COLOR = "#e94560"
ACCENT_COLOR = "#0f3460"
HIGHLIGHT_COLOR = "#16c79a"
TEXT_COLOR = "#eeeeee"
CODE_BG = "#16213e"

# Model-specific colors
MODEL_COLORS = {
    "act": "#FF6B6B",           # coral red
    "diffusion": "#4ECDC4",     # teal
    "pi0": "#45B7D1",           # sky blue
    "pi0_fast": "#96CEB4",      # sage green
    "pi05": "#FFEAA7",          # soft yellow
    "smolvla": "#DDA0DD",       # plum
    "fast": "#F39C12",          # orange
}

# ── Text Styles ────────────────────────────────────────────────────────
def title_text(text: str, **kwargs) -> Text:
    return Text(text, color=TITLE_COLOR, font="sans-serif", weight=BOLD, **kwargs)

def body_text(text: str, **kwargs) -> Text:
    return Text(text, color=TEXT_COLOR, font="sans-serif", **kwargs)

def code_text(text: str, **kwargs) -> Text:
    return Text(text, color=HIGHLIGHT_COLOR, font="monospace", **kwargs)

def model_label(text: str, model: str) -> Text:
    color = MODEL_COLORS.get(model, TEXT_COLOR)
    return Text(text, color=color, font="sans-serif", weight=BOLD)

# ── Layout Helpers ─────────────────────────────────────────────────────
def slide_bg(scene):
    """Set dark background for the scene."""
    scene.camera.background_color = BG_COLOR

def add_footer(scene, text: str):
    """Add a small footer to the bottom of the slide."""
    footer = Text(text, color="#555555", font_size=18).to_edge(DOWN)
    scene.add(footer)

# ── Common Diagrams ────────────────────────────────────────────────────
def architecture_box(
    label: str,
    color: str = ACCENT_COLOR,
    width: float = 3.0,
    height: float = 1.0,
    **kwargs
) -> VGroup:
    """Create a labeled architecture box."""
    rect = Rectangle(
        width=width, height=height,
        color=color, fill_color=color, fill_opacity=0.3,
        stroke_width=2,
        **kwargs
    )
    text = Text(label, color=TEXT_COLOR, font_size=20).move_to(rect.get_center())
    return VGroup(rect, text)

def arrow_between(start, end, color=TEXT_COLOR):
    """Create an arrow between two mobjects."""
    return Arrow(start.get_bottom(), end.get_top(), color=color, buff=0.2)

def model_comparison_table() -> Table:
    """Return a Table showing all 6 models' action representations."""
    return Table(
        [
            ["Model", "Action Format", "Expert?", "Tokens?"],
            ["ACT", "Continuous (CVAE)", "No", "None"],
            ["Diffusion", "Continuous (DDPM)", "No", "None"],
            ["pi0", "Flow Matching", "Yes", "None"],
            ["SmolVLA", "Cross-Attn", "Yes", "None"],
            ["pi0.5", "Flow Matching", "Yes", "None"],
            ["pi0-FAST", "FAST Tokens", "No", "DCT+BPE"],
        ],
        include_outer_lines=True,
        line_config={"color": TEXT_COLOR},
    )
