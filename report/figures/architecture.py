"""
architecture.py
---------------
Generates a system architecture diagram for AlphaPortfolio and saves it
as report/figures/architecture.png at 300 DPI.

Run:
    python report/figures/architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Output path ────────────────────────────────────────────────────────
OUT_PATH = Path(__file__).parent / "architecture.png"

# ── Colour palette ─────────────────────────────────────────────────────
LAYER_COLORS = {
    1: "#2C7BB6",   # blue  — environment
    2: "#1A9641",   # green — agents
    3: "#D7191C",   # red   — orchestrator
    4: "#756BB1",   # purple — training
}
LAYER_ALPHA   = 0.13   # background band fill alpha
BOX_ALPHA     = 0.90   # component box fill alpha
ARROW_COLOR   = "#444444"
TEXT_COLOR    = "#111111"
LABEL_COLOR   = "#FFFFFF"
BAND_EDGE     = "#CCCCCC"


def _hex_lighten(hex_color: str, factor: float = 0.55) -> str:
    """Blend a hex colour toward white by `factor`."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02X}{g2:02X}{b2:02X}"


def _draw_box(
    ax: plt.Axes,
    cx: float, cy: float,
    w: float, h: float,
    label: str,
    sublabel: str = "",
    color: str = "#2C7BB6",
    fontsize: int = 10,
    subfontsize: int = 8,
    radius: float = 0.04,
) -> None:
    """Draw a rounded rectangle with centred label (and optional sublabel)."""
    face = _hex_lighten(color, 0.30)
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.6,
        edgecolor=color,
        facecolor=face,
        alpha=BOX_ALPHA,
        zorder=3,
    )
    ax.add_patch(box)

    if sublabel:
        ax.text(cx, cy + h * 0.14, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=4)
        ax.text(cx, cy - h * 0.22, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color=color, style="italic", zorder=4)
    else:
        ax.text(cx, cy, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color, zorder=4)


def _draw_band(
    ax: plt.Axes,
    y_center: float,
    height: float,
    color: str,
    layer_num: int,
    layer_title: str,
    x_left: float = 0.02,
    x_right: float = 0.98,
) -> None:
    """Draw a full-width background band for one layer."""
    face = _hex_lighten(color, 0.78)
    band = FancyBboxPatch(
        (x_left, y_center - height / 2),
        x_right - x_left,
        height,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=BAND_EDGE,
        facecolor=face,
        alpha=1.0,
        zorder=1,
    )
    ax.add_patch(band)

    # Layer label on the left margin
    ax.text(
        x_left + 0.012, y_center,
        f"Layer {layer_num}",
        ha="left", va="center",
        fontsize=8, fontweight="bold",
        color=color, rotation=90,
        zorder=2,
    )
    ax.text(
        x_left + 0.038, y_center,
        layer_title,
        ha="left", va="center",
        fontsize=9, fontweight="bold",
        color=color,
        zorder=2,
    )


def _arrow(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    label: str = "",
    color: str = ARROW_COLOR,
    bidirectional: bool = False,
) -> None:
    """Draw an annotated arrow between two points."""
    style = "Simple,tail_width=1.5,head_width=8,head_length=6"
    arrowstyle = "<->" if bidirectional else "->"

    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        xycoords="data", textcoords="data",
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=1.4,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.015, my, label,
                ha="left", va="center", fontsize=7.5,
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.965,
        "AlphaPortfolio — System Architecture",
        ha="center", va="top",
        fontsize=16, fontweight="bold", color="#222222",
    )
    fig.text(
        0.5, 0.942,
        "Multi-Agent Reinforcement Learning for Equity Portfolio Management",
        ha="center", va="top",
        fontsize=10, color="#555555", style="italic",
    )

    # ── Layer geometry  (y centres, top → bottom) ──────────────────────
    #   Layer 4 (Training)     y = 0.80
    #   Layer 3 (Orchestrator) y = 0.575
    #   Layer 2 (Agents)       y = 0.355
    #   Layer 1 (Environment)  y = 0.13
    BAND_H = 0.175
    L_X0, L_X1 = 0.08, 0.98   # band x extents (leaves margin for layer label)

    layers = {
        4: (0.800, LAYER_COLORS[4], "Training Pipeline"),
        3: (0.575, LAYER_COLORS[3], "Orchestrator"),
        2: (0.355, LAYER_COLORS[2], "Specialist Agents"),
        1: (0.130, LAYER_COLORS[1], "Market Environment"),
    }

    for num, (yc, col, title) in layers.items():
        _draw_band(ax, yc, BAND_H, col, num, title, x_left=L_X0, x_right=L_X1)

    # ── Layer 1: Environment ───────────────────────────────────────────
    yc1 = layers[1][0]
    col1 = layers[1][1]
    BOX_H = 0.085

    _draw_box(ax, 0.310, yc1, 0.22, BOX_H,
              "MarketEnv", "Gymnasium  |  252-step episode", col1, fontsize=10)
    _draw_box(ax, 0.620, yc1, 0.22, BOX_H,
              "DataLoader", "yfinance  |  AAPL MSFT GOOGL JPM GLD", col1, fontsize=10)

    # State / obs label
    ax.text(0.875, yc1 + 0.055, "obs  (41,)",
            ha="center", va="center", fontsize=7.5,
            color=col1, fontweight="bold")
    ax.text(0.875, yc1 + 0.010, "7 tech × 5 stocks",
            ha="center", va="center", fontsize=7, color=col1, style="italic")
    ax.text(0.875, yc1 - 0.030, "+ weights + cash",
            ha="center", va="center", fontsize=7, color=col1, style="italic")

    # Arrow between DataLoader → MarketEnv
    _arrow(ax, 0.505, yc1, 0.425, yc1,
           label="OHLCV features\n& prices", color=col1)

    # ── Layer 2: Agents ────────────────────────────────────────────────
    yc2 = layers[2][0]
    col2 = layers[2][1]

    _draw_box(ax, 0.235, yc2, 0.195, BOX_H,
              "MomentumAgent", "γ=0.995  |  ε_end=0.02", col2)
    _draw_box(ax, 0.500, yc2, 0.205, BOX_H,
              "MeanReversionAgent", "γ=0.97  |  ε_end=0.08  |  h=512", col2)
    _draw_box(ax, 0.765, yc2, 0.195, BOX_H,
              "MacroAgent", "γ=0.99  |  ε_end=0.05", col2)

    # Shared DQN note
    ax.text(0.500, yc2 - 0.073,
            "All agents: Double DQN  ·  3-step returns  ·  replay buffer 100 k  ·  hidden 256",
            ha="center", va="center", fontsize=7.5, color=col2, style="italic")

    # ── Layer 3: Orchestrator ──────────────────────────────────────────
    yc3 = layers[3][0]
    col3 = layers[3][1]

    _draw_box(ax, 0.235, yc3, 0.215, BOX_H,
              "UCB Bandit", "context: low/med/high vol\nc = 2.0", col3)
    _draw_box(ax, 0.500, yc3, 0.215, BOX_H,
              "Reward Coordinator", "r_blend = α·r_local + (1−α)·r_global\nα = 0.7", col3)
    _draw_box(ax, 0.765, yc3, 0.215, BOX_H,
              "LLM Regime Detector", "claude-haiku  |  trending /\nmean_reverting / volatile", col3)

    # Internal orchestrator arrows
    _arrow(ax, 0.348, yc3, 0.388, yc3, color=col3)   # bandit → coordinator
    _arrow(ax, 0.613, yc3, 0.653, yc3, color=col3)   # coordinator → LLM

    # ── Layer 4: Training ──────────────────────────────────────────────
    yc4 = layers[4][0]
    col4 = layers[4][1]

    _draw_box(ax, 0.235, yc4, 0.200, BOX_H,
              "Phase 1", "Specialist pre-training\n(solo env, n eps / agent)", col4)
    _draw_box(ax, 0.500, yc4, 0.210, BOX_H,
              "Phase 1.5  Warmup", "Solo in orch. env\nfills replay buffer", col4)
    _draw_box(ax, 0.765, yc4, 0.200, BOX_H,
              "Phase 2", "Full orchestrator training\n(bandit + coord + agents)", col4)

    # Phase arrows (left to right)
    _arrow(ax, 0.340, yc4, 0.390, yc4, color=col4)
    _arrow(ax, 0.608, yc4, 0.660, yc4, color=col4)

    # ── Inter-layer arrows ─────────────────────────────────────────────
    # Layer 1 → Layer 2  (obs up)
    _arrow(ax, 0.310, yc1 + BOX_H / 2, 0.310, yc2 - BOX_H / 2,
           label="obs s_t", color="#777777")
    # Layer 2 → Layer 1  (action down) — slight x offset to avoid overlap
    _arrow(ax, 0.390, yc2 - BOX_H / 2, 0.390, yc1 + BOX_H / 2,
           label="action a_t", color="#777777")
    # Layer 1 → Layer 2  reward
    _arrow(ax, 0.500, yc1 + BOX_H / 2, 0.500, yc2 - BOX_H / 2,
           label="reward r_t", color="#777777")

    # Layer 2 → Layer 3  (Q-values / actions up)
    _arrow(ax, 0.600, yc2 + BOX_H / 2, 0.500, yc3 - BOX_H / 2,
           label="Q-values\n(confidence)", color="#777777")
    # Layer 3 → Layer 2  (blended rewards down)
    _arrow(ax, 0.680, yc3 - BOX_H / 2, 0.680, yc2 + BOX_H / 2,
           label="r_blended", color="#777777")

    # Layer 3 → Layer 4  (training signals up)
    _arrow(ax, 0.350, yc3 + BOX_H / 2, 0.350, yc4 - BOX_H / 2,
           label="losses /\ngradients", color="#777777")
    # Layer 4 → Layer 3  (policy updates down)
    _arrow(ax, 0.430, yc4 - BOX_H / 2, 0.430, yc3 + BOX_H / 2,
           label="updated\npolicies", color="#777777")

    # ── Legend ─────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=_hex_lighten(LAYER_COLORS[1], 0.30),
                       edgecolor=LAYER_COLORS[1], label="Layer 1 — Market Environment"),
        mpatches.Patch(facecolor=_hex_lighten(LAYER_COLORS[2], 0.30),
                       edgecolor=LAYER_COLORS[2], label="Layer 2 — Specialist Agents (DQN)"),
        mpatches.Patch(facecolor=_hex_lighten(LAYER_COLORS[3], 0.30),
                       edgecolor=LAYER_COLORS[3], label="Layer 3 — Orchestrator (UCB + MARL)"),
        mpatches.Patch(facecolor=_hex_lighten(LAYER_COLORS[4], 0.30),
                       edgecolor=LAYER_COLORS[4], label="Layer 4 — Training Pipeline"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4,
        fontsize=8.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )

    # ── Save ───────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.0, 1, 0.935])
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUT_PATH.resolve()}")
    print(f"Size:  {OUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
