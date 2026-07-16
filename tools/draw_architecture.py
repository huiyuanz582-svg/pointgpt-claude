# -*- coding: utf-8 -*-
"""Draw the method/architecture figure for the denoising paper.

Layout: the inference pipeline runs left-to-right in one band (noisy cloud ->
patch partition -> embedding -> GPT extractor -> GPT generator -> residual
add -> weighted fusion -> denoised cloud); training-only parts (pre-training
init, Chamfer+EMD loss against the clean cloud) hang below with dashed lines.
Contribution modules carry an accent color and numbered badges that the paper
text can reference.

Outputs both a PNG preview and a vector PDF (use the PDF in LaTeX: small file,
crisp at any zoom, no Overleaf compile-time cost).

Usage:
  python tools/draw_architecture.py out/architecture   # -> .png + .pdf
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---- palette (restrained: one accent for contributions, neutrals elsewhere) --
INK = "#1F2937"          # primary text
MUTED = "#6B7280"        # secondary text / neutral edges
NEUTRAL_FILL = "#F1F3F7"
ACCENT = "#3B5BDB"       # contribution modules
ACCENT_FILL = "#E8EDFB"
CLOUD_NOISY = "#7A85D6"
CLOUD_CLEAN = "#5FA97F"
LOSS_EDGE = "#B34A4A"

BOX_H = 1.75


def box(ax, x, y, w, h, title, sub=None, accent=False, badge=None,
        dashed=False, title_size=10.5):
    fc, ec = (ACCENT_FILL, ACCENT) if accent else (NEUTRAL_FILL, MUTED)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor=fc, edgecolor=ec, linewidth=1.6 if accent else 1.1,
        linestyle="--" if dashed else "-", zorder=2))
    if sub:
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
                fontsize=title_size, color=INK, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + 0.55, sub, ha="center", va="center",
                fontsize=8.4, color=MUTED, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, color=INK, fontweight="bold", zorder=3)
    if badge:
        r = 0.19
        bx, by = x + w - 0.02, y + h - 0.02
        ax.add_patch(Circle((bx, by), r, facecolor=ACCENT, edgecolor="white",
                            linewidth=1.2, zorder=4))
        ax.text(bx, by, badge, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=5)
    return x + w  # right edge, handy for chaining


def arrow(ax, x0, y0, x1, y1, dashed=False, color=MUTED, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.3, color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def cloud(ax, cx, cy, r, color, noise, label, seed=0, n=900):
    """A small torus point cloud glyph, optionally noisy."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (1.0 + 0.45 * np.cos(v)) * np.cos(u)
    y = (1.0 + 0.45 * np.cos(v)) * np.sin(u) * 0.55 + 0.45 * np.sin(v) * 0.8
    pts = np.column_stack([x, y]) * r * 0.62
    pts += rng.normal(scale=noise * r, size=pts.shape)
    ax.scatter(cx + pts[:, 0], cy + pts[:, 1], s=1.1, c=color, linewidths=0,
               zorder=3)
    ax.text(cx, cy - r - 0.25, label, ha="center", va="top", fontsize=9.5,
            color=INK, fontweight="bold")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "architecture"

    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 12.6)
    ax.set_axis_off()

    y = 7.2                     # main pipeline band (box bottom)
    cy = y + BOX_H / 2          # arrow height

    # ---- main pipeline -----------------------------------------------------
    cloud(ax, 1.9, cy, 1.3, CLOUD_NOISY, noise=0.06,
          label="Noisy input\n$P \\in \\mathbb{R}^{N\\times 3}$", seed=1)

    x = 3.9
    arrow(ax, 3.3, cy, x - 0.1, cy)
    x = box(ax, x, y, 3.5, BOX_H, "Patch Partition",
            "FPS + $k$NN, Morton order\n$G$ patches $\\times$ $K$ pts")
    arrow(ax, x + 0.1, cy, x + 0.75, cy)
    x = box(ax, x + 0.85, y, 3.7, BOX_H, "Patch Embedding",
            "PointNet encoder +\nabs. / rel. position enc.")
    arrow(ax, x + 0.1, cy, x + 0.75, cy)
    gpt_x = x + 0.85
    x = box(ax, gpt_x, y, 4.0, BOX_H, "GPT Extractor",
            "$12\\times$ causal Transformer\n(generatively pre-trained)",
            accent=True, badge="1")
    arrow(ax, x + 0.1, cy, x + 0.75, cy)
    gen_x = x + 0.85
    x = box(ax, gen_x, y, 3.8, BOX_H, "GPT Generator",
            "$4\\times$ blocks $\\rightarrow$ per-point\ndisplacement $\\Delta$",
            accent=True, badge="2")

    # residual add node
    plus_x = x + 1.0
    ax.add_patch(Circle((plus_x, cy), 0.30, facecolor="white",
                        edgecolor=ACCENT, linewidth=1.6, zorder=3))
    ax.text(plus_x, cy, "+", ha="center", va="center", fontsize=15,
            color=ACCENT, fontweight="bold", zorder=4)
    arrow(ax, x + 0.1, cy, plus_x - 0.32, cy)
    # skip connection: noisy patches bypass over the top
    arrow(ax, 5.6, y + BOX_H + 0.12, plus_x, cy + 0.34, rad=-0.22,
          color=ACCENT)
    ax.text((5.6 + plus_x) / 2, y + BOX_H + 1.55,
            "noisy patches (residual)", ha="center", fontsize=8.8,
            color=ACCENT, style="italic")

    fuse_x = plus_x + 0.65
    arrow(ax, plus_x + 0.32, cy, fuse_x - 0.1, cy)
    x = box(ax, fuse_x, y, 4.0, BOX_H, "Weighted Patch Fusion",
            "overlapping patches\n$\\rightarrow$ $N$ points", accent=True,
            badge="3", title_size=10)
    arrow(ax, x + 0.1, cy, x + 0.7, cy)
    out_cx = x + 2.1
    cloud(ax, out_cx, cy, 1.3, CLOUD_NOISY, noise=0.004,
          label="Denoised output $\\hat{P}$", seed=1)

    # ---- pre-training init (below the extractor, dashed) --------------------
    pre_y = 3.3
    box(ax, gpt_x + 0.15, pre_y, 3.7, 1.6, "PointGPT pre-training",
        "auto-regressive generation\non ShapeNet", dashed=True, title_size=9.5)
    arrow(ax, gpt_x + 2.0, pre_y + 1.75, gpt_x + 2.0, y - 0.12,
          dashed=True, color=ACCENT)
    ax.text(gpt_x + 2.15, (pre_y + 1.75 + y) / 2, "init.", fontsize=8.6,
            color=ACCENT, ha="left")

    # ---- training loss (bottom right, dashed) --------------------------------
    loss_x, loss_y = 23.2, 3.4
    box(ax, loss_x, loss_y, 3.6, 1.2,
        "$\\mathcal{L} = 3\\,\\mathrm{CD} + 10\\,\\mathrm{EMD}$",
        dashed=True, title_size=10)
    cloud(ax, 20.0, loss_y + 0.6, 1.05, CLOUD_CLEAN, noise=0.004,
          label="Clean GT $P_{gt}$", seed=1, n=700)
    arrow(ax, 21.2, loss_y + 0.6, loss_x - 0.12, loss_y + 0.6,
          dashed=True, color=LOSS_EDGE)
    arrow(ax, out_cx - 0.7, cy - 2.2, loss_x + 3.1, loss_y + 1.4, dashed=True,
          color=LOSS_EDGE, rad=0.15)
    ax.text(loss_x + 1.8, loss_y - 0.4, "training only", ha="center",
            fontsize=8.6, color=MUTED, style="italic")

    # ---- legend --------------------------------------------------------------
    lx, ly = 1.2, 1.3
    ax.add_patch(FancyBboxPatch((lx, ly - 0.22), 0.55, 0.45,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor=ACCENT_FILL, edgecolor=ACCENT,
                                linewidth=1.5))
    ax.text(lx + 0.8, ly, "our contributions (1–3)", fontsize=9, color=INK,
            va="center")
    ax.plot([lx + 5.4, lx + 6.1], [ly, ly], linestyle="--", color=MUTED,
            linewidth=1.3)
    ax.text(lx + 6.3, ly, "training / initialization only", fontsize=9,
            color=INK, va="center")

    for ext in (".png", ".pdf"):
        fig.savefig(out + ext, dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"saved {out}.png and {out}.pdf")


if __name__ == "__main__":
    main()
