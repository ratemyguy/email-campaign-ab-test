"""Shared matplotlib styling: white background, takeaway titles, source note."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as cfg  # noqa: E402

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 8.5,
        "savefig.facecolor": "white",
        "text.parse_math": False,  # "$100-$199" is money, not LaTeX math
    }
)


def new_figure(nrows: int = 1, ncols: int = 1, figsize=cfg.FIGSIZE, **kwargs):
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def finish(
    fig,
    title: str,
    path: Path,
    subtitle: str | None = None,
    note: str = cfg.SOURCE_NOTE,
    top: float = 0.84,
    bottom: float = 0.13,
) -> Path:
    """Add the takeaway title (+ optional subtitle) and source note, save at 150 dpi, close."""
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color="#111111")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=8.3, color="#444444")
    fig.text(0.01, 0.015, note, ha="left", va="bottom", fontsize=6.8, color="#555555")
    fig.subplots_adjust(top=top, bottom=bottom)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.DPI)
    plt.close(fig)
    return path
