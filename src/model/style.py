"""Shared colours and plot style for the fair-value notebooks.

One colour per flag, used in every figure:
    overpriced     orange
    fair           blue
    below typical  green (teal)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PAL = {
    "primary": "#2a78d6",   # blue: fair
    "accent": "#eb6834",    # orange: overpriced
    "teal": "#1baf7a",      # green: below typical
    "ink": "#1f1f1e",
    "muted": "#6b6a66",
    "grid": "#e6e5e1",
    "surface": "#fcfcfb",
}
LABEL_COL = {"overpriced": PAL["accent"], "fair": PAL["primary"], "below_typical": PAL["teal"]}
LABEL_ORDER = ["overpriced", "fair", "below_typical"]
MODEL_COL = {"baseline_median": "#9a9893", "ridge": PAL["teal"], "gradient_boosting": PAL["primary"]}

EUR = mtick.FuncFormatter(lambda v, _: f"€{v:,.0f}")


def apply_style() -> None:
    """Set matplotlib defaults: dark ink text, recessive grid, and larger labels for reports."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": PAL["surface"],
        "axes.facecolor": PAL["surface"],
        "axes.edgecolor": "#c9c8c3",
        "text.color": PAL["ink"],
        "axes.labelcolor": PAL["ink"],
        "axes.titlecolor": PAL["ink"],
        "xtick.color": PAL["muted"],
        "ytick.color": PAL["muted"],
        "xtick.labelcolor": PAL["ink"],
        "ytick.labelcolor": PAL["ink"],
        "legend.labelcolor": PAL["ink"],
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": PAL["grid"],
        "grid.linewidth": 0.8,
    })
