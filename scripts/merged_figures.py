"""Figures 1-11 of paper3, exported unchanged from the V13ADmergedfigures /
V7ADmerged notebooks (paper2 figure code + the layout fixes for the merged
gene lists). Reads data/ only."""

"""Shared visual language for every figure in the paper.

One module so that a panel letter, a tick, a bar gap or a significance bracket
looks the same wherever it appears -- the thing that separates a figure set from
a pile of plots. Everything here is layout and typography; the numbers come from
../data and are never touched.

Colour is the data-viz reference palette used unchanged: categorical slots 1-3
(blue / orange / aqua) for the three model arms, which is the documented
all-pairs colourblind-safe subset, and the documented blue-red diverging pair
with a neutral grey midpoint for up/down regulation. Identity never rests on
colour alone -- every series also carries a marker shape, a direct label or a
legend entry.
"""
import os
import warnings
from pathlib import Path

import matplotlib as mpl
# PAPER_INLINE is set when this same code runs inside the Kaggle notebook, where
# the figures must render in the output rather than only be written to disk.
INLINE = bool(os.environ.get("PAPER_INLINE"))
if not INLINE:
    mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Paths are overridable so that the notebook can point them at /kaggle/working
# and run this file unmodified -- one source of truth for every figure.
if "PAPER_DATA" in os.environ:
    DATA = Path(os.environ["PAPER_DATA"])
    FIG = Path(os.environ.get("PAPER_FIGS", DATA / "figures"))
else:
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "data"
    FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Target printed width. The LaTeX text block is 6.27 in wide, so a figure drawn
# much wider than this is downscaled by \includegraphics and every font in it
# shrinks with the drawing -- the usual reason a figure set looks amateur on the
# page. Everything here is laid out at TW and lands at ~0.9 scale.
TW = 6.9

# ---- palette -------------------------------------------------------------
# A cool, low-chroma set in the register of a printed journal figure: navy,
# wine and violet rather than the primary blue / orange / green of a screen
# palette. The three model arms stay separable for the common forms of colour
# blindness -- they differ in hue *and* in lightness (navy 30%, wine 42%,
# violet 55%), so they also survive a greyscale print -- and identity never
# rests on colour alone, since every series carries a marker or a direct label.
PLS     = "#173451"   # categorical slot 1 -- navy
CVAE_T  = "#8f3a4f"   # slot 2 -- wine
CVAE_N  = "#4f4478"   # slot 3 -- muted violet
AD      = "#8f3a4f"   # diverging warm pole -- wine
CTRL    = "#173451"   # diverging cool pole -- navy
UP, DOWN = AD, CTRL
MID     = "#8a8a92"   # neutral midpoint ink, faintly cool
ACCENT  = "#1f6372"   # reserved for "found by both" -- petrol blue
INK     = "#17171c"
INK2    = "#54545e"
INK3    = "#8a8a92"
RULE    = "#c9c9d1"
GRID    = "#e9e9ef"
SURFACE = "#ffffff"

# Diverging ramp for probability surfaces: cool pole -> neutral grey -> warm
# pole. Deliberately light at both ends -- this ramp is only ever a background
# under data marks, and a saturated pole swallows them.
DIVERGING = LinearSegmentedColormap.from_list(
    "ctrl_ad", ["#7d92ad", "#a8b5c9", "#d3d9e3", "#eeeef1",
                "#e3ccd3", "#c99aa8", "#a96d80"])

# Pale stepped bands of P(AD) for the latent-space maps (Figure 3): light enough
# that held-out markers in the two pole colours stand out against them.
from matplotlib.colors import ListedColormap, BoundaryNorm
GEO_LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
GEO_BANDS = ListedColormap(["#cdd6e3", "#e2e7ef", "#f3e7eb", "#e6cdd5"])
GEO_NORM = BoundaryNorm(GEO_LEVELS, 4)

# Sequential ramps for the confusion matrices, built from the two arm colours so
# the matrices sit inside the same muted palette as everything else instead of
# borrowing matplotlib's saturated "Blues"/"Oranges".
SEQ_PLS = LinearSegmentedColormap.from_list(
    "seq_pls", ["#ffffff", "#e7ecf2", "#c3d1e0", "#93aac6", "#5c7fa6", PLS, "#173252"])
SEQ_CVAE_N = LinearSegmentedColormap.from_list(
    "seq_cvae_n", ["#ffffff", "#efedf5", "#d9d4e8", "#b7aed2", "#9286b5", CVAE_N, "#463d66"])
SEQ_CVAE = LinearSegmentedColormap.from_list(
    "seq_cvae", ["#ffffff", "#f5e9ec", "#e6c9d1", "#d0a1ae", "#b06c81", CVAE_T, "#63242f"])

MODEL_COLOR = {"pls": PLS, "cvae_true": CVAE_T, "cvae_neutral": CVAE_N,
               "PLS-DA": PLS, "CVAE_true": CVAE_T, "CVAE_neutral": CVAE_N}
MODEL_LABEL = {"pls": "PLS-DA",
               "cvae_true": "CVAE (label-conditioned)",
               "cvae_neutral": "CVAE (label-agnostic)",
               "PLS-DA": "PLS-DA",
               "CVAE_true": "CVAE (label-conditioned)",
               "CVAE_neutral": "CVAE (label-agnostic)"}
GROUP_COLOR = {"AD": AD, "Control": CTRL}
GROUP_MARKER = {"AD": "^", "Control": "o"}

# database sources in the g:Profiler output
SRC_COLOR = {"GO:MF": "#24486b", "GO:BP": "#2f7d8c", "GO:CC": "#6f6396",
             "CORUM": "#8f3a4f", "KEGG": "#4b7ca8", "REAC": "#57516e",
             "WP": "#3f6d80", "MIRNA": "#a3798f", "TF": "#54545e", "HP": "#7a4a5e"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "axes.titlepad": 7,
    "axes.labelsize": 8,
    "axes.labelpad": 3.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.1,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


# ---- primitives ----------------------------------------------------------
def save(fig, name):
    """Vector PDF for pdflatex, 400-dpi PNG for the pandoc/docx build."""
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight",
                    pad_inches=0.03, **kw)
    if INLINE:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:      # running the notebook's cells outside a kernel
            pass
    plt.close(fig)
    print(f"  {FIG.name}/{name}.pdf + .png")


def panel(ax, letter, title=None, subtitle=None, dx=-0.13, dy=1.0):
    """Panel letter in sans-serif, hung outside the axes at a fixed offset so
    the letters line up down the figure regardless of tick-label width.

    Title and subtitle are both annotations rather than `set_title`, because a
    real title always sits at one fixed height and would land on top of the
    subtitle whenever both are present.
    """
    ax.annotate(letter, xy=(dx, dy), xycoords="axes fraction",
                xytext=(0, 2), textcoords="offset points",
                fontsize=9.5, fontweight="bold", family="DejaVu Sans",
                color=INK, ha="left", va="bottom", annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 13 if subtitle else 3), textcoords="offset points",
                    fontsize=8.6, color=INK, ha="left", va="bottom",
                    annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=6.8, color=INK3, ha="left", va="bottom",
                    annotation_clip=False)


def ygrid(ax):
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)


def xgrid(ax):
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)


def despine(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def load(name):
    return pd.read_csv(DATA / name)


def scal(df=None):
    """summary_scalars.csv -> dict, numeric where possible."""
    df = df if df is not None else load("summary_scalars.csv")
    out = {}
    for _, r in df.iterrows():
        try:
            out[r["key"]] = float(r["value"])
        except (TypeError, ValueError):
            out[r["key"]] = r["value"]
    return out


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def pmath(p, prefix="p"):
    """`$p = ...$` with a real times-ten-to-the exponent, not `1e-05`."""
    if p >= 1e-3:
        return f"${prefix} = {p:.3f}$"
    if not p > 1e-300:          # V7: with thousands of pairs a p-value can underflow to 0
        return f"${prefix} < 10^{{-300}}$"
    e = int(np.floor(np.log10(p)))
    return f"${prefix} = {p/10**e:.1f}\\times 10^{{{e}}}$"


def italic_ticks(ax, names, fontsize=6.0, axis="y"):
    """Italic gene symbols as real text, so hyphens stay hyphens and digits do
    not pick up math-italic spacing (`HLA-DQA2`, not `HLA - DQA2`)."""
    ticks = np.arange(len(names))
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(names, fontsize=fontsize, fontstyle="italic")
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, fontsize=fontsize, fontstyle="italic")


def bracket(ax, x0, x1, y, text, tick=None, fontsize=7.2, color=INK):
    """Significance bracket that never overlaps the marks it spans."""
    tick = tick if tick is not None else abs(y) * 0.02 + 1e-9
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick],
            color=color, lw=0.7, clip_on=False, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y + tick * 0.5, text, ha="center", va="bottom",
            fontsize=fontsize, color=color, clip_on=False)


def boxstrip(ax, values, pos, color, width=0.44, jitter=0.055, seed=0, size=4):
    """A box over its own points: the distribution and its summary together."""
    rs = np.random.RandomState(int(round(seed)))
    ax.scatter(rs.normal(pos, jitter, len(values)), values, s=size, color=color,
               alpha=0.28, linewidth=0, zorder=1, rasterized=True)
    bp = ax.boxplot([values], positions=[pos], widths=width, showfliers=False,
                    patch_artist=True, zorder=3, manage_ticks=False)
    bp["boxes"][0].set(facecolor=SURFACE, edgecolor=color, linewidth=1.2, alpha=0.95)
    for part in ("whiskers", "caps", "medians"):
        for a in bp[part]:
            a.set(color=color, linewidth=1.2)
    return bp


def swatches(labels_colors, marker="s", size=5):
    return [Line2D([], [], marker=marker, ls="", color=c, markersize=size, label=l)
            for l, c in labels_colors]


def repel_labels(ax, xs, ys, labels, fontsize=5.4, color=None, italic=True,
                 clearance=1.6, leader_color=None):
    """Place a label for every point without letting any two labels touch.

    A dependency-free replacement for adjustText: each label is tried at a ring
    of candidate offsets, in decreasing order of preference, and the first
    offset whose estimated box overlaps neither an already-placed label nor its
    own point is taken. Everything is measured in typographic points, which is
    exactly the unit `annotate` takes for `xytext`, so no round trip through
    data coordinates is needed and the result is stable at any figure size.
    """
    color = INK if color is None else color
    leader_color = RULE if leader_color is None else leader_color
    fig = ax.figure
    ppd = fig.dpi / 72.0                     # points -> display pixels
    pts = ax.transData.transform(np.column_stack([xs, ys]))

    # candidate offsets: near ring first, then progressively further out
    cands = []
    for radius in (7, 11, 16, 22, 30, 40, 52, 66, 82, 100):   # V7: wider search
        for ang in range(0, 360, 30):
            a = np.deg2rad(ang)
            cands.append((radius * np.cos(a), radius * np.sin(a)))

    placed = []                              # display-space boxes already taken
    leg = ax.get_legend()                    # a label under the key is unreadable
    if leg is not None:
        bb = leg.get_window_extent()
        placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
    order = np.argsort(-np.asarray(ys))      # most significant genes get first pick

    def overlaps(box):
        x0, y0, x1, y1 = box
        for p0, q0, p1, q1 in placed:
            if x0 < p1 and p0 < x1 and y0 < q1 and q0 < y1:
                return True
        return False

    axbb = ax.get_window_extent()            # V7: labels must stay inside the axes
    inside = lambda b: (b[0] >= axbb.x0 and b[2] <= axbb.x1 and
                        b[1] >= axbb.y0 and b[3] <= axbb.y1)
    forced = 0
    for i in order:
        text = labels[i]
        w = len(text) * fontsize * 0.56 * ppd + clearance * ppd
        h = fontsize * 1.25 * ppd + clearance * ppd
        px, py = pts[i]
        boxes = []
        for dx, dy in cands:
            cx = px + dx * ppd + (w / 2 - clearance) * (1 if dx >= 0 else -1)
            cy = py + dy * ppd
            boxes.append(((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dx, dy))
        ok = [t for t in boxes if inside(t[0]) and not overlaps(t[0])]
        if not ok:                           # never drop a gene: least-bad inside spot
            forced += 1
            ok = [t for t in boxes if inside(t[0])] or boxes
        for box, dx, dy in ok[:1]:
            if True:
                placed.append(box)
                ax.annotate(text, (xs[i], ys[i]), xytext=(dx, dy),
                            textcoords="offset points", fontsize=fontsize,
                            color=color, fontstyle="italic" if italic else "normal",
                            ha="left" if dx >= 0 else "right", va="center",
                            zorder=5, annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=leader_color,
                                            lw=0.35, shrinkA=0.5, shrinkB=1.5))
    if forced:
        print(f"  repel_labels: {forced} label(s) had no collision-free spot")


def auc_rank(y, s):
    """ROC-AUC from ranks (Mann-Whitney U); NaN when a group is absent."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def column_labels(ax, xs, ys, labels, colors, x_text, y_lo, y_hi, side, fontsize=5.0, spacing=None):
    """Name every point from a column at x_text, labels spread evenly between
    y_lo and y_hi in height order, each tied to its point by a thin leader. No
    two labels can touch whatever the density of the points."""
    # each label starts level with its own point; overlapping neighbours are
    # pushed apart by the minimum spacing only, so leaders stay short and flat
    order = np.argsort(ys)
    want = np.asarray(ys, float)[order]
    gap = (y_hi - y_lo) / max(len(order), 1) * 0.98 if spacing is None else spacing
    slots = np.maximum(want, y_lo)
    for k in range(1, len(slots)):
        slots[k] = max(slots[k], slots[k - 1] + gap)
    if len(slots) and slots[-1] > y_hi:
        slots[-1] = y_hi
        for k in range(len(slots) - 2, -1, -1):
            slots[k] = min(slots[k], slots[k + 1] - gap)
    for yt, i in zip(slots, order):
        ax.annotate(labels[i], xy=(xs[i], ys[i]), xytext=(x_text, yt), textcoords="data",
                    fontsize=fontsize, fontstyle="italic", color=colors[i],
                    ha="left" if side == "right" else "right", va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=tint(colors[i], 0.45), lw=0.35,
                                    shrinkA=1.0, shrinkB=1.5))


def kde2d(xy, grid_x, grid_y, bw_scale=1.0):
    """Gaussian KDE evaluated on a grid, without a scipy import at call sites."""
    from scipy.stats import gaussian_kde
    k = gaussian_kde(xy.T)
    k.set_bandwidth(k.factor * bw_scale)
    XX, YY = np.meshgrid(grid_x, grid_y)
    return k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

# the two objects every figure function draws from
S = scal()
runs = load("run_metrics.csv")
print(f"{len(S)} scalars, {len(runs)} runs")

"""Figure 1 -- the study design, drawn rather than measured.

The only figure in the paper that carries no data: it exists so the reader can
see, before any result, that the two models are matched on the thing that
matters (an 8-dimensional bottleneck fitted on identical splits) and differ only
in how they get there.
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BOX_KW = dict(boxstyle="round,pad=0.30,rounding_size=0.09", linewidth=0.9)


def box(ax, x, y, w, h, text, face, edge, fontsize=7.2, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge, **BOX_KW))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight=weight, linespacing=1.35)


def arrow(ax, xy0, xy1, color=INK2, lw=0.9, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=8,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}"))


def tint(hex_color, alpha):
    """Flat tint over the white surface -- keeps the PDF free of transparency."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(1 - alpha * (1 - c) for c in (r, g, b))


def fig_design():
    coh = load("cohort_summary.csv").iloc[0]
    fig = plt.figure(figsize=(13.6, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.9], hspace=0.10)

    # ---------------- (a) cohort and resampling protocol -----------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 23); ax.axis("off")
    panel(ax, "a", dx=-0.012, dy=1.02)
    ax.text(0, 21.6, "Cohort, preprocessing and resampling protocol",
            fontsize=8.8, color=INK, ha="left", va="bottom")

    box(ax, 0.5, 8, 15, 11,
        f"{int(coh.n_cohorts)} GEO cohorts\n{int(coh.n_subjects)} arrays\n{int(coh.n_donors)} donors, age $\\geq${int(coh.age_min)}",
        tint(MID, 0.22), INK3)
    box(ax, 19, 8, 16, 11,
        f"Probe $\\rightarrow$ symbol\n(NCBI Gene, all cohorts)\n{int(coh.n_genes):,} genes",
        tint(MID, 0.22), INK3)
    box(ax, 38.5, 8, 17, 11,
        f"Donor-grouped split\n60 / 20 / 20\ntrain / val / test",
        tint(MID, 0.22), INK3)
    arrow(ax, (15.5, 13.5), (19, 13.5)); arrow(ax, (35, 13.5), (38.5, 13.5))

    # the two resampling arms
    box(ax, 60, 14, 18, 8, f"$\\times\\,20$ real-label splits\nrank statistics",
        tint(PLS, 0.30), PLS)
    box(ax, 60, 3.5, 18, 8, "$\\times\\,10$ label-permuted splits\nempirical null",
        tint(MID, 0.30), INK3)
    arrow(ax, (55.5, 14.5), (60, 18), rad=0.15)
    arrow(ax, (55.5, 12.5), (60, 7.5), rad=-0.15)

    box(ax, 82, 8, 17, 11,
        "One-sided Mann–Whitney\n$\\rightarrow$ BH FDR $q<0.05$\n$+$ stability selection",
        CVAE_T, CVAE_T, color="white", weight="bold")
    arrow(ax, (78, 18), (82, 14.6), rad=-0.15)
    arrow(ax, (78, 7.5), (82, 12.4), rad=0.15)

    ax.text(27, 5.4, "z-scored with training-fold parameters only",
            fontsize=6.6, color=INK3, ha="center", style="italic")

    # ---------------- (b) the two matched architectures ------------------
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 100); ax.set_ylim(-4.0, 57); ax.axis("off")
    panel(ax, "b", dx=-0.012, dy=1.01)
    ax.text(0, 55.2, "Two representations of the same 8-dimensional bottleneck",
            fontsize=8.8, color=INK, ha="left", va="bottom")

    # ---- the two shared inputs, drawn once on the left -------------------
    box(ax, 0.5, 21, 11.5, 10,
        f"$\\mathbf{{X}}$\n$n\\times{int(coh.n_genes):,}$\nexpression",
        tint(MID, 0.22), INK3, fontsize=7.0)
    box(ax, 0.5, 35, 11.5, 7.5, "$\\mathbf{Y}$\none-hot labels",
        tint(CVAE_N, 0.30), RULE, fontsize=7.0)
    ax.text(6.25, 19.6, "same split for both lanes", fontsize=6.2, color=INK3,
            ha="center", va="top", style="italic")

    # ================= PLS-DA lane =======================================
    lane = Rectangle((14.0, 29.5), 84.5, 25.0, facecolor=tint(PLS, 0.10),
                     edgecolor="none", zorder=0)
    ax.add_patch(lane)
    ax.text(15.0, 53.4, "PLS-DA  ·  linear, supervised projection",
            fontsize=7.6, color=PLS, ha="left", va="top", fontweight="bold")

    box(ax, 15.5, 38.0, 14.5, 9.0,
        "NIPALS\n$\\mathbf{T}=\\mathbf{XW}$\n8 components", tint(PLS, 0.30), PLS)
    box(ax, 33.5, 38.0, 12.5, 9.0, "Scores\n$\\mathbf{T}\\in\\mathbb{R}^{n\\times 8}$",
        PLS, PLS, fontsize=7.3, weight="bold", color="white")
    box(ax, 50.5, 44.6, 15.5, 6.0, "Logistic regression", tint(PLS, 0.30), PLS,
        fontsize=6.9)
    box(ax, 50.5, 36.8, 15.5, 6.4, "Loadings $\\mathbf{P}$\n$\\mathbf{T}\\mathbf{P}^{\\top}$",
        tint(PLS, 0.30), PLS, fontsize=6.9)
    box(ax, 70.0, 44.6, 14.0, 6.0, "$\\hat{y}$  prediction", tint(MID, 0.22), INK3,
        fontsize=6.9)
    box(ax, 70.0, 36.8, 14.0, 6.4, "$\\hat{\\mathbf{X}}$  reconstruction",
        tint(MID, 0.22), INK3, fontsize=6.9)
    box(ax, 33.5, 30.2, 12.5, 5.4, "x-weights $\\mathbf{W}$", tint(PLS, 0.22), PLS,
        fontsize=6.7)
    box(ax, 50.5, 30.2, 15.5, 5.4, "VIP score per gene", tint(PLS, 0.42), PLS,
        fontsize=6.9)

    arrow(ax, (12.0, 28.0), (15.5, 41.0), color=PLS, rad=0.14)
    arrow(ax, (12.0, 38.5), (15.5, 43.5), color=CVAE_N, rad=-0.10)
    arrow(ax, (30.0, 42.5), (33.5, 42.5), color=PLS)
    arrow(ax, (46.0, 43.6), (50.5, 47.6), color=PLS, rad=0.10)
    arrow(ax, (46.0, 41.5), (50.5, 40.0), color=PLS, rad=-0.10)
    arrow(ax, (66.0, 47.6), (70.0, 47.6), color=PLS)
    arrow(ax, (66.0, 40.0), (70.0, 40.0), color=PLS)
    arrow(ax, (39.7, 38.0), (39.7, 35.6), color=PLS, ls=(0, (2.4, 1.6)))
    arrow(ax, (46.0, 32.9), (50.5, 32.9), color=PLS)

    # ================= beta-CVAE lane ====================================
    lane = Rectangle((14.0, 1.0), 84.5, 25.0, facecolor=tint(CVAE_T, 0.10),
                     edgecolor="none", zorder=0)
    ax.add_patch(lane)
    ax.text(15.0, 24.9, "$\\beta$-CVAE  ·  non-linear, semi-supervised generative model",
            fontsize=7.6, color=CVAE_T, ha="left", va="top", fontweight="bold")

    box(ax, 15.5, 11.6, 14.0, 9.0,
        "Encoder $q_\\phi(\\mathbf{z}\\mid\\mathbf{x},\\mathbf{c})$\n"
        "$256\\rightarrow$ BN, ReLU\ndropout 0.2", tint(CVAE_T, 0.30), CVAE_T,
        fontsize=6.8)
    box(ax, 32.5, 16.4, 9.5, 4.6, "$\\mu$", tint(CVAE_T, 0.30), CVAE_T, fontsize=7.2)
    box(ax, 32.5, 10.6, 9.5, 4.6, "$\\log\\sigma^{2}$", tint(CVAE_T, 0.30), CVAE_T,
        fontsize=7.2)
    box(ax, 45.5, 12.6, 13.5, 6.8,
        "$\\mathbf{z}=\\mu+\\sigma\\odot\\epsilon$\n$\\in\\mathbb{R}^{n\\times 8}$",
        CVAE_T, CVAE_T, fontsize=7.2, weight="bold", color="white")
    box(ax, 45.5, 5.4, 13.5, 4.6, "$\\epsilon\\sim\\mathcal{N}(0,\\mathbf{I})$",
        tint(MID, 0.22), INK3, fontsize=6.7)
    box(ax, 62.5, 17.4, 14.0, 6.2, "Classifier head\nFC-ReLU-FC", tint(CVAE_T, 0.30),
        CVAE_T, fontsize=6.7)
    box(ax, 62.5, 9.6, 14.0, 6.2, "Decoder $p_\\theta(\\mathbf{x}\\mid\\mathbf{z},\\mathbf{c})$",
        tint(CVAE_T, 0.30), CVAE_T, fontsize=6.7)
    box(ax, 80.5, 17.4, 14.5, 6.2, "$\\hat{y}$  prediction", tint(MID, 0.22), INK3,
        fontsize=6.9)
    box(ax, 80.5, 9.6, 14.5, 6.2, "$\\hat{\\mathbf{X}}$  reconstruction",
        tint(MID, 0.22), INK3, fontsize=6.9)
    box(ax, 80.5, 2.0, 14.5, 5.6, "Integrated Gradients\nper gene",
        tint(CVAE_T, 0.42), CVAE_T, fontsize=6.7)

    arrow(ax, (6.25, 21.0), (15.5, 17.5), color=CVAE_T, rad=-0.10)
    arrow(ax, (29.5, 17.5), (32.5, 18.7), color=CVAE_T)
    arrow(ax, (29.5, 14.5), (32.5, 12.9), color=CVAE_T)
    arrow(ax, (42.0, 18.7), (45.5, 17.2), color=CVAE_T)
    arrow(ax, (42.0, 12.9), (45.5, 14.6), color=CVAE_T)
    arrow(ax, (52.2, 10.0), (52.2, 12.6), color=INK3, ls=(0, (2.4, 1.6)))
    arrow(ax, (59.0, 17.4), (62.5, 20.0), color=CVAE_T, rad=0.10)
    arrow(ax, (59.0, 14.6), (62.5, 12.7), color=CVAE_T, rad=-0.10)
    arrow(ax, (76.5, 20.5), (80.5, 20.5), color=CVAE_T)
    arrow(ax, (76.5, 12.7), (80.5, 12.7), color=CVAE_T)
    # the attribution route runs outside the output boxes, down the lane edge
    ax.plot([95.0, 97.2, 97.2], [20.5, 20.5, 4.8], color=CVAE_T, lw=0.9, ls=(0, (2.4, 1.6)))
    arrow(ax, (97.2, 4.8), (95.0, 4.8), color=CVAE_T)

    # the conditioning vector: the one input the two lanes do not share
    box(ax, 15.5, 3.4, 14.0, 5.4, "$\\mathbf{c}$  conditioning", tint(CVAE_N, 0.30),
        CVAE_N, fontsize=6.8)
    arrow(ax, (22.5, 8.8), (22.5, 11.6), color=CVAE_N, ls=(0, (2.4, 1.6)))
    # the conditioning route to the decoder runs beneath the epsilon box
    arrow(ax, (29.5, 4.4), (69.5, 4.4), color=CVAE_N, ls=(0, (2.4, 1.6)), style="-")
    arrow(ax, (69.5, 4.4), (69.5, 9.6), color=CVAE_N, ls=(0, (2.4, 1.6)))
    ax.text(31.0, 2.6, "true one-hot label  |  neutral $1/K$",
            fontsize=6.2, color=CVAE_N, ha="left", va="center")

    # the matched constraint, drawn once between the two bottleneck boxes
    ax.text(15.5, 27.9, "matched: $d=8$, identical folds",
            fontsize=6.8, color=INK2, ha="left", va="center", style="italic")

    # One rule, stated once: grey is data entering or leaving a lane, violet is
    # label information, a lane-coloured box is an operation of that lane, and
    # the one box per lane with a heavier fill is the matched bottleneck. Every
    # box in the drawing follows it, which is what makes the key worth reading.
    key = [("data in / out", [(tint(MID, 0.22), INK3)]),
           ("label information", [(tint(CVAE_N, 0.30), CVAE_N)]),
           ("operation", [(tint(PLS, 0.30), PLS), (tint(CVAE_T, 0.30), CVAE_T)]),
           ("matched $d=8$ bottleneck", [(PLS, PLS),
                                         (CVAE_T, CVAE_T)])]
    x0 = 15.0
    for lab, sw in key:
        for j, (fc, ec) in enumerate(sw):
            ax.add_patch(Rectangle((x0 + j * 2.4, -0.9), 2.1, 1.6, facecolor=fc,
                                   edgecolor=ec, linewidth=0.7, clip_on=False))
        x0 += 2.4 * (len(sw) - 1) + 3.0
        ax.text(x0, -0.1, lab, fontsize=6.4, color=INK2, ha="left", va="center")
        x0 += len(lab) * 0.95 + 5.0
    ax.text(15.0, -3.0, "Lane colour marks which model an operation belongs to; "
            "a heavier fill marks the bottleneck the two models share.",
            fontsize=6.2, color=INK3, ha="left", va="center", style="italic")

    save(fig, "fig1_design")



fig_design()

# =====================================================================
# Figure 2 -- classification
# =====================================================================
def fig_classification(runs, S):
    roc, pr = load("roc_curves_raw.csv"), load("pr_curves_raw.csv")
    cm = load("confusion_matrices.csv").set_index("model")
    op = load("oof_predictions.csv")
    base = np.linspace(0, 1, 101)
    arms = ("pls", "cvae_neutral", "cvae_true")
    SHORT = {"pls": "PLS-DA", "cvae_neutral": "CVAE agnostic", "cvae_true": "CVAE conditioned"}

    fig = plt.figure(figsize=(TW, 9.3))
    gs = fig.add_gridspec(4, 3, hspace=0.86, wspace=0.42, height_ratios=[1.0, 1.10, 0.95, 0.72])
    top = gs[0, :].subgridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.48)

    # ---- (a) ROC -----------------------------------------------------
    ax = fig.add_subplot(top[0])
    for m in arms:
        v = roc[roc.model == m][[f"tpr_{i}" for i in range(101)]].values
        mu, sd = v.mean(0), v.std(0)
        key = "auc_pls" if m == "pls" else "auc_cvae_lat_" + m.split("_", 1)[1]
        ax.fill_between(base, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=MODEL_COLOR[m], alpha=0.12, lw=0)
        ax.plot(base, mu, color=MODEL_COLOR[m], lw=1.3, ls="-" if m != "cvae_true" else (0, (3, 1.5)),
                label=f"{SHORT[m]}  {S[key + '_mean']:.3f}")
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.004); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.legend(loc="lower right", fontsize=5.6, title="mean AUC", title_fontsize=5.8, alignment="left",
              frameon=True, framealpha=1.0, facecolor=SURFACE, edgecolor="none", handlelength=1.4,
              borderaxespad=0.1)
    panel(ax, "a", "ROC", "mean $\\pm$ SD, 20 splits", dx=-0.30)

    # ---- (b) precision-recall ---------------------------------------
    ax = fig.add_subplot(top[1])
    for m in arms:
        v = pr[pr.model == m][[f"prec_{i}" for i in range(101)]].values
        mu, sd = v.mean(0), v.std(0)
        ax.fill_between(base, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=MODEL_COLOR[m], alpha=0.12, lw=0)
        ax.plot(base, mu, color=MODEL_COLOR[m], lw=1.3, ls="-" if m != "cvae_true" else (0, (3, 1.5)))
    prev = S["n_AD"] / S["n_subjects"] if "n_AD" in S else (op.group == "AD").mean()
    ax.axhline(prev, color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.03, prev - 0.03, "AD prevalence", fontsize=5.8, color=INK3, va="top")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.004); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    panel(ax, "b", "Precision-recall", "same splits", dx=-0.30)

    # ---- (c) slope chart over all three arms, baselines as bands ----------------
    ax = fig.add_subplot(top[2])
    A3 = [runs.auc_pls.values, runs.auc_cvae_lat_neutral.values, runs.auc_cvae_lat_true.values]
    C3 = [PLS, CVAE_N, CVAE_T]
    for col, lab in (("auc_age_only", "age alone"), ("auc_cohort_only", "cohort alone")):
        m_, s_ = runs[col].mean(), runs[col].std()
        ax.axhline(m_, color=tint(MID, 0.8), lw=0.7, ls=(0, (1, 1.6)), zorder=1)
        ax.text(2.62, m_ + 0.004, f"{lab}  {m_:.2f} $\\pm$ {s_:.2f}", ha="right", va="bottom", fontsize=5.6,
                color=INK3)
    for i in range(len(runs)):
        up = A3[1][i] > A3[0][i]
        ax.plot([0, 1], [A3[0][i], A3[1][i]], color=CVAE_N if up else PLS, lw=0.7, alpha=0.5 if up else 0.85,
                zorder=2)
        ax.plot([1, 2], [A3[1][i], A3[2][i]], color=CVAE_T, lw=0.55, alpha=0.55, zorder=2)
    for x, v, c in zip((0, 1, 2), A3, C3):
        m_, s_ = v.mean(), v.std()
        ax.add_patch(Rectangle((x - 0.13, m_ - s_), 0.26, 2 * s_, facecolor=tint(c, 0.16), edgecolor="none",
                               zorder=1))
        ax.scatter(np.full(len(v), x), v, s=6, color=c, lw=0, zorder=3)
        ax.plot([x - 0.17, x + 0.17], [m_, m_], color=c, lw=2.0, solid_capstyle="butt", zorder=4)
        ax.text(x, 0.968, f"{m_:.3f}", ha="center", va="bottom", fontsize=6.8, color=c)
        ax.text(x, 0.962, f"$\\pm${s_:.3f}", ha="center", va="top", fontsize=5.4, color=tint(c, 0.8))
    wins = int((A3[1] > A3[0]).sum()); p_ = S["auc_cvae_neutral_vs_pls_p"]
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["PLS-DA", "CVAE\nagnostic", "CVAE\nconditioned"], fontsize=6.2)
    for t, c in zip(ax.get_xticklabels(), C3):
        t.set_color(c)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.45, 2.65); ax.set_ylim(0.55, 1.0); ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("Test ROC-AUC"); ax.spines["bottom"].set_visible(False)
    panel(ax, "c", "Paired AUC across splits",
          f"higher for CVAE agnostic in {wins}/20, paired {pmath(p_)}", dx=-0.27)

    # ---- (d) small multiples: one cohort per mini-panel, age removed left -> right ----
    v15 = None
    for f_ in [DATA / "validation" / "v15adage" / "comparison_per_cohort.csv"] + \
              [Path(h) for h in __import__("glob").glob("/kaggle/input/**/comparison_per_cohort.csv", recursive=True)]:
        if Path(f_).exists():
            v15 = pd.read_csv(f_); break
    nd_ = op.drop_duplicates("donor").groupby(["cohort", "group"]).size()
    cohorts = sorted(op.cohort.unique(), key=lambda c: -op[op.cohort == c].donor.nunique())
    ARMS_ = ["V7, no age handling", "A, age-corrected", "B, age-matched"]
    LINES = [("auc_age_alone", "#a4a4ad", "D", "age alone", (0, (3, 1.6))),
             ("auc_pls", PLS, "o", "PLS-DA", "-"), ("auc_cvae_agn", CVAE_N, "s", "CVAE agnostic", "-")]
    sub = gs[1, :].subgridspec(1, len(cohorts), wspace=0.10)
    xs_ = np.array([0, 1, 2])
    axd = []
    for k, c in enumerate(cohorts):
        ax = fig.add_subplot(sub[k], sharey=axd[0] if axd else None)
        axd.append(ax)
        ax.axhspan(0.35, 0.5, color="#f5f5f8", lw=0, zorder=0)
        ax.axhline(0.5, color=INK3, lw=0.5, ls=(0, (1, 2)), zorder=1)
        rr = [v15[(v15.arm == a) & (v15.cohort == c)].iloc[0] for a in ARMS_]
        for col, colr, mk, lab, ls in LINES:
            v = np.array([r[col] for r in rr])
            if col + "_sd" in v15.columns and col != "auc_age_alone":
                s = np.array([r[col + "_sd"] for r in rr])
                ax.fill_between(xs_, np.clip(v - s, 0.35, 1), np.clip(v + s, 0.35, 1), color=colr, alpha=0.13,
                                lw=0, zorder=2)
            ax.plot(xs_, v, color=colr, lw=1.5 if col != "auc_age_alone" else 1.1, ls=ls, zorder=3,
                    solid_capstyle="round")
            ax.scatter(xs_, v, marker=mk, s=18 if mk != "D" else 14, color=colr, edgecolor=SURFACE,
                       linewidth=0.6, zorder=4)
        # the one number per line a reader needs: where it ends up once age is matched
        ends = sorted([(rr[2][col], colr) for col, colr, *_ in LINES])
        last = -1
        for v, colr in ends:
            yv = max(v, last + 0.035)
            ax.text(2.16, yv, f"{v:.2f}", ha="left", va="center", fontsize=5.6, color=colr)
            last = yv
        ax.set_xlim(-0.25, 2.55); ax.set_ylim(0.38, 1.02)
        ax.set_xticks(xs_); ax.set_xticklabels(["merged", "age-\ncorrected", "age-\nmatched"], fontsize=5.6,
                                                 color=INK2)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", visible=True)
        if k:
            plt.setp(ax.get_yticklabels(), visible=False); ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        else:
            ax.set_ylabel("ROC-AUC within cohort"); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.text(0.0, 1.06, c, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.9, color=INK)
        ax.text(0.0, 1.0, f"{nd_[c]['Control']} ctrl · {nd_[c]['AD']} AD", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5.6, color=INK3)
    axd[0].text(0.02, 0.405, "chance", fontsize=5.4, color=INK3, style="italic")
    hl = [Line2D([], [], color=PLS, lw=1.5, marker="o", ms=4, label="PLS-DA"),
          Line2D([], [], color=CVAE_N, lw=1.5, marker="s", ms=3.8, label="CVAE agnostic"),
          Line2D([], [], color="#a4a4ad", lw=1.1, ls=(0, (3, 1.6)), marker="D", ms=3.4, label="age alone"),
          Rectangle((0, 0), 1, 1, facecolor=tint(INK2, 0.15), edgecolor="none", label="$\\pm$ SD over 20 splits")]
    axd[2].legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=6.2,
                  frameon=False, columnspacing=1.4)
    v9_ = [DATA / "validation" / "v9adexternal" / "loco_run_metrics.csv"] + \
          [Path(h) for h in __import__("glob").glob("/kaggle/input/**/loco_run_metrics.csv", recursive=True)]
    v9_ = [f for f in v9_ if Path(f).exists()]
    if v9_ and "GSE132903" in cohorts:
        r9 = pd.read_csv(v9_[0])
        axk = axd[cohorts.index("GSE132903")]
        axk.text(1.0, 0.72, f"cohort fully held out\n(separate run):\nPLS-DA {r9.auc_pls.mean():.3f}\n"
                            f"CVAE {r9.auc_cvae_neutral.mean():.3f}", ha="center", va="center", fontsize=5.6,
                 color=INK2, style="italic", linespacing=1.35)
    a0_ = axd[0]
    a0_.annotate("d", xy=(-0.42, 1.0), xycoords="axes fraction", xytext=(0, 36), textcoords="offset points",
                 fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, ha="left", va="bottom",
                 annotation_clip=False)
    a0_.annotate("Removing age leaves the disease signal; only age alone falls to chance", xy=(0, 1.0),
                 xycoords="axes fraction", xytext=(0, 34), textcoords="offset points", fontsize=8.6, color=INK,
                 ha="left", va="bottom", annotation_clip=False)
    a0_.annotate("test AUC within each cohort: as merged, with ageing slopes removed, and with donors matched on "
                 "age; donors unseen, cohorts seen in training", xy=(0, 1.0), xycoords="axes fraction",
                 xytext=(0, 24), textcoords="offset points", fontsize=6.8, color=INK3, ha="left", va="bottom",
                 annotation_clip=False)

    # ---- (e-g) confusion matrices, all three arms -------------------------
    for i, m in enumerate(arms):
        ax = fig.add_subplot(gs[2, i])
        if m in cm.index:
            r = cm.loc[m]
            M = np.array([[r.tn, r.fp], [r.fn, r.tp]], float)
        else:
            yy = (op.group == "AD").values; pp = (op.p_cvae_lat_neutral >= 0.5).values
            M = np.array([[(~yy & ~pp).sum(), (~yy & pp).sum()], [(yy & ~pp).sum(), (yy & pp).sum()]], float)
        Mn = M / M.sum(1, keepdims=True)
        ax.imshow(Mn, cmap={"pls": SEQ_PLS, "cvae_neutral": SEQ_CVAE_N, "cvae_true": SEQ_CVAE}[m], vmin=0, vmax=1)
        for u in range(2):
            for v in range(2):
                dark = Mn[u, v] > 0.55
                ax.text(v, u - 0.06, f"{Mn[u, v]:.2f}", ha="center", va="center", fontsize=10,
                        color=SURFACE if dark else INK)
                ax.text(v, u + 0.25, f"n = {int(M[u, v]):,}", ha="center", va="center", fontsize=5.6,
                        color=SURFACE if dark else INK2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "AD"], fontsize=6.4)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "AD"] if i == 0 else ["", ""], fontsize=6.4)
        ax.set_xlabel("Predicted", fontsize=6.8); ax.set_ylabel("True" if i == 0 else "", fontsize=6.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        sens, spec = Mn[1, 1], Mn[0, 0]
        sub = f"sens. {sens:.2f} · spec. {spec:.2f}"
        if m == "cvae_true":
            sub = "reference: sees the test label"
        panel(ax, "efg"[i], SHORT[m], sub, dx=-0.30 if i == 0 else -0.12)
        ax.set_position(ax.get_position().translated(0, -0.012))

    # ---- (h) predicted probability by diagnosis ----------------------
    ax = fig.add_subplot(gs[3, :])
    grid = np.linspace(0, 1, 300)
    from scipy.stats import gaussian_kde
    for j, (col, ls) in enumerate((("p_pls", "-"), ("p_cvae_lat_neutral", (0, (4, 2))))):
        for g in ("Control", "AD"):
            v = op.loc[op.group == g, col].values
            d = gaussian_kde(v)(grid)
            ax.plot(grid, d, color=GROUP_COLOR[g], lw=1.3, ls=ls)
            if j == 0:
                ax.fill_between(grid, 0, d, color=GROUP_COLOR[g], alpha=0.10, lw=0)
    ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(0, 1); ax.set_xlabel("Predicted $P(\\mathrm{AD})$ on held-out arrays")
    ax.set_ylabel("Density"); ygrid(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(handles=[*swatches([("Control", CTRL), ("AD", AD)]),
                       Line2D([], [], color=INK2, ls="-", label="PLS-DA"),
                       Line2D([], [], color=INK2, ls=(0, (4, 2)), label="CVAE agnostic")],
              loc="upper center", ncol=4, fontsize=6.2, frameon=True, framealpha=1.0, facecolor=SURFACE,
              edgecolor="none")
    panel(ax, "h", "Predicted probability by diagnosis",
          f"all {len(op):,} held-out array–split pairs", dx=-0.075)
    save(fig, "fig2_classification")


fig_classification(runs, S)

# =====================================================================
# Figure 3 -- latent geometry
# =====================================================================
def fig_geometry():
    from sklearn.metrics import silhouette_score
    from scipy.stats import gaussian_kde
    emb = load("latent_embeddings.csv")
    Zc = load("latent_coords_representative.csv")
    tags = [("PLS-DA", "PLS-DA", PLS), ("CVAE_neutral", "CVAE agnostic", CVAE_N),
            ("CVAE_true", "CVAE conditioned", CVAE_T)]
    projs = [(("pca1", "pca2"), "PCA"), (("tsne1", "tsne2"), "t-SNE")]
    if "umap1" in emb.columns and emb.umap1.notna().any():
        projs.append((("umap1", "umap2"), "UMAP"))
    zc = [f"z{k}" for k in range(1, 9)]
    sil = {(t, sp): silhouette_score(Zc[(Zc.model == t) & (Zc.split == sp)][zc],
                                     Zc[(Zc.model == t) & (Zc.split == sp)].group)
           for t, _, _ in tags for sp in ("train", "test")}

    nr = len(projs)
    fig = plt.figure(figsize=(TW, TW / 3 * nr + 1.05))
    gs = fig.add_gridspec(nr, 3, left=0.075, right=0.995, top=0.875, bottom=0.075, wspace=0.05, hspace=0.05)
    letters = iter("abcdefghi")
    for r, (proj, pname) in enumerate(projs):
        for c, (tag, tname, tcol) in enumerate(tags):
            ax = fig.add_subplot(gs[r, c])
            d = emb[emb.model == tag]
            X = d[list(proj)].values
            lo_, hi_ = np.percentile(X, [1, 99], axis=0); pad_ = 0.10 * (hi_ - lo_)
            lo_, hi_ = lo_ - pad_, hi_ + pad_
            tr = d[d.split == "train"]; te = d[d.split == "test"]
            gx = np.linspace(lo_[0], hi_[0], 200); gy = np.linspace(lo_[1], hi_[1], 200)
            GX, GY = np.meshgrid(gx, gy); G = np.vstack([GX.ravel(), GY.ravel()])
            # landscape learned from TRAINING arrays: P(AD | position), equal priors
            dens = {g: gaussian_kde(tr[tr.group == g][list(proj)].values.T, bw_method="scott")(G).reshape(GX.shape)
                    for g in ("AD", "Control")}
            post = dens["AD"] / (dens["AD"] + dens["Control"] + 1e-300)
            # pale stepped bands, so that the held-out markers -- coloured by diagnosis in
            # the bands' own two hues -- carry the eye and show the separation directly
            im = ax.contourf(gx, gy, post, levels=GEO_LEVELS, cmap=GEO_BANDS, norm=GEO_NORM, zorder=0)
            ax.contour(gx, gy, post, levels=[0.5], colors=[INK], linewidths=1.0, linestyles=[(0, (4, 2.2))],
                       zorder=2)
            ax.scatter(tr[proj[0]], tr[proj[1]], s=1.4, color=INK3, alpha=0.18, linewidth=0, zorder=1,
                       rasterized=True)
            # held-out arrays, judged against the training landscape
            kt = {g: gaussian_kde(tr[tr.group == g][list(proj)].values.T, bw_method="scott")
                  for g in ("AD", "Control")}
            pt = kt["AD"](te[list(proj)].values.T); pc = kt["Control"](te[list(proj)].values.T)
            isad = (te.group == "AD").values
            ok = (pt / (pt + pc + 1e-300) > 0.5) == isad
            for g, kw in (("Control", dict(marker="o", color=CTRL, s=13, edgecolor=SURFACE, linewidth=0.45)),
                          ("AD", dict(marker="^", color=AD, s=15, edgecolor=SURFACE, linewidth=0.45))):
                s = te[te.group == g]
                ax.scatter(s[proj[0]], s[proj[1]], zorder=4, **kw)
            ax.text(0.965, 0.035, f"correct side  {ok.mean():.0%}\ncontrols {ok[~isad].mean():.0%} · "
                                  f"AD {ok[isad].mean():.0%}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=5.8, color=INK, linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none", alpha=0.9), zorder=6)
            ax.set_xlim(lo_[0], hi_[0]); ax.set_ylim(lo_[1], hi_[1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(RULE); sp.set_linewidth(0.6)
            ax.text(0.035, 0.965, next(letters), transform=ax.transAxes, ha="left", va="top", fontsize=9,
                    fontweight="bold", family="DejaVu Sans", color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.75))
            if r == 0:
                ax.text(0.5, 1.10, tname, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
                        color=tcol)
                ax.text(0.5, 1.035, f"silhouette  held-out {sil[(tag, 'test')]:.2f}  ·  training "
                                    f"{sil[(tag, 'train')]:.2f}", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=6.3, color=INK2)
            if c == 0:
                ax.text(-0.06, 0.5, pname, transform=ax.transAxes, ha="right", va="center", rotation=90,
                        fontsize=8.4, color=INK2)
    hl = [Line2D([], [], marker="o", ls="", color=CTRL, markeredgecolor=SURFACE, markersize=5,
                 label="control, held out"),
          Line2D([], [], marker="^", ls="", color=AD, markeredgecolor=SURFACE, markersize=5.4, label="AD, held out"),
          Line2D([], [], marker="o", ls="", color=INK3, alpha=0.4, markersize=2.5, label="training arrays"),
          Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2.2)), label="$P = 0.5$ boundary")]
    fig.legend(handles=hl, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2, fontsize=7.0, frameon=False,
               columnspacing=1.6)
    cax = fig.add_axes([0.62, 0.022, 0.30, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cb.ax.tick_params(labelsize=6.0, length=2); cb.outline.set_visible(False)
    cb.set_label("$P(\\mathrm{AD} \\mid$ position$)$, learned from training arrays", fontsize=6.4, color=INK2,
                 labelpad=2)
    fig.text(0.075, 0.975, "Latent space of one representative split: the landscape learned in training, "
                           "and where unseen donors land", fontsize=8.8, color=INK, ha="left", va="top")
    fig.text(0.075, 0.952, "bands: P(AD) learned from training arrays, in steps of 0.25 · markers: held-out arrays, "
                           "coloured by diagnosis", fontsize=6.6, color=INK3, ha="left", va="top")
    save(fig, "fig3_latent_geometry")


fig_geometry()

# =====================================================================
# Figure 4 -- what the geometry is worth, and what conditioning buys
# =====================================================================
def fig_separation(S):
    from scipy.stats import ttest_rel, t as tdist
    # with two classes Davies-Bouldin = 2 / (inter/intra), so the two are one measurement
    specs = [("sil", "Silhouette", +1), ("ari", "Adjusted Rand", +1),
             ("iir", "Inter / intra distance\n(= 2 / Davies–Bouldin)", +1), ("ch", "Calinski–Harabasz", +1)]
    DPLS, DAGN, DCND = PLS, CVAE_N, CVAE_T
    arms = [("pls", DPLS, "PLS-DA"), ("cvae_neutral", DAGN, "CVAE\nagn."), ("cvae_true", DCND, "CVAE\ncond.")]
    fig = plt.figure(figsize=(TW, 6.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], hspace=0.44, wspace=0.42)

    # ---- (a) one paired slope chart per metric --------------------------------
    for k, (m, name, sgn) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        V = [runs[f"{m}_{a}"].values for a, _, _ in arms]
        for i in range(len(runs)):
            up = (V[1][i] - V[0][i]) * sgn > 0
            ax.plot([0, 1], [V[0][i], V[1][i]], color=DAGN if up else DPLS, lw=0.65,
                    alpha=0.62 if up else 0.95, zorder=2)
            ax.plot([1, 2], [V[1][i], V[2][i]], color=DCND, lw=0.5, alpha=0.55, zorder=1)
        for x, (v, (a, c, _)) in enumerate(zip(V, arms)):
            ax.scatter(np.full(len(v), x), v, s=6, color=c, lw=0, zorder=3)
            mu, sd = v.mean(), v.std()
            ax.add_patch(Rectangle((x - 0.17, mu - sd), 0.34, 2 * sd, facecolor=tint(c, 0.24), edgecolor="none",
                                   zorder=0))
            ax.plot([x - 0.22, x + 0.22], [mu, mu], color=c, lw=1.8, solid_capstyle="butt", zorder=4)
        wins = int(((V[1] - V[0]) * sgn > 0).sum())
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([a[2] for a in arms], fontsize=6.0)
        for t_, (_, c, _) in zip(ax.get_xticklabels(), arms):
            t_.set_color(c)
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=5.8)
        ax.set_xlim(-0.45, 2.45); ax.spines["bottom"].set_visible(False); ygrid(ax)
        if m == "ari":
            ax.axhline(0, color=INK3, lw=0.5, ls=(0, (1, 1.6)), zorder=1)
        if sgn < 0:
            ax.invert_yaxis()
        ax.text(0.5, 1.10, name.split("\n")[0], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, color=INK)
        ax.text(0.5, 1.03, f"CVAE agn. better in {wins}/20",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color=INK3)
    a0 = fig.axes[0]
    a0.annotate("a", xy=(-0.42, 1.10), xycoords="axes fraction", fontsize=9.5, fontweight="bold",
                family="DejaVu Sans", color=INK, ha="left", va="bottom", annotation_clip=False)
    a0.annotate("Every split, every metric", xy=(-0.42, 1.30), xycoords="axes fraction", fontsize=8.6,
                color=INK, ha="left", va="bottom", annotation_clip=False)
    a0.annotate("held-out arrays; one line per split; bar = mean, band = $\\pm$SD; higher = better throughout",
                xy=(-0.42, 1.22), xycoords="axes fraction",
                fontsize=6.4, color=INK3, ha="left", va="bottom", annotation_clip=False)

    # ---- (b) distribution of the per-split gain, two scales ----------------------
    from scipy.stats import gaussian_kde
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[3.1, 1.0], wspace=0.10)
    ari_slot = sub[1].subgridspec(3, 1, hspace=0.0)[1:2, 0]
    def rows_for(metrics, ratio):
        out = []
        for m, name, sgn in metrics:
            b_ = runs[f"{m}_pls"].values
            per = {}
            for arm in ("cvae_neutral", "cvae_true"):
                a_ = runs[f"{m}_{arm}"].values
                per[arm] = (np.log2(a_ / b_) * sgn) if ratio else (a_ - b_)
            out.append((name, per, ttest_rel(runs[f"{m}_cvae_neutral"], b_).pvalue))
        return out
    blocks = [(sub[0], rows_for([s for s in specs if s[0] != "ari"], True), True),
              (ari_slot, rows_for([s for s in specs if s[0] == "ari"], False), False)]
    axes_b = []
    for spec_, R_, ratio in blocks:
        ax = fig.add_subplot(spec_); axes_b.append(ax)
        allv = np.concatenate([np.r_[r[1]["cvae_neutral"], r[1]["cvae_true"]] for r in R_])
        lo, hi = min(allv.min(), 0) - 0.08 * np.ptp(allv), allv.max() + 0.08 * np.ptp(allv)
        for i, (name, per, p_) in enumerate(R_):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f3f3f7", lw=0, zorder=0)
            d = per["cvae_neutral"]
            g = np.linspace(d.min() - 0.05 * np.ptp(d), d.max() + 0.05 * np.ptp(d), 200)
            k = gaussian_kde(d)(g); k = k / k.max() * 0.36
            ax.fill_between(g, i - 0.04, i - 0.04 - k, color=tint(DAGN, 0.30), lw=0, zorder=1)
            ax.plot(g, i - 0.04 - k, color=DAGN, lw=0.8, zorder=2)
            ax.scatter(d, np.full(len(d), i + 0.10) + np.random.RandomState(i).uniform(-0.05, 0.05, len(d)),
                       s=7, color=DAGN, alpha=0.85, lw=0, zorder=3)
            mu = d.mean(); h = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            ax.plot([mu - h, mu + h], [i + 0.27, i + 0.27], color=INK, lw=1.4, solid_capstyle="butt", zorder=4)
            ax.scatter(mu, i + 0.27, s=26, marker="D", color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=5)
            dc = per["cvae_true"]; mc = dc.mean()
            hc = tdist.ppf(0.975, len(dc) - 1) * dc.std(ddof=1) / np.sqrt(len(dc))
            ax.plot([mc - hc, mc + hc], [i + 0.39, i + 0.39], color=DCND, lw=0.9, zorder=4)
            ax.scatter(mc, i + 0.39, s=13, marker="D", facecolor=SURFACE, edgecolor=DCND, linewidth=0.8, zorder=5)
            lab = (f"×{2 ** mu:.2f}  [{2 ** (mu - h):.2f}–{2 ** (mu + h):.2f}]" if ratio
                   else f"+{mu:.3f}  [{mu - h:.3f}–{mu + h:.3f}]")
            if ratio:
                ax.text(1.0, i - 0.30, lab, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=6.1, color=INK, fontweight="bold")
                ax.text(1.0, i - 0.12, pmath(p_), transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=5.7, color=INK2)
            else:
                ax.text(0.0, 1.36, "Adjusted Rand (k-means agreement)", transform=ax.transAxes, ha="left",
                        va="bottom", fontsize=6.5, color=INK)
                ax.text(0.0, 1.08, f"{lab}   {pmath(p_)}", transform=ax.transAxes, ha="left",
                        va="bottom", fontsize=5.9, color=INK, fontweight="bold")
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_xlim(lo, hi * 1.42 if ratio else hi)
        ax.set_ylim(len(R_) - 0.45, -0.55)
        ax.set_yticks(range(len(R_)))
        if ratio:
            ax.set_yticklabels([r[0] for r in R_], fontsize=6.5, linespacing=1.15)
            tv = [1, 1.5, 2, 3, 4]
            ax.set_xticks([np.log2(v) for v in tv]); ax.set_xticklabels([f"×{v:g}" for v in tv])
            ax.set_xlabel("fold change, CVAE agnostic / PLS-DA on the same split (log scale)", fontsize=6.6)
        else:
            ax.set_yticklabels([""])
            ax.set_xlabel("difference, CVAE agn. − PLS-DA", fontsize=6.4)
        ax.tick_params(axis="y", length=0); xgrid(ax)
        for s_ in ("left", "right"):
            ax.spines[s_].set_visible(False)
    hl = [Line2D([], [], marker="o", ls="", color=DAGN, markersize=3.5, label="one split"),
          Rectangle((0, 0), 1, 1, facecolor=tint(DAGN, 0.30), edgecolor=DAGN, lw=0.6, label="distribution"),
          Line2D([], [], marker="D", color=INK, lw=1.4, markersize=4.2, label="mean, 95% CI"),
          Line2D([], [], marker="D", color=DCND, markerfacecolor=SURFACE, markeredgecolor=DCND,
                 lw=0.9, markersize=3.6, label="CVAE conditioned (reference)")]
    axes_b[0].legend(handles=hl, loc="upper center", bbox_to_anchor=(0.66, -0.24), ncol=4, fontsize=6.2,
                     frameon=False, columnspacing=1.4)
    panel(axes_b[0], "b", "How much better the CVAE separates the classes, split by split",
          "fold change on three measures (left); adjusted Rand as a difference (right); right of the line = CVAE better", dx=-0.16)
    save(fig, "fig4_latent_separation")


fig_separation(S)

# =====================================================================
# Figure 5 -- posterior probability landscapes over the latent spaces
# =====================================================================
def fig_landscape():
    emb = load("latent_embeddings.csv")
    tags = ["PLS-DA", "CVAE_neutral"]
    projs = [(("pca1", "pca2"), "PCA"), (("tsne1", "tsne2"), "t-SNE")]
    if "umap1" in emb.columns and emb.umap1.notna().any():
        projs.append((("umap1", "umap2"), "UMAP"))

    fig, axes = plt.subplots(2, len(projs), figsize=(TW, TW / len(projs) * 2 + 0.7))
    axes = np.atleast_2d(axes)
    im = None
    for row, tag in enumerate(tags):
        for col, (proj, name) in enumerate(projs):
            ax = axes[row, col]
            sub = emb[emb.model == tag]
            XY = sub[list(proj)].values
            pad = 0.09 * (XY.max(0) - XY.min(0))
            gx = np.linspace(XY[:, 0].min() - pad[0], XY[:, 0].max() + pad[0], 220)
            gy = np.linspace(XY[:, 1].min() - pad[1], XY[:, 1].max() + pad[1], 220)
            # class-conditional KDEs -> posterior P(AD | position) at equal priors
            dens = {g: kde2d(sub.loc[sub.group == g, list(proj)].values, gx, gy, 1.15)
                    for g in ("AD", "Control")}
            post = dens["AD"] / (dens["AD"] + dens["Control"] + 1e-300)
            im = ax.pcolormesh(gx, gy, post, cmap=DIVERGING, vmin=0, vmax=1,
                               shading="gouraud", rasterized=True)
            ax.contour(gx, gy, post, levels=[0.5], colors=[INK2], linewidths=1.0,
                       linestyles=[(0, (3.5, 2.0))])
            # The surface already carries class by colour, so the marks must not:
            # a blue dot on a blue field is invisible. Shape plus fill/no-fill in
            # ink reads on every part of the ramp.
            mark = {"Control": dict(marker="o", facecolor=SURFACE, edgecolor=INK),
                    "AD": dict(marker="^", facecolor=INK, edgecolor=SURFACE)}
            for g in ("Control", "AD"):
                s = sub[sub.group == g]
                ax.scatter(s[proj[0]], s[proj[1]], s=13, linewidth=0.6,
                           label=g, zorder=3, **mark[g])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
            for sp in ax.spines.values():
                sp.set_visible(False)
            panel(ax, "abcdef"[row * len(projs) + col],
                  name if row == 0 else None, dx=-0.11)
            if col == 0:
                ax.annotate(MODEL_LABEL[tag], xy=(-0.06, 0.5),
                            xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=7.6, color=INK2,
                            annotation_clip=False)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", markerscale=1.25, labelcolor=INK,
                          facecolor=SURFACE, framealpha=0.88, frameon=True,
                          edgecolor="none", borderpad=0.45)
    cax = fig.add_axes([0.24, -0.022, 0.44, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Posterior $P(\\mathrm{AD}\\mid$ latent position$)$ from class-conditional KDE",
                 fontsize=7, color=INK2)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    fig.text(0.72, -0.020, "dashed contour: $P=0.5$", fontsize=6.6, color=INK2,
             ha="left", style="italic")
    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    save(fig, "fig5_probability_landscape")



# fig_landscape(): merged into Figure 3 (latent geometry with the training-set landscape)

# =====================================================================
# Figure 6 -- reconstruction fidelity
# =====================================================================
def fig_reconstruction(S):
    sc = load("recon_scatter_representative.csv")
    oof = load("oof_reconstruction.csv")

    fig = plt.figure(figsize=(TW, 6.5))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.50, bottom=0.10)
    lim = (-4.5, 4.5)      # where 99.9% of the values are; the rest is noted on the panel
    beyond = np.mean((sc.observed.abs() > 4.5) | (sc.recon_pls.abs() > 4.5) | (sc.recon_cvae.abs() > 4.5))

    def clean_box(ax, v, x, c, w=0.34, label_y=None, side_label=True):
        """Slim box and 5-95% whisker over a faint strip; median and mean written above."""
        rs = np.random.RandomState(int(x * 10))
        ax.scatter(x + rs.uniform(-w * 0.42, w * 0.42, len(v)), v, s=2.2, color=c, alpha=0.18,
                   linewidth=0, zorder=1, rasterized=True)
        q5, q1, q2, q3, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([x, x], [q5, q1], color=c, lw=1.0, zorder=2); ax.plot([x, x], [q3, q95], color=c, lw=1.0, zorder=2)
        ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, facecolor=tint(c, 0.18), edgecolor=c, lw=1.1,
                               zorder=3))
        ax.plot([x - w / 2, x + w / 2], [q2, q2], color=c, lw=2.0, solid_capstyle="butt", zorder=4)
        ax.scatter([x], [v.mean()], s=16, marker="D", color=SURFACE, edgecolor=c, linewidth=1.0, zorder=5)
        # values written beside the box, level with it, clear of the points
        if side_label: ax.text(x + w / 2 + 0.06, q2, f"median {q2:.3f}\nmean {v.mean():.3f}", ha="left", va="center",
                fontsize=5.9, color=c, linespacing=1.3, zorder=6,
                bbox=dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none", alpha=0.85))

    for i, (col, name, cmap, key) in enumerate(
            (("recon_pls", "PLS-DA", SEQ_PLS, "mse_pls_mean_mean"),
             ("recon_cvae", "CVAE (label-conditioned)", SEQ_CVAE, "mse_cvae_true_mean_mean"))):
        ax = fig.add_subplot(gs[0, i]); ax.set_box_aspect(0.82)
        x, y = sc.observed.values, sc[col].values
        ax.hexbin(x, y, gridsize=52, bins="log", cmap=cmap, mincnt=1, linewidths=0, rasterized=True,
                  extent=(*lim, *lim))
        ax.plot(lim, lim, color=INK2, lw=0.7, ls=(0, (3.5, 2.5)))
        b, a0 = np.polyfit(x, y, 1)
        ax.plot(np.array(lim), a0 + b * np.array(lim), color=MODEL_COLOR["pls" if i == 0 else "cvae_true"], lw=1.1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_xlabel("Observed expression ($z$)")
        ax.set_ylabel("Reconstructed ($z$)" if i == 0 else "")
        ax.text(0.04, 0.95, f"mean MSE {S[key]:.3f}\nfitted slope {b:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.35)
        ax.text(0.96, 0.04, f"dashed = identity · {beyond:.1%} beyond ±4.5", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.6, color=INK3)
        panel(ax, "ab"[i], name, "one representative held-out split" if i == 0 else
              "decoder output saved for this arm only", dx=-0.20)

    # (c) paired per-array error
    ax = fig.add_subplot(gs[1, 0])
    d = {"PLS-DA": oof.mse_pls.values, "CVAE (agn.)": oof.mse_cvae_neutral.values}
    for j, (nm, color) in enumerate((("PLS-DA", PLS), ("CVAE (agn.)", CVAE_N))):
        clean_box(ax, d[nm], j, color, w=0.34)
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(d)); ax.tick_params(axis="x", length=0)
    for t_, c in zip(ax.get_xticklabels(), (PLS, CVAE_N)):
        t_.set_color(c)
    ax.set_yscale("log"); ax.set_ylim(0.07, 22); ax.set_xlim(-0.45, 1.85)
    ax.set_yticks([0.1, 0.3, 1, 3, 10]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
    ax.set_ylabel("Per-array MSE (log)"); ygrid(ax)
    p, r = S["recon_cvae_neutral_vs_pls_wilcoxon_p"], S["recon_cvae_neutral_vs_pls_rank_biserial"]
    bracket(ax, 0, 1, 11.5, f"{stars(p)}  Wilcoxon {pmath(p)},  $r = {r:.2f}$", tick=1.8, fontsize=6.4)
    panel(ax, "c", "Every held-out array–split pair",
          f"$n = {int(S['recon_n_oof_samples']):,}$, paired; box = IQR, whiskers 5–95%, ◇ = mean", dx=-0.20)

    # (d) does either model reconstruct one diagnosis better?
    ax = fig.add_subplot(gs[1, 1])
    pos = [0, 1.0, 2.6, 3.6]; k = 0; tl = []
    for nm, col, color, tag in (("PLS-DA", "mse_pls", PLS, "pls"),
                                ("CVAE (agn.)", "mse_cvae_neutral", CVAE_N, "cvaeneutral")):
        for g in ("Control", "AD"):
            v = oof.loc[oof.group == g, col].values
            clean_box(ax, v, pos[k], GROUP_COLOR[g], w=0.34, side_label=False)
            tl.append((f"{g}\nmedian {np.median(v):.3f}\nmean {v.mean():.3f}", GROUP_COLOR[g]))
            k += 1
        pg = S[f"recon_group_{tag}_mannwhitney_p"]
        a_, b_ = pos[k - 2], pos[k - 1]
        bracket(ax, a_, b_, 6.0, f"{stars(pg)}  {pmath(pg)}", tick=0.6, fontsize=6.2)
        ax.text((a_ + b_) / 2, -0.255, nm, transform=ax.get_xaxis_transform(), ha="center", fontsize=7.4,
                color=color)
    ax.set_xticks(pos); ax.set_xticklabels([t[0] for t in tl], fontsize=6.0, linespacing=1.3)
    for t_, (_, c) in zip(ax.get_xticklabels(), tl):
        t_.set_color(c)
    ax.tick_params(axis="x", length=0)
    ax.set_yscale("log"); ax.set_ylim(0.07, 11); ax.set_xlim(-0.6, 4.2)
    ax.set_yticks([0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.1", "0.3", "1", "3"])
    ax.set_ylabel("Per-array MSE (log)"); ygrid(ax)
    panel(ax, "d", "Does error depend on diagnosis?", "Mann–Whitney on held-out arrays, within each model",
          dx=-0.20)
    save(fig, "fig6_reconstruction")


fig_reconstruction(S)

# =====================================================================
# Figure 7 -- out-of-fold uncertainty
# =====================================================================
def fig_uncertainty(S):
    unc = load("oof_uncertainty.csv")
    panels = [("kl_neutral", "KL divergence", "label-agnostic CVAE", True),
              ("kl_true", "KL divergence", "label-conditioned CVAE (reference)", True),
              ("meanvar_neutral", "Mean latent variance", "label-agnostic CVAE", False),
              ("meanvar_true", "Mean latent variance", "label-conditioned CVAE (reference)", False)]

    def clean_box(ax, v, x, c, w=0.36):
        rs = np.random.RandomState(int(x * 10) + 3)
        ax.scatter(x + rs.uniform(-w * 0.45, w * 0.45, len(v)), v, s=2.2, color=c, alpha=0.18,
                   linewidth=0, zorder=1, rasterized=True)
        q5, q1, q2, q3, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([x, x], [q5, q1], color=c, lw=1.0, zorder=2); ax.plot([x, x], [q3, q95], color=c, lw=1.0, zorder=2)
        ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, facecolor=tint(c, 0.18), edgecolor=c, lw=1.1, zorder=3))
        ax.plot([x - w / 2, x + w / 2], [q2, q2], color=c, lw=2.0, solid_capstyle="butt", zorder=4)
        ax.scatter([x], [v.mean()], s=16, marker="D", color=SURFACE, edgecolor=c, linewidth=1.0, zorder=5)
        return q2, v.mean()

    fig, axes = plt.subplots(2, 2, figsize=(TW, 6.4))
    fig.subplots_adjust(wspace=0.32, hspace=0.62, bottom=0.14, top=0.93)
    for i, (col, ylab, title, logy) in enumerate(panels):
        ax = axes.ravel()[i]
        tl = []
        for j, g in enumerate(("Control", "AD")):
            v = unc.loc[unc.group == g, col].values
            med, mean = clean_box(ax, v, j, GROUP_COLOR[g])
            fmt = (lambda z: f"{z:.3f}") if not logy else (lambda z: f"{z:.2f}")
            tl.append(f"{g}\nmedian {fmt(med)}\nmean {fmt(mean)}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(tl, fontsize=6.0, linespacing=1.3)
        for t_, g in zip(ax.get_xticklabels(), ("Control", "AD")):
            t_.set_color(GROUP_COLOR[g])
        ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, 1.6)
        p = S[f"{col}_mannwhitney_p"]
        if logy:
            ax.set_yscale("log"); ax.set_ylim(0.003, 150)
            ax.set_yticks([0.01, 0.1, 1, 10, 100]); ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
            bracket(ax, 0, 1, 45, f"{stars(p)}  Mann–Whitney {pmath(p)}", tick=12, fontsize=6.2)
            ax.set_ylabel(f"{ylab}, nats (log)" if i % 2 == 0 else "")
        else:
            ax.set_ylim(0.38, 1.13)
            bracket(ax, 0, 1, 1.065, f"{stars(p) or 'n.s.'}  Mann–Whitney {pmath(p)}", tick=0.015, fontsize=6.2)
            ax.set_ylabel(ylab if i % 2 == 0 else "")
        ygrid(ax)
        delta = S[f"{col}_pct_increase_AD"]
        panel(ax, "abcd"[i], f"{ylab} — {title.split(' CVAE')[0]}",
              f"mean in AD {delta:+.1f}% vs control" + (" · reference arm" if "reference" in title else ""), dx=-0.20)
    fig.text(0.5, 0.012, f"Held-out arrays pooled over the 20 splits ({int(S['unc_n_AD']):,} AD, "
                         f"{int(S['unc_n_control']):,} control). Box = IQR, whiskers 5–95%, open diamond = mean.\n"
                         "KL: the median is lower but the mean higher in AD, so the difference sits in the upper tail.",
             ha="center", fontsize=6.4, color=INK2, style="italic")
    save(fig, "fig7_uncertainty")


fig_uncertainty(S)

# =====================================================================
# Figure 8 -- gene discovery and its discordance
# =====================================================================
def fig_genes(S):
    import glob as _glob
    from pathlib import Path as _P
    pls = load("PLSDA_gene_significance.csv").set_index("gene")
    cv = load("CVAE_gene_significance.csv").set_index("gene")
    de = load("gene_de_stats.csv").set_index("gene")
    ps, cs = set(pls.index[pls.significant]), set(cv.index[cv.significant])
    both = ps & cs
    prov = {g: ("both" if g in both else "pls" if g in ps else "cvae") for g in ps | cs}
    PC = {"pls": PLS, "both": ACCENT, "cvae": CVAE_N}
    PM = {"pls": "o", "both": "D", "cvae": "s"}
    PN = {"pls": "PLS-DA only", "both": "both models", "cvae": "CVAE only"}

    def _find(rel, marker):
        f = DATA / "validation" / rel
        if f.exists():
            return f
        hits = sorted(_glob.glob(f"/kaggle/input/**/{marker}", recursive=True))
        return _P(hits[0]) if hits else None
    # replication in the four cohorts V7 never used (meta-analysis, new cohorts only)
    fm = _find("v10adgenevalidation/meta_analysis_new_cohorts_only.csv", "meta_analysis_new_cohorts_only.csv")
    M = pd.read_csv(fm, index_col=0) if fm is not None else pd.DataFrame(columns=["pooled", "q", "k"])
    bg = M.join(de, how="inner")
    rep_all = (bg.q < 0.05) & (np.sign(bg.pooled) == np.sign(bg.logFC))
    rep = lambda g: bool(rep_all.get(g, False))
    # V15 age analyses
    age_sel = {}
    for arm, tag in (("A_age_corrected", "corr"), ("B_age_matched", "match")):
        sets = []
        for f_ in ("PLSDA_gene_significance.csv", "CVAE_gene_significance.csv"):
            ff = _find(f"v15adage/{arm}/{f_}", f"{arm}/{f_}")
            d_ = pd.read_csv(ff) if ff is not None else pd.DataFrame(columns=["gene", "significant"])
            sets.append(set(d_.gene[d_.significant]))
        age_sel[tag] = sets

    # landscape: a-d on the left, the mirror chart e down the full right-hand side
    fig = plt.figure(figsize=(9.6, 7.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.10, left=0.06, right=0.99, top=0.90,
                             bottom=0.07)
    gs = outer[0].subgridspec(2, 3, height_ratios=[0.62, 1.38], hspace=0.26, wspace=0.62)

    # ---- (a, b) each gene against its own permutation null ---------------------------
    allg = list(de.index)
    for j, (real_f, null_f, sel_, mdl, c, lab) in enumerate(
            (("vip_real.npy", "vip_null.npy", ps, "PLS-DA", PLS, "VIP"),
             ("ig_real.npy", "ig_null.npy", cs, "CVAE", CVAE_N, "|IG|"))):
        ax = fig.add_subplot(gs[0, j]); ax.set_box_aspect(1)
        mr, mn = np.load(DATA / real_f).mean(0), np.load(DATA / null_f).mean(0)
        si = np.array([g in sel_ for g in allg])
        lo, hi = np.percentile(np.r_[mr, mn], 0.5), max(mr.max(), mn.max()) * 1.2
        GN = LinearSegmentedColormap.from_list(f"gn{j}", ["#ffffff", "#ececf0", "#d2d2da", "#a9a9b5"])
        ax.hexbin(np.log10(mn[~si]), np.log10(mr[~si]), gridsize=36, bins="log", cmap=GN, mincnt=1,
                  linewidths=0, rasterized=True, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)))
        ax.plot([np.log10(lo), np.log10(hi)], [np.log10(lo), np.log10(hi)], color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.scatter(np.log10(mn[si]), np.log10(mr[si]), s=9, color=c, edgecolor=SURFACE, lw=0.35, zorder=3)
        ax.set_xlim(np.log10(lo), np.log10(hi)); ax.set_ylim(np.log10(lo), np.log10(hi))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{lab}, permuted labels (log)", fontsize=6.2); ax.set_ylabel(f"{lab}, real labels (log)", fontsize=6.2)
        fold = np.median(mr[si] / mn[si]); bgf = np.median(mr[~si] / mn[~si])
        ax.text(0.96, 0.17, f"{si.sum()} selected: {fold:.1f}× null", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=5.9, color=c)
        ax.text(0.96, 0.05, f"all others: {bgf:.1f}× null", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color=INK3)
        panel(ax, "ab"[j], f"{mdl}: real vs null", "each gene vs itself, permuted labels", dx=-0.14)

    # ---- (c) two rankings of one genome ----------------------------------------------
    ax = fig.add_subplot(gs[0, 2]); ax.set_box_aspect(1)
    rp = np.log10(pls.mean_score.rank(ascending=False) / len(pls))
    rc = np.log10(cv.mean_score.reindex(pls.index).rank(ascending=False) / len(pls))
    GRE = LinearSegmentedColormap.from_list("gre", ["#ffffff", "#ececf0", "#d6d6dd", "#b8b8c2"])
    ax.hexbin(rp, rc, gridsize=38, bins="log", cmap=GRE, mincnt=1, linewidths=0, rasterized=True,
              extent=(-4.15, 0, -4.15, 0))
    ax.plot([-4.15, 0], [-4.15, 0], color=INK3, lw=0.5, ls=(0, (3, 2)))
    for k in ("pls", "cvae", "both"):
        g_ = [g for g, v in prov.items() if v == k]
        ax.scatter(rp[g_], rc[g_], s=18 if k == "both" else 10, marker=PM[k], color=PC[k], edgecolor=SURFACE,
                   linewidth=0.4, zorder=3)
    tk = [-4, -3, -2, -1, 0]; tl = ["0.01%", "0.1%", "1%", "10%", "100%"]
    ax.set_xticks(tk); ax.set_xticklabels(tl); ax.set_yticks(tk); ax.set_yticklabels(tl)
    ax.set_xlim(-4.15, 0); ax.set_ylim(-4.15, 0)
    ax.set_xlabel("PLS-DA rank (top fraction)", fontsize=6.2)
    ax.set_ylabel("CVAE rank (top fraction)", fontsize=6.2)
    ax.tick_params(labelsize=5.6)
    fig.canvas.draw()
    bl = sorted(both)
    # label only the shared genes that also replicate in unseen cohorts: the panel stays readable
    bl = [g for g in bl if rep(g)]
    # names in a column in the empty upper-left corner, each tied to its point by a hairline
    column_labels(ax, rp[bl].values, rc[bl].values, bl, [ACCENT] * len(bl), -4.05, -2.75, -0.55, "right",
                  fontsize=5.2, spacing=0.30)
    for t_ in ax.texts:
        ap = getattr(t_, "arrow_patch", None)
        if ap is not None:
            ap.set_alpha(0.6); ap.set_linewidth(0.35)
    ax.text(0.97, 0.03, "named: shared and replicated", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.4, color=INK3, style="italic")
    panel(ax, "c", "Two rankings, one genome",
          f"genome-wide $\\rho$ = {S['spearman_rho_vip_ig']:.3f}", dx=-0.22)

    # ---- (c) volcano with every selected gene named -----------------------------------
    ax = fig.add_subplot(gs[1, :])
    d = de.copy(); d["nlp"] = -np.log10(d.DE_pvalue.clip(lower=1e-300))
    ns = d[~d.index.isin(prov)]
    ax.scatter(ns.logFC, ns.nlp, s=1.4, color="#c9c9d1", linewidth=0, rasterized=True, zorder=1)
    sel = d[d.index.isin(prov)].copy(); sel["k"] = [prov[g] for g in sel.index]
    sel["rep"] = [rep(g) for g in sel.index]
    for k in ("pls", "cvae", "both"):
        s_ = sel[sel.k == k]
        ax.scatter(s_.logFC, s_.nlp, s=22 if k == "both" else 14, marker=PM[k], color=PC[k], edgecolor=SURFACE,
                   linewidth=0.45, zorder=3)
    r_ = sel[sel.rep]
    ax.scatter(r_.logFC, r_.nlp, s=70, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=2)
    ax.axvline(0, color=INK3, lw=0.6, ls=(0, (3, 2)))
    xmin_, xmax_ = d.logFC.min(), d.logFC.max(); span_ = xmax_ - xmin_
    ax.set_xlim(xmin_ - 0.42 * span_, xmax_ + 0.40 * span_)
    top_ = sel.nlp.max() * 1.04
    ax.set_ylim(-0.02 * top_, top_)
    # two label columns per side, alternating by height, so each column is half as dense and
    # every leader stays short; leaders are hairlines so the names, not the lines, carry the eye
    for side, s_ in (("left", sel[sel.logFC < 0]), ("right", sel[sel.logFC >= 0])):
        s_ = s_.sort_values("nlp")
        for col_i in (1, 0):          # outer column first, so inner names mask its leaders
            part = s_.iloc[col_i::2]
            xt = (xmin_ - (0.06 + 0.17 * col_i) * span_) if side == "left" else (xmax_ + (0.06 + 0.17 * col_i) * span_)
            column_labels(ax, part.logFC.values, part.nlp.values, list(part.index), [PC[k] for k in part.k],
                          xt, -0.01 * top_, top_ * 0.985, side, fontsize=5.0, spacing=top_ * 0.026)
    for ln in ax.texts:
        ap = getattr(ln, "arrow_patch", None)
        if ap is not None:
            ap.set_alpha(0.35); ap.set_linewidth(0.3)
            ln.set_bbox(dict(boxstyle="square,pad=0.08", facecolor=SURFACE, edgecolor="none"))
    ax.set_xlabel("Standardised expression difference, AD − control (merged cohorts)")
    ax.set_ylabel("$-\\log_{10}$ $p$ (differential expression)"); ygrid(ax)
    hl = [Line2D([], [], marker=PM[k], ls="", color=PC[k], markersize=4.5,
                 label=f"{PN[k]} ({sum(v == k for v in prov.values())})") for k in ("pls", "both", "cvae")]
    hl += [Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK, markersize=7,
                  label="replicates in independent cohorts"),
           Line2D([], [], marker="o", ls="", color="#c9c9d1", markersize=3, label="not selected")]
    ax.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=6.2, frameon=True,
              framealpha=1.0, facecolor=SURFACE, edgecolor="none")
    panel(ax, "d", "Where every selected gene sits genome-wide",
          "selection is not simply the strongest differential expression; names coloured by model", dx=-0.075)

    # ---- (e) mirror chart: the leading genes of each model, side by side ----------------
    top_n = 25
    tp = list(pls.mean_score.sort_values(ascending=False).index[:top_n])
    tc = list(cv.mean_score.sort_values(ascending=False).index[:top_n])
    union = list(dict.fromkeys(tp + tc))
    rk_p = pls.mean_score.rank(ascending=False); rk_c = cv.mean_score.rank(ascending=False)
    union.sort(key=lambda g: min(rk_p[g], rk_c[g]) + 0.001 * (rk_p[g] + rk_c[g]))
    sub = outer[1].subgridspec(1, 3, width_ratios=[1, 0.62, 1], wspace=0.0)
    axL, axM, axR = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[2])
    y = np.arange(len(union))
    pmax, cmax = pls.mean_score.max(), cv.mean_score.max()
    for ax_, T, mx, c, sel_, sgn in ((axL, pls, pmax, PLS, ps, -1), (axR, cv, cmax, CVAE_N, cs, 1)):
        v = T.loc[union, "mean_score"].values / mx; e = T.loc[union, "std_score"].values / mx
        on = np.array([g in sel_ for g in union])
        ax_.barh(y, sgn * v, 0.66, color=[c if o else tint(c, 0.25) for o in on], lw=0)
        ax_.errorbar(sgn * v, y, xerr=e, fmt="none", ecolor=[INK2 if o else RULE for o in on], elinewidth=0.6,
                     capsize=0)
        for yi, g in zip(y, union):
            r_ = int((rk_p if T is pls else rk_c)[g])
            ax_.text(sgn * (v[yi] + e[yi] + 0.03), yi, f"#{r_}", ha="left" if sgn > 0 else "right", va="center",
                     fontsize=4.9, color=c if on[yi] else INK3)
        ax_.set_ylim(len(union) - 0.4, -0.6); ax_.set_yticks([])
        ax_.set_xlim((-1.35, 0) if sgn < 0 else (0, 1.35))
        ax_.set_xticks([sgn * t for t in (0, 0.5, 1.0)]); ax_.set_xticklabels(["0", "0.5", "1"], fontsize=5.8)
        ax_.grid(axis="x", visible=True)
        ax_.spines["left" if sgn > 0 else "right"].set_visible(False)
        if sgn < 0:
            ax_.spines["left"].set_visible(False); ax_.spines["right"].set_visible(True)
        ax_.set_xlabel(("PLS-DA VIP" if T is pls else "CVAE |IG|") + " (relative)", fontsize=6.4)
    axM.set_ylim(len(union) - 0.4, -0.6); axM.axis("off")
    for yi, g in zip(y, union):
        k = prov.get(g)
        col = PC[k] if k else INK3
        arrow = "↑" if de.loc[g, "logFC"] > 0 else "↓"
        axM.text(0.5, yi, f"{g}", ha="center", va="center", fontsize=6.0, fontstyle="italic", color=col,
                 fontweight="bold" if k == "both" else "normal")
        axM.text(0.92, yi, arrow, ha="center", va="center", fontsize=6.4, color=AD if arrow == "↑" else CTRL)
        if yi % 2 == 0:
            for ax_ in (axL, axR):
                ax_.axhspan(yi - 0.5, yi + 0.5, color="#f5f5f8", lw=0, zorder=0)
    axL.annotate("e", xy=(-0.20, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                 fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom", annotation_clip=False)
    axL.annotate(f"Leading genes of each model (top {top_n} of each)",
                 xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 21), textcoords="offset points", fontsize=8.6,
                 color=INK, va="bottom", annotation_clip=False)
    axL.annotate("mean $\\pm$ SD over 20 splits, scaled to each model's top gene; #rank;\npale = not selected "
                 "by that model; bold name = both models; arrow = direction in AD", xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 3),
                 textcoords="offset points", fontsize=6.0, color=INK3, va="bottom", annotation_clip=False,
                 linespacing=1.3)
    save(fig, "fig8_gene_discovery")


fig_genes(S)

# =====================================================================
# Figure 9 -- directional expression profile of every selected gene
# =====================================================================
def fig_profile():
    import glob as _glob
    from pathlib import Path as _P
    pls = load("PLSDA_gene_significance.csv")
    cv = load("CVAE_gene_significance.csv")
    de = load("gene_de_stats.csv").set_index("gene")
    pls_set = set(pls.loc[pls.significant, "gene"]); cv_set = set(cv.loc[cv.significant, "gene"])
    both = pls_set & cv_set
    # within each block, genes run from most reduced to most increased in AD, so both traces
    # sweep smoothly round the circle instead of zig-zagging gene to gene
    dz = lambda g: de.loc[g, "mean_z_AD"] - de.loc[g, "mean_z_control"]
    groups = [("PLS-DA only", sorted(pls_set - cv_set, key=dz), PLS),
              ("Found by both", sorted(both, key=dz), ACCENT),
              ("CVAE only", sorted(cv_set - pls_set, key=dz), CVAE_N)]
    # independent replication (4 cohorts never used), for a marker at each name
    f_ = DATA / "validation" / "v10adgenevalidation" / "meta_analysis_new_cohorts_only.csv"
    if not f_.exists():
        hits = sorted(_glob.glob("/kaggle/input/**/meta_analysis_new_cohorts_only.csv", recursive=True))
        f_ = _P(hits[0]) if hits else None
    M = pd.read_csv(f_, index_col=0) if f_ is not None else pd.DataFrame(columns=["pooled", "q"])
    rep = lambda g: g in M.index and M.loc[g, "q"] < 0.05 and np.sign(M.loc[g, "pooled"]) == np.sign(de.loc[g, "logFC"])

    gap = 2.2                                   # empty spokes between blocks
    slots, genes, gcol = [], [], []
    pos = 0.0
    for name, gl, c in groups:
        for g in gl:
            slots.append(pos); genes.append(g); gcol.append(c); pos += 1
        pos += gap
    n_slots = pos
    ang = np.array(slots) / n_slots * 2 * np.pi
    ad = np.array([de.loc[g, "mean_z_AD"] for g in genes]); ct = np.array([de.loc[g, "mean_z_control"] for g in genes])
    r0, sc = 1.0, 1.15                          # zero ring and radial scale
    rA, rC = r0 + sc * ad, r0 + sc * ct
    r_out = r0 + sc * max(np.abs(np.r_[ad, ct])) + 0.12

    fig = plt.figure(figsize=(TW, TW + 0.35))
    ax = fig.add_axes([0.13, 0.11, 0.74, 0.74], projection="polar")
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, r_out + 0.02)

    # provenance wedges and rim arcs
    start = 0.0
    for name, gl, c in groups:
        a0 = (start - 0.5) / n_slots * 2 * np.pi; a1 = (start + len(gl) - 0.5) / n_slots * 2 * np.pi
        arc = np.linspace(a0, a1, 160)
        ax.fill_between(arc, 0.18, r_out, color=c, alpha=0.045, lw=0, zorder=0)
        ax.plot(arc, np.full_like(arc, r_out), color=c, lw=2.6, solid_capstyle="butt", zorder=5)
        mid = (a0 + a1) / 2
        ax.text(mid, r_out * 1.36, f"{name}\n{len(gl)} genes", ha="center", va="center", fontsize=7.6,
                color=c, linespacing=1.25, fontweight="bold")
        start += len(gl) + gap

    # reference rings
    ring = np.linspace(0, 2 * np.pi, 400)
    for z, ls_, lw_ in ((-0.5, "-", 0.4), (0.0, (0, (3, 2)), 0.9), (0.5, "-", 0.4)):
        ax.plot(ring, np.full_like(ring, r0 + sc * z), color=INK3 if z else INK2, lw=lw_, ls=ls_, zorder=1)
    for z in (-0.5, 0.0, 0.5):
        ax.text(np.deg2rad(-3.5), r0 + sc * z, f"{z:+.1f}" if z else "0", ha="center", va="center",
                fontsize=5.4, color=INK3, bbox=dict(boxstyle="square,pad=0.05", facecolor=SURFACE, edgecolor="none"))

    # one radial segment per gene: control to AD, coloured by direction
    for a, r1, r2 in zip(ang, rC, rA):
        ax.plot([a, a], [r1, r2], color=AD if r2 > r1 else CTRL, lw=1.6, alpha=0.38, solid_capstyle="butt", zorder=2)
    # smooth traces within each block
    k = 0
    for name, gl, c in groups:
        sl = slice(k, k + len(gl))
        ax.plot(ang[sl], rC[sl], color=CTRL, lw=1.0, zorder=3)
        ax.plot(ang[sl], rA[sl], color=AD, lw=1.0, ls=(0, (3.5, 1.8)), zorder=3)
        k += len(gl)
    ax.scatter(ang, rC, s=9, color=CTRL, edgecolor=SURFACE, linewidth=0.3, zorder=4)
    ax.scatter(ang, rA, s=11, marker="^", color=AD, edgecolor=SURFACE, linewidth=0.3, zorder=4)

    # names along their own spokes; a small dot marks independent replication
    for a, g, c in zip(ang, genes, gcol):
        phi = (90.0 - np.degrees(a)) % 360
        flip = 90 < phi < 270
        if rep(g):
            ax.scatter([a], [r_out + 0.05], s=5, color=INK, zorder=6, clip_on=False)
        ax.text(a, r_out + 0.10, g, rotation=phi - 180 if flip else phi, rotation_mode="anchor",
                ha="right" if flip else "left", va="center", fontsize=4.9, fontstyle="italic", color=c,
                fontweight="bold" if g in both else "normal", clip_on=False)

    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False); ax.spines["polar"].set_visible(False)
    # centre key
    n_up = int((ad > ct).sum())
    ax.text(0, 0, f"{len(genes)} genes\n{n_up} higher in AD\n{len(genes) - n_up} lower", ha="center", va="center",
            fontsize=6.8, color=INK, linespacing=1.35, zorder=6)
    fig.legend(handles=[Line2D([], [], color=CTRL, lw=1.1, marker="o", markersize=3.8, label="control mean"),
                        Line2D([], [], color=AD, lw=1.1, ls=(0, (3.5, 1.8)), marker="^", markersize=4.2,
                               label="AD mean"),
                        Line2D([], [], color=AD, lw=2.2, alpha=0.45, label="higher in AD"),
                        Line2D([], [], color=CTRL, lw=2.2, alpha=0.45, label="lower in AD"),
                        Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2)), label="cohort mean ($z$ = 0)"),
                        Line2D([], [], marker="o", ls="", color=INK, markersize=2.6, label="replicates in unseen cohorts")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.4, frameon=False,
               columnspacing=1.6)
    fig.text(0.04, 0.985, "Directional expression of every selected gene", fontsize=8.8, color=INK, va="top")
    fig.text(0.04, 0.962, "radius = mean $z$ in each diagnostic group; genes sorted within each block from most "
                          "reduced to most increased in AD; shared genes in bold", fontsize=6.4, color=INK3, va="top")
    save(fig, "fig9_expression_profile")


fig_profile()

# =====================================================================
# Figure 10 -- how each latent dimension maps back onto genes
# =====================================================================
def fig_latent_map():
    pls = load("PLSDA_gene_significance.csv")
    cv = load("CVAE_gene_significance.csv")
    snr = load("latent_snr.csv")
    genes_all = list(load("gene_de_stats.csv").gene)
    gidx = {g: i for i, g in enumerate(genes_all)}
    ps, cs = set(pls.loc[pls.significant, "gene"]), set(cv.loc[cv.significant, "gene"])
    WP = np.abs(np.load(DATA / "latent_gene_weights_PLS-DA.npy")); WP = WP / WP.max()
    WC = np.abs(np.load(DATA / "latent_gene_weights_CVAE_neutral.npy")); WC = WC / WC.max()
    blocks = [("PLS-DA only", ps - cs, PLS), ("both", ps & cs, ACCENT), ("CVAE only", cs - ps, CVAE_N)]
    # within each block, genes grouped by the PLS-DA component that carries them most
    rows = []
    for name, gs_, c in blocks:
        gl = sorted(gs_, key=lambda g: (int(np.argmax(WP[:, gidx[g]])), -WP[:, gidx[g]].max()))
        rows += [(g, c, name) for g in gl]
    half = int(np.ceil(len(rows) / 2))
    parts = [rows[:half], rows[half:]]

    row_in = 0.092
    H = half * row_in + 2.75
    fig = plt.figure(figsize=(TW, H))
    top_frac = (half * row_in) / H
    gs = fig.add_gridspec(2, 1, height_ratios=[half * row_in, 1.55], hspace=1.05 / ((half * row_in + 1.55) / 2),
                          top=1 - 0.62 / H, bottom=0.40 / H)
    upper = gs[0].subgridspec(1, 2, wspace=0.42)
    cm_p, cm_c = SEQ_PLS, SEQ_CVAE_N
    for k, part in enumerate(parts):
        ax = fig.add_subplot(upper[k])
        n = len(part)
        cols = [gidx[g] for g, _, _ in part]
        Mp, Mc = WP[:, cols].T, WC[:, cols].T          # rows = genes, cols = dimensions
        ax.imshow(Mp, extent=(0.5, 8.5, n - 0.5, -0.5), cmap=cm_p, vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        ax.imshow(Mc, extent=(9.3, 17.3, n - 0.5, -0.5), cmap=cm_c, vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest")
        for i, (g, c, name) in enumerate(part):
            ax.add_patch(Rectangle((-0.35, i - 0.5), 0.6, 1.0, facecolor=c, edgecolor=SURFACE, lw=0.3))
            if i and part[i - 1][2] != name:
                ax.plot([-0.35, 17.3], [i - 0.5, i - 0.5], color=INK, lw=0.7)
        ax.set_xlim(-0.4, 17.4); ax.set_ylim(n - 0.5, -0.5)
        italic_ticks(ax, [g for g, _, _ in part], 5.2)
        for t_, (_, c, _) in zip(ax.get_yticklabels(), part):
            t_.set_color(c)
        ax.set_xticks(list(range(1, 9)) + [x + 0.8 for x in range(9, 17)])
        ax.set_xticklabels([f"{i}" for i in range(1, 9)] * 2, fontsize=5.4)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=1.5)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.text(4.5, -2.6, "PLS-DA LV", ha="center", va="bottom", fontsize=6.6, color=PLS)
        ax.text(13.3, -2.6, "CVAE z (agnostic)", ha="center", va="bottom", fontsize=6.6, color=CVAE_N)
        if k == 0:
            for x0f, cm_, lab, c in ((0.051, cm_p, "PLS-DA |weight|", PLS), (0.575, cm_c, "CVAE |weight|", CVAE_N)):
                cax = ax.inset_axes([x0f, -0.035, 0.40, 0.010])
                cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm_, norm=plt.Normalize(0, 1)), cax=cax,
                                  orientation="horizontal")
                cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
                cb.outline.set_visible(False)
                cax.set_xlabel(lab, fontsize=5.4, color=c, labelpad=1)
            ax.annotate("a", xy=(-0.30, 1.0), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points",
                        fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom",
                        annotation_clip=False)
            ax.annotate("How each latent dimension maps back onto the 111 selected genes", xy=(0, 1.0),
                        xycoords="axes fraction", xytext=(0, 30), textcoords="offset points", fontsize=8.6,
                        color=INK, va="bottom", annotation_clip=False)
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=c, label=nm) for nm, _, c in blocks],
               loc="upper right", bbox_to_anchor=(0.995, 1 - 0.18 / H), ncol=1, fontsize=5.8, frameon=False,
               title="selected by", title_fontsize=5.8)

    # ---- (b) signal-to-noise per dimension, all three arms ---------------------------
    lower = gs[1].subgridspec(1, 1)
    ax = fig.add_subplot(lower[0])
    pos_ = ax.get_position(); ax.set_position([pos_.x0 + 0.04, pos_.y0, pos_.width - 0.04, pos_.height])
    arms = [("PLS-DA", PLS, "PLS-DA (LV)"), ("CVAE_neutral", CVAE_N, "CVAE agnostic (z)"),
            ("CVAE_true", CVAE_T, "CVAE conditioned (z)")]
    w = 0.26
    for j, (tag, c, lab) in enumerate(arms):
        d = snr[snr.model == tag].reset_index(drop=True)
        x = np.arange(len(d)) + (j - 1) * w
        ax.bar(x, d.snr, w * 0.9, color=c, lw=0, label=lab)
        for xi, v in zip(x, d.snr):
            ax.text(xi, v + 0.06, f"{v:.1f}", ha="center", va="bottom", fontsize=4.6, color=c)
    ax.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(range(8)); ax.set_xticklabels([f"dimension {i}" for i in range(1, 9)], fontsize=6.2)
    ax.set_ylabel("signal-to-noise"); ax.set_ylim(0, snr.snr.max() * 1.22); ygrid(ax)
    ax.legend(loc="upper left", ncol=3, fontsize=6.2, frameon=False)
    panel(ax, "b", "Signal-to-noise of each latent dimension",
          "mean |weight| of the 111 selected genes ÷ that of all other genes; dashed = parity; "
          "heatmap weights scaled to each model's maximum", dx=-0.075)
    save(fig, "fig10_latent_gene_map")


fig_latent_map()

# =====================================================================
# Figure 11 -- functional enrichment
# =====================================================================
def fig_enrichment():
    """Enrichment as biology, every term shown. Terms are grouped into themes by the genes
    that carry them; terms resting on one gene form gene-specific themes, drawn hollow and
    marked as single-gene evidence rather than hidden."""
    SETS = [("plsda_down", "PLS-DA\nlower", PLS), ("plsda_up", "PLS-DA\nhigher", PLS),
            ("cvae_down", "CVAE\nlower", CVAE_N), ("cvae_up", "CVAE\nhigher", CVAE_N)]
    pls = load("PLSDA_gene_significance.csv"); cv = load("CVAE_gene_significance.csv")
    de = load("gene_de_stats.csv").set_index("gene")
    ps, cs = set(pls.gene[pls.significant]), set(cv.gene[cv.significant])
    prov = lambda g: ACCENT if (g in ps and g in cs) else PLS if g in ps else CVAE_N
    D, stats_ = {}, {}
    for key, _, _ in SETS:
        d = pd.read_csv(DATA / "enrichment" / f"{key}.csv")
        d = d.sort_values("adjusted_p_value").drop_duplicates(["source", "term_name"]).copy()
        d["genes"] = d.intersections.fillna("").apply(lambda s: frozenset(x for x in s.split(",") if x))
        stats_[key] = (len(d), int((d.intersection_size >= 2).sum()))
        D[key] = d

    entries = [(k, r) for k in D for r in D[k].itertuples()]
    multi_counts = {}
    for k, r in entries:
        if len(r.genes) >= 2:
            multi_counts[r.genes] = multi_counts.get(r.genes, 0) + 1
    cores = sorted(multi_counts, key=lambda s: (-multi_counts[s], len(s)))
    HB = frozenset({"HBB", "HBD"})
    MINOR = {"TRMT5", "ELMO2", "POLR1D", "BDNF"}           # one-off phenotype / enzyme terms

    def core_of(k, r):
        if len(r.genes) == 1:
            g = next(iter(r.genes))
            if g in HB:
                return HB
            return ("ONE", "minor" if g in MINOR else g)
        if r.source == "MIRNA":
            return "MIRNA"
        if r.source == "HP":
            return "HP"
        best = None
        for c in cores:
            if c <= r.genes and len(r.genes) - len(c) <= 1 and multi_counts[c] >= 2:
                if best is None or multi_counts[c] > multi_counts[best]:
                    best = c
        return best if best is not None else r.genes
    themes = {}
    for k, r in entries:
        themes.setdefault(core_of(k, r), []).append((k, r))

    NAMES = {HB: "Hemoglobin and oxygen transport*",
             frozenset({"NPTX2", "RPH3A", "BDNF", "TUBB2B", "CARTPT", "VGF"}): "Synaptic signalling",
             frozenset({"CARTPT", "VGF"}): "Neuropeptide and neuroendocrine",
             frozenset({"NPTX2", "VGF"}): "Anterior pituitary expression (HPA)",
             "MIRNA": "microRNA target sets (miR-7, -892b, -4436b)",
             "HP": "Human phenotype associations (HPO)",
             frozenset({"CARTPT", "TNR"}): "Transmission of nerve impulse",
             frozenset({"BDNF", "SLC2A3"}): "Neuronal glucose handling",
             frozenset({"BDNF", "NKX2-5"}): "Myotube differentiation",
             frozenset({"CHP1", "TOM1L1"}): "Protein autophosphorylation",
             ("ONE", "RAF1"): "RAF1–MAPK signalling complexes",
             ("ONE", "TAB2"): "TAK1–TAB2 inflammatory signalling",
             ("ONE", "KIF5A"): "Kinesin KIF5A transport complexes",
             ("ONE", "NEDD9"): "SMAD3–HEF1 (NEDD9) complex",
             ("ONE", "SRSF4"): "SR-protein splicing complex",
             ("ONE", "TUBB2B"): "Tubulin and cortical malformation",
             ("ONE", "DUSP6"): "MAP kinase phosphatase",
             ("ONE", "NPAS2"): "Circadian ARNTL–NPAS2 complex",
             ("ONE", "ASF1B"): "ASF1 histone-chaperone complexes",
             ("ONE", "CPSF6"): "CPSF6 mRNA 3'-end processing complexes",
             ("ONE", "YEATS4"): "NuA4/Tip60 chromatin remodelling",
             ("ONE", "MSL2"): "MSL histone-acetyltransferase complex",
             ("ONE", "minor"): "Other terms (TRMT5, ELMO2, POLR1D, BDNF)"}
    RANK_MULTI = ["Synaptic signalling", "Neuropeptide and neuroendocrine", "Anterior pituitary expression (HPA)",
                  "Transmission of nerve impulse", "Neuronal glucose handling", "Myotube differentiation",
                  "Protein autophosphorylation", "microRNA target sets (miR-7, -892b, -4436b)",
                  "Human phenotype associations (HPO)"]
    T = []
    for core, mem in themes.items():
        single = isinstance(core, tuple)
        name = NAMES.get(core, min(mem, key=lambda kr: kr[1].adjusted_p_value)[1].term_name[:38])
        genes = sorted(set().union(*[r.genes for _, r in mem]))
        per = {}
        for k, r in mem:
            n_, q_ = per.get(k, (0, 1.0)); per[k] = (n_ + 1, min(q_, r.adjusted_p_value))
        rank = (RANK_MULTI.index(name) if name in RANK_MULTI else 50 if single else 40)
        if core == HB:
            rank = 100
        T.append(dict(name=name, genes=genes, per=per, best=min(r.adjusted_p_value for _, r in mem),
                      n=len(mem), mem=mem, single=single, rank=rank,
                      home=min(per, key=lambda k: [s[0] for s in SETS].index(k))))
    # single-gene themes ordered by the gene set they come from, then by strength
    order_sets = [s[0] for s in SETS]
    T.sort(key=lambda t: (t["rank"] == 100, order_sets.index(t["home"]), t["rank"], t["best"]))
    nT = len(T)

    fig = plt.figure(figsize=(12.4, 7.8))
    Y0, Y1 = 0.115, 0.885

    # ---- (a) theme x gene set ----------------------------------------------------------------
    ax = fig.add_axes([0.155, Y0, 0.150, Y1 - Y0])
    nmax = max(t["n"] for t in T)
    qmax = -np.log10(min(t["best"] for t in T))
    first_single = next(i for i, t in enumerate(T) if t["single"])
    for i, t in enumerate(T):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f4f7", lw=0, zorder=0)
        for j, (k, lab, c) in enumerate(SETS):
            if k in t["per"]:
                n_, q_ = t["per"][k]
                s = 18 + 120 * np.sqrt(n_ / nmax)
                lv = 0.35 + 0.65 * (-np.log10(q_)) / qmax
                ax.scatter(j, i, s=s, color=tint(c, lv), edgecolor=c, linewidth=0.8, zorder=3)
                ax.text(j + 0.30, i, f"{n_}", ha="left", va="center", fontsize=5.0, color=INK2)
    ax.set_xlim(-0.5, 3.8); ax.set_ylim(nT - 0.5 + 0.9, -0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([s[1] for s in SETS], fontsize=5.8)
    for t_, s in zip(ax.get_xticklabels(), SETS):
        t_.set_color(s[2])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks(range(nT)); ax.set_yticklabels([t["name"] for t in T], fontsize=6.0)
    for t_ in ax.get_yticklabels():
        t_.set_color(INK)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for j, (k, _, c) in enumerate(SETS):
        ax.text(j, nT + 0.15, f"{stats_[k][0]}\nterms", ha="center", va="top", fontsize=4.9, color=INK3,
                linespacing=1.1)
    fig.text(0.008, 0.965, "a", fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom")
    fig.text(0.024, 0.965, "Every enriched term, grouped into biological themes", fontsize=8.8, color=INK,
             va="bottom")

    # ---- (b) genes behind each theme ------------------------------------------------------------
    axb = fig.add_axes([0.310, Y0, 0.225, Y1 - Y0], sharey=ax)
    order = []
    for t in T:
        for g in t["genes"]:
            if g not in order:
                order.append(g)
    for i, t in enumerate(T):
        if i % 2 == 0:
            axb.axhspan(i - 0.5, i + 0.5, color="#f4f4f7", lw=0, zorder=0)
        xs = [order.index(g) for g in t["genes"]]
        axb.plot([min(xs), max(xs)], [i, i], color=RULE, lw=0.7, zorder=1)
        for g in t["genes"]:
            both_ = g in ps and g in cs
            axb.scatter(order.index(g), i, s=16, marker="D" if both_ else "o", zorder=3,
                        facecolor=prov(g), edgecolor=SURFACE, linewidth=0.5)
    for x_, g in enumerate(order):
        up_ = de.loc[g, "logFC"] > 0
        axb.scatter(x_, nT - 0.5 + 0.45, marker="^" if up_ else "v", s=8, color=AD if up_ else CTRL, lw=0,
                    clip_on=False, zorder=4)
    axb.set_xlim(-0.8, len(order) - 0.2)
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels(order, rotation=90, fontsize=4.6, fontstyle="italic")
    for t_, g in zip(axb.get_xticklabels(), order):
        t_.set_color(prov(g))
    axb.xaxis.tick_top(); axb.tick_params(length=0, pad=1)
    plt.setp(axb.get_yticklabels(), visible=False)
    for s_ in axb.spines.values():
        s_.set_visible(False)
    fig.text(0.310, 0.965, "b", fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom")
    fig.text(0.326, 0.965, "The genes behind each theme", fontsize=8.8, color=INK, va="bottom")

    # ---- (c) leading terms of each theme, with their genes ------------------------------------------
    rows = []
    for t in T:
        mem = sorted(t["mem"], key=lambda kr: kr[1].adjusted_p_value)
        cap = 2 if (t["single"] or t["n"] <= 2) else 3
        seen = []
        for k, r in mem:
            if r.term_name in seen:
                continue
            seen.append(r.term_name); rows.append((t, k, r))
            if len(seen) == cap:
                break
    axc = fig.add_axes([0.735, Y0, 0.105, Y1 - Y0])
    xmax = max(-np.log10(r.adjusted_p_value) for _, _, r in rows) * 1.08
    prev = None
    for yi, (t, k, r) in enumerate(rows):
        c = PLS if k.startswith("plsda") else CVAE_N
        v = -np.log10(r.adjusted_p_value)
        if t is not prev:
            if prev is not None:
                axc.axhline(yi - 0.5, color=RULE, lw=0.5, xmin=-1.55, xmax=2.5, clip_on=False)
            prev = t
        axc.hlines(yi, 0, v, color=c, lw=1.0, alpha=0.6, zorder=1)
        mk = "o" if "down" in k else "^"
        axc.scatter(v, yi, s=9 + 6 * min(r.intersection_size, 12), marker=mk, zorder=3,
                    facecolor=c, edgecolor=SURFACE, linewidth=0.5)
        nm = r.term_name if len(r.term_name) <= 48 else r.term_name[:47] + "…"
        axc.text(-0.03, yi, nm, transform=axc.get_yaxis_transform(), ha="right", va="center", fontsize=5.2,
                 color=INK2)
        gl = sorted(r.genes)
        txt = ", ".join(gl if len(gl) <= 5 else gl[:5] + [f"+{len(gl) - 5}"])
        axc.text(1.04, yi, f"{txt}  · {r.source}", transform=axc.get_yaxis_transform(), ha="left", va="center",
                 fontsize=4.9, fontstyle="italic", color=c)
    axc.axvline(-np.log10(0.05), color=INK3, lw=0.6, ls=(0, (3, 2)))
    axc.set_xlim(0, xmax); axc.set_ylim(len(rows) - 0.5, -0.6)
    axc.set_yticks([]); axc.set_xlabel("$-\\log_{10}$ FDR $q$", fontsize=6.4); xgrid(axc)
    axc.spines["left"].set_visible(False); axc.tick_params(axis="x", labelsize=5.8)
    fig.text(0.555, 0.965, "c", fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom")
    fig.text(0.571, 0.965, "Leading terms of every theme, same order, with their genes", fontsize=8.8,
             color=INK, va="bottom")

    hl = [Line2D([], [], marker="o", ls="", color=PLS, markersize=4.2, label="PLS-DA set"),
          Line2D([], [], marker="o", ls="", color=CVAE_N, markersize=4.2, label="CVAE set"),
          Line2D([], [], marker="D", ls="", color=ACCENT, markersize=3.8, label="gene chosen by both"),
          Line2D([], [], marker="^", ls="", color=INK2, markersize=4.2, label="genes higher in AD"),
          Line2D([], [], marker="v", ls="", color=INK2, markersize=4.2, label="genes lower in AD")]
    fig.legend(handles=hl, loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=6, fontsize=6.2, frameon=False)
    fig.text(0.5, 0.004, ha="center", s= "Bubble area = number of terms in the theme (count beside it); fill depth = best FDR $q$. "
             "Themes of one-gene terms are named by us from the terms' content. *Hemoglobin terms likely reflect "
             "residual blood and are not interpreted.", fontsize=5.8, color=INK3, style="italic")
    save(fig, "fig11_enrichment")


fig_enrichment()