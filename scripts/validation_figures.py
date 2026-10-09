"""Validation figures V1-V6 of paper3, exported unchanged from V12ADfigures.
Reads data/validation/ only."""

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

# Sequential ramps for the confusion matrices, built from the two arm colours so
# the matrices sit inside the same muted palette as everything else instead of
# borrowing matplotlib's saturated "Blues"/"Oranges".
SEQ_PLS = LinearSegmentedColormap.from_list(
    "seq_pls", ["#ffffff", "#e7ecf2", "#c3d1e0", "#93aac6", "#5c7fa6", PLS, "#173252"])
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


def kde2d(xy, grid_x, grid_y, bw_scale=1.0):
    """Gaussian KDE evaluated on a grid, without a scipy import at call sites."""
    from scipy.stats import gaussian_kde
    k = gaussian_kde(xy.T)
    k.set_bandwidth(k.factor * bw_scale)
    XX, YY = np.meshgrid(grid_x, grid_y)
    return k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

# ======================================================================
# V12 figure set -- reads the saved results of the merge, V9, V10 and V11
# notebooks and draws the project's publication figures. Nothing is re-fitted.
# ======================================================================
import glob, json, re, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

INPUT_ROOT = os.environ.get("V12_INPUT", "/kaggle/input")


def src(slug, name):
    """Path of `name` written by the notebook `slug` (Kaggle mounts each input
    notebook's output under /kaggle/input/...)."""
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True))
    mine = [h for h in hits if f"/{slug}/" in h or f"/{slug}" in os.path.dirname(h)]
    hits = mine or hits
    if not hits:
        raise FileNotFoundError(f"{name} from {slug}: attach that notebook's output as an input")
    return hits[0]


rd = lambda slug, name, **kw: pd.read_csv(src(slug, name), **kw)
M, V9, V10, V11 = "ad-merged-geo-cohorts", "v9adexternal", "v10adgenevalidation", "v11adrobustness"

# ---- shared vocabulary ------------------------------------------------------
TRAIN = ["GSE48350", "GSE33000", "GSE118553", "GSE84422"]
HOLDOUT = "GSE132903"
VALID = ["GSE15222", "GSE29378", "GSE122063", "GSE36980"]
RNASEQ = "GSE125583"
SEVERITY = ["GSE106241", "GSE1297"]
DXC = [HOLDOUT] + VALID + [RNASEQ]                 # every AD-vs-control test cohort
REGION = {"GSE48350": "4 regions (HC, EC, SFG, PCG)", "GSE33000": "prefrontal cortex",
          "GSE118553": "temporal, entorhinal, frontal cx", "GSE84422": "15 cortical / hippocampal",
          "GSE132903": "middle temporal gyrus", "GSE15222": "temporal cortex",
          "GSE29378": "hippocampus (CA1, CA3)", "GSE122063": "frontal + temporal cortex",
          "GSE36980": "frontal cortex", "GSE125583": "fusiform gyrus",
          "GSE106241": "inferior temporal cortex", "GSE1297": "hippocampus"}
PLATFORM = {"GPL570": "Affymetrix U133 Plus 2", "GPL4372": "Agilent/Rosetta 44K",
            "GPL10558": "Illumina HT-12 v4", "GPL96": "Affymetrix U133A",
            "GPL2700": "Illumina RefSeq-8", "GPL6947": "Illumina HT-12 v3",
            "GPL16699": "Agilent 8x60K", "GPL6244": "Affymetrix HuGene 1.0 ST",
            "GPL24170": "Agilent 8x60K custom", "GPL16791": "RNA-seq (HiSeq 2500)"}
ROLE_COLOR = {"training": PLS, "external test": AD, "validation": ACCENT,
              "RNA-seq validation": CVAE_N, "severity": MID}
SET_COLOR = {"PLS-DA": PLS, "CVAE": CVAE_N, "Core": ACCENT}
SET_MARK = {"PLS-DA": "o", "CVAE": "s", "Core": "D"}
CELL_ORDER = ["neurons", "astrocytes", "microglia", "oligodendrocytes", "OPCs", "endothelial"]


def short(c):
    return c.replace("GSE", "GSE ") if False else c


def tint(hex_color, alpha):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def rnaseq_counts():
    """AD / control donors of GSE125583, read from V11's own log."""
    try:
        L = json.load(open(src(V11, "v11adrobustness.log")))
        t = "".join(x.get("data", "") for x in L)
        m = re.search(r"GSE125583: \d+ arrays -> (\d+) donors.*?'AD': (\d+), 'Control': (\d+)", t)
        if m:
            return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except Exception:
        pass
    # fallback: GEO's own sample records (same age >= 60 rule as V11)
    import gzip, requests
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125583/matrix/GSE125583_series_matrix.txt.gz"
    r = requests.get(url, headers={"User-Agent": "ad-merged-geo-cohorts (kaggle)"}, timeout=180)
    r.raise_for_status()
    txt = gzip.decompress(r.content).decode("utf-8", "ignore")
    rows = {}
    for line in txt.splitlines():
        if line.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip('"') for v in line.split("\t")[1:]]
            key = vals[0].split(":")[0].strip().lower()
            rows[key] = [v.split(":", 1)[1].strip() if ":" in v else "" for v in vals]
    dx = np.array(rows["diagnosis"]); age = pd.to_numeric(pd.Series(rows["age"]).str.extract(r"(\d+)")[0], errors="coerce").values
    keep = np.isnan(age) | (age >= 60)          # V11 keeps donors with no recorded age
    nad = int(((dx == "Alzheimer's disease") & keep).sum()); nct = int(((dx == "control") & keep).sum())
    return nad + nct, nad, nct


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def box(ax, x, y, w, h, text, face, edge, fs=6.6, weight="normal", ha="center", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, zorder=3,
            fontweight=weight, linespacing=1.25)


def arrow(ax, p0, p1, color=INK3, rad=0.0, lw=0.9):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=1, shrinkB=1,
                                mutation_scale=9, connectionstyle=f"arc3,rad={rad}"))


def repel_avoiding(ax, xs, ys, labels, ox, oy, fontsize=5.8, clearance=3.2):
    """Like figstyle.repel_labels, but every plotted point is also an obstacle,
    so a label never sits on a marker; leader lines join label and point."""
    fig = ax.figure; ppd = fig.dpi / 72.0
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    obs = ax.transData.transform(np.column_stack([ox, oy]))
    placed = [(x - 3.5 * ppd, y - 3.5 * ppd, x + 3.5 * ppd, y + 3.5 * ppd) for x, y in obs]
    leg = ax.get_legend()
    if leg is not None:
        bb = leg.get_window_extent(); placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
    axbb = ax.get_window_extent()
    cands = [(r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a)))
             for r in (8, 12, 17, 23, 30, 38, 48, 60, 74, 90) for a in range(0, 360, 20)]
    for i in np.argsort(-np.asarray(ys)):
        t = labels[i]; w = len(t) * fontsize * 0.56 * ppd + clearance * ppd; h = fontsize * 1.3 * ppd + clearance * ppd
        px, py = pts[i]; best = None
        for dx, dy in cands:
            cx = px + dx * ppd + (w / 2) * (1 if dx >= 0 else -1); cy = py + dy * ppd
            b = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if not (b[0] >= axbb.x0 and b[2] <= axbb.x1 and b[1] >= axbb.y0 and b[3] <= axbb.y1):
                continue
            if not any(b[0] < q[2] and q[0] < b[2] and b[1] < q[3] and q[1] < b[3] for q in placed):
                best = (b, dx, dy); break
        if best is None:
            continue
        b, dx, dy = best; placed.append(b)
        ax.annotate(t, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    color=INK, fontstyle="italic", ha="left" if dx >= 0 else "right", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=INK3, lw=0.4, shrinkA=0.5, shrinkB=2.5))


# =====================================================================
# Figure 1 -- study design and cohorts
# =====================================================================
def fig1_design():
    ct = rd(M, "cohort_table.csv"); ct = ct[ct.cohort != "TOTAL"]
    vc = rd(V10, "validation_cohorts.csv")
    sel = rd(V9, "loco_selected_genes.csv")
    n_pls = sel[(sel.method == "PLS-DA")].gene.nunique(); n_cvae = sel[(sel.method == "CVAE")].gene.nunique()
    gv = rd(V10, "gene_validation_table.csv"); n_core = int(gv.sets.str.contains("Core").sum())
    rs = rnaseq_counts()

    rows = []
    for _, r in ct.iterrows():
        role = "external test" if r.cohort == HOLDOUT else "training"
        rows.append((r.cohort, role, int(r.donors), int(r.AD_donors), int(r.control_donors), r.platform))
    for _, r in vc.iterrows():
        if r.cohort == HOLDOUT:
            continue
        role = "severity" if r.cohort in SEVERITY else "validation"
        rows.append((r.cohort, role, int(r.donors), int(r.AD), int(r.control), r.platform))
    rows.append((RNASEQ, "RNA-seq validation", rs[0], rs[1], rs[2], "GPL16791"))
    order = {"training": 0, "external test": 1, "validation": 2, "RNA-seq validation": 3, "severity": 4}
    rows.sort(key=lambda r: (order[r[1]], -np.nan_to_num(r[2])))

    fig = plt.figure(figsize=(TW, 10.2))
    # ---- (a) flow, drawn across the full width ----------------------------
    ax = fig.add_axes([0.0, 0.777, 1.0, 0.201]); ax.set_xlim(-0.02, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.0, "a", fontsize=9.5, fontweight="bold", family="DejaVu Sans", va="top")
    ax.text(0.03, 1.0, "Study design", fontsize=8.6, color=INK, va="top")
    ax.text(0.03, 0.90, "each step is its own Kaggle notebook; the test cohorts never touch training or gene selection",
            fontsize=6.6, color=INK3, va="top")
    ctd = ct.set_index("cohort")
    tr_txt = "Training cohorts\n" + "\n".join(f"{c}  n={int(ctd.loc[c, 'donors'])}" for c in TRAIN)
    yc = 0.40
    box(ax, 0.000, yc - 0.20, 0.160, 0.40, tr_txt, tint(PLS, 0.10), PLS, fs=6.0)
    box(ax, 0.200, yc - 0.14, 0.155, 0.28, "Preprocessing\nprobe → gene\nage ≥ 60, z per cohort",
        tint(MID, 0.10), RULE, fs=6.0)
    box(ax, 0.395, yc - 0.15, 0.170, 0.30, "PLS-DA  &  β-CVAE\n8-d latent space\n20 donor-grouped runs\n10 permuted-label runs",
        tint(MID, 0.10), RULE, fs=6.0)
    box(ax, 0.605, yc - 0.15, 0.130, 0.30, f"Frozen gene lists\nPLS-DA {n_pls} · CVAE {n_cvae}\nCore {n_core}",
        tint(ACCENT, 0.12), ACCENT, fs=6.0)
    for x0, x1 in ((0.160, 0.200), (0.355, 0.395), (0.565, 0.605)):
        arrow(ax, (x0 + 0.004, yc), (x1 - 0.004, yc), color=INK2, lw=1.0)
    outs = [(0.66, f"External test · {HOLDOUT}\nAUC and age checks", AD),
            (0.40, "Gene validation\n5 cohorts + 2 severity cohorts", ACCENT),
            (0.14, "Robustness\ncell mix + RNA-seq", CVAE_N)]
    for y0, t, c in outs:
        box(ax, 0.785, y0 - 0.10, 0.20, 0.20, t, tint(c, 0.12), c, fs=6.0)
    arrow(ax, (0.48, yc + 0.155), (0.785, 0.66), rad=-0.22, color=INK2, lw=1.0)
    ax.text(0.62, 0.80, "trained models", fontsize=5.8, color=INK3, style="italic", ha="center")
    arrow(ax, (0.739, yc), (0.781, 0.40), color=INK2, lw=1.0)
    arrow(ax, (0.739, yc - 0.05), (0.781, 0.16), rad=-0.15, color=INK2, lw=1.0)
    ax.text(0.0, -0.03, "Notebooks:  ad-merged-geo-cohorts  ›  V9ADexternal  ›  V10ADgenevalidation  ›  "
                        "V11ADrobustness  ›  V14ADfigures", fontsize=5.8, color=INK3)

    # ---- (b) cohorts ------------------------------------------------------
    ax = fig.add_axes([0.16, 0.307, 0.52, 0.395])
    y = np.arange(len(rows))[::-1]
    maxn = max(np.nan_to_num(r[2]) for r in rows)
    for yi, (c, role, n, nad, nct, plat) in zip(y, rows):
        if role == "severity":
            ax.barh(yi, n, 0.62, color=tint(MID, 0.45), edgecolor=SURFACE)
        else:
            ax.barh(yi, nad, 0.62, color=AD, edgecolor=SURFACE)
            ax.barh(yi, nct, 0.62, left=nad, color=CTRL, edgecolor=SURFACE)
        ax.text(n + maxn * 0.015, yi, f"{int(n)}", va="center", fontsize=6.0, color=INK2)
        ax.text(1.03, yi, f"{PLATFORM.get(plat, plat)}", va="center", fontsize=5.9, color=INK2,
                transform=ax.get_yaxis_transform())
        ax.text(1.40, yi, REGION.get(c, ""), va="center", fontsize=5.9, color=INK3,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    for lab, r in zip(ax.get_yticklabels(), rows):
        lab.set_color(ROLE_COLOR[r[1]])
    # role brackets
    for role in ROLE_COLOR:
        ys = [yi for yi, r in zip(y, rows) if r[1] == role]
        if ys:
            ax.annotate("", xy=(-0.205, min(ys) - 0.35), xytext=(-0.205, max(ys) + 0.35),
                        xycoords=("axes fraction", "data"),
                        arrowprops=dict(arrowstyle="-", color=ROLE_COLOR[role], lw=2.0), annotation_clip=False)
            pass
    ax.text(1.03, len(rows) - 0.25, "platform", fontsize=6.0, color=INK, transform=ax.get_yaxis_transform())
    ax.text(1.40, len(rows) - 0.25, "region", fontsize=6.0, color=INK, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, maxn * 1.10); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("donors aged ≥ 60"); xgrid(ax); despine(ax, left=False)
    ax.tick_params(axis="y", length=0)
    l1 = ax.legend(handles=swatches([("AD", AD), ("control", CTRL), ("severity cohort (Braak, MMSE)", tint(MID, 0.45))]),
                   loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=6.0, frameon=False)
    ax.add_artist(l1)
    rh = [Line2D([], [], color=v, lw=2.4, label=k) for k, v in ROLE_COLOR.items()]
    ax.legend(handles=rh, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=5, fontsize=6.0, frameon=False,
              handlelength=1.4, columnspacing=1.0)
    panel(ax, "b", "Cohorts", "names and left bars coloured by role", dx=-0.31)

    # ---- (c) donor age by diagnosis: the confound the age checks address ----
    ax = fig.add_axes([0.09, 0.035, 0.38, 0.14])
    meta = rd(M, "merged_metadata.csv").drop_duplicates("donor")
    order_ = TRAIN + [HOLDOUT]
    for i, c in enumerate(order_):
        d = meta[meta.cohort == c]
        for g, o in (("Control", -0.19), ("AD", 0.19)):
            boxstrip(ax, d.loc[d.dx == g, "age"].dropna().values, i + o, GROUP_COLOR[g], width=0.32,
                     jitter=0.04, seed=i, size=2.5)
        dm = d.loc[d.dx == "AD", "age"].median() - d.loc[d.dx == "Control", "age"].median()
        ax.text(i, 109, f"$\\Delta$ {dm:+.0f} y", ha="center", fontsize=6.0,
                color=AD if abs(dm) >= 5 else INK2, fontweight="bold" if abs(dm) >= 5 else "normal")
    ax.set_xticks(range(len(order_))); ax.set_xticklabels(order_, fontsize=6.0)
    for t, c in zip(ax.get_xticklabels(), order_):
        t.set_color(AD if c == HOLDOUT else PLS)
    ax.set_ylim(58, 114); ax.set_yticks([60, 70, 80, 90, 100]); ax.set_ylabel("age at death"); ygrid(ax)
    ax.legend(handles=swatches([("control", CTRL), ("AD", AD)]), loc="lower right", ncol=2, fontsize=5.8,
              frameon=True, framealpha=0.95, facecolor=SURFACE, edgecolor="none")
    panel(ax, "c", "Donor age by cohort", "$\\Delta$ = AD $-$ control median; external cohort age-matched", dx=-0.17)

    # ---- (d) the two gene-list versions in the paper ------------------------
    ax = fig.add_axes([0.60, 0.035, 0.37, 0.14])
    def _v7(name):
        here = Path(INPUT_ROOT).parent / name
        if here.exists():
            return pd.read_csv(here)
        hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True))
        hits = [h for h in hits if "v7admerged" in h] or hits
        return pd.read_csv(hits[0])
    P7, C7 = _v7("PLSDA_gene_significance.csv"), _v7("CVAE_gene_significance.csv")
    p7, c7 = set(P7.gene[P7.significant]), set(C7.gene[C7.significant])
    p9 = set(sel.gene[sel.method == "PLS-DA"]); c9 = set(sel.gene[sel.method == "CVAE"])
    core9 = set(gv.gene[gv.sets.str.contains("Core")])
    rows_ = [("PLS-DA", p7, p9, PLS), ("CVAE", c7, c9, CVAE_N), ("shared / core", p7 & c7, core9, ACCENT)]
    for i, (nm, a_, b_, c) in enumerate(rows_):
        parts = [(len(a_ - b_), tint(c, 0.30)), (len(a_ & b_), c), (len(b_ - a_), tint(c, 0.62))]
        left = 0
        for n_, fc in parts:
            ax.barh(i, n_, 0.62, left=left, color=fc, edgecolor=SURFACE, linewidth=0.8)
            if n_ >= 6:
                ax.text(left + n_ / 2, i, str(n_), ha="center", va="center", fontsize=6.0,
                        color=SURFACE if fc == c else INK)
            left += n_
        ax.text(left + 2, i, f"{len(a_ & b_)} of {len(a_)} kept", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows_], fontsize=6.4); ax.invert_yaxis()
    for t, r in zip(ax.get_yticklabels(), rows_):
        t.set_color(r[3])
    ax.tick_params(axis="y", length=0); ax.set_xlabel("genes"); xgrid(ax)
    ax.set_xlim(0, max(len(a_ | b_) for _, a_, b_, _ in rows_) * 1.32)
    ax.legend(handles=swatches([("5-cohort run only", tint(INK3, 0.30)), ("both runs", INK2),
                                ("held-out run only", tint(INK3, 0.62))]),
              loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=5.8, frameon=False)
    panel(ax, "d", "Two versions of the gene lists",
          "5-cohort run (gene-discovery figures) vs GSE132903-held-out run (tested here)", dx=-0.25)
    save(fig, "figV1_design_cohorts")


# =====================================================================
# Figure 2 -- external test on GSE132903 and the age checks
# =====================================================================
def fig2_external():
    runs = rd(V9, "loco_run_metrics.csv"); roc = rd(V9, "loco_roc_curves.csv")
    mp = rd(V9, "external_mean_predictions.csv"); bands = rd(V9, "external_age_bands.csv")
    A = dict(rd(V9, "external_age_validation.csv").values)
    arms = [("pls", "PLS-DA", PLS), ("cvae_neutral", "CVAE", CVAE_N), ("age", "age only", MID)]
    fig = plt.figure(figsize=(TW, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.15, 0.85, 1.0],
                          hspace=0.50, wspace=0.42)
    base = np.linspace(0, 1, 101)

    ax = fig.add_subplot(gs[0, 0])
    for m, lab, col in arms:
        v = roc[roc.model == m][[f"tpr_{i}" for i in range(101)]].values
        mu, sd = v.mean(0), v.std(0)
        ax.fill_between(base, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.13, lw=0)
        ax.plot(base, mu, color=col, lw=1.5 if m != "age" else 1.1, ls="-" if m != "age" else (0, (4, 2)),
                label=f"{lab}  {runs[f'auc_{m}'].mean():.3f}")
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right", title="mean AUC", title_fontsize=6.2, fontsize=6.2, alignment="left",
              frameon=True, framealpha=0.95, facecolor=SURFACE, edgecolor="none")
    panel(ax, "a", f"Unseen cohort {HOLDOUT}", f"{int(A['n_AD'])} AD / {int(A['n_control'])} control donors")

    ax = fig.add_subplot(gs[0, 1])
    for i, (m, lab, col) in enumerate(arms):
        boxstrip(ax, runs[f"auc_{m}"].values, i, col, width=0.52, seed=i, size=6)
    t_p = A.get("lr_pls_beyond_age_p", np.nan)
    ax.axhline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in arms], fontsize=6.6)
    ax.set_ylim(0.4, 0.95); ax.set_ylabel("ROC-AUC"); ygrid(ax)
    panel(ax, "b", "20 training runs", "one point per run")

    ax = fig.add_subplot(gs[0, 2])
    groups = [("age", [A["auc_age_only_raw"]], MID),
              ("PLS-DA", [A["auc_pls_mean_prob"], A["auc_pls_plus_age_insample"]], PLS),
              ("CVAE", [A["auc_cvae_neutral_mean_prob"], A["auc_cvae_neutral_plus_age_insample"]], CVAE_N)]
    for gi, (name, vals, c) in enumerate(groups):
        offs = [0.0] if len(vals) == 1 else [-0.19, 0.19]
        for k_, (v, o) in enumerate(zip(vals, offs)):
            ax.bar(gi + o, v, 0.36, color=c if k_ == 0 else tint(c, 0.3), edgecolor=c, linewidth=0.9)
            ax.text(gi + o, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.6, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=6.6)
    ax.set_ylim(0.4, 1.12); ax.set_ylabel("ROC-AUC"); ygrid(ax)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=PLS, edgecolor=PLS, label="model alone"),
                       Rectangle((0, 0), 1, 1, facecolor=tint(PLS, 0.3), edgecolor=PLS, label="model + age")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=5.8, frameon=False, handlelength=0.9)
    ax.text(0.03, 0.97, f"adds to age:\nPLS-DA {pmath(A['lr_pls_beyond_age_p'])}\nCVAE {pmath(A['lr_cvae_neutral_beyond_age_p'])}",
            transform=ax.transAxes, va="top", fontsize=5.8, color=INK2)
    panel(ax, "c", "Beyond age", "likelihood-ratio test")

    ax = fig.add_subplot(gs[1, 0:2])
    rng = np.random.RandomState(3)
    for g in ("Control", "AD"):
        s = mp[mp.group == g]
        ax.scatter(s.age + rng.uniform(-0.35, 0.35, len(s)), s.p_pls, s=12, color=GROUP_COLOR[g],
                   marker=GROUP_MARKER[g], alpha=0.78, edgecolor=SURFACE, linewidth=0.35, label=g)
    ax.set_xlabel("age at death (years)"); ax.set_ylabel("predicted P(AD), PLS-DA"); ax.set_ylim(-0.03, 1.32)
    ygrid(ax)
    ax.text(0.015, 0.975, f"risk vs age within group:  controls ρ = {A['rho_age_pls_control']:+.2f} "
                          f"({pmath(A['rho_age_pls_control_p'])}),  AD ρ = {A['rho_age_pls_AD']:+.2f} "
                          f"({pmath(A['rho_age_pls_AD_p'])})",
            transform=ax.transAxes, va="top", fontsize=6.0, color=INK2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=6.2, frameon=True, framealpha=0.95,
              facecolor=SURFACE, edgecolor="none", markerscale=1.2)
    panel(ax, "d", "Predicted risk does not follow age", f"AD and controls age-matched (median "
          f"{A['age_median_AD']:.0f} vs {A['age_median_control']:.0f} years)", dx=-0.075)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(bands))
    for j, (m, lab, col) in enumerate(arms):
        ax.errorbar(x + (j - 1) * 0.2, bands[f"auc_{m}_mean"], yerr=bands[f"auc_{m}_sd"],
                    fmt=["o", "s", "^"][j], color=col, ms=4.6, lw=0.9, capsize=1.8, label=lab)
    ax.axhline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels([f"{b.band}\n{b.n_AD}/{b.n_control}" for b in bands.itertuples()],
                                         fontsize=6.0)
    ax.set_xlim(-0.5, len(bands) - 0.5); ax.set_ylim(0.3, 1.08); ax.set_ylabel("ROC-AUC"); ygrid(ax)
    ax.legend(loc="lower left", fontsize=5.8, ncol=3, frameon=True, framealpha=0.95, facecolor=SURFACE,
              edgecolor="none", columnspacing=0.6, handletextpad=0.2)
    panel(ax, "e", "Inside age bands", "years; AD / control donors")
    save(fig, "figV2_external_validation")


# =====================================================================
# Figure 3 -- the gene signatures in six independent cohorts
# =====================================================================
def fig3_replication():
    sv = rd(V10, "set_validation.csv"); rna = rd(V11, "rnaseq_signature.csv")
    gv = rd(V10, "gene_validation_table.csv")
    rows = []
    for _, r in sv[sv.cohort != "META"].iterrows():
        rows.append(dict(set=r.set, cohort=r.cohort, auc=r.sig_auc, lo=r.null_auc_lo, hi=r.null_auc_hi,
                         p=r.p_auc, top=r.topDE_auc))
    for _, r in rna.iterrows():
        rows.append(dict(set=r.set, cohort=RNASEQ, auc=r.signature_auc, lo=r.null_lo, hi=r.null_hi,
                         p=r.p_vs_matched, top=r.topDE_auc))
    R = pd.DataFrame(rows)
    fig = plt.figure(figsize=(TW, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.50, wspace=0.16,
                          width_ratios=[1, 1, 1])
    cohorts = DXC
    for k, s in enumerate(["PLS-DA", "CVAE", "Core"]):
        ax = fig.add_subplot(gs[0, k])
        d = R[R.set == s].set_index("cohort").reindex(cohorts)
        y = np.arange(len(cohorts))[::-1]
        for yi, c in zip(y, cohorts):
            if c not in d.index or np.isnan(d.loc[c, "auc"]):
                ax.text(0.75, yi, "not tested" if c == HOLDOUT else "", ha="center", va="center",
                        fontsize=5.6, color=INK3, style="italic")
                continue
            r = d.loc[c]
            ax.add_patch(Rectangle((r.lo, yi - 0.28), r.hi - r.lo, 0.56, color=GRID, lw=0, zorder=1))
            ax.plot([r.top, r.top], [yi - 0.30, yi + 0.30], color=INK2, lw=1.3, zorder=2)
            ax.scatter(r.auc, yi, marker=SET_MARK[s], s=34, color=SET_COLOR[s], zorder=3, edgecolor=SURFACE,
                       linewidth=0.6)
            ax.text(max(r.auc, r.hi, r.top) + 0.022, yi, f"{r.auc:.2f}{star(r.p)}", ha="left", va="center",
                    fontsize=6.0, color=INK2)
        ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(0.45, 1.06); ax.set_ylim(-0.6, len(cohorts) - 0.4); ax.set_xticks([0.5, 0.7, 0.9])
        ax.set_yticks(y); ax.set_yticklabels(cohorts if k == 0 else [], fontsize=6.6)
        ax.tick_params(axis="y", length=0); xgrid(ax)
        ax.set_xlabel("signature ROC-AUC")
        n = int(gv.sets.str.contains(s).sum())
        panel(ax, "abc"[k], f"{s} set ({n} genes)",
              "grey = random 95%, bar = top-N DE" if k == 0 else None, dx=-0.37 if k == 0 else -0.12)
    fig.text(0.125, 0.515, "* better than matched random genes (* p<0.05, ** p<0.01, *** p<0.001); "
                           "signature = mean(up) − mean(down), no model fitted", fontsize=5.8, color=INK3,
             style="italic")

    # (d) meta-analysis volcano of every selected gene
    gsb = gs[1, :].subgridspec(1, 2, width_ratios=[2.1, 1.0], wspace=0.32)
    ax = fig.add_subplot(gsb[0])
    g = gv.copy()
    g["sign"] = np.where(g.train_direction == "up", 1, -1)
    g["eff"] = g.meta_pooled * g.sign
    g["nlq_true"] = -np.log10(g.meta_q.clip(lower=1e-300))
    YCAP = 12.0                                   # GFAP would otherwise compress every other gene
    g["nlq"] = g.nlq_true.clip(upper=YCAP)
    cls = np.where(g.sets.str.contains("Core"), "Core",
                   np.where(g.sets.str.contains("PLS-DA") & g.sets.str.contains("CVAE"), "both",
                            np.where(g.sets.str.contains("PLS-DA"), "PLS-DA", "CVAE")))
    g["cls"] = cls
    style = {"CVAE": (CVAE_N, "s", 14), "PLS-DA": (PLS, "o", 14), "both": (MID, "o", 16), "Core": (ACCENT, "D", 30)}
    for k_ in ("CVAE", "PLS-DA", "both", "Core"):
        s_ = g[g.cls == k_]; c, mk, sz = style[k_]
        ax.scatter(s_.eff, s_.nlq, s=sz, marker=mk, color=c, alpha=0.85 if k_ != "Core" else 1,
                   edgecolor=SURFACE, linewidth=0.4, zorder=3 if k_ == "Core" else 2,
                   label={"both": "both models", "Core": "Core genes"}.get(k_, f"{k_} only"))
    ax.axhline(-np.log10(0.05), color=INK3, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0, color=INK3, lw=0.6)
    ax.set_xlabel("pooled effect in the trained direction (SD, 5 cohorts)")
    ax.set_ylabel("$-\\log_{10}$ meta q (capped at 12)")
    ax.set_ylim(0, YCAP * 1.12); ygrid(ax)
    capped = g[g.nlq_true > YCAP]
    done = []
    for _, r in capped.sort_values("nlq_true", ascending=False).iterrows():
        if any(abs(r.eff - x) < 0.08 for x in done):
            continue
        done.append(r.eff)
        ax.annotate(f"\u2191{r.nlq_true:.0f}", (r.eff, YCAP), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=5.4, color=INK3)
    core = g[g.cls == "Core"]
    ax.set_xlim(g.eff.min() - 0.1, g.eff.max() + 0.25)
    fig.canvas.draw()
    repel_avoiding(ax, core.eff.values, core.nlq.values, list(core.gene), g.eff.values, g.nlq.values, fontsize=5.8)
    ax.text(0.01, -np.log10(0.05) + 0.25, "q = 0.05", fontsize=5.6, color=INK3,
            transform=ax.get_yaxis_transform())
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, framealpha=0.95, facecolor=SURFACE,
              edgecolor="none", ncol=2)
    ok = (g.eff > 0) & (g.meta_q < 0.05)
    panel(ax, "d", "Every selected gene, random-effects meta-analysis",
          f"{int(ok.sum())} of {len(g)} replicate (right of 0, above line); age/sex-adjusted, donor level", dx=-0.075)

    # (e) gene-level replication vs baselines (honest: not above baseline)
    ax = fig.add_subplot(gsb[1])
    mt = sv[sv.cohort == "META"].set_index("set")
    for j, s in enumerate(["PLS-DA", "CVAE", "Core"]):
        if s not in mt.index:
            continue
        r = mt.loc[s]
        ax.add_patch(Rectangle((j - 0.22, r.null_rep_lo), 0.44, r.null_rep_hi - r.null_rep_lo, color=GRID, lw=0))
        ax.plot([j - 0.25, j + 0.25], [r.topDE_replication] * 2, color=INK2, lw=1.3)
        ax.scatter(j, r.replication, marker=SET_MARK[s], s=40, color=SET_COLOR[s], zorder=3, edgecolor=SURFACE)
        ax.text(j, r.replication - 0.075, f"{r.replication:.0%}", ha="center", fontsize=5.8, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PLS-DA", "CVAE", "Core"], fontsize=6.6)
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.6, 2.6); ax.set_ylabel("genes replicated (meta)"); ygrid(ax)
    panel(ax, "e", "Gene by gene", "grey: matched random; bar: top-N DE", dx=-0.36)
    save(fig, "figV3_signature_replication")


# =====================================================================
# Figure 4 -- what cell composition explains, and what survives it
# =====================================================================
def fig4_composition():
    Q = rd(V11, "cell_composition_qc.csv"); B = rd(V11, "signature_beyond_composition.csv")
    G = rd(V11, "gene_effects_composition_adjusted.csv")
    piv = Q.dropna(subset=["AD_minus_control_d"]).pivot(index="cohort", columns="cell", values="AD_minus_control_d")
    piv = piv.reindex([c for c in DXC if c in piv.index])[[c for c in CELL_ORDER if c in piv.columns]]
    fig = plt.figure(figsize=(TW, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.25], width_ratios=[1.0, 1.0], hspace=0.62, wspace=0.55)

    ax = fig.add_subplot(gs[0, 0])
    lim = np.nanmax(np.abs(piv.values))
    ax.imshow(piv.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=5.8,
                    color=SURFACE if abs(v) > 0.85 * lim else INK)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=35, ha="right", fontsize=6.2)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=6.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel(ax, "a", "Cell mix, AD − control", "standardised difference; markers: McKenzie 2018", dx=-0.42)

    ax = fig.add_subplot(gs[0, 1])
    cohorts = [c for c in DXC if c in set(B.cohort)]
    y = np.arange(len(cohorts))[::-1]
    for yi, c in zip(y, cohorts):
        d = B[B.cohort == c].set_index("set")
        b0 = d.cv_auc_age_sex_cells.iloc[0]
        ax.scatter(b0, yi, s=26, facecolor=SURFACE, edgecolor=MID, linewidth=1.1, zorder=3)
        for k_, s in enumerate(("PLS-DA", "CVAE")):
            if s not in d.index:
                continue
            r = d.loc[s]; yy = yi + (0.17 if s == "PLS-DA" else -0.17)
            ax.plot([b0, r.cv_auc_plus_signature], [yy, yy], color=tint(SET_COLOR[s], 0.55), lw=1.4, zorder=2)
            ax.scatter(r.cv_auc_plus_signature, yy, marker=SET_MARK[s], s=26, color=SET_COLOR[s], zorder=4,
                       edgecolor=SURFACE, linewidth=0.5)
            if r.p_vs_matched_dauc < 0.05:
                ax.text(r.cv_auc_plus_signature + 0.008, yy, "*", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(cohorts, fontsize=6.2); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.66, 0.97); ax.set_ylim(-0.6, len(cohorts) - 0.4)
    ax.set_xlabel("cross-validated ROC-AUC"); xgrid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MID,
                              label="age + sex + 6 cell types"),
                       Line2D([], [], marker="o", ls="", color=PLS, label="+ PLS-DA signature"),
                       Line2D([], [], marker="s", ls="", color=CVAE_N, label="+ CVAE signature")],
              loc="upper left", bbox_to_anchor=(-0.05, -0.17), fontsize=5.8, ncol=3, frameon=False,
              columnspacing=1.0, handletextpad=0.5)
    panel(ax, "b", "Signature beyond cell mix", "* gain beats matched random genes", dx=-0.36)

    ax = fig.add_subplot(gs[1, :])
    C = G[G.set == "Core"].sort_values("pooled_raw", ascending=True)
    yy = np.arange(len(C))
    ax.hlines(yy, C.pooled_adj, C.pooled_raw, color=RULE, lw=1.0, zorder=1)
    ax.hlines(yy, C.lo_adj, C.hi_adj, color=tint(ACCENT, 0.6), lw=2.2, zorder=2)
    ax.scatter(C.pooled_raw, yy, s=28, facecolor=SURFACE, edgecolor=INK3, linewidth=1.0, zorder=3,
               label="age + sex adjusted")
    sig = C.q_adj < 0.05
    ax.scatter(C.pooled_adj[sig], yy[sig.values], s=34, color=ACCENT, edgecolor=SURFACE, linewidth=0.5, zorder=4,
               label="+ 6 cell types, q < 0.05")
    ax.scatter(C.pooled_adj[~sig], yy[~sig.values], s=30, facecolor=tint(ACCENT, 0.25), edgecolor=ACCENT,
               linewidth=0.8, zorder=4, label="+ 6 cell types, n.s.")
    for i, (q, fr) in enumerate(zip(C.q_adj, C.retained_frac)):
        ax.text(1.005, i, (f"{fr:.0%} kept" if fr <= 1 else f"grew {fr:.1f}$\\times$") + f"   q {q:.1g}",
                transform=ax.get_yaxis_transform(), va="center",
                fontsize=5.8, color=INK if q < 0.05 else INK3)
    ax.axvline(0, color=INK3, lw=0.6, ls=(0, (3, 2)))
    italic_ticks(ax, [f"{g} ({d})" for g, d in zip(C.gene, C.direction)], 6.4)
    ax.set_ylim(-0.7, len(C) - 0.3); ax.set_xlabel("pooled effect in the trained direction (SD)")
    xgrid(ax); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=0.95, facecolor=SURFACE, edgecolor="none")
    panel(ax, "c", "Core genes before and after cell-mix adjustment",
          "meta over the 5 cohorts not used to define them (incl. RNA-seq)", dx=-0.13)
    save(fig, "figV4_cell_composition")


# =====================================================================
# Figure 5 -- dose-response with neuropathology and cognition
# =====================================================================
def fig5_dose():
    P10 = rd(V10, "dose_response_points.csv"); P11 = rd(V11, "dose_response_points_v11.csv")
    D = rd(V11, "dose_response_composition_adjusted.csv")
    pts = pd.concat([P10[P10.set == "PLS-DA"][["cohort", "measure", "score", "severity"]],
                     P11[P11.cohort == RNASEQ][["cohort", "measure", "score", "severity"]]])
    show = [("GSE106241", "braak", "Braak stage"), (RNASEQ, "braak", "Braak stage"),
            ("GSE29378", "plaque", "plaque score (0–3)"), ("GSE1297", "mmse", "MMSE")]
    fig = plt.figure(figsize=(TW, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.52, wspace=0.55)
    for k, (c, m, lab) in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        p = pts[(pts.cohort == c) & (pts.measure == m)].dropna()
        lv = sorted(p.severity.unique())
        if len(lv) <= 7:
            for v in lv:
                boxstrip(ax, p[p.severity == v].score.values, v, PLS, width=0.55, seed=int(v * 10), size=4)
            ax.set_xticks(lv); ax.set_xticklabels([str(int(v)) for v in lv], fontsize=6.2)
        else:
            ax.scatter(p.severity, p.score, s=11, color=PLS, alpha=0.8, edgecolor=SURFACE, linewidth=0.3)
            z = np.polyfit(p.severity, p.score, 1); xx = np.linspace(p.severity.min(), p.severity.max(), 5)
            ax.plot(xx, np.polyval(z, xx), color=PLS, lw=1.0)
        d = D[(D.cohort == c) & (D.measure == m) & (D.set == "PLS-DA")].iloc[0]
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.42 * (hi - lo))
        ax.text(0.03, 0.97, f"ρ = {d.rho_age_adjusted:+.2f}\n+cells {d.rho_age_cells_adjusted:+.2f}"
                            f"{star(d.p_vs_matched_cells_adjusted)}",
                transform=ax.transAxes, va="top", fontsize=5.8, color=INK2)
        ax.set_xlabel(lab, fontsize=6.6); ax.set_ylabel("PLS-DA score" if k == 0 else ""); ygrid(ax)
        panel(ax, "abcd"[k], c, f"n = {int(d.n)}", dx=-0.42 if k == 0 else -0.30)

    ax = fig.add_subplot(gs[1, :])
    rows = D[["cohort", "measure"]].drop_duplicates().values.tolist()
    lab = {"braak": "Braak", "mmse": "MMSE", "nft": "tangles", "plaque": "plaques"}
    x = np.arange(len(rows))
    for j, s in enumerate(["PLS-DA", "CVAE", "Core"]):
        for i, (c, m) in enumerate(rows):
            d = D[(D.cohort == c) & (D.measure == m) & (D.set == s)]
            if not len(d):
                continue
            d = d.iloc[0]; e = d.expected_sign; xx = i + (j - 1) * 0.24
            ax.add_patch(Rectangle((xx - 0.07, d.null_lo), 0.14, d.null_hi - d.null_lo, color=GRID, lw=0))
            ax.plot([xx, xx], [d.rho_age_adjusted * e, d.rho_age_cells_adjusted * e], color=tint(SET_COLOR[s], 0.5),
                    lw=1.0)
            ax.scatter(xx, d.rho_age_adjusted * e, marker=SET_MARK[s], s=18, facecolor=SURFACE,
                       edgecolor=SET_COLOR[s], linewidth=0.9, zorder=3)
            ax.scatter(xx, d.rho_age_cells_adjusted * e, marker=SET_MARK[s], s=24, color=SET_COLOR[s], zorder=4,
                       edgecolor=SURFACE, linewidth=0.5, label=s if i == 0 else None)
            if d.p_vs_matched_cells_adjusted < 0.05:
                ax.text(xx + 0.075, d.rho_age_cells_adjusted * e, "*", ha="left", va="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK3, lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{lab.get(m, m)}" for c, m in rows], fontsize=6.0)
    ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(-0.35, 0.95)
    ax.set_ylabel("partial ρ (expected direction)"); ygrid(ax)
    ax.legend(loc="upper right", ncol=3, fontsize=6.0, frameon=True, framealpha=0.95, facecolor=SURFACE,
              edgecolor="none")
    panel(ax, "e", "Severity, all cohorts and sets",
          "hollow: age removed · filled: age + cell mix removed · grey: matched random · "
          "* beats matched random", dx=-0.075)
    save(fig, "figV5_dose_response")


# =====================================================================
# Figure 6 -- evidence scorecard
# =====================================================================
def fig6_scorecard():
    sv = rd(V10, "set_validation.csv"); rna = rd(V11, "rnaseq_signature.csv")
    B = rd(V11, "signature_beyond_composition.csv"); D = rd(V11, "dose_response_composition_adjusted.csv")
    A = dict(rd(V9, "external_age_validation.csv").values); runs = rd(V9, "loco_run_metrics.csv")
    cols = DXC + SEVERITY
    tests = []
    for s in ("PLS-DA", "CVAE"):
        key = "pls" if s == "PLS-DA" else "cvae_neutral"
        auc = runs[f"auc_{key}"].mean()
        tests.append((s, "trained-model AUC, unseen cohort",
                      {HOLDOUT: (auc, 0.0 if auc > runs.auc_age.mean() else 1, "auc")}))
        tests.append((s, "model adds to age (LR test)", {HOLDOUT: (np.nan, A[f"lr_{key}_beyond_age_p"], "p")}))
        rr = rna[rna.set == s]
        sub = sv[(sv.set == s) & (sv.cohort != "META")]
        cells = {r.cohort: (r.sig_auc, r.p_auc, "auc") for _, r in sub.iterrows()}
        if len(rr):
            cells[RNASEQ] = (rr.signature_auc.iloc[0], rr.p_vs_matched.iloc[0], "auc")
        tests.append((s, "signature AUC > matched random", cells))
        cells = {r.cohort: (r.sig_auc - r.topDE_auc, 0.01 if r.sig_auc > r.topDE_auc else 1, "diff")
                 for _, r in sub.iterrows()}
        if len(rr):
            dlt = rr.signature_auc.iloc[0] - rr.topDE_auc.iloc[0]
            cells[RNASEQ] = (dlt, 0.01 if dlt > 0 else 1, "diff")
        tests.append((s, "signature AUC \u2212 top-N DE AUC", cells))
        cells = {r.cohort: (r.delta_auc, r.p_vs_matched_dauc, "diff") for _, r in B[B.set == s].iterrows()}
        tests.append((s, "AUC gain beyond age, sex, cell mix", cells))
        cells = {r.cohort: (r.rho_age_cells_adjusted * r.expected_sign, r.p_vs_matched_cells_adjusted, "rho")
                 for _, r in D[(D.set == s) & (D.measure == "braak")].iterrows()}
        tests.append((s, "tracks Braak beyond cell mix", cells))
    fig, ax = plt.subplots(figsize=(TW, 0.30 * len(tests) + 1.5))
    for i, (s, name, cells) in enumerate(tests):
        for j, c in enumerate(cols):
            if c not in cells:
                ax.add_patch(Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, facecolor=SURFACE, edgecolor=GRID, lw=0.6))
                continue
            v, p, tag = cells[c]
            ok = p < 0.05
            face = SET_COLOR[s] if ok else tint(SET_COLOR[s], 0.12)
            ax.add_patch(Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, facecolor=face, edgecolor=SURFACE, lw=1.0))
            if tag == "p":
                txt = f"p={p:.0e}" if p < 1e-3 else f"p={p:.2f}"
            elif tag == "auc":
                txt = f"{v:.2f}"
            else:
                txt = f"{v:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6, color=SURFACE if ok else INK2)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(tests) - 0.5, -0.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=6.4)
    ax.set_yticks(range(len(tests))); ax.set_yticklabels([t[1] for t in tests], fontsize=6.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    half = len(tests) // 2
    for k, s in enumerate(("PLS-DA", "CVAE")):
        ax.annotate(s, xy=(-0.43, k * half + (half - 1) / 2), xycoords=("axes fraction", "data"),
                    ha="center", va="center", rotation=90, fontsize=8, color=SET_COLOR[s], fontweight="bold",
                    annotation_clip=False)
        ax.annotate("", xy=(-0.40, k * half - 0.4), xytext=(-0.40, k * half + half - 0.6),
                    xycoords=("axes fraction", "data"), arrowprops=dict(arrowstyle="-", color=SET_COLOR[s], lw=1.6),
                    annotation_clip=False)
    ax.axhline(half - 0.5, color=INK3, lw=0.8)
    ax.axvline(len(DXC) - 0.5, color=RULE, lw=0.8)
    ax.text(len(DXC) + 0.5, len(tests) + 0.9, "severity cohorts", ha="center", fontsize=6.0, color=INK3, style="italic")
    panel(ax, "", "Evidence at a glance",
          "filled = passes (p < 0.05 or above baseline) \u00b7 pale = does not \u00b7 blank = not tested",
          dx=-0.45, dy=1.04)
    save(fig, "figV6_evidence_scorecard")


for _f in (fig1_design, fig2_external, fig3_replication, fig4_composition, fig5_dose, fig6_scorecard):
    try:
        _f()
    except Exception as _e:
        import traceback; traceback.print_exc()
        print(f"!! {_f.__name__} failed: {_e}")
