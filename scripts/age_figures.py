"""Figure of paper3: the benchmark repeated with age taken out (V15ADage).

(a) classification under the three ways of handling age, (b, c) how far each model's
genome-wide gene ranking moves, (d) the shared genes ranked under every analysis,
(e) how much of each gene list survives, (f) whether the selected genes are age genes.
Reads data/ (V7ADmerged) and data/validation/v15adage only; reuses the style of
validation_figures.py (its header is executed, not its figures)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
exec((HERE / "validation_figures.py").read_text().split("\ndef fig1_design")[0])
from scipy.stats import spearmanr
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle

V15 = "v15adage"
ARMS = [("V7", "as merged"), ("A_age_corrected", "age-corrected"), ("B_age_matched", "age-matched")]
MODELS = [("PLSDA", "PLS-DA", PLS, "o"), ("CVAE", "CVAE", CVAE_N, "s")]
BG = "#f5f5f8"
TERM_LONG = {"V7, no age handling": "V7", "A, age-corrected": "A_age_corrected", "B, age-matched": "B_age_matched"}


def _load(arm, what):
    """V7 files live in data/, the age arms under data/validation/v15adage/<arm>/."""
    if arm == "V7":
        base = Path(os.environ.get("PAPER_DATA", HERE.parent / "data"))
        return pd.read_csv(base / what)
    return pd.read_csv(_arm_path(arm, what))


def _arm_path(arm, what):
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{arm}/{what}", recursive=True))
    hits = [h for h in hits if V15 in h] or hits
    if not hits:
        raise FileNotFoundError(f"{arm}/{what}")
    return hits[0]


def fig7_age():
    runs = {a: _load(a, "run_metrics.csv") for a, _ in ARMS}
    sig = {(a, m): _load(a, f"{m}_gene_significance.csv").set_index("gene") for a, _ in ARMS for m, *_ in MODELS}
    AG = rd(V15, "comparison_age_genes.csv")

    fig = plt.figure(figsize=(TW, 9.0))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.92, 0.62], hspace=0.38,
                             left=0.085, right=0.985, top=0.885, bottom=0.03)
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.42)
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.40)

    fig.text(0.085, 0.985, "Taking age out moves the edges of the gene lists, not the classification or the core",
             fontsize=10.2, color=INK, va="top", fontweight="bold")
    fig.text(0.085, 0.962, "the full benchmark repeated with each gene's ageing slope removed (age-corrected) and with "
             "donors matched on age within cohort (age-matched)", fontsize=6.8, color=INK3, va="top", style="italic")

    # ---- (a) classification under the three age handlings ----------------------------------
    ax = fig.add_subplot(r1[0])
    xs = {}
    for i, (arm, lab) in enumerate(ARMS):
        x0 = i * 1.0
        r = runs[arm]
        ax.axvspan(x0 - 0.45, x0 + 0.45, color=BG if i % 2 == 0 else SURFACE, lw=0, zorder=0)
        a_, b_ = r.auc_pls.values, r.auc_cvae_lat_neutral.values
        for k in range(len(r)):
            up = b_[k] > a_[k]
            ax.plot([x0 - 0.17, x0 + 0.17], [a_[k], b_[k]], color=CVAE_N if up else PLS, lw=0.55,
                    alpha=0.45 if up else 0.85, zorder=2)
        for dx, v, col, mk in ((-0.17, a_, PLS, "o"), (0.17, b_, CVAE_N, "s")):
            ax.scatter(np.full(len(v), x0 + dx), v, s=7, marker=mk, color=col, lw=0, zorder=3, alpha=0.9)
            ax.plot([x0 + dx - 0.11, x0 + dx + 0.11], [v.mean()] * 2, color=col, lw=2.2, zorder=4,
                    solid_capstyle="butt")
        ag = r.auc_age_only.mean()
        ax.plot([x0 - 0.38, x0 + 0.38], [ag, ag], color=MID, lw=1.6, ls=(0, (2, 1.4)), zorder=2)
        ax.text(x0, ag - 0.022, f"age {ag:.2f}", ha="center", va="top", fontsize=5.6, color=INK3)
        xs[arm] = int((b_ > a_).sum())
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in ARMS], fontsize=6.4)
    for i, (arm, _) in enumerate(ARMS):
        r = runs[arm]
        ax.text(i - 0.03, 0.948, f"{r.auc_pls.mean():.3f}", ha="right", va="bottom", fontsize=6.0,
                color=PLS, fontweight="bold")
        ax.text(i + 0.03, 0.948, f"{r.auc_cvae_lat_neutral.mean():.3f}", ha="left", va="bottom", fontsize=6.0,
                color=CVAE_N, fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.38, 0.975); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_ylabel("test ROC-AUC, 20 splits"); ax.grid(axis="y", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=PLS, markersize=4.2, label="PLS-DA"),
                       Line2D([], [], marker="s", ls="", color=CVAE_N, markersize=4.0, label="CVAE agnostic"),
                       Line2D([], [], color=MID, lw=1.6, ls=(0, (2, 1.4)), label="age alone")],
              loc="lower left", ncol=1, fontsize=5.8, frameon=False, bbox_to_anchor=(0.0, 0.10))
    panel(ax, "a", "Classification does not move",
          f"means on top; CVAE ahead in {xs['V7']}/{xs['A_age_corrected']}/{xs['B_age_matched']} of 20", dx=-0.17)
    ax.annotate("", xy=(0, 1), xycoords="axes fraction")
    for t_ in ax.texts[-2:]:
        pass

    # ---- (b, c) genome-wide ranking: as merged vs age-corrected ------------------------------
    GRE = LinearSegmentedColormap.from_list("gre", ["#ffffff", "#ececf1", "#d4d4dc", "#a9a9b5", "#7c7c88"])
    for j, (m, name, col, mk) in enumerate(MODELS):
        ax = fig.add_subplot(r1[1 + j]); ax.set_box_aspect(1)
        s0 = sig[("V7", m)].mean_score
        n = len(s0)
        r0 = np.log10(s0.rank(ascending=False) / n)
        rows = []
        for arm, lab in ARMS[1:]:
            s1 = sig[(arm, m)].mean_score.reindex(s0.index)
            top = s0.rank(ascending=False) <= 200
            rows.append((lab, spearmanr(s0, s1).correlation, spearmanr(s0[top], s1[top]).correlation))
        s1 = sig[("A_age_corrected", m)].mean_score.reindex(s0.index)
        r1_ = np.log10(s1.rank(ascending=False) / n)
        ax.hexbin(r0, r1_, gridsize=34, bins="log", cmap=GRE, mincnt=1, linewidths=0, rasterized=True,
                  extent=(-4.1, 0, -4.1, 0))
        sel0 = sig[("V7", m)].significant
        kept = sel0 & sig[("A_age_corrected", m)].significant.reindex(s0.index).fillna(False)
        ax.scatter(r0[sel0 & ~kept], r1_[sel0 & ~kept], s=11, marker=mk, facecolor=SURFACE, edgecolor=col, lw=0.8,
                   zorder=3)
        ax.scatter(r0[kept], r1_[kept], s=13, marker=mk, color=col, edgecolor=SURFACE, lw=0.4, zorder=4)
        ax.plot([-4.1, 0], [-4.1, 0], color=INK2, lw=0.6, ls=(0, (3, 2)))
        ax.axvline(np.log10(sel0.sum() / n), color=col, lw=0.5, alpha=0.5)
        tk = [-4, -3, -2, -1, 0]; tl = ["0.01%", "0.1%", "1%", "10%", "100%"]
        ax.set_xticks(tk); ax.set_xticklabels(tl, fontsize=5.4); ax.set_yticks(tk); ax.set_yticklabels(tl, fontsize=5.4)
        ax.set_xlim(-4.1, 0); ax.set_ylim(-4.1, 0)
        ax.set_xlabel("rank, as merged", fontsize=6.2); ax.set_ylabel("rank, age-corrected" if j == 0 else "",
                                                                       fontsize=6.2)
        (l1, ra1, rt1), (l2, ra2, rt2) = rows
        panel(ax, "bc"[j], f"{name}: genome-wide ρ {ra1:.2f} / {ra2:.2f}",
              f"corrected / matched; top 200: {rt1:.2f} / {rt2:.2f}", dx=-0.32)

    # ---- (d) the 18 shared genes, ranked under every analysis ---------------------------------
    ax = fig.add_subplot(outer[1])
    p7, c7 = sig[("V7", "PLSDA")], sig[("V7", "CVAE")]
    shared = sorted(set(p7.index[p7.significant]) & set(c7.index[c7.significant]))
    rows = [(arm, lab, m, name, col) for arm, lab in ARMS for m, name, col, _ in MODELS]
    robust = [g for g in shared if all(sig[(a, m)].loc[g, "significant"] for a, _ in ARMS for m, *_ in MODELS)]
    score = {g: sum(sig[(a, m)].loc[g, "significant"] for a, _ in ARMS for m, *_ in MODELS) for g in shared}
    shared.sort(key=lambda g: (-score[g], p7.loc[g, "mean_score"] * -1))
    for i, (arm, lab, m, name, col) in enumerate(rows):
        S = sig[(arm, m)]
        rk = S.mean_score.rank(ascending=False)
        cm_ = LinearSegmentedColormap.from_list(f"rk{i}", ["#ffffff", tint(col, 0.15), tint(col, 0.45), col,
                                                            tuple(v * 0.75 for v in to_rgb(col))])
        yy = i + (i // 2) * 0.35
        for j, g in enumerate(shared):
            r = rk[g]; sel = bool(S.loc[g, "significant"])
            f = np.clip((np.log10(2000) - np.log10(r)) / np.log10(2000), 0, 1)
            ax.add_patch(FancyBboxPatch((j - 0.42, yy - 0.40), 0.84, 0.80, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=cm_(f) if sel else SURFACE, edgecolor=col if sel else RULE,
                                        linewidth=0.0 if sel else 0.6, zorder=2))
            ax.text(j, yy, f"{int(r)}" if r < 1000 else f"{r / 1000:.1f}k", ha="center", va="center",
                    fontsize=4.9, color=SURFACE if (sel and f > 0.55) else (INK2 if sel else INK3), zorder=3)
        ax.text(-0.75, yy, f"{name} · {lab}", ha="right", va="center", fontsize=6.2, color=col)
    nrow_y = (len(rows) - 1) + ((len(rows) - 1) // 2) * 0.35
    for j, g in enumerate(shared):
        isr = g in robust
        ax.text(j, nrow_y + 0.75, g, ha="center", va="top", rotation=90, fontsize=6.4, fontstyle="italic",
                color=ACCENT if isr else INK2, fontweight="bold" if isr else "normal")
        if isr:
            ax.add_patch(Rectangle((j - 0.5, -0.62), 1.0, nrow_y + 1.24, facecolor="none", edgecolor=ACCENT,
                                   linewidth=1.1, zorder=4))
    ax.set_xlim(-0.6, len(shared) - 0.4); ax.set_ylim(nrow_y + 2.75, -0.8)
    ax.axis("off")
    ax.annotate("d", xy=(-0.20, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom", annotation_clip=False)
    ax.annotate(f"The {len(shared)} genes both models chose, ranked under every analysis: "
                f"{len(robust)} stay selected by both models throughout",
                xy=(-0.17, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=8.4, color=INK, va="bottom", annotation_clip=False)
    ax.text(-0.6, nrow_y + 2.72, "number = genome-wide rank of 11,214 genes · filled = selected · teal frame = "
            "selected by both models in all three analyses", ha="left", va="bottom", fontsize=5.6, color=INK3,
            style="italic")

    # ---- (e) how much of each list survives ------------------------------------------------------
    ax = fig.add_subplot(r3[0])
    y = 0
    for m, name, col, mk in MODELS:
        s0 = set(sig[("V7", m)].index[sig[("V7", m)].significant])
        sa = set(sig[("A_age_corrected", m)].index[sig[("A_age_corrected", m)].significant])
        sb = set(sig[("B_age_matched", m)].index[sig[("B_age_matched", m)].significant])
        parts = [(len(s0 & sa & sb), col, "kept in both"), (len(s0 & (sa ^ sb)), tint(col, 0.50), "kept in one"),
                 (len(s0 - sa - sb), "#e3e3ea", "dropped in both")]
        left = 0
        for n_, c_, lab in parts:
            ax.barh(y, n_, 0.55, left=left, color=c_, lw=0)
            if n_ >= 4:
                ax.text(left + n_ / 2, y, str(n_), ha="center", va="center", fontsize=6.6,
                        color=SURFACE if c_ == col else INK, fontweight="bold" if c_ == col else "normal")
            left += n_
        new_a, new_b = len(sa - s0), len(sb - s0)
        ax.text(left + 1.5, y, f"of {len(s0)}   ·   new: {new_a} corrected, {new_b} matched", va="center",
                fontsize=5.8, color=INK2)
        ax.text(-1.5, y, name, ha="right", va="center", fontsize=7.0, color=col, fontweight="bold")
        y += 1
    ax.set_xlim(0, 125); ax.set_ylim(1.55, -0.6); ax.axis("off")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=INK2, label="kept under both age handlings"),
                       Rectangle((0, 0), 1, 1, color=tint(INK2, 0.5), label="kept under one"),
                       Rectangle((0, 0), 1, 1, color="#e3e3ea", label="dropped under both")],
              loc="upper center", bbox_to_anchor=(0.42, 0.06), ncol=3, fontsize=5.8, frameon=False)
    ax.annotate("e", xy=(-0.17, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom", annotation_clip=False)
    ax.annotate("How much of each gene list survives", xy=(-0.10, 1.0), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points", fontsize=8.4, color=INK, va="bottom", annotation_clip=False)

    # ---- (f) are the selected genes age genes? -----------------------------------------------------
    ax = fig.add_subplot(r3[1])
    bg = 100 * AG.frac_age_genes[AG.list == "background"].iloc[0]
    for i, (arm, lab) in enumerate(ARMS):
        key = [k for k, v in TERM_LONG.items() if v == arm][0]
        for dx, (m, name, col, mk) in zip((-0.17, 0.17), MODELS):
            v = 100 * AG.frac_age_genes[(AG.arm == key) & (AG.list == name)].iloc[0]
            ax.plot([i + dx, i + dx], [bg, v], color=col, lw=2.4, solid_capstyle="butt", zorder=2)
            ax.scatter(i + dx, v, s=30, marker=mk, color=col, edgecolor=SURFACE, lw=0.6, zorder=3)
            ax.text(i + dx, v + (2.2 if v > bg else -2.2), f"{v:.0f}", ha="center", va="bottom" if v > bg else "top",
                    fontsize=6.0, color=col)
    ax.axhline(bg, color=INK, lw=0.8)
    ax.text(2.48, bg - 0.8, f"all genes {bg:.0f}%", ha="right", va="top", fontsize=5.8, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in ARMS], fontsize=6.4); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(30, 72); ax.set_ylabel("% age-associated in controls", fontsize=6.6)
    ax.grid(axis="y", visible=True); ax.spines["bottom"].set_visible(False)
    panel(ax, "f", "Selected genes are not age genes", "% of each list age-associated in controls", dx=-0.22)
    save(fig, "figV7_age_robustness")


fig7_age()
