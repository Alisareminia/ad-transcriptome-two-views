"""Figure of paper3: why the two models select different sides of AD (V16ADgenesides).

(a) every gene's AD effect in bulk tissue against its effect after accounting for cell
composition, with the selected genes of each model; (b) how each list's AD effects
divide into direct, compositional, masked and undetected; (c) the size of the AD
shift; (d) the share of that shift that remains beyond cell composition; (e)
high-specificity cell-type marker genes; (f) what each importance score follows
genome-wide; (g) co-expression within each list against effect-matched genes.
Reads data/validation/v16adgenesides only and reuses the style of
validation_figures.py (its header is executed, not its figures)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
exec((HERE / "validation_figures.py").read_text().split("\ndef fig1_design")[0])
from matplotlib.patches import Rectangle

V16 = "v16adgenesides"
LISTS = [("PLS-DA only", "PLS-DA only", PLS, "o"),
         ("shared", "both models", ACCENT, "D"),
         ("CVAE only", "CVAE only", CVAE_N, "s")]
CLASS = [("direct", "direct", "#2e4b63"),
         ("compositional", "compositional", "#9aa7b8"),
         ("masked", "masked", "#b0868f"),
         ("neither", "not detected", "#e4e4ea")]
PANEL_BG = "#f6f6f9"


def fig_gene_sides():
    G = rd(V16, "gene_sides_per_gene.csv").set_index("gene")
    EC = rd(V16, "gene_sides_effect_classes.csv")
    SM = rd(V16, "gene_sides_summary.csv")
    CO = rd(V16, "gene_sides_coherence.csv")
    CN = rd(V16, "gene_sides_coherence_null.csv")
    MK = rd(V16, "gene_sides_celltypes.csv")
    AT = rd(V16, "gene_sides_attribution.csv")
    FD = rd(V16, "gene_sides_fisher.csv").iloc[0]

    fig = plt.figure(figsize=(TW, 9.15))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.32, 0.80, 0.86], hspace=0.52,
                             left=0.075, right=0.985, top=0.905, bottom=0.045)
    r1 = outer[0].subgridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.30)
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.48)
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.42)

    fig.text(0.075, 0.992, "Two attribution rules, two sides of the same disease",
             fontsize=10.2, color=INK, va="top", fontweight="bold")
    fig.text(0.075, 0.970, "donor-level properties of the genes each model selected, against unselected genes "
             "matched on the size of their AD shift", fontsize=6.8, color=INK3, va="top", style="italic")

    # ---- (a) AD effect in bulk and beyond cell composition ----------------------------
    ax = fig.add_subplot(r1[0])
    thr_raw = G.loc[G.q_raw < 0.05, "t_raw"].abs().min()
    thr_adj = G.loc[G.q_adj < 0.05, "t_adj"].abs().min()
    un = G[G.group == "unselected"]
    ax.hexbin(un.t_raw, un.t_adj, gridsize=58, cmap=LinearSegmentedColormap.from_list(
        "g", ["#ffffff", "#e3e3ea", "#c4c4cf", "#9d9dab"]), mincnt=1, bins="log", linewidths=0, zorder=1)
    lim_x, lim_y = (-17, 17), (-11, 11)
    ax.axvspan(-thr_raw, thr_raw, color=PANEL_BG, zorder=0, lw=0)
    ax.axhspan(-thr_adj, thr_adj, color=PANEL_BG, zorder=0, lw=0)
    for v in (-thr_raw, thr_raw):
        ax.axvline(v, color=RULE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    for v in (-thr_adj, thr_adj):
        ax.axhline(v, color=RULE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot([-17, 17], [-17, 17], color=INK3, lw=0.6, ls=(0, (1, 2)), zorder=1)
    for key, lab, col, mk in LISTS:
        s = G[G.group == key]
        ax.scatter(s.t_raw, s.t_adj, s=19 if mk != "D" else 17, marker=mk, color=col,
                   edgecolor=SURFACE, linewidth=0.45, zorder=4, label=f"{lab} ({len(s)})")
    lab_genes = ["VGF", "DUSP6", "SPRY4", "RAF1", "ELK1", "KDM2A", "BDNF", "NPTX2", "GFAP", "RPH3A",
                 "CLDN5", "IFITM1", "FCGR2C", "CD52", "LAMA2", "VEGFA", "PPM1A", "ASF1B", "GPR17", "ACSBG1"]
    lg = G.loc[lab_genes]
    ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)
    REG = (("direct", thr_raw + 0.5, 10.45, "left"), ("direct", -thr_raw - 0.5, -10.45, "right"),
           ("compositional", 16.6, -0.0, "right"), ("masked", 0.0, -10.45, "center"))
    for txt, x, y, ha in REG:
        ax.text(x, y, txt, fontsize=6.4, color=INK3, ha=ha, va="center", style="italic", zorder=2)
    ax.set_xlabel("AD effect in bulk tissue ($t$, donor level)")
    ax.set_ylabel("AD effect after accounting for cell mix ($t$)")
    ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.3, borderaxespad=0.4, markerscale=1.1)
    sel_ = G[G.group != "unselected"]
    # every selected marker and every region label is an obstacle for the gene labels
    ox = np.r_[sel_.t_raw.values, [thr_raw + 1.5 + k for k in range(0, 5)], [-thr_raw - 1.5 - k for k in range(0, 5)],
               [12.5, 13.5, 14.5, 15.5, 16.3], [-0.8, 0.0, 0.8]]
    oy = np.r_[sel_.t_adj.values, [10.45] * 5, [-10.45] * 5, [0.0] * 5, [-10.45] * 3]
    repel_avoiding(ax, lg.t_raw.values, lg.t_adj.values, list(lg.index), ox, oy, fontsize=5.6, clearance=1.2)
    panel(ax, "a", "AD effect in bulk tissue and beyond the cell mix",
          "shaded: not significant ($q\\geq0.05$); dotted: no change on adjustment", dx=-0.115)

    # ---- (b) effect classes per list ------------------------------------------------
    ax = fig.add_subplot(r1[1])
    rows = [("PLS-DA only", "PLS-DA only", PLS), ("shared", "both models", ACCENT),
            ("CVAE only", "CVAE only", CVAE_N), ("all genes", "all genes", INK3)]
    for i, (key, lab, col) in enumerate(rows):
        e = EC[EC.group == key].set_index("effect_class")
        left = 0.0
        for cls, clab, ccol in CLASS:
            w = 100 * e.at[cls, "fraction"]
            ax.barh(i, w, left=left, height=0.62, color=ccol, edgecolor=SURFACE, linewidth=0.8, zorder=2)
            if w >= 7:
                txt = f"{int(e.at[cls, 'n']):,}" if key == "all genes" else f"{int(e.at[cls, 'n'])}"
                ax.text(left + w / 2, i, txt, ha="center", va="center", fontsize=6.3,
                        color=SURFACE if cls in ("direct", "masked") else INK, zorder=3)
            left += w
        if key != "all genes":
            exp = 100 * e.at["direct", "null_matched_fraction"]
            ax.plot([exp, exp], [i - 0.40, i + 0.40], color=INK, lw=1.3, zorder=4, solid_capstyle="butt")
            p = e.at["direct", "p_vs_matched"]
            ax.text(101.5, i, f"{100 * e.at['direct', 'fraction']:.0f}% direct\n"
                    f"exp. {exp:.0f}%{' ' + star(p) if star(p) else ''}",
                    fontsize=5.9, color=col, va="center", ha="left", linespacing=1.1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], fontsize=7)
    for t, (_, _, col) in zip(ax.get_yticklabels(), rows):
        t.set_color(col)
    ax.set_ylim(len(rows) - 0.45, -0.6); ax.set_xlim(0, 100)
    ax.set_xlabel("% of genes in the list")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hl = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in CLASS]
    ax.legend(hl, [l for _, l, _ in CLASS], ncol=4, loc="upper left", bbox_to_anchor=(-0.02, -0.17),
              fontsize=5.9, handlelength=0.9, columnspacing=0.8, handletextpad=0.35)
    panel(ax, "b", "How each list's AD effects relate to the cell mix",
          "black tick: direct share expected for genes of the same AD shift", dx=-0.30)

    # ---- (c) size of the AD shift, (d) share beyond cell mix ------------------------
    def strip(ax, metric, scale, ylab, clip=None, nullband=True):
        for i, (key, lab, col, mk) in enumerate(LISTS):
            v = G.loc[G.group == key, metric].values
            if clip is not None:
                v = np.clip(v, *clip)
            v = v * scale
            rs = np.random.RandomState(3 + i)
            ax.scatter(i + rs.uniform(-0.17, 0.17, len(v)), v, s=9, marker=mk, color=col, alpha=0.55,
                       linewidth=0, zorder=3)
            med = np.median(v)
            ax.plot([i - 0.28, i + 0.28], [med, med], color=col, lw=2.0, zorder=4, solid_capstyle="butt")
            r = SM[(SM.group == key) & (SM.metric == metric)].iloc[0]
            if nullband:
                ax.add_patch(Rectangle((i - 0.33, r.null_lo * scale), 0.66, (r.null_hi - r.null_lo) * scale,
                                       facecolor=tint(INK3, 0.16), edgecolor="none", zorder=1))
                ax.plot([i - 0.33, i + 0.33], [r.null_matched_median * scale] * 2, color=INK3, lw=0.8,
                        ls=(0, (2, 1.5)), zorder=2)
            ax.text(i, 0.985, f"{med:.2f}" if scale == 1 else f"{med:.0f}%", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.4, color=col, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"))
        ax.set_xticks(range(3)); ax.set_xticklabels([l[1].replace(" ", "\n", 1) for l in LISTS], fontsize=6.4)
        for t, l in zip(ax.get_xticklabels(), LISTS):
            t.set_color(l[2])
        ax.set_ylabel(ylab); ygrid(ax); ax.set_xlim(-0.55, 2.55)

    ax = fig.add_subplot(r2[0])
    ax.set_ylim(0, 1.72)
    strip(ax, "abs_d", 1, "|Cohen's $d$|, AD vs control", nullband=False)
    gen_med = G.abs_d.median()
    ax.axhline(gen_med, color=INK3, lw=0.7, ls=(0, (3, 2)), zorder=1)
    p = SM[(SM.group == "PLS-DA only vs CVAE only") & (SM.metric == "abs_d")].p_vs_matched.iloc[0]
    panel(ax, "c", "Size of the AD shift", f"dashed: all genes ({gen_med:.2f}); {pmath(p)}", dx=-0.32, dy=1.07)

    ax = fig.add_subplot(r2[1])
    ax.set_ylim(-50, 162)
    strip(ax, "effect_retained", 100, "% of the AD effect kept\nafter accounting for cell mix", clip=(-0.45, 1.45))
    ax.axhline(100, color=RULE, lw=0.6, zorder=1); ax.axhline(0, color=RULE, lw=0.6, zorder=1)
    p = SM[(SM.group == "PLS-DA only vs CVAE only") & (SM.metric == "effect_retained")].p_vs_matched.iloc[0]
    panel(ax, "d", "Effect beyond the cell mix",
          f"grey: effect-matched genes; {pmath(p)}", dx=-0.32, dy=1.07)

    # ---- (e) cell-type marker genes ---------------------------------------------------
    ax = fig.add_subplot(r2[2])
    CT = [("astrocytes", "astrocyte", "#6d8aa6"), ("endothelial", "endothelial", "#a0566b"),
          ("microglia", "microglia", "#7a6d9e"), ("OPCs", "OPC", "#5f8f8a"),
          ("neurons", "neuron", "#3d4f66"), ("oligodendrocytes", "oligodendrocyte", "#b59a6a")]
    for i, (key, lab, col, mk) in enumerate(LISTS):
        r = MK[MK.group == key].iloc[0]
        n = r.n_genes; left = 0.0
        for ct, clab, ccol in CT:
            w = 100 * r[ct] / n
            if w > 0:
                ax.bar(i, w, bottom=left, width=0.55, color=ccol, edgecolor=SURFACE, linewidth=0.6, zorder=3)
                left += w
        exp = 100 * r.null_matched_fraction
        ax.plot([i - 0.36, i + 0.36], [exp, exp], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
        ax.text(i, left + 0.8, f"{int(r.n_markers)}/{int(n)}{' ' + star(r.p_vs_matched) if star(r.p_vs_matched) else ''}",
                ha="center", va="bottom", fontsize=6.2, color=col, fontweight="bold")
    ax.set_ylim(0, 33); ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(3)); ax.set_xticklabels([l[1].replace(" ", "\n", 1) for l in LISTS], fontsize=6.4)
    for t, l in zip(ax.get_xticklabels(), LISTS):
        t.set_color(l[2])
    ax.set_ylabel("% of the list that are\ncell-type marker genes"); ygrid(ax)
    used = [c for c in CT if any(MK[MK.group.isin(["PLS-DA only", "shared", "CVAE only"])][c[0]] > 0)]
    ax.legend([Rectangle((0, 0), 1, 1, color=c) for _, _, c in used], [l for _, l, _ in used],
              loc="upper right", fontsize=5.6, ncol=2, handlelength=0.8, columnspacing=0.6,
              handletextpad=0.3, borderaxespad=0.1)
    panel(ax, "e", "Cell-type marker genes", "top-200 published markers; tick: expected", dx=-0.30, dy=1.07)

    # ---- (f) what each importance score follows genome-wide --------------------------
    ax = fig.add_subplot(r3[0])
    PROPS = [("abs_d", "size of the AD shift, |$d$|", "spearman"),
             ("abs_t_adj", "AD shift beyond the cell mix", "partial_spearman_given_abs_d"),
             ("connectivity", "co-expression with the genome", "partial_spearman_given_abs_d"),
             ("R2_cells", "variance explained by the cell mix", "partial_spearman_given_abs_d"),
             ("lnVR", "variance ratio, AD / control", "partial_spearman_given_abs_d")]
    for i, (prop, lab, col_) in enumerate(PROPS):
        v = AT[(AT.score == "vip") & (AT.property == prop)][col_].iloc[0]
        w = AT[(AT.score == "ig") & (AT.property == prop)][col_].iloc[0]
        ax.plot([v, w], [i, i], color=RULE, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([v], [i], s=34, marker="o", color=PLS, zorder=3, edgecolor=SURFACE, linewidth=0.6)
        ax.scatter([w], [i], s=32, marker="s", color=CVAE_N, zorder=3, edgecolor=SURFACE, linewidth=0.6)
        for val, col, side in ((v, PLS, 1 if v >= w else -1), (w, CVAE_N, 1 if w > v else -1)):
            ax.text(val + side * 0.035, i, f"{val:+.2f}", fontsize=6.1, color=col, va="center",
                    ha="left" if side > 0 else "right", fontweight="bold")
    ax.axvline(0, color=INK3, lw=0.7, zorder=0)
    ax.set_yticks(range(len(PROPS))); ax.set_yticklabels([p[1] for p in PROPS], fontsize=6.7)
    ax.set_ylim(len(PROPS) - 0.5, -0.65); ax.set_xlim(-0.42, 0.62)
    ax.set_xlabel("Spearman $\\rho$ with the importance score, all 11,214 genes")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); xgrid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=PLS, markersize=5, label="PLS-DA VIP"),
                       Line2D([], [], marker="s", ls="", color=CVAE_N, markersize=5, label="CVAE $|$IG$|$")],
              loc="lower right", fontsize=6.2, ncol=2, columnspacing=0.8)
    panel(ax, "f", "Genome-wide: what each importance score follows",
          "rows 2--5 partial $\\rho$ holding |$d$| fixed", dx=-0.47, dy=1.03)

    # ---- (g) co-expression within each list ------------------------------------------
    ax = fig.add_subplot(r3[1])
    for i, (key, lab, col, mk) in enumerate(LISTS):
        nv = CN[CN.group == key].coherence.values
        vp = ax.violinplot([nv], positions=[i], widths=0.62, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(tint(INK3, 0.22)); b.set_edgecolor("none"); b.set_alpha(1)
        obs = CO[CO.group == key].coherence.iloc[0]
        ax.scatter([i], [obs], s=46, marker=mk, color=col, zorder=4, edgecolor=SURFACE, linewidth=0.7)
        ax.text(i + 0.12, obs, f"{obs:.2f}", fontsize=6.3, color=col, va="center", ha="left", fontweight="bold")
    p = rd(V16, "gene_sides_coherence_diff.csv").permutation_p.iloc[0]
    ax.set_xticks(range(3)); ax.set_xticklabels([l[1].replace(" ", "\n", 1) for l in LISTS], fontsize=6.4)
    for t, l in zip(ax.get_xticklabels(), LISTS):
        t.set_color(l[2])
    ax.set_ylabel("mean |$r$| between genes,\ndiagnosis effect removed"); ygrid(ax)
    ax.set_xlim(-0.55, 2.6)
    panel(ax, "g", "Co-expression within each list",
          f"grey: effect-matched sets; PLS-DA vs CVAE only {pmath(p)}", dx=-0.30, dy=1.03)

    save(fig, "figV16_gene_sides")


fig_gene_sides()
