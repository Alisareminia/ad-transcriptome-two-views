"""The single core figure of the external validation in paper3.

(a) the held-out cohort, (b) its age bands, (c) the genes both models chose before
and after cell-composition adjustment, and (d) an evidence map: every validation
cohort against every test, PLS-DA and CVAE side by side as the two halves of one
disc. Reads V9ADexternal, V10ADgenevalidation and V11ADrobustness outputs only and
reuses the style of validation_figures.py (its header is executed, not its figures)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
exec((HERE / "validation_figures.py").read_text().split("\ndef fig1_design")[0])
from matplotlib.patches import Wedge

V9, V10, V11 = "v9adexternal", "v10adgenevalidation", "v11adrobustness"
SETS = [("PLS-DA", PLS), ("CVAE", CVAE_N)]
PAPER_BG = "#f6f6f9"


def half_disc(ax, x, y, r, passes):
    """Left half PLS-DA, right half CVAE; solid when the test passes, pale ring when not."""
    for (s, col), (t1, t2), ok in zip(SETS, ((90, 270), (-90, 90)), passes):
        if ok is None:
            continue
        ax.add_patch(Wedge((x, y), r, t1, t2, facecolor=col if ok else tint(col, 0.13),
                           edgecolor=col, linewidth=0.0 if ok else 0.9, zorder=3))
    ax.plot([x, x], [y - r, y + r], color=SURFACE, lw=0.9, zorder=4)


def fig_core():
    sv = rd(V10, "set_validation.csv"); sv = sv[sv.cohort != "META"]
    rs = rd(V11, "rnaseq_signature.csv"); bc = rd(V11, "signature_beyond_composition.csv")
    D = rd(V11, "dose_response_composition_adjusted.csv")
    G = rd(V11, "gene_effects_composition_adjusted.csv"); G = G[G.set == "Core"]
    vc = rd(V10, "validation_cohorts.csv").set_index("cohort")
    runs = rd(V9, "loco_run_metrics.csv"); A = dict(rd(V9, "external_age_validation.csv").values)
    bands = rd(V9, "external_age_bands.csv"); roc = rd(V9, "loco_roc_curves.csv")
    rna = rnaseq_counts()

    fig = plt.figure(figsize=(TW, 8.7))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.20, left=0.07, right=0.985,
                             top=0.955, bottom=0.015)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 0.82, 1.35], wspace=0.62)

    # ---- (a) the held-out cohort ------------------------------------------------------------
    ax = fig.add_subplot(top[0]); ax.set_box_aspect(1); ax.set_anchor("N")
    base = np.linspace(0, 1, 101)
    for m, c, ls, lw in (("age", MID, (0, (3, 2)), 1.0), ("cvae_neutral", CVAE_N, "-", 1.5), ("pls", PLS, "-", 1.7)):
        v = roc[roc.model == m][[f"tpr_{i}" for i in range(101)]].values
        mu, sd = v.mean(0), v.std(0)
        ax.fill_between(base, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=c, alpha=0.12, lw=0)
        ax.plot(base, mu, color=c, lw=lw, ls=ls)
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.5)
    for k, (m, lab, c) in enumerate((("pls", "PLS-DA", PLS), ("cvae_neutral", "CVAE", CVAE_N), ("age", "age", MID))):
        ax.text(0.97, 0.05 + 0.13 * (2 - k), f"{runs[f'auc_{m}'].mean():.2f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9.5 if m != "age" else 7.5, color=c, fontweight="bold" if m != "age" else "normal")
        ax.text(0.70, 0.065 + 0.13 * (2 - k), lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
                color=c)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel("false-positive rate", fontsize=6.6); ax.set_ylabel("true-positive rate", fontsize=6.6)
    panel(ax, "a", "A cohort never seen", f"{HOLDOUT} · {int(A['n_AD'])} AD / {int(A['n_control'])} control",
          dx=-0.36)

    # ---- (b) age bands ----------------------------------------------------------------------------
    ax = fig.add_subplot(top[1])
    x = np.arange(len(bands))
    for m, c, mk, dx_ in (("pls", PLS, "o", -0.2), ("cvae_neutral", CVAE_N, "s", 0.0), ("age", MID, "D", 0.2)):
        v = bands[f"auc_{m}_mean"].values; e = bands[f"auc_{m}_sd"].values
        ax.errorbar(x + dx_, v, yerr=e, fmt=mk, color=c, ms=4.6 if m != "age" else 3.8, lw=0.9, capsize=0, zorder=3)
    ax.plot(x - 0.2, bands.auc_pls_mean, color=PLS, lw=0.6, alpha=0.5)
    ax.plot(x, bands.auc_cvae_neutral_mean, color=CVAE_N, lw=0.6, alpha=0.5)
    ax.axhspan(0.4, 0.55, color=PAPER_BG, lw=0, zorder=0)
    ax.axhline(0.5, color=INK3, lw=0.5, ls=(0, (3, 2)))
    ax.text(len(bands) - 0.55, 0.43, "chance", ha="right", fontsize=5.6, color=INK3, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([f"{b.band} y\n{b.n_AD}/{b.n_control}" for b in bands.itertuples()],
                                         fontsize=5.9)
    ax.tick_params(axis="x", length=0); ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.set_ylim(0.4, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_ylabel("ROC-AUC", fontsize=6.6)
    ygrid(ax)
    panel(ax, "b", "Not an age effect", "AUC inside each age band", dx=-0.42)

    # ---- (c) genes both models chose: before and after cell composition --------------------------
    ax = fig.add_subplot(top[2])
    G = G.sort_values("pooled_adj").reset_index(drop=True)
    keep = G.q_adj < 0.05
    for r, ok in zip(G.itertuples(), keep):
        c = ACCENT if ok else "#b7b7c2"
        ax.plot([0, 1], [r.pooled_raw, r.pooled_adj], color=c, lw=1.8 if ok else 0.8, zorder=3 if ok else 2,
                solid_capstyle="round")
        ax.scatter([0], [r.pooled_raw], s=14 if ok else 8, color=c, zorder=4, lw=0)
        ax.scatter([1], [r.pooled_adj], s=20 if ok else 8, color=c, zorder=4, lw=0)
    ys = G.pooled_adj.values; order = np.argsort(ys)
    gap = 0.075; pos = []
    for v in ys[order]:
        pos.append(max(v, pos[-1] + gap) if pos else v)
    over = pos[-1] - 1.0
    if over > 0:
        pos = [p_ - over for p_ in pos]
    for p_, i in zip(pos, order):
        r = G.iloc[i]; ok = r.q_adj < 0.05
        qs = "q<0.001" if r.q_adj < 0.001 else f"q {r.q_adj:.2g}"
        ax.annotate(f"{r.gene}", xy=(1, r.pooled_adj), xytext=(1.12, p_), textcoords="data",
                    fontsize=6.0 if ok else 5.4, fontstyle="italic", va="center", ha="left",
                    color=ACCENT if ok else INK3, fontweight="bold" if ok else "normal", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=RULE, lw=0.35, shrinkA=0, shrinkB=2))
        ax.text(1.62, p_, qs, fontsize=5.2 if not ok else 5.6, va="center", ha="left",
                color=ACCENT if ok else INK3, fontweight="bold" if ok else "normal")
    ax.axhline(0, color=INK3, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlim(-0.12, 2.05); ax.set_ylim(-0.15, 1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["age, sex", "+ 6 cell\ntypes"], fontsize=6.0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("pooled effect, trained direction (SD)", fontsize=6.4); ygrid(ax)
    ax.spines["bottom"].set_visible(False)
    panel(ax, "c", f"{int(keep.sum())} of {len(G)} shared genes survive cell mix",
          "meta-analysis, five cohorts not used to define them", dx=-0.25)

    # ---- (d) the evidence map ---------------------------------------------------------------------
    ax = fig.add_subplot(outer[1])
    ROWS = [("held out", AD, [HOLDOUT]), ("independent microarray", ACCENT, VALID),
            ("RNA-seq", CVAE_N, [RNASEQ]), ("pathology only", MID, SEVERITY)]
    COLS = ["Separates AD\nfrom control", "Beats matched\nrandom genes", "Beats plain\ntop-N DE",
            "Adds beyond\ncell mix", "Tracks Braak\nstage", "Tracks plaques\nor MMSE"]
    cx = [6.75 + 1.55 * k for k in range(len(COLS))]
    R = 0.30
    y = 0.0; nrow = 0
    tally = {s: [[0, 0] for _ in COLS] for s, _ in SETS}

    def sig_row(c, s):
        if c == RNASEQ:
            r = rs[rs.set == s].iloc[0]
            return r.signature_auc, r.p_vs_matched, r.topDE_auc
        r = sv[(sv.set == s) & (sv.cohort == c)]
        if not len(r):
            return None
        r = r.iloc[0]
        return r.sig_auc, r.p_auc, r.topDE_auc

    for gname, gcol, cohorts in ROWS:
        y0 = y
        for c in cohorts:
            nrow += 1
            if nrow % 2 == 1:
                ax.add_patch(Rectangle((0.0, y - 0.68), cx[-1] + 0.95, 1.30, facecolor=PAPER_BG, lw=0, zorder=0))
            if c == RNASEQ:
                n_ad, n_ct, plat = rna[1], rna[2], "GPL16791"
            else:
                n_ad, n_ct, plat = int(vc.loc[c, "AD"]), int(vc.loc[c, "control"]), vc.loc[c, "platform"]
            ax.text(0.55, y + 0.20, c, fontsize=6.8, color=INK, va="center", fontweight="bold")
            ax.text(0.55, y - 0.20, f"{PLATFORM.get(plat, plat)} · {REGION.get(c, '')}", fontsize=5.0, color=INK2,
                    va="center")
            if n_ad + n_ct > 0:
                w = 0.80
                ax.add_patch(Rectangle((4.75, y - 0.09), w * n_ad / 400, 0.18, facecolor=AD, lw=0))
                ax.add_patch(Rectangle((4.75 + w * n_ad / 400, y - 0.09), w * n_ct / 400, 0.18, facecolor=CTRL, lw=0))
                ax.text(4.75 + w * (n_ad + n_ct) / 400 + 0.08, y, f"{n_ad}/{n_ct}", fontsize=5.2, color=INK2,
                        va="center")
            else:
                ax.text(4.75, y, f"{int(vc.loc[c, 'donors'])} graded", fontsize=5.2, color=INK3, va="center",
                        style="italic")
            vals = {}
            for s, _ in SETS:
                sr = sig_row(c, s)
                b = bc[(bc.set == s) & (bc.cohort == c)]
                d_b = D[(D.cohort == c) & (D.set == s) & (D.measure == "braak")]
                d_o = D[(D.cohort == c) & (D.set == s) & (D.measure.isin(["plaque", "mmse"]))]
                vals[s] = [
                    (sr[0], sr[1] < 0.05) if sr else None,
                    (sr[1], sr[1] < 0.05) if sr else None,
                    (sr[0] - sr[2], sr[0] > sr[2]) if sr else None,
                    (b.delta_auc.iloc[0], b.p_vs_matched_dauc.iloc[0] < 0.05) if len(b) else None,
                    (d_b.rho_age_cells_adjusted.iloc[0] * d_b.expected_sign.iloc[0],
                     d_b.p_vs_matched_cells_adjusted.iloc[0] < 0.05) if len(d_b) else None,
                    (d_o.rho_age_cells_adjusted.iloc[0] * d_o.expected_sign.iloc[0],
                     d_o.p_vs_matched_cells_adjusted.iloc[0] < 0.05) if len(d_o) else None]
            for k in range(len(COLS)):
                pl, cv_ = vals["PLS-DA"][k], vals["CVAE"][k]
                if pl is None and cv_ is None:
                    ax.plot([cx[k] - 0.12, cx[k] + 0.12], [y, y], color=RULE, lw=0.8)
                    continue
                half_disc(ax, cx[k], y, R, [True, True] if k == 0 else
                          [pl[1] if pl else None, cv_[1] if cv_ else None])
                for s_i, (s, col) in enumerate(SETS):
                    vv = vals[s][k]
                    if vv is None:
                        continue
                    if k > 0:
                        tally[s][k][1] += 1; tally[s][k][0] += int(vv[1])
                    if k == 0:
                        txt = f"{vv[0]:.2f}"
                    elif k == 1:
                        txt = "p<.001" if vv[0] < 0.001 else f"p {vv[0]:.2f}".replace("0.", ".")
                    else:
                        txt = f"{vv[0]:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")
                    ax.text(cx[k] + (-0.05 if s_i == 0 else 0.05), y - R - 0.17, txt,
                            ha="right" if s_i == 0 else "left", va="center", fontsize=5.0, color=col)
            y -= 1.36
        ax.plot([0.25, 0.25], [y0 + 0.5, y + 1.36 - 0.6], color=gcol, lw=2.6, solid_capstyle="butt")
        ax.text(0.10, (y0 + y + 1.36) / 2 - 0.05, gname, rotation=90, ha="right", va="center", fontsize=5.8, color=gcol)
        y -= 0.35
    ytop = 1.15
    for k, lab in enumerate(COLS):
        ax.text(cx[k], ytop, lab, ha="center", va="bottom", fontsize=5.9, color=INK, linespacing=1.15)
        if k > 0:
            t_p, t_c = tally["PLS-DA"][k], tally["CVAE"][k]
            ax.text(cx[k] - 0.08, y + 0.55, f"{t_p[0]}/{t_p[1]}", ha="right", va="center", fontsize=7.0, color=PLS,
                    fontweight="bold")
            ax.text(cx[k] + 0.08, y + 0.55, f"{t_c[0]}/{t_c[1]}", ha="left", va="center", fontsize=7.0, color=CVAE_N,
                    fontweight="bold")
    ax.text(0.55, y + 0.55, "tests passed (PLS-DA | CVAE)", fontsize=6.4, color=INK, va="center", fontweight="bold")
    ax.plot([0.0, cx[-1] + 0.95], [y + 1.05, y + 1.05], color=INK2, lw=0.6)
    ax.text(0.55, ytop + 0.05, "Cohort", fontsize=6.3, color=INK, va="bottom")
    ax.text(4.75, ytop + 0.05, "AD/ctrl", fontsize=6.3, color=INK, va="bottom")
    kx, ky = 0.85, y - 0.30
    half_disc(ax, kx, ky, 0.26, [True, True])
    ax.text(kx + 0.42, ky, "left half PLS-DA, right half CVAE · solid = test passed ($p<0.05$, or above the comparison "
            "for plain DE) · pale = not passed\nvalues below each disc: AUC, $p$ against matched random genes, "
            "AUC minus plain DE, ΔAUC over cell mix, partial $\\rho$ with pathology", fontsize=5.6, color=INK2, va="center", linespacing=1.4)
    ax.set_xlim(-0.35, cx[-1] + 1.0); ax.set_ylim(ky - 0.45, ytop + 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    ax.annotate("d", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(-2, 4), textcoords="offset points",
                fontsize=9.5, fontweight="bold", family="DejaVu Sans", color=INK, va="bottom", ha="right",
                annotation_clip=False)
    ax.annotate("Evidence map: frozen gene signatures in every validation cohort, no model refitted",
                xy=(0.0, 1.0), xycoords="axes fraction", xytext=(4, 4), textcoords="offset points",
                fontsize=8.4, color=INK, va="bottom", annotation_clip=False)
    save(fig, "figV_core_validation")


fig_core()
