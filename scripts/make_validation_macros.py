"""Turn the external-validation results (V9ADexternal, V10ADgenevalidation,
V11ADrobustness) in data/validation/ into LaTeX macros in
generated/validation_values.tex, so every validation number in the text is
generated, never typed."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "validation"
GEN = ROOT / "generated"
spec = importlib.util.spec_from_file_location("mm", ROOT / "scripts" / "make_macros.py")
src = (ROOT / "scripts" / "make_macros.py").read_text().split("\ndef main():")[0]
mm = {"__file__": str(ROOT / "scripts" / "make_macros.py")}
exec(src, mm)
macro_name, fmt = mm["macro_name"], mm["fmt"]

rows = []


def put(key, value):
    rows.append(("val_" + key, value))


# ---- V9: external test on GSE132903 -----------------------------------------
v9 = VAL / "v9adexternal"
runs = pd.read_csv(v9 / "loco_run_metrics.csv")
A = dict(pd.read_csv(v9 / "external_age_validation.csv").values)
bands = pd.read_csv(v9 / "external_age_bands.csv")
sel = pd.read_csv(v9 / "loco_selected_genes.csv")
put("n_external", A["n_external"]); put("n_ext_AD", A["n_AD"]); put("n_ext_control", A["n_control"])
put("age_median_AD", int(round(A["age_median_AD"]))); put("age_median_control", int(round(A["age_median_control"])))
put("age_mannwhitney_p", A["age_mannwhitney_p"]); put("auc_age_raw", A["auc_age_only_raw"])
for m, tag in (("pls", "pls"), ("cvae_neutral", "cvae"), ("cvae_true", "cvaecond"), ("age", "age")):
    for k in ("auc", "ap", "bacc"):
        put(f"ext_{k}_{tag}_mean", runs[f"{k}_{m}"].mean()); put(f"ext_{k}_{tag}_sd", runs[f"{k}_{m}"].std())
put("ext_cvae_vs_pls_p", ttest_rel(runs.auc_cvae_neutral, runs.auc_pls).pvalue)
for m, tag in (("pls", "pls"), ("cvae_neutral", "cvae")):
    put(f"ext_lr_beyond_age_{tag}_p", A[f"lr_{m}_beyond_age_p"])
    for g in ("control", "AD"):
        put(f"ext_rho_age_{tag}_{g}", A[f"rho_age_{m}_{g}"]); put(f"ext_rho_age_{tag}_{g}_p", A[f"rho_age_{m}_{g}_p"])
for i, b in enumerate(bands.itertuples()):
    w = ["one", "two", "three"][i]
    put(f"band_{w}_n_AD", b.n_AD); put(f"band_{w}_n_control", b.n_control)
    put(f"band_{w}_auc_pls", b.auc_pls_mean); put(f"band_{w}_auc_cvae", b.auc_cvae_neutral_mean)
    put(f"band_{w}_auc_age", b.auc_age_mean)
put("frac_age_genes_background", A["frac_age_genes_background"])
put("frac_age_genes_pls", A["frac_age_genes_PLS-DA"]); put("frac_age_genes_cvae", A["frac_age_genes_CVAE"])
put("n_genes_frozen_pls", sel[sel.method == "PLS-DA"].gene.nunique())
put("n_genes_frozen_cvae", sel[sel.method == "CVAE"].gene.nunique())

# ---- V10: frozen lists in independent cohorts ---------------------------------
v10 = VAL / "v10adgenevalidation"
sv = pd.read_csv(v10 / "set_validation.csv")
vc = pd.read_csv(v10 / "validation_cohorts.csv")
for r in vc.itertuples():
    put(f"{r.cohort.lower()}_donors", r.donors)
SET = {"PLS-DA": "pls", "CVAE": "cvae", "Core": "core"}
for r in sv.itertuples():
    t = f"{SET[r.set]}_{r.cohort.lower()}"
    put(f"conc_{t}", r.concordance); put(f"rep_{t}", r.replication)
    put(f"nullrep_{t}", r.null_rep_mean); put(f"toprep_{t}", r.topDE_replication)
    if r.cohort != "META":
        put(f"sigauc_{t}", r.sig_auc); put(f"sigauc_{t}_p", r.p_auc)
        put(f"nullauc_{t}", r.null_auc_mean); put(f"topauc_{t}", r.topDE_auc)
    put(f"n_{t}", r.n)
gv = pd.read_csv(v10 / "gene_validation_table.csv")
put("n_core_genes", int(gv.sets.str.contains("Core").sum()))
put("n_meta_replicated", int(gv.meta_replicated.sum())); put("n_meta_tested", len(gv))
dose = pd.read_csv(v10 / "dose_response.csv")
for r in dose.itertuples():
    put(f"dose_{SET[r.set]}_{r.cohort.lower()}_{r.measure}", r.partial_rho)
    put(f"dose_{SET[r.set]}_{r.cohort.lower()}_{r.measure}_p", r.p_vs_matched)

# ---- V11: beyond cell composition, RNA-seq ------------------------------------
v11 = VAL / "v11adrobustness"
B = pd.read_csv(v11 / "signature_beyond_composition.csv")
for r in B.itertuples():
    t = f"{SET[r.set]}_{r.cohort.lower()}"
    put(f"comp_base_{t}", r.cv_auc_age_sex_cells); put(f"comp_plus_{t}", r.cv_auc_plus_signature)
    put(f"comp_gain_{t}", r.delta_auc); put(f"comp_gain_{t}_p", r.p_vs_matched_dauc)
put("comp_pls_n_pass", int(((B.set == "PLS-DA") & (B.p_vs_matched_dauc < 0.05)).sum()))
put("comp_cvae_n_pass", int(((B.set == "CVAE") & (B.p_vs_matched_dauc < 0.05)).sum()))
put("comp_n_cohorts", B.cohort.nunique())
G = pd.read_csv(v11 / "gene_effects_composition_adjusted.csv")
for s in ("PLS-DA", "CVAE", "Core"):
    g = G[G.set == s]
    put(f"{SET[s]}_rep_raw", int(g.replicates_raw.sum())); put(f"{SET[s]}_rep_adj", int(g.replicates_adj.sum()))
    put(f"{SET[s]}_rep_total", len(g))
for r in G[G.set == "Core"].itertuples():
    k = r.gene.lower().replace("-", "")
    put(f"core_{k}_raw", r.pooled_raw); put(f"core_{k}_adj", r.pooled_adj)
    put(f"core_{k}_q_adj", r.q_adj); put(f"core_{k}_kept", 100 * r.retained_frac)
Q = pd.read_csv(v11 / "cell_composition_qc.csv").dropna(subset=["AD_minus_control_d"])
neu = Q[Q.cell == "neurons"].AD_minus_control_d
put("neuron_d_min", neu.min()); put("neuron_d_max", neu.max())
R = pd.read_csv(v11 / "rnaseq_signature.csv")
for r in R.itertuples():
    put(f"rnaseq_auc_{SET[r.set]}", r.signature_auc); put(f"rnaseq_auc_{SET[r.set]}_p", r.p_vs_matched)
    put(f"rnaseq_top_{SET[r.set]}", r.topDE_auc); put(f"rnaseq_null_{SET[r.set]}", r.null_mean)
D = pd.read_csv(v11 / "dose_response_composition_adjusted.csv")
for r in D.itertuples():
    t = f"{SET[r.set]}_{r.cohort.lower()}_{r.measure}"
    put(f"dosecell_{t}", r.rho_age_cells_adjusted * r.expected_sign)
    put(f"dosecell_{t}_p", r.p_vs_matched_cells_adjusted)

# ---- pass counts over all six AD-vs-control cohorts (5 microarray + RNA-seq) ----
for s_, t_ in SET.items():
    m = sv[(sv.set == s_) & (sv.cohort != "META")]
    rr = R[R.set == s_]
    n_pass = int((m.p_auc < 0.05).sum()) + int((rr.p_vs_matched < 0.05).sum())
    n_top = int((m.sig_auc > m.topDE_auc).sum()) + int((rr.signature_auc > rr.topDE_auc).sum())
    put(f"sig_{t_}_n_pass", n_pass); put(f"sig_{t_}_n_cohorts", len(m) + len(rr)); put(f"sig_{t_}_n_beats_top", n_top)

# ---- V15: the benchmark with age taken out (sensitivity analysis) -------------
v15 = VAL / "v15adage"
ARM = {"V7, no age handling": "vseven", "A, age-corrected": "corr", "B, age-matched": "match"}
Hd = pd.read_csv(v15 / "comparison_headline.csv")
for r in Hd.itertuples():
    a = ARM[r.arm]
    put(f"v15_donors_{a}", int(r.donors)); put(f"v15_arrays_{a}", int(r.arrays))
    for m in ("pls", "cvae_agn", "age_only"):
        put(f"v15_auc_{m}_{a}", getattr(r, f"auc_{m}")); put(f"v15_auc_{m}_{a}_sd", getattr(r, f"auc_{m}_sd"))
    put(f"v15_wins_{a}", int(r.cvae_agn_wins)); put(f"v15_cvae_vs_pls_{a}_p", r.cvae_agn_vs_pls_p)
    put(f"v15_n_pls_{a}", int(r.n_genes_pls)); put(f"v15_n_cvae_{a}", int(r.n_genes_cvae))
    put(f"v15_n_shared_{a}", int(r.n_shared))
Kc = pd.read_csv(v15 / "comparison_per_cohort.csv")
for r in Kc.itertuples():
    t = f"v15_{r.cohort.lower()}_{ARM[r.arm]}"
    put(f"{t}_pls", r.auc_pls); put(f"{t}_cvae", r.auc_cvae_agn); put(f"{t}_age", r.auc_age_alone)
Gk = pd.read_csv(v15 / "comparison_genes.csv")
LST = {"PLS-DA": "pls", "CVAE": "cvae", "shared by both models": "shared"}
for r in Gk.itertuples():
    put(f"v15_kept_{LST[r.list]}_{ARM[r.arm]}", int(r.kept_from_V7))
put("v15_v7_n_shared", int(Gk.V7[Gk.list == "shared by both models"].iloc[0]))
Ag = pd.read_csv(v15 / "comparison_age_genes.csv")
put("v15_agefrac_background", int(round(100 * Ag.frac_age_genes[Ag.list == "background"].iloc[0])))
for r in Ag[Ag.list != "background"].itertuples():
    put(f"v15_agefrac_{LST[r.list]}_{ARM[r.arm]}", int(round(100 * r.frac_age_genes)))
Sg = pd.read_csv(v15 / "v7_shared_genes_by_arm.csv").set_index("gene")
robust = sorted(Sg.index[Sg.all(axis=1)])
put("v15_n_robust", len(robust))
put("v15_robust_genes", ", ".join(rf"\gene{{{g}}}" for g in robust[:-1]) + rf" and \gene{{{robust[-1]}}}")
bm = pd.read_csv(v15 / "B_age_matched" / "arm_age_balance.csv").set_index("cohort")
put("v15_match_gse33000_ad", int(bm.loc["GSE33000", "AD_donors"]))
put("v15_match_gse33000_ctrl", int(bm.loc["GSE33000", "control_donors"]))

# ---- V16: why the two models see different sides of AD -------------------------
v16 = VAL / "v16adgenesides"
GRP = {"shared": "shared", "PLS-DA only": "plsonly", "CVAE only": "cvaeonly", "all genes": "all"}
GS = pd.read_csv(v16 / "gene_sides_per_gene.csv").set_index("gene")
for g_, t_ in GRP.items():
    if g_ != "all genes":
        put(f"gs_n_{t_}", int((GS.group == g_).sum()))
SM = pd.read_csv(v16 / "gene_sides_summary.csv")
for r in SM.itertuples():
    if r.group == "PLS-DA only vs CVAE only":
        put(f"gs_{r.metric.replace('_', '')}_diff_p", r.p_vs_matched)
        continue
    t_ = GRP[r.group]; m_ = r.metric.replace("_", "")
    if r.metric == "effect_retained":
        put(f"gs_retained_{t_}_pct", int(round(100 * r.median)))
        put(f"gs_retained_{t_}_null_pct", int(round(100 * r.null_matched_median)))
    else:
        put(f"gs_{m_}_{t_}", r.median); put(f"gs_{m_}_{t_}_null", r.null_matched_median)
    put(f"gs_{m_}_{t_}_p", r.p_vs_matched)
EC = pd.read_csv(v16 / "gene_sides_effect_classes.csv")
for r in EC.itertuples():
    t_ = GRP[r.group]
    put(f"gs_cls_{t_}_{r.effect_class}_n", int(r.n))
    put(f"gs_cls_{t_}_{r.effect_class}_pct", int(round(100 * r.fraction)))
    if r.group != "all genes":
        put(f"gs_cls_{t_}_{r.effect_class}_null_pct", int(round(100 * r.null_matched_fraction)))
        put(f"gs_cls_{t_}_{r.effect_class}_p", r.p_vs_matched)
FD = pd.read_csv(v16 / "gene_sides_fisher.csv").iloc[0]
put("gs_direct_or", float(FD.odds_ratio)); put("gs_direct_fisher_p", FD.p)
CO = pd.read_csv(v16 / "gene_sides_coherence.csv")
for r in CO.itertuples():
    t_ = GRP[r.group]
    put(f"gs_coh_{t_}", r.coherence); put(f"gs_coh_{t_}_null", r.null_matched_mean)
    put(f"gs_coh_{t_}_fold", r.fold_vs_matched); put(f"gs_coh_{t_}_p", r.p_vs_matched)
put("gs_coh_diff_p", pd.read_csv(v16 / "gene_sides_coherence_diff.csv").permutation_p.iloc[0])
MK = pd.read_csv(v16 / "gene_sides_celltypes.csv")
for r in MK.itertuples():
    t_ = GRP[r.group]
    put(f"gs_mk_{t_}_n", int(r.n_markers)); put(f"gs_mk_{t_}_pct", int(round(100 * r.n_markers / r.n_genes)))
    if r.group != "all genes":
        put(f"gs_mk_{t_}_null_pct", int(round(100 * r.null_matched_fraction))); put(f"gs_mk_{t_}_p", r.p_vs_matched)
put("gs_mk_fisher_p", pd.read_csv(v16 / "gene_sides_fisher_markers.csv").p.iloc[0])
AT = pd.read_csv(v16 / "gene_sides_attribution.csv")
for r in AT.itertuples():
    m_ = r.property.replace("_", "")
    put(f"gs_att_{r.score}_{m_}", r.spearman)
    if not np.isnan(r.partial_spearman_given_abs_d):
        put(f"gs_att_{r.score}_{m_}_partial", r.partial_spearman_given_abs_d)
CS = pd.read_csv(v16 / "gene_sides_cell_scores.csv")
for r in CS.itertuples():
    put(f"gs_cellshift_{r.cell.lower()}", r.mean_AD_minus_control)


def gene_list(genes):
    genes = list(genes)
    if len(genes) == 1:
        return rf"\gene{{{genes[0]}}}"
    return ", ".join(rf"\gene{{{g}}}" for g in genes[:-1]) + rf" and \gene{{{genes[-1]}}}"


for g_, t_ in (("PLS-DA only", "plsonly"), ("CVAE only", "cvaeonly"), ("shared", "shared")):
    sub = GS[GS.group == g_]
    for cls in ("direct", "compositional", "masked", "neither"):
        s_ = sub[sub.effect_class == cls]
        s_ = s_.reindex(s_.t_adj.abs().sort_values(ascending=False).index)
        if len(s_):
            put(f"gs_genes_{t_}_{cls}", gene_list(s_.index))
    mk_ = sub[sub.marker_top.fillna("") != ""].sort_values("marker_rank")
    if len(mk_):
        put(f"gs_genes_{t_}_markers", ", ".join(rf"\gene{{{g}}}" for g in mk_.index))

# sex sensitivity (V16 version 2)
SXC = pd.read_csv(v16 / "gene_sides_sex_cohorts.csv").set_index("cohort")
put("gs_sex_ad_female_pct", int(round(100 * SXC.at["merged", "AD_female_frac"])))
put("gs_sex_ctrl_female_pct", int(round(100 * SXC.at["merged", "control_female_frac"])))
put("gs_sex_gse33000_ad_female_pct", int(round(100 * SXC.at["GSE33000", "AD_female_frac"])))
put("gs_sex_gse33000_ctrl_female_pct", int(round(100 * SXC.at["GSE33000", "control_female_frac"])))
_sx = SXC.drop(index="merged").auc_sex_alone
put("gs_sex_auc_max", float(_sx.max())); put("gs_sex_auc_max_cohort", _sx.idxmax())
SXG = pd.read_csv(v16 / "gene_sides_sex_genes.csv").set_index("group")
for g_, t_ in GRP.items():
    put(f"gs_sex_n_{t_}", int(SXG.at[g_, "n_sex_associated_controls"]))
put("gs_sex_all_pct", float(100 * SXG.at["all genes", "frac_sex_associated_controls"]))
SXK = pd.read_csv(v16 / "gene_sides_sex_canonical.csv")
put("gs_sex_canon_n", len(SXK)); put("gs_sex_canon_best_vip_rank", int(SXK.vip_rank.min()))
put("gs_sex_canon_best_ig_rank", int(SXK.ig_rank.min()))
# tissue specificity of the selected genes (Human Protein Atlas)
HP_ = pd.read_csv(v16 / "gene_sides_hpa_tissue.csv")
put("gs_hpa_n_total", int(HP_.non_brain_enriched.sum()))
for g_, t_ in (("shared", "shared"), ("PLS-DA only", "plsonly"), ("CVAE only", "cvaeonly")):
    s_ = HP_[(HP_.group == g_) & HP_.non_brain_enriched]
    put(f"gs_hpa_n_{t_}", len(s_))
    if len(s_):
        put(f"gs_hpa_genes_{t_}", gene_list(sorted(s_.gene)))
put("gs_hpa_brain_not_detected", gene_list(sorted(HP_.gene[HP_.brain_distribution == "Not detected"])))
# what the enrichment terms are: term counts by database
ENR = {n: pd.read_csv(ROOT / "data" / "enrichment" / f"{n}.csv") for n in ("plsda_down", "plsda_up", "cvae_down", "cvae_up")}
put("enr_all_terms_n", int(sum(len(d) for d in ENR.values())))
_pd = ENR["plsda_down"]
put("enr_pdown_terms_n", len(_pd)); put("enr_pdown_hp_n", int((_pd.source == "HP").sum()))
put("enr_pdown_hpa_n", int((_pd.source == "HPA").sum()))
_hp = _pd[_pd.source == "HP"]
_hpg = pd.Series([g for x in _hp.intersections for g in x.split(",")]).value_counts()
put("enr_pdown_hp_spry_n", int(_hpg.get("SPRY4", 0))); put("enr_pdown_hp_hbb_n", int(_hpg.get("HBB", 0)))
put("enr_pdown_hp_tubb_n", int(_hpg.get("TUBB2B", 0)))
put("enr_pdown_hp_maxsize_k2", int(_hp[_hp.intersection_size <= 2].term_size.max()))


# the functional map of the selected genes: themes (curated from NCBI Gene, in
# data/gene_themes.csv) by provenance, each gene with its direction in AD and its
# composition class from V16
TH = pd.read_csv(ROOT / "data" / "gene_themes.csv")
MARK = {"direct": "", "compositional": "^{c}", "masked": "^{m}", "neither": r"^{\circ}"}
ARROW = {"up": r"\uparrow", "down": r"\downarrow"}
ORDER_DIR = {"down": 0, "up": 1}
th_lines = []
for theme, sub in TH.groupby("theme", sort=False):
    cells = []
    for g_ in ("shared", "PLS-DA only", "CVAE only"):
        gs_ = [g for g in sub.gene if GS.at[g, "group"] == g_]
        def _key(g):
            c_ = GS.at[g, "effect_class"]
            t_ = GS.at[g, "t_adj"] if c_ == "masked" else GS.at[g, "t_raw"]
            return (2 if c_ == "neither" else (0 if t_ < 0 else 1), g)
        gs_.sort(key=_key)
        items = []
        for g in gs_:
            # the arrow is the tested donor-level effect: in bulk for direct and
            # compositional genes, after cell adjustment for masked genes, none otherwise
            cls_ = GS.at[g, "effect_class"]
            t_ = GS.at[g, "t_adj"] if cls_ == "masked" else GS.at[g, "t_raw"]
            arrow = "" if cls_ == "neither" else (r"\uparrow" if t_ > 0 else r"\downarrow")
            mark = MARK[cls_]
            tail = f"${arrow}{mark}$" if (arrow or mark) else ""
            items.append(rf"\gene{{{g}}}{tail}")
        cells.append(", ".join(items) if items else "--")
    th_lines.append(rf"{theme} & {cells[0]} & {cells[1]} & {cells[2]} \\[2pt]")
th_lines.append("\\bottomrule")   # inside the file: \bottomrule after \input breaks the alignment
(GEN / "gene_themes_table.tex").write_text("\n".join(th_lines) + "\n")
for g_, t_ in (("shared", "shared"), ("PLS-DA only", "plsonly"), ("CVAE only", "cvaeonly")):
    n_th = TH[TH.gene.map(GS.group) == g_].theme.nunique()
    put(f"gs_n_themes_{t_}", int(n_th))

# ---- figure-level values for the redesigned main figures ---------------------
D7 = ROOT / "data"
rec = pd.read_csv(D7 / "oof_reconstruction.csv")
for col, tag in (("mse_pls", "pls"), ("mse_cvae_neutral", "cvae")):
    for g in ("AD", "Control"):
        v = rec.loc[rec.group == g, col]
        put(f"fig_rec_{tag}_{g.lower()}_median", v.median()); put(f"fig_rec_{tag}_{g.lower()}_mean", v.mean())
    put(f"fig_rec_{tag}_median", rec[col].median())
put("fig_rec_cvae_worse_pct", 100 * (rec.mse_cvae_neutral > rec.mse_pls).mean())
rm_ = pd.read_csv(D7 / "run_metrics.csv")
for m, sgn in (("sil", 1), ("iir", 1), ("ch", 1)):
    put(f"fig_sep_{m}_fold", float(2 ** np.mean(np.log2(rm_[f"{m}_cvae_neutral"] / rm_[f"{m}_pls"]) * sgn)))
    put(f"fig_sep_{m}_wins", int((rm_[f"{m}_cvae_neutral"] > rm_[f"{m}_pls"]).sum()))
put("fig_sep_ari_diff", float((rm_.ari_cvae_neutral - rm_.ari_pls).mean()))
put("fig_sep_ari_wins", int((rm_.ari_cvae_neutral > rm_.ari_pls).sum()))
put("fig_auc_wins", int((rm_.auc_cvae_lat_neutral > rm_.auc_pls).sum()))
Pg = pd.read_csv(D7 / "PLSDA_gene_significance.csv"); Cg = pd.read_csv(D7 / "CVAE_gene_significance.csv")
de_ = pd.read_csv(D7 / "gene_de_stats.csv").set_index("gene")
Mn = pd.read_csv(VAL / "v10adgenevalidation" / "meta_analysis_new_cohorts_only.csv", index_col=0)
bgm = Mn.join(de_, how="inner")
repl = (bgm.q < 0.05) & (np.sign(bgm.pooled) == np.sign(bgm.logFC))
ps_, cs_ = set(Pg.gene[Pg.significant]), set(Cg.gene[Cg.significant])
for nm, s_ in (("plsonly", ps_ - cs_), ("both", ps_ & cs_), ("cvaeonly", cs_ - ps_), ("all", ps_ | cs_)):
    s_ = [g for g in s_ if g in repl.index]
    put(f"fig_rep_{nm}_n", int(repl[s_].sum())); put(f"fig_rep_{nm}_pct", int(round(100 * repl[s_].mean())))
put("fig_rep_background_pct", int(round(100 * repl.mean())))
for f_, tag, sel_ in (("vip", "pls", ps_), ("ig", "cvae", cs_)):
    mr, mn = np.load(D7 / f"{f_}_real.npy").mean(0), np.load(D7 / f"{f_}_null.npy").mean(0)
    si = np.array([g in sel_ for g in de_.index])
    put(f"fig_null_{tag}_fold", float(np.median(mr[si] / mn[si])))
put("fig_up_n", int(sum(de_.loc[g, "mean_z_AD"] > de_.loc[g, "mean_z_control"] for g in ps_ | cs_)))

# ---- core validation figure: margins, dose tallies, core survivors, list overlap --
svm = sv[sv.cohort != "META"]
for s_, t_ in (("PLS-DA", "pls"), ("CVAE", "cvae")):
    m_ = svm[svm.set == s_]; rr_ = R[R.set == s_]
    marg = np.r_[(m_.sig_auc - m_.topDE_auc).values, (rr_.signature_auc - rr_.topDE_auc).values]
    put(f"core_topde_margin_{t_}_min", float(marg.min())); put(f"core_topde_margin_{t_}_max", float(marg.max()))
    put(f"core_topde_margin_{t_}_median", float(np.median(marg)))
    put(f"core_dose_{t_}_n_pass", int(((D.set == s_) & (D.p_vs_matched_cells_adjusted < 0.05)).sum()))
nulls = np.r_[svm[svm.set != "Core"].null_auc_mean.values, R[R.set != "Core"].null_mean.values]
put("core_null_auc_min", float(nulls.min())); put("core_null_auc_max", float(nulls.max()))
put("core_dose_n_tests", int((D.set == "PLS-DA").sum()))
Gc = G[G.set == "Core"]
put("core_n_survive", int((Gc.q_adj < 0.05).sum())); put("core_n_total", len(Gc))
P7 = pd.read_csv(ROOT / "data" / "PLSDA_gene_significance.csv"); C7 = pd.read_csv(ROOT / "data" / "CVAE_gene_significance.csv")
p7, c7 = set(P7.gene[P7.significant]), set(C7.gene[C7.significant])
p9, c9 = set(sel.gene[sel.method == "PLS-DA"]), set(sel.gene[sel.method == "CVAE"])
put("core_overlap_pls", len(p7 & p9)); put("core_overlap_pls_total", len(p7))
put("core_overlap_cvae", len(c7 & c9)); put("core_overlap_cvae_total", len(c7))

# ---- cohort table for the Methods -------------------------------------------
ct = pd.read_csv(VAL / "ad-merged-geo-cohorts" / "cohort_table.csv")
PLAT = {"GPL570": "Affymetrix U133 Plus 2.0", "GPL4372": "Agilent/Rosetta 44K", "GPL10558": "Illumina HT-12 v4",
        "GPL96": "Affymetrix U133A"}
REG = {"GSE48350": "HC, EC, SFG, PCG", "GSE33000": "prefrontal cortex", "GSE132903": "middle temporal gyrus",
       "GSE118553": "temporal, entorhinal, frontal", "GSE84422": "15 cortical/hippocampal"}
CITE = {"GSE48350": "berchtold2008gene", "GSE33000": "narayanan2014common", "GSE132903": "piras2019transcriptome",
        "GSE118553": "patel2019transcriptomic", "GSE84422": "wang2016integrative"}
tl = []
for r in ct[ct.cohort != "TOTAL"].itertuples():
    tl.append(f"{r.cohort} & {PLAT[r.platform]} & {REG[r.cohort]} & "
              f"{int(r.after_age_filter):,} & {int(r.donors):,} & {int(r.AD_donors)}/{int(r.control_donors)} & "
              f"{r.age_median_AD:.0f}/{r.age_median_control:.0f} \\\\".replace(",", "{,}"))
t = ct[ct.cohort == "TOTAL"].iloc[0]
tl.append("\\midrule")
tl.append(f"Merged & & & {int(float(t.after_age_filter)):,} & {int(float(t.donors)):,} & "
          f"{int(float(t.AD_donors))}/{int(float(t.control_donors))} & "
          f"{float(t.age_median_AD):.0f}/{float(t.age_median_control):.0f} \\\\".replace(",", "{,}"))
tl.append("\\bottomrule")
(GEN / "cohort_table.tex").write_text("\n".join(tl) + "\n")

lines = ["% Generated by scripts/make_validation_macros.py -- do not edit by hand.",
         "% Sources: data/validation/{v9adexternal,v10adgenevalidation,v11adrobustness}", ""]
seen = {}
for k, v in rows:
    name = macro_name(k)
    if name in seen:
        raise SystemExit(f"macro collision: {k} and {seen[name]}")
    seen[name] = k
    if isinstance(v, (float, np.floating)) and k.split("_")[-1] in ("kept",):
        val = f"{v:.0f}"
    elif isinstance(v, (int, np.integer)):
        val = f"{int(v):,}".replace(",", "{,}")
    else:
        val = fmt(k, v)
    lines.append(rf"\newcommand{{\{name}}}{{{val}}}")
(GEN / "validation_values.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {GEN / 'validation_values.tex'} with {len(seen)} macros")
