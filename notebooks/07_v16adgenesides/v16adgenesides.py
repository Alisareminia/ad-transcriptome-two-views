# %% [markdown]
# # V16ADgenesides -- why the two models see different sides of AD
#
# The PLS-DA and CVAE gene lists of V7ADmerged agree on a core and diverge outside
# it. This notebook tests three properties of the genes each model selected, on
# the same merged matrix and at donor level, against matched random genes:
#
# 1. **Co-expression coherence** -- do a model's genes form one co-regulated
#    programme (a module) or a set of unrelated genes (a mosaic)? Measured on
#    expression with the diagnosis effect removed, so the AD shift itself cannot
#    create the correlation.
# 2. **Cell-type alignment** -- how much of each gene's variation follows the
#    published marker-based cell-type scores (McKenzie et al. 2018), which cell type
#    it follows, and how much of its AD effect remains after adjusting for them.
# 3. **Uniform versus uneven AD effect** -- is the gene shifted in most AD donors,
#    or strongly changed in a subset (larger variance in AD, an excess tail)?
#
# and then asks the same questions genome-wide of the two importance scores
# (VIP and |IG|). Every number is written to CSV for the paper.

# %%
# ======================================================================
# Setup, inputs and gene symbols
# ======================================================================
import os, glob, gzip, tarfile, subprocess, sys, hashlib
import numpy as np, pandas as pd, requests
from scipy import stats

OUT = os.environ.get("V16_OUT", "/kaggle/working")
TMP = os.environ.get("V16_TMP", "/kaggle/tmp")
IN_ROOT = os.environ.get("V16_INPUT", "/kaggle/input")
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
HDR = {"User-Agent": "v16adgenesides (kaggle)"}
RNG = np.random.default_rng(2026)
N_NULL = int(os.environ.get("V16_NNULL", 5000))


def find(name):
    hits = sorted(glob.glob(f"{IN_ROOT}/**/{name}", recursive=True))
    assert hits, f"input file not found: {name}"
    return hits[0]


def fetch(url, name):
    path = f"{TMP}/{name}"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for attempt in range(5):
        try:
            with requests.get(url, headers=HDR, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            print(f"downloaded {name}: {os.path.getsize(path)/1e6:.1f} MB")
            return path
        except Exception as e:                       # noqa: BLE001
            print(f"  retry {attempt+1} for {name}: {e}")
    raise RuntimeError(f"could not download {url}")


# NCBI Gene: the merged matrix already carries current official symbols; the
# marker lists are mapped onto the same symbols with the project's recipe.
gi = pd.read_csv(fetch("https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
                       "Homo_sapiens.gene_info.gz"), sep="\t", dtype=str,
                 usecols=["GeneID", "Symbol", "Synonyms"])
ENTREZ2SYM = dict(zip(gi.GeneID, gi.Symbol))
OFFICIAL = set(gi.Symbol)
_syn = {}
for gid, syns in zip(gi.GeneID, gi.Synonyms):
    if isinstance(syns, str) and syns != "-":
        for s in syns.split("|"):
            _syn.setdefault(s, set()).add(gid)
SYN2SYM = {s: ENTREZ2SYM[next(iter(g))] for s, g in _syn.items() if len(g) == 1 and s not in OFFICIAL}


def norm_symbol(s):
    if not isinstance(s, str) or not s or s == "-":
        return None
    s = s.strip()
    return s if s in OFFICIAL else SYN2SYM.get(s)


for f in ("merged_expression.csv.gz", "PLSDA_gene_significance.csv", "CVAE_gene_significance.csv"):
    print(f"{f}: {find(f)}  sha256 {hashlib.sha256(open(find(f), 'rb').read()).hexdigest()[:16]}")

# %%
# ======================================================================
# Donor-level matrix
# ======================================================================
# Arrays of one donor are not independent, so every statistic below is computed
# on donors. Region differences are removed first (each gene centred within
# cohort x region), the arrays of a donor are then averaged, and each gene is
# standardised within cohort -- the same per-cohort scaling as the main analysis.
X = pd.read_csv(find("merged_expression.csv.gz"), index_col=0).astype(np.float32)
M = pd.read_csv(find("merged_metadata.csv"), index_col=0).loc[X.index]
key = (M.cohort + "|" + M.region.astype(str)).values
Xc = X - X.groupby(key).transform("mean")
D = Xc.groupby(M.donor.values).mean()
DM = M.groupby("donor").agg(cohort=("cohort", "first"), dx=("dx", "first"),
                            age=("age", "first"), sex=("sex", "first")).loc[D.index]
_g = D.groupby(DM.cohort.values)
_sd = _g.transform("std", ddof=0)
D = ((D - _g.transform("mean")) / _sd.where(_sd > 0, 1.0)).fillna(0.0).astype(np.float32)
GENES = np.array(D.columns)
AD = (DM.dx == "AD").values
COH = DM.cohort.values
COHORTS = sorted(set(COH))
print(f"donor-level matrix: {D.shape[0]} donors x {D.shape[1]:,} genes "
      f"({AD.sum()} AD, {(~AD).sum()} control)")
print(DM.groupby(["cohort", "dx"]).size().unstack().to_string())

# %%
# ======================================================================
# The gene groups
# ======================================================================
pls = pd.read_csv(find("PLSDA_gene_significance.csv")).set_index("gene")
cv = pd.read_csv(find("CVAE_gene_significance.csv")).set_index("gene")
P = set(pls.index[pls.significant.astype(str).str.lower() == "true"])
C = set(cv.index[cv.significant.astype(str).str.lower() == "true"])
GROUPS = {"shared": sorted(P & C), "PLS-DA only": sorted(P - C), "CVAE only": sorted(C - P)}
SELECTED = P | C
for g, v in GROUPS.items():
    print(f"{g:12s} {len(v):3d}")
assert len(P) == 54 and len(C) == 75 and len(P & C) == 18, "gene lists differ from V7ADmerged"
DIRECTION = {g: (pls.direction.get(g) if g in P else cv.direction.get(g)) for g in SELECTED}

# %%
# ======================================================================
# 3. Uniform versus uneven AD effects (per gene, within cohort, pooled)
# ======================================================================
Dv = D.values
rows = []
d_c, lnvr_c, tail_c, w_c = [], [], [], []
for c in COHORTS:
    m = COH == c
    a, k = Dv[m & AD], Dv[m & ~AD]
    na, nk = len(a), len(k)
    sa, sk = a.std(0, ddof=1), k.std(0, ddof=1)
    sp = np.sqrt(((na - 1) * sa**2 + (nk - 1) * sk**2) / (na + nk - 2))
    delta = a.mean(0) - k.mean(0)
    d_c.append(delta / np.where(sp > 0, sp, 1))
    # log variance ratio with the small-sample correction of Nakagawa et al. (2015)
    lnvr_c.append(np.log(np.where(sa > 0, sa, 1e-6) / np.where(sk > 0, sk, 1e-6))
                  + 1 / (2 * (na - 1)) - 1 / (2 * (nk - 1)))
    # excess tail: share of AD donors beyond the control 90th percentile (in the
    # direction of the gene's change) minus the share expected if every AD donor
    # were the control distribution moved by the same mean shift -- positive when
    # the effect is carried by a subset of donors rather than by all of them
    sgn = np.sign(delta); sgn[sgn == 0] = 1
    a_s, k_s, dl = a * sgn, k * sgn, delta * sgn
    q90 = np.quantile(k_s, 0.90, axis=0)
    obs = (a_s > q90).mean(0)
    exp = ((k_s + dl) > q90).mean(0)
    tail_c.append(obs - exp)
    w_c.append(na * nk / (na + nk))
w_c = np.array(w_c)[:, None]
G = pd.DataFrame({
    "gene": GENES,
    "d": (np.array(d_c) * w_c).sum(0) / w_c.sum(),
    "lnVR": (np.array(lnvr_c) * w_c).sum(0) / w_c.sum(),
    "excess_tail": (np.array(tail_c) * w_c).sum(0) / w_c.sum(),
}).set_index("gene")
G["abs_d"] = G.d.abs()

# %%
# ======================================================================
# 1. Co-expression with the diagnosis effect removed
# ======================================================================
# Residuals after removing the cohort x diagnosis mean of every gene: what is
# left is donor-to-donor variation *within* a diagnostic group, so a correlation
# between two genes here is co-regulation, not a shared AD shift.
grp = np.char.add(COH.astype(str), np.where(AD, "|AD", "|C"))
R = D - D.groupby(grp).transform("mean")
R = ((R - R.mean()) / R.std(ddof=0).replace(0, 1)).values.astype(np.float32)
NDON = R.shape[0]
IDX = {g: i for i, g in enumerate(GENES)}


def coherence(genes):
    """Mean absolute pairwise residual correlation, and PC1 share of variance."""
    ii = [IDX[g] for g in genes]
    Cm = (R[:, ii].T @ R[:, ii]) / NDON
    k = len(ii)
    iu = np.triu_indices(k, 1)
    ev = np.linalg.eigvalsh(Cm)
    return float(np.abs(Cm[iu]).mean()), float(ev[-1] / k)


# global connectivity of every gene: mean |r| with 2,000 random reference genes
ref = RNG.choice(len(GENES), 2000, replace=False)
CR = (R.T @ R[:, ref]) / NDON
CR[ref, np.arange(len(ref))] = np.nan
G["connectivity"] = np.nanmean(np.abs(CR), axis=1)
del CR

# effect-matched null: every selected gene is replaced by a random unselected gene
# from the same twentieth of the genome-wide |d| distribution
BINS = pd.qcut(G.abs_d.rank(method="first"), 20, labels=False).values
G["d_bin"] = BINS
POOL = {b: np.array([g for g in GENES[BINS == b] if g not in SELECTED]) for b in range(20)}


def matched(genes, rng):
    out, used = [], set()
    for g in genes:
        cand = POOL[G.at[g, "d_bin"]]
        pick = cand[rng.integers(len(cand))]
        while pick in used:
            pick = cand[rng.integers(len(cand))]
        used.add(pick); out.append(pick)
    return out


coh_rows, coh_null_rows = [], []
for name, genes in GROUPS.items():
    obs_r, obs_pc = coherence(genes)
    null = np.array([coherence(matched(genes, RNG)) for _ in range(N_NULL)])
    unm = np.array([coherence(list(RNG.choice(GENES, len(genes), replace=False))) for _ in range(N_NULL)])
    coh_rows.append({"group": name, "n_genes": len(genes),
                     "coherence": obs_r, "pc1_share": obs_pc,
                     "null_matched_mean": null[:, 0].mean(), "null_matched_hi": np.quantile(null[:, 0], 0.975),
                     "fold_vs_matched": obs_r / null[:, 0].mean(),
                     "p_vs_matched": (1 + np.sum(null[:, 0] >= obs_r)) / (1 + N_NULL),
                     "pc1_null_matched_mean": null[:, 1].mean(),
                     "p_pc1_vs_matched": (1 + np.sum(null[:, 1] >= obs_pc)) / (1 + N_NULL),
                     "null_random_mean": unm[:, 0].mean(),
                     "p_vs_random": (1 + np.sum(unm[:, 0] >= obs_r)) / (1 + N_NULL)})
    coh_null_rows += [{"group": name, "null": "matched", "coherence": v} for v in null[:, 0]]
    print(f"{name:12s} coherence {obs_r:.3f} (matched null {null[:, 0].mean():.3f}, "
          f"x{obs_r / null[:, 0].mean():.2f}, p {coh_rows[-1]['p_vs_matched']:.2g}); PC1 {obs_pc:.2f}")

# the two model-specific lists against each other: permute membership of the union
a_, b_ = GROUPS["PLS-DA only"], GROUPS["CVAE only"]
diff_obs = coherence(a_)[0] - coherence(b_)[0]
union = np.array(a_ + b_)
perm = []
for _ in range(N_NULL):
    s = RNG.permutation(union)
    perm.append(coherence(list(s[:len(a_)]))[0] - coherence(list(s[len(a_):]))[0])
perm = np.array(perm)
COH_DIFF_P = (1 + np.sum(np.abs(perm) >= abs(diff_obs))) / (1 + N_NULL)
print(f"PLS-DA only minus CVAE only coherence: {diff_obs:+.3f} (permutation p {COH_DIFF_P:.2g})")

# %%
# ======================================================================
# 2. Cell-type alignment (McKenzie et al. 2018 markers, as in V11ADrobustness)
# ======================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyreadr"], check=False)
import pyreadr
br = fetch("https://cran.r-project.org/src/contrib/BRETIGEA_1.0.3.tar.gz", "BRETIGEA_1.0.3.tar.gz")
with tarfile.open(br) as tf:
    tf.extract("BRETIGEA/data/markers_df_human_brain.rda", TMP)
mk = list(pyreadr.read_r(f"{TMP}/BRETIGEA/data/markers_df_human_brain.rda").values())[0]
mk.columns = [c.lower() for c in mk.columns]
gcol = [c for c in mk.columns if "marker" in c or "gene" in c][0]
ccol = [c for c in mk.columns if "cell" in c][0]
CELL_NAMES = {"ast": "astrocytes", "end": "endothelial", "mic": "microglia",
              "neu": "neurons", "oli": "oligodendrocytes", "opc": "OPCs"}
N_MARKERS = 50
MARKERS, MARKER_RANK = {}, {}
for ct, sub in mk.groupby(ccol, sort=False):
    name = CELL_NAMES.get(ct, ct)
    genes = [norm_symbol(g) or g for g in sub[gcol].astype(str)]
    for r_, g in enumerate(genes, 1):
        if g not in MARKER_RANK or r_ < MARKER_RANK[g][1]:
            MARKER_RANK[g] = (name, r_)
    # a selected gene never helps define the cell score it is compared with
    MARKERS[name] = [g for g in genes if g not in SELECTED and g in IDX][:N_MARKERS]
CELLS = list(MARKERS)
print({k: len(v) for k, v in MARKERS.items()}, "markers per cell type (selected genes excluded)")

S = pd.DataFrame(0.0, index=D.index, columns=CELLS)
for c in COHORTS:
    m = COH == c
    for ct in CELLS:
        Z = D.loc[m, MARKERS[ct]].values
        Z = Z - Z.mean(0)
        U, s, _ = np.linalg.svd(Z, full_matrices=False)
        pc = U[:, 0] * s[0]
        if np.corrcoef(pc, Z.mean(1))[0, 1] < 0:
            pc = -pc
        S.loc[m, ct] = (pc - pc.mean()) / pc.std()
Sv = S.values
cell_d = {ct: float(np.mean([(Sv[(COH == c) & AD, j].mean() - Sv[(COH == c) & ~AD, j].mean())
                             for c in COHORTS])) for j, ct in enumerate(CELLS)}
print("mean AD-minus-control cell score by type:", {k: round(v, 2) for k, v in cell_d.items()})

# marginal correlation of every gene with every cell score (both centred within cohort)
Dz = (Dv - Dv.mean(0)) / Dv.std(0)
Sz = (Sv - Sv.mean(0)) / Sv.std(0)
RC = (Dz.T @ Sz) / len(Dz)                                  # genes x cells
for j, ct in enumerate(CELLS):
    G[f"r_{ct}"] = RC[:, j]
# a gene is assigned to the cell type whose score it follows most closely in the
# same direction (r >= 0.30); a negative correlation only says that the gene is
# expressed elsewhere -- GFAP, for example, falls as the neuronal score rises
best = RC.argmax(1)
G["cell_r"] = RC[np.arange(len(GENES)), best]
G["cell_type"] = np.array(CELLS)[best]
G["cell_assigned"] = np.where(G.cell_r >= 0.30, G.cell_type, "unassigned")

# share of each gene's donor-level variance explained by the six cell scores
Coh = pd.get_dummies(COH).values.astype(float)
Xs = np.column_stack([Coh, Sv])
beta, *_ = np.linalg.lstsq(Xs, Dv, rcond=None)
res = Dv - Xs @ beta
G["R2_cells"] = 1 - (res**2).sum(0) / ((Dv - Coh @ np.linalg.lstsq(Coh, Dv, rcond=None)[0])**2).sum(0)

# AD effect before and after adjusting for the six cell scores (donor-level OLS
# with cohort fixed effects). Each gene is then placed in one of four classes:
#   direct         significant before and after adjustment, same sign
#   compositional  significant before, not after: the shift follows the cell mix
#   masked         significant only after adjustment (or the sign reverses): a
#                  per-cell change hidden in bulk tissue by the change in cell mix
#   neither        no significant AD effect in either model
def ols_t(Xd, Y):
    XtXi = np.linalg.inv(Xd.T @ Xd)
    B = XtXi @ Xd.T @ Y
    resid = Y - Xd @ B
    dof = Xd.shape[0] - Xd.shape[1]
    se = np.sqrt((resid**2).sum(0) / dof * XtXi[-1, -1])
    t = B[-1] / se
    return B[-1], t, 2 * stats.t.sf(np.abs(t), dof)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1)
    return out


X0 = np.column_stack([Coh, AD.astype(float)])
X1 = np.column_stack([Coh, Sv, AD.astype(float)])
b0, t0, p0 = ols_t(X0, Dv)
b1, t1, p1 = ols_t(X1, Dv)
q0, q1 = bh(p0), bh(p1)
G["ad_effect_raw"], G["ad_effect_adj"] = b0, b1
G["t_raw"], G["t_adj"], G["q_raw"], G["q_adj"] = t0, t1, q0, q1
G["abs_t_raw"], G["abs_t_adj"] = np.abs(t0), np.abs(t1)
G["delta_abs_t"] = np.abs(t1) - np.abs(t0)
G["effect_retained"] = np.where(np.abs(b0) > 1e-6, b1 / b0, np.nan)
sig0, sig1, same = q0 < 0.05, q1 < 0.05, np.sign(b0) == np.sign(b1)
G["effect_class"] = np.select([sig0 & sig1 & same, sig0 & ~sig1, sig1 & (~sig0 | ~same)],
                              ["direct", "compositional", "masked"], "neither")
G["marker_of"] = [MARKER_RANK.get(g, ("", np.nan))[0] for g in GENES]
G["marker_rank"] = [MARKER_RANK.get(g, ("", np.nan))[1] for g in GENES]

# %%
# ======================================================================
# Group summaries against effect-matched random genes
# ======================================================================
METRICS = ["abs_d", "connectivity", "R2_cells", "effect_retained", "abs_t_raw", "abs_t_adj",
           "delta_abs_t", "lnVR", "excess_tail"]
G["group"] = "unselected"
for name, genes in GROUPS.items():
    G.loc[genes, "group"] = name
G["direction"] = [DIRECTION.get(g, "") for g in G.index]
G["vip"] = pls.mean_score.reindex(G.index).values
G["ig"] = cv.mean_score.reindex(G.index).values

GI = {g: i for i, g in enumerate(G.index)}
ARR = {m: G[m].values.astype(float) for m in METRICS}
CLASSES = ["direct", "compositional", "masked", "neither"]
CLS = G.effect_class.values
summ, cls_rows = [], []
for name, genes in GROUPS.items():
    gi = np.array([GI[g] for g in genes])
    nulls = {m: [] for m in METRICS}
    ncls = {c: [] for c in CLASSES}
    for _ in range(N_NULL):
        mi = np.array([GI[g] for g in matched(genes, RNG)])
        for m in METRICS:
            nulls[m].append(np.nanmedian(ARR[m][mi]))
        for c in CLASSES:
            ncls[c].append(np.mean(CLS[mi] == c))
    for m in METRICS:
        v = np.nanmedian(ARR[m][gi])
        nv = np.array(nulls[m])
        summ.append({"group": name, "metric": m, "median": v,
                     "null_matched_median": float(np.median(nv)),
                     "null_lo": float(np.quantile(nv, 0.025)), "null_hi": float(np.quantile(nv, 0.975)),
                     "p_vs_matched": float(min(1.0, 2 * min((1 + np.sum(nv >= v)) / (1 + N_NULL),
                                                            (1 + np.sum(nv <= v)) / (1 + N_NULL))))})
    for c in CLASSES:
        v = float(np.mean(CLS[gi] == c)); nv = np.array(ncls[c])
        cls_rows.append({"group": name, "effect_class": c, "n": int(np.sum(CLS[gi] == c)),
                         "n_genes": len(gi), "fraction": v,
                         "null_matched_fraction": float(nv.mean()),
                         "null_lo": float(np.quantile(nv, 0.025)), "null_hi": float(np.quantile(nv, 0.975)),
                         "p_vs_matched": float(min(1.0, 2 * min((1 + np.sum(nv >= v)) / (1 + N_NULL),
                                                                (1 + np.sum(nv <= v)) / (1 + N_NULL))))})
for c in CLASSES:
    cls_rows.append({"group": "all genes", "effect_class": c, "n": int(np.sum(CLS == c)),
                     "n_genes": len(CLS), "fraction": float(np.mean(CLS == c))})
CLS_TAB = pd.DataFrame(cls_rows)
_a = G.loc[GROUPS["PLS-DA only"], "effect_class"].eq("direct")
_b = G.loc[GROUPS["CVAE only"], "effect_class"].eq("direct")
FISHER_DIRECT = stats.fisher_exact([[_a.sum(), (~_a).sum()], [_b.sum(), (~_b).sum()]])
print(CLS_TAB.round(3).to_string(index=False))
print(f"direct effects, PLS-DA only {_a.mean():.2f} vs CVAE only {_b.mean():.2f}: "
      f"Fisher OR {FISHER_DIRECT[0]:.1f}, p {FISHER_DIRECT[1]:.2g}")
SUMM = pd.DataFrame(summ)
# PLS-DA only versus CVAE only, gene by gene
for m in METRICS:
    a = G.loc[GROUPS["PLS-DA only"], m].dropna(); b = G.loc[GROUPS["CVAE only"], m].dropna()
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    SUMM.loc[len(SUMM)] = {"group": "PLS-DA only vs CVAE only", "metric": m,
                           "median": float(a.median() - b.median()), "null_matched_median": np.nan,
                           "null_lo": np.nan, "null_hi": np.nan, "p_vs_matched": float(u.pvalue)}
print(SUMM.pivot(index="metric", columns="group", values="median").round(3).to_string())
print(SUMM.pivot(index="metric", columns="group", values="p_vs_matched").round(4).to_string())

# cell-type make-up of each list, from the published marker lists themselves
# (a gene counts as a marker if it is among the 200 most specific genes of a cell
# type), so that the annotation does not depend on this data set. Correlation with
# bulk cell scores is kept in the per-gene table but is not used here: in bulk
# tissue all glial scores rise together in AD and blur the assignment.
TOP_MARKER = 200
G["marker_top"] = np.where(G.marker_rank <= TOP_MARKER, G.marker_of, "")
CT_ORDER = ["neurons", "astrocytes", "microglia", "endothelial", "oligodendrocytes", "OPCs"]
mrows = []
IS_MK = (G.marker_top != "").values
for name, genes in list(GROUPS.items()) + [("all genes", list(G.index))]:
    sub = G.loc[genes]
    row = {"group": name, "n_genes": len(sub), "n_markers": int((sub.marker_top != "").sum())}
    row.update({ct: int((sub.marker_top == ct).sum()) for ct in CT_ORDER})
    row["markers"] = ", ".join(f"{g} ({sub.at[g, 'marker_top']} #{int(sub.at[g, 'marker_rank'])})"
                               for g in sub.index[sub.marker_top != ""])
    if name != "all genes":
        nv = np.array([IS_MK[[GI[g] for g in matched(genes, RNG)]].mean() for _ in range(N_NULL)])
        row.update({"null_matched_fraction": float(nv.mean()),
                    "p_vs_matched": float(min(1.0, 2 * min((1 + np.sum(nv >= row["n_markers"] / len(sub))) / (1 + N_NULL),
                                                           (1 + np.sum(nv <= row["n_markers"] / len(sub))) / (1 + N_NULL))))})
    mrows.append(row)
comp = pd.DataFrame(mrows)
_ma = G.loc[GROUPS["PLS-DA only"], "marker_top"].ne(""); _mb = G.loc[GROUPS["CVAE only"], "marker_top"].ne("")
FISHER_MARKER = stats.fisher_exact([[_ma.sum(), (~_ma).sum()], [_mb.sum(), (~_mb).sum()]])
print(comp.drop(columns="markers").to_string(index=False))
for _, r in comp.iterrows():
    if r.group != "all genes":
        print(f"  {r.group}: {r.markers}")
print(f"markers, PLS-DA only vs CVAE only: Fisher OR {FISHER_MARKER[0]:.2f}, p {FISHER_MARKER[1]:.2g}")

# %%
# ======================================================================
# Genome-wide: what does each importance score track?
# ======================================================================
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


att = []
ok = G[["vip", "ig"] + METRICS].dropna()
for score in ("vip", "ig"):
    for m in METRICS:
        rho, p = stats.spearmanr(ok[score], ok[m])
        if m in ("abs_d", "abs_t_raw"):
            prho, pp = np.nan, np.nan
        else:
            prho, pp = partial_spearman(ok[score].values, ok[m].values, ok["abs_d"].values)
        att.append({"score": score, "property": m, "spearman": rho, "p": p,
                    "partial_spearman_given_abs_d": prho, "p_partial": pp, "n_genes": len(ok)})
ATT = pd.DataFrame(att)
print(ATT.round(3).to_string(index=False))

# %%
# ======================================================================
# Write everything
# ======================================================================
G.reset_index().to_csv(f"{OUT}/gene_sides_per_gene.csv", index=False)
pd.DataFrame(coh_rows).to_csv(f"{OUT}/gene_sides_coherence.csv", index=False)
pd.DataFrame(coh_null_rows).to_csv(f"{OUT}/gene_sides_coherence_null.csv", index=False)
pd.DataFrame([{"comparison": "PLS-DA only minus CVAE only", "coherence_difference": diff_obs,
               "permutation_p": COH_DIFF_P}]).to_csv(f"{OUT}/gene_sides_coherence_diff.csv", index=False)
SUMM.to_csv(f"{OUT}/gene_sides_summary.csv", index=False)
comp.to_csv(f"{OUT}/gene_sides_celltypes.csv", index=False)
pd.DataFrame([{"test": "cell-type markers, PLS-DA only vs CVAE only", "odds_ratio": FISHER_MARKER[0],
               "p": FISHER_MARKER[1]}]).to_csv(f"{OUT}/gene_sides_fisher_markers.csv", index=False)
ATT.to_csv(f"{OUT}/gene_sides_attribution.csv", index=False)
CLS_TAB.to_csv(f"{OUT}/gene_sides_effect_classes.csv", index=False)
pd.DataFrame([{"test": "direct effects, PLS-DA only vs CVAE only", "odds_ratio": FISHER_DIRECT[0],
               "p": FISHER_DIRECT[1]}]).to_csv(f"{OUT}/gene_sides_fisher.csv", index=False)
pd.DataFrame([{"cell": k, "mean_AD_minus_control": v, "markers_used": len(MARKERS[k])}
              for k, v in cell_d.items()]).to_csv(f"{OUT}/gene_sides_cell_scores.csv", index=False)
print("written:", sorted(f for f in os.listdir(OUT) if f.startswith("gene_sides")))

# %%
# ======================================================================
# Sensitivity: is either list driven by sex?
# ======================================================================
# AD and control donors differ in sex composition in some cohorts (most clearly
# GSE33000), and the main analysis does not adjust for sex. If sex drove the
# selection, the selected genes would be sex-associated among controls and the
# canonical sex-chromosome genes would rank high.
from sklearn.metrics import roc_auc_score
MALE = (DM.sex == "male").values
sex_rows = []
for c in COHORTS:
    m = COH == c
    sex_rows.append({"cohort": c, "AD_female_frac": float(1 - MALE[m & AD].mean()),
                     "control_female_frac": float(1 - MALE[m & ~AD].mean()),
                     "auc_sex_alone": float(roc_auc_score(AD[m], (~MALE[m]).astype(float)))})
sex_rows.append({"cohort": "merged", "AD_female_frac": float(1 - MALE[AD].mean()),
                 "control_female_frac": float(1 - MALE[~AD].mean()), "auc_sex_alone": np.nan})
SEXCOH = pd.DataFrame(sex_rows)
print(SEXCOH.round(3).to_string(index=False))

ctl = ~AD
Xs_ = np.column_stack([Coh[ctl], MALE[ctl].astype(float)])
_, ts_, ps_ = ols_t(Xs_, Dv[ctl])
G["q_sex_controls"] = bh(ps_)
gi_chr = pd.read_csv(f"{TMP}/Homo_sapiens.gene_info.gz", sep="\t", dtype=str,
                     usecols=["Symbol", "chromosome"]).drop_duplicates("Symbol").set_index("Symbol").chromosome
G["chromosome"] = [gi_chr.get(g, "") for g in G.index]
SEXCHR = G.chromosome.isin(["X", "Y", "X|Y"])
sexg_rows = []
for name, genes in list(GROUPS.items()) + [("all genes", list(G.index))]:
    sub = G.loc[genes]
    sexg_rows.append({"group": name, "n_genes": len(sub),
                      "n_sex_associated_controls": int((sub.q_sex_controls < 0.05).sum()),
                      "frac_sex_associated_controls": float((sub.q_sex_controls < 0.05).mean()),
                      "n_sex_chromosome": int(SEXCHR[genes].sum()),
                      "sex_associated_genes": ", ".join(sub.index[sub.q_sex_controls < 0.05]) if name != "all genes" else "",
                      "sex_chromosome_genes": ", ".join(sub.index[SEXCHR[genes].values]) if name != "all genes" else ""})
SEXG = pd.DataFrame(sexg_rows)
print(SEXG.drop(columns=[]).to_string(index=False))
CANON = [g for g in ["XIST", "RPS4Y1", "DDX3Y", "UTY", "EIF1AY", "KDM5D", "USP9Y", "ZFY"] if g in G.index]
vip_rank = G.vip.rank(ascending=False); ig_rank = G.ig.rank(ascending=False)
SEXCANON = pd.DataFrame({"gene": CANON, "q_sex_controls": G.loc[CANON, "q_sex_controls"].values,
                         "vip_rank": vip_rank[CANON].values.astype(int), "ig_rank": ig_rank[CANON].values.astype(int),
                         "n_genes": len(G)})
print(SEXCANON.to_string(index=False))
SEXCOH.to_csv(f"{OUT}/gene_sides_sex_cohorts.csv", index=False)
SEXG.to_csv(f"{OUT}/gene_sides_sex_genes.csv", index=False)
SEXCANON.to_csv(f"{OUT}/gene_sides_sex_canonical.csv", index=False)

# %%
# ======================================================================
# Where else are the selected genes expressed? (Human Protein Atlas)
# ======================================================================
# A gene that is barely expressed in brain gives a weak array signal that may
# reflect background or cross-hybridisation rather than the gene itself. HPA's
# consensus tissue-specificity call flags genes whose expression is concentrated in
# a non-brain tissue.
import time, json as _json
BRAIN_WORDS = ("brain", "cerebral", "cerebell", "hippocamp", "amygdala", "basal ganglia",
               "midbrain", "spinal cord", "hypothalamus", "pituitary", "retina", "choroid plexus")
hpa_rows = []
for g in sorted(SELECTED):
    url = ("https://www.proteinatlas.org/api/search_download.php?search=" + g +
           "&format=json&columns=g,rnats,rnatsm,rnabrd&compress=no")
    rec = {}
    for attempt in range(4):
        try:
            rec = next((x for x in requests.get(url, headers=HDR, timeout=60).json() if x.get("Gene") == g), {})
            break
        except Exception:                            # noqa: BLE001
            time.sleep(2)
    tissues = rec.get("RNA tissue specific nTPM") or {}
    spec = rec.get("RNA tissue specificity", "")
    brain_dist = rec.get("RNA brain regional distribution", "")
    nonbrain = (spec in ("Tissue enriched", "Group enriched") and
                not any(w in t.lower() for t in tissues for w in BRAIN_WORDS)) or brain_dist == "Not detected"
    hpa_rows.append({"gene": g, "group": G.at[g, "group"], "rna_tissue_specificity": spec,
                     "specific_tissues": "; ".join(f"{t} {v}" for t, v in tissues.items()),
                     "brain_distribution": brain_dist, "non_brain_enriched": bool(nonbrain)})
    time.sleep(0.3)
HPA = pd.DataFrame(hpa_rows)
HPA.to_csv(f"{OUT}/gene_sides_hpa_tissue.csv", index=False)
print(HPA[HPA.non_brain_enriched].to_string(index=False))
print(HPA.groupby("group").non_brain_enriched.agg(["sum", "count"]))
print("HPA records found:", int((HPA.rna_tissue_specificity != "").sum()), "of", len(HPA))
