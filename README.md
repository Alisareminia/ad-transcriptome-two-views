# Linear and variational latent models of the Alzheimer's disease brain transcriptome

Code, notebooks and result files for the study *Linear and Variational Latent Models of the
Alzheimer's Disease Brain Transcriptome Select Overlapping but Distinct Genes* (A. Sareminia,
manuscript in preparation).

Five post-mortem Alzheimer's disease (AD) cohorts from the Gene Expression Omnibus are merged
(1,596 arrays, 817 donors, 11,214 genes). Partial least squares discriminant analysis (PLS-DA) and
a semi-supervised β-conditional variational autoencoder (β-CVAE) are fitted under the same
eight-dimensional latent budget on 20 donor-grouped splits with 10 label-permuted null splits.
Genes are attributed by VIP and Integrated Gradients, filtered by permutation FDR and stability
selection, annotated with g:Profiler, characterised at donor level, and validated in cohorts that
were not used to select them, including an RNA-seq cohort, with age and cell-composition
sensitivity analyses.

Every number, table and figure of the paper is generated from the files in `data/` by the scripts
in `scripts/`; nothing is transcribed by hand.

## Repository layout

```
notebooks/   the eight analysis notebooks, in the order they were run, each with the
             kernel-metadata.json that lists its inputs (exact code of the version that
             produced the results in data/)
inputs/      the preprocessed GSE48350 matrix and labels read by notebook 01
data/        every result file the notebooks wrote (V7ADmerged at the top level, the
             other notebooks under data/validation/<notebook>/), with their run logs
scripts/     figure and number generation from data/ (no network, no refitting)
figures/     the 19 figures of the paper, as PDF and 400-dpi PNG
tools/       builders that assemble notebooks 07 and 08 from the scripts
```

## Pipeline

| # | Notebook | Reads | Writes (in this repository) | Paper |
|---|----------|-------|-----------------------------|-------|
| 01 | `ad-merged-geo-cohorts` | GEO (downloaded), `inputs/gse48350/` | `data/validation/ad-merged-geo-cohorts/` (+ `merged_expression.csv.gz`, regenerated, not stored) | Methods 2.1, Table 1 |
| 02 | `V7ADmerged` | 01 | `data/` | Results 3.1–3.7, Section 4 (enrichment) |
| 03 | `V9ADexternal` | 01 | `data/validation/v9adexternal/` | Section 5 (held-out cohort GSE132903) |
| 04 | `V10ADgenevalidation` | 01, 03 | `data/validation/v10adgenevalidation/` | Section 5 (independent cohorts, meta-analysis, severity) |
| 05 | `V11ADrobustness` | 01, 03 | `data/validation/v11adrobustness/` | Section 5 (cell composition, RNA-seq) |
| 06 | `V15ADage` | 01, 02 | `data/validation/v15adage/` | Methods 2.10, Section 6 (robustness to age) |
| 07 | `V16ADgenesides` | 01, 02 | `data/validation/v16adgenesides/` | Methods 2.11, Results 3.8, Discussion 7.2 |
| 08 | `V14ADfigures` | 01–07 | all figures | every figure |

The notebooks ran on Kaggle (CPU, internet on) and read their inputs from `/kaggle/input/**` and
write to `/kaggle/working`. They download the remaining GEO cohorts, NCBI Gene, the BRETIGEA
marker lists and the g:Profiler annotations themselves.

## Reproducing the results

### 1. Numbers and figures from the stored results (minutes, no network)

```bash
pip install -r requirements.txt
python scripts/make_macros.py            # main-analysis numbers  -> generated/values.tex
python scripts/make_enrichment.py        # enrichment tables      -> generated/enrichment_*.tex
python scripts/make_validation_macros.py # validation, age, gene-sides numbers -> generated/
python scripts/make_figures.py           # all 19 figures         -> figures/
```

`generated/*.tex` holds every number quoted in the paper as a LaTeX macro (for example
`\vAucPlsMean`), so each value in the text can be traced to the file and line that produced it.

### 2. The full analysis from the raw data

Run the notebooks in the order 01 → 08, giving each the outputs of the notebooks it reads
(see `kernel-metadata.json` in each folder). Notebook 01 additionally needs the two files in
`inputs/gse48350/`; decompress the matrix first:

```bash
gunzip -k inputs/gse48350/GSE48350_subjects_by_genes.csv.gz
```

| File | SHA-256 |
|------|---------|
| `GSE48350_subjects_by_genes.csv` (after `gunzip`) | `f181920b801fe1899dfac7020256acc9e002d2bcc0a8426951d542c2cd1af7a3` |
| `GSE48350_clean_labels.csv` | `276c998d072add31f64baab1ab7f7e42d8d52338476e8ed27bcb01b0f5ef3c7f` |

On Kaggle, upload these files as a dataset and attach it to notebook 01, and attach each
notebook's output to the notebooks that read it. To run elsewhere, place the inputs under any
folder and point the paths at it: every notebook locates its inputs with
`glob("/kaggle/input/**/<file name>")`.

Every split, label permutation and network initialisation is seeded (the 20 real-label splits
use seeds 0–19), models are trained on CPU, and the output of notebook 07 was confirmed to match
an independent local run to within 10⁻⁷.

## Software

Analysis notebooks (Kaggle image): Python 3.13, PyTorch 2.11 (CPU), NumPy 2.1.3, pandas 2.3.3,
scikit-learn 1.6.1, SciPy 1.16.3, Captum and umap-learn (installed in the notebooks), pyreadr
(notebooks 05 and 07). The exact environment of each run is recorded in `data/env_report.csv`
and `data/validation/*/env_report.csv`.

Figure and number scripts: see `requirements.txt`.

## Figures

| Paper | File |
|-------|------|
| Figure 1 | `fig1_design` |
| Figure 2 | `fig2_classification` |
| Figure 3 | `fig3_latent_geometry` |
| Figure 4 | `fig4_latent_separation` |
| Figure 5 | `fig6_reconstruction` |
| Figure 6 | `fig7_uncertainty` |
| Figure 7 | `fig8_gene_discovery` |
| Figure 8 | `fig9_expression_profile` |
| Figure 9 | `fig10_latent_gene_map` |
| Figure 10 | `figV16_gene_sides` |
| Figure 11 | `fig11_enrichment` |
| Figure 12 | `figV_core_validation` |
| Figure 13 | `figV7_age_robustness` |
| Figures S1–S6 | `figV1_design_cohorts`, `figV2_external_validation`, `figV3_signature_replication`, `figV4_cell_composition`, `figV5_dose_response`, `figV6_evidence_scorecard` |

## Data sources

All expression data are public in the Gene Expression Omnibus: GSE48350, GSE33000, GSE132903,
GSE118553 and GSE84422 (analysis cohorts); GSE15222, GSE29378, GSE122063, GSE36980 and the RNA-seq
cohort GSE125583 (validation); GSE106241 and GSE1297 (association with neuropathology). The
functional themes of the selected genes (`data/gene_themes.csv`) are assigned from NCBI Gene
descriptions, with the Gene ID of each gene.
