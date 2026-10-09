"""Rebuild the V14ADfigures Kaggle notebook from paper3/scripts (run from paper3/kaggle)."""
import json, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "scripts"
base = json.load(open(HERE.parent / "v14adfigures.ipynb"))
intro, pip, locate = base["cells"][0], base["cells"][1], base["cells"][2]
FILES = ["merged_figures.py", "validation_figures.py", "age_figures.py", "core_validation_figure.py",
         "gene_sides_figure.py"]
RUNS = [("Main figures (merged cohorts)", "merged_figures.py"),
        ("External validation at a glance (core figure)", "core_validation_figure.py"),
        ("Supplementary external-validation figures", "validation_figures.py"),
        ("Robustness to age (V15ADage)", "age_figures.py"),
        ("Why the two models see different sides of AD (V16ADgenesides)", "gene_sides_figure.py")]
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
cells = [intro, pip, locate]
for n in FILES:
    s = (SRC / n).read_text(); ast.parse(s)
    v = "SRC_" + n.split(".")[0].upper()
    cells += [md(f"## `scripts/{n}`\n"),
              code(f"{v} = {json.dumps(s)}\nopen('/kaggle/working/scripts/{n}', 'w').write({v})\nprint('wrote {n}', len({v}), 'chars')\n")]
cells.append(code("import runpy\n"))
for title, n in RUNS:
    cells += [md(f"## {title}\n"), code(f"runpy.run_path('/kaggle/working/scripts/{n}', run_name='__main__')\n")]
cells += [md("## Output manifest\n"), base["cells"][-1]]
nb = {"cells": cells, "metadata": base["metadata"], "nbformat": 4, "nbformat_minor": 4}
out = HERE / "build"; out.mkdir(exist_ok=True)
json.dump(nb, open(out / "v14adfigures.ipynb", "w"), indent=1)
json.dump({"id": "alisaremi/v14adfigures", "title": "V14ADfigures", "code_file": "v14adfigures.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
           "dataset_sources": [], "competition_sources": [], "model_sources": [],
           "kernel_sources": ["alisaremi/v7admerged", "alisaremi/ad-merged-geo-cohorts", "alisaremi/v9adexternal",
                              "alisaremi/v10adgenevalidation", "alisaremi/v11adrobustness", "alisaremi/v15adage",
                              "alisaremi/v16adgenesides"]},
          open(out / "kernel-metadata.json", "w"), indent=2)
print("built", len(cells), "cells")
