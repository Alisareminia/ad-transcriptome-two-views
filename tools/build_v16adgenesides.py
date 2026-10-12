"""Build the V16ADgenesides Kaggle notebook from v16adgenesides.py.

Cells are delimited by '# %%' (code) and '# %% [markdown]' (markdown) so that the
notebook and the local script are one source. Output: build/ with the notebook and
kernel-metadata.json, and a copy at paper3/kaggle/v16adgenesides.ipynb.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v16adgenesides.py"
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)

cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
        cells.append({"cell_type": "markdown", "id": f"c{len(cells)}", "metadata": {}, "source": text})
    else:
        cells.append({"cell_type": "code", "id": f"c{len(cells)}", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text})


for line in SRC.read_text().splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(BUILD / "v16adgenesides.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": "alisaremi/v16adgenesides", "title": "V16ADgenesides",
        "code_file": "v16adgenesides.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": ["alisaremi/ad-merged-geo-cohorts", "alisaremi/v7admerged"]}
(BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
shutil.copy(BUILD / "v16adgenesides.ipynb", HERE.parent / "v16adgenesides.ipynb")
print(f"built {len(cells)} cells")
