"""Regenerate every figure of paper3 from data/ (no network, no refitting)."""
import os, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ["PAPER_DATA"] = str(ROOT / "data")
os.environ["PAPER_FIGS"] = str(ROOT / "figures")
os.environ["PAPER_INLINE"] = ""
os.environ["V12_INPUT"] = str(ROOT / "data" / "validation")
os.environ["MPLBACKEND"] = "Agg"
runpy.run_path(str(ROOT / "scripts" / "merged_figures.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "validation_figures.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "age_figures.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "core_validation_figure.py"), run_name="__main__")
runpy.run_path(str(ROOT / "scripts" / "gene_sides_figure.py"), run_name="__main__")
