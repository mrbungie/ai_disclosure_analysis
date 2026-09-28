"""Shared chart style and data access for the defense deck.

Every chart is written as SVG with text converted to paths, so it renders
identically on any projector machine. Figures are designed at 1 inch = 72
display px: a chart shown 1100 px wide is built with figsize width 1100/72.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = HERE.parent
REPO = DECK.parent
OUT = DECK / "figures"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "scripts" / "common"))
import layers as L  # noqa: E402

for _f in (DECK / "assets" / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(_f))

# Palette: the thesis MIB base palette (ink, terracotta, eucalyptus, violet-grey,
# ochre, deep) as main accents on white; the thesis chart colours (sage, greys) as secondary.
C = {
    "bg": "#FFFFFF",
    "ink": "#1C1F22",
    "ink_2": "#454E54",
    "ink_3": "#73828C",
    "grid": "#E6E8EB",
    "rule": "#A5AFB6",
    "petrol": "#446582",
    "petrol_dark": "#263B4A",
    "petrol_light": "#7D97AC",
    "rust": "#C9826B",
    "rust_light": "#E2B4A4",
    "teal": "#67958A",
    "sage": "#8EA7A3",
    "ochre": "#B69A59",
    "slate": "#9385A6",
    "mist": "#D5D9DD",
}
# One colour per archetype, used identically on every slide.
ARCH = {
    "Vocal Substantives": C["petrol"],
    "Governance-Led Disclosers": C["ochre"],
    "Defensive Disclosers": C["slate"],
    "No AI": C["mist"],
}
ARCH_SHORT = {
    "Vocal Substantives": "Vocal Substantives",
    "Governance-Led Disclosers": "Governance-Led",
    "Defensive Disclosers": "Defensive",
    "No AI": "No AI",
}

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 17,
    "font.weight": "regular",
    "text.color": C["ink"],
    "axes.labelcolor": C["ink_2"],
    "axes.labelsize": 16,
    "axes.edgecolor": C["rule"],
    "axes.linewidth": 1.0,
    "axes.facecolor": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": C["grid"],
    "grid.linewidth": 1.0,
    "xtick.color": C["ink_3"],
    "ytick.color": C["ink_3"],
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "legend.frameon": False,
    "legend.fontsize": 15,
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "svg.fonttype": "path",
    "lines.solid_capstyle": "round",
})


def fig(width_px: float, height_px: float):
    """A figure sized in display pixels (72 px per inch)."""
    return plt.figure(figsize=(width_px / 72, height_px / 72))


def save(f, name: str) -> Path:
    p = OUT / f"{name}.svg"
    f.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(f)
    print(f"wrote {p.relative_to(REPO)}")
    return p


def results(topic: str, name: str) -> Path:
    return L.RESULTS / topic / name


def results_json(topic: str, name: str) -> dict:
    return json.loads(results(topic, name).read_text())


def gold(kind: str, grain: str, name: str) -> Path:
    return L.GOLD / kind / grain / f"{name}.parquet"


def read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if path.suffix == ".csv" else pd.read_parquet(path, **kw)
