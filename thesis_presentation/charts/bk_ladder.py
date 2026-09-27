"""Backup chart: nested grounding ladder, cumulative share of disclosed AI
activities clearing each successive concreteness requirement."""
from matplotlib.colors import LinearSegmentedColormap

from style import C, fig, save, results, read

row = read(results("washing", "ladder_substance.parquet")).iloc[0]
n_total = int(row["n_total"])
counts = [int(row[f"c{i}"]) for i in range(1, 6)]
pcts = [row[f"pct_c{i}"] for i in range(1, 6)]

labels = [
    "AI activity claim",
    "+ specific business function",
    "+ deployed / scaled stage",
    "+ named product or process",
    "+ quantified outcome",
]

cmap = LinearSegmentedColormap.from_list("bars", [C["mist"], C["petrol_dark"]])
colors = [cmap(t) for t in [0.02, 0.28, 0.54, 0.80, 1.0]]

f = fig(1100, 520)
ax = f.add_axes([0.30, 0.08, 0.62, 0.86])

y = list(range(len(labels)))[::-1]
bars = ax.barh(y, pcts, height=0.58, color=colors, zorder=3)

for yi, p, c, lab_i in zip(y, pcts, counts, range(len(labels))):
    emphasize = lab_i == 4
    val_txt = f"{p:.1f}%"
    ax.text(p + 2.0, yi, val_txt, va="center", ha="left", fontsize=17,
            fontweight="bold" if emphasize else "semibold",
            color=C["rust"] if emphasize else C["ink"])
    ax.text(p + 17.5, yi, f"{c:,} of {n_total:,}",
            va="center", ha="left", fontsize=12, color=C["ink_3"])

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=16, color=C["ink"])
ax.set_xlim(0, 128)
ax.set_xticks([])
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

save(f, "ladder")
