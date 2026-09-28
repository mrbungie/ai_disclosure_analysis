"""Backup chart: sector representation by disclosure archetype, observed vs.
expected share, log2 diverging scale."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from style import C, ARCH, ARCH_SHORT, fig, save, results, read

ARCHS = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers", "No AI"]

df = read(results("posture", "archetype_sector.parquet"))
rel = df.pivot(index="sector", columns="archetype", values="relative_representation")[ARCHS]
n_firms = df.pivot(index="sector", columns="archetype", values="n_firms")[ARCHS]
rel = rel.sort_values(by="Vocal Substantives", ascending=False)
n_firms = n_firms.loc[rel.index]

cmap = LinearSegmentedColormap.from_list("div", [C["rust"], C["bg"], C["petrol"]])
vmax = 1.5
log_ratio = np.log2(np.clip(rel.values, 0.2, 5.0))

n_rows, n_cols = rel.shape

f = fig(660, 450)
ax = f.add_axes([0.30, 0.02, 0.69, 0.88])
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(-0.5, n_rows - 0.5)
ax.invert_yaxis()
ax.axis("off")

cell = 0.90
for i in range(n_rows):
    for j in range(n_cols):
        t = np.clip((log_ratio[i, j] + vmax) / (2 * vmax), 0, 1)
        color = cmap(t)
        rect = FancyBboxPatch((j - cell / 2, i - cell / 2), cell, cell,
                               boxstyle="round,pad=0,rounding_size=0.06",
                               linewidth=0, facecolor=color, zorder=2)
        ax.add_patch(rect)
        val = rel.values[i, j]
        n_f = int(n_firms.values[i, j])
        small_n = n_f < 5
        txt_color = "white" if abs(log_ratio[i, j]) > 0.75 else C["ink"]
        style = "italic" if small_n else "normal"
        alpha = 0.75 if small_n else 1.0
        ax.text(j, i, f"{val:.1f}×", ha="center", va="center",
                fontsize=15, fontweight="semibold", color=txt_color,
                style=style, alpha=alpha, zorder=3)

for j, a in enumerate(ARCHS):
    ax.text(j, -0.62, ARCH_SHORT[a].replace("Vocal Substantives", "Vocal"), ha="center", va="bottom", fontsize=14.5,
            fontweight="semibold", color=C["ink_3"] if a == "No AI" else ARCH[a])

for i, sec in enumerate(rel.index):
    ax.text(-0.58, i, sec, ha="right", va="center", fontsize=14, color=C["ink"])



save(f, "sector_heatmap")
