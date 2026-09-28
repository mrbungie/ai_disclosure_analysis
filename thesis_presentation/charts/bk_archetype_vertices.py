"""Backup chart: standardized vertex coordinates (z-scores) of the three
posture archetypes across the seven posture features."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from style import C, ARCH, ARCH_SHORT, fig, save, results, read

FEATURES = ["promotional_posture", "hedging_posture", "risk_orientation",
            "governance_orientation", "temporal_posture", "ai_positioning",
            "specificity"]
COL_LABELS = ["Promotional", "Hedging", "Risk", "Governance",
              "Realized", "Customer-\nfacing", "Specificity"]
ARCHETYPES = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]

vert = read(results("posture", "archetype_vertices.parquet"))
mat = vert.pivot(index="archetype", columns="feature", values="value").reindex(
    index=ARCHETYPES, columns=FEATURES)

cmap = LinearSegmentedColormap.from_list("div", [C["rust"], C["bg"], C["petrol"]])
vmax = 3.0

f = fig(880, 420)
ax = f.add_axes([0.20, 0.14, 0.78, 0.72])
ax.set_xlim(-0.5, len(FEATURES) - 0.5)
ax.set_ylim(-0.5, len(ARCHETYPES) - 0.5)
ax.invert_yaxis()
ax.axis("off")

cell = 0.90
for j, arche in enumerate(ARCHETYPES):
    for i, feat in enumerate(FEATURES):
        v = mat.loc[arche, feat]
        t = np.clip((v + vmax) / (2 * vmax), 0, 1)
        color = cmap(t)
        rect = FancyBboxPatch((i - cell / 2, j - cell / 2), cell, cell,
                               boxstyle="round,pad=0,rounding_size=0.06",
                               linewidth=0, facecolor=color, zorder=2)
        ax.add_patch(rect)
        txt_color = "white" if abs(v) > 1.7 else C["ink"]
        ax.text(i, j, f"{v:+.2f}", ha="center", va="center", fontsize=15.5,
                fontweight="semibold", color=txt_color, zorder=3)

for i, lab in enumerate(COL_LABELS):
    ax.text(i, -0.72, lab, ha="center", va="bottom", fontsize=14.5,
            color=C["ink_2"], fontweight="semibold")

for j, arche in enumerate(ARCHETYPES):
    ax.text(-0.72, j, ARCH_SHORT[arche], ha="right", va="center", fontsize=15.5,
            color=ARCH[arche], fontweight="semibold")

f.text(0.20, 0.02, "z-scores vs. pooled mean", fontsize=13.5, color=C["ink_3"],
       style="italic")

save(f, "archetype_vertices")
