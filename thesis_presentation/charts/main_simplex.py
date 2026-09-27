"""Chart 2: ternary simplex of archetype weights for the 451 disclosing firms."""
import sys
from pathlib import Path

import numpy as np

from style import C, ARCH, fig, save, REPO

sys.path.insert(0, str(REPO / "scripts" / "gold" / "posture"))
sys.path.insert(0, str(REPO / "scripts" / "common"))
import joblib  # noqa: E402
import layers as L  # noqa: E402

bundle = joblib.load(REPO / "models" / "posture_archetype_static" / "model.pkl")
model, scaler, feature_names, cluster_names = (
    bundle["model"], bundle["scaler"], bundle["feature_names"], bundle["cluster_names"],
)
df = L.read_gold("firm", ("covariates", "posture_archetype_static"))
fit_pop = df[df["archetype"] != "No AI"].copy().reset_index(drop=True)
W = model.transform(scaler.transform(fit_pop[feature_names].values))
name_to_col = {name: cid for cid, name in cluster_names.items()}

w_def = W[:, name_to_col["Defensive Disclosers"]]
w_voc = W[:, name_to_col["Vocal Substantives"]]
w_gov = W[:, name_to_col["Governance-Led Disclosers"]]

# Vertices: Defensive top, Vocal bottom-left, Governance bottom-right.
TOP = np.array([0.5, np.sqrt(3) / 2])
BL = np.array([0.0, 0.0])
BR = np.array([1.0, 0.0])

xy = np.outer(w_def, TOP) + np.outer(w_voc, BL) + np.outer(w_gov, BR)
x, y = xy[:, 0], xy[:, 1]

colors = fit_pop["archetype"].map(ARCH).values
tickers = fit_pop["ticker"].values
archetype = fit_pop["archetype"].values

f = fig(820, 800)
ax = f.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.06, 1.06)
ax.set_ylim(-0.19, 1.04)
ax.set_aspect("equal")
ax.axis("off")

# Triangle edges
tri = np.array([TOP, BL, BR, TOP])
ax.plot(tri[:, 0], tri[:, 1], color=C["rule"], lw=1.4, zorder=1)

# Faint internal guides for w>0.5 regions: line from each edge midpoint to
# the centroid-side point where that vertex weight = 0.5 (i.e. midpoints of
# the two adjacent edges, connected across the triangle).
mid_top_bl = (TOP + BL) / 2
mid_top_br = (TOP + BR) / 2
mid_bl_br = (BL + BR) / 2
# w_def = 0.5 boundary: line parallel to BL-BR through midpoints of TOP-BL and TOP-BR
ax.plot([mid_top_bl[0], mid_top_br[0]], [mid_top_bl[1], mid_top_br[1]],
        color=C["grid"], lw=1.1, linestyle=(0, (2, 3)), zorder=1)
# w_voc = 0.5 boundary: parallel to TOP-BR through midpoints of BL-TOP and BL-BR
mid_bl_top = mid_top_bl
mid_bl_br2 = mid_bl_br
ax.plot([mid_bl_top[0], mid_bl_br2[0]], [mid_bl_top[1], mid_bl_br2[1]],
        color=C["grid"], lw=1.1, linestyle=(0, (2, 3)), zorder=1)
# w_gov = 0.5 boundary: parallel to TOP-BL through midpoints of BR-TOP and BR-BL
mid_br_top = mid_top_br
mid_br_bl = mid_bl_br
ax.plot([mid_br_top[0], mid_br_bl[0]], [mid_br_top[1], mid_br_bl[1]],
        color=C["grid"], lw=1.1, linestyle=(0, (2, 3)), zorder=1)

# Points
ax.scatter(x, y, s=46, c=colors, alpha=0.75, linewidths=0.5,
           edgecolors="white", zorder=3)

# Vertex labels: name (semibold, archetype colour) over a one-line descriptor
def vertex_label(xy, name, desc, n, ha, below):
    x0, y0 = xy
    y_name, y_desc = (y0 - 0.075, y0 - 0.135) if below else (y0 + 0.125, y0 + 0.068)
    ax.text(x0, y_name, f"{name.replace(' Disclosers', '')}  ·  {n}", ha=ha, va="center",
            fontsize=21, fontweight="semibold", color=ARCH[name], zorder=5)
    ax.text(x0, y_desc, desc, ha=ha, va="center", fontsize=16, color=C["ink_3"], zorder=5)

n_def = int((archetype == "Defensive Disclosers").sum())
n_voc = int((archetype == "Vocal Substantives").sum())
n_gov = int((archetype == "Governance-Led Disclosers").sum())
ax.set_ylim(-0.19, TOP[1] + 0.17)
vertex_label(TOP, "Defensive Disclosers", "risk & caveats", n_def, "center", False)
vertex_label(BL, "Vocal Substantives", "customer-facing, specific", n_voc, "left", True)
vertex_label(BR, "Governance-Led Disclosers", "oversight & responsible use", n_gov, "right", True)

# Ticker labels for well-known firms
labels = {
    "MSFT": (0, 10, "center"), "NVDA": (0, -12, "center"), "GOOGL": (10, 4, "left"),
    "AMZN": (10, -4, "left"), "AAPL": (-10, 4, "right"), "C": (-10, -6, "right"),
    "JPM": (10, 8, "left"), "GS": (10, -8, "left"),
}
tmap = {t: (xi, yi) for t, xi, yi in zip(tickers, x, y)}
for t, (dx, dy, ha) in labels.items():
    if t not in tmap:
        continue
    px, py = tmap[t]
    ax.annotate(t, (px, py), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=16, fontweight="bold", color=C["ink"],
                zorder=6,
                arrowprops=dict(arrowstyle="-", color=C["ink_3"], lw=0.8,
                                 shrinkA=3, shrinkB=3) if (dx, dy) != (0, 0) else None)

save(f, "simplex")
