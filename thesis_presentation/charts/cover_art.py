"""Cover motif: the 451 disclosing firms on the posture simplex, drawn quietly
for the dark cover slide (no labels; decorative but real data)."""
import sys

import numpy as np

from style import REPO, fig, save

sys.path.insert(0, str(REPO / "scripts" / "gold" / "posture"))
import joblib  # noqa: E402
import layers as L  # noqa: E402

bundle = joblib.load(REPO / "models" / "posture_archetype_static" / "model.pkl")
model, scaler, feats, names = bundle["model"], bundle["scaler"], bundle["feature_names"], bundle["cluster_names"]
df = L.read_gold("firm", ("covariates", "posture_archetype_static"))
pop = df[df["archetype"] != "No AI"].reset_index(drop=True)
W = model.transform(scaler.transform(pop[feats].values))
col = {n: i for i, n in names.items()}

TOP, BL, BR = np.array([0.5, np.sqrt(3) / 2]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
xy = (np.outer(W[:, col["Defensive Disclosers"]], TOP) + np.outer(W[:, col["Vocal Substantives"]], BL)
      + np.outer(W[:, col["Governance-Led Disclosers"]], BR))

tone = {"Vocal Substantives": "#7AA2AE", "Governance-Led Disclosers": "#D9B86A", "Defensive Disclosers": "#9FB0B8"}

f = fig(640, 580)
ax = f.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.04, 0.91)
ax.set_aspect("equal")
ax.axis("off")
tri = np.array([TOP, BL, BR, TOP])
ax.plot(tri[:, 0], tri[:, 1], color="#FAF8F4", lw=1.2, alpha=0.35)
for t in (1 / 3, 2 / 3):  # faint inner lattice
    for a, b, c in ((TOP, BL, BR), (BL, BR, TOP), (BR, TOP, BL)):
        p, q = a + t * (b - a), a + t * (c - a)
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#FAF8F4", lw=0.6, alpha=0.10)
ax.scatter(xy[:, 0], xy[:, 1], s=16, c=pop["archetype"].map(tone).values, alpha=0.8, linewidths=0)
ax.scatter([TOP[0], BL[0], BR[0]], [TOP[1], BL[1], BR[1]], s=70, color="#B85F42", zorder=3)
save(f, "cover_art")
