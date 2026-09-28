"""Chart: narrative evolution. Left, AI disclosure intensity D by year.
Right, the seven posture inputs of the archetypal analysis by year: per
firm-year, the posture rates over its 10-K, 8-K and DEF 14A AI frames, kept
when the firm-year has at least MIN_FRAMES frames and shrunk toward that year's
prior (posture_features, as in the fit); lines are the mean across firms,
before standardization."""
import sys

import pandas as pd

from style import C, fig, save, gold, read, REPO

sys.path.insert(0, str(REPO / "scripts" / "gold" / "posture"))
from posture_features import MIN_FRAMES, POSTURE, build_posture, load_frames, shrink_to_prior  # noqa: E402

YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

dv = read(gold("covariates", "firm_year", "disclosure_volume"), columns=["year", "frames_per_1k"])
d_mean = dv.groupby("year")["frames_per_1k"].mean()

fy = build_posture(load_frames(), ["ticker", "year"])
fy = fy[fy["year"].isin(YEARS) & (fy["n_posture_frames"] >= MIN_FRAMES)]
shrunk = pd.concat([shrink_to_prior(g[POSTURE], g["n_posture_frames"]).assign(year=y)
                    for y, g in fy.groupby("year")])
by_year = shrunk.groupby("year")[POSTURE].mean() * 100
n_firms = shrunk.groupby("year").size()

DIMS = [
    ("temporal_posture", "Realized", C["petrol_dark"]),
    ("risk_orientation", "Risk", C["rust"]),
    ("ai_positioning", "Customer-facing", C["teal"]),
    ("specificity", "Specificity", C["slate"]),
    ("promotional_posture", "Promotional", C["petrol_light"]),
    ("governance_orientation", "Governance", C["ochre"]),
    ("hedging_posture", "Hedging", C["rust_light"]),
]

f = fig(1290, 430)
axL = f.add_axes([0.0, 0.14, 0.2, 0.74])
axR = f.add_axes([0.27, 0.14, 0.53, 0.74])

# ---- Left: D by year ----
xs = list(range(len(YEARS)))
vals = [d_mean[y] for y in YEARS]
cols = [C["petrol"]] * 5 + [C["petrol_light"]]
axL.bar(xs, vals, width=0.66, color=cols, zorder=3)
for x, v, y in zip(xs, vals, YEARS):
    if y in (2021, 2023, 2025, 2026):
        axL.annotate(f"{v:.2f}", (x, v), xytext=(0, 6), textcoords="offset points", ha="center",
                     fontsize=14, fontweight="bold", color=C["ink_3"] if y == 2026 else C["petrol_dark"])
axL.set_xticks(xs)
axL.set_xticklabels(["'21", "'22", "'23", "'24", "'25", "'26\nYTD"], fontsize=14, color=C["ink_2"])
axL.set_ylim(0, max(vals) * 1.2)
axL.set_yticks([])
for s in ("left", "top", "right"):
    axL.spines[s].set_visible(False)
axL.spines["bottom"].set_color(C["rule"])
axL.tick_params(axis="x", length=0, pad=6)
ratio = d_mean[2025] / d_mean[2021]
axL.annotate(f"{ratio:.1f}×", (1.0, max(vals) * 0.62), ha="center", fontsize=30, fontweight="bold", color=C["rust"])
axL.annotate("2021 → 2025", (1.0, max(vals) * 0.62), xytext=(0, -22), textcoords="offset points",
             ha="center", fontsize=14, color=C["ink_3"])
axL.text(0.0, 1.06, "D: AI frames per 1,000 words", transform=axL.transAxes, fontsize=16,
         fontweight="semibold", color=C["ink_2"], ha="left", va="bottom")

# ---- Right: the seven posture inputs ----
EVENTS = [(2022.9, "ChatGPT"), (2023.95, "SEC warning"), (2025.05, "DeepSeek")]
for x, lab in EVENTS:
    axR.axvline(x, color=C["rule"], lw=1.2, linestyle=(0, (3, 3)), zorder=1)
    axR.text(x, 81, lab, ha="center", va="bottom", fontsize=13.5, color=C["ink_3"])

mid = [y + 0.5 for y in YEARS]
for col, name, color in DIMS:
    v = by_year[col]
    axR.plot(mid[:5], [v[y] for y in YEARS[:5]], color=color, lw=3, marker="o", markersize=6,
            markeredgecolor=C["bg"], markeredgewidth=1.2, zorder=3)
    axR.plot(mid[4:], [v[y] for y in YEARS[4:]], color=color, lw=2.4, linestyle=(0, (2, 2)), zorder=3)
    axR.plot([mid[-1]], [v[2026]], marker="o", markersize=6, markerfacecolor=C["bg"], markeredgecolor=color,
            markeredgewidth=1.8, zorder=4)


def spread(items, gap):
    """Label y positions, pushed apart bottom-up so none overlap."""
    placed, out = [], {}
    for key, y in sorted(items, key=lambda t: t[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        out[key] = y
    return out


left = spread([(c, by_year[c][2021]) for c, *_ in DIMS], 4.4)
right = spread([(c, by_year[c][2026]) for c, *_ in DIMS], 5.4)
for col, name, color in DIMS:
    v = by_year[col]
    axR.text(mid[0] - 0.14, left[col], f"{v[2021]:.0f}%", ha="right", va="center", fontsize=14,
            fontweight="bold", color=color)
    axR.text(mid[-1] + 0.2, right[col], f"{name}  {v[2021]:.0f}% → {v[2025]:.0f}%", ha="left", va="center",
            fontsize=15, fontweight="bold", color=color)

axR.set_xlim(2020.95, 2026.75)
axR.set_ylim(0, 85)
axR.set_xticks(mid)
axR.set_xticklabels([f"{y if y < 2026 else '2026 YTD'}\n{n_firms[y]} firms" for y in YEARS],
                   fontsize=14, color=C["ink_2"])
axR.set_yticks([0, 20, 40, 60, 80])
axR.set_yticklabels(["0", "20", "40", "60", "80%"], fontsize=13.5, color=C["ink_3"])
axR.yaxis.grid(True, color=C["grid"], lw=1.0, zorder=0)
for s in ("left", "top", "right", "bottom"):
    axR.spines[s].set_visible(False)
axR.tick_params(axis="both", length=0, pad=6)

axR.text(0.0, 1.06, "Posture inputs P: firms with ≥ 5 AI frames, average share of their frames", transform=axR.transAxes,
         fontsize=16, fontweight="semibold", color=C["ink_2"], ha="left", va="bottom")

save(f, "narrative")
