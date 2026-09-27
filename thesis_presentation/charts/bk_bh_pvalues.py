"""Backup chart: Benjamini-Hochberg multiple-testing correction across the 48
disclosure-variable p-values in the crash/archetype outcome battery."""
import numpy as np

from style import C, fig, save, results, read

VARS = ["w_call", "w_expanding", "intensity_expanding", "has_posture", "w_voc", "w_gov"]
VAR_SHORT = {"w_call": "W_call", "w_expanding": "W_exp", "intensity_expanding": "D",
             "has_posture": "Posture", "w_voc": "W_voc", "w_gov": "Gov"}
TARGET_SHORT = {"beta_shift_delta": "β shift", "car_p2_p63": "Drift",
                "revenue_growth_ttm_post_pct": "Revenue"}

df = read(results("crash_archetypes", "call_archetype_full_battery_targets.csv"))
sub = df[df["variable"].isin(VARS)].copy()
n = len(sub)
sub = sub.sort_values("p").reset_index(drop=True)
sub["rank"] = np.arange(1, n + 1)
sub["bh_thresh"] = 0.05 * sub["rank"] / n
crossings = sub.index[sub["p"] <= sub["bh_thresh"]]
max_rank = sub.loc[crossings, "rank"].max() if len(crossings) else 0
sub["bh_survive"] = sub["rank"] <= max_rank
sub["nominal"] = sub["p"] < 0.05

n_bh = int(sub["bh_survive"].sum())
n_nom = int(sub["nominal"].sum())
print(f"n={n} BH survivors={n_bh} nominal p<.05={n_nom}")

f = fig(1400, 560)
ax = f.add_axes([0.075, 0.14, 0.90, 0.80])

ax.set_yscale("log")
ranks_line = np.arange(1, n + 1)
bh_line = 0.05 * ranks_line / n
ax.plot(ranks_line, bh_line, color=C["petrol"], ls=(0, (5, 4)), lw=1.8, zorder=2)
ax.axhline(0.05, color=C["rust"], ls=(0, (1, 2)), lw=1.6, zorder=2)

other = ~sub["bh_survive"] & ~sub["nominal"]
nominal_only = ~sub["bh_survive"] & sub["nominal"]

ax.scatter(sub.loc[other, "rank"], sub.loc[other, "p"], s=26, color=C["mist"],
           edgecolor=C["ink_3"], linewidths=0.6, zorder=3)
ax.scatter(sub.loc[nominal_only, "rank"], sub.loc[nominal_only, "p"], s=60,
           facecolor="none", edgecolor=C["slate"], linewidths=1.6, zorder=4)
ax.scatter(sub.loc[sub["bh_survive"], "rank"], sub.loc[sub["bh_survive"], "p"],
           s=110, color=C["petrol"], edgecolor=C["bg"], linewidths=1.2, zorder=5)

for i, (_, r) in enumerate(sub[sub["bh_survive"]].iterrows()):
    lab = f"{TARGET_SHORT.get(r['target'], r['target'])} · {VAR_SHORT.get(r['variable'], r['variable'])}"
    dy = 14 if i % 2 == 0 else 40
    dx = -18 if i % 2 == 0 else 18
    ha = "right" if i % 2 == 0 else "left"
    ax.annotate(lab, xy=(r["rank"], r["p"]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, fontsize=12,
                fontweight="semibold", color=C["petrol_dark"])

ax.text(n * 0.60, 0.35, "Benjamini–Hochberg 5%", color=C["petrol"],
        fontsize=13, fontweight="semibold", rotation=18)
ax.text(n * 0.02, 0.062, "nominal p = 0.05", color=C["rust"], fontsize=13,
        fontweight="semibold", va="bottom")

ax.text(0.02, 0.96, "6 of 48 survive", transform=ax.transAxes, ha="left",
        va="top", fontsize=19, fontweight="bold", color=C["petrol_dark"])
ax.text(0.02, 0.87, f"{n_nom} nominally significant", transform=ax.transAxes,
        ha="left", va="top", fontsize=14.5, color=C["ink_2"])

ax.set_xlim(0, n + 1)
ax.set_ylim(1e-5, 1.3)
yt = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
ax.set_yticks(yt)
ax.set_yticklabels(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"])
ax.set_xlabel("Rank (ascending p-value, n = 48)", fontsize=15, color=C["ink_2"])
ax.set_ylabel("p-value (log scale)", fontsize=15, color=C["ink_2"])
ax.grid(axis="y", color=C["grid"], lw=0.8, zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

save(f, "bh_pvalues")
