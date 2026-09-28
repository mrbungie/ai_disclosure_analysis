"""Chart 3: ratio of disclosed AI activities vs Defensive, at equal disclosure volume."""
from style import C, ARCH, ARCH_SHORT, fig, save, results_json

d = results_json("posture", "strategy_dimensions_diagnostics.json")["activity_volume_regression"]["volume_adjusted"]

rows = [
    ("Vocal Substantives", d["Vocal Substantives"]["ratio"], d["Vocal Substantives"]["ratio_ci95"]),
    ("Governance-Led Disclosers", d["Governance-Led Disclosers"]["ratio"], d["Governance-Led Disclosers"]["ratio_ci95"]),
    ("Defensive Disclosers", 1.0, None),
]

f = fig(1000, 520)
ax = f.add_axes([0.04, 0.16, 0.72, 0.72])

ys = [2, 1, 0]
ax.axvline(1.0, color=C["ink_3"], lw=1.1, linestyle=(0, (3, 3)), zorder=1)

for (name, ratio, ci), y in zip(rows, ys):
    color = ARCH[name]
    label = ARCH_SHORT[name]
    if ci is not None:
        ax.plot(ci, [y, y], color=color, alpha=0.35, lw=9, solid_capstyle="round", zorder=2)
        ax.plot([ratio], [y], marker="o", markersize=15, color=color, zorder=3)
        ax.annotate(f"{ratio:.2f}×", (ci[1], y), xytext=(16, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=24, fontweight="bold", color=color, zorder=4)
        ax.annotate(f"[{ci[0]:.2f}–{ci[1]:.2f}]", (ci[1], y), xytext=(16, -26),
                    textcoords="offset points", ha="left", va="center", fontsize=16, color=C["ink_3"])
    else:
        ax.plot([ratio], [y], marker="o", markersize=15, markerfacecolor="none",
                 markeredgecolor=color, markeredgewidth=2.2, zorder=3)
        ax.annotate("reference", (ratio, y), xytext=(16, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=17, color=C["ink_3"])
    ax.annotate(label, (0.8, y), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", fontsize=17, fontweight="semibold", color=color)

ax.set_xlim(0.8, 2.8)
ax.set_ylim(-0.7, 2.7)
ax.set_yticks([])
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(C["rule"])
ax.set_xticks([1.0, 1.5, 2.0, 2.5])
ax.tick_params(axis="x", length=0, pad=10, labelsize=17)
ax.set_xlabel("Disclosed AI activities relative to Defensive, holding disclosure volume fixed",
              fontsize=17, color=C["ink_2"], labelpad=14)

save(f, "activity_ratio")
