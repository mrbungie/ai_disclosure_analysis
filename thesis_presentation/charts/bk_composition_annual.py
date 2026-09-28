"""Backup chart: 100% stacked annual composition of disclosure archetypes,
2021-2026 (year to date)."""
import numpy as np

from style import C, ARCH, ARCH_SHORT, fig, save, results, read

ARCH_ORDER = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers", "No AI"]

comp = read(results("posture", "archetype_composition_annual.parquet"))
share = comp.pivot(index="year", columns="archetype", values="share_pct").reindex(columns=ARCH_ORDER)
years = share.index.tolist()

f = fig(600, 450)
ax = f.add_axes([0.0, 0.09, 0.74, 0.89])

x = np.arange(len(years))
bottom = np.zeros(len(years))
for a in ARCH_ORDER:
    vals = share[a].values
    dark = a in ("Vocal Substantives", "Defensive Disclosers")
    bars = ax.bar(x, vals, bottom=bottom, color=ARCH[a], width=0.62, zorder=3)
    for xi, v, b in zip(x, vals, bottom):
        if v >= 6:
            txt_color = "white" if dark else C["ink"]
            ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=14, fontweight="semibold", color=txt_color, zorder=4)
    bottom += vals

# direct labels to the right of the 2026 bar
last_x = x[-1]
last_bottom = 0.0
for a in ARCH_ORDER:
    v = share[a].iloc[-1]
    ax.annotate(ARCH_SHORT[a].replace("Vocal Substantives", "Vocal"), xy=(last_x + 0.32, last_bottom + v / 2),
                xytext=(8, 0), textcoords="offset points", va="center",
                ha="left", fontsize=15, fontweight="semibold",
                color=C["ink_3"] if a == "No AI" else ARCH[a])
    last_bottom += v

xticklabels = [f"'{str(y)[2:]}" if y != 2026 else "'26\nYTD" for y in years]
ax.set_xticks(x)
ax.set_xticklabels(xticklabels, fontsize=15, color=C["ink_2"])
ax.set_ylim(0, 100)
ax.set_yticks([])
ax.set_xlim(-0.6, len(years) - 0.2)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

save(f, "composition_annual")
