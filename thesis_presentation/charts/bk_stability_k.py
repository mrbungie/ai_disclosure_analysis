"""Backup chart: bootstrap Jaccard stability across candidate k, frame- and
firm-level resampling regimes, side by side."""
from style import C, fig, save, results_json

d = results_json("posture", "bootstrap_jaccard_results.json")
ks = [2, 3, 4, 5]


def series(level):
    means = [d[level][str(k)]["overall_mean"] for k in ks]
    mins = [d[level][str(k)]["min"] for k in ks]
    return means, mins


f = fig(1500, 520)
ax1 = f.add_axes([0.065, 0.16, 0.41, 0.78])
ax2 = f.add_axes([0.565, 0.16, 0.41, 0.78], sharey=ax1)

panels = [
    (ax1, "frame_level", "Frame-level resampling\n(measurement error)"),
    (ax2, "firm_level", "Firm-level resampling\n(sample composition)"),
]

for ax, level, label in panels:
    means, mins = series(level)

    # k = 3 highlight band
    ax.axvspan(2.62, 3.38, color=C["petrol_light"], alpha=0.15, lw=0, zorder=0)
    ax.text(3.0, 0.365, "k = 3", ha="center", va="bottom", fontsize=14.5,
            fontweight="semibold", color=C["petrol_dark"])

    ax.axhline(0.60, color=C["rust"], ls=(0, (1, 2)), lw=1.6, zorder=1)

    ax.plot(ks, means, "o-", color=C["petrol"], lw=2.4, ms=7,
            mfc=C["petrol"], mec=C["bg"], mew=1.2, zorder=3, label="Mean Jaccard")
    ax.plot(ks, mins, "s--", color=C["rust"], lw=2.0, ms=6.5,
            mfc=C["rust"], mec=C["bg"], mew=1.0, zorder=3,
            label="Minimum-archetype Jaccard")

    for k, mn, mean in zip(ks, mins, means):
        below_floor = mn < 0.60
        offset = -20 if mn <= mean else 15
        ax.annotate(f"{mn:.2f}", (k, mn), xytext=(0, offset),
                    textcoords="offset points", ha="center", fontsize=13.5,
                    fontweight="semibold", color=C["rust"] if below_floor else C["ink_2"])

    ax.set_xticks(ks)
    ax.set_xlim(1.8, 5.2)
    ax.set_xlabel("Number of archetypes (k)", fontsize=15, color=C["ink_2"])
    ax.text(0.03, 0.96, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="semibold", color=C["ink"])
    ax.grid(False)
    ax.spines["left"].set_visible(ax is ax1)

ax1.set_ylim(0.30, 1.0)
ax1.set_ylabel("Bootstrap Jaccard overlap", fontsize=15, color=C["ink_2"])
ax2.tick_params(axis="y", labelleft=False)
ax2.spines["left"].set_color(C["grid"])

ax1.text(1.85, 0.615, "stability floor 0.60", fontsize=12.5, color=C["rust"],
         va="bottom", ha="left", style="italic")

handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
         ncol=2, fontsize=14.5, frameon=False)

save(f, "stability_k")
