"""Chart 4: customer-facing share of disclosed activities, by archetype."""
from style import C, ARCH, ARCH_SHORT, fig, save, results, read

df = read(results("posture", "archetype_activity.parquet"))
sub = df[df["metric"] == "Customer-facing share"].copy()

order = ["Vocal Substantives", "Governance-Led Disclosers", "Defensive Disclosers"]
sub["order"] = sub["archetype"].map({n: i for i, n in enumerate(order)})
sub = sub.sort_values("order")
pooled = sub["pooled_rate"].iloc[0]

f = fig(700, 520)
ax = f.add_axes([0.09, 0.12, 0.86, 0.8])

xs = range(len(order))
vals = [sub[sub["archetype"] == n]["archetype_rate"].iloc[0] for n in order]
colors = [ARCH[n] for n in order]

ax.bar(xs, vals, width=0.56, color=colors, zorder=3)
for x, v, c in zip(xs, vals, colors):
    ax.annotate(f"{v:.1f}%", (x, v), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=20, fontweight="bold", color=c)

ax.set_xlim(-0.6, 2.6)
ax.axhline(pooled, color=C["ink_3"], lw=1.3, linestyle=(0, (4, 3)), zorder=2)
ax.annotate(f"pooled {pooled:.1f}%", (2.6, pooled), xytext=(-6, 8), textcoords="offset points",
            ha="right", va="bottom", fontsize=13, color=C["ink_3"])

ax.set_xticks(list(xs))
ax.set_xticklabels([ARCH_SHORT[n] for n in order], fontsize=15, color=C["ink_2"])
ax.set_ylim(0, max(vals) * 1.28)
ax.set_yticks([])
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(C["rule"])
ax.tick_params(axis="x", length=0, pad=10)
ax.tick_params(axis="y", length=0)

save(f, "customer_share")
