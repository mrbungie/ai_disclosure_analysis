"""Chart 6: within-firm-year call-minus-filing gap (pp), by claim family."""
from style import C, fig, save, results_json

d = results_json("washing", "activity_grounding.json")["channel_gap"]["pooled"]

rows = [
    ("Quantified outcome or metric", "quantified_outcome", "evidence"),
    ("Named product or process", "named_product_or_process", "evidence"),
    ("Named business function", "named_function", "evidence"),
    ("Named third-party provider", "third_party_named_provider", "evidence"),
    ("Customer-facing deployment", "customer_facing_deployment", "mix"),
    ("Internal deployment", "internal_deployment", "mix"),
    ("Proprietary AI development", "proprietary_ai", "mix"),
    ("Governance or restriction", "governance_or_restriction", "caveat"),
]

f = fig(900, 500)
ax = f.add_axes([0.32, 0.08, 0.62, 0.84])

n = len(rows)
# Extra vertical gap after a group boundary so group headers have room.
ys = []
cursor = 0.0
prev_group = rows[0][2]
for _, _, group in rows:
    if group != prev_group:
        cursor += 0.9
    ys.append(cursor)
    cursor += 1.0
    prev_group = group
top = ys[-1]
ys = [top - y for y in ys]
ax.axvline(0, color=C["ink_3"], lw=1.2, zorder=1)

group_label_y = {}
for (label, key, group), y in zip(rows, ys):
    r = d[key]
    gap = r["gap_pp"]
    t = r["t"]
    se = abs(gap / t) if t != 0 else 0
    lo, hi = gap - 1.96 * se, gap + 1.96 * se
    sig = r["p"] < 0.05
    if not sig:
        color = C["slate"]
    elif gap > 0:
        color = C["petrol"]
    else:
        color = C["rust"]

    if sig:
        ax.plot([lo, hi], [y, y], color=color, alpha=0.35, lw=7, solid_capstyle="round", zorder=2)
        ax.plot([gap], [y], marker="o", markersize=11, color=color, zorder=3)
    else:
        ax.plot([lo, hi], [y, y], color=color, alpha=0.3, lw=7, solid_capstyle="round", zorder=2)
        ax.plot([gap], [y], marker="o", markersize=11, markerfacecolor=C["bg"],
                 markeredgecolor=color, markeredgewidth=2.0, zorder=3)

    label_x = hi + 0.4 if gap >= 0 else lo - 0.4
    ha = "left" if gap >= 0 else "right"
    txt = f"+{gap:.1f} pp" if gap >= 0 else f"−{abs(gap):.1f} pp"
    if not sig:
        txt += "  (ns)"
    ax.annotate(txt, (label_x, y), ha=ha, va="center", fontsize=16,
                fontweight="bold" if sig else "regular", color=color)
    group_label_y[y] = group

# Row labels on the left (axes fraction for x)
for (label, key, group), y in zip(rows, ys):
    ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=16.5, color=C["ink_2"])

# Separators between groups
sep1_y = (ys[3] + ys[4]) / 2
sep2_y = (ys[6] + ys[7]) / 2
ax.axhline(sep1_y, color=C["grid"], lw=1.0, xmin=-0.55, xmax=1.0, clip_on=False, zorder=0)
ax.axhline(sep2_y, color=C["grid"], lw=1.0, xmin=-0.55, xmax=1.0, clip_on=False, zorder=0)

ax.text(-0.02, ys[0] + 0.62, "EVIDENCE", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=C["ink_3"])
ax.text(-0.02, ys[4] + 0.55, "ACTIVITY MIX", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=C["ink_3"])
ax.text(-0.02, ys[7] + 0.55, "CAVEATS", transform=ax.get_yaxis_transform(),
        ha="right", va="center", fontsize=12.5, fontweight="bold", color=C["ink_3"])

ax.set_xlim(-17, 13)
ax.set_ylim(-1.1, ys[0] + 0.9)
ax.set_yticks([])
ax.set_xticks([])
for s in ("left", "top", "right", "bottom"):
    ax.spines[s].set_visible(False)

bottom_y = -1.0
ax.annotate("← more in filings", (-16.6, bottom_y), ha="left", va="center", fontsize=15,
            fontweight="bold", color=C["rust"])
ax.annotate("more in calls →", (12.6, bottom_y), ha="right", va="center", fontsize=15,
            fontweight="bold", color=C["petrol"])

save(f, "venue_gap")
