"""Chart 7: matrix of market-battery coefficients, with BH-FDR 5% markers."""
import numpy as np

from style import C, fig, save, results, read

df = read(results("crash_archetypes", "call_archetype_full_battery_targets.csv"))

col_vars = ["w_call", "w_expanding", "intensity_expanding", "has_posture", "w_voc", "w_gov"]
col_labels = ["Decoupling W,\nthis call", "Decoupling W,\naccumulated", "Cumulative\nintensity D",
              "Discloses AI\n(has)", "Vocal vs.\nDefensive", "Governance vs.\nDefensive"]

row_vars = ["beta_shift_delta", "car_m1_p1", "car_p2_p63", "ncskew_wk_post", "duvol_wk_post",
            "revenue_growth_ttm_post_pct", "gross_margin_ttm_change_pp", "roic_minus_wacc_ttm_change_pp"]
row_labels = ["Beta shift around call", "CAR [−1,+1]", "Drift CAR [+2,+63]",
              "Crash risk NCSKEW", "Crash risk DUVOL", "Revenue growth (4Q)",
              "Gross margin change", "ROIC−WACC change"]
row_groups = [0, 1, 1, 2, 2, 3, 3, 3]  # 0 risk, 1 returns, 2 tail risk, 3 fundamentals
group_names = {0: "RISK", 1: "RETURNS", 2: "TAIL RISK", 3: "FUNDAMENTALS"}

sub = df[df["variable"].isin(col_vars)].copy()

# Benjamini-Hochberg at 5% over all 48 cells.
sub = sub.sort_values("p").reset_index(drop=True)
sub["rank"] = sub.index + 1
sub["crit"] = 0.05 * sub["rank"] / len(sub)
cand = sub["p"] <= sub["crit"]
maxrank = sub.loc[cand, "rank"].max() if cand.any() else 0
sub["bh"] = sub["rank"] <= maxrank

cell = {(r["target"], r["variable"]): r for _, r in sub.iterrows()}


# Layout in data units: one column per disclosure variable, one row per outcome.
# Sized so that 1 pt = 1.333 display px lands text at ~20 px on the slide.
nrow, ncol = len(row_vars), len(col_vars)
f = fig(1290, 385)
ax = f.add_axes([0, 0, 1, 1])
LEFT = -2.35                       # room for group + row labels
ax.set_xlim(LEFT, ncol - 0.35)
ax.set_ylim(nrow - 0.45, -1.75)    # row 0 at top, header above it
ax.axis("off")

# zebra bands per outcome group
for g in sorted(set(row_groups)):
    rows = [i for i, gg in enumerate(row_groups) if gg == g]
    if g % 2 == 0:
        ax.axhspan(rows[0] - 0.5, rows[-1] + 0.5, xmin=0, xmax=1, color="#F1F2F4", zorder=0, lw=0)
    ax.text(LEFT + 0.02, (rows[0] + rows[-1]) / 2, group_names[g], ha="left", va="center",
            fontsize=14, fontweight="semibold", color=C["ink_3"])

for ri, lab in enumerate(row_labels):
    ax.text(-0.55, ri, lab, ha="right", va="center", fontsize=14.5, color=C["ink"])
for ci, lab in enumerate(col_labels):
    ax.text(ci, -1.05, lab, ha="center", va="center", fontsize=14.5, color=C["ink_2"], linespacing=1.2)

bmax = max(abs(float(r["beta_std"])) for k, r in cell.items() if r["bh"])
for ri, t in enumerate(row_vars):
    for ci, v in enumerate(col_vars):
        r = cell.get((t, v))
        if r is None:
            continue
        b, p = float(r["beta_std"]), float(r["p"])
        if r["bh"]:
            col = C["petrol"] if b > 0 else C["rust"]
            s = 180 + 900 * abs(b) / bmax
            ax.scatter([ci], [ri], s=s, color=col, zorder=4, linewidths=0)
            ax.text(ci + 0.2, ri, f"{b:+.2f}", ha="left", va="center", fontsize=15,
                    fontweight="bold", color=col, zorder=5)
        elif p < 0.05:
            ax.scatter([ci], [ri], s=150, facecolor="none", edgecolor=C["slate"], linewidths=1.8, zorder=3)
        else:
            ax.scatter([ci], [ri], s=26, color=C["rule"], zorder=2, linewidths=0)

save(f, "market_grid")
