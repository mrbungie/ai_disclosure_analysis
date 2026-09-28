"""Chart 1: share of S&P 500 firms disclosing AI, by year."""
from style import C, fig, save, gold, read

df = read(gold("covariates", "firm_year", "disclosure_volume"), columns=["year", "any_ai"])
s = df.groupby("year")["any_ai"].mean() * 100
years = s.index.tolist()

f = fig(1100, 620)
ax = f.add_axes([0.06, 0.1, 0.90, 0.84])

solid_years = [y for y in years if y <= 2025]
solid_vals = [s[y] for y in solid_years]
dotted_years = [y for y in years if y >= 2025]
dotted_vals = [s[y] for y in dotted_years]

# GenAI boom shading 2023-2024
ax.axvspan(2022.5, 2024.5, color=C["rust_light"], alpha=0.25, lw=0, zorder=0)
ax.text(2023.5, 97.5, "GENAI BOOM", ha="center", va="top", fontsize=18,
        fontweight="bold", color=C["rust"], alpha=0.9, zorder=3)

# Area fill under solid segment
ax.fill_between(solid_years, solid_vals, 0, color=C["petrol"], alpha=0.08, lw=0, zorder=1)

# Solid line 2021-2025
ax.plot(solid_years, solid_vals, color=C["petrol"], lw=2.6, marker="o",
        markersize=7, markerfacecolor=C["petrol"], markeredgecolor=C["bg"],
        markeredgewidth=1.4, zorder=4, solid_capstyle="round")
# Dotted 2025-2026
ax.plot(dotted_years, dotted_vals, color=C["petrol"], lw=2.2, linestyle=(0, (2, 2)),
        zorder=4)
ax.plot([2026], [s[2026]], marker="o", markersize=7, markerfacecolor=C["bg"],
        markeredgecolor=C["petrol"], markeredgewidth=1.8, zorder=5)

# Value labels
for y in years:
    v = s[y]
    if y == 2021:
        ax.annotate(f"{v:.1f}%", (y, v), xytext=(-2, 16), textcoords="offset points",
                    ha="center", fontsize=21, fontweight="bold", color=C["petrol_dark"])
    elif y == 2025:
        ax.annotate(f"{v:.1f}%", (y, v), xytext=(-2, 16), textcoords="offset points",
                    ha="center", fontsize=21, fontweight="bold", color=C["petrol_dark"])
    elif y == 2026:
        ax.annotate(f"{v:.1f}%", (y, v), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=19, fontweight="bold", color=C["ink_2"])
        ax.annotate("2026 YTD", (y, v), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=18, color=C["ink_3"])
    elif y in (2023, 2024):
        ax.annotate(f"{v:.1f}%", (y, v), xytext=(-10, 6), textcoords="offset points",
                    ha="right", fontsize=18, color=C["ink_2"])
    else:
        ax.annotate(f"{v:.1f}%", (y, v), xytext=(0, 18), textcoords="offset points",
                    ha="center", fontsize=18, color=C["ink_2"])

ax.set_ylim(0, 100)
ax.set_xlim(2020.6, 2026.5)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0", "25", "50", "75", "100%"])
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years])
ax.yaxis.grid(True, color=C["grid"], lw=1.0, zorder=0)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="both", length=0, labelsize=18)

save(f, "diffusion")
