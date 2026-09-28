"""Backup chart: mean AI activity grounding for a fixed 2021 cohort vs. each
year's new filers, first year on record."""
from style import C, fig, save, results, read

df = read(results("washing", "appendix_g_cohort_grounding.csv")).sort_values("year")

f = fig(1000, 520)
ax = f.add_axes([0.11, 0.14, 0.72, 0.78])

fixed = df.dropna(subset=["fixed_cohort_grounding"])
ax.plot(fixed["year"], fixed["fixed_cohort_grounding"], "-", color=C["petrol"],
        lw=2.6, zorder=3)
ax.plot(fixed["year"][:-1], fixed["fixed_cohort_grounding"][:-1], "o",
        color=C["petrol"], ms=7, mec=C["bg"], mew=1.2, zorder=4)
last_fixed = fixed.iloc[-1]
ax.plot([last_fixed["year"]], [last_fixed["fixed_cohort_grounding"]], "o", ms=10,
        mfc=C["bg"], mec=C["petrol"], mew=2.0, zorder=5)

new = df.dropna(subset=["new_filer_year_grounding"])
new = new[new["year"] >= 2022]
ax.plot(new["year"], new["new_filer_year_grounding"], "--", color=C["rust"],
        lw=2.2, zorder=3)
ax.plot(new["year"][:-1], new["new_filer_year_grounding"][:-1], "s",
        color=C["rust"], ms=6.5, mec=C["bg"], mew=1.0, zorder=4)
last_new = new.iloc[-1]
ax.plot([last_new["year"]], [last_new["new_filer_year_grounding"]], "s", ms=9,
        mfc=C["bg"], mec=C["rust"], mew=2.0, zorder=5)

ax.annotate("Fixed 2021 cohort", xy=(last_fixed["year"], last_fixed["fixed_cohort_grounding"]),
            xytext=(12, 8), textcoords="offset points", fontsize=15.5,
            fontweight="semibold", color=C["petrol"], va="center")
ax.annotate("New filers,\nfirst year", xy=(last_new["year"], last_new["new_filer_year_grounding"]),
            xytext=(12, -10), textcoords="offset points", fontsize=15.5,
            fontweight="semibold", color=C["rust"], va="center")

ax.annotate("YTD", xy=(last_fixed["year"], last_fixed["fixed_cohort_grounding"]),
            xytext=(0, 16), textcoords="offset points", fontsize=13.5, ha="center",
            color=C["ink_3"], style="italic")

ax.set_xlim(2020.6, 2027.6)
ax.set_ylim(0.30, 0.50)
ax.set_xticks(range(2021, 2027))
ax.set_ylabel("Mean grounding G (0-1)", fontsize=15, color=C["ink_2"])
ax.grid(axis="y", color=C["grid"], lw=1.0, zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

save(f, "cohort_grounding")
